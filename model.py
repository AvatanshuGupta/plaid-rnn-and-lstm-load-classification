#!/usr/bin/env python3
"""
Train sequential classifiers (vanilla RNN, LSTM, optional GRU) on the windowed
electrical features produced by plaid_features.py and write a detailed report.

Each PLAID record (id) becomes ONE time series: its feature rows ordered by
window_idx.  Training samples are fixed-length chunks (seq_len windows) cut out
of the training records; at test time every record is cut into chunks and the
chunk probabilities are averaged -> one prediction per record.

Anti-leakage: the train / val / test split is done BY RECORD ID (never by
window), stratified on the target label.

Usage
-----
  python train_sequence_models.py --features features/all_features.csv \
         --out_dir results --target appliance_type --models rnn lstm

Outputs (in --out_dir)
----------------------
  report.md / report.json         full performance report
  curves.png, cm_<model>.png, per_class_f1.png
  <model>.pt                      weights + preprocessing + class names
  shap_<model>/                   SHAP plots + importance CSVs (with --shap)
  predictions_<model>.csv         per-record test predictions
"""
import argparse
import copy
import json
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")

NON_FEATURE = {"id", "window_idx", "start_sample", "end_sample", "t_start_s",
               "t_end_s", "appliance_type", "load", "status", "location"}


# =============================================================================
# Data
# =============================================================================
def load_data(path, target, drop_cols, min_per_class):
    df = pd.read_csv(path, dtype={"id": str})
    df = df[df[target].notna() & (df[target].astype(str).str.strip() != "")]
    df[target] = df[target].astype(str)

    rec_label = df.groupby("id")[target].first()
    counts = rec_label.value_counts()
    small = counts[counts < min_per_class]
    if len(small):
        print(f"[warn] dropping classes with < {min_per_class} records: "
              f"{small.to_dict()}")
        df = df[~df[target].isin(small.index)]

    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in NON_FEATURE and c not in drop_cols]
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    if "window_idx" in df.columns:
        df = df.sort_values(["id", "window_idx"], kind="stable")

    seqs, labels = {}, {}
    for rid, g in df.groupby("id", sort=False):
        seqs[rid] = g[feat_cols].to_numpy(np.float32)
        labels[rid] = g[target].iloc[0]
    return seqs, labels, feat_cols


class Preprocessor:
    """median-impute -> clip to train percentiles -> standardise (train only)."""

    def fit(self, arrays):
        X = np.vstack(arrays)
        self.keep = ~np.all(np.isnan(X), axis=0)          # all-NaN cols dropped
        X = X[:, self.keep]
        self.median = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), self.median, X)
        self.lo, self.hi = np.percentile(Xi, [0.5, 99.5], axis=0)
        Xc = np.clip(Xi, self.lo, self.hi)
        self.mean, self.std = Xc.mean(0), Xc.std(0) + 1e-8
        return self

    def transform(self, a):
        a = a[:, self.keep]
        a = np.where(np.isnan(a), self.median, a)
        return ((np.clip(a, self.lo, self.hi) - self.mean) / self.std).astype(np.float32)


def make_chunks(seqs, ids, y, seq_len, stride):
    X, Y, R = [], [], []
    for rid in ids:
        s = seqs[rid]
        if len(s) < seq_len:                                   # pad short records
            s = np.vstack([s, np.repeat(s[-1:], seq_len - len(s), 0)])
        starts = list(range(0, len(s) - seq_len + 1, stride))
        if starts[-1] != len(s) - seq_len:
            starts.append(len(s) - seq_len)
        for st in starts:
            X.append(s[st:st + seq_len]); Y.append(y[rid]); R.append(rid)
    return np.stack(X), np.array(Y), np.array(R)


# =============================================================================
# Model
# =============================================================================
class SeqClassifier(nn.Module):
    def __init__(self, n_feat, n_cls, cell="lstm", hidden=64, layers=2,
                 dropout=0.3, bidir=False):
        super().__init__()
        kw = dict(input_size=n_feat, hidden_size=hidden, num_layers=layers,
                  batch_first=True, bidirectional=bidir,
                  dropout=dropout if layers > 1 else 0.0)
        self.rnn = {"rnn": lambda: nn.RNN(nonlinearity="tanh", **kw),
                    "lstm": lambda: nn.LSTM(**kw),
                    "gru": lambda: nn.GRU(**kw)}[cell]()
        d = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, hidden),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_cls))

    def forward(self, x):
        out, _ = self.rnn(x)                                   # (B,T,d)
        z = torch.cat([out.mean(1), out[:, -1]], dim=1)        # mean-pool + last step
        return self.head(z)


@torch.no_grad()
def predict_probs(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.concatenate(out)


def aggregate_by_record(probs, y, rids):
    ids = np.unique(rids)
    P = np.stack([probs[rids == r].mean(0) for r in ids])
    Y = np.array([y[rids == r][0] for r in ids])
    return ids, P, Y


def train_one(cell, a, Xtr, ytr, Xva, yva, rva, n_cls, device, class_w):
    torch.manual_seed(a.seed)
    model = SeqClassifier(Xtr.shape[2], n_cls, cell, a.hidden, a.layers,
                          a.dropout, a.bidir).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32,
                                                    device=device),
                                label_smoothing=0.05)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
            "val_macro_f1": [], "lr": []}
    best, best_state, bad, best_ep = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), a.batch):
            idx = perm[i:i + a.batch]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            xb = xb + a.noise * torch.randn_like(xb)           # light augmentation
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        pv = predict_probs(model, Xva, device)
        vl = log_loss(yva, pv, labels=list(range(n_cls)))
        _, Pv, Yv = aggregate_by_record(pv, yva, rva)
        hist["epoch"].append(ep); hist["train_loss"].append(tot / len(Xt))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(accuracy_score(Yv, Pv.argmax(1)))
        hist["val_macro_f1"].append(f1_score(Yv, Pv.argmax(1), average="macro"))
        hist["lr"].append(opt.param_groups[0]["lr"])
        sched.step(vl)
        if vl < best - 1e-4:
            best, best_state, bad, best_ep = vl, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  [{cell}] ep {ep:3d} train {hist['train_loss'][-1]:.4f} "
                  f"val {vl:.4f} val_acc(rec) {hist['val_acc'][-1]:.3f}")
        if bad >= a.patience:
            print(f"  [{cell}] early stop at epoch {ep} (best {best_ep})")
            break
    model.load_state_dict(best_state)
    return model, hist, dict(n_params=n_params, train_time_s=time.time() - t0,
                             best_epoch=best_ep, epochs_run=len(hist["epoch"]))


# =============================================================================
# Metrics
# =============================================================================
def full_metrics(y, P, class_names, boot=1000, seed=0):
    n_cls = len(class_names)
    pred = P.argmax(1)
    present = sorted(set(y) | set(pred))
    m = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, labels=present, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, labels=present, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, labels=present, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, labels=present, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "log_loss": log_loss(y, P, labels=list(range(n_cls))),
    }
    if n_cls >= 4:
        top3 = np.mean([yy in np.argsort(p)[-3:] for yy, p in zip(y, P)])
        m["top3_accuracy"] = float(top3)
    try:                                                      # macro OvR AUC on classes present
        aucs = {}
        for c in sorted(set(y)):
            yb = (y == c).astype(int)
            if 0 < yb.sum() < len(yb):
                aucs[class_names[c]] = roc_auc_score(yb, P[:, c])
        m["roc_auc_macro_ovr"] = float(np.mean(list(aucs.values())))
        m["roc_auc_per_class"] = aucs
    except ValueError:
        pass

    rng = np.random.default_rng(seed)                         # bootstrap CIs
    accs, f1s = [], []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[i], pred[i]))
        f1s.append(f1_score(y[i], pred[i], average="macro", zero_division=0))
    m["accuracy_ci95"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
    m["macro_f1_ci95"] = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]

    m["per_class"] = classification_report(
        y, pred, labels=list(range(n_cls)), target_names=class_names,
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(n_cls)))
    m["confusion_matrix"] = cm.tolist()
    conf = [(class_names[i], class_names[j], int(cm[i, j]))
            for i in range(n_cls) for j in range(n_cls) if i != j and cm[i, j] > 0]
    m["top_confusions"] = sorted(conf, key=lambda t: -t[2])[:8]
    return m


# =============================================================================
# Plots
# =============================================================================
def plot_curves(hists, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for name, h in hists.items():
        ax[0].plot(h["epoch"], h["train_loss"], label=f"{name} train")
        ax[0].plot(h["epoch"], h["val_loss"], "--", label=f"{name} val")
        ax[1].plot(h["epoch"], h["val_acc"], label=name)
        ax[2].plot(h["epoch"], h["val_macro_f1"], label=name)
    for x, t in zip(ax, ["Loss", "Validation accuracy (record-level)",
                         "Validation macro-F1 (record-level)"]):
        x.set_title(t); x.set_xlabel("epoch"); x.legend(); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_cm(cm, names, title, path):
    cm = np.array(cm); norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, .7 * len(names) + 3),) * 2)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right"); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.colorbar(im, fraction=.046); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_per_class_f1(results, names, path):
    fig, ax = plt.subplots(figsize=(max(8, len(names) * .9), 4))
    w = .8 / len(results)
    for k, (mname, m) in enumerate(results.items()):
        f1 = [m["per_class"][c]["f1-score"] for c in names]
        ax.bar(np.arange(len(names)) + k * w, f1, w, label=mname)
    ax.set_xticks(np.arange(len(names)) + .4 - w / 2)
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


# =============================================================================
# Report
# =============================================================================
def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    return s + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"


def f(x, d=4):
    return f"{x:.{d}f}" if isinstance(x, (float, np.floating)) else str(x)


def write_report(path, info, results, run_info, split_info, class_names):
    L = ["# Load classification: sequential model report\n"]
    L.append("## 1. Setup\n")
    L.append(md_table(["item", "value"], [(k, v) for k, v in info.items()]))
    L.append("\n## 2. Data split (by record id, stratified)\n")
    L.append(md_table(["class", "train", "val", "test"],
                      [(c, *split_info["per_class"][c]) for c in class_names] +
                      [("**total records**", *split_info["totals"])]))
    L.append("\n## 3. Headline comparison (test set, record-level)\n")
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "macro_precision", "macro_recall", "cohen_kappa", "mcc", "log_loss",
            "top3_accuracy", "roc_auc_macro_ovr"]
    L.append(md_table(["metric"] + list(results),
                      [(k, *[f(m[k]) if k in m else "-" for m in results.values()])
                       for k in keys] +
                      [("accuracy 95% CI", *[f"[{m['accuracy_ci95'][0]:.3f}, {m['accuracy_ci95'][1]:.3f}]"
                                             for m in results.values()]),
                       ("macro-F1 95% CI", *[f"[{m['macro_f1_ci95'][0]:.3f}, {m['macro_f1_ci95'][1]:.3f}]"
                                             for m in results.values()])]))
    L.append("\n## 4. Training summary\n")
    L.append(md_table(["model", "params", "epochs run", "best epoch", "train time (s)",
                       "chunk-level test acc", "chunk-level test macro-F1"],
                      [(n, r["n_params"], r["epochs_run"], r["best_epoch"],
                        f"{r['train_time_s']:.1f}", f(r["chunk_acc"]), f(r["chunk_f1"]))
                       for n, r in run_info.items()]))
    L.append("\n![curves](curves.png)\n")
    for name, m in results.items():
        L.append(f"\n## 5. Detail: {name}\n")
        rep = m["per_class"]
        L.append(md_table(["class", "precision", "recall", "f1", "support", "AUC (OvR)"],
                          [(c, f(rep[c]["precision"]), f(rep[c]["recall"]),
                            f(rep[c]["f1-score"]), int(rep[c]["support"]),
                            f(m.get("roc_auc_per_class", {}).get(c, float("nan"))))
                           for c in class_names]))
        L.append("\nMost frequent confusions (true -> predicted, count):\n")
        L.append(md_table(["true", "predicted", "count"], m["top_confusions"]) if m["top_confusions"]
                 else "_none, no misclassifications_\n")
        if name != "RandomForest (baseline)":
            L.append(f"\n![cm](cm_{name}.png)\n")
    L.append("\n![f1](per_class_f1.png)\n")
    L.append("\n## 6. How to read this\n")
    L.append("- Headline numbers are **record-level** (chunk probabilities averaged per record), "
             "computed on records never seen in training.\n"
             "- Macro-F1 / balanced accuracy matter more than accuracy if classes are imbalanced.\n"
             "- If the test set is small the CIs above are wide; differences between models "
             "inside the CI are not significant.\n"
             "- The RandomForest row uses per-record mean/std/min/max of the same features; "
             "if the RNN/LSTM does not beat it, the temporal structure is not adding much.\n"
             "- PLAID has several records per physical appliance; a split by id can still put "
             "the same appliance in train and test, so real-world generalisation may be lower.\n")
    open(path, "w").write("\n".join(L))



# =============================================================================
# SHAP explainability
# =============================================================================
def run_shap(model, cell, Xtr, Xte, feat_names, class_names, out_dir, device,
             n_bg=100, n_explain=200, seed=0, nsamples=300):
    """
    GradientExplainer (expected gradients) on the trained sequence model.
    Input to the model is (B, T, F), so SHAP values come out as (B, T, F, C):
    one value per sample, time step, feature and output class (logit).
    """
    import shap
    rng = np.random.default_rng(seed)
    bg = Xtr[rng.choice(len(Xtr), min(n_bg, len(Xtr)), replace=False)]
    idx = rng.choice(len(Xte), min(n_explain, len(Xte)), replace=False)
    Xex = Xte[idx]
    model = model.to(device).eval()
    bg_t = torch.from_numpy(bg).to(device)
    ex_t = torch.from_numpy(Xex).to(device)

    # cuDNN RNN kernels refuse backward() in eval mode -> disable cuDNN for this call
    with torch.backends.cudnn.flags(enabled=False):
        sv = shap.GradientExplainer(model, bg_t).shap_values(ex_t, nsamples=nsamples)
        with torch.no_grad():
            f_ex, f_bg = model(ex_t).cpu().numpy(), model(bg_t).cpu().numpy()
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)   # (B,T,F,C)

    # sanity check: SHAP values should sum to f(x) - E[f(background)] per class
    gap = np.abs(sv.sum((1, 2)) - (f_ex - f_bg.mean(0))).mean()
    print(f"  [shap] additivity error (mean |sum(SHAP) - (f(x)-E[f])|): {gap:.4f} logits")

    d = os.path.join(out_dir, f"shap_{cell}")
    os.makedirs(d, exist_ok=True)
    A = np.abs(sv)
    imp_fc = np.abs(sv.sum(1)).mean(0)               # (F, C): |total contribution over the sequence|, averaged over samples
    imp = pd.DataFrame(imp_fc, index=feat_names, columns=class_names)
    imp.insert(0, "overall", imp_fc.mean(1))
    imp = imp.sort_values("overall", ascending=False)
    imp.to_csv(os.path.join(d, "feature_importance.csv"))
    imp_t = A.sum(2).mean((0, 2))                    # (T,) total |attribution| per time step (all features)
    pd.DataFrame({"time_step": range(len(imp_t)), "mean_abs_shap": imp_t}
                 ).to_csv(os.path.join(d, "temporal_importance.csv"), index=False)

    # 1) global bar chart
    top = imp.head(20)[::-1]
    plt.figure(figsize=(7, 6)); plt.barh(top.index, top["overall"])
    plt.xlabel("mean |total SHAP| per sequence (logit units, averaged over classes)")
    plt.title(f"{cell.upper()}: global feature importance"); plt.tight_layout()
    plt.savefig(os.path.join(d, "global_importance.png"), dpi=130); plt.close()

    # 2) class x feature heat-map
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(class_names) + 3), 6))
    im = ax.imshow(top[class_names].to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=60, ha="right")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
    ax.set_title(f"{cell.upper()}: mean |SHAP| per class"); plt.colorbar(im)
    plt.tight_layout(); plt.savefig(os.path.join(d, "class_feature_heatmap.png"), dpi=130); plt.close()

    # 3) which part of the sequence matters
    plt.figure(figsize=(7, 3.5)); plt.plot(imp_t, marker="o")
    plt.xlabel("time step inside the sequence (window index)"); plt.ylabel("sum over features of mean |SHAP|")
    plt.title(f"{cell.upper()}: temporal importance"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(d, "temporal_importance.png"), dpi=130); plt.close()

    # 4) beeswarm per class. SHAP is additive, so summing over time gives each
    #    feature's total contribution; the colour is the feature's mean value over the sequence.
    feat_val = Xex.mean(1)
    for c, cname in enumerate(class_names):
        try:
            shap.summary_plot(sv[..., c].sum(1), feat_val, feature_names=feat_names,
                              max_display=15, show=False)
            plt.title(f"{cell.upper()} - class: {cname}"); plt.tight_layout()
            plt.savefig(os.path.join(d, f"beeswarm_{cname}.png".replace("/", "_")), dpi=120)
        except Exception as e:                             # plotting must never kill the run
            print(f"  [shap] beeswarm for {cname} skipped: {e}")
        plt.close("all")
    return imp, float(gap), d


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--target", default="appliance_type", help="appliance_type | load | status")
    p.add_argument("--models", nargs="+", default=["rnn", "lstm"], choices=["rnn", "lstm", "gru"])
    p.add_argument("--seq_len", type=int, default=30, help="windows per training sequence")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bidir", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.05, help="gaussian input-noise std (augmentation)")
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--min_per_class", type=int, default=6)
    p.add_argument("--drop_cols", nargs="*", default=["f0_Hz"],
                   help="feature columns to exclude (f0_Hz can fingerprint a recording session)")
    p.add_argument("--shap", action="store_true", help="run SHAP analysis on each trained model")
    p.add_argument("--shap_models", nargs="+", default=None, help="subset of models to explain (default: all)")
    p.add_argument("--shap_background", type=int, default=100, help="background chunks for SHAP")
    p.add_argument("--shap_samples", type=int, default=200, help="test chunks to explain")
    p.add_argument("--shap_nsamples", type=int, default=300, help="expected-gradient samples (higher = more accurate, slower)")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    seqs, labels, feat_cols = load_data(a.features, a.target, set(a.drop_cols), a.min_per_class)
    class_names = sorted(set(labels.values()))
    c2i = {c: i for i, c in enumerate(class_names)}
    y_rec = {r: c2i[l] for r, l in labels.items()}
    ids = np.array(list(seqs)); ys = np.array([y_rec[r] for r in ids])
    n_cls = len(class_names)
    print(f"{len(ids)} records, {n_cls} classes, {len(feat_cols)} features")

    # ---- split by record id ----
    try:
        tv_ids, te_ids, tv_y, _ = train_test_split(ids, ys, test_size=a.test_size,
                                                   stratify=ys, random_state=a.seed)
        tr_ids, va_ids = train_test_split(tv_ids, test_size=a.val_size / (1 - a.test_size),
                                          stratify=tv_y, random_state=a.seed)
    except ValueError as e:
        warnings.warn(f"stratified split failed ({e}); falling back to random split")
        tv_ids, te_ids = train_test_split(ids, test_size=a.test_size, random_state=a.seed)
        tr_ids, va_ids = train_test_split(tv_ids, test_size=a.val_size / (1 - a.test_size),
                                          random_state=a.seed)

    # ---- preprocessing (fit on train only) ----
    pre = Preprocessor().fit([seqs[r] for r in tr_ids])
    seqs = {r: pre.transform(s) for r, s in seqs.items()}
    n_feat = int(pre.keep.sum())
    used_feats = [c for c, k in zip(feat_cols, pre.keep) if k]

    stride_tr = max(1, a.seq_len // 4)
    Xtr, ytr, _ = make_chunks(seqs, tr_ids, y_rec, a.seq_len, stride_tr)
    Xva, yva, rva = make_chunks(seqs, va_ids, y_rec, a.seq_len, a.seq_len)
    Xte, yte, rte = make_chunks(seqs, te_ids, y_rec, a.seq_len, a.seq_len)
    print(f"chunks  train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  "
          f"(seq_len={a.seq_len}, features={n_feat})")

    cnt = np.bincount(ytr, minlength=n_cls).astype(float)
    class_w = np.where(cnt > 0, cnt.sum() / (n_cls * np.maximum(cnt, 1)), 0.0)

    results, hists, run_info, shap_res = {}, {}, {}, {}
    for cell in a.models:
        print(f"\n=== training {cell.upper()} ===")
        model, hist, ri = train_one(cell, a, Xtr, ytr, Xva, yva, rva, n_cls, device, class_w)
        pt = predict_probs(model, Xte, device)
        rids, P, Y = aggregate_by_record(pt, yte, rte)
        m = full_metrics(Y, P, class_names)
        ri["chunk_acc"] = accuracy_score(yte, pt.argmax(1))
        ri["chunk_f1"] = f1_score(yte, pt.argmax(1), average="macro", zero_division=0)
        results[cell.upper()], hists[cell.upper()], run_info[cell.upper()] = m, hist, ri
        plot_cm(m["confusion_matrix"], class_names, f"{cell.upper()} (test, record-level)",
                os.path.join(a.out_dir, f"cm_{cell.upper()}.png"))
        pd.DataFrame({"id": rids, "true": [class_names[i] for i in Y],
                      "pred": [class_names[i] for i in P.argmax(1)],
                      "confidence": P.max(1), "correct": P.argmax(1) == Y}
                     ).to_csv(os.path.join(a.out_dir, f"predictions_{cell.upper()}.csv"), index=False)
        torch.save({"state_dict": model.state_dict(), "classes": class_names,
                    "features": used_feats, "args": vars(a),
                    "pre": {k: getattr(pre, k).tolist() for k in
                            ("keep", "median", "lo", "hi", "mean", "std")}},
                   os.path.join(a.out_dir, f"{cell}.pt"))
        print(f"  test (record-level): acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")
        if a.shap and (a.shap_models is None or cell in a.shap_models):
            print(f"  running SHAP for {cell.upper()} ...")
            shap_res[cell.upper()] = run_shap(model, cell, Xtr, Xte, used_feats, class_names,
                                              a.out_dir, device, a.shap_background,
                                              a.shap_samples, a.seed, a.shap_nsamples)

    if not a.no_baseline:
        print("\n=== RandomForest baseline (per-record aggregate features) ===")
        agg = lambda r: np.concatenate([np.nanmean(seqs[r], 0), seqs[r].std(0),
                                        seqs[r].min(0), seqs[r].max(0)])
        rf = RandomForestClassifier(500, class_weight="balanced", random_state=a.seed, n_jobs=-1)
        t0 = time.time()
        rf.fit(np.stack([agg(r) for r in tr_ids]), [y_rec[r] for r in tr_ids])
        Prf = np.zeros((len(te_ids), n_cls))
        Prf[:, rf.classes_] = rf.predict_proba(np.stack([agg(r) for r in te_ids]))
        name = "RandomForest (baseline)"
        results[name] = full_metrics(np.array([y_rec[r] for r in te_ids]), Prf, class_names)
        run_info[name] = dict(n_params="-", train_time_s=time.time() - t0, best_epoch="-",
                              epochs_run="-", chunk_acc=float("nan"), chunk_f1=float("nan"))
        print(f"  test: acc {results[name]['accuracy']:.4f}  macro-F1 {results[name]['macro_f1']:.4f}")

    plot_curves(hists, os.path.join(a.out_dir, "curves.png"))
    plot_per_class_f1(results, class_names, os.path.join(a.out_dir, "per_class_f1.png"))

    def cc(idset):
        return np.bincount([y_rec[r] for r in idset], minlength=n_cls)
    split_info = {"per_class": {c: (int(cc(tr_ids)[i]), int(cc(va_ids)[i]), int(cc(te_ids)[i]))
                                for i, c in enumerate(class_names)},
                  "totals": (len(tr_ids), len(va_ids), len(te_ids))}
    lens = [len(s) for s in seqs.values()]
    info = {"features file": a.features, "target": a.target, "classes": n_cls,
            "records": len(ids), "windows per record (min/median/max)":
                f"{min(lens)} / {int(np.median(lens))} / {max(lens)}",
            "feature count used": n_feat, "excluded columns": ", ".join(a.drop_cols) or "none",
            "sequence length (windows)": a.seq_len,
            "chunks train/val/test": f"{len(Xtr)} / {len(Xva)} / {len(Xte)}",
            "architecture": f"{a.layers}-layer, hidden={a.hidden}, dropout={a.dropout}, "
                            f"bidirectional={a.bidir}",
            "optimiser": f"AdamW lr={a.lr} wd={a.wd}, batch={a.batch}, label-smoothing 0.05, "
                         f"class-weighted CE, grad-clip 1.0, input noise {a.noise}",
            "early stopping": f"val loss, patience {a.patience}", "seed": a.seed,
            "device": device}
    write_report(os.path.join(a.out_dir, "report.md"), info, results, run_info, split_info, class_names)
    if shap_res:
        with open(os.path.join(a.out_dir, "report.md"), "a") as fh:
            fh.write("\n## 7. SHAP feature attribution (GradientExplainer)\n")
            fh.write("SHAP values are in logit units, computed on standardised features. "
                     "Importance = |sum of SHAP over the sequence's time steps| averaged over explained test chunks and classes. "
                     "Plots and full tables are in `shap_<model>/`.\n")
            for mname, (imp, gap, d) in shap_res.items():
                fh.write(f"\n### {mname} (additivity error {gap:.4f} logits)\n")
                fh.write(md_table(["rank", "feature", "mean |SHAP|"],
                                  [(k + 1, n, f"{v:.4f}") for k, (n, v) in
                                   enumerate(imp["overall"].head(10).items())]))
                fh.write(f"\n![global](shap_{mname.lower()}/global_importance.png)\n"
                         f"\n![heatmap](shap_{mname.lower()}/class_feature_heatmap.png)\n"
                         f"\n![temporal](shap_{mname.lower()}/temporal_importance.png)\n")
    json.dump({"info": info, "split": split_info, "run_info": run_info,
               "results": results, "history": hists},
              open(os.path.join(a.out_dir, "report.json"), "w"), indent=2, default=float)
    print(f"\nReport written to {os.path.join(a.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()