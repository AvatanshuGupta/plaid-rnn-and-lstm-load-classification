#!/usr/bin/env python3
"""
FastAPI server + web UI for the PLAID load-classification project.

Put this file (and the `static/` folder) in the SAME root folder as
  plaid_features.py, train_sequence_models.py
  features/all_features.csv           (output of plaid_features.py)
  results/ (rnn.pt, lstm.pt, report.json, report.md, shap_<model>/ ...)
                                       (output of train_sequence_models.py)

Run:   uvicorn app:app --reload          then open  http://127.0.0.1:8000
   or: python app.py

Folders are auto-detected ("results" or the root itself; "features" or the root).
Override with environment variables PLAID_RESULTS and PLAID_FEATURES (csv path).

What it serves
  /                       web UI (Overview, Performance, SHAP, Predict, Full report)
  /api/info               models, classes, feature list, paths
  /api/report             report.json + report.md
  /api/shap/{model}       SHAP feature-importance tables for a trained model
  /api/records            records available in all_features.csv (for the picker)
  POST /api/predict       classify a record (existing id / features csv / raw i-v csv)
  POST /api/explain       local SHAP explanation of one prediction
  /results/...            static access to the generated PNG/CSV files
"""
import json
import math
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from model import SeqClassifier  # noqa: E402  (model definition)

CELLS = ("rnn", "lstm", "gru")


# ----------------------------------------------------------------------------
# Locate files
# ----------------------------------------------------------------------------
def _find_results():
    env = os.environ.get("PLAID_RESULTS")
    if env:
        return Path(env).resolve()
    for c in ("results", "."):
        p = (ROOT / c).resolve()
        if p.is_dir() and (any(p.glob("*.pt")) or (p / "report.json").exists()):
            return p
    return ROOT / "results"


def _find_features():
    env = os.environ.get("PLAID_FEATURES")
    if env:
        return Path(env).resolve()
    for c in ("features/all_features.csv", "all_features.csv", "results/all_features.csv"):
        p = (ROOT / c).resolve()
        if p.exists():
            return p
    return None


RESULTS = _find_results()
FEATURES_CSV = _find_features()


# ----------------------------------------------------------------------------
# JSON helpers (NaN is not valid JSON)
# ----------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def ok(o):
    return JSONResponse(clean(o))


# ----------------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------------
_models, _lock = {}, threading.Lock()


def available_models():
    if not RESULTS.is_dir():
        return []
    return sorted([p.stem for p in RESULTS.glob("*.pt") if p.stem in CELLS],
                  key=lambda n: CELLS.index(n))


def get_model(name):
    name = (name or "").lower()
    with _lock:
        if name in _models:
            return _models[name]
        path = RESULTS / f"{name}.pt"
        if name not in CELLS or not path.exists():
            raise HTTPException(404, f"model '{name}' not found in {RESULTS}")
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(path, map_location="cpu", weights_only=False)  # your own checkpoint
        a, feats, classes = ck["args"], ck["features"], ck["classes"]
        net = SeqClassifier(len(feats), len(classes), name, a["hidden"], a["layers"],
                            a["dropout"], a.get("bidir", False))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        pre = {k: np.asarray(v, dtype=np.float32) for k, v in ck["pre"].items() if k != "keep"}
        _models[name] = dict(name=name, net=net, classes=classes, features=feats, args=a,
                             pre=pre, seq_len=int(a["seq_len"]),
                             n_params=sum(p.numel() for p in net.parameters()))
        return _models[name]


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
_feat = {"df": None}


def feat_df():
    if FEATURES_CSV is None or not FEATURES_CSV.exists():
        raise HTTPException(404, "all_features.csv not found (set PLAID_FEATURES)")
    if _feat["df"] is None:
        _feat["df"] = pd.read_csv(FEATURES_CSV, dtype={"id": str})
    return _feat["df"]


def test_ids():
    """ids of the held-out test records (from predictions_<model>.csv)."""
    for m in available_models():
        p = RESULTS / f"predictions_{m.upper()}.csv"
        if p.exists():
            return set(pd.read_csv(p, dtype={"id": str})["id"])
    return set()


def infer_window_step():
    try:
        d = pd.read_csv(FEATURES_CSV, nrows=3)
        w = int(d["end_sample"][0] - d["start_sample"][0] + 1)
        s = int(d["start_sample"][1] - d["start_sample"][0]) if len(d) > 1 else w
        return w, s
    except Exception:
        return 500, 500


def standardise(b, df):
    feats, p = b["features"], b["pre"]
    miss = [c for c in feats if c not in df.columns]
    if miss:
        raise HTTPException(400, f"input is missing {len(miss)} feature column(s), e.g. {miss[:6]}")
    if "window_idx" in df.columns:
        df = df.sort_values("window_idx", kind="stable")
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32, copy=True)
    if len(X) == 0:
        raise HTTPException(400, "no feature rows found")
    X[~np.isfinite(X)] = np.nan
    X = np.where(np.isnan(X), p["median"], X)
    X = np.clip(X, p["lo"], p["hi"])
    return ((X - p["mean"]) / p["std"]).astype(np.float32)


def chunk(X, L):
    if len(X) < L:                                   # pad short records (same as training)
        X = np.vstack([X, np.repeat(X[-1:], L - len(X), 0)])
    starts = list(range(0, len(X) - L + 1, L))
    if starts[-1] != len(X) - L:
        starts.append(len(X) - L)
    return np.stack([X[s:s + L] for s in starts]), starts


@torch.no_grad()
def probs_of(net, chunks):
    return torch.softmax(net(torch.from_numpy(chunks)), -1).numpy()


def _label(df, target):
    """Ground-truth label of a record for the column the model was trained on."""
    if target not in df.columns or not len(df):
        return None
    v = df[target].iloc[0]
    return None if pd.isna(v) or str(v).strip() == "" else str(v).strip()


def target_of(b):
    return b["args"].get("target", "appliance_type")


def read_input(source, record_id, file, window, step, fs, order, f0, target="appliance_type", manual="", b=None):
    """Return (feature dataframe, true label or None, meta dict)."""
    meta = {"source": source}
    if source == "record":
        df = feat_df()
        sub = df[df["id"] == str(record_id).strip()]
        if sub.empty:
            raise HTTPException(404, f"record id '{record_id}' not found in {FEATURES_CSV.name}")
        label = _label(sub, target)
        meta.update(record_id=str(record_id).strip(), in_test=str(record_id).strip() in test_ids())
        return sub, label, meta
    if source == "manual":
        df, m = manual_frame(b, manual)
        meta.update(m)
        return df, None, meta
    if file is None:
        raise HTTPException(400, "please attach a CSV file")
    content = file.file.read(60 * 1024 * 1024)
    if source == "features":
        try:
            import io
            df = pd.read_csv(io.BytesIO(content), dtype={"id": str})
        except Exception as e:
            raise HTTPException(400, f"could not read CSV: {e}")
        label = _label(df, target)
        return df, label, meta
    if source == "raw":
        try:
            import plaid_features as pf
        except ImportError:
            raise HTTPException(500, "plaid_features.py must be in the same folder as app.py")
        w0, s0 = infer_window_step() if FEATURES_CSV else (500, 500)
        window = window or w0
        step = step or window
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            path = tmp.name
        try:
            i, v = pf.load_iv(path, order)
            if len(i) < window:
                raise HTTPException(400, f"record has {len(i)} samples, shorter than window {window}")
            df = pf.process_record(i, v, fs, window, step, f0=f0 or None)
        finally:
            os.unlink(path)
        meta.update(window=window, step=step, fs=fs, samples=int(len(i)))
        return df, None, meta
    raise HTTPException(400, f"unknown source '{source}'")



# ----------------------------------------------------------------------------
# Feature metadata (for the manual-entry form) and derivation of dependent values
# ----------------------------------------------------------------------------
_F = [
    ("v_mean", "Time domain", "V", "DC component of voltage"), ("i_mean", "Time domain", "A", "DC component of current"),
    ("v_rms", "Time domain", "V", "RMS voltage"), ("i_rms", "Time domain", "A", "RMS current"),
    ("v_rms_ac", "Time domain", "V", "RMS voltage without DC"), ("i_rms_ac", "Time domain", "A", "RMS current without DC"),
    ("v_peak", "Time domain", "V", "peak |voltage|"), ("i_peak", "Time domain", "A", "peak |current|"),
    ("v_pp", "Time domain", "V", "peak-to-peak voltage"), ("i_pp", "Time domain", "A", "peak-to-peak current"),
    ("i_crest_factor", "Time domain", "", "current peak / RMS (1.41 for a sine)"),
    ("v_crest_factor", "Time domain", "", "voltage peak / RMS (1.41 for a sine)"),
    ("i_form_factor", "Time domain", "", "current RMS / mean |i| (1.11 for a sine)"),
    ("i_std", "Time domain", "A", "standard deviation of current"), ("i_skew", "Time domain", "", "skewness of current"),
    ("i_kurtosis", "Time domain", "", "kurtosis of current (1.5 for a sine)"),
    ("i_zero_cross", "Time domain", "", "zero crossings per window"),
    ("P_real_W", "Power", "W", "real (active) power"), ("S_apparent_VA", "Power", "VA", "apparent power Vrms*Irms"),
    ("Q_nonactive_var", "Power", "var", "non-active power sqrt(S^2-P^2)"), ("pf", "Power", "", "true power factor P/S"),
    ("energy_Wh", "Power", "Wh", "energy used in the window"), ("p_inst_max", "Power", "W", "max instantaneous power"),
    ("p_inst_min", "Power", "W", "min instantaneous power"), ("Z_ohm", "Power", "ohm", "impedance Vrms/Irms"),
    ("R_eff_ohm", "Power", "ohm", "effective resistance P/Irms^2"),
    ("f0_Hz", "Fundamental", "Hz", "mains frequency"), ("v1_rms", "Fundamental", "V", "RMS of fundamental voltage"),
    ("i1_rms", "Fundamental", "A", "RMS of fundamental current"),
    ("phase_deg", "Fundamental", "deg", "voltage-current phase (+ inductive, - capacitive)"),
    ("cos_phi", "Fundamental", "", "displacement power factor"), ("sin_phi", "Fundamental", "", "sine of phase angle"),
    ("P1_W", "Fundamental", "W", "fundamental real power"), ("Q1_var", "Fundamental", "var", "fundamental reactive power"),
    ("S1_VA", "Fundamental", "VA", "fundamental apparent power"), ("D_distortion_VA", "Fundamental", "VA", "distortion power"),
    ("Z1_ohm", "Fundamental", "ohm", "fundamental impedance"), ("R1_ohm", "Fundamental", "ohm", "resistance"),
    ("X1_ohm", "Fundamental", "ohm", "reactance (+ inductive)"), ("Y1_siemens", "Fundamental", "S", "admittance"),
    ("G1_siemens", "Fundamental", "S", "conductance"), ("B1_siemens", "Fundamental", "S", "susceptance"),
    ("L_equiv_H", "Fundamental", "H", "equivalent inductance"), ("C_equiv_F", "Fundamental", "F", "equivalent capacitance"),
    ("i_h3_ratio", "Harmonics", "", "3rd harmonic / fundamental current"), ("i_h5_ratio", "Harmonics", "", "5th harmonic / fundamental"),
    ("i_h7_ratio", "Harmonics", "", "7th harmonic / fundamental"), ("i_thd", "Harmonics", "", "current THD (0.2 = 20 %)"),
    ("v_thd", "Harmonics", "", "voltage THD"), ("i_odd_even_ratio", "Harmonics", "", "odd / even harmonic energy"),
    ("i_harm_phase_h3_deg", "Harmonics", "deg", "3rd-harmonic phase vs fundamental"),
    ("i_spec_centroid_Hz", "Spectral", "Hz", "spectral centroid of current"),
    ("i_spec_entropy", "Spectral", "", "spectral entropy of current"),
    ("i_hf_energy_ratio", "Spectral", "", "energy above the 10th harmonic"),
]
FEATURE_META = {n: dict(group=g, unit=u, desc=d) for n, g, u, d in _F}
KEY_FEATURES = ["v_rms", "i_rms", "P_real_W", "phase_deg", "i_thd", "i_crest_factor"]


def derive(v):
    """Fill dependent electrical quantities from the ones the user typed."""
    d, added = dict(v), []

    def put(k, val):
        if k not in d and np.isfinite(val):
            d[k] = float(val)
            added.append(k)

    if "phase_deg" not in d and "cos_phi" in d:                    # phase from cos(phi), assume inductive
        put("phase_deg", math.degrees(math.acos(min(1.0, max(-1.0, d["cos_phi"])))))
    V1, I1 = d.get("v1_rms", d.get("v_rms")), d.get("i1_rms", d.get("i_rms"))
    if "phase_deg" in d and V1 and I1:                              # fundamental phasor quantities
        th = math.radians(d["phase_deg"])
        put("cos_phi", math.cos(th)); put("sin_phi", math.sin(th))
        put("v1_rms", V1); put("i1_rms", I1)
        S1 = V1 * I1
        put("S1_VA", S1); put("P1_W", S1 * math.cos(th)); put("Q1_var", S1 * math.sin(th))
        Z1, Y1 = V1 / max(I1, 1e-9), I1 / max(V1, 1e-9)
        put("Z1_ohm", Z1); put("R1_ohm", Z1 * math.cos(th)); put("X1_ohm", Z1 * math.sin(th))
        put("Y1_siemens", Y1); put("G1_siemens", Y1 * math.cos(th)); put("B1_siemens", -Y1 * math.sin(th))
        put("P_real_W", S1 * math.cos(th))                           # assumes sinusoidal current
    if "v_rms" in d and "i_rms" in d:
        S = d["v_rms"] * d["i_rms"]
        put("S_apparent_VA", S); put("Z_ohm", d["v_rms"] / max(d["i_rms"], 1e-9))
        if "P_real_W" in d:
            P = d["P_real_W"]
            put("pf", P / max(S, 1e-9)); put("Q_nonactive_var", math.sqrt(max(S * S - P * P, 0)))
            put("R_eff_ohm", P / max(d["i_rms"] ** 2, 1e-12))
            if "P1_W" in d and "Q1_var" in d:
                put("D_distortion_VA", math.sqrt(max(S * S - d["P1_W"] ** 2 - d["Q1_var"] ** 2, 0)))
    return d, added


def manual_frame(b, manual):
    try:
        raw = json.loads(manual or "{}")
        vals = {k: float(v) for k, v in raw.items() if k in b["features"] and v not in ("", None)}
    except (ValueError, TypeError) as e:
        raise HTTPException(400, f"invalid manual values: {e}")
    if not vals:
        raise HTTPException(400, "enter at least one value")
    full, derived = derive(vals)
    med = dict(zip(b["features"], b["pre"]["median"].tolist()))
    row = {f: full.get(f, med[f]) for f in b["features"]}
    meta = dict(manual=dict(entered=len(vals), derived=derived,
                            defaulted=len(b["features"]) - len(full)))
    return pd.DataFrame([row]), meta

# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(title="PLAID load classifier")


@app.get("/api/info")
def info():
    models = []
    for n in available_models():
        b = get_model(n)
        a = b["args"]
        models.append(dict(name=n, params=b["n_params"], seq_len=b["seq_len"], hidden=a["hidden"],
                           layers=a["layers"], bidir=a.get("bidir", False),
                           n_features=len(b["features"]),
                           has_shap=(RESULTS / f"shap_{n}" / "feature_importance.csv").exists()))
    first = get_model(models[0]["name"]) if models else None
    w, s = infer_window_step() if FEATURES_CSV else (None, None)
    return ok(dict(
        target=target_of(first) if first else None,
        models=models, default_model="rnn" if any(m["name"] == "rnn" for m in models)
        else (models[0]["name"] if models else None),
        classes=first["classes"] if first else [], features=first["features"] if first else [],
        window=w, step=s, results_dir=str(RESULTS), features_csv=str(FEATURES_CSV),
        has_report=(RESULTS / "report.json").exists(),
        images=[p.name for p in sorted(RESULTS.glob("*.png"))] if RESULTS.is_dir() else []))


@app.get("/api/feature_spec")
def feature_spec(model: Optional[str] = None):
    names = available_models()
    b = get_model(model or ("rnn" if "rnn" in names else names[0]))
    p = b["pre"]
    return ok(dict(key=[f for f in KEY_FEATURES if f in b["features"]], target=target_of(b), features=[
        dict(name=f, median=p["median"][i], lo=p["lo"][i], hi=p["hi"][i],
             **FEATURE_META.get(f, dict(group="Other", unit="", desc=f)))
        for i, f in enumerate(b["features"])]))


@app.get("/api/report")
def report():
    p = RESULTS / "report.json"
    if not p.exists():
        raise HTTPException(404, f"report.json not found in {RESULTS}. Run train_sequence_models.py first.")
    import json
    data = json.loads(p.read_text())
    md = (RESULTS / "report.md").read_text() if (RESULTS / "report.md").exists() else ""
    return ok({"report": data, "md": md})


@app.get("/api/shap/{model}")
def shap_tables(model: str):
    model = model.lower()
    d = RESULTS / f"shap_{model}"
    f = d / "feature_importance.csv"
    if not f.exists():
        raise HTTPException(404, f"no SHAP results for '{model}'. Train with --shap.")
    imp = pd.read_csv(f, index_col=0)
    classes = [c for c in imp.columns if c != "overall"]
    tmp = d / "temporal_importance.csv"
    temporal = pd.read_csv(tmp)["mean_abs_shap"].tolist() if tmp.exists() else []
    bees = {c: f"/results/shap_{model}/beeswarm_{c.replace('/', '_')}.png"
            for c in classes if (d / f"beeswarm_{c.replace('/', '_')}.png").exists()}
    md = (RESULTS / "report.md").read_text() if (RESULTS / "report.md").exists() else ""
    m = re.search(rf"### {model.upper()} \(additivity error ([0-9.]+)", md)
    return ok(dict(features=imp.index.tolist(), classes=classes, overall=imp["overall"].tolist(),
                   matrix=imp[classes].values.tolist(), temporal=temporal, beeswarm=bees,
                   additivity_error=float(m.group(1)) if m else None,
                   csv=f"/results/shap_{model}/feature_importance.csv",
                   plots={k: f"/results/shap_{model}/{k}.png" for k in
                          ("global_importance", "class_feature_heatmap", "temporal_importance")
                          if (d / f"{k}.png").exists()}))


@app.get("/api/records")
def records(model: Optional[str] = None):
    df = feat_df()
    names = available_models()
    target = target_of(get_model(model or (names[0] if names else "rnn"))) if names else "appliance_type"
    g = df.groupby("id", sort=False)
    label = g[target].first().fillna("").astype(str) if target in df.columns else ""
    out = pd.DataFrame({"n": g.size(), "label": label}).reset_index()
    t = test_ids()
    out["in_test"] = out["id"].isin(t)
    return ok(out.to_dict("records"))


def _predict_core(model, source, record_id, file, window, step, fs, order, f0, manual=""):
    b = get_model(model)
    df, label, meta = read_input(source, record_id, file, window, step, fs, order, f0,
                                 target_of(b), manual, b)
    X = standardise(b, df)
    chunks, starts = chunk(X, b["seq_len"])
    P = probs_of(b["net"], chunks)
    return b, df, label, meta, chunks, starts, P


@app.post("/api/predict")
def predict(model: str = Form("rnn"), source: str = Form("record"), record_id: str = Form(""),
            window: int = Form(0), step: int = Form(0), fs: float = Form(30000.0),
            order: str = Form("iv"), f0: float = Form(0.0), manual: str = Form(""),
            file: Optional[UploadFile] = File(None)):
    b, df, label, meta, chunks, starts, P = _predict_core(model, source, record_id, file, window,
                                                          step, fs, order, f0, manual)
    rec = P.mean(0)
    order_idx = np.argsort(-rec)
    cls = b["classes"]
    return ok(dict(
        model=b["name"], target=target_of(b), prediction=cls[order_idx[0]], confidence=rec[order_idx[0]],
        ranking=[{"class": cls[i], "prob": rec[i]} for i in order_idx],
        true_label=label, label_in_classes=(label in cls) if label else None,
        correct=(label == cls[order_idx[0]]) if label in cls else None,
        n_windows=len(df), n_chunks=len(chunks), seq_len=b["seq_len"], classes=cls,
        chunks=[{"start_window": s, "probs": p} for s, p in zip(starts, P)],
        chunk_votes={cls[i]: int((P.argmax(1) == i).sum()) for i in order_idx
                     if (P.argmax(1) == i).any()},
        meta=meta))


@app.post("/api/explain")
def explain(model: str = Form("rnn"), source: str = Form("record"), record_id: str = Form(""),
            window: int = Form(0), step: int = Form(0), fs: float = Form(30000.0),
            order: str = Form("iv"), f0: float = Form(0.0), target_class: str = Form(""),
            n_chunks: int = Form(8), nsamples: int = Form(200), manual: str = Form(""),
            file: Optional[UploadFile] = File(None)):
    """Local SHAP explanation (GradientExplainer) of the model's output for one record."""
    try:
        import shap
    except ImportError:
        raise HTTPException(500, "pip install shap")
    b, df, label, meta, chunks, starts, P = _predict_core(model, source, record_id, file, window,
                                                          step, fs, order, f0, manual)
    cls = b["classes"]
    c = cls.index(target_class) if target_class in cls else int(P.mean(0).argmax())
    idx = np.unique(np.linspace(0, len(chunks) - 1, min(len(chunks), max(1, n_chunks))).astype(int))
    Xc = chunks[idx]
    T, F = Xc.shape[1:]
    # baseline = the "average window" of the training data (0 after standardisation)
    bg = np.random.default_rng(0).normal(0, 0.05, (20, T, F)).astype(np.float32)
    net = b["net"]
    with torch.backends.cudnn.flags(enabled=False):
        sv = shap.GradientExplainer(net, torch.from_numpy(bg)).shap_values(
            torch.from_numpy(Xc), nsamples=int(nsamples))
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)   # (B,T,F,C)
    s = sv[..., c]                                                       # (B,T,F)
    with torch.no_grad():
        base = float(net(torch.from_numpy(bg))[:, c].mean())
        out = float(net(torch.from_numpy(Xc))[:, c].mean())
    tot = s.sum(1).mean(0)                                               # (F,) signed totals
    val = Xc.mean((0, 1))
    top = np.argsort(-np.abs(tot))[:15]
    top10 = top[:10]
    return ok(dict(
        model=b["name"], target_class=cls[c], predicted=cls[int(P.mean(0).argmax())],
        baseline_logit=base, output_logit=out, shap_sum=float(tot.sum()),
        n_chunks_explained=len(idx), seq_len=T,
        features=[{"feature": b["features"][i], "shap": tot[i], "value_z": val[i]} for i in top],
        per_timestep=s.sum(2).mean(0),
        heat_features=[b["features"][i] for i in top10],
        heat=s.mean(0)[:, top10].T))                                      # (10, T)


@app.get("/api/health")
def health():
    return dict(status="ok", results_dir=str(RESULTS), features_csv=str(FEATURES_CSV),
                models=available_models())


if RESULTS.is_dir():
    app.mount("/results", StaticFiles(directory=RESULTS), name="results")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)