#!/usr/bin/env python3
"""
PLAID windowed electrical-feature extractor.

For every id in the metadata JSON it reads <id>.csv (current + voltage samples),
slides a window of N samples over the signal and computes electrical parameters
for each window.

Output
------
  <out_dir>/<id>_features.csv   one row per window, one column per parameter
  <out_dir>/all_features.csv    every id stacked, with id + appliance label columns

Usage
-----
  python plaid_features.py --meta metadata.json --data_dir ./csv --out_dir ./features \
         --window 500 --step 500

Notes
-----
* Best accuracy: make --window a whole number of mains cycles.
  At 30 kHz and 60 Hz one cycle = 500 samples (PLAID was recorded in the US).
  100 / 200 sample windows (<1 cycle) still work: RMS/P/S/PF are still computed,
  and the fundamental phasor is fitted by least squares at the known mains
  frequency, but harmonics / THD are NaN because they cannot be resolved.
* If your CSV columns are (voltage, current) instead of (current, voltage) and
  have no header, pass --order vi.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_metadata(path):
    """Return list of metadata dicts (accepts a list or {id: meta} dict)."""
    with open(path) as f:
        meta = json.load(f)
    if isinstance(meta, dict):
        out = []
        for k, v in meta.items():
            v = dict(v)
            v.setdefault("id", k)
            out.append(v)
        return out
    return meta


def parse_fs(meta_item, default=30000.0):
    s = str(meta_item.get("meta", {}).get("header", {}).get("sampling_frequency", ""))
    digits = "".join(c for c in s if c.isdigit() or c == ".")
    return float(digits) if digits else default


def load_iv(path, order="iv"):
    """Read a CSV and return (i, v) numpy arrays. Handles header / no header."""
    df = pd.read_csv(path, header=None)
    try:
        df.iloc[0].astype(float)          # first row numeric -> no header
    except (ValueError, TypeError):
        df = pd.read_csv(path)            # has header

    cols = {str(c).strip().lower(): c for c in df.columns}
    i_key = next((cols[k] for k in ("i", "current", "amps") if k in cols), None)
    v_key = next((cols[k] for k in ("v", "voltage", "volts") if k in cols), None)
    if i_key is None or v_key is None:    # positional fallback
        a, b = df.columns[:2]
        i_key, v_key = (a, b) if order == "iv" else (b, a)
    return df[i_key].to_numpy(float), df[v_key].to_numpy(float)


# ----------------------------------------------------------------------------
# Signal processing
# ----------------------------------------------------------------------------
def estimate_mains_freq(v, fs, lo=45.0, hi=65.0):
    """Mains frequency from a zero-padded FFT of the whole voltage record."""
    v = v - v.mean()
    nfft = 1 << int(np.ceil(np.log2(len(v) * 8)))
    spec = np.abs(np.fft.rfft(v * np.hanning(len(v)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    m = (f >= lo) & (f <= hi)
    k = np.argmax(spec[m])
    return float(f[m][k])


def build_projector(n, fs, f0, n_harm):
    """
    Least-squares projector for  x(t) = c0 + sum_h a_h cos(h w t) + b_h sin(h w t).
    Computed once per window length -> every window is a single matrix product.
    Works for windows shorter than one cycle when n_harm = 1 (fundamental only).
    """
    t = np.arange(n) / fs
    w = 2 * np.pi * f0
    cols = [np.ones(n)]
    for h in range(1, n_harm + 1):
        cols += [np.cos(h * w * t), np.sin(h * w * t)]
    return np.linalg.pinv(np.stack(cols, axis=1))     # (2H+1, n)


def phasors(coef, n_harm):
    """coef -> arrays of harmonic amplitude (peak) and phase for h = 1..H."""
    a = coef[1::2][:n_harm]
    b = coef[2::2][:n_harm]
    return np.hypot(a, b), np.arctan2(-b, a)           # A cos(wt+phi)


def window_features(i, v, fs, f0, proj, n_harm, harmonic_ok):
    n = len(i)
    eps = 1e-12
    f = {}

    # ---- time-domain ------------------------------------------------------
    f["v_mean"], f["i_mean"] = v.mean(), i.mean()
    f["v_rms"] = np.sqrt(np.mean(v ** 2))
    f["i_rms"] = np.sqrt(np.mean(i ** 2))
    f["v_rms_ac"] = np.sqrt(np.mean((v - v.mean()) ** 2))
    f["i_rms_ac"] = np.sqrt(np.mean((i - i.mean()) ** 2))
    f["v_peak"], f["i_peak"] = np.abs(v).max(), np.abs(i).max()
    f["v_pp"], f["i_pp"] = np.ptp(v), np.ptp(i)
    f["i_crest_factor"] = f["i_peak"] / (f["i_rms"] + eps)
    f["v_crest_factor"] = f["v_peak"] / (f["v_rms"] + eps)
    f["i_form_factor"] = f["i_rms"] / (np.mean(np.abs(i)) + eps)
    f["i_std"] = i.std()
    f["i_skew"] = float(np.mean(((i - i.mean()) / (i.std() + eps)) ** 3))
    f["i_kurtosis"] = float(np.mean(((i - i.mean()) / (i.std() + eps)) ** 4))
    f["i_zero_cross"] = int(np.sum(np.diff(np.signbit(i - i.mean())) != 0))

    # ---- power (time domain, include all harmonics) -----------------------
    p_inst = v * i
    f["P_real_W"] = p_inst.mean()
    f["S_apparent_VA"] = f["v_rms"] * f["i_rms"]
    f["Q_nonactive_var"] = np.sqrt(max(f["S_apparent_VA"] ** 2 - f["P_real_W"] ** 2, 0.0))
    f["pf"] = f["P_real_W"] / (f["S_apparent_VA"] + eps)
    f["energy_Wh"] = f["P_real_W"] * n / fs / 3600.0
    f["p_inst_max"], f["p_inst_min"] = p_inst.max(), p_inst.min()
    f["Z_ohm"] = f["v_rms"] / (f["i_rms"] + eps)
    f["R_eff_ohm"] = f["P_real_W"] / (f["i_rms"] ** 2 + eps)     # P / Irms^2

    # ---- fundamental (least-squares phasor fit at mains frequency) --------
    cv = proj @ v
    ci = proj @ i
    Av, phv = phasors(cv, n_harm)
    Ai, phi = phasors(ci, n_harm)

    V1, I1 = Av[0] / np.sqrt(2), Ai[0] / np.sqrt(2)               # RMS
    theta = np.angle(np.exp(1j * (phv[0] - phi[0])))              # +ve = current lags
    f["f0_Hz"] = f0
    f["v1_rms"], f["i1_rms"] = V1, I1
    f["phase_deg"] = np.degrees(theta)
    f["cos_phi"] = np.cos(theta)                                  # displacement PF
    f["sin_phi"] = np.sin(theta)
    f["P1_W"] = V1 * I1 * np.cos(theta)
    f["Q1_var"] = V1 * I1 * np.sin(theta)                         # + inductive, - capacitive
    f["S1_VA"] = V1 * I1
    f["D_distortion_VA"] = np.sqrt(max(f["S_apparent_VA"] ** 2 - f["P1_W"] ** 2 - f["Q1_var"] ** 2, 0.0))
    f["Z1_ohm"] = V1 / (I1 + eps)
    f["R1_ohm"] = f["Z1_ohm"] * np.cos(theta)
    f["X1_ohm"] = f["Z1_ohm"] * np.sin(theta)                     # + inductive, - capacitive
    f["Y1_siemens"] = I1 / (V1 + eps)
    f["G1_siemens"] = f["Y1_siemens"] * np.cos(theta)             # conductance
    f["B1_siemens"] = -f["Y1_siemens"] * np.sin(theta)            # susceptance
    w = 2 * np.pi * f0
    f["L_equiv_H"] = f["X1_ohm"] / w if f["X1_ohm"] > 0 else np.nan
    f["C_equiv_F"] = -1.0 / (w * f["X1_ohm"]) if f["X1_ohm"] < 0 else np.nan

    # ---- harmonics / THD (only if the window holds >= 1 full cycle) -------
    if harmonic_ok:
        for h in (3, 5, 7):
            f[f"i_h{h}_ratio"] = Ai[h - 1] / (Ai[0] + eps)        # relative to fundamental
        f["i_thd"] = np.sqrt(np.sum(Ai[1:] ** 2)) / (Ai[0] + eps)
        f["v_thd"] = np.sqrt(np.sum(Av[1:] ** 2)) / (Av[0] + eps)
        f["i_odd_even_ratio"] = (np.sum(Ai[0::2] ** 2) / (np.sum(Ai[1::2] ** 2) + eps))
        f["i_harm_phase_h3_deg"] = np.degrees(np.angle(np.exp(1j * (phi[2] - 3 * phi[0]))))
    else:
        for k in ("i_h3_ratio", "i_h5_ratio", "i_h7_ratio", "i_thd", "v_thd",
                  "i_odd_even_ratio", "i_harm_phase_h3_deg"):
            f[k] = np.nan

    # ---- spectral (plain FFT, only meaningful for >= 1 cycle) --------------
    if harmonic_ok:
        spec = np.abs(np.fft.rfft(i * np.hanning(n))) / n
        fr = np.fft.rfftfreq(n, 1 / fs)
        ps = spec ** 2
        tot = ps.sum() + eps
        f["i_spec_centroid_Hz"] = float((fr * ps).sum() / tot)
        pn = ps / tot
        f["i_spec_entropy"] = float(-(pn[pn > 0] * np.log2(pn[pn > 0])).sum())
        hf = fr > 10 * f0
        f["i_hf_energy_ratio"] = float(ps[hf].sum() / tot)        # >10th harmonic content
    else:
        f["i_spec_centroid_Hz"] = f["i_spec_entropy"] = f["i_hf_energy_ratio"] = np.nan

    return f


# ----------------------------------------------------------------------------
# Per-record driver
# ----------------------------------------------------------------------------
def process_record(i, v, fs, window, step, n_harm_max=15, f0=None):
    f0 = f0 or estimate_mains_freq(v, fs)
    cycle = fs / f0
    harmonic_ok = window >= int(round(cycle))
    # keep harmonics below Nyquist and resolvable by the window
    n_harm = n_harm_max if harmonic_ok else 1
    n_harm = max(1, min(n_harm, int(fs / 2 / f0) - 1))
    proj = build_projector(window, fs, f0, n_harm)

    rows = []
    for k, s in enumerate(range(0, len(i) - window + 1, step)):
        sl = slice(s, s + window)
        r = {"window_idx": k, "start_sample": s, "end_sample": s + window - 1,
             "t_start_s": s / fs, "t_end_s": (s + window) / fs}
        r.update(window_features(i[sl], v[sl], fs, f0, proj, n_harm, harmonic_ok))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="metadata JSON (list or dict)")
    ap.add_argument("--data_dir", required=True, help="folder holding <id>.csv")
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--window", type=int, default=500, help="samples per window")
    ap.add_argument("--step", type=int, default=None, help="hop (default = window)")
    ap.add_argument("--order", default="iv", choices=["iv", "vi"],
                    help="column order if CSV has no header")
    ap.add_argument("--f0", type=float, default=None, help="force mains frequency (Hz)")
    a = ap.parse_args()

    step = a.step or a.window
    os.makedirs(a.out_dir, exist_ok=True)
    frames = []

    for item in load_metadata(a.meta):
        rid = str(item["id"])
        path = os.path.join(a.data_dir, f"{rid}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue

        i, v = load_iv(path, a.order)
        fs = parse_fs(item)
        df = process_record(i, v, fs, a.window, step, f0=a.f0)

        app = item.get("meta", {}).get("appliance", {})
        df.insert(0, "id", rid)
        df["appliance_type"] = app.get("type", "")
        df["load"] = app.get("load", "")
        df["status"] = app.get("status", "")
        df["location"] = item.get("meta", {}).get("location", "")

        df.to_csv(os.path.join(a.out_dir, f"{rid}_features.csv"), index=False)
        frames.append(df)
        print(f"[ok] id={rid}: {len(df)} windows x {df.shape[1]} columns")

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(a.out_dir, "all_features.csv"), index=False)
        print(f"Saved combined file -> {os.path.join(a.out_dir, 'all_features.csv')}")


if __name__ == "__main__":
    main()