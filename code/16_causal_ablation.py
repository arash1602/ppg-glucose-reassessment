"""Reviewer analyses (step 16).

A. Causal (pre-measurement) PPG: the PPG features are recomputed from signal BEFORE the glucose draw only
   (last 8, 2 and 1 min) and compared with the centred +/-8 min window, without and with calibration.
B. Order-independent ablation: the full multi-signal model, and the same model with each signal group left out.

  python 16_causal_ablation.py          (feature step uses several CPU cores; ~5 min, cached)
Output: results/causal_windows.csv, results/ablation_logo.csv, log on screen
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as C
import ppg

E = __import__("10_multisignal_eval")
PPGF = E.PPG
FS = C.FS
CENTER = C.HALF_WINDOW_S * FS                   # sample index of the glucose draw (48,000)
WINDOWS = {"pre8": 8 * 60 * FS, "pre2": 2 * 60 * FS, "pre1": 60 * FS}


def feats(raw):
    xf = ppg.bandpass(raw)
    B, P, q = ppg.extract_beats(xf)
    f = ppg.morphology_features(B, P, raw)
    f["accept_frac"], f["cos_mean"] = q["accept_frac"], q["cos_mean"]
    f["n_beats"] = q["n_beats"]
    return f


def case_job(args):
    cid, items = args
    z = np.load(C.RAW_DIR / f"{int(cid)}.npz")
    out = []
    for rid_, ric in items:
        x = z["ppg"][int(ric)].astype(float)
        row = {"rid": int(rid_)}
        for tag, n in WINDOWS.items():
            seg = x[CENTER - n:CENTER]
            if np.isnan(seg).mean() > 0.2:
                continue
            seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy()
            try:
                f = feats(seg)
            except Exception:
                continue
            if f["n_beats"] < 3:
                continue
            row.update({f"{tag}_{k}": v for k, v in f.items() if k in PPGF})
        out.append(row)
    return out


def causal_features():
    path = C.PROC_DIR / "ppg_causal_features.csv"
    if path.exists():
        return pd.read_csv(path)
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    jobs = [(cid, list(zip(g["rid"], g["reading_in_case"]))) for cid, g in R.groupby("caseid")]
    rows, t0 = [], time.time()
    with Pool(max(1, min(6, (os.cpu_count() or 2) - 2))) as pool:
        for k, res in enumerate(pool.imap_unordered(case_job, jobs, chunksize=8)):
            rows += res
            if (k + 1) % 500 == 0:
                print(f"  causal features: {k + 1}/{len(jobs)} cases, {time.time() - t0:.0f} s", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(path, index=False)
    return F


def calib_frame(R, ppg_cols):
    S = R.sort_values(["subjectid", "caseid", "dt"])
    first_idx = S.groupby("subjectid").head(1).index
    first = S.loc[first_idx].set_index("subjectid")
    L = S.drop(first_idx).copy()
    ref = first.reindex(L["subjectid"])
    L["ref_glucose"] = ref["glucose"].to_numpy()
    same = ref["caseid"].to_numpy() == L["caseid"].to_numpy()
    L["elapsed_min"] = np.where(same, (L["dt"].to_numpy() - ref["dt"].to_numpy()) / 60, np.nan)
    L["same_case"] = same.astype(float)
    for c in ppg_cols:
        L[f"d_{c}"] = L[c].to_numpy() - ref[c].to_numpy()
    return L.reset_index(drop=True)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    R = E.load()
    F = causal_features()
    R = R.merge(F, on="rid", how="left")
    base = E.CLIN + E.TIME
    print("\n=== A. causal PPG windows (person-level 5-fold) ===")
    sets = {"centred +/-8 min (main analysis)": PPGF}
    for tag in WINDOWS:
        sets[{"pre8": "8 min before the draw", "pre2": "2 min before", "pre1": "1 min before"}[tag]] = [f"{tag}_{c}" for c in PPGF]
    preds = {"clinical + time (no PPG)": E.cv(R, base)}
    for name, cols in sets.items():
        preds[name] = E.cv(R, base + cols)
    pairs = [(k, "clinical + time (no PPG)") for k in sets]
    ci, dci = E.paired(R, preds, pairs)
    y = R["glucose"].to_numpy()
    rows = []
    for k, p in preds.items():
        d = dci.get((k, "clinical + time (no PPG)"))
        rows.append({"setting": "no calibration", "ppg_window": k, "n": len(y), "rmse": rmse(y, p),
                     "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                     "ppg_gain": None if d is None else rmse(y, p) - rmse(y, preds["clinical + time (no PPG)"]),
                     "ppg_gain_ci": None if d is None else f"[{d[0]:+.2f}, {d[1]:+.2f}]"})
    # calibrated
    all_ppg = sum(sets.values(), [])
    L = calib_frame(R, all_ppg)
    cbase = ["ref_glucose", "elapsed_min", "same_case"] + base
    cp = {"clinical + time (no PPG)": E.cv(L, cbase, base="ref_glucose")}
    for name, cols in sets.items():
        cp[name] = E.cv(L, cbase + cols + [f"d_{c}" for c in cols], base="ref_glucose")
    ci, dci = E.paired(L, cp, pairs)
    yl = L["glucose"].to_numpy()
    for k, p in cp.items():
        d = dci.get((k, "clinical + time (no PPG)"))
        rows.append({"setting": "one calibration value", "ppg_window": k, "n": len(yl), "rmse": rmse(yl, p),
                     "r2": 1 - np.sum((yl - p) ** 2) / np.sum((yl - yl.mean()) ** 2),
                     "ppg_gain": None if d is None else rmse(yl, p) - rmse(yl, cp["clinical + time (no PPG)"]),
                     "ppg_gain_ci": None if d is None else f"[{d[0]:+.2f}, {d[1]:+.2f}]"})
    T = pd.DataFrame(rows)
    T.to_csv(C.RESULTS_DIR / "causal_windows.csv", index=False)
    print(T.round(3).to_string(index=False))

    print("\n=== B. leave-one-group-out ablation (no calibration) ===")
    groups = {"clinical": E.CLIN, "time since anaesthesia": E.TIME, "PPG": PPGF, "ECG + arterial": E.ECGART,
              "monitor": E.MON, "drugs": E.DRUGS}
    full = sum(groups.values(), [])
    ap = {"full model": E.cv(R, full)}
    for g, cols in groups.items():
        ap[f"without {g}"] = E.cv(R, [c for c in full if c not in cols])
    pairs = [(f"without {g}", "full model") for g in groups]
    ci, dci = E.paired(R, ap, pairs)
    rows = []
    for k, p in ap.items():
        d = dci.get((k, "full model"))
        rows.append({"model": k, "rmse": rmse(y, p), "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                     "rmse_increase": None if d is None else rmse(y, p) - rmse(y, ap["full model"]),
                     "ci": None if d is None else f"[{d[0]:+.2f}, {d[1]:+.2f}]"})
    A = pd.DataFrame(rows)
    A.to_csv(C.RESULTS_DIR / "ablation_logo.csv", index=False)
    print(A.round(3).to_string(index=False))
    print("\nDONE")


if __name__ == "__main__":
    main()
