"""Step 13 - train and evaluate the ESP32 models on features computed exactly as the device does
(60-s window centred on each glucose reading), person-level 5-fold, then export to C.

  python 13_device_train.py            # 1) features (cached), 2) CV results, 3) final models + C export
Outputs: processed/device_features.csv, results/device60_free.csv, results/device60_cal.csv,
         esp32/firmware/include/device_model.h, esp32/pc_test/device_reference.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
import data
import device_features as DF

E = __import__("10_multisignal_eval")
PROFILE = ["age", "sex_m", "bmi", "preop_dm"]
FREE = DF.FEATURES + PROFILE
D_FEATS = [f"d_{c}" for c in DF.FEATURES]
CAL = ["ref_glucose", "minutes_since_cal"] + PROFILE + DF.FEATURES + D_FEATS
INC = C.CODE_DIR / "esp32" / "firmware" / "include"


def gbm(n=300):
    return HistGradientBoostingRegressor(max_iter=n, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def compute_features(budget_s=float("inf")):
    """resumable: progress is saved every 200 cases; returns None until all cases are done."""
    out, part = C.PROC_DIR / "device_features.csv", C.PROC_DIR / "device_features_partial.csv"
    if out.exists():
        return pd.read_csv(out)
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {int(r["rid"]) for r in rows}
    groups = [(cid, g) for cid, g in R.groupby("caseid") if not set(g["rid"].astype(int)) <= done]
    t0 = time.time()
    for k, (cid, g) in enumerate(groups):
        z = np.load(C.RAW_DIR / f"{int(cid)}.npz")
        for _, r in g.iterrows():
            f = DF.device_features(DF.centre_window(z["ppg"][int(r["reading_in_case"])]))
            f["rid"] = int(r["rid"]); rows.append(f)
        if (k + 1) % 200 == 0 or k + 1 == len(groups) or time.time() - t0 > budget_s:
            pd.DataFrame(rows).to_csv(part, index=False)
            print(f"  features: {len({int(r['rid']) for r in rows})}/{len(R)} readings", flush=True)
            if time.time() - t0 > budget_s and k + 1 < len(groups):
                return None
    F = pd.DataFrame(rows).drop_duplicates("rid")
    F.to_csv(out, index=False)
    return F


def calib_pairs(R):
    S = R.sort_values(["subjectid", "caseid", "dt"]); fi = S.groupby("subjectid").head(1).index
    first = S.loc[fi].set_index("subjectid"); L = S.drop(fi).copy(); ref = first.reindex(L["subjectid"])
    same = ref["caseid"].to_numpy() == L["caseid"].to_numpy()          # same wear session
    L, ref = L[same].copy(), ref[same]
    L["ref_glucose"] = ref["glucose"].to_numpy()
    L["minutes_since_cal"] = (L["dt"].to_numpy() - ref["dt"].to_numpy()) / 60
    for c in DF.FEATURES:
        L[f"d_{c}"] = L[c].to_numpy() - ref[c].to_numpy()
    return L.reset_index(drop=True)


def export_trees(model, cols, name):
    """HistGradientBoostingRegressor -> flat C arrays. Prediction = base + sum of leaf values."""
    feat, thr, left, right, miss_left, val, roots = [], [], [], [], [], [], []
    for (pred,) in model._predictors:
        nodes = pred.nodes
        off = len(feat)
        roots.append(off)
        for nd in nodes:
            leaf = bool(nd["is_leaf"])
            feat.append(-1 if leaf else int(nd["feature_idx"]))
            thr.append(float(nd["num_threshold"]))
            left.append(0 if leaf else off + int(nd["left"]))
            right.append(0 if leaf else off + int(nd["right"]))
            miss_left.append(int(nd["missing_go_to_left"]))
            val.append(float(nd["value"]))
    base = float(np.ravel(model._baseline_prediction)[0])
    f = lambda a, t, fmt: f"static const {t} {name}_{a[0]}[{len(a[1])}] = {{" + ",".join(fmt(v) for v in a[1]) + "};\n"
    num = lambda v: (repr(float(v)) if np.isfinite(v) else "INFINITY")
    s = [f"// {name}: {len(roots)} trees, {len(feat)} nodes, inputs: {', '.join(cols)}\n",
         f"#define {name.upper()}_N_TREES {len(roots)}\n#define {name.upper()}_N_INPUTS {len(cols)}\n",
         f"static const double {name}_BASE = {base!r};\n",
         f(("ROOT", roots), "int32_t", str), f(("FEAT", feat), "int16_t", str), f(("THR", thr), "double", num),
         f(("LEFT", left), "int32_t", str), f(("RIGHT", right), "int32_t", str),
         f(("MISSL", miss_left), "uint8_t", str), f(("VAL", val), "double", num)]
    return "".join(s), len(feat)


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else float("inf")
    F = compute_features(budget)
    if F is None:
        print("features not finished yet - run again to continue")
        return
    R = E.load()
    R = R.drop(columns=[c for c in DF.FEATURES + ["n_beats"] if c in R.columns]).merge(F, on="rid", how="left")
    print(f"{len(R)} readings; 60-s windows with >= 3 beats: {(R.n_beats >= 3).mean():.1%}")

    print("\n=== ESP32 inputs (60-s features), calibration-free, person-level 5-fold ===")
    sets = [("profile only", PROFILE), ("PPG (60 s) only", DF.FEATURES), ("PPG (60 s) + profile", FREE)]
    p = {n: E.cv(R, c) for n, c in sets}
    E.table(R, p, [n for n, _ in sets], "dev60_free").to_csv(C.RESULTS_DIR / "device60_free.csv", index=False)

    L = calib_pairs(R)
    print("\n=== ESP32 inputs, calibrated (fingerstick at start of the session) ===")
    cs = [("persistence", None), ("fingerstick + time + profile", ["ref_glucose", "minutes_since_cal"] + PROFILE),
          ("+ PPG (60 s) and its change", CAL)]
    p = {n: (L["ref_glucose"].to_numpy() if c is None else E.cv(L, c, base="ref_glucose")) for n, c in cs}
    E.table(L, p, [n for n, _ in cs], "dev60_cal").to_csv(C.RESULTS_DIR / "device60_cal.csv", index=False)

    # ---- final models on all data -> C
    mf = gbm().fit(R[FREE].astype(float), R["glucose"])
    mc = gbm().fit(L[CAL].astype(float), L["glucose"] - L["ref_glucose"])
    hf, nf = export_trees(mf, FREE, "gbf")
    hc, nc = export_trees(mc, CAL, "gbc")
    INC.mkdir(parents=True, exist_ok=True)
    hdr = ["// Auto-generated by 13_device_train.py - do not edit\n#pragma once\n#include <stdint.h>\n#include <math.h>\n",
           f"#define DEV_N_FEATURES {len(DF.FEATURES)}\n#define DEV_N_PROFILE {len(PROFILE)}\n",
           "// feature order: " + ", ".join(DF.FEATURES) + "\n// profile order: " + ", ".join(PROFILE) + "\n",
           "// free model inputs   = features, profile\n",
           "// calib model inputs  = ref_glucose, minutes_since_cal, profile, features, (features - features at calibration)\n",
           hf, hc]
    (INC / "device_model.h").write_text("".join(hdr))
    print(f"\nexported: free model {nf} nodes, calibrated model {nc} nodes -> {INC / 'device_model.h'}")

    # ---- reference vectors for the C check: 40 real windows with their Python features + predictions
    ref, rng = [], np.random.default_rng(0)
    pick = R[R.n_beats >= 3].sample(40, random_state=0)
    for _, r in pick.iterrows():
        z = np.load(C.RAW_DIR / f"{int(r.caseid)}.npz")
        x = DF.centre_window(z["ppg"][int(r.reading_in_case)]).astype(np.float32)
        f = DF.device_features(x)
        prof = [float(r[c]) if pd.notna(r[c]) else float("nan") for c in PROFILE]
        xf = np.array([[f[c] for c in DF.FEATURES] + prof])
        ref.append({"rid": int(r.rid), "label": float(r.glucose), "x": x.tolist(), "profile": prof,
                    "features": [None if not np.isfinite(f[c]) else f[c] for c in DF.FEATURES],
                    "n_beats": f["n_beats"], "pred_free": float(mf.predict(xf)[0])})
    # calibrated check: pairs of windows from the same case (first = calibration)
    for cid, g in L.groupby("caseid"):
        if len(ref) >= 60:
            break
        r = g.iloc[0]
        first = R[(R.caseid == cid)].sort_values("dt").iloc[0]
        z = np.load(C.RAW_DIR / f"{int(cid)}.npz")
        x0 = DF.centre_window(z["ppg"][int(first.reading_in_case)]).astype(np.float32)
        x1 = DF.centre_window(z["ppg"][int(r.reading_in_case)]).astype(np.float32)
        f0, f1 = DF.device_features(x0), DF.device_features(x1)
        if f0["n_beats"] < 3 or f1["n_beats"] < 3:
            continue
        prof = [float(r[c]) if pd.notna(r[c]) else float("nan") for c in PROFILE]
        xc = np.array([[float(first.glucose), float(r.minutes_since_cal)] + prof + [f1[c] for c in DF.FEATURES]
                       + [f1[c] - f0[c] for c in DF.FEATURES]])
        ref.append({"rid": int(r.rid), "label": float(r.glucose), "x": x1.tolist(), "x_cal": x0.tolist(),
                    "cal_glucose": float(first.glucose), "minutes_since_cal": float(r.minutes_since_cal),
                    "profile": prof, "features": [None if not np.isfinite(f1[c]) else f1[c] for c in DF.FEATURES],
                    "n_beats": f1["n_beats"], "pred_free": float(mf.predict(np.array([[f1[c] for c in DF.FEATURES] + prof]))[0]),
                    "pred_cal": float(mc.predict(xc)[0] + float(first.glucose))})
    out = C.CODE_DIR / "esp32" / "pc_test" / "device_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref))
    print(f"reference vectors: {len(ref)} windows ({sum('x_cal' in r for r in ref)} calibrated) -> {out}")


if __name__ == "__main__":
    main()
