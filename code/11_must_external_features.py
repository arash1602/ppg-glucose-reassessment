"""Step 11 - external test on MUST with the feature-based models (CPU only).

Train on ALL VitalDB readings, test on MUST (never seen in training, different people,
different sensor, 10-s recordings, non-surgical). Models:
  * PPG features only (gradient boosting)
  * PPG features + age, sex, BMI
  * age, sex, BMI only
and constant comparators (VitalDB mean; MUST mean = an oracle no real model can know).
MUST signals and labels are matched by file name.
Run: python 11_must_external_features.py
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample_poly
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
import metrics as M
import ppg

PPG = ["hr", "ibi_sd", "rmssd", "amp_raw", "raw_sd", "t_rise", "width50", "area_ratio", "max_slope", "min_slope",
       "apg_ba", "apg_ea", "dicrotic_ratio", "skew", "kurt", "accept_frac", "cos_mean"]
DEMO = ["age", "sex_m", "bmi"]


def must_table():
    root = C.MUST_DIR
    key = lambda f: re.sub(r"^\D*", "", Path(f).stem)
    mats = {key(f): f for f in glob.glob(str(root / "**" / "*.mat"), recursive=True)}
    labs = {key(f): f for f in glob.glob(str(root / "**" / "label*.csv"), recursive=True)}
    rows = []
    for k in sorted(set(mats) & set(labs)):
        lab = pd.read_csv(labs[k]).iloc[0]
        x = np.asarray(scipy.io.loadmat(mats[k])["signal"], float).ravel()
        fs = int(round(len(x) / 10.0))
        x100 = resample_poly(x, C.FS, fs)
        B, P, q = ppg.extract_beats(ppg.bandpass(x100))
        f = ppg.morphology_features(B, P, x100)
        f.update(q)
        h = float(lab["Height"]) / 100
        rows.append({"file": k, "subjectid": int(lab["ID"]), "glucose": float(lab["Glucose"]),
                     "age": float(lab["Age"]), "sex_m": float(str(lab["Gender"]).lower().startswith("m")),
                     "bmi": float(lab["Weight"]) / h ** 2, "fs_in": fs, **f})
    T = pd.DataFrame(rows)
    print(f"MUST: {len(mats)} signals, {len(labs)} labels, {len(T)} matched; {T.subjectid.nunique()} people; "
          f"fs {T.fs_in.unique().tolist()} Hz; glucose {T.glucose.min():.0f}-{T.glucose.max():.0f} mg/dL; "
          f"median beats per recording {int(T.n_beats.median())}")
    return T


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def main():
    V = pd.read_csv(C.PROC_DIR / "readings.csv")
    T = must_table()
    res = {}
    for name, cols in [("PPG features (trained on VitalDB)", PPG), ("PPG + age, sex, BMI", PPG + DEMO),
                       ("age, sex, BMI only", DEMO)]:
        m = gbm().fit(V[cols].astype(float), V["glucose"])
        res[name] = m.predict(T[cols].astype(float))
    res["constant: VitalDB mean"] = np.full(len(T), V["glucose"].mean())
    res["constant: MUST mean (oracle)"] = np.full(len(T), T["glucose"].mean())
    rows = []
    for name, p in res.items():
        mm = M.basic(T["glucose"], p); mm.update(M.clarke(T["glucose"], p))
        rows.append({"model": name, "rmse": mm["rmse"], "mae": mm["mae"], "mard": mm["mard"], "r2": mm["r2"],
                     "pearson_r": mm["pearson_r"], "clarke_A": mm["clarke_A"],
                     "clarke_AB": mm["clarke_A"] + mm["clarke_B"], "pred_range": f"{p.min():.0f}-{p.max():.0f}"})
    R = pd.DataFrame(rows)
    R.to_csv(C.RESULTS_DIR / "must_external_features.csv", index=False)
    T.to_csv(C.RESULTS_DIR / "must_features.csv", index=False)
    with pd.option_context("display.width", 200):
        print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
