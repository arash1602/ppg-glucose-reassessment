"""Step 19 - MUST, like-for-like comparison (Table 14): every model scored on the MUST recordings
that the deep networks could process (>= 3 accepted beats). Needs results/must_features.csv
(step 11) and results/preds/must_*_beat_subject.csv (step 06). Output: results/must_matched.csv
  python 19_must_matched.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C

PPG = ["hr", "ibi_sd", "rmssd", "amp_raw", "raw_sd", "t_rise", "width50", "area_ratio", "max_slope", "min_slope",
       "apg_ba", "apg_ea", "dicrotic_ratio", "skew", "kurt", "accept_frac", "cos_mean"]
DEMO = ["age", "sex_m", "bmi"]


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def main(B=1000, seed=0):
    V = pd.read_csv(C.PROC_DIR / "readings.csv")
    T = pd.read_csv(C.RESULTS_DIR / "must_features.csv", dtype={"file": str})
    deep = {n: pd.read_csv(C.RESULTS_DIR / "preds" / f"must_{k}_beat_subject.csv", dtype={"file": str})
            for n, k in [("ResNet34-1D (person split)", "resnet34"), ("TinyCNN (person split)", "tiny")]}
    files = set.intersection(*[set(d.file) for d in deep.values()])
    M = T[T.file.isin(files)].reset_index(drop=True)
    print(f"matched set: {len(M)} recordings, {M.subjectid.nunique()} people")
    preds = {}
    for name, cols in [("PPG features (GBM)", PPG), ("PPG + age, sex, BMI (GBM)", PPG + DEMO), ("Age, sex, BMI (GBM)", DEMO)]:
        preds[name] = gbm().fit(V[cols].astype(float), V["glucose"]).predict(M[cols].astype(float))
    preds["Constant: VitalDB mean"] = np.full(len(M), V["glucose"].mean())
    for n, d in deep.items():
        preds[n] = M[["file"]].merge(d[["file", "pred"]], on="file")["pred"].to_numpy()
    y = M["glucose"].to_numpy(); g = M["subjectid"].to_numpy(); subs = np.unique(g)
    idx = {s: np.where(g == s)[0] for s in subs}; rng = np.random.default_rng(seed)
    draws = [np.concatenate([idx[s] for s in rng.choice(subs, len(subs))]) for _ in range(B)]
    rm = lambda p, r: np.sqrt(np.mean((y[r] - p[r]) ** 2))
    c = preds["Constant: VitalDB mean"]; rows = []
    for n, p in preds.items():
        bs = [rm(p, r) for r in draws]; dv = [rm(p, r) - rm(c, r) for r in draws]
        rows.append({"model": n, "rmse": rm(p, np.arange(len(y))),
                     "ci": f"{np.percentile(bs, 2.5):.1f}-{np.percentile(bs, 97.5):.1f}",
                     "mard": np.mean(np.abs(p - y) / y) * 100, "r": np.corrcoef(p, y)[0, 1] if np.std(p) > 0 else np.nan,
                     "vs_const": rm(p, np.arange(len(y))) - rm(c, np.arange(len(y))),
                     "vs_ci": f"[{np.percentile(dv, 2.5):+.1f}, {np.percentile(dv, 97.5):+.1f}]"})
    R = pd.DataFrame(rows); R.to_csv(C.RESULTS_DIR / "must_matched.csv", index=False)
    print(R.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
