"""Step 4 - non-deep baselines on the SAME person-level folds.

  mean          : predict the training-set mean glucose (no information at all)
  demo          : ridge regression on age, sex, BMI
  preop_carry   : predict the patient's pre-operative glucose (a real clinical prior)
  ppg_gbm       : gradient boosting on hand-crafted PPG morphology/HR features
  ppg_demo_gbm  : the same plus age/sex/BMI
Also runs ppg_gbm with a reading-random split to show how much person leakage
inflates even a simple model.
Output: results/preds/baseline_<name>_<split>.csv and results/runs.jsonl
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

import config as C
import data
import metrics as M

PPG_FEATS = ["hr", "ibi_sd", "rmssd", "amp_raw", "raw_sd", "t_rise", "width50", "area_ratio",
             "max_slope", "min_slope", "apg_ba", "apg_ea", "dicrotic_ratio", "skew", "kurt",
             "accept_frac", "cos_mean"]


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15,
                                         random_state=C.SEED)


def run_cv(R, folds, name, fit_predict, split):
    preds = np.full(len(R), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        preds[te] = fit_predict(R[tr], R[te])
    P = R[["rid", "subjectid", "caseid", "dt"]].copy()
    P["y"], P["pred"], P["fold"] = R["glucose"].values, preds, folds
    P = P.dropna(subset=["pred"])
    out = C.RESULTS_DIR / "preds"
    out.mkdir(exist_ok=True)
    run = f"baseline_{name}_{split}"
    P.to_csv(out / f"{run}.csv", index=False)
    m = M.all_metrics(P)
    with open(C.RESULTS_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps({"run": run, "split": split, "per_reading": m}) + "\n")
    print(f"{run:40s} RMSE {m['rmse']:6.1f}  MAE {m['mae']:5.1f}  MARD {m['mard']:5.1f}%  "
          f"R2 {m['r2']:+.3f}  ClarkeA {m['clarke_A']:5.1f}%  within-subj r {m['ws_r']:+.2f}  (n={m['n']})")


def main():
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    f_subj = data.subject_folds(R)
    rng = np.random.default_rng(C.SEED)
    f_read = rng.integers(0, C.N_FOLDS, len(R))  # reading-random (person leakage)

    mean_fp = lambda tr, te: np.full(len(te), tr["glucose"].mean())

    def demo_fp(tr, te):
        Xtr, Xte = data.demo_matrix(tr, ref=tr), data.demo_matrix(te, ref=tr)
        return RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xtr, tr["glucose"]).predict(Xte)

    def preop_fp(tr, te):
        if "preop_gluc" not in te:
            return np.full(len(te), np.nan)
        return te["preop_gluc"].fillna(tr["glucose"].mean()).to_numpy()

    def gbm_fp(cols):
        return lambda tr, te: gbm().fit(tr[cols], tr["glucose"]).predict(te[cols])

    run_cv(R, f_subj, "mean", mean_fp, "subject")
    run_cv(R, f_subj, "demo", demo_fp, "subject")
    if "preop_gluc" in R:
        run_cv(R, f_subj, "preop_carry", preop_fp, "subject")
    run_cv(R, f_subj, "ppg_gbm", gbm_fp(PPG_FEATS), "subject")
    run_cv(R, f_subj, "ppg_demo_gbm", gbm_fp(PPG_FEATS + data.DEMO_COLS), "subject")
    run_cv(R, f_read, "ppg_gbm", gbm_fp(PPG_FEATS), "readingrandom")


if __name__ == "__main__":
    main()
