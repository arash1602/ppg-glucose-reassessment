"""Step 5 - personalised (calibrated) estimation, the realistic use case.

For people with >=2 readings, order readings in time. The FIRST reading is used
as a fingerstick calibration; we score only the later readings.
  persistence : later glucose = calibration value (no PPG at all)  <- the bar to beat
  model_raw   : model prediction, uncalibrated
  model_offset: model prediction + (calibration value - model prediction at calibration)
  model_delta : calibration value + shrunk model change, shrink factor fitted on training folds
Input: any per-reading prediction file from a SUBJECT split (default: all of them).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import metrics as M


def personalise(P):
    P = P.sort_values(["subjectid", "caseid", "dt"]).copy()
    P = P.groupby("subjectid").filter(lambda g: len(g) >= 2)
    first = P.groupby("subjectid").head(1).set_index("subjectid")
    later = P[~P.index.isin(P.groupby("subjectid").head(1).index)].copy()
    later["cal_y"] = later["subjectid"].map(first["y"])
    later["cal_p"] = later["subjectid"].map(first["pred"])
    later["persistence"] = later["cal_y"]
    later["model_raw"] = later["pred"]
    later["model_offset"] = later["pred"] + (later["cal_y"] - later["cal_p"])
    # shrinkage of predicted change, fitted out-of-fold (fold k fitted on the other folds)
    later["model_delta"] = np.nan
    for k in later["fold"].unique():
        tr = later[later.fold != k]
        dp, dy = tr["pred"] - tr["cal_p"], tr["y"] - tr["cal_y"]
        b = float(np.clip((dp * dy).sum() / max((dp ** 2).sum(), 1e-9), 0, 1)) if len(tr) else 0.0
        m = later.fold == k
        later.loc[m, "model_delta"] = later.loc[m, "cal_y"] + b * (later.loc[m, "pred"] - later.loc[m, "cal_p"])
    return later


def main():
    files = sys.argv[1:] or sorted(glob.glob(str(C.RESULTS_DIR / "preds" / "*_subject*.csv")))
    rows = []
    for f in files:
        P = pd.read_csv(f)
        L = personalise(P)
        if L.empty:
            continue
        for strat in ["persistence", "model_raw", "model_offset", "model_delta"]:
            m = M.basic(L["y"], L[strat])
            m.update(M.clarke(L["y"], L[strat]))
            rows.append({"source": Path(f).stem, "strategy": strat, "n_subjects": L.subjectid.nunique(), **m})
    T = pd.DataFrame(rows)
    T.to_csv(C.RESULTS_DIR / "personalisation.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(T[["source", "strategy", "n", "n_subjects", "rmse", "mae", "mard", "clarke_A"]].round(2))


if __name__ == "__main__":
    main()
