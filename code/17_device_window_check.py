"""Step 17 - device model by PPG window (paper Table 16).

Re-evaluates the wearable-input device model with features from the last K one-minute windows
before the recorded sample time (K = 1, 3, 5, 10), retrospectively over all 16 min, and with the
60-s window centred on the recorded time. Needs the outputs of steps 13 and 14
(processed/device_features.csv, processed/device_features_16min.npz).
Output: results/device_window_check.csv
  python 17_device_window_check.py
"""
import numpy as np
import pandas as pd

import config as C
import device_features as DF

E = __import__("10_multisignal_eval")
T13 = __import__("13_device_train")
T14 = __import__("14_device_avg")


def evaluate(R, name, A):
    P = R.merge(A, on="rid", how="left"); y = P.glucose.to_numpy()
    p = E.cv(P, DF.FEATURES + T13.PROFILE)
    L = T13.calib_pairs(P); base = ["ref_glucose", "minutes_since_cal"] + T13.PROFILE
    p0 = E.cv(L, base, base="ref_glucose"); p1 = E.cv(L, T13.CAL, base="ref_glucose"); yl = L.glucose.to_numpy()
    _, dci = E.paired(L, {"no": p0, "ppg": p1}, [("ppg", "no")], n=500)
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    row = {"window": name, "rmse_free": rm(y, p), "r2_free": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
           "n_cal": len(L), "rmse_cal_no_ppg": rm(yl, p0), "rmse_cal_ppg": rm(yl, p1),
           "ppg_gain": rm(yl, p1) - rm(yl, p0), "gain_lo": dci[("ppg", "no")][0], "gain_hi": dci[("ppg", "no")][1]}
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


def main():
    rid, F, nb = T14.per_minute_features(float("inf"))
    R = E.load(); R = R.drop(columns=[c for c in DF.FEATURES + ["n_beats"] if c in R.columns])
    rows = []
    for K in [1, 3, 5, 10, 16]:
        A = pd.DataFrame(T14.averaged(F, K), columns=DF.FEATURES); A["rid"] = rid
        rows.append(evaluate(R, f"last {K} min" if K < 16 else "all 16 min (retrospective)", A))
    Fc = pd.read_csv(C.PROC_DIR / "device_features.csv")[["rid"] + DF.FEATURES]
    rows.append(evaluate(R, "60 s centred on recorded time (retrospective)", Fc))
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "device_window_check.csv", index=False)


if __name__ == "__main__":
    main()
