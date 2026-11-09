"""Step 12 - models restricted to inputs a wearable device actually has, person-level 5-fold.
  calibration-free: PPG features + age, sex, BMI, diabetes (+ skin/body temperature)
  calibrated     : fingerstick value + minutes since it + the same + PPG change since the fingerstick
Picks the device model; saves the final models trained on ALL data for the ESP32 export.
"""
import pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import config as C, data
E = __import__("10_multisignal_eval")

PPG = E.PPG
PROFILE = ["age", "sex_m", "bmi", "preop_dm"]
SETS = [("profile only (age, sex, BMI, diabetes)", PROFILE),
        ("PPG only", PPG),
        ("PPG + profile", PPG + PROFILE),
        ("PPG + profile + temperature", PPG + PROFILE + ["temp"])]


def gbm(n=300):
    return HistGradientBoostingRegressor(max_iter=n, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def main():
    R = E.load()
    print("=== device, calibration-free ===")
    preds = {n: E.cv(R, c) for n, c in SETS}
    E.table(R, preds, [n for n, _ in SETS], "device_free").to_csv(C.RESULTS_DIR / "device_free.csv", index=False)

    S = R.sort_values(["subjectid", "caseid", "dt"]); fi = S.groupby("subjectid").head(1).index
    first = S.loc[fi].set_index("subjectid"); L = S.drop(fi).copy(); ref = first.reindex(L["subjectid"])
    same = ref["caseid"].to_numpy() == L["caseid"].to_numpy()
    L = L[same].copy(); ref = ref[same]                      # a device is calibrated within one wear session
    L["ref_glucose"] = ref["glucose"].to_numpy()
    L["minutes_since_cal"] = (L["dt"].to_numpy() - ref["dt"].to_numpy()) / 60
    for c in PPG + ["temp"]:
        L[f"d_{c}"] = L[c].to_numpy() - ref[c].to_numpy()
    L = L.reset_index(drop=True)
    base = ["ref_glucose", "minutes_since_cal"] + PROFILE
    cal = [("persistence", None), ("fingerstick + time + profile (no PPG)", base),
           ("+ PPG and PPG change", base + PPG + [f"d_{c}" for c in PPG]),
           ("+ temperature and its change", base + PPG + [f"d_{c}" for c in PPG] + ["temp", "d_temp"])]
    print("\n=== device, calibrated (same wear session) ===")
    cp = {n: (L["ref_glucose"].to_numpy() if c is None else E.cv(L, c, base="ref_glucose")) for n, c in cal}
    E.table(L, cp, [n for n, _ in cal], "device_cal").to_csv(C.RESULTS_DIR / "device_cal.csv", index=False)

    # final models on ALL data (for the device)
    out = {}
    cols = PPG + PROFILE
    out["free"] = {"cols": cols, "model": gbm().fit(R[cols].astype(float), R["glucose"]), "base": None}
    cols = base + PPG + [f"d_{c}" for c in PPG]
    out["cal"] = {"cols": cols, "model": gbm().fit(L[cols].astype(float), L["glucose"] - L["ref_glucose"]),
                  "base": "ref_glucose"}
    (C.DATA_DIR / "models").mkdir(exist_ok=True)
    pickle.dump(out, open(C.DATA_DIR / "models" / "device_gbm.pkl", "wb"))
    for k, v in out.items():
        n_nodes = sum(len(p[0].nodes) for p in v["model"]._predictors)
        print(f"device model '{k}': {len(v['cols'])} inputs, {len(v['model']._predictors)} trees, {n_nodes} nodes")


if __name__ == "__main__":
    main()
