"""Step 10 - what does each signal group add? Person-level 5-fold CV (same folds as everything else).

Tiers, each adding one group to the previous:
  T0 clinical       : age, sex, BMI, diabetes, ASA, emergency surgery
  T1 + time         : minutes since anaesthesia start
  T2 + PPG          : 17 PPG waveform features (the only signal the original paper used)
  T3 + ECG/arterial : heart rate, HRV, blood pressure, pulse arrival times
  T4 + monitor      : temperature, SpO2, EtCO2, BIS (+ 30-min changes)
  T5 + drugs        : anaesthetic and vasoactive infusion rates
plus "PPG-wearable": clinical + time + PPG + temperature + HR only (signals a wrist device could have).

Calibrated variant: first reading per person = fingerstick; later readings predicted (delta model),
with changes since the fingerstick of every signal feature.

Each tier is compared with the tier before it by a paired bootstrap over PEOPLE (1000 resamples).
Run (CPU only): python 10_multisignal_eval.py
Out: results/multisignal_tiers.csv, results/multisignal_calibrated.csv, results/preds/ms_*.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
import data
import metrics as M

PPG = ["hr", "ibi_sd", "rmssd", "amp_raw", "raw_sd", "t_rise", "width50", "area_ratio", "max_slope", "min_slope",
       "apg_ba", "apg_ea", "dicrotic_ratio", "skew", "kurt", "accept_frac", "cos_mean"]
CLIN = ["age", "sex_m", "bmi", "preop_dm", "asa", "emop"]
TIME = ["t_since_anes_min"]
ECGART = ["ecg_hr", "ecg_sdnn", "ecg_rmssd", "sbp", "dbp", "map_wave", "pp", "dpdt_max", "pat_ppg", "pat_art"]
MON = [f"{v}{s}" for v in ["temp", "spo2", "etco2", "bis", "hr_mon", "mbp_mon", "nibp_mbp"] for s in ["", "_chg30"]]
DRUGS = ["propofol", "remifentanil", "norepinephrine", "phenylephrine", "epinephrine", "dopamine", "dobutamine",
         "vasopressin"]
TIERS = [("T0 clinical", CLIN), ("T1 + time in surgery", CLIN + TIME), ("T2 + PPG", CLIN + TIME + PPG),
         ("T3 + ECG & arterial", CLIN + TIME + PPG + ECGART), ("T4 + monitor (temp, SpO2, EtCO2, BIS)",
         CLIN + TIME + PPG + ECGART + MON), ("T5 + drug infusions", CLIN + TIME + PPG + ECGART + MON + DRUGS)]
WEARABLE = ("Wearable set: clinical + time + PPG + temp + HR", CLIN + TIME + PPG + ["temp", "temp_chg30", "hr_mon"])


def load():
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    cases = pd.read_csv(C.DATA_DIR / "cases.csv")
    extra = [c for c in ["preop_dm", "asa", "emop", "anestart"] if c not in R.columns]
    R = R.merge(cases[["caseid"] + extra], on="caseid", how="left")
    R["t_since_anes_min"] = (R["dt"] - R["anestart"]) / 60
    ms = pd.read_csv(C.PROC_DIR / "multisignal.csv")
    R = R.merge(ms, on="rid", how="left", suffixes=("", "_ms"))
    R["fold"] = data.subject_folds(R)
    # sensor artefacts -> missing
    for c, lo, hi in [("temp", 30, 42), ("bis", 1, 100), ("spo2", 50, 100), ("etco2", 5, 80),
                      ("ecg_hr", 30, 200), ("hr_mon", 30, 200), ("mbp_mon", 25, 200), ("nibp_mbp", 25, 200)]:
        if c in R:
            bad = (R[c] < lo) | (R[c] > hi)
            R.loc[bad, c] = np.nan
            if f"{c}_chg30" in R:
                R.loc[bad, f"{c}_chg30"] = np.nan
    # out-of-fold prediction of the multi-beat deep model (08_improve.py), same person folds -> no leakage
    ds = C.RESULTS_DIR / "preds" / "improve_deepset_demo.csv"
    if ds.exists():
        R = R.merge(pd.read_csv(ds)[["rid", "pred"]].rename(columns={"pred": "deepset_pred"}), on="rid", how="left")
    for c in sum([t[1] for t in TIERS], []) + ["deepset_pred"]:
        if c not in R:
            R[c] = np.nan
    return R.reset_index(drop=True)


def usable(P, cols, min_cov=0.02):
    """drop features (almost) never recorded - an all-missing column breaks the tree binning"""
    return [c for c in cols if P[c].notna().mean() >= min_cov]


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def cv(P, cols, base=None):
    cols = usable(P, cols)
    out = np.full(len(P), np.nan)
    y = P["glucose"].to_numpy() - (P[base].to_numpy() if base else 0)
    for k in range(C.N_FOLDS):
        tr, te = (P.fold != k).to_numpy(), (P.fold == k).to_numpy()
        m = gbm().fit(P.loc[tr, cols].astype(float), y[tr])
        out[te] = m.predict(P.loc[te, cols].astype(float)) + (P[base].to_numpy()[te] if base else 0)
    return out


def paired(P, preds, pairs, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    g = P["subjectid"].to_numpy(); u = np.unique(g); idx = {s: np.where(g == s)[0] for s in u}
    y = P["glucose"].to_numpy()
    rm = lambda p, r: np.sqrt(np.mean((y[r] - p[r]) ** 2))
    res = {k: [] for k in preds}; dif = {p: [] for p in pairs}
    for _ in range(n):
        r = np.concatenate([idx[s] for s in rng.choice(u, len(u))])
        v = {k: rm(p, r) for k, p in preds.items()}
        for k in preds: res[k].append(v[k])
        for a, b in pairs: dif[(a, b)].append(v[a] - v[b])
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {k: q(v) for k, v in res.items()}, {k: q(v) for k, v in dif.items()}


def table(P, preds, order, tag):
    pairs = [(order[i], order[i - 1]) for i in range(1, len(order))]
    ci, dci = paired(P, preds, pairs)
    rows = []
    for i, k in enumerate(order):
        D = P[["rid", "subjectid", "caseid", "dt", "fold"]].assign(y=P["glucose"].to_numpy(), pred=preds[k])
        m = M.all_metrics(D)
        d = dci.get((k, order[i - 1])) if i else None
        rows.append({"model": k, "n": m["n"], "people": P.subjectid.nunique(), "rmse": round(m["rmse"], 2),
                     "rmse_ci": f"{ci[k][0]:.1f}-{ci[k][1]:.1f}", "mae": round(m["mae"], 2), "mard": round(m["mard"], 2),
                     "r2": round(m["r2"], 3), "clarke_A": round(m["clarke_A"], 1), "within_person_r": round(m["ws_r"], 3),
                     "delta_rmse_vs_previous": None if d is None else f"{np.sqrt(np.mean((P.glucose - preds[k]) ** 2)) - np.sqrt(np.mean((P.glucose - preds[order[i-1]]) ** 2)):+.2f} [{d[0]:+.2f}, {d[1]:+.2f}]"})
        slug = "".join(ch if ch.isalnum() else "_" for ch in k.lower())[:40]
        (C.RESULTS_DIR / "preds").mkdir(exist_ok=True)
        D.to_csv(C.RESULTS_DIR / "preds" / f"ms_{tag}_{slug}.csv", index=False)
    T = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 48):
        print(T.drop(columns=["n", "people"]).to_string(index=False))
    return T


def main():
    R = load()
    feats = sorted(set(sum([t[1] for t in TIERS], [])))
    cov = R[ECGART + MON + DRUGS].notna().mean()
    print(f"{len(R)} readings, {R.subjectid.nunique()} people. Feature coverage: ECG HR {cov['ecg_hr']:.0%}, "
          f"arterial SBP {cov['sbp']:.0%}, temperature {cov['temp']:.0%}, BIS {cov['bis']:.0%}, propofol {cov['propofol']:.0%}")

    print("\n=== Calibration-free (person-level 5-fold) ===")
    preds = {name: cv(R, cols) for name, cols in TIERS}
    order = [t[0] for t in TIERS]
    T = table(R, preds, order, "free")
    extra = [order[1]]
    preds[WEARABLE[0]] = cv(R, WEARABLE[1]); extra.append(WEARABLE[0])
    if R["deepset_pred"].notna().any():
        wd = WEARABLE[0] + " + multi-beat CNN"
        preds[wd] = cv(R, WEARABLE[1] + ["deepset_pred"]); extra.append(wd)
        best = order[-1] + " + multi-beat CNN"
        preds[best] = cv(R, TIERS[-1][1] + ["deepset_pred"])
        print("\n--- wearable-type inputs (vs clinical + time), and the multi-beat CNN as an extra input ---")
        w = table(R, preds, extra, "free_wear").iloc[1:]
        b = table(R, preds, [order[-1], best], "free_best").iloc[1:]
        pd.concat([T, w, b]).to_csv(C.RESULTS_DIR / "multisignal_tiers.csv", index=False)
    else:
        w = table(R, preds, extra, "free_wear").iloc[1:]
        pd.concat([T, w]).to_csv(C.RESULTS_DIR / "multisignal_tiers.csv", index=False)

    print("\n=== Calibrated: first reading = fingerstick, later readings scored ===")
    S = R.sort_values(["subjectid", "caseid", "dt"])
    first_idx = S.groupby("subjectid").head(1).index
    first = S.loc[first_idx].set_index("subjectid")
    L = S.drop(first_idx).copy()
    ref = first.reindex(L["subjectid"])
    L["ref_glucose"] = ref["glucose"].to_numpy()
    same = ref["caseid"].to_numpy() == L["caseid"].to_numpy()
    L["elapsed_min"] = np.where(same, (L["dt"].to_numpy() - ref["dt"].to_numpy()) / 60, np.nan)
    L["same_case"] = same.astype(float)
    sig = PPG + ECGART + MON + DRUGS
    for c in sig:
        L[f"d_{c}"] = L[c].to_numpy() - ref[c].to_numpy()
    L = L.reset_index(drop=True)
    base = ["ref_glucose", "elapsed_min", "same_case"] + CLIN + TIME
    cal = [("persistence (no model)", None),
           ("C0 fingerstick + time + clinical", base),
           ("C1 + PPG change", base + PPG + [f"d_{c}" for c in PPG]),
           ("C2 + ECG & arterial change", base + PPG + ECGART + [f"d_{c}" for c in PPG + ECGART]),
           ("C3 + monitor & drugs change", base + sig + [f"d_{c}" for c in sig])]
    cp = {name: (L["ref_glucose"].to_numpy() if cols is None else cv(L, cols, base="ref_glucose")) for name, cols in cal}
    T2 = table(L, cp, [c[0] for c in cal], "calib")
    T2.to_csv(C.RESULTS_DIR / "multisignal_calibrated.csv", index=False)
    print("\nsaved results/multisignal_tiers.csv and results/multisignal_calibrated.csv")


if __name__ == "__main__":
    main()
