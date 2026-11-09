"""Step 14 - does averaging device features over several minutes help? (firmware improvement)

For every glucose reading, the device features are computed on each of the 16 consecutive 60-s
windows of its +/-8 min recording (exactly as on the ESP32). The device would keep a running mean
of the last K windows; here K = 1, 3, 5, 10, 16 (centred on the reading) is evaluated with the same
person-level folds, calibration-free and calibrated. The best K is then trained on all data and
exported to the firmware (device_model.h, K written as DEV_AVG_WINDOWS).

  python 14_device_avg.py      (feature step uses all CPU cores)
"""
import json
import sys
import time

import numpy as np
import pandas as pd

import config as C
import device_features as DF

E = __import__("10_multisignal_eval")
T13 = __import__("13_device_train")
PROFILE = T13.PROFILE
KS = [1, 3, 5, 10, 16]
NW = 16


def _case_job(args):
    cid, items = args
    z = np.load(C.RAW_DIR / f"{int(cid)}.npz")
    out = []
    for rid_, ric in items:
        x = z["ppg"][int(ric)]
        F = np.full((NW, len(DF.FEATURES)), np.nan); nb = np.zeros(NW, int)
        for w in range(NW):
            f = DF.device_features(x[w * DF.WIN:(w + 1) * DF.WIN])
            nb[w] = f["n_beats"]
            if f["n_beats"] >= 3:
                F[w] = [f[c] for c in DF.FEATURES]
        out.append((int(rid_), F, nb))
    return out


def per_minute_features(budget_s):
    out = C.PROC_DIR / "device_features_16min.npz"
    if out.exists():
        z = np.load(out)
        return z["rid"], z["F"], z["nb"]
    import os
    from multiprocessing import Pool
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    jobs = [(cid, list(zip(g["rid"], g["reading_in_case"]))) for cid, g in R.groupby("caseid")]
    rids, Fs, nbs, t0 = [], [], [], time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for k, res in enumerate(pool.imap_unordered(_case_job, jobs, chunksize=8)):
            for r_, F, nb in res:
                rids.append(r_); Fs.append(F); nbs.append(nb)
            if (k + 1) % 300 == 0:
                print(f"  per-minute features: {k + 1}/{len(jobs)} cases, {time.time() - t0:.0f} s", flush=True)
    np.savez(out, rid=np.array(rids), F=np.array(Fs), nb=np.array(nbs))
    return np.array(rids), np.array(Fs), np.array(nbs)


def averaged(F, K):
    """running mean of the K windows ending at the reading's centre (windows 8-K .. 7), NaN-aware"""
    end = NW // 2
    sl = F[:, max(0, end - K):end] if K < NW else F
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(sl, axis=1)


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else float("inf")
    res = per_minute_features(budget)
    if res is None:
        return
    rid, F, nb = res
    R = E.load()
    R = R.drop(columns=[c for c in DF.FEATURES + ["n_beats"] if c in R.columns])
    rows_free, rows_cal, store = [], [], {}
    for K in KS:
        A = pd.DataFrame(averaged(F, K), columns=DF.FEATURES); A["rid"] = rid
        P = R.merge(A, on="rid", how="left")
        p = E.cv(P, DF.FEATURES + PROFILE)
        y = P["glucose"].to_numpy()
        rows_free.append({"K_minutes": K, "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                          "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))})
        L = T13.calib_pairs(P)
        base = ["ref_glucose", "minutes_since_cal"] + PROFILE
        p0 = E.cv(L, base, base="ref_glucose")
        p1 = E.cv(L, T13.CAL, base="ref_glucose")
        ci, dci = E.paired(L, {"no": p0, "ppg": p1}, [("ppg", "no")], n=500)
        yl = L["glucose"].to_numpy()
        rows_cal.append({"K_minutes": K, "rmse_no_ppg": float(np.sqrt(np.mean((yl - p0) ** 2))),
                         "rmse_with_ppg": float(np.sqrt(np.mean((yl - p1) ** 2))),
                         "ppg_gain_ci": f"[{dci[('ppg', 'no')][0]:+.2f}, {dci[('ppg', 'no')][1]:+.2f}]"})
        store[K] = (P, L)
        print(f"K={K:2d} min | free RMSE {rows_free[-1]['rmse']:.2f} (R2 {rows_free[-1]['r2']:.3f}) | calibrated "
              f"{rows_cal[-1]['rmse_with_ppg']:.2f} vs {rows_cal[-1]['rmse_no_ppg']:.2f} without PPG, "
              f"PPG change {rows_cal[-1]['rmse_with_ppg'] - rows_cal[-1]['rmse_no_ppg']:+.2f} {rows_cal[-1]['ppg_gain_ci']}",
              flush=True)
    Tf, Tc = pd.DataFrame(rows_free), pd.DataFrame(rows_cal)
    Tf.merge(Tc, on="K_minutes").to_csv(C.RESULTS_DIR / "device_averaging.csv", index=False)
    best = int(Tc.sort_values("rmse_with_ppg").iloc[0]["K_minutes"])
    print(f"\nbest averaging window (calibrated RMSE): K = {best} min")
    P, L = store[best]
    mf = T13.gbm().fit(P[T13.FREE].astype(float), P["glucose"])
    mc = T13.gbm().fit(L[T13.CAL].astype(float), L["glucose"] - L["ref_glucose"])
    hf, nf = T13.export_trees(mf, T13.FREE, "gbf")
    hc, nc = T13.export_trees(mc, T13.CAL, "gbc")
    hdr = ["// Auto-generated by 14_device_avg.py - do not edit\n#pragma once\n#include <stdint.h>\n#include <math.h>\n",
           f"#define DEV_N_FEATURES {len(DF.FEATURES)}\n#define DEV_N_PROFILE {len(PROFILE)}\n",
           f"#define DEV_AVG_WINDOWS {best}   // features = NaN-aware mean over the last {best} one-minute windows\n",
           "// feature order: " + ", ".join(DF.FEATURES) + "\n// profile order: " + ", ".join(PROFILE) + "\n", hf, hc]
    (T13.INC / "device_model.h").write_text("".join(hdr))
    print(f"exported K={best} models ({nf} + {nc} nodes) -> device_model.h")

    # reference: sequences of K one-minute windows -> averaged features -> Python predictions
    Rr = pd.read_csv(C.PROC_DIR / "readings.csv").set_index("rid")
    idx = {int(r): i for i, r in enumerate(rid)}
    end = NW // 2
    def seq(r):
        cid, ric = int(Rr.loc[r, "caseid"]), int(Rr.loc[r, "reading_in_case"])
        x = np.load(C.RAW_DIR / f"{cid}.npz")["ppg"][ric]
        return [x[w * DF.WIN:(w + 1) * DF.WIN].astype(np.float32).tolist() for w in (range(NW) if best >= NW else range(max(0, end - best), end))]
    ref = []
    for _, r in P[P["hr"].notna()].sample(20, random_state=1).iterrows():
        prof = [float(r[c]) if pd.notna(r[c]) else float("nan") for c in PROFILE]
        f = [r[c] for c in DF.FEATURES]
        ref.append({"rid": int(r.rid), "label": float(r.glucose), "windows": seq(int(r.rid)), "profile": prof,
                    "features": [None if not np.isfinite(v) else float(v) for v in f],
                    "pred_free": float(mf.predict(np.array([f + prof], float))[0])})
    for _, r in L[L["hr"].notna()].sample(10, random_state=1).iterrows():
        first = P[P.caseid == r.caseid].sort_values("dt").iloc[0]
        prof = [float(r[c]) if pd.notna(r[c]) else float("nan") for c in PROFILE]
        xc = np.array([[r[c] for c in T13.CAL]], float)
        ref.append({"rid": int(r.rid), "label": float(r.glucose), "windows": seq(int(r.rid)),
                    "cal_windows": seq(int(first.rid)), "cal_glucose": float(r.ref_glucose),
                    "minutes_since_cal": float(r.minutes_since_cal), "profile": prof,
                    "features": [None if not np.isfinite(r[c]) else float(r[c]) for c in DF.FEATURES],
                    "pred_free": float(mf.predict(np.array([[r[c] for c in DF.FEATURES] + prof], float))[0]),
                    "pred_cal": float(mc.predict(xc)[0] + r.ref_glucose)})
    out = C.CODE_DIR / "esp32" / "pc_test" / "device_reference_avg.json"
    out.write_text(json.dumps({"K": best, "cases": ref}))
    print(f"reference: {len(ref)} sequences of {best} windows -> {out}")


if __name__ == "__main__":
    main()
