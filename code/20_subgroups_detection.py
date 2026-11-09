"""Step 20 - subgroups (Table 11), hypo-/hyperglycaemia detection (Table 12) and the paired PPG gain
by diabetes status. Reads per-reading predictions in results/preds/ (steps 03, 04, 10).
Outputs: results/subgroups.csv, results/detection.csv, results/diabetes_ppg_gain.csv
  python 20_subgroups_detection.py
"""
import numpy as np
import pandas as pd

import config as C

RUNS = [("Training-set mean", "baseline_mean_subject"), ("ResNet34-1D", "resnet34_beat_subject"),
        ("Clinical + time", "ms_free_t1___time_in_surgery"), ("Clinical + time + PPG", "ms_free_t2___ppg"),
        ("All signals, no calibration", "ms_free_t5___drug_infusions"),
        ("Calibrated, no PPG", "ms_calib_c0_fingerstick___time___clinical"), ("Calibrated + PPG", "ms_calib_c1___ppg_change"),
        ("Calibrated, all signals", "ms_calib_c3___monitor___drugs_change")]
GROUPS = [("All", lambda d: d.y > -1), ("No diabetes", lambda d: d.preop_dm == 0), ("Diabetes", lambda d: d.preop_dm == 1),
          ("< 70 mg/dL", lambda d: d.y < 70), ("70-180 mg/dL", lambda d: (d.y >= 70) & (d.y <= 180)), ("> 180 mg/dL", lambda d: d.y > 180)]
HYPO, HYPER = 70, 180


def boot_idx(d, B, rng):
    subs = d.subjectid.unique(); idx = {s: np.where(d.subjectid.values == s)[0] for s in subs}
    return [np.concatenate([idx[s] for s in rng.choice(subs, len(subs))]) for _ in range(B)]


def main():
    R = pd.read_csv(C.PROC_DIR / "readings.csv")[["rid", "preop_dm"]] if "preop_dm" in pd.read_csv(C.PROC_DIR / "readings.csv", nrows=1) \
        else __import__("10_multisignal_eval").load()[["rid", "preop_dm"]]
    P = {k: pd.read_csv(C.RESULTS_DIR / "preds" / f"{f}.csv").merge(R, on="rid") for k, f in RUNS}
    rng = np.random.default_rng(0)
    rows = []
    for gname, gf in GROUPS:
        for mname in ["Training-set mean", "ResNet34-1D", "Clinical + time", "Clinical + time + PPG",
                      "Calibrated, no PPG", "Calibrated + PPG", "Calibrated, all signals"]:
            q = P[mname][gf(P[mname])]
            if len(q) == 0:
                continue
            e = (q.pred - q.y).to_numpy(); bs = [np.sqrt(np.mean(e[i] ** 2)) for i in boot_idx(q, 500, rng)]
            r = dict(group=gname, model=mname, n=len(q), people=q.subjectid.nunique(), rmse=np.sqrt(np.mean(e ** 2)),
                     lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), mard=np.mean(np.abs(e) / q.y) * 100, bias=e.mean())
            if gname == "< 70 mg/dL": r["detected_lt70"] = (q.pred < HYPO).mean() * 100
            if gname == "> 180 mg/dL": r["detected_gt180"] = (q.pred > HYPER).mean() * 100
            rows.append(r)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "subgroups.csv", index=False)

    def rates(y, p):
        out = {}
        for nm, pos_y, pos_p in [("hypo", y < HYPO, p < HYPO), ("hyper", y > HYPER, p > HYPER)]:
            tp = np.sum(pos_y & pos_p); fn = np.sum(pos_y & ~pos_p); fp = np.sum(~pos_y & pos_p); tn = np.sum(~pos_y & ~pos_p)
            out[f"{nm}_sens"] = 100 * tp / (tp + fn) if tp + fn else np.nan
            out[f"{nm}_spec"] = 100 * tn / (tn + fp) if tn + fp else np.nan
            out[f"{nm}_ppv"] = 100 * tp / (tp + fp) if tp + fp else np.nan
        return out
    det = []
    for mname in ["ResNet34-1D", "Clinical + time + PPG", "All signals, no calibration", "Calibrated, no PPG",
                  "Calibrated + PPG", "Calibrated, all signals"]:
        d = P[mname]; y, p = d.y.to_numpy(), d.pred.to_numpy(); point = rates(y, p)
        bs = pd.DataFrame([rates(y[i], p[i]) for i in boot_idx(d, 1000, rng)])
        row = {"model": mname, "n": len(d), "n_hypo": int((y < HYPO).sum()), "n_hyper": int((y > HYPER).sum())}
        for k, v in point.items():
            row[k] = v; row[k + "_ci"] = f"{np.nanpercentile(bs[k], 2.5):.1f}-{np.nanpercentile(bs[k], 97.5):.1f}" if bs[k].notna().any() else "nan-nan"
        det.append(row)
    pd.DataFrame(det).to_csv(C.RESULTS_DIR / "detection.csv", index=False)

    gains = []
    for g in [0, 1]:
        for a_, b_ in [("Clinical + time", "Clinical + time + PPG"), ("Calibrated, no PPG", "Calibrated + PPG")]:
            m = P[a_][P[a_].preop_dm == g][["rid", "subjectid", "y", "pred"]].merge(
                P[b_][P[b_].preop_dm == g][["rid", "pred"]], on="rid", suffixes=("_a", "_b"))
            ea, eb = (m.pred_a - m.y).values ** 2, (m.pred_b - m.y).values ** 2
            d = [np.sqrt(ea[i].mean()) - np.sqrt(eb[i].mean()) for i in boot_idx(m, 1000, rng)]
            gains.append({"diabetes": g, "comparison": f"{b_} vs {a_}", "patients": m.subjectid.nunique(),
                          "ppg_gain": np.sqrt(ea.mean()) - np.sqrt(eb.mean()), "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    G = pd.DataFrame(gains); G.to_csv(C.RESULTS_DIR / "diabetes_ppg_gain.csv", index=False)
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(det).round(1).to_string(index=False)); print(G.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
