"""Step 18 - seed robustness (Table 6) and the split x scoring factorial (Table 5) from results/runs.jsonl.
Needs the runs of run_all.bat and run_robustness.bat. Output: results/robustness_table.csv, results/factorial_table.csv
  python 18_robustness_factorial.py
"""
import json

import numpy as np
import pandas as pd

import config as C


def main():
    runs = {}
    for line in open(C.RESULTS_DIR / "runs.jsonl"):
        d = json.loads(line)
        runs[d["run"]] = d                      # later entries of the same run replace earlier ones
    groups = {"ResNet34-1D, leaky": ["resnet34_beat_leaky", "resnet34_beat_leaky_seed1"],
              "ResNet34-1D, person": ["resnet34_beat_subject", "resnet34_beat_subject_seed1", "resnet34_beat_subject_seed2"],
              "TinyCNN, person": ["tiny_beat_subject"] + [f"tiny_beat_subject_seed{s}" for s in range(1, 5)],
              "ResNet34-1D, person, lr 3e-4": ["resnet34_beat_subject_lr3e-4"],
              "ResNet34-1D, person, 96 beats per reading": ["resnet34_beat_subject_pr96"]}
    rows = []
    for name, keys in groups.items():
        m = np.array([[runs[k]["per_reading"][x] for x in ("rmse", "r2", "pearson_r", "pred_sd", "clarke_A")]
                      for k in keys if k in runs])
        sd = m.std(0, ddof=1) if len(m) > 1 else np.full(5, np.nan)
        rows.append({"model": name, "runs": len(m), **{f"{x}_mean": v for x, v in zip(("rmse", "r2", "r", "pred_sd", "clarke_A"), m.mean(0))},
                     **{f"{x}_sd": v for x, v in zip(("rmse", "r2", "r", "pred_sd", "clarke_A"), sd)}})
    T = pd.DataFrame(rows); T.to_csv(C.RESULTS_DIR / "robustness_table.csv", index=False)
    print(T.round(3).to_string(index=False))
    fac = []
    for name, lk, sk in [("ResNet34-1D", "resnet34_beat_leaky", "resnet34_beat_subject"),
                         ("ResNet34-1D, 10-s segments", "resnet34_seg10_leaky", "resnet34_seg10_subject"),
                         ("TinyCNN", "tiny_beat_leaky", "tiny_beat_subject")]:
        fac.append({"model": name, "segment_random": runs[lk]["per_sample"]["r2"], "segment_person": runs[sk]["per_sample"]["r2"],
                    "reading_random": runs[lk]["per_reading"]["r2"], "reading_person": runs[sk]["per_reading"]["r2"]})
    F = pd.DataFrame(fac); F.to_csv(C.RESULTS_DIR / "factorial_table.csv", index=False)
    print(F.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
