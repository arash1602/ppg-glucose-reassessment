"""Synthetic stand-in for VitalDB, used ONLY to test that the pipeline runs and
that the leakage check works. Each person gets an individual pulse shape and an
individual glucose level, but the pulse shape carries NO information about
glucose changes. A correct pipeline must therefore look good under the leaky
split and collapse to the mean baseline under the subject split.

Run: python make_synthetic.py --subjects 60   (writes into DATA_DIR/vitaldb_raw)
"""
import argparse

import numpy as np
import pandas as pd

import config as C


def beat_shape(t, p):
    sys_ = np.exp(-((t - p[0]) / p[1]) ** 2)
    dia = p[2] * np.exp(-((t - p[3]) / p[4]) ** 2)
    return sys_ + dia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_win = 2 * C.HALF_WINDOW_S * C.FS
    cases, cid = [], 0
    for s in range(args.subjects):
        shape = [rng.uniform(0.18, 0.28), rng.uniform(0.05, 0.09), rng.uniform(0.2, 0.6),
                 rng.uniform(0.45, 0.6), rng.uniform(0.06, 0.12)]
        hr = rng.uniform(55, 95)
        g_level = rng.uniform(80, 200)
        for c in range(rng.integers(1, 3)):
            cid += 1
            n_read = int(rng.integers(1, 5))
            wins, dts, glus = [], [], []
            for r in range(n_read):
                t = np.arange(n_win) / C.FS
                ibi = 60 / (hr + rng.normal(0, 3))
                phase = (t % ibi) / ibi
                x = beat_shape(phase, shape) * rng.uniform(800, 1200)
                x += 50 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 15, n_win)
                wins.append(x.astype(np.float32))
                dts.append(1200.0 + 1800 * r)
                glus.append(float(np.clip(g_level + rng.normal(0, 25), 40, 400)))
            np.savez_compressed(C.RAW_DIR / f"{cid}.npz", ppg=np.stack(wins), dt=np.array(dts),
                                glucose=np.array(glus), nan_frac=np.zeros(n_read))
            cases.append({"caseid": cid, "subjectid": s + 1, "age": rng.uniform(20, 85),
                          "sex": rng.choice(["M", "F"]), "bmi": rng.uniform(18, 32),
                          "preop_gluc": g_level + rng.normal(0, 20), "casestart": 0})
    pd.DataFrame(cases).to_csv(C.DATA_DIR / "cases.csv", index=False)
    print(f"synthetic: {args.subjects} subjects, {cid} cases written to {C.RAW_DIR}")


if __name__ == "__main__":
    main()
