"""Step 6 - external test on the MUST dataset (never used for training).

Download the MUST dataset from Mendeley Data (doi:10.17632/37pm7jk7jn.3) and unzip
it so that  <DATA_DIR>/must/  contains  RawData/*.mat  and  Labels/*.csv .

Every signal is matched to its label by the shared file-name suffix.
The sampling rate is inferred from the recording length (10-s recordings).

Run: python 06_must_external.py --run tiny_beat_subject
     (uses the 5 subject-fold checkpoints of that run as an ensemble)
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import torch
from scipy.signal import resample_poly

import config as C
import metrics as M
import models
import ppg


def find_pairs(root):
    mats = {re.sub(r"^\D*", "", Path(f).stem): f for f in glob.glob(str(root / "**" / "*.mat"), recursive=True)}
    labs = {re.sub(r"^\D*", "", Path(f).stem): f for f in glob.glob(str(root / "**" / "*.csv"), recursive=True)
            if Path(f).stem.lower().startswith("label")}
    keys = sorted(set(mats) & set(labs))
    print(f"MUST: {len(mats)} signals, {len(labs)} labels, {len(keys)} matched by file name")
    return [(k, mats[k], labs[k]) for k in keys]


def load_beats(mat_file, rec_seconds=10.0):
    m = scipy.io.loadmat(mat_file)
    key = "signal" if "signal" in m else [k for k in m if not k.startswith("__")][0]
    x = np.asarray(m[key], float).ravel()
    fs_in = int(round(len(x) / rec_seconds))
    x100 = resample_poly(x, C.FS, fs_in)
    xf = ppg.bandpass(x100)
    B, P, q = ppg.extract_beats(xf)
    return B, fs_in, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tiny_beat_subject")
    args = ap.parse_args()
    ckpts = sorted((C.DATA_DIR / "models").glob(f"{args.run}_fold*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints for {args.run} in {C.DATA_DIR/'models'}")
    dev = torch.device("cpu")
    nets = []
    for f in ckpts:
        ck = torch.load(f, map_location=dev)
        if ck["args"].get("demo"):
            raise SystemExit("use a run without --demo for MUST (no BMI available)")
        net = models.build(ck["args"]["model"]).to(dev)
        net.load_state_dict(ck["state"]); net.eval()
        nets.append((net, ck["mu"], ck["sd"]))
    vit_mean = float(np.mean([mu for _, mu, _ in nets]))

    rows = []
    for key, mf, lf in find_pairs(C.MUST_DIR):
        lab = pd.read_csv(lf)
        gcol = [c for c in lab.columns if "gluc" in c.lower()][0]
        icol = [c for c in lab.columns if c.lower() == "id"]
        B, fs_in, q = load_beats(mf)
        if len(B) < 3:
            continue
        xb = torch.from_numpy(B).unsqueeze(1)
        with torch.no_grad():
            ens = np.mean([net(xb).squeeze(1).numpy() * sd + mu for net, mu, sd in nets], 0)
        rows.append({"file": key, "subjectid": lab[icol[0]].iloc[0] if icol else key,
                     "y": float(lab[gcol].iloc[0]), "pred": float(np.median(ens)),
                     "n_beats": len(B), "fs_in": fs_in})
    P = pd.DataFrame(rows)
    if P.empty:
        raise SystemExit("no usable MUST recordings found - check the folder layout")
    print("inferred input sampling rates:", P.fs_in.value_counts().to_dict())
    P["const_vitaldb_mean"] = vit_mean
    P["const_must_mean_oracle"] = P["y"].mean()
    P.to_csv(C.RESULTS_DIR / "preds" / f"must_{args.run}.csv", index=False)
    out = {}
    for col in ["pred", "const_vitaldb_mean", "const_must_mean_oracle"]:
        m = M.basic(P.y, P[col]); m.update(M.clarke(P.y, P[col])); out[col] = m
        print(f"{col:24s} RMSE {m['rmse']:5.1f}  MAE {m['mae']:5.1f}  MARD {m['mard']:5.1f}%  "
              f"ClarkeA {m['clarke_A']:5.1f}%  A+B {m['clarke_A']+m['clarke_B']:5.1f}%  r {m['pearson_r']:+.2f}")
    print(f"MUST glucose range {P.y.min():.0f}-{P.y.max():.0f} mg/dL, sd {P.y.std():.1f}; "
          f"{len(P)} recordings, {P.subjectid.nunique()} subjects")
    with open(C.RESULTS_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps({"run": f"must_{args.run}", "external_must": out}) + "\n")


if __name__ == "__main__":
    main()
