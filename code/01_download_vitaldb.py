"""Step 1 - download VitalDB PPG windows around every intra-operative glucose lab value.

For each case that has glucose ('gluc') lab results and a SNUADC/PLETH waveform:
  * load the PLETH track at 500 Hz from the open-dataset .vital file
  * cut +/- 8 min around each glucose time (lab dt is relative to casestart)
  * anti-alias decimate to 100 Hz
  * save one compressed .npz per case

Resumable: cases already saved (or marked as failed/empty) are skipped.
Run:  python 01_download_vitaldb.py [--workers 4] [--limit N]
"""
import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import decimate

import config as C

API = "https://api.vitaldb.net"


def get_tables():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_f, labs_f = C.DATA_DIR / "cases.csv", C.DATA_DIR / "labs.csv"
    if not cases_f.exists():
        pd.read_csv(f"{API}/cases").to_csv(cases_f, index=False)
    if not labs_f.exists():
        pd.read_csv(f"{API}/labs").to_csv(labs_f, index=False)
    cases = pd.read_csv(cases_f)
    labs = pd.read_csv(labs_f)
    glu = labs[labs["name"].str.lower() == "gluc"].copy()
    if glu.empty:
        raise SystemExit("no 'gluc' rows in VitalDB labs; available lab names: "
                         + ", ".join(sorted(labs["name"].astype(str).unique())))
    glu = glu[(glu["result"] >= C.GLU_MIN) & (glu["result"] <= C.GLU_MAX)]
    return cases, glu


def ffill_nan(x):
    """forward-fill NaNs (paper), then back-fill a leading gap."""
    s = pd.Series(x)
    return s.ffill().bfill().to_numpy()


def process_case(caseid, case_row, glu_rows):
    import vitaldb  # imported here so worker threads are independent
    out_f = C.RAW_DIR / f"{caseid}.npz"
    sig = vitaldb.load_case(int(caseid), [C.PPG_TRACK], 1.0 / C.FS_RAW)
    if sig is None or len(sig) == 0:
        return "no_track"
    sig = sig[:, 0].astype(np.float64)
    casestart = float(case_row.get("casestart", 0) or 0)
    hw = C.HALF_WINDOW_S * C.FS_RAW
    wins, meta = [], []
    for _, r in glu_rows.iterrows():
        center = int(round((casestart + float(r["dt"])) * C.FS_RAW))
        a, b = center - hw, center + hw
        if a < 0 or b > len(sig):
            continue  # paper: drop readings without the full +/-8 min
        w = sig[a:b]
        nan_frac = float(np.isnan(w).mean())
        if nan_frac > 0.5:
            continue
        w = ffill_nan(w)
        w100 = decimate(w, C.FS_RAW // C.FS, ftype="fir", zero_phase=True)
        wins.append(w100.astype(np.float32))
        meta.append((float(r["dt"]), float(r["result"]), nan_frac))
    if not wins:
        np.savez_compressed(out_f, ppg=np.zeros((0, 2 * C.HALF_WINDOW_S * C.FS), np.float32),
                            dt=np.zeros(0), glucose=np.zeros(0), nan_frac=np.zeros(0))
        return "no_windows"
    m = np.array(meta)
    np.savez_compressed(out_f, ppg=np.stack(wins), dt=m[:, 0], glucose=m[:, 1], nan_frac=m[:, 2])
    return f"ok:{len(wins)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (for a quick test)")
    args = ap.parse_args()

    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cases, glu = get_tables()
    case_ids = sorted(glu["caseid"].unique())
    if args.limit:
        case_ids = case_ids[: args.limit]
    todo = [c for c in case_ids if not (C.RAW_DIR / f"{c}.npz").exists()]
    print(f"{len(case_ids)} cases with glucose labs; {len(todo)} still to download")
    cases_idx = cases.set_index("caseid")
    log_f = C.DATA_DIR / "download_log.jsonl"

    def job(cid):
        for attempt in range(3):
            try:
                return cid, process_case(cid, cases_idx.loc[cid], glu[glu["caseid"] == cid])
            except Exception as e:  # network hiccups: retry
                err = f"error:{type(e).__name__}:{e}"
                time.sleep(3 * (attempt + 1))
        return cid, err

    t0, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex, open(log_f, "a") as log:
        futs = [ex.submit(job, c) for c in todo]
        for f in as_completed(futs):
            cid, status = f.result()
            log.write(json.dumps({"caseid": int(cid), "status": status}) + "\n")
            log.flush()
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] {el/60:.1f} min elapsed, "
                      f"~{el/done*(len(todo)-done)/60:.0f} min left (last: case {cid} {status})")
    print("done. Raw windows in", C.RAW_DIR)


if __name__ == "__main__":
    main()
