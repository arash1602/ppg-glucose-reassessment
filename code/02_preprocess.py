"""Step 2 - turn raw +/-8 min windows into model inputs.

Per glucose reading we produce:
  * up to MAX_BEATS_PER_READING  1-s beats   (paper's proposed input)
  * up to MAX_SEG10_PER_READING 10-s segments (paper's baseline input)
  * one row of morphology/quality features
Everything keeps caseid AND subjectid so later splits can be done by person.

Run: python 02_preprocess.py [--workers 4]
"""
import argparse
import io
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as C
import ppg

OUT_CASES = C.PROC_DIR / "cases"


def even_pick(n, k, rng):
    if n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, k, replace=False))


def process_case_file(path):
    cid = int(path.stem)
    out = OUT_CASES / f"{cid}.npz"
    if out.exists():
        return str(out)
    d = np.load(path)
    rng = np.random.default_rng(cid)
    beats, beat_r, segs, seg_r, rows = [], [], [], [], []
    for i in range(len(d["glucose"])):
        raw = d["ppg"][i].astype(np.float64)
        xf = ppg.bandpass(raw)
        B, P, q = ppg.extract_beats(xf)
        S = ppg.segments_10s(xf)
        okS = np.array([ppg.segment_ok(s) for s in S], bool) if len(S) else np.zeros(0, bool)
        S = S[okS]
        row = {"caseid": cid, "reading_in_case": i, "dt": float(d["dt"][i]),
               "glucose": float(d["glucose"][i]), "nan_frac": float(d["nan_frac"][i]),
               "n_seg10_ok": int(len(S)), **q}
        row.update(ppg.morphology_features(B, P, raw))
        rows.append(row)
        bi = even_pick(len(B), C.MAX_BEATS_PER_READING, rng)
        si = even_pick(len(S), C.MAX_SEG10_PER_READING, rng)
        beats.append(B[bi]); beat_r.append(np.full(len(bi), i))
        segs.append(S[si]); seg_r.append(np.full(len(si), i))
    L, L10 = C.BEAT_LEN, C.FS * C.SEG10_S
    np.savez(out,
             beats=np.concatenate(beats).astype(np.float16) if beats else np.zeros((0, L), np.float16),
             beat_r=np.concatenate(beat_r).astype(np.int32) if beat_r else np.zeros(0, np.int32),
             segs=np.concatenate(segs).astype(np.float16) if segs else np.zeros((0, L10), np.float16),
             seg_r=np.concatenate(seg_r).astype(np.int32) if seg_r else np.zeros(0, np.int32),
             rows=pd.DataFrame(rows).to_json())
    return str(out)


def merge(min_beats=10):
    cases = pd.read_csv(C.DATA_DIR / "cases.csv")
    keep_cols = [c for c in ["caseid", "subjectid", "age", "sex", "height", "weight", "bmi",
                             "asa", "department", "optype", "dx", "preop_dm", "preop_htn",
                             "preop_gluc", "preop_hb", "casestart", "caseend"] if c in cases.columns]
    cases = cases[keep_cols]
    files = sorted(OUT_CASES.glob("*.npz"), key=lambda p: int(p.stem))
    tabs, beats, beat_rid, segs, seg_rid = [], [], [], [], []
    rid = 0
    for f in files:
        d = np.load(f)
        rows = pd.read_json(io.StringIO(str(d["rows"])))
        if rows.empty:
            continue
        for i, r in rows.iterrows():
            nb = int((d["beat_r"] == r["reading_in_case"]).sum())
            if nb < min_beats:
                continue
            r = r.copy(); r["rid"] = rid
            tabs.append(r)
            m = d["beat_r"] == r["reading_in_case"]
            beats.append(d["beats"][m]); beat_rid.append(np.full(m.sum(), rid, np.int32))
            m = d["seg_r"] == r["reading_in_case"]
            segs.append(d["segs"][m]); seg_rid.append(np.full(m.sum(), rid, np.int32))
            rid += 1
    R = pd.DataFrame(tabs).merge(cases, on="caseid", how="left")
    if "subjectid" not in R.columns or R["subjectid"].isna().all():
        print("WARNING: no subjectid column - falling back to caseid as the person id")
        R["subjectid"] = R["caseid"]
    R["subjectid"] = R["subjectid"].fillna(-R["caseid"]).astype(int)
    R["sex_m"] = (R.get("sex", pd.Series(index=R.index)) == "M").astype(float)
    R.to_csv(C.PROC_DIR / "readings.csv", index=False)
    np.save(C.PROC_DIR / "beats.npy", np.concatenate(beats))
    np.save(C.PROC_DIR / "beat_rid.npy", np.concatenate(beat_rid))
    np.save(C.PROC_DIR / "segs.npy", np.concatenate(segs))
    np.save(C.PROC_DIR / "seg_rid.npy", np.concatenate(seg_rid))
    print(f"readings: {len(R)}  cases: {R.caseid.nunique()}  subjects: {R.subjectid.nunique()}")
    print(f"beats: {sum(map(len, beats))}  10-s segments: {sum(map(len, segs))}")
    print("glucose mg/dL: mean %.1f sd %.1f min %.0f max %.0f" %
          (R.glucose.mean(), R.glucose.std(), R.glucose.min(), R.glucose.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT_CASES.mkdir(parents=True, exist_ok=True)
    files = sorted(C.RAW_DIR.glob("*.npz"))
    print(f"{len(files)} raw case files")
    with Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(process_case_file, files, chunksize=4), 1):
            if k % 100 == 0 or k == len(files):
                print(f"  processed {k}/{len(files)}")
    merge()


if __name__ == "__main__":
    main()
