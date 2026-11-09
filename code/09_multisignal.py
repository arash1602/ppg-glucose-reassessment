"""Step 9 - multi-signal features from the SAME VitalDB surgeries and glucose readings.

The original paper used only the finger PPG (SNUADC/PLETH). VitalDB records far more at the
same time. For every glucose reading already in processed/readings.csv this script downloads
the case file once and computes, per reading:

  waveforms (100 Hz, +/- 8 min, same window as the PPG):
    ECG lead II  -> heart rate, SDNN, RMSSD
    arterial line -> systolic / diastolic / mean pressure, pulse pressure, max dP/dt
    ECG + PPG    -> pulse arrival time to the finger (R-peak -> PPG peak)
    ECG + arterial -> pulse arrival time to the arterial line
  monitor numerics (1 s, the 10 min BEFORE the reading, so nothing after the lab draw is used):
    body temperature, SpO2, end-tidal CO2, BIS, heart rate, mean arterial pressure,
    + their change over the preceding 30 min
  infusion pumps (10 min before): propofol, remifentanil, norepinephrine, phenylephrine,
    epinephrine, dopamine, dobutamine, vasopressin rates

Saved per case as small JSON (resumable): <DATA_DIR>/multisignal/<caseid>.json
Merge into one table at the end: <DATA_DIR>/processed/multisignal.csv
Run: python 09_multisignal.py [--workers 4] [--limit N]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

import config as C

FS = 100
WAVE = ["SNUADC/ECG_II", "SNUADC/ART", "SNUADC/PLETH"]
NUM = {"Solar8000/BT": "temp", "Solar8000/PLETH_SPO2": "spo2", "Solar8000/ETCO2": "etco2", "BIS/BIS": "bis",
       "Solar8000/HR": "hr_mon", "Solar8000/ART_MBP": "mbp_mon", "Solar8000/NIBP_MBP": "nibp_mbp"}
DRUG = {"Orchestra/PPF20_RATE": "propofol", "Orchestra/RFTN20_RATE": "remifentanil",
        "Orchestra/NEPI_RATE": "norepinephrine", "Orchestra/PHEN_RATE": "phenylephrine",
        "Orchestra/EPI_RATE": "epinephrine", "Orchestra/DOPA_RATE": "dopamine",
        "Orchestra/DOBU_RATE": "dobutamine", "Orchestra/VASO_RATE": "vasopressin"}
OUT = C.DATA_DIR / "multisignal"

_ECG_SOS = butter(3, (5, 25), btype="band", fs=FS, output="sos")
_PPG_SOS = butter(3, (0.5, 8), btype="band", fs=FS, output="sos")


def _clean(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.mean() < 0.5:
        return None
    if not ok.all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    return x


def ecg_rpeaks(x):
    x = _clean(x)
    if x is None or np.nanstd(x) < 1e-6:
        return None
    y = sosfiltfilt(_ECG_SOS, x)
    if np.abs(np.percentile(y, 1)) > np.abs(np.percentile(y, 99)):   # inverted lead
        y = -y
    z = (y - np.median(y)) / (np.median(np.abs(y - np.median(y))) * 1.4826 + 1e-9)
    p, _ = find_peaks(z, distance=int(0.3 * FS), prominence=4)
    if len(p) >= 20:                                   # drop T waves / noise: < 40 % of typical R height
        p = p[z[p] >= 0.4 * np.percentile(z[p], 90)]
    return p if len(p) >= 20 else None


def rr_features(p):
    rr = np.diff(p) / FS
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) < 20:
        return {}
    d = np.diff(rr)
    return {"ecg_hr": 60 / np.median(rr), "ecg_sdnn": float(np.std(rr)), "ecg_rmssd": float(np.sqrt(np.mean(d ** 2)))}


def art_features(x):
    x = _clean(x)
    if x is None:
        return {}, None
    if np.median(x) < 20 or np.median(x) > 200:     # line not connected / flushed
        return {}, None
    pk, _ = find_peaks(x, distance=int(0.3 * FS), prominence=10)
    tr, _ = find_peaks(-x, distance=int(0.3 * FS), prominence=10)
    if len(pk) < 20 or len(tr) < 20:
        return {}, None
    sbp, dbp = np.median(x[pk]), np.median(x[tr])
    if not (40 < sbp < 260 and 15 < dbp < sbp):
        return {}, None
    dpdt = np.diff(x) * FS
    return {"sbp": sbp, "dbp": dbp, "map_wave": float(np.mean(x)), "pp": sbp - dbp,
            "dpdt_max": float(np.percentile(dpdt, 99))}, pk


def arrival_time(r, target_peaks, lo=0.08, hi=0.6):
    """median delay from each ECG R-peak to the next peak of another signal"""
    if r is None or target_peaks is None or len(target_peaks) < 10:
        return np.nan
    j = np.searchsorted(target_peaks, r)
    ok = j < len(target_peaks)
    d = (target_peaks[j[ok]] - r[ok]) / FS
    d = d[(d >= lo) & (d <= hi)]
    return float(np.median(d)) if len(d) >= 10 else np.nan


def ppg_peaks(x):
    x = _clean(x)
    if x is None:
        return None
    y = sosfiltfilt(_PPG_SOS, x)
    z = (y - np.median(y)) / (np.median(np.abs(y - np.median(y))) * 1.4826 + 1e-9)
    p, _ = find_peaks(z, distance=int(0.3 * FS), prominence=0.8)
    return p if len(p) >= 20 else None


def window_median(v, a, b):
    a, b = max(a, 0), min(b, len(v))
    if b - a < 30:
        return np.nan
    s = v[a:b]
    s = s[np.isfinite(s)]
    return float(np.median(s)) if len(s) >= 10 else np.nan


def reading_features(wav, num, drug, t_s):
    """t_s = seconds from the start of the file to the glucose reading."""
    f = {}
    c = int(round(t_s * FS)); hw = C.HALF_WINDOW_S * FS
    if c - hw >= 0 and c + hw <= len(wav):
        seg = wav[c - hw:c + hw]
        r = ecg_rpeaks(seg[:, 0])
        if r is not None:
            f.update(rr_features(r))
        af, apk = art_features(seg[:, 1])
        f.update(af)
        f["pat_ppg"] = arrival_time(r, ppg_peaks(seg[:, 2]))
        f["pat_art"] = arrival_time(r, apk)
    t = int(round(t_s))
    for j, name in enumerate(NUM.values()):
        v = num[:, j]
        now = window_median(v, t - 600, t)
        f[name] = now
        f[f"{name}_chg30"] = now - window_median(v, t - 1800, t - 1500)
    for j, name in enumerate(DRUG.values()):
        v = drug[:, j]
        m = window_median(v, t - 600, t)
        f[name] = 0.0 if np.isnan(m) and np.isfinite(v).any() else m   # pump present but idle -> 0
    return f


def case_url(cid):
    from vitaldb import dataset as vd
    ver = getattr(vd, "DATASET_VERSION", "")
    return f"{vd.api_url}/{ver}/{int(cid)}.vital" if ver else f"{vd.api_url}/{int(cid)}.vital"


def process_case(cid, rows, casestart):
    import vitaldb
    vf = vitaldb.VitalFile(case_url(cid), WAVE + list(NUM) + list(DRUG))
    wav = vf.to_numpy(WAVE, 1 / FS)
    num = vf.to_numpy(list(NUM), 1)
    drug = vf.to_numpy(list(DRUG), 1)
    out = []
    for _, r in rows.iterrows():
        f = reading_features(wav, num, drug, casestart + r["dt"])
        f["rid"] = int(r["rid"])
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    cs = pd.read_csv(C.DATA_DIR / "cases.csv").set_index("caseid")["casestart"].fillna(0)
    cases = sorted(R.caseid.unique())
    if a.limit:
        cases = cases[:a.limit]
    def done_ok(c):
        f = OUT / f"{c}.json"
        return f.exists() and not json.loads(f.read_text()).get("error")
    todo = [c for c in cases if not done_ok(c)]          # failed cases are retried
    print(f"{len(cases)} cases, {len(todo)} to download")
    if todo:                                              # connectivity check on one case, with the exact URL
        print("test URL:", case_url(todo[0]), flush=True)
        try:
            import vitaldb
            vitaldb.VitalFile(case_url(todo[0]), ["Solar8000/HR"])
            print("test download OK", flush=True)
        except Exception as e:
            print(f"test download FAILED: {type(e).__name__}: {e}", flush=True)
            try:                                          # does the library's own loader work?
                import vitaldb
                vitaldb.load_case(int(todo[0]), ["Solar8000/HR"], 1)
                print("  ...but vitaldb.load_case works", flush=True)
            except Exception as e2:
                print(f"  vitaldb.load_case also fails: {type(e2).__name__}: {e2}", flush=True)
            if "403" in str(e):
                raise SystemExit("VitalDB refuses downloads right now (HTTP 403). Wait an hour and run again.")

    def job(cid):
        err = None
        for attempt in range(4):
            try:
                return cid, process_case(cid, R[R.caseid == cid], float(cs.get(cid, 0))), None
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep((60 if "403" in err or "429" in err else 3) * (attempt + 1))   # back off when refused
        return cid, None, err

    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(job, c) for c in todo]):
            cid, feats, err = fut.result()
            (OUT / f"{cid}.json").write_text(json.dumps({"rows": feats or [], "error": err}, default=float))
            done += 1
            if err:
                print(f"  case {cid}: {err}", flush=True)
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] {el/60:.0f} min, ~{el/done*(len(todo)-done)/60:.0f} min left", flush=True)

    rows = []
    for f in OUT.glob("*.json"):
        rows += json.loads(f.read_text())["rows"]
    M = pd.DataFrame(rows)
    M.to_csv(C.PROC_DIR / "multisignal.csv", index=False)
    cov = M.drop(columns="rid").notna().mean().sort_values(ascending=False)
    print(f"{len(M)} readings with multi-signal features; coverage per feature:")
    print((cov * 100).round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
