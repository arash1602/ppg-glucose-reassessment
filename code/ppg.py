"""PPG signal processing shared by VitalDB, MUST and the embedded export.

Implements the paper's 1-s 'precision interval segmentation' (peak detection ->
1-s window centred on the systolic peak -> exactly one peak per window ->
template = mean window -> keep windows with cosine similarity >= 0.85),
plus simple, documented signal-quality and morphology features.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

import config as C

_SOS = butter(C.FILTER_ORDER, C.BAND, btype="band", fs=C.FS, output="sos")


def bandpass(x):
    return sosfiltfilt(_SOS, x)


def minmax_pm1(x, axis=-1):
    lo = x.min(axis=axis, keepdims=True)
    hi = x.max(axis=axis, keepdims=True)
    return 2 * (x - lo) / np.maximum(hi - lo, 1e-8) - 1


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-8)


def detect_peaks(xz, fs=C.FS):
    # distance 0.4 s allows HR up to 150 bpm (paper used 0.8 s -> max 75 bpm, which
    # silently drops every beat of a tachycardic patient); prominence rejects noise.
    p, _ = find_peaks(xz, distance=int(0.4 * fs), prominence=0.8)
    return p


def extract_beats(x_filt, fs=C.FS, L=C.BEAT_LEN, cos_min=C.COS_SIM_MIN):
    """Return (beats [n,L] scaled to [-1,1], peak indices, quality dict)."""
    xz = robust_z(x_filt)
    peaks = detect_peaks(xz, fs)
    half = L // 2
    wins, keep_p = [], []
    for p in peaks:
        a, b = p - half, p + half
        if a < 0 or b > len(xz):
            continue
        # paper: exactly one detected peak inside the window
        n_in = np.sum((peaks >= a) & (peaks < b))
        if n_in != 1:
            continue
        wins.append(xz[a:b])
        keep_p.append(p)
    q = {"n_peaks": int(len(peaks)), "n_windows": len(wins), "n_beats": 0,
         "accept_frac": 0.0, "cos_mean": np.nan}
    if len(wins) < 3:
        return np.zeros((0, L), np.float32), np.array(keep_p, int), q
    W = np.stack(wins)
    T = W.mean(0)
    cos = (W @ T) / (np.linalg.norm(W, axis=1) * np.linalg.norm(T) + 1e-8)
    ok = cos >= cos_min
    q.update(n_beats=int(ok.sum()), accept_frac=float(ok.mean()),
             cos_mean=float(cos[ok].mean()) if ok.any() else np.nan)
    return minmax_pm1(W[ok]).astype(np.float32), np.array(keep_p, int)[ok], q


def morphology_features(beats, peaks, x_raw, fs=C.FS):
    """Hand-crafted features for one reading (used by the gradient-boosting baseline)."""
    f = {}
    if len(peaks) >= 3:
        ibi = np.diff(peaks) / fs
        ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    else:
        ibi = np.array([])
    f["hr"] = 60 / np.median(ibi) if len(ibi) else np.nan
    f["ibi_sd"] = float(np.std(ibi)) if len(ibi) > 2 else np.nan
    f["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 3 else np.nan
    # raw amplitude (monitor units; note: device auto-gain makes this only partly physiological)
    if len(peaks):
        xr = x_raw - np.median(x_raw)
        f["amp_raw"] = float(np.median(xr[peaks]) - np.percentile(xr, 5))
        f["raw_sd"] = float(np.std(x_raw))
    else:
        f["amp_raw"] = f["raw_sd"] = np.nan
    if len(beats) == 0:
        for k in ["t_rise", "width50", "area_ratio", "max_slope", "min_slope", "apg_ba",
                  "apg_ea", "dicrotic_ratio", "skew", "kurt"]:
            f[k] = np.nan
        return f
    T = np.median(beats, 0)
    c = len(T) // 2
    foot = int(np.argmin(T[:c]))
    f["t_rise"] = (c - foot) / fs
    above = np.where(T > (T[c] + T[foot]) / 2)[0]
    f["width50"] = (above[-1] - above[0]) / fs if len(above) else np.nan
    pos = T - T.min()
    f["area_ratio"] = float(pos[:c].sum() / max(pos.sum(), 1e-8))
    d1 = np.gradient(T) * fs
    f["max_slope"] = float(d1.max())
    f["min_slope"] = float(d1.min())
    d2 = np.gradient(d1) * fs
    a_i = int(np.argmax(d2[:c])) if c > 0 else 0
    a = d2[a_i]
    b = d2[a_i:c + 10].min() if a_i < c + 10 else np.nan
    e = d2[c:c + 35].max() if c + 35 <= len(d2) else np.nan
    f["apg_ba"] = float(b / a) if a else np.nan
    f["apg_ea"] = float(e / a) if a else np.nan
    tail = T[c + 15:]
    f["dicrotic_ratio"] = float((tail.max() - T.min()) / max(T[c] - T.min(), 1e-8)) if len(tail) else np.nan
    z = (T - T.mean()) / (T.std() + 1e-8)
    f["skew"] = float(np.mean(z ** 3))
    f["kurt"] = float(np.mean(z ** 4))
    return f


def segments_10s(x_filt, fs=C.FS, seg_s=C.SEG10_S):
    n = len(x_filt) // (fs * seg_s)
    S = x_filt[: n * fs * seg_s].reshape(n, fs * seg_s)
    return minmax_pm1(S).astype(np.float32)


def segment_ok(seg, beats_per_seg_min=4):
    """crude 10-s quality gate: enough regular peaks."""
    p = detect_peaks(robust_z(seg))
    return len(p) >= beats_per_seg_min
