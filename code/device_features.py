"""Features exactly as the ESP32 computes them (60-s window, double precision).
The C port in esp32/firmware/src/features.c must reproduce this function number for number.
Raw-amplitude features are left out on purpose: their units depend on the sensor, so they
would not transfer from the hospital monitor (training) to a different PPG sensor (device).
"""
import numpy as np

import config as C
import ppg

WIN = 60 * C.FS
FEATURES = ["hr", "ibi_sd", "rmssd", "t_rise", "width50", "area_ratio", "max_slope", "min_slope",
            "apg_ba", "apg_ea", "dicrotic_ratio", "skew", "kurt", "accept_frac", "cos_mean"]


def centre_window(x, n=WIN):
    s = len(x) // 2 - n // 2
    return x[s:s + n]


def device_features(x_raw):
    """x_raw: 1-D PPG at 100 Hz (the 60-s window). Returns dict with FEATURES + n_beats (NaN = not computable)."""
    x = np.asarray(x_raw, dtype=np.float32).astype(np.float64)      # the device receives float32 samples
    B, P, q = ppg.extract_beats(ppg.bandpass(x))
    f = {k: np.nan for k in FEATURES}
    f["n_beats"] = int(len(B))
    f["accept_frac"] = float(q["accept_frac"]) if q["n_windows"] >= 3 else np.nan
    f["cos_mean"] = float(q["cos_mean"]) if len(B) else np.nan
    if len(B) < 3:
        return f
    fs = C.FS
    ibi = np.diff(P) / fs
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    if len(ibi):
        f["hr"] = 60.0 / np.median(ibi)
    if len(ibi) > 2:
        f["ibi_sd"] = float(np.std(ibi))
    if len(ibi) > 3:
        f["rmssd"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    T = np.median(B.astype(np.float64), axis=0)
    c = len(T) // 2
    foot = int(np.argmin(T[:c]))
    f["t_rise"] = (c - foot) / fs
    above = np.where(T > (T[c] + T[foot]) / 2)[0]
    f["width50"] = (above[-1] - above[0]) / fs if len(above) else np.nan
    pos = T - T.min()
    f["area_ratio"] = float(pos[:c].sum() / max(pos.sum(), 1e-8))
    d1 = np.gradient(T) * fs
    f["max_slope"], f["min_slope"] = float(d1.max()), float(d1.min())
    d2 = np.gradient(d1) * fs
    a_i = int(np.argmax(d2[:c]))
    a = d2[a_i]
    b = d2[a_i:c + 10].min()
    e = d2[c:c + 35].max()
    f["apg_ba"] = float(b / a) if a != 0 else np.nan
    f["apg_ea"] = float(e / a) if a != 0 else np.nan
    tail = T[c + 15:]
    f["dicrotic_ratio"] = float((tail.max() - T.min()) / max(T[c] - T.min(), 1e-8))
    z = (T - T.mean()) / (T.std() + 1e-8)
    f["skew"] = float(np.mean(z ** 3))
    f["kurt"] = float(np.mean(z ** 4))
    return f
