"""Compare the C pipeline with Python on many difficult synthetic windows
(HR 45-150 bpm, noise, motion spikes, baseline drift, clipping/plateaus).
Usage: python stress_test.py --ckpt <tiny checkpoint> [--n 200]"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent)); sys.path.insert(0, str(H.parent.parent))
import export_esp32 as ex, models  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu"); net = models.build("tiny"); net.load_state_dict(ck["state"]); net.eval().double()
exe = H / ("glu_test.exe" if sys.platform == "win32" else "glu_test")
rng = np.random.default_rng(0); fails = 0; dmax = 0.0; dq = []
for i in range(a.n):
    n = 6000; t = np.arange(n) / 100
    hr = rng.uniform(45, 150) + rng.uniform(0, 8) * np.sin(2 * np.pi * rng.uniform(.01, .1) * t)
    ph = np.cumsum(hr / 60 / 100) % 1
    x = np.exp(-((ph - .25) / rng.uniform(.05, .1)) ** 2) + rng.uniform(.1, .6) * np.exp(-((ph - .55) / .1) ** 2)
    x = rng.uniform(200, 3000) * x + rng.uniform(0, 300) * np.sin(2 * np.pi * rng.uniform(.1, .4) * t)
    x += rng.normal(0, rng.uniform(1, 80), n)
    for _ in range(rng.integers(0, 4)):                       # motion spikes
        s = rng.integers(0, n - 100); x[s:s + rng.integers(5, 80)] += rng.normal(0, 1500)
    if rng.random() < .3: x = np.clip(x, None, np.percentile(x, 97))   # clipping -> plateaus
    x = np.round(x).astype(np.float32)                          # integer ADC-like values -> exact ties
    np.savetxt(H / "_w.txt", x)
    ref = ex.python_reference(net, x, float(ck["mu"]), float(ck["sd"]))
    c = json.loads(subprocess.run([str(exe), str(H / "_w.txt")], capture_output=True, text=True, check=True).stdout)["float"]
    same = c["n_beats"] == ref["n_beats"] and c["peaks"] == ref["peaks"]
    if same and ref["n_beats"]:
        dmax = max(dmax, abs(c["glucose_median"] - ref["glucose_median"]))
    if not same:
        fails += 1; print(f"window {i}: C {c['n_peaks']}/{c['n_beats']} vs Python {ref['n_peaks']}/{ref['n_beats']}")
print(f"{a.n - fails}/{a.n} windows identical beat selection; max glucose diff {dmax:.4f} mg/dL")
(H / "_w.txt").unlink()
