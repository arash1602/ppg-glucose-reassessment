"""Compile the firmware C (glucose.c + device.c) for the PC, stream the 60 reference VitalDB windows
through the same command protocol the ESP32 uses, and compare features and predictions with Python."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; FW = H.parent / "firmware"
exe = H / ("dev_test.exe" if sys.platform == "win32" else "dev_test")
subprocess.run(["gcc", "-O2", "-std=c99", f"-I{FW/'include'}", str(H / "device_main.c"), str(FW / "src" / "glucose.c"),
                str(FW / "src" / "device.c"), "-lm", "-o", str(exe)], check=True)
ref = json.loads((H / "device_reference.json").read_text())
fmt = lambda v: "nan" if v is None or not np.isfinite(v) else repr(float(v))
lines, expect = [], []
for r in ref:
    lines.append("PROFILE " + " ".join(fmt(v) for v in r["profile"]))
    if "x_cal" in r:
        lines += [f"CAL {r['cal_glucose']}", f"BEGIN {len(r['x_cal'])}"] + [repr(float(np.float32(v))) for v in r["x_cal"]] + ["END"]
        expect.append(None)
        lines.append(f"MIN {r['minutes_since_cal']!r}")
    lines += [f"BEGIN {len(r['x'])}"] + [repr(float(np.float32(v))) for v in r["x"]] + ["END"]
    expect.append(r)
out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
res = [json.loads(l) for l in out.splitlines() if l.startswith('{"tag":"window"')]
assert len(res) == len(expect), (len(res), len(expect))
fd, pf, pc, nb = [], [], [], 0
for d, r in zip(res, expect):
    if r is None:
        continue
    nb += d["n_beats"] == r["n_beats"]
    a = np.array([np.nan if v is None else v for v in d["features"]], float)
    b = np.array([np.nan if v is None else v for v in r["features"]], float)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    fd.append(np.nanmax(np.abs(a - b) / np.maximum(1, np.abs(b))) if same_nan else np.inf)
    pf.append(abs(d["glucose_free"] - r["pred_free"]))
    if "pred_cal" in r:
        pc.append(abs(d["glucose_cal"] - r["pred_cal"]))
n = len(fd)
ok = nb == n and max(fd) < 1e-9 and max(pf) < 1e-6 and (not pc or max(pc) < 1e-6)
print(f"windows {n} ({len(pc)} calibrated) | identical beat counts {nb}/{n} | max feature diff {max(fd):.2e} (relative)")
print(f"free model:       max |C - Python| = {max(pf):.2e} mg/dL")
if pc: print(f"calibrated model: max |C - Python| = {max(pc):.2e} mg/dL")
print("ALL DEVICE CHECKS PASSED" if ok else "SOME DEVICE CHECKS FAILED")
sys.exit(0 if ok else 1)
