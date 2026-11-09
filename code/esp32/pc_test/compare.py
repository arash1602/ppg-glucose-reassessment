"""Compile the firmware C code for the PC, run it on the exported test window and compare
with the Python reference (reference.json written by export_esp32.py)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
FW = H.parent / "firmware"
exe = H / ("glu_test.exe" if sys.platform == "win32" else "glu_test")
cc = "gcc"
subprocess.run([cc, "-O2", "-std=c99", f"-I{FW/'include'}", str(H / "test_main.c"), str(FW / "src" / "glucose.c"),
                "-lm", "-o", str(exe)], check=True)
c = json.loads(subprocess.run([str(exe), str(H / "test_window.txt")], capture_output=True, text=True, check=True).stdout)
r = json.loads((H / "reference.json").read_text())
ok = True
def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
d = np.max(np.abs(np.array(c["filtered_first10"]) - np.array(r["filtered_first10"])))
check("band-pass filter", d < 1e-6, f"max abs diff {d:.2e}")
f = c["float"]
check("peaks / windows / beats", (f["n_windows"], f["n_beats"]) == (r["n_windows"], r["n_beats"]) and f["n_peaks"] == r["n_peaks"],
      f"C {f['n_peaks']}/{f['n_windows']}/{f['n_beats']}  Python {r['n_peaks']}/{r['n_windows']}/{r['n_beats']}")
check("beat positions", f["peaks"] == r["peaks"], "identical" if f["peaks"] == r["peaks"] else "differ")
if f["n_beats"] == r["n_beats"] and f["n_beats"]:
    dp = np.max(np.abs(np.array(f["beat_preds"]) - np.array(r["beat_preds"])))
    check("per-beat predictions (float32)", dp < 0.05, f"max diff {dp:.4f} mg/dL")
check("glucose (float32)", abs(f["glucose_median"] - r["glucose_median"]) < 0.05,
      f"C {f['glucose_median']:.3f}  Python {r['glucose_median']:.3f} mg/dL")
q = c["int8"]
dq = abs(q["glucose_median"] - r["glucose_median"])
check("glucose (int8 weights)", dq < 2.0, f"C {q['glucose_median']:.3f} mg/dL, diff {dq:.3f} (quantisation)")
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
