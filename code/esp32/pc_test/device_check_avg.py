"""PC check of the firmware's running feature average (DEV_AVG_WINDOWS one-minute windows).

  python esp32/pc_test/device_check_avg.py            uses device_reference_avg.json (from 14_device_avg.py)
  python esp32/pc_test/device_check_avg.py --selftest K   no reference needed: averages K of the 60-s
                                                       windows in device_reference.json and compares the
                                                       C features with numpy (device_features.py)
Protocol per case:  PROFILE / RESET / [cal windows, CAL before the last] / MIN / RESET / K windows.
"""
import json, os, subprocess, sys, warnings
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; FW = H.parent / "firmware"
sys.path.insert(0, str(H.parent.parent))
fmt = lambda v: "nan" if v is None or not np.isfinite(v) else repr(float(v))


def build(extra=()):
    exe = H / ("dev_avg_test.exe" if sys.platform == "win32" else "dev_avg_test")
    subprocess.run(["gcc", "-O2", "-std=c99", *extra, *os.environ.get("XFLAGS", "").split(), f"-I{FW/'include'}", str(H / "device_main.c"),
                    str(FW / "src" / "glucose.c"), str(FW / "src" / "device.c"), "-lm", "-o", str(exe)], check=True)
    return exe


def win(x):
    return [f"BEGIN {len(x)}"] + [repr(float(np.float32(v))) for v in x] + ["END"]


def protocol(r):
    L = ["PROFILE " + " ".join(fmt(v) for v in r["profile"]), "RESET"]
    if "cal_windows" in r:
        for x in r["cal_windows"][:-1]:
            L += win(x)
        L += [f"CAL {r['cal_glucose']}"] + win(r["cal_windows"][-1])
        L += ["RESET"]
    for i, x in enumerate(r["windows"]):
        if i == len(r["windows"]) - 1 and "cal_windows" in r:
            L.append(f"MIN {r['minutes_since_cal']!r}")
        L += win(x)
    return L


def run(exe, cases):
    lines, ends = [], []
    for r in cases:
        lines += protocol(r)
        ends.append(None)
    out = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    res = [json.loads(l) for l in out.splitlines() if l.startswith('{"tag":"window"')]
    last, k = [], 0
    for r in cases:                                      # the answer for each case = its final window
        k += (len(r.get("cal_windows", [])) + len(r["windows"]))
        last.append(res[k - 1])
    return last


def fdiff(d, ref):
    a = np.array([np.nan if v is None else v for v in d["features"]], float)
    b = np.array([np.nan if v is None else v for v in ref], float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    return float(np.nanmax(np.abs(a - b) / np.maximum(1, np.abs(b)))) if np.isfinite(b).any() else 0.0


def selftest(K):
    import device_features as DF
    ref = json.loads((H / "device_reference.json").read_text())
    xs = [r["x"] for r in ref]
    cases = []
    for i in range(0, len(xs) - K + 1, K):
        W = xs[i:i + K]
        F = []
        for x in W:
            f = DF.device_features(np.asarray(x, np.float32))
            F.append([f[c] for c in DF.FEATURES] if f["n_beats"] >= 3 else [np.nan] * len(DF.FEATURES))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(np.array(F, float)[None], axis=1)[0]
        cases.append({"profile": ref[i]["profile"], "windows": W, "features": [None if not np.isfinite(v) else v for v in m]})
    res = run(build([f"-DDEV_AVG_WINDOWS={K}"]), cases)
    d = [fdiff(a, c["features"]) for a, c in zip(res, cases)]
    ok = max(d) < 1e-9 and all(a["avg_k"] == K for a in res)
    print(f"self-test K={K}: {len(cases)} sequences | max averaged-feature diff {max(d):.2e} (relative)")
    return ok


def reference():
    R = json.loads((H / "device_reference_avg.json").read_text())
    K, cases = R["K"], R["cases"]
    res = run(build(), cases)
    fd = [fdiff(a, c["features"]) for a, c in zip(res, cases)]
    pf = [abs(a["glucose_free"] - c["pred_free"]) for a, c in zip(res, cases)]
    pc = [abs(a["glucose_cal"] - c["pred_cal"]) for a, c in zip(res, cases) if "pred_cal" in c]
    kk = {a["avg_k"] for a in res}
    ok = kk == {K} and max(fd) < 1e-9 and max(pf) < 1e-6 and (not pc or max(pc) < 1e-6)
    print(f"K={K} ({kk} in firmware) | {len(cases)} sequences ({len(pc)} calibrated) | max feature diff {max(fd):.2e}")
    print(f"free model:       max |C - Python| = {max(pf):.2e} mg/dL")
    if pc:
        print(f"calibrated model: max |C - Python| = {max(pc):.2e} mg/dL")
    return ok


if __name__ == "__main__":
    ok = selftest(int(sys.argv[sys.argv.index("--selftest") + 1])) if "--selftest" in sys.argv else reference()
    print("ALL AVERAGING CHECKS PASSED" if ok else "SOME AVERAGING CHECKS FAILED")
    sys.exit(0 if ok else 1)
