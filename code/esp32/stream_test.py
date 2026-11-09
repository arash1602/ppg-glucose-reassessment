"""Hardware-in-the-loop test of the device model on the ESP32.

Streams the 60 real VitalDB reference windows (pc_test/device_reference.json, written by
13_device_train.py) to the board with the same commands a phone app would use:
    PROFILE <age> <sex_m> <bmi> <diabetes>
    [CAL <fingerstick mg/dL>  + calibration window,  MIN <minutes since>]
    BEGIN 6000 / samples / END
and compares the board's glucose estimates with the Python model, plus per-window timing.

  python esp32/stream_test.py --port COM3          one-minute windows (device_reference.json, step 13)
  python esp32/stream_test.py --port COM3 --avg    sequences of K one-minute windows (device_reference_avg.json,
                                                   step 14): RESET, K windows -> averaged estimate
The --avg mode is chosen automatically when device_model.h was exported by 14_device_avg.py.
Output: results/esp32_hil.csv (or esp32_hil_avg.csv) and a summary for the paper.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial          # pip install pyserial

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent))
import config as C     # noqa: E402


CHUNK, PAUSE = 500, 0.003


def send(ser, lines):
    ser.write(("\n".join(lines) + "\n").encode())


def window(ser, x, timeout=30):
    """send one window in chunks of 250 samples; wait for the board's ack after each chunk"""
    ser.reset_input_buffer()
    send(ser, [f"BEGIN {len(x)}"])
    for i in range(0, len(x), 250):
        send(ser, [repr(float(np.float32(v))) for v in x[i:i + 250]])
        if i + 250 < len(x):
            t0 = time.time()
            while time.time() - t0 < 5:
                line = ser.readline().decode(errors="ignore").strip()
                if line.startswith('{"ack"'):
                    break
            else:
                print(f"    no ack after {i + 250} samples")
                return None
    send(ser, ["END"])
    t0, other = time.time(), []
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith('{"tag":"window"'):
            return json.loads(line)
        if line:
            other.append(line)
    print("    no result line from the board; it said:" if other else "    no result line and no other output from the board")
    for l in other[-12:]:
        print("      |", l[:150])
    return None


def wait_ready(ser, timeout=20):
    """the board runs its self-test when the port opens; wait for it to finish"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("  board:", line[:110])
        if line.startswith("ready:"):
            return True
    return False


def fmt(v):
    return "nan" if v is None or not np.isfinite(v) else repr(float(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--slow", action="store_true", help="smaller chunks + longer pauses if samples are lost")
    ap.add_argument("--avg", action="store_true", help="K-minute averaged sequences (device_reference_avg.json)")
    ap.add_argument("--baud", type=int, default=921600, help="921600 for ESP32-S3, 460800 for the STM32 boards")
    ap.add_argument("--out", default=None, help="output CSV name in results/ (default esp32_hil[_avg].csv)")
    a = ap.parse_args()
    global CHUNK, PAUSE
    if a.slow:
        CHUNK, PAUSE = 100, 0.02
    hdr = (H / "firmware" / "include" / "device_model.h").read_text().splitlines()[0]
    if a.avg or "14_device_avg.py" in hdr:
        return main_avg(a)
    ref = json.loads((H / "pc_test" / "device_reference.json").read_text())
    ser = serial.Serial(a.port, a.baud, timeout=2)
    if not wait_ready(ser):
        print("board not ready - sending a self-test request")
        send(ser, ["s"])
        if not wait_ready(ser):
            print("no 'ready' from the board - is the serial monitor still open?"); return
    rows = []
    for i, r in enumerate(ref):
        send(ser, ["PROFILE " + " ".join(fmt(v) for v in r["profile"])])
        time.sleep(0.05)
        if "x_cal" in r:
            send(ser, [f"CAL {r['cal_glucose']}"])
            time.sleep(0.05)
            if window(ser, r["x_cal"]) is None:
                print(f"[{i}] no answer for the calibration window"); continue
            send(ser, [f"MIN {r['minutes_since_cal']!r}"])
            time.sleep(0.05)
        d = window(ser, r["x"])
        if d is None:
            print(f"[{i}] no answer"); continue
        cal = "x_cal" in r
        if d.get("glucose_free") is None or d.get("n") != len(r["x"]):
            print(f"[{i}] board reply: received {d.get('n')} of {len(r['x'])} samples, rc {d.get('rc')}, "
                  f"{d.get('n_beats')} beats (Python {r['n_beats']}) -> {json.dumps(d)[:160]}")
            if d.get("n") != len(r["x"]):
                print("    samples were lost on the USB link - see --slow")
            continue
        row = {"rid": r["rid"], "lab_mgdl": r["label"], "calibrated": cal, "n_beats": d["n_beats"],
               "py_beats": r["n_beats"], "esp32_free": d["glucose_free"], "python_free": r["pred_free"],
               "esp32_cal": d["glucose_cal"] if cal else np.nan, "python_cal": r.get("pred_cal", np.nan),
               "ms_extract": d["us_extract"] / 1e3, "ms_features": d["us_features"] / 1e3,
               "ms_trees": d["us_trees"] / 1e3, "ms_total": d["us_total"] / 1e3}
        rows.append(row)
        g = row["esp32_cal"] if cal else row["esp32_free"]
        p = row["python_cal"] if cal else row["python_free"]
        print(f"[{i+1:2d}/{len(ref)}] {'calibrated' if cal else 'free      '}  lab {r['label']:5.0f}  "
              f"ESP32 {g:8.3f}  Python {p:8.3f}  ({d['n_beats']} beats, {row['ms_total']:.0f} ms)")
    ser.close()
    if not rows:
        print("no usable windows - see the lines above"); return
    D = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    D.to_csv(C.RESULTS_DIR / "esp32_hil.csv", index=False)
    c = D[D.calibrated]
    print(f"\n{len(D)} windows on the ESP32 | identical beat counts {(D.n_beats == D.py_beats).mean():.0%}")
    print(f"free model:       max |ESP32 - Python| = {np.abs(D.esp32_free - D.python_free).max():.2e} mg/dL")
    if len(c):
        print(f"calibrated model: max |ESP32 - Python| = {np.abs(c.esp32_cal - c.python_cal).max():.2e} mg/dL")
    print(f"time per 60-s window (median): extract {D.ms_extract.median():.0f} ms, features {D.ms_features.median():.1f} ms, "
          f"trees {D.ms_trees.median():.2f} ms, total {D.ms_total.median():.0f} ms")
    print("saved", C.RESULTS_DIR / "esp32_hil.csv")


def num(v):
    """JSON value from the board -> float (None / "nan" -> NaN, e.g. the WB55 has no calibrated model)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main_avg(a):
    port = a.port
    R = json.loads((H / "pc_test" / "device_reference_avg.json").read_text())
    K, ref = R["K"], R["cases"]
    print(f"averaged mode: {len(ref)} sequences of K = {K} one-minute window(s)")
    ser = serial.Serial(port, a.baud, timeout=2)
    if not wait_ready(ser):
        print("board not ready - sending a self-test request")
        send(ser, ["s"])
        if not wait_ready(ser):
            print("no 'ready' from the board - is the serial monitor still open?"); return
    rows = []
    for i, r in enumerate(ref):
        send(ser, ["PROFILE " + " ".join(fmt(v) for v in r["profile"]), "RESET"])
        time.sleep(0.05)
        cal = "cal_windows" in r
        if cal:
            for x in r["cal_windows"][:-1]:
                window(ser, x)
            send(ser, [f"CAL {r['cal_glucose']}"]); time.sleep(0.05)
            if window(ser, r["cal_windows"][-1]) is None:
                print(f"[{i}] no answer for the calibration window"); continue
            send(ser, ["RESET"]); time.sleep(0.05)
        d, ms = None, []
        for j, x in enumerate(r["windows"]):
            if cal and j == len(r["windows"]) - 1:
                send(ser, [f"MIN {r['minutes_since_cal']!r}"]); time.sleep(0.05)
            d = window(ser, x)
            if d is None or d.get("n") != len(x):
                break
            ms.append(d["us_total"] / 1e3)
        if d is None or d.get("glucose_free") is None:
            print(f"[{i}] no usable answer -> {json.dumps(d)[:160] if d else None}"); continue
        if d.get("avg_k") != K:
            print(f"the board averages {d.get('avg_k')} window(s) but the reference has K = {K}: reflash (flash_esp32.bat)"); break
        row = {"rid": r["rid"], "lab_mgdl": r["label"], "calibrated": cal, "n_avg": d["n_avg"],
               "esp32_free": d["glucose_free"], "python_free": r["pred_free"],
               "esp32_cal": num(d.get("glucose_cal")) if cal else np.nan, "python_cal": r.get("pred_cal", np.nan),
               "ms_per_window": float(np.median(ms))}
        rows.append(row)
        g, p = (row["esp32_cal"], row["python_cal"]) if cal and np.isfinite(row["esp32_cal"]) \
            else (row["esp32_free"], row["python_free"])
        print(f"[{i+1:2d}/{len(ref)}] {'calibrated' if cal else 'free      '}  lab {r['label']:5.0f}  ESP32 {g:8.3f}  "
              f"Python {p:8.3f}  ({d['n_avg']}/{len(r['windows'])} windows, {row['ms_per_window']:.0f} ms each)")
    ser.close()
    if not rows:
        print("no usable sequences - see the lines above"); return
    D = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out = C.RESULTS_DIR / (a.out or "esp32_hil_avg.csv")
    D.to_csv(out, index=False)
    c = D[D.calibrated & np.isfinite(D.esp32_cal)]
    print(f"\n{len(D)} estimates (K = {K} min) on the board ({len(c)} calibrated; a free-only build reports none)")
    print(f"free model:       max |ESP32 - Python| = {np.abs(D.esp32_free - D.python_free).max():.2e} mg/dL")
    if len(c):
        print(f"calibrated model: max |ESP32 - Python| = {np.abs(c.esp32_cal - c.python_cal).max():.2e} mg/dL")
    print(f"time per one-minute window (median): {D.ms_per_window.median():.0f} ms")
    print("saved", out)


if __name__ == "__main__":
    main()
