"""Energy per glucose estimate, measured with a multimeter (all three boards).

The board alternates an IDLE phase (CPU sleeping between timer ticks) and a RUN phase (back-to-back
estimates on a real VitalDB window). You read the supply current in each phase and type it in.
Energy per estimate = (I_run - I_idle) x V x time per estimate.

  python esp32/energy_bench.py --port COM5 --volts 5.0 --where "USB 5 V"      (board-level, any board)
  python esp32/energy_bench.py --port COM7 --volts 3.3 --where "JP2 (MCU VDD)" (NUCLEO-WB55 MCU only)
Results are appended to results/energy.csv.
"""
import argparse, csv, sys, time
from pathlib import Path
import serial

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent))
import config as C  # noqa: E402


def wait_for(ser, token, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("  board:", line[:120])
        if line.startswith(token):
            return line
    raise SystemExit(f"no '{token}' from the board - is the serial monitor still open? wrong --baud?")


def ask(prompt):
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            print("  please type a number, e.g. 41.7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=None, help="921600 for ESP32-S3, 460800 for the STM32 boards")
    ap.add_argument("--volts", type=float, required=True, help="supply voltage at the measuring point")
    ap.add_argument("--where", default="USB 5 V", help="where the meter is inserted (for the record)")
    ap.add_argument("--seconds", type=int, default=40)
    a = ap.parse_args()
    ser = serial.Serial(a.port, a.baud or 921600, timeout=2)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(f"BENCH {a.seconds}\n".encode())
    wait_for(ser, "BENCH IDLE start")
    i_idle = ask(f"\nIDLE phase ({a.seconds} s): wait ~10 s for the meter to settle, then type the current in mA: ")
    wait_for(ser, "BENCH RUN start")
    i_run = ask(f"\nRUN phase ({a.seconds} s): wait ~10 s for the meter to settle, then type the current in mA: ")
    res = wait_for(ser, "BENCH RESULT")
    kv = dict(p.split("=") for p in res.replace("BENCH RESULT ", "").split())
    ms = float(kv["ms_per_estimate"])
    e_mj = (i_run - i_idle) * a.volts * ms / 1000.0
    p_run = i_run * a.volts
    print(f"\n{kv['board']}: {ms:.1f} ms per estimate | I_idle {i_idle} mA, I_run {i_run} mA at {a.volts} V ({a.where})")
    print(f"  extra power while estimating {(i_run - i_idle) * a.volts:.1f} mW -> energy per estimate {e_mj:.3f} mJ")
    print(f"  (one estimate per minute: {e_mj * 60 * 24 / 1000:.2f} J per day for the computation)")
    out = C.RESULTS_DIR / "energy.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["board", "where", "volts", "i_idle_mA", "i_run_mA", "ms_per_estimate", "estimates", "energy_mJ", "run_power_mW"])
        w.writerow([kv["board"], a.where, a.volts, i_idle, i_run, ms, kv["estimates"], round(e_mj, 4), round(p_run, 1)])
    print("saved", out)


if __name__ == "__main__":
    main()
