# ESP32 deployment (TinyCNN, plain C)

Pipeline on the device = the Python pipeline, step for step:
band-pass (scipy `sosfiltfilt`) -> robust z-score -> `find_peaks` (distance 0.4 s, prominence 0.8)
-> 1-s windows with exactly one peak -> template (mean) + cosine >= 0.85 -> min-max [-1, 1]
-> TinyCNN (12.3 k parameters, BatchNorm folded) per beat -> median over beats.

| File | Purpose |
|---|---|
| `export_esp32.py` | checkpoint -> `firmware/include/glucose_model.h` (float32 + int8 weights), `dsp_coeffs.h`, `test_window.h` |
| `firmware/` | PlatformIO project (ESP32-S3-DevKitC-1; also esp32dev, esp32-c3) |
| `firmware/src/glucose.c` | DSP + inference, no dynamic memory, no libraries |
| `firmware/src/main.cpp` | boot self-test vs Python reference; serial streaming mode |
| `pc_test/` | compile the same C on a PC (gcc) and compare with Python: `compare.py`, `stress_test.py` |
| `stream_test.py` | hardware-in-the-loop: real VitalDB windows -> ESP32 -> compare + timing (`results/esp32_hil.csv`) |
| `flash_esp32.bat` | export + build + flash + serial monitor |
| `hil_test.bat COM5` | run the hardware-in-the-loop test |

## Verified (PC, gcc)
* 300/300 difficult synthetic windows: identical beat selection, 0.0000 mg/dL difference (float32).
* int8 weights: 0.016 mg/dL difference on the test window; model 48 kB (float32) / 12 kB (int8).

## Known limitation inherited from the original method
A 1-s window must contain exactly one peak, so beats are rejected above ~120 bpm.

## Final model for the device
Use a TinyCNN trained on all subjects (not a cross-validation fold) once the evaluation is finished,
then re-run `flash_esp32.bat` with that checkpoint.
