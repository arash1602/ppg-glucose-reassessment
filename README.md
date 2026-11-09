# Reassessing PPG-based blood glucose estimation — code and reproduction package

Code, simulation, firmware and manuscript sources for:

> M. Ghamari, M. Chhetri, S. Kundu, K. Alipour, B. Tarvirdizadeh. *Reassessing photoplethysmography-based blood glucose estimation:
> patient-independent validation, optical modeling, and embedded implementation.* (manuscript, 2026)

It extends Zeynali et al., *Sci. Rep.* 15:581 (2025) to patient-wise cross-validation on
VitalDB, tests on MUST, models the optics of three-layer skin with Monte Carlo (MCX), and runs a
wearable-input model on an ESP32-S3 (and STM32 boards) in plain C.

```
ppg-glucose-reassessment/
├── code/                 analysis pipeline (Python), steps 01–21, Windows .bat launchers
│   └── esp32/            firmware (PlatformIO: ESP32-S3, STM32F746G-DISCO, NUCLEO-WB55),
│                         PC bit-exactness checks, hardware-in-the-loop and energy tools
├── simulation/           Monte Carlo skin model (MCX/pmcx) and confounder / net-analyte analysis
├── paper/                manuscript (LaTeX), figure script, data-collection script
├── expected_results/     the result files behind every table, for comparison with your run
├── requirements.txt      Python packages (versions used for the tables)
└── README.md             this file
```

The Monte Carlo skin model was developed by Milan Chhetri ([Glucose_Sensor_Ranking](https://github.com/milanbhutan/Glucose_Sensor_Ranking) framework);
`simulation/` contains the version used for the paper with our extensions (660–970 nm, public water
spectrum, confounder, pulsatile-channel and net-analyte-signal analyses).

---------------------------------------------------------------------------------------------------
## 1. Requirements

| | |
|---|---|
| OS | Windows 10/11 (launchers are `.bat`); every step is a plain `python` command and also runs on Linux/macOS |
| Python | 3.11–3.13 |
| GPU | NVIDIA GPU with CUDA for the deep networks (step 03) and the Monte Carlo simulation (≈ 4–6 GB VRAM is enough; developed on an RTX 4050 laptop GPU) |
| Disk | ≈ 15 GB for VitalDB windows (kept outside the project folder) |
| Time | download 4–10 h (network bound); deep networks ≈ 6 h; feature models < 1 h; simulation ≈ 6–8 h |
| Hardware (optional) | ESP32-S3-DevKitC-1; STM32F746G-DISCO; NUCLEO-WB55; a multimeter for energy |

Install (Windows): run `code\setup.bat` once. It creates `C:\glucose_venv`, installs PyTorch with CUDA
and `code\requirements.txt`. On Linux/macOS:

```
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126     # or the CPU build
pip install -r requirements.txt
```

**Numerical reproducibility.** Folds, training and bootstraps use fixed seeds (`config.SEED = 42`,
bootstrap seed 0). The feature-model tables (Tables 7–17) were produced with scikit-learn 1.7.2,
NumPy 2.2.6, pandas 2.3.3 and SciPy 1.15.3; other versions change gradient-boosting results in the
second decimal (≤ 0.15 mg/dL). Deep-network results vary with GPU, CUDA and PyTorch versions within
the seed-to-seed spread reported in Table 6.

## 2. Data

* **VitalDB** (open, https://vitaldb.net): downloaded automatically by step 01 through the `vitaldb`
  package; nothing to do by hand.
* **MUST** (Mendeley Data, doi:10.17632/37pm7jk7jn.3): download and unzip into `<DATA_DIR>/must`.

`DATA_DIR` defaults to `C:\glucose_data` (Windows) or `~/glucose_data` (Linux/macOS); override with the
environment variable `GLU_DATA_DIR`. Results go to `results/` next to `code/` (override with
`GLU_RESULTS_DIR`). Raw data are never written into the project folder.

## 3. Run order

All commands are run from `code/` (Windows launchers in brackets). Every step can be re-run; the
download and feature steps resume where they stopped.

| # | Command | What it produces |
|---|---|---|
| 0 | `python make_synthetic.py` then `[run_quicktest.bat]` | optional 15-min check on synthetic and 40 real cases |
| 1 | `python 01_download_vitaldb.py --workers 6` | ±8-min PLETH windows around every intra-operative glucose lab (`DATA_DIR/vitaldb_raw`) |
| 2 | `python 02_preprocess.py --workers 6` | band-pass, 1-s beats (template matching), 10-s segments, PPG features, `processed/readings.csv` |
| 3 | `python 04_baselines.py` | constant, demographic, pre-op carry-forward and PPG-feature baselines |
| 4 | `[run_all.bat]` step 4: `python 03_train_eval.py ...` (7 runs, see `run_all.bat`) | ResNet34-1D, 10-s ResNet, TinyCNN; segment-level and person-level splits |
| 5 | `python 05_personalize.py`, `python 07_report.py` | calibration baseline, `results/summary_table.csv` (Table 4), Clarke grids |
| 6 | `[run_improve.bat]` `python 08_improve.py` | multi-beat (deep-set) CNN, PPG-change calibrated models |
| 7 | `[run_multisignal.bat]` `python 09_multisignal.py --workers 2` and `python 10_multisignal_eval.py` | ECG, arterial, monitor and drug features; Tables 7 and 9 |
| 8 | `python 11_must_external_features.py`; `[run_must_cnn.bat]` | MUST all-recordings test (Table 14) |
| 9 | `[run_reviewer.bat]` `python 16_causal_ablation.py` | causal windows (Table 11) and leave-one-group-out (Table 8) |
| 10 | `[run_robustness.bat]` | extra seeds and training variants (runs of `03_train_eval.py`) |
| 11 | `python 18_robustness_factorial.py` | Tables 5 (factorial) and 6 (robustness) |
| 12 | `python 19_must_matched.py` | Table 15 (MUST, matched 48 recordings) |
| 13 | `python 20_subgroups_detection.py` | Tables 12 and 13, diabetes PPG gain |
| 14 | `python 13_device_train.py`, then `[run_device_avg.bat]` `python 14_device_avg.py` | device features, device model, firmware export (`esp32/firmware/include/device_model.h`) |
| 15 | `python 17_device_window_check.py` | Table 18 (device model by PPG window) |
| 16 | `[run_foundation.bat]`: `python 21_foundation_models.py download`, `embed`, `finetune`, `finetune --demo`, `probe` | pretrained PPG foundation models PaPaGei-S and Pulse-PPG, frozen and fine-tuned (Table 10; GPU, about 1.5 h; weights downloaded from Zenodo) |

Step 12 (`12_device_models.py`) is an earlier version of the device model kept for completeness; the
paper uses steps 13, 14 and 17.

### Simulation (`simulation/`)
```
pip install pmcx pyyaml
python src/main_extended.py                                  # 78 wavelengths x 5 batches (GPU, hours)
python src/analysis/physio_confounders.py results_extended   # glucose-equivalent errors, pulsatile channel, NAS
python src/analysis/nas_uncertainty.py results_extended      # rank / conditioning and Monte Carlo uncertainty
```
(`run_extended_sim.bat` runs all three.)

Tissue-model sensitivity analysis (Table 17 and Fig. 9d): `run_tissue_sensitivity.bat`, or
```
python src/main_tissue2.py                                     # alternative continuous tissue model, 86 wavelengths x 5 batches (GPU, ~6.5 h)
python src/analysis/physio_confounders.py results_tissue2
python src/analysis/nas_uncertainty.py results_tissue2
python src/analysis/compare_tissue_models.py results_extended results_tissue2
python src/analysis/robust_nas.py results_extended results_tissue2   # exhaustive four-wavelength search (Tables 16-17)
```
The alternative model (`src/tissue_model_v2/params2.py`) adapts M. Chhetri's continuous 600-1700 nm model
(Salomatina 2006 scattering with a power-law fit above 1330 nm, continuous anisotropy, glucose in the dermis). Outputs are in `results_extended/physio_analysis/`; the
reference outputs of the paper's run are included, so the analysis scripts can be re-run without the
GPU simulation. Absorbance is `A = -ln R`; glucose sensitivities are per mmol/L.

### Figures and manuscript (`paper/`)
```
python collect_data.py     # copies the needed results and simulation outputs into paper/data
python make_figs.py        # all figures into paper/figs
pdflatex paper.tex  (three times)
```

## 4. Where each result comes from

| Paper item | Script | Output file |
|---|---|---|
| Fig. 2 participant flow | 01, 02, 06/11 (logs) | `run_all_log.txt`, `must_cnn_log.txt` |
| Table 4, Figs. 5 and 6 | 03, 04, 07 | `summary_table.csv`, `preds/*.csv` |
| Table 5 factorial | 03, 18 | `factorial_table.csv` |
| Table 6 robustness | 03 (`run_robustness.bat`), 18 | `robustness_table.csv` |
| Tables 7 and 9, Fig. 7 | 09, 10 | `multisignal_tiers.csv`, `multisignal_calibrated.csv` |
| Table 8 leave-one-group-out | 16 | `ablation_logo.csv` |
| Table 10 foundation models | 21 (encoders in `fm_models.py`) | `foundation_models.csv`, `foundation_models_must.csv`, `preds/fm_*.csv` |
| Table 11 causal windows | 16 | `causal_windows.csv` |
| Tables 12 and 13, Fig. 8 | 20 (Fig. 8 from `preds/`) | `subgroups.csv`, `detection.csv`, `diabetes_ppg_gain.csv` |
| Table 14 MUST, all recordings | 06, 11 | `must_external_features.csv`, `preds/must_*.csv` |
| Table 15 MUST, matched 48 | 19 | `must_matched.csv` |
| Table 16, Fig. 9 simulation | `simulation/` | `results_extended/physio_analysis/*`, `results_tissue2/physio_analysis/robust_nas.csv` |
| Table 17 tissue-model sensitivity | `simulation/` (tissue sensitivity) | `results_tissue2/physio_analysis/tissue_model_comparison.csv`, `robust_nas.csv` |
| Table 18 device model | 13, 14, 17 | `device_window_check.csv` |
| Table 19, Fig. 10 ESP32 | `esp32/` (section 5) | board self-test output, `esp32_hil_avg.csv` |
| Table 20 summary | collects Tables 4, 7, 9, 10, 18 | |

`expected_results/` holds the files of the paper's run for comparison.

## 5. Firmware and hardware tests (`code/esp32/`)

The device model is plain C (`firmware/src/glucose.c`, `device.c`), no ML library.

1. **PC check (no board):** `cd esp32/pc_test` and `python device_check_avg.py` — compiles the C code
   with gcc and compares every feature and estimate with Python (expected: 0.00 mg/dL).
   `device_check.py` checks the earlier centred-window model of step 13.
2. **Flash:** `esp32\flash_esp32.bat` (ESP32-S3, 921600 baud) or `esp32\flash_stm32.bat f746|wb55`
   (STM32, 460800 baud; the WB55 build is calibration-free only for memory). The board prints a
   self-test (`SELFTEST PASS`). PlatformIO downloads the toolchains on first use.
3. **Hardware-in-the-loop:** `esp32\hil_test_avg.bat COM3` (ESP32) or `esp32\hil_test_stm32.bat COM7`
   streams 30 VitalDB recordings and compares each estimate with Python.
4. **Energy per estimate:** `esp32\energy_test.bat COMx <baud> <volts> "<where>"` — the board runs an
   idle phase and a compute phase (`BENCH` command); type the multimeter readings when prompted.
   Energy = (I_run − I_idle) × V × time per estimate. Results: `results/energy.csv`.

`esp32/export_device_test.py` regenerates the self-test header from the exported model.

## 6. Differences from the original evaluation protocol (Zeynali et al. 2025)

| Aspect | Original study | This study |
|---|---|---|
| Data split | random split of segment files (segments of one person can be in training and test) | 5-fold split by VitalDB `subjectid`; validation set also by person |
| Test readings | held-out random segments | every glucose reading tested once, out of fold |
| Unit of evaluation | each 10-s / 1-s segment | one prediction per glucose reading (median over its beats) |
| MUST signal-label matching | by directory order | by file name |
| Peak spacing | 0.8 s (up to 75 bpm) | 0.4 s (up to 150 bpm) |
| Baselines | not reported | constant mean, demographics, pre-op glucose, PPG features, clinical context |
| Within-person tracking | not reported | within-subject correlation and trend accuracy |

## 7. Licence and citation

Code is released under the MIT licence (see `LICENSE`).
Please cite the paper above and, for the datasets, Lee et al., *Sci. Data* 9:279 (2022) (VitalDB) and
Kermani & Esmaeili, Mendeley Data, doi:10.17632/37pm7jk7jn (2023) (MUST).
