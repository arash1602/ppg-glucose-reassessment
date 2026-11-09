# Patient-wise PPG → glucose pipeline (VitalDB + MUST)

Patient-wise evaluation of the approach of Zeynali et al., *Sci. Rep.* 15:581 (2025), using the same data,
the same 1-s beat segmentation and the same ResNet34-1D (7.38 M parameters),
but splitting train/test **by person** instead of by randomly shuffled segments.

## Differences from the original evaluation protocol
| Aspect | Original study | This study |
|---|---|---|
| Data split | random split of segment files (segments of one person can be in training and test) | 5-fold split by VitalDB `subjectid`; validation set also by person |
| Test readings | held-out random segments | every glucose reading tested once, out of fold |
| Unit of evaluation | each 10-s / 1-s segment | one prediction per glucose reading (median over its beats) |
| MUST signal-label matching | by directory order | by file name |
| Peak spacing | 0.8 s (up to 75 bpm) | 0.4 s (up to 150 bpm) |
| Baselines | not reported | constant mean, demographics, pre-op glucose, PPG features, clinical context |
| Within-person tracking | not reported | within-subject correlation and trend accuracy |

The segment-level protocol of the original study is also run (`--split leaky`, the option name in the code), so the effect of the split is measured directly.

## How to run (Windows)
1. `setup.bat` – once. Creates `C:\glucose_venv` and installs PyTorch with CUDA.
2. `run_quicktest.bat` – 10–20 min on 40 cases. Checks everything works on your PC.
3. `run_all.bat` – the full experiment. The download is the slow part; it can be stopped and restarted.
4. Optional: download MUST from Mendeley (doi:10.17632/37pm7jk7jn) and unzip it into `C:\glucose_data\must`.
   Then run `python 06_must_external.py --run tiny_beat_subject`.

Data (many GB) is stored in `C:\glucose_data`, outside the project folder.
Results go to `..\results` (`summary.md`, `summary_table.csv`, `figures\`, `preds\`).

## Files
| Step | File | Purpose |
|---|---|---|
| 1 | `01_download_vitaldb.py` | ±8-min PLETH windows around every intra-operative glucose lab value |
| 2 | `02_preprocess.py`, `ppg.py` | band-pass, 1-s beats (template matching), 10-s segments, morphology features |
| 3 | `03_train_eval.py`, `models.py` | ResNet34-1D / TinyCNN, segment-level vs person-level splits |
| 4 | `04_baselines.py` | mean, demographics, pre-op glucose carry-forward, PPG-feature GBM |
| 5 | `05_personalize.py` | first reading = calibration; compare against "carry the calibration forward" |
| 6 | `06_must_external.py` | external test on MUST, with constant-predictor comparators |
| 7 | `07_report.py` | tables with subject-level bootstrap 95 % CIs, Clarke error grids |
| – | `make_synthetic.py` | synthetic data for testing: the segment-level split must score well here and the person-level split must not |

## Validation on synthetic data (glucose NOT encoded in the waveform)
TinyCNN: segment-level split R² = +0.53, person split R² = −0.23. A reading-random split also
raises the score of a simple gradient-boosting model. This confirms the pipeline detects person overlap between training and test sets.
