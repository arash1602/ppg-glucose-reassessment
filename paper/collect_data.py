"""Copy the pipeline and simulation outputs that the figures need into paper/data/.
Run after the pipeline (code/) and the simulation (simulation/) have finished:
  python collect_data.py            then   python make_figs.py   and   pdflatex paper.tex (3x)
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))
import config as C  # noqa: E402

OUT = HERE / "data"; (OUT / "ext").mkdir(parents=True, exist_ok=True)
RES, PRED = C.RESULTS_DIR, C.RESULTS_DIR / "preds"
copies = {
    RES / "summary_table.csv": OUT, RES / "multisignal_tiers.csv": OUT, RES / "multisignal_calibrated.csv": OUT,
    RES / "esp32_hil_avg.csv": OUT, RES / "must_external_features.csv": OUT, RES / "personalisation.csv": OUT,
    RES / "subgroups.csv": OUT, RES / "detection.csv": OUT,
    C.PROC_DIR / "readings.csv": OUT, C.RAW_DIR / "1.npz": OUT / "case1.npz",
}
for f in ["baseline_demo_subject", "baseline_mean_subject", "resnet34_beat_leaky", "resnet34_beat_subject",
          "ms_calib_c0_fingerstick___time___clinical", "ms_calib_c1___ppg_change", "ms_calib_c3___monitor___drugs_change",
          "ms_calib_persistence__no_model_", "ms_free_t1___time_in_surgery", "ms_free_t2___ppg", "ms_free_t5___drug_infusions"]:
    copies[PRED / f"{f}.csv"] = OUT
SIM = ROOT / "simulation" / "results_extended" / "physio_analysis"
SIM2 = ROOT / "simulation" / "results_tissue2" / "physio_analysis"
for f in ["robust_nas.csv", "tissue_model_comparison.csv"]:
    copies[SIM2 / f] = OUT / "ext"
for f in ["dc_glucose_equivalent_errors.csv", "ac_pulsatile_analysis.csv", "net_analyte_signal.csv", "scattering_assumption_scan.csv"]:
    copies[SIM / f] = OUT / "ext"
for f in ["Metric_Mean_dA_dCg_total.csv"]:
    copies[ROOT / "simulation" / "results_extended" / "mean_metrics" / f] = OUT / "ext"
for f in ["Metric_Error_dA_dCg_total.csv"]:
    copies[ROOT / "simulation" / "results_extended" / "error_metrics" / f] = OUT / "ext"
missing = []
for src, dst in copies.items():
    if src.exists():
        shutil.copy(src, dst)
    else:
        missing.append(str(src))
print("copied", len(copies) - len(missing), "files" + ("; missing:\n  " + "\n  ".join(missing) if missing else ""))
