"""Tissue-model sensitivity analysis: the same sweep as main_extended.py, with the alternative continuous
600-1700 nm tissue model (tissue_model_v2/params2.py) instead of the Iino / composition split.

Everything else is unchanged: geometry, detector rings, 1e8 photons, 5 batches, glucose in the DERMIS only
(dmu_a/dCg from Ge et al. 2024 >= 1000 nm, 0 below; dmu_s/mu_s = -0.29 % per mmol/L). The fat-layer and
below-1000-nm glucose variants are computed afterwards from the saved path lengths
(src/analysis/compare_tissue_models.py), so they need no extra GPU time.

Run from the simulation (MCeXtreme_Sim) folder:   python src/main_tissue2.py        (GPU, about 7-9 h)
Dry run without a GPU (checks the pipeline with fake numbers):  python src/main_tissue2.py --dry
Output: results_tissue2/ (mean_metrics, error_metrics, inputs_per_wavelength.csv)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import tissue_model_v2.params2 as params

DRY = "--dry" in sys.argv
OUT = Path("results_tissue2_dry" if DRY else "results_tissue2")
WAVELENGTHS = sorted(set(list(range(600, 1000, 40)) + [660, 760, 810, 850, 880, 940, 970] + list(range(1000, 1701, 10))))
BATCHES = 5
SDS = ["SDS_1.7mm", "SDS_2.0mm", "SDS_2.3mm", "SDS_2.6mm", "SDS_2.9mm"]
DMU_S_DCG_REL = -0.0029

ge = pd.read_csv("Glucose_Absorption_Data/ge2024_dmu_a_dCg.csv").set_index("wavelength_nm").sort_index()
dmu_a_dCg = pd.Series({w: (ge.loc[w, "dmu_a_dCg_mm_inv_per_mmol_L"] if w in ge.index else 0.0) for w in WAVELENGTHS})


def layer(prefix, w):
    return [float(getattr(params, f"{prefix}_mu_a_sim")[w]), float(getattr(params, f"{prefix}_mu_s_sim")[w]),
            float(getattr(params, f"{prefix}_g_sim")[w]), float(getattr(params, f"{prefix}_n_sim"))]


def fake_run(glass, epi, derm, sub):
    """Plausible stand-in for photon_sim.monte_carlo.run_simulation (dry run only)."""
    rng = np.random.default_rng()
    r = np.array([1.7, 2.0, 2.3, 2.6, 2.9]); att = np.exp(-(derm[0] + 0.2 * sub[0]) * 5 * r)
    Ld, Ls = 2.0 * r * att ** 0.1 * (1 + 0.01 * rng.standard_normal(5)), 1.2 * r * (1 + 0.01 * rng.standard_normal(5))
    return {"mean_path": {"epi_mean_path": 0.3 * r, "derm_mean_path": Ld, "sub_mean_path": Ls},
            "dA_dmu_s": {"epi_dA_dmu_s": -0.01 * r, "derm_dA_dmu_s": -0.05 * r * (1 + 0.01 * rng.standard_normal(5)),
                         "sub_dA_dmu_s": -0.04 * r},
            "sensitivity": {"epi_sensitivity": 0.1 + 0 * r, "derm_sensitivity": 0.5 + 0 * r, "sub_sensitivity": 0.4 + 0 * r},
            "power_dB": -15 - 5 * r - 10 * derm[0]}


def main():
    run = fake_run if DRY else __import__("photon_sim.monte_carlo", fromlist=["run_simulation"]).run_simulation
    keys = ["dA_dCg_total", "dA_dCg_mu_a", "dA_dCg_mu_s", "dA_dmu_s", "mean_path_epi", "mean_path_derm",
            "mean_path_sub", "epi_dA_dmu_s", "derm_dA_dmu_s", "sub_dA_dmu_s", "beta_epi", "beta_derm",
            "beta_sub", "power_dB"]
    batches = {k: [] for k in keys}
    glass = [params.glass_mu_a_sim, params.glass_mu_s_sim, params.glass_g_sim, params.glass_n_sim]
    t0 = time.time()
    for b in range(BATCHES):
        frames = {k: pd.DataFrame(np.nan, index=WAVELENGTHS, columns=SDS) for k in keys}
        for i, w in enumerate(WAVELENGTHS):
            r = run(glass, layer("epi", w), layer("derm", w), layer("sub", w))
            mus = float(params.derm_mu_s_sim[w])
            L_d, s_d = r["mean_path"]["derm_mean_path"], r["dA_dmu_s"]["derm_dA_dmu_s"]
            frames["dA_dCg_mu_a"].loc[w] = L_d * dmu_a_dCg[w]
            frames["dA_dCg_mu_s"].loc[w] = s_d * DMU_S_DCG_REL * mus
            frames["dA_dCg_total"].loc[w] = frames["dA_dCg_mu_a"].loc[w] + frames["dA_dCg_mu_s"].loc[w]
            frames["dA_dmu_s"].loc[w] = s_d
            for lay in ["epi", "derm", "sub"]:
                frames[f"mean_path_{lay}"].loc[w] = r["mean_path"][f"{lay}_mean_path"]
                frames[f"{lay}_dA_dmu_s"].loc[w] = r["dA_dmu_s"][f"{lay}_dA_dmu_s"]
                frames[f"beta_{lay}"].loc[w] = r["sensitivity"][f"{lay}_sensitivity"]
            frames["power_dB"].loc[w] = r["power_dB"]
            done = b * len(WAVELENGTHS) + i + 1; total = BATCHES * len(WAVELENGTHS)
            eta = (time.time() - t0) / done * (total - done) / 3600
            print(f"batch {b+1}/{BATCHES}  {w} nm  ({done}/{total}, about {eta:.1f} h left)", flush=True)
        for k in keys:
            batches[k].append(frames[k])
    (OUT / "mean_metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "error_metrics").mkdir(parents=True, exist_ok=True)
    for k, v in batches.items():
        arr = np.stack([f.to_numpy() for f in v])
        pd.DataFrame(arr.mean(0), index=WAVELENGTHS, columns=SDS).to_csv(
            OUT / "mean_metrics" / f"Metric_Mean_{k}.csv", index_label="wavelength_nm")
        pd.DataFrame(arr.std(0, ddof=1) / np.sqrt(BATCHES), index=WAVELENGTHS, columns=SDS).to_csv(
            OUT / "error_metrics" / f"Metric_Error_{k}.csv", index_label="wavelength_nm")
    pd.DataFrame({"derm_mu_s_mm-1": [float(params.derm_mu_s_sim[w]) for w in WAVELENGTHS],
                  "derm_g": [float(params.derm_g_sim[w]) for w in WAVELENGTHS],
                  "sub_mu_s_mm-1": [float(params.sub_mu_s_sim[w]) for w in WAVELENGTHS],
                  "dmu_a_dCg_mm-1_per_mmol": dmu_a_dCg.values},
                 index=pd.Index(WAVELENGTHS, name="wavelength_nm")).to_csv(OUT / "inputs_per_wavelength.csv")
    (OUT / "tissue_model.txt").write_text(params.__doc__ + "\nscattering power-law fits: " + str(params.FIT_INFO) + "\n")
    print("saved to", OUT.resolve())


if __name__ == "__main__":
    main()
