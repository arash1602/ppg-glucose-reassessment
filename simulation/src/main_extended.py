"""
Extended wavelength sweep: pulse-oximeter wavelengths (660-970 nm) + 1000-1700 nm.

Same Monte Carlo model and metrics as main.py, plus:
  * the wavelengths used by clinical pulse oximeters (660 / 940 nm, as in VitalDB),
    so the physics can be compared directly with the VitalDB machine-learning results;
  * 5 batches (instead of 2) so the reported standard errors mean something;
  * dermal mu_s and dmu_a/dCg used at each wavelength are saved next to the metrics,
    so src/analysis/physio_confounders.py can run on these results.

Glucose absorption below 1000 nm: Ge et al. 2024 starts at 1000 nm. Glucose absorption
at 600-1000 nm (3rd overtone, ~940 nm) is far weaker and is set to 0 here, so below
1000 nm only the scattering route (dmu_s/dCg) contributes. Replace with literature
values if available.

Run from the MCeXtreme_Sim folder:  python src/main_extended.py
Output: results_extended/mean_metrics, results_extended/error_metrics
Runtime: ~77 wavelengths x 5 batches x 1e8 photons (about 1 h on an RTX 4050).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import tissue_model.tissue_params as params
import photon_sim.monte_carlo as mc

OUT = Path("results_extended")
WAVELENGTHS = [660, 760, 810, 850, 880, 940, 970] + list(range(1000, 1701, 10))
BATCHES = 5
SDS = ["SDS_1.7mm", "SDS_2.0mm", "SDS_2.3mm", "SDS_2.6mm", "SDS_2.9mm"]
DMU_S_DCG_REL = -0.0029     # same assumption as main.py (relative change of mu_s per mmol/L)

ge = pd.read_csv("Glucose_Absorption_Data/ge2024_dmu_a_dCg.csv").set_index("wavelength_nm").sort_index()
dmu_a_dCg = pd.Series({w: (ge.loc[w, "dmu_a_dCg_mm_inv_per_mmol_L"] if w in ge.index else 0.0)
                       for w in WAVELENGTHS})


def layer(prefix, w):
    return [getattr(params, f"{prefix}_mu_a_sim")[w], getattr(params, f"{prefix}_mu_s_sim")[w],
            getattr(params, f"{prefix}_g_sim")[w], getattr(params, f"{prefix}_n_sim")]


def main():
    keys = ["dA_dCg_total", "dA_dCg_mu_a", "dA_dCg_mu_s", "dA_dmu_s", "mean_path_epi", "mean_path_derm",
            "mean_path_sub", "epi_dA_dmu_s", "derm_dA_dmu_s", "sub_dA_dmu_s", "beta_epi", "beta_derm",
            "beta_sub", "power_dB"]
    batches = {k: [] for k in keys}
    glass = [params.glass_mu_a_sim, params.glass_mu_s_sim, params.glass_g_sim, params.glass_n_sim]
    for b in range(BATCHES):
        frames = {k: pd.DataFrame(np.nan, index=WAVELENGTHS, columns=SDS) for k in keys}
        for i, w in enumerate(WAVELENGTHS):
            r = mc.run_simulation(glass, layer("epi", w), layer("derm", w), layer("sub", w))
            mus = params.derm_mu_s_sim[w]
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
            print(f"batch {b+1}/{BATCHES}  {w} nm  ({i+1}/{len(WAVELENGTHS)})", flush=True)
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
    pd.DataFrame({"derm_mu_s_mm-1": [params.derm_mu_s_sim[w] for w in WAVELENGTHS],
                  "derm_g": [params.derm_g_sim[w] for w in WAVELENGTHS],
                  "dmu_a_dCg_mm-1_per_mmol": dmu_a_dCg.values},
                 index=pd.Index(WAVELENGTHS, name="wavelength_nm")).to_csv(OUT / "inputs_per_wavelength.csv")
    print("saved to", OUT.resolve())


if __name__ == "__main__":
    main()
