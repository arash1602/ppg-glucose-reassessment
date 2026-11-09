import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from pysr import PySRRegressor as SR
# import physo
# import torch
# from scipy.optimize import least_squares
# from pathlib import Path

import tissue_model.layer_calc as calc
import tissue_model.tissue_params as params
import analysis.error_analysis as err
import analysis.uncertainty_analysis as uncertain
import photon_sim.monte_carlo as mc
from pathlib import Path

mean_metrics_path = Path("results/mean_metrics")
mean_metrics_path.mkdir(parents = True, exist_ok = True)

error_metrics_path = Path("results/error_metrics")
error_metrics_path.mkdir(parents = True, exist_ok = True)

wavelength_range = range(1000, 1701, 10) # was 10
batches = 2
sds = ["SDS_1.7mm", "SDS_2.0mm", "SDS_2.3mm", "SDS_2.6mm", "SDS_2.9mm"]
dmu_s_dCg = -0.0029

df_dmu_a_dCg = (
    pd.read_csv("Glucose_Absorption_Data/ge2024_dmu_a_dCg.csv")
    .set_index("wavelength_nm")
    .sort_index()
)
dmu_a_dCg = df_dmu_a_dCg.loc[:, "dmu_a_dCg_mm_inv_per_mmol_L"]




plt.figure()
plt.plot(df_dmu_a_dCg)
plt.title("$d\mu_a/dC_g$ from Ge 2024", fontsize = 18)
plt.xlabel("Wavelength (nm)", fontsize = 18)
plt.xticks(fontsize = 18)
plt.ylabel("$d\mu_a/dC_g$", fontsize = 18)
plt.yticks(fontsize = 18)



batch_metrics = {
    "dA_dCg_total": [None] * batches,
    "dA_dmu_a": [None] * batches,
    "dA_dmu_s": [None] * batches,
    "dA_dCg_mu_a": [None] * batches,
    "dA_dCg_mu_s": [None] * batches,
    "mean_path_epi": [None] * batches,
    "mean_path_derm": [None] * batches,
    "mean_path_sub": [None] * batches,
    "epi_dA_dmu_s": [None] * batches,
    "derm_dA_dmu_s": [None] * batches,
    "sub_dA_dmu_s": [None] * batches,
    "beta_epi": [None] * batches,
    "beta_derm": [None] * batches,
    "beta_sub": [None] * batches,
    "power_dB": [None] * batches,
}

mean_df = dict.fromkeys(batch_metrics, None)
stderr_df = dict.fromkeys(batch_metrics, None)

sim_count = 0
for batch in range(batches):
    # Create dataframes for one batch
    dA_dCg_total = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    dA_dmu_a = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    dA_dmu_s = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    dA_dCg_mu_a = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    dA_dCg_mu_s = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )


    mean_path_epi = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    mean_path_derm = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    mean_path_sub = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )


    epi_dA_dmu_s = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    derm_dA_dmu_s = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    sub_dA_dmu_s = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )


    beta_epi = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    beta_derm = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    beta_sub = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )


    power_dB = pd.DataFrame(
        np.nan,
        index = wavelength_range,
        columns = sds
    )
    for sim_wavelength_nm in wavelength_range:
    #Following Ge 2024 procedure for computing dA/dCg
        baseline = mc.run_simulation(
            [
            params.glass_mu_a_sim,
            params.glass_mu_s_sim,
            params.glass_g_sim,
            params.glass_n_sim
            ],

            [
            params.epi_mu_a_sim[sim_wavelength_nm], 
            params.epi_mu_s_sim[sim_wavelength_nm], 
            params.epi_g_sim[sim_wavelength_nm], 
            params.epi_n_sim],

            [
            params.derm_mu_a_sim[sim_wavelength_nm], 
            params.derm_mu_s_sim[sim_wavelength_nm], 
            params.derm_g_sim[sim_wavelength_nm], 
            params.derm_n_sim],

            [
            params.sub_mu_a_sim[sim_wavelength_nm], 
            params.sub_mu_s_sim[sim_wavelength_nm], 
            params.sub_g_sim[sim_wavelength_nm], 
            params.sub_n_sim]
        )

        # #Increment mu_a by 1
        # base_dmu_a = mc.run_simulation(
        #     [
        #     params.glass_mu_a_sim,
        #     params.glass_mu_s_sim,
        #     params.glass_g_sim,
        #     params.glass_n_sim
        #     ],

        #     [
        #     params.epi_mu_a_sim[sim_wavelength_nm], 
        #     params.epi_mu_s_sim[sim_wavelength_nm], 
        #     params.epi_g_sim[sim_wavelength_nm], 
        #     params.epi_n_sim ],

        #     [
        #     params.derm_mu_a_sim[sim_wavelength_nm] + 1, 
        #     params.derm_mu_s_sim[sim_wavelength_nm], 
        #     params.derm_g_sim[sim_wavelength_nm], 
        #     params.derm_n_sim],

        #     [
        #     params.sub_mu_a_sim[sim_wavelength_nm], 
        #     params.sub_mu_s_sim[sim_wavelength_nm], 
        #     params.sub_g_sim[sim_wavelength_nm], 
        #     params.sub_n_sim]
        # )

        # #Increment mu_s by 1
        # base_dmu_s = mc.run_simulation(
        #     [
        #     params.glass_mu_a_sim,
        #     params.glass_mu_s_sim,
        #     params.glass_g_sim,
        #     params.glass_n_sim
        #     ],
            
        #     [
        #     params.epi_mu_a_sim[sim_wavelength_nm], 
        #     params.epi_mu_s_sim[sim_wavelength_nm], 
        #     params.epi_g_sim[sim_wavelength_nm], 
        #     params.epi_n_sim 
        #     ],

        #     [
        #     params.derm_mu_a_sim[sim_wavelength_nm], 
        #     params.derm_mu_s_sim[sim_wavelength_nm] + 1, 
        #     params.derm_g_sim[sim_wavelength_nm], 
        #     params.derm_n_sim
        #     ],

        #     [
        #     params.sub_mu_a_sim[sim_wavelength_nm], 
        #     params.sub_mu_s_sim[sim_wavelength_nm], 
        #     params.sub_g_sim[sim_wavelength_nm], 
        #     params.sub_n_sim
        #     ]
        # )

        dA_dmu_a.loc[sim_wavelength_nm] = (
            baseline["mean_path"]["derm_mean_path"]
            #(base_dmu_a["absorbance"] - baseline["absorbance"])
            )

        dA_dmu_s.loc[sim_wavelength_nm] = (
            baseline["dA_dmu_s"]["derm_dA_dmu_s"]
            #(base_dmu_s["absorbance"] - baseline["absorbance"])
            )

        dA_dCg_mu_a.loc[sim_wavelength_nm] = (
            baseline["mean_path"]["derm_mean_path"] * dmu_a_dCg.loc[sim_wavelength_nm]
            #(base_dmu_a["absorbance"] - baseline["absorbance"]) * dmu_a_dCg.loc[sim_wavelength_nm]
            )

        dA_dCg_mu_s.loc[sim_wavelength_nm] = (
            baseline["dA_dmu_s"]["derm_dA_dmu_s"] * dmu_s_dCg * params.derm_mu_s_sim[sim_wavelength_nm]
            #(base_dmu_s["absorbance"] - baseline["absorbance"]) * dmu_s_dCg * params.derm_mu_s_sim[sim_wavelength_nm] 

            )

        dA_dCg_total.loc[sim_wavelength_nm] = (
            dA_dCg_mu_a.loc[sim_wavelength_nm] + dA_dCg_mu_s.loc[sim_wavelength_nm]
            )

        mean_path_epi.loc[sim_wavelength_nm] = baseline["mean_path"]["epi_mean_path"]
        mean_path_derm.loc[sim_wavelength_nm] = baseline["mean_path"]["derm_mean_path"]
        mean_path_sub.loc[sim_wavelength_nm] = baseline["mean_path"]["sub_mean_path"]

        epi_dA_dmu_s.loc[sim_wavelength_nm] = baseline["dA_dmu_s"]["epi_dA_dmu_s"]
        derm_dA_dmu_s.loc[sim_wavelength_nm] = baseline["dA_dmu_s"]["derm_dA_dmu_s"]
        sub_dA_dmu_s.loc[sim_wavelength_nm] = baseline["dA_dmu_s"]["sub_dA_dmu_s"]
        
        beta_epi.loc[sim_wavelength_nm] = baseline["sensitivity"]["epi_sensitivity"]
        beta_derm.loc[sim_wavelength_nm] = baseline["sensitivity"]["derm_sensitivity"]
        beta_sub.loc[sim_wavelength_nm] = baseline["sensitivity"]["sub_sensitivity"]

        power_dB.loc[sim_wavelength_nm] = baseline["power_dB"]

    #Add wavelength sweep results as a batch entry
    batch_metrics["dA_dCg_total"][batch] = dA_dCg_total
    batch_metrics["dA_dmu_a"][batch] = dA_dmu_a
    batch_metrics["dA_dmu_s"][batch] = dA_dmu_s
    batch_metrics["dA_dCg_mu_a"][batch] = dA_dCg_mu_a
    batch_metrics["dA_dCg_mu_s"][batch] = dA_dCg_mu_s
    batch_metrics["mean_path_epi"][batch] = mean_path_epi
    batch_metrics["mean_path_derm"][batch] = mean_path_derm
    batch_metrics["mean_path_sub"][batch] = mean_path_sub
    batch_metrics["epi_dA_dmu_s"][batch] = epi_dA_dmu_s
    batch_metrics["derm_dA_dmu_s"][batch] = derm_dA_dmu_s
    batch_metrics["sub_dA_dmu_s"][batch] = sub_dA_dmu_s
    batch_metrics["beta_epi"][batch] = beta_epi
    batch_metrics["beta_derm"][batch] = beta_derm
    batch_metrics["beta_sub"][batch] = beta_sub
    batch_metrics["power_dB"][batch] = power_dB

for key, value in batch_metrics.items():
    mean_df[key] = pd.DataFrame(
        data = np.average(value, axis = 0),
        index = wavelength_range,
        columns = sds
    )
    stderr_df[key] = pd.DataFrame(
        data = np.std(value, axis = 0, ddof = 1) / np.sqrt(batches),
        index = wavelength_range,
        columns = sds
    )


fig_gluc, axes_gluc = plt.subplots(2, 3, layout = "constrained")

for ax in axes_gluc.flat:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)

# keys = ["dA_dmu_a","dA_dmu_s","dA_dCg_mu_a","dA_dCg_mu_s","dA_dCg_total"]
# for ax, key in zip(axes_gluc.flat, keys):
#     ax.errorbar(
#         x = mean_df[key].index,
#         y = mean_df[key]["SDS_1.7_mm"],
#         width = stderr_df[key],
#         color = "blue"

#     )


    

mean_df["dA_dmu_a"].plot(ax = axes_gluc[0, 0])
axes_gluc[0, 0].set_ylabel("$\delta A/\delta\mu_a$", fontsize = 18)


mean_df["dA_dmu_s"].plot(ax = axes_gluc[1, 0])
axes_gluc[1, 0].set_ylabel("$\delta A/\delta\mu_s$", fontsize = 18)

mean_df["dA_dCg_mu_a"].plot(ax = axes_gluc[0, 1])
axes_gluc[0, 1].set_ylabel("$(\delta A/\delta\mu_a)*(d\mu_a /dC_g)$", fontsize = 18)


mean_df["dA_dCg_mu_s"].plot(ax = axes_gluc[1, 1])
axes_gluc[1, 1].set_ylabel("$(\delta A/\delta\mu_s)*(d\mu_s /dC_g)$", fontsize = 18)


mean_df["dA_dCg_total"].plot(ax = axes_gluc[0, 2])
axes_gluc[0, 2].set_ylabel("$dA/dC_g$", fontsize = 18)



fig_mpath, axes_mpath = plt.subplots(1, 3, layout = "constrained")
for ax in axes_mpath:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)
    ax.set_ylabel("Distance (mm)", fontsize = 18)

mean_df["mean_path_epi"].plot(ax = axes_mpath[0])
axes_mpath[0].set_title("Mean Pathlength: Epidermis")



mean_df["mean_path_derm"].plot(ax = axes_mpath[1])
axes_mpath[1].set_title("Mean Pathlength: Dermis")


mean_df["mean_path_sub"].plot(ax = axes_mpath[2])
axes_mpath[2].set_title("Mean Pathlength: Subcutaneous")


fig_dA_dmu_s, axes_dA_dmu_s = plt.subplots(1, 3, layout = "constrained")
for ax in axes_dA_dmu_s:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)
    ax.set_ylabel("$\delta A/\delta\mu_s$", fontsize = 18)

mean_df["epi_dA_dmu_s"].plot(ax = axes_dA_dmu_s[0])
axes_dA_dmu_s[0].set_title("Epidermis")



mean_df["derm_dA_dmu_s"].plot(ax = axes_dA_dmu_s[1])
axes_dA_dmu_s[1].set_title("Dermis")


mean_df["sub_dA_dmu_s"].plot(ax = axes_dA_dmu_s[2])
axes_dA_dmu_s[2].set_title("Subcutaneous")




fig_sens, axes_sens = plt.subplots(1, 3, layout = "constrained")
for ax in axes_sens:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)
    ax.set_ylabel("Path Fraction (%)", fontsize = 18)


(100 * mean_df["beta_epi"]).plot(ax = axes_sens[0])
axes_sens[0].set_title(r"$\beta_{epi}$")


(100 * mean_df["beta_derm"]).plot(ax = axes_sens[1])
axes_sens[1].set_title(r"$\beta_{derm}$")


(100 * mean_df["beta_sub"]).plot(ax = axes_sens[2])
axes_sens[2].set_title(r"$\beta_{sub}$")


mean_df["power_dB"].plot()
plt.title("Detected Optical Power")
plt.xlabel("Wavelength (nm)", fontsize = 18)
plt.ylabel("Power (dB)", fontsize = 18)
plt.tick_params(axis = "both", labelsize = 18)
plt.legend(["SDS_1.7mm", "SDS_2.0mm", "SDS_2.3mm", "SDS_2.6mm", "SDS_2.9mm"])


for metric, df in mean_df.items():
    df.to_csv(mean_metrics_path / f"Metric_Mean_{metric}.csv", index = True, index_label = "wavelength_nm")

for metric, df in stderr_df.items():
    df.to_csv(error_metrics_path / f"Metric_Error_{metric}.csv", index = True, index_label = "wavelength_nm")


plt.show()