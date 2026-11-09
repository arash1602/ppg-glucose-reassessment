import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tissue_model.tissue_params as params
import analysis.uncertainty_analysis as uncertain
import tissue_model.exp_data as exp


plt.rcParams["text.usetex"] = False



# Absorption Data
df_Wan_epi_mu_a = (
    pd.read_csv("Exp_Absorption/wan_epidermis_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Shimojo_epi_mu_a = (
    pd.read_csv("Exp_Absorption/shimojo_epidermis_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)


df_Simpson_derm_mu_a = (
    pd.read_csv("Exp_Absorption/simpson_dermis_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)
mu_a_derm_Simpson = df_Simpson_derm_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2, "caucasian_dermis_mu_a_mm^-1"]

df_Shimojo_derm_mu_a = (
    pd.read_csv("Exp_Absorption/shimojo_dermis_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Jonasson_derm_mu_a = (
    pd.read_csv("Exp_Absorption/jonasson_lower_dermis_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Ganesan_sub_mu_a = (
    pd.read_csv("Exp_Absorption/ganesan_subcutaneous_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)
mu_a_sub_Ganesan = df_Ganesan_sub_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2, "mu_a_mm^-1"]

df_Simpson_sub_mu_a = (
    pd.read_csv("Exp_Absorption/simpson_subcutaneous_absorption.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)
mu_a_sub_Simpson = df_Simpson_sub_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2, "caucasian_subdermis_mu_a_mm^-1"]

# Scattering Data
df_Shimojo_epi_red_mu_s = (
    pd.read_csv("Exp_Scattering/shimojo_epidermis_reduced_scattering.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Shimojo_derm_red_mu_s = (
    pd.read_csv("Exp_Scattering/shimojo_dermis_reduced_scattering.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Simpson_derm_red_mu_s = (
    pd.read_csv("Exp_Scattering/simpson_dermis_reduced_scattering.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Shimojo_sub_red_mu_s = (
    pd.read_csv("Exp_Scattering/shimojo_subcutaneous_reduced_scattering.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Ganesan_sub_red_mu_s = (
    pd.read_csv("Exp_Scattering/ganesan_subcutaneous_reduced_scattering.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

df_Simpson_sub_red_mu_s = (
    pd.read_csv("Exp_Scattering/simpson_subcutaneous_reduced_scattering.csv", comment = "#")
    .set_index("wavelength_nm")
    .sort_index()
)

# ===============================================
# Plotting tissue spectra from 600 - 1100nm
# ===============================================

# Note: This subcutaneous spectra is only valid within 429-1100 nm where reliable tissue constituent abosroption data is availible
fig_mu_a, axes_mu_a = plt.subplots(1, 3, layout = "constrained")
fig_mu_a.get_layout_engine().set(h_pad = 0.1, w_pad = 0.1)
fig_mu_a.suptitle("Absorption Spectra for Three-Layer Tissue Model", fontsize = 18)

for ax in axes_mu_a:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.set_ylabel("Absorption (1/mm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)


# Simulated absorption spectra plots
axes_mu_a[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_a_sim[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  label = "Analytical Model", 
                  color = "red")
axes_mu_a[0].text(600, 
                  0.2, 
                  "M=2.5%", 
                  color = "red",
                  fontsize = 10)

axes_mu_a[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_a_sim_Jolivot_SPT_II[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red")


axes_mu_a[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_a_sim_Jolivot_SPT_III[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red")

axes_mu_a[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_a_sim_Jolivot_SPT_IV[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red")

axes_mu_a[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_a_sim_Jolivot_SPT_V[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red")

axes_mu_a[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_a_sim_Jolivot_SPT_VI[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red")
axes_mu_a[0].text(713.5, 
                  9.11, 
                  "M=44.7%", 
                  color = "red",
                  fontsize = 10)

axes_mu_a[0].set_title(r"$\mu_a$ epidermis", fontsize = 18)


axes_mu_a[1].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.derm_mu_a_sim[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red",
                  label = "Analytical Model")

axes_mu_a[1].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:850], 
                  params.Jonasson_derm_mu_a_sim[exp.wavelength_bound_1:850], 
                  color = "red",
                  )
axes_mu_a[1].text(850, 
                  0, 
                  "Simplified dermis\nabsorption model\n", 
                  color = "red",
                  fontsize = 10)

axes_mu_a[1].plot(
    uncertain.derm_mu_a_sim_df[exp.wavelength_bound_1:exp.wavelength_bound_2], 
    color = "red",
    alpha = 0.02
    )
# axes_mu_a[1].plot(
#     uncertain.Jonasson_derm_mu_a_sim_df[exp.wavelength_bound_1:exp.wavelength_bound_2], 
#     color = "red",
#     alpha = 0.02
#     )
axes_mu_a[1].set_title(r"$\mu_a$ dermis",  fontsize = 18)



axes_mu_a[2].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.sub_mu_a_sim[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "red",
                  label = "Analytical Model")

axes_mu_a[2].plot(
    uncertain.sub_mu_a_sim_df[exp.wavelength_bound_1:exp.wavelength_bound_2], 
    color = "red",
    alpha = 0.02
    )

axes_mu_a[2].set_title(r"$\mu_a$ subcutaneous", fontsize = 18)


# Experimental absorption spectra plots
axes_mu_a[0].plot(df_Wan_epi_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "orange", 
                  label = "Wan 1981: Dark Skin",
                  linestyle = "--")
axes_mu_a[0].plot(df_Shimojo_epi_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "black", 
                  label = "Shimojo 2020: Asian Skin",
                  linestyle = "--")

axes_mu_a[1].plot(df_Simpson_derm_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "green",
                  label = "Simpson 1998",
                  linestyle = "--")

axes_mu_a[1].plot(df_Jonasson_derm_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "purple",
                  label = "Jonasson 2023",
                  linestyle = "--")

# axes_mu_a[1].plot(df_Shimojo_derm_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
#                   color = "black",
#                   linestyle = "--")

axes_mu_a[2].plot(df_Ganesan_sub_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "black", 
                  label = "Ganesan 2016", 
                  linestyle = "--")
axes_mu_a[2].plot(df_Simpson_sub_mu_a.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "green", 
                  label = "Simpson 1998", 
                  linestyle = "--")
axes_mu_a[2].scatter(x = 650, 
                     y = 0.006,
                     color = "pink",
                     label = "Weber 2009",
                     s = 100)

axes_mu_a[0].legend(fontsize = 10)
axes_mu_a[1].legend(fontsize = 10)
axes_mu_a[2].legend(fontsize = 10)

fig_mu_s, axes_mu_s = plt.subplots(1, 3, layout = "constrained")
fig_mu_s.get_layout_engine().set(h_pad = 0.1, w_pad = 0.1)
fig_mu_s.suptitle("Reduced Scattering Spectra for Three-Layer Tissue Model", fontsize = 18)

for ax in axes_mu_s.flat:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.set_ylabel("Reduced Scattering (1/mm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)

# Simulated reduced scattering coefficients
axes_mu_s[0].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.epi_mu_s_red_sim[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  label = "Jacques 2013 Model", 
                  color = "red")
axes_mu_s[0].set_title(r"$\mu_s^'$ epidermis", fontsize = 18)



axes_mu_s[1].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.derm_mu_s_red_sim[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  label = "Jacques 2013 Model",
                  color = "red")
axes_mu_s[1].set_title(r"$\mu^'_s$ dermis", fontsize = 18)

axes_mu_s[2].plot(params.norm_wavelength_nm[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  params.sub_mu_s_red_sim[exp.wavelength_bound_1:exp.wavelength_bound_2],
                  label = "Jacques 2013 Model", 
                  color = "red")
axes_mu_s[2].set_title(r"$\mu^'_s$ subcutaneous tissue", fontsize = 18)

# Experimental reduced scattering coefficients
axes_mu_s[0].plot(df_Shimojo_epi_red_mu_s.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "cyan", 
                  label = "Shimojo 2020", 
                  linestyle = "--")

axes_mu_s[1].plot(df_Shimojo_derm_red_mu_s.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "cyan",
                  label = "Shimojo 2020", 
                  linestyle = "--")
axes_mu_s[1].plot(df_Simpson_derm_red_mu_s.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "green", 
                  label = "Simpson 1998", 
                  linestyle = "--")

axes_mu_s[2].plot(df_Shimojo_sub_red_mu_s.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "cyan",
                  label = "Shimojo 2020", 
                  linestyle = "--")
axes_mu_s[2].plot(df_Simpson_sub_red_mu_s.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "green",
                  label = "Simpson 1998",
                  linestyle = "--")
axes_mu_s[2].plot(df_Ganesan_sub_red_mu_s.loc[exp.wavelength_bound_1:exp.wavelength_bound_2], 
                  color = "purple", 
                  label = "Ganesan 2016", 
                  linestyle = "--")

axes_mu_s[0].legend(fontsize = 10)
axes_mu_s[1].legend(fontsize = 10)
axes_mu_s[2].legend(fontsize = 10)

# 930 nm lipid absorption feature
axes_mu_a[2].vlines(
    x = 930,
    ymin = 0.01,
    ymax = 0.020,
    linestyle=":",
    linewidth = 1.5,
    color = "black"
)

axes_mu_a[2].text(
    900,
    0.020,
    "930 nm\nC-H overtone",
    ha = "center",
    va = "bottom",
    fontsize = 10
)

# 970 nm water absorption feature
axes_mu_a[1].vlines(
    x = 970,
    ymin = 0.027,
    ymax = 0.05,
    linestyle=":",
    linewidth = 1.5,
    color = "black"
)

axes_mu_a[1].text(
    970,
    0.05,
    "970 nm\nH$_2$O O-H \novertone",
    ha = "center",
    va = "bottom",
    fontsize = 10
)

axes_mu_a[2].vlines(
    x = 970,
    ymin = 0.01,
    ymax = 0.025,
    linestyle=":",
    linewidth = 1.5,
    color = "black"
)

axes_mu_a[2].text(
    970,
    0.025,
    "970 nm\nH$_2$O O-H \novertone",
    ha = "center",
    va = "bottom",
    fontsize = 10
)

# Deoxy Hb absorption
axes_mu_a[1].vlines(
    x = 760,
    ymin = 0.002,
    ymax = 0.022,
    linestyle=":",
    linewidth=1.5,
    color="black"
)

axes_mu_a[1].text(
    760,
    0.022,
    "760 nm\nHb Electron Transfer",
    ha = "center",
    va = "bottom",
    fontsize = 10
)

axes_mu_a[2].vlines(
    x = 760,
    ymin = 0.0025,
    ymax = 0.014,
    linestyle=":",
    linewidth=1.5,
    color="black"
)

axes_mu_a[2].text(
    760,
    0.014,
    "760 nm\nHb Electron Transfer",
    ha = "center",
    va = "bottom",
    fontsize = 10
)


plt.show()