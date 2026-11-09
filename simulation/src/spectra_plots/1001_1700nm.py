
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tissue_model.tissue_params as params
import matplotlib.pyplot as plt
import analysis.uncertainty_analysis as uncertain
import tissue_model.exp_data as exp

# ===============================================
# Plotting tissue spectra from 1001 - 1700nm
# ===============================================

fig_mu_a, axes_mu_a = plt.subplots(1, 3, layout = "constrained")
fig_mu_a.get_layout_engine().set(h_pad = 0.1, w_pad = 0.1)
fig_mu_a.suptitle("Absorption Spectra for Three-Layer Tissue Model", fontsize = 18)

for ax in axes_mu_a:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.set_ylabel("Absorption (1/mm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)


# Simulated absorption spectra plots

axes_mu_a[0].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.epi_mu_a_sim.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  label = "Analytical Model", 
                  color = "red")


axes_mu_a[0].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.epi_mu_a_sim_Jolivot_SPT_II.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red")


axes_mu_a[0].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.epi_mu_a_sim_Jolivot_SPT_III.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red")

axes_mu_a[0].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.epi_mu_a_sim_Jolivot_SPT_IV.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red")

axes_mu_a[0].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.epi_mu_a_sim_Jolivot_SPT_V.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red")

axes_mu_a[0].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.epi_mu_a_sim_Jolivot_SPT_VI.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red")
axes_mu_a[0].text(713.5, 
                  9.11, 
                  "M=44.7%", 
                  color = "red",
                  fontsize = 10)

axes_mu_a[0].set_title(r"$\mu_a$ epidermis", fontsize = 18)


axes_mu_a[1].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.derm_mu_a_sim.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red",
                  label = "Analytical Model")

axes_mu_a[1].plot(
    uncertain.derm_mu_a_sim_df[exp.wavelength_bound_3:exp.wavelength_bound_4], 
    color = "red",
    alpha = 0.02
    )
axes_mu_a[1].set_title(r"$\mu_a$ dermis",  fontsize = 18)



axes_mu_a[2].plot(params.norm_wavelength_nm.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  params.sub_mu_a_sim.loc[exp.wavelength_bound_3:exp.wavelength_bound_4], 
                  color = "red",
                  label = "Analytical Model")

axes_mu_a[2].plot(
    uncertain.sub_mu_a_sim_df[exp.wavelength_bound_3:exp.wavelength_bound_4], 
    color = "red",
    alpha = 0.02
    )

axes_mu_a[2].set_title(r"$\mu_a$ subcutaneous", fontsize = 18)


#Experimental Absorption from Iino 2003

axes_mu_a[0].plot(exp.Iino_epidermis_mu_a.loc[exp.wavelength_bound_3:exp.wavelength_bound_4],
                  label = "Iino 2003",
                  linestyle = "--",
                  color = "black")

axes_mu_a[1].plot(exp.Iino_dermis_mu_a.loc[exp.wavelength_bound_3:exp.wavelength_bound_4],
                  label = "Iino 2003",
                  linestyle = "--",
                  color = "black")

axes_mu_a[2].plot(exp.Iino_subcutaneous_mu_a.loc[exp.wavelength_bound_3:exp.wavelength_bound_4],
                  label = "Iino 2003",
                  linestyle = "--",
                  color = "black")

axes_mu_a[0].legend(fontsize = 10)
axes_mu_a[1].legend(fontsize = 10)
axes_mu_a[2].legend(fontsize = 10)

fig_mu_s, axes_mu_s = plt.subplots(1, 3, layout = "constrained")
fig_mu_s.get_layout_engine().set(h_pad = 0.1, w_pad = 0.1)
fig_mu_s.suptitle("Reduced Scattering Spectra for Three-Layer Tissue Model", fontsize = 18)

for ax in axes_mu_s:
    ax.set_xlabel("Wavelength (nm)", fontsize = 18)
    ax.set_ylabel("Reduced Scattering (1/mm)", fontsize = 18)
    ax.tick_params(axis = "both", labelsize = 18)

#Experimental Scattering from Iino 2003

axes_mu_s[0].plot(exp.Iino_epidermis_mu_red_s.loc[exp.wavelength_bound_3:exp.wavelength_bound_4],
                  label = "Iino 2003",
                  linestyle = "--",
                  color = "black")

axes_mu_s[1].plot(exp.Iino_dermis_mu_red_s.loc[exp.wavelength_bound_3:exp.wavelength_bound_4],
                  label = "Iino 2003",
                  linestyle = "--",
                  color = "black")

axes_mu_s[2].plot(exp.Iino_subcutaneous_mu_red_s.loc[exp.wavelength_bound_3:exp.wavelength_bound_4],
                  label = "Iino 2003",
                  linestyle = "--",
                  color = "black")

axes_mu_s[0].legend(fontsize = 10)
axes_mu_s[1].legend(fontsize = 10)
axes_mu_s[2].legend(fontsize = 10)

plt.show()