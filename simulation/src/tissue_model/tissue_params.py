import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tissue_model.layer_calc as calc


norm_wavelength_nm_np = np.arange(1, 2001, 1)
norm_wavelength_nm = pd.Series(data = norm_wavelength_nm_np, index = norm_wavelength_nm_np)

baseline_epi = (8.53 * np.exp(-(norm_wavelength_nm - 154)/66.2) + 0.0244)
# Dermis baseline (Saidi / Jacques: 7.84e8 * lambda^-3.255 cm^-1) already converted to mm^-1 (7.84e7).
# Confirmed by Milan 2026-10-30. Only affects wavelengths < 1000 nm.
baseline_derm = 7.84 * 10**7 * norm_wavelength_nm ** -3.255
baseline_sub = (1.682 * np.exp(-(norm_wavelength_nm - 400)/80.5) + 0.0244)


derm_params = [
    "blood",
    "oxygen_saturation",
    "water",
]

sub_params = [
    "blood",
    "oxygen_saturation",
    "water",
    "fat",
]


epi_Jacques = {
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20, #from Meglinski 2002, was 0
    "fat" : 0,
    "melanosome" : 0.025,
}
epi_Jacques["baseline"] = (
    1 - 
    epi_Jacques["blood"] - 
    epi_Jacques["water"] - 
    epi_Jacques["fat"] - 
    epi_Jacques["melanosome"]
    )


epi_Jolivot_low_mel ={
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20,
    "fat" : 0,
    "melanosome": 0.013
}
epi_Jolivot_low_mel["baseline"] = (
    1 - 
    epi_Jolivot_low_mel["blood"] - 
    epi_Jolivot_low_mel["water"] - 
    epi_Jolivot_low_mel["fat"] - 
    epi_Jolivot_low_mel["melanosome"]
    )


epi_Jolivot_SPT_II ={
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20,
    "fat" : 0,
    "melanosome": 0.101
}
epi_Jolivot_SPT_II["baseline"] = (
    1 - 
    epi_Jolivot_SPT_II["blood"] - 
    epi_Jolivot_SPT_II["water"] - 
    epi_Jolivot_SPT_II["fat"] - 
    epi_Jolivot_SPT_II["melanosome"]
    )


epi_Jolivot_SPT_III ={
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20,
    "fat" : 0,
    "melanosome": 0.187
}
epi_Jolivot_SPT_III["baseline"] = (
    1 - 
    epi_Jolivot_SPT_III["blood"] - 
    epi_Jolivot_SPT_III["water"] - 
    epi_Jolivot_SPT_III["fat"] - 
    epi_Jolivot_SPT_III["melanosome"]
    )


epi_Jolivot_SPT_IV ={
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20,
    "fat" : 0,
    "melanosome": 0.265
}
epi_Jolivot_SPT_IV["baseline"] = (
    1 - 
    epi_Jolivot_SPT_IV["blood"] - 
    epi_Jolivot_SPT_IV["water"] - 
    epi_Jolivot_SPT_IV["fat"] - 
    epi_Jolivot_SPT_IV["melanosome"]
    )


epi_Jolivot_SPT_V ={
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20,
    "fat" : 0,
    "melanosome": 0.358
}
epi_Jolivot_SPT_V["baseline"] = (
    1 - 
    epi_Jolivot_SPT_V["blood"] - 
    epi_Jolivot_SPT_V["water"] - 
    epi_Jolivot_SPT_V["fat"] - 
    epi_Jolivot_SPT_V["melanosome"]
    )


epi_Jolivot_SPT_VI = {
    "blood" : 0,
    "oxygen_saturation" : 0,
    "water" : 0.20,
    "fat" : 0,
    "melanosome": 0.447
}
epi_Jolivot_SPT_VI["baseline"] = (
    1 - 
    epi_Jolivot_SPT_VI["blood"] - 
    epi_Jolivot_SPT_VI["water"] - 
    epi_Jolivot_SPT_VI["fat"] - 
    epi_Jolivot_SPT_VI["melanosome"]
    )


derm_Jacques = {
    "blood" : 0.0020,  
    "oxygen_saturation" : 0.39,
    "water" : 0.65,
    "fat" : 0,
    "melanosome" : 0,
}
derm_Jacques["baseline"] = (
    1 - 
    derm_Jacques["blood"] - 
    derm_Jacques["water"] - 
    derm_Jacques["fat"] - 
    derm_Jacques["melanosome"]
    )


lower_derm_Jonasson = {
    "blood" : 0.0075,  
    "oxygen_saturation" : 0.50, 
    "water" : 0.0, # Jonasson neglects water
    "fat" : 0,
    "melanosome" : 0,
}
lower_derm_Jonasson["baseline"] = (
    1 - 
    lower_derm_Jonasson["blood"] - 
    lower_derm_Jonasson["water"] - 
    lower_derm_Jonasson["fat"] - 
    lower_derm_Jonasson["melanosome"]
    )


sub_Jacques = {
    "blood" : 0.0054,
    "oxygen_saturation" : 0.76,
    "water" : 0.11,
    "fat" : 0.69,
    "melanosome" : 0,
}
sub_Jacques["baseline"] = (
    1 - 
    sub_Jacques["blood"] - 
    sub_Jacques["water"] - 
    sub_Jacques["fat"] - 
    sub_Jacques["melanosome"]
    )


sub_Ganesan = {
    "blood" : 0.00634,
    "oxygen_saturation" : 0.60,
    "water" : 0.2184,
    "fat" : 0.6433,
    "melanosome" : 0,
}
sub_Ganesan["baseline"] = (
    1 - 
    sub_Ganesan["blood"] - 
    sub_Ganesan["water"] - 
    sub_Ganesan["fat"] - 
    sub_Ganesan["melanosome"]
    )


# Upper bound are provided by Jonasson 2023 (Blood)
derm_upper_bound = {
    "blood" : 0.015,
    "oxygen_saturation" : 0.65,    
    "water" : 0.75,

}

# Lower bound are provided by Jacques 2013
derm_lower_bound = {
    "blood" : 0.002,
    "oxygen_saturation" : 0.39,    
    "water" : 0.30, # was 0.65

}

# Upper bound are provided by Ganesan 2016
sub_upper_bound = {
    "blood" : 0.00813,
    "oxygen_saturation" : 0.67,    
    "water" : 0.304,
    "fat" : 0.702,

}

sub_lower_bound = {
    "blood" : 0.00398,
    "oxygen_saturation" : 0.493,    
    "water" : 0.00, # was 0.145
    "fat" : 0.599, 

}

# ============== MC simulation parameters ==============

glass_thickness_mm = 0.2
epi_thickness_mm = 0.1     
derm_thickness_mm = 1      
sub_thickness_mm = 10      
vox_size_mm = 0.1      

glass_thickness_vox = int(glass_thickness_mm / vox_size_mm)
epi_thickness_vox = int(epi_thickness_mm / vox_size_mm)
derm_thickness_vox = int(derm_thickness_mm / vox_size_mm)
sub_thickness_vox = int(sub_thickness_mm / vox_size_mm)

bot_glass_depth_vox = glass_thickness_vox
bot_epi_depth_vox = epi_thickness_vox + glass_thickness_vox
bot_derm_depth_vox = epi_thickness_vox + derm_thickness_vox + glass_thickness_vox
bot_sub_depth_vox = epi_thickness_vox + derm_thickness_vox + sub_thickness_vox + glass_thickness_vox

vox_x = 300
vox_y = 300
vox_z = bot_sub_depth_vox

SDS_1_mm = 1.7
SDS_2_mm = 2.0
SDS_3_mm = 2.3
SDS_4_mm = 2.6
SDS_5_mm = 2.9

SDS_1_vox = round(SDS_1_mm / vox_size_mm)
SDS_2_vox = round(SDS_2_mm / vox_size_mm)
SDS_3_vox = round(SDS_3_mm / vox_size_mm)
SDS_4_vox = round(SDS_4_mm / vox_size_mm)
SDS_5_vox = round(SDS_5_mm / vox_size_mm)


src_x_vox = vox_x / 4
src_y_vox = vox_y / 2
src_z_vox = 0

src_x_mm = src_x_vox * vox_size_mm
src_y_mm = src_y_vox * vox_size_mm
src_z_mm = src_z_vox * vox_size_mm

# Create one large circular detector which will be divided into 5 concentric ring detectors
det_x_vox = src_x_vox
det_y_vox = src_y_vox
det_z_vox = src_z_vox
det_r_vox = 31

ring_1_inner_rad = SDS_1_mm - 0.1
ring_1_outer_rad = SDS_1_mm + 0.1

ring_2_inner_rad = SDS_2_mm - 0.1
ring_2_outer_rad = SDS_2_mm + 0.1

ring_3_inner_rad = SDS_3_mm - 0.1
ring_3_outer_rad = SDS_3_mm + 0.1

ring_4_inner_rad = SDS_4_mm - 0.1
ring_4_outer_rad = SDS_4_mm + 0.1

ring_5_inner_rad = SDS_5_mm - 0.1
ring_5_outer_rad = SDS_5_mm + 0.1

glass_mu_a_sim, glass_mu_s_sim, glass_g_sim, glass_n_sim = [0, 0, 1, 1.4]

epi_mu_a_sim, epi_mu_s_sim, epi_g_sim, epi_n_sim, epi_mu_s_red_sim = calc.epi(
    epi_Jacques["blood"], 
    epi_Jacques["oxygen_saturation"], 
    epi_Jacques["water"], 
    epi_Jacques["fat"], 
    epi_Jacques["melanosome"],
    baseline_epi,
    )


epi_mu_a_sim_Jolivot_SPT_II = calc.epi(
    epi_Jolivot_SPT_II["blood"], 
    epi_Jolivot_SPT_II["oxygen_saturation"], 
    epi_Jolivot_SPT_II["water"], 
    epi_Jolivot_SPT_II["fat"], 
    epi_Jolivot_SPT_II["melanosome"],
    baseline_epi,
    )[0]

epi_mu_a_sim_Jolivot_SPT_III = calc.epi(
    epi_Jolivot_SPT_III["blood"], 
    epi_Jolivot_SPT_III["oxygen_saturation"], 
    epi_Jolivot_SPT_III["water"], 
    epi_Jolivot_SPT_III["fat"], 
    epi_Jolivot_SPT_III["melanosome"],
    baseline_epi,
    )[0]

epi_mu_a_sim_Jolivot_SPT_IV = calc.epi(
    epi_Jolivot_SPT_IV["blood"], 
    epi_Jolivot_SPT_IV["oxygen_saturation"], 
    epi_Jolivot_SPT_IV["water"], 
    epi_Jolivot_SPT_IV["fat"], 
    epi_Jolivot_SPT_IV["melanosome"],
    baseline_epi
    )[0]

epi_mu_a_sim_Jolivot_SPT_V = calc.epi(
    epi_Jolivot_SPT_V["blood"], 
    epi_Jolivot_SPT_V["oxygen_saturation"], 
    epi_Jolivot_SPT_V["water"], 
    epi_Jolivot_SPT_V["fat"], 
    epi_Jolivot_SPT_V["melanosome"],
    baseline_epi
    )[0]

epi_mu_a_sim_Jolivot_SPT_VI = calc.epi(
    epi_Jolivot_SPT_VI["blood"], 
    epi_Jolivot_SPT_VI["oxygen_saturation"], 
    epi_Jolivot_SPT_VI["water"], 
    epi_Jolivot_SPT_VI["fat"], 
    epi_Jolivot_SPT_VI["melanosome"],
    baseline_epi
    )[0]

derm_mu_a_sim, derm_mu_s_sim, derm_g_sim, derm_n_sim, derm_mu_s_red_sim = calc.derm(
    derm_Jacques["blood"], 
    derm_Jacques["oxygen_saturation"], 
    derm_Jacques["water"], 
    derm_Jacques["fat"], 
    derm_Jacques["melanosome"],
    baseline_derm,
    )

Jonasson_derm_mu_a_sim = calc.derm(
    lower_derm_Jonasson["blood"], 
    lower_derm_Jonasson["oxygen_saturation"], 
    lower_derm_Jonasson["water"], 
    lower_derm_Jonasson["fat"], 
    lower_derm_Jonasson["melanosome"],
    0, # No baseline in Jonasson 2023
    )[0]

sub_mu_a_sim, sub_mu_s_sim, sub_g_sim, sub_n_sim, sub_mu_s_red_sim = calc.sub(
    sub_Ganesan["blood"], 
    sub_Ganesan["oxygen_saturation"],
    sub_Ganesan["water"],
    sub_Ganesan["fat"],
    sub_Ganesan["melanosome"],
    0, # No baseline applied by Lapitan 2024
    )

