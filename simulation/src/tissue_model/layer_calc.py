import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tissue_model.exp_data as exp



# ==========================================================
# Using Jacques 2013 generic tissue model
# Per layer: (B * S * mu_a_HbO2) + (B * (1-S) * mu_a_Hb) + (W * mu_a_water) + (F * mu_a_fat) + (M * mu_a_melanosome) + baseline
# where B, S, W, F, and M represent blood, oxygen, water, fat, and melanosome compostion of tissue respectively
# ==========================================================


norm_wavelength_nm_np = np.arange(1, 2001, 1)
norm_wavelength_nm = pd.Series(data = norm_wavelength_nm_np, index = norm_wavelength_nm_np)


# Jacques 2013
n_dry = 1.5
n_water = 1.33

# van Gemert 1989, valid for 200nm - 1000nm
g_200_1000 = 0.62 + (0.29 * norm_wavelength_nm.loc[200: 1000] * 10 ** -3)
g_1000_1700 = 0.9

g_200_1000_ser = pd.Series(data = g_200_1000, index = norm_wavelength_nm.loc[200: 1000])
g_1000_1700_ser = pd.Series(data = g_1000_1700, index = norm_wavelength_nm.loc[1001: 1700])

g = pd.concat([g_200_1000_ser, g_1000_1700_ser])

def epi(
    B: float,
    S: float,
    W: float,
    F: float,
    M: float,
    baseline: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, float, pd.Series]:
    R = 1 - (B + W + F + M) # need R calculated here for uncertainty analysis
    
    if(R < 0):
        R = 0
    mu_a_1 = ( 
    (B * S * exp.Bosschaart_2014_interp_oxy_ser)
    + (B * (1-S) * exp.Bosschaart_2014_interp_deoxy_ser)
    + (W * exp.Fewell_2019_interp_ser) 
    + (F * exp.Veen_Anderson_interp_ser) 
    + (M * exp.Jacques_1998_interp_ser)
    + (R * baseline) # baseline tissue absorption
    ).loc[1:999]
    mu_a_2 = exp.Iino_epidermis_mu_a.loc[1000:1700]

    mu_a = pd.concat([mu_a_1, mu_a_2])

    # epidermis scattering
    mu_red_s_1 = 6.87 * (norm_wavelength_nm.loc[1: 999] / 500) ** -1.161
    mu_red_s_2 = exp.Iino_epidermis_mu_red_s.loc[1000:1700]
    mu_red_s = pd.concat([mu_red_s_1, mu_red_s_2])

    mu_s = mu_red_s / (1 - g)

    # Jacques 2013 refractive index
    #n = ((1 - W) * n_dry) + W * n_water
    n = 1.37
    
    return mu_a, mu_s, g, n, mu_red_s

def derm(
    B: float,
    S: float,
    W: float,
    F: float,
    M: float,
    baseline: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, float, pd.Series]:
    R = 1 - (B + W + F + M) # need R calculated here for uncertainty analysis
    

    if(R < 0):
        R = 0
    mu_a_1 = ( 
    (B * S * exp.Bosschaart_2014_interp_oxy_ser)
    + (B * (1-S) * exp.Bosschaart_2014_interp_deoxy_ser)
    + (W * exp.Fewell_2019_interp_ser) 
    + (F * exp.Veen_Anderson_interp_ser) 
    + (M * exp.Jacques_1998_interp_ser)
    + (R * baseline) # baseline tissue absorption
    ).loc[1:999]
    mu_a_2 = exp.Iino_dermis_mu_a.loc[1000:1700]

    mu_a = pd.concat([mu_a_1, mu_a_2])

    # dermis scattering
    mu_red_s_1 = 4.53 * (norm_wavelength_nm.loc[1: 999] / 500) ** -1.292
    mu_red_s_2 = exp.Iino_dermis_mu_red_s.loc[1000:1700]
    mu_red_s = pd.concat([mu_red_s_1, mu_red_s_2])

    mu_s = mu_red_s / (1 - g)


    # Jacques 2013 refractive index
    #n = ((1 - W) * n_dry) + W * n_water
    n = 1.37
    
    return mu_a, mu_s, g, n, mu_red_s


def sub(
    B: float,
    S: float,
    W: float,
    F: float,
    M: float,
    baseline: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, float, pd.Series]:
    R = 1 - (B + W + F + M) # need R calculated here for uncertainty analysis
    
    
    if(R < 0):
        R = 0
    mu_a_1 = ( 
    (B * S * exp.Bosschaart_2014_interp_oxy_ser)
    + (B * (1-S) * exp.Bosschaart_2014_interp_deoxy_ser)
    + (W * exp.Fewell_2019_interp_ser) 
    + (F * exp.Veen_Anderson_interp_ser) 
    + (M * exp.Jacques_1998_interp_ser)
    + (R * baseline) # baseline tissue absorption
    ).loc[1:999]
    mu_a_2 = exp.Iino_subcutaneous_mu_a.loc[1000:1700]

    mu_a = pd.concat([mu_a_1, mu_a_2])


    
    # subcutaneous scattering
    mu_red_s_1 = 1.84 * (norm_wavelength_nm.loc[1: 999] / 500) ** -0.672
    mu_red_s_2 = exp.Iino_subcutaneous_mu_red_s.loc[1000:1700]
    mu_red_s = pd.concat([mu_red_s_1, mu_red_s_2])

    mu_s = mu_red_s / (1 - g)


    # Jacques 2013 refractive index
    #n = ((1 - W) * n_dry) + W * n_water
    n = 1.37
    
    return mu_a, mu_s, g, n, mu_red_s
