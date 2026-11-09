"""Alternative (continuous) tissue model, 600-1700 nm, for the tissue-model sensitivity analysis.

Based on M. Chhetri's updated three-layer model (Glucose_Sensor_Ranking, Nov 2026), with corrections:
  * absorption of every layer from composition over the whole range (no change of data source at 1000 nm):
      mu_a = B*S*mu_a,HbO2 + B*(1-S)*mu_a,Hb + W*mu_a,water + F*mu_a,fat + M*mu_a,mel + R*baseline
    blood: Bosschaart et al. 2014; water: Segelstein 1981 (public); fat: van Veen 2004 / Anderson 2006;
    melanosome: Jacques; baselines as in Jacques 2013 / Saidi 1992 (subcutaneous: none).
    Compositions: epidermis (Jacques), dermis (Jacques), subcutaneous fat (Ganesan et al. 2016).
  * reduced scattering of every layer from Salomatina et al. 2006 (measured, 600-1330 nm). Between 1330
    and 1700 nm the measured curves show a peak at the 1450 nm water band, a known absorption-scattering
    cross-talk artefact of such inversions, and the data end near 1600 nm; there a power law
    a*(lambda/500 nm)^-b fitted to the same layer's 600-1330 nm data is used, scaled to join at 1330 nm.
  * anisotropy g = 0.62 + 0.29e-3*lambda (van Gemert 1989) up to 1000 nm and held at its 1000-nm value
    (0.91) above, so that mu_s = mu_s'/(1-g) is continuous (the original model switched from 0.91 to 0.90).
  * refractive index n = (1-W)*1.5 + W*1.33 (Jacques 2013).
Geometry (layer thicknesses, glass window, detector rings) is unchanged and comes from tissue_model.tissue_params.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]            # .../MCeXtreme_Sim (simulation/)
WL = np.arange(600, 1701)


def _series(csv, col=None, scale=1.0):
    d = pd.read_csv(ROOT / csv, comment="#")
    x = d.iloc[:, 0].to_numpy(float); y = (d[col] if col else d.iloc[:, 1]).to_numpy(float) * scale
    o = np.argsort(x)
    return pd.Series(np.interp(WL, x[o], y[o]), index=WL)


water = _series("Water_Absorption_Data_Public/water_segelstein1981_400_2000nm.csv", "mu_a_water_mm-1")
hbo2 = _series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_oxy_mm-1")
hb = _series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_deoxy_mm-1")
fat = _series("Fat_Absorption_Data/fat_429_1700nm.csv", "mu_a")
mel = pd.Series(1.7e12 * WL.astype(float) ** -3.48 / 10, index=WL)          # mm^-1 (Jacques)

lam = WL.astype(float)
baseline = {"epi": pd.Series(8.53 * np.exp(-(lam - 154) / 66.2) + 0.0244, index=WL),
            "derm": pd.Series(7.84e7 * lam ** -3.255, index=WL),
            "sub": pd.Series(0.0, index=WL)}
COMP = {"epi": dict(B=0.0, S=0.0, W=0.20, F=0.0, M=0.025),
        "derm": dict(B=0.0020, S=0.39, W=0.65, F=0.0, M=0.0),
        "sub": dict(B=0.00634, S=0.60, W=0.2184, F=0.6433, M=0.0)}


def _mua(layer):
    c = COMP[layer]; R = max(0.0, 1 - c["B"] - c["W"] - c["F"] - c["M"])
    return (c["B"] * c["S"] * hbo2 + c["B"] * (1 - c["S"]) * hb + c["W"] * water + c["F"] * fat
            + c["M"] * mel + R * baseline[layer])


JOIN = 1330
FIT_INFO = {}


def _musp(layer, file):
    raw = _series(f"Exp_Scattering/{file}")
    d = pd.read_csv(ROOT / "Exp_Scattering" / file, comment="#")
    x = d.iloc[:, 0].to_numpy(float); y = d.iloc[:, 1].to_numpy(float)
    keep = (x >= 600) & (x <= JOIN)
    slope, icpt = np.polyfit(np.log(x[keep] / 500.0), np.log(y[keep]), 1)
    fit = pd.Series(np.exp(icpt) * (lam / 500.0) ** slope, index=WL)
    fit *= raw.loc[JOIN] / fit.loc[JOIN]                       # continuous at the join
    out = raw.copy(); out.loc[JOIN:] = fit.loc[JOIN:]
    FIT_INFO[layer] = {"a_mm-1": float(np.exp(icpt)), "b": float(-slope),
                       "raw_vs_fit_1450_%": float(100 * (raw.loc[1450] / fit.loc[1450] - 1))}
    return out


g = pd.Series(np.where(lam <= 1000, 0.62 + 0.29e-3 * lam, 0.62 + 0.29e-3 * 1000), index=WL)
musp = {"epi": _musp("epi", "salomatina_2006_epidermis_reduced_scattering.csv"),
        "derm": _musp("derm", "salomatina_2006_dermis_reduced_scattering.csv"),
        "sub": _musp("sub", "salomatina_2006_subcutaneous_fat_reduced_scattering.csv")}
n_index = {k: (1 - c["W"]) * 1.5 + c["W"] * 1.33 for k, c in COMP.items()}

# same names as tissue_model.tissue_params, so the sweep code can use either model
for _k in ["epi", "derm", "sub"]:
    globals()[f"{_k}_mu_a_sim"] = _mua(_k)
    globals()[f"{_k}_mu_s_sim"] = musp[_k] / (1 - g)
    globals()[f"{_k}_g_sim"] = g
    globals()[f"{_k}_n_sim"] = n_index[_k]
glass_mu_a_sim, glass_mu_s_sim, glass_g_sim, glass_n_sim = 0, 0, 1, 1.4
SUB_WATER_REL = COMP["sub"]["W"] / COMP["derm"]["W"]     # interstitial-fluid weighting of glucose in fat (variant)

if __name__ == "__main__":
    for w in [600, 660, 940, 999, 1000, 1001, 1300, 1450, 1600, 1700]:
        print(w, "derm mua %.4f  mus %.2f  mus' %.3f  g %.4f | sub mua %.4f mus' %.3f" % (
            derm_mu_a_sim[w], derm_mu_s_sim[w], musp["derm"][w], g[w], sub_mu_a_sim[w], musp["sub"][w]))
    print(FIT_INFO)
