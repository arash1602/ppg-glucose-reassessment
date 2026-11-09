"""Tissue-model sensitivity analysis: the key optical results under the paper's tissue model (A, results_extended)
and the alternative continuous model (B, results_tissue2), plus two variants of B computed from its saved
path lengths (no new simulation):
  B+fat : glucose also acts in the subcutaneous layer, weighted by its water fraction relative to the dermis
          (glucose is in the interstitial fluid; 0.2184 / 0.65 = 0.34); and at full strength (Ge et al.'s assumption)
  B+abs<1000 : a small glucose absorption below 1000 nm, linearly interpolated between the molar absorptivities at 660 and
               940 nm reported by Haque et al. 2021 (0.0002 and 0.001 L/mol/cm; dmu_a/dC = ln(10)*eps*1e-4 per mmol/L)
Confounders, noise and the net-analyte-signal method are exactly those of physio_confounders.py.
Run from the simulation folder (after physio_confounders.py on both result folders):
  python src/analysis/compare_tissue_models.py results_extended results_tissue2
Output: <B>/physio_analysis/tissue_model_comparison.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physio_confounders as PC

P, MG = PC.P, PC.MGDL_PER_MMOL


def load(res):
    res = Path(res)
    m = {k: pd.read_csv(res / "mean_metrics" / f"Metric_Mean_{k}.csv", index_col="wavelength_nm")
         for k in ["mean_path_derm", "mean_path_sub", "derm_dA_dmu_s", "sub_dA_dmu_s", "power_dB"]}
    wl = m["mean_path_derm"].index.to_numpy()
    inp = pd.read_csv(res / "inputs_per_wavelength.csv", index_col="wavelength_nm")
    inp = inp[~inp.index.duplicated()].reindex(wl)
    sp = {"w": PC.load_series("Water_Absorption_Data_Public/water_segelstein1981_400_2000nm.csv", "mu_a_water_mm-1", wl),
          "o": PC.load_series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_oxy_mm-1", wl),
          "d": PC.load_series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_deoxy_mm-1", wl)}
    return m, wl, inp, sp


def evaluate(res, fat_weight=0.0, dmua_below_1000=None):
    m, wl, inp, sp = load(res)
    L, dAs = m["mean_path_derm"], m["derm_dA_dmu_s"]
    dmua = inp["dmu_a_dCg_mm-1_per_mmol"].copy()
    if dmua_below_1000 is not None:
        dmua[wl < 1000] = dmua_below_1000.reindex(wl[wl < 1000]).values
    mus = inp["derm_mu_s_mm-1"]
    S_abs = L.mul(dmua, axis=0)
    S = S_abs + dAs.mul(mus * P["dmus_dCg_rel"], axis=0)
    if fat_weight:
        mus_sub = inp["sub_mu_s_mm-1"]
        S_fat_abs = m["mean_path_sub"].mul(dmua, axis=0) * fat_weight
        S_abs = S_abs + S_fat_abs
        S = S + S_fat_abs + m["sub_dA_dmu_s"].mul(mus_sub * P["dmus_dCg_rel"], axis=0) * fat_weight
    mua_bd = P["derm_SO2"] * sp["o"] + (1 - P["derm_SO2"]) * sp["d"]
    dA = {"water": L.mul(sp["w"] * P["d_water_abs"], axis=0),
          "blood": L.mul((mua_bd - P["isf_water_frac"] * sp["w"]) * P["d_blood_abs"], axis=0),
          "scat": dAs.mul(mus * P["d_mus_rel"], axis=0), "drift": L * 0 + P["drift_rel"]}
    if fat_weight:
        dA["fat water"] = m["mean_path_sub"].mul(sp["w"] * P["d_water_abs"], axis=0)
        dA["fat scattering"] = m["sub_dA_dmu_s"].mul(inp["sub_mu_s_mm-1"] * P["d_mus_rel"], axis=0)
    rss = lambda Sx: np.sqrt(sum(((v.abs() / Sx.abs()) * MG) ** 2 for v in dA.values()))
    R, Ra = rss(S), rss(S_abs)
    best = R.stack().idxmin()
    out = {"best single channel (mg/dL)": float(R.stack().min()), "best wavelength (nm)": int(best[0]), "best SDS": best[1],
           "absorption only, best single (mg/dL)": float(Ra.stack().min())}
    for w in (660, 940):
        out[f"{w} nm, steady channel (mg/dL)"] = float(R.loc[w].min()) if w in R.index else np.nan
    mua_art = P["art_SO2"] * sp["o"] + (1 - P["art_SO2"]) * sp["d"]
    contrast = mua_art - P["isf_water_frac"] * sp["w"]
    rel = (dmua / contrast).abs()
    ac = np.sqrt((P["eps_AC_rel"] / rel) ** 2 + ((sp["o"] - sp["d"]) * P["d_SO2"] / contrast / rel) ** 2
                 + (sp["w"] * P["d_blood_water_abs"] / contrast / rel) ** 2) * MG
    ac = ac.replace([np.inf], np.nan)
    out["pulsatile, best below 1000 nm (mg/dL)"] = float(ac[wl < 1000].min()) if ac[wl < 1000].notna().any() else np.inf
    out["pulsatile, best 1000-1700 nm (mg/dL)"] = float(ac[wl >= 1000].min())
    # net analyte signal: exhaustive best pair, then greedy, all 5 rings per wavelength
    K = {"water": L.mul(sp["w"], axis=0), "blood": L.mul(mua_bd - P["isf_water_frac"] * sp["w"], axis=0),
         "scat": dAs.mul(mus, axis=0), "SO2": L.mul((sp["o"] - sp["d"]) * 0.01, axis=0), "off": L * 0 + 1.0}
    if fat_weight:   # glucose in fat is only fair if water and blood changes in fat are confounders too
        Ls = m["mean_path_sub"]
        K["fat water"] = Ls.mul(sp["w"], axis=0)
        K["fat blood"] = Ls.mul(mua_bd - P["isf_water_frac"] * sp["w"], axis=0)
        K["fat scattering"] = m["sub_dA_dmu_s"].mul(inp["sub_mu_s_mm-1"], axis=0)

    def nas(sel, Sx):
        r = Sx.index.isin(sel); g = Sx[r].to_numpy().ravel()
        Km = np.column_stack([v[r].to_numpy().ravel() for v in K.values()])
        n = np.linalg.norm(g - Km @ np.linalg.lstsq(Km, g, rcond=None)[0])
        return P["drift_rel"] / n * MG if n > 0 else np.inf          # same rule as physio_confounders.py
    pairs = [(a, b) for i, a in enumerate(wl) for b in wl[i + 1:]]
    chosen = list(min(pairs, key=lambda ab: nas(list(ab), S)))
    for k in range(2, 7):
        if k > 2:
            chosen.append(min((w for w in wl if w not in chosen), key=lambda w: nas(chosen + [w], S)))
        out[f"{k} wavelengths (mg/dL)"] = nas(chosen, S)
        if k in (4, 5, 6):
            out[f"{k}-wavelength set"] = ",".join(map(str, sorted(chosen)))
    out["all wavelengths (mg/dL)"] = nas(list(wl), S)
    return out


def main():
    A, B = sys.argv[1], sys.argv[2]
    lin = pd.read_csv("Glucose_Absorption_Data/haque2021_linear_glucose_absorption_600_999nm.csv").set_index("wavelength_nm").iloc[:, 0]
    sub_rel = 0.2184 / 0.65
    cols = {"A: paper model": evaluate(A), "B: continuous model": evaluate(B),
            "B + glucose in fat": evaluate(B, fat_weight=sub_rel),
            "B + glucose in fat (full strength)": evaluate(B, fat_weight=1.0),
            "B + absorption < 1000 nm": evaluate(B, dmua_below_1000=lin)}
    T = pd.DataFrame(cols)
    (Path(B) / "physio_analysis").mkdir(parents=True, exist_ok=True)
    T.to_csv(Path(B) / "physio_analysis" / "tissue_model_comparison.csv")
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        print(T.to_string())


if __name__ == "__main__":
    main()
