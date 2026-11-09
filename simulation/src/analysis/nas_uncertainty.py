"""Rank/conditioning of the confounder Jacobian and Monte Carlo uncertainty of the net-analyte results.
Run from MCeXtreme_Sim:  python src/analysis/nas_uncertainty.py results_extended
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import physio_confounders as PC

RES_ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results_extended")
P = PC.P
mean = {k: pd.read_csv(RES_ROOT / "mean_metrics" / f"Metric_Mean_{k}.csv", index_col="wavelength_nm") for k in ["mean_path_derm", "derm_dA_dmu_s"]}
se = {k: pd.read_csv(RES_ROOT / "error_metrics" / f"Metric_Error_{k}.csv", index_col="wavelength_nm") for k in ["mean_path_derm", "derm_dA_dmu_s"]}
wl = mean["mean_path_derm"].index.to_numpy()
mua_w = PC.load_series("Water_Absorption_Data_Public/water_segelstein1981_400_2000nm.csv", "mu_a_water_mm-1", wl)
mua_hbo2 = PC.load_series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_oxy_mm-1", wl)
mua_hb = PC.load_series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_deoxy_mm-1", wl)
inp = pd.read_csv(RES_ROOT / "inputs_per_wavelength.csv", index_col="wavelength_nm")
inp = inp[~inp.index.duplicated()].reindex(wl)
dmua, mus = inp["dmu_a_dCg_mm-1_per_mmol"], inp["derm_mu_s_mm-1"]
mua_bd = P["derm_SO2"] * mua_hbo2 + (1 - P["derm_SO2"]) * mua_hb


def build(L, dAs):
    g = L.mul(dmua, axis=0) + dAs.mul(mus * P["dmus_dCg_rel"], axis=0)
    K = {"water": L.mul(mua_w, axis=0), "blood volume": L.mul(mua_bd - P["isf_water_frac"] * mua_w, axis=0),
         "scattering": dAs.mul(mus, axis=0), "SO2": L.mul((mua_hbo2 - mua_hb) * 0.01, axis=0), "offset": L * 0 + 1.0}
    return g, K


def nas(g, K, sel):
    r = g.index.isin(sel)
    gv = g[r].to_numpy().ravel(); Km = np.column_stack([v[r].to_numpy().ravel() for v in K.values()])
    res = gv - Km @ np.linalg.lstsq(Km, gv, rcond=None)[0]
    n = np.linalg.norm(res)
    return (P["drift_rel"] / n * 18.016 if n > 1e-12 * np.linalg.norm(gv) else np.inf), Km, gv


g0, K0 = build(mean["mean_path_derm"], mean["derm_dA_dmu_s"])
# wavelength sets found by physio_confounders.py (exhaustive best pair, then greedy addition)
_nas = pd.read_csv(RES_ROOT / "physio_analysis" / "net_analyte_signal.csv")
_nas = _nas[_nas.glucose_model.str.startswith("glucose total") & ~_nas.wavelengths.str.startswith("all")]
sets = {len(w.split(",")): [int(float(x)) for x in w.split(",")] for w in _nas.wavelengths}
rows = []
for k, sel in sets.items():
    e, Km, gv = nas(g0, K0, sel)
    A = np.column_stack([Km, gv]); A = A / np.linalg.norm(A, axis=0)
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int((sv > 1e-9 * sv[0]).sum())
    rows.append({"n_wavelengths": k, "channels": 5 * k, "rank_[K g]": rank, "rank_K": np.linalg.matrix_rank(Km / np.linalg.norm(Km, axis=0), tol=1e-9),
                 "smallest_sv_norm": sv[-1], "condition_[K g]": sv[0] / sv[-1], "error_mgdl": e})
# Monte Carlo uncertainty: perturb the per-channel sensitivities by their batch standard errors
rng = np.random.default_rng(0); draws = {k: [] for k in sets}
for _ in range(500):
    L = mean["mean_path_derm"] + rng.standard_normal(mean["mean_path_derm"].shape) * se["mean_path_derm"]
    dA = mean["derm_dA_dmu_s"] + rng.standard_normal(mean["derm_dA_dmu_s"].shape) * se["derm_dA_dmu_s"]
    g, K = build(L, dA)
    for k, sel in sets.items():
        draws[k].append(nas(g, K, sel)[0])
for r in rows:
    d = np.array(draws[r["n_wavelengths"]]); d = d[np.isfinite(d)]
    r["mc_median"] = np.median(d) if len(d) else np.inf
    r["mc_95ci"] = f"{np.percentile(d, 2.5):.1f}-{np.percentile(d, 97.5):.1f}" if len(d) else "inf"
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.to_string(index=False))
T.to_csv(RES_ROOT / "physio_analysis" / "nas_conditioning.csv", index=False)
