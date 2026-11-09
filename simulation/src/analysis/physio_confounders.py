"""
Pulsatile-signal and confounder analysis for the NIR glucose sensor design.

Builds on the Monte Carlo results in results/mean_metrics (one run per wavelength,
five ring detectors). Everything here is FIRST-ORDER (Jacobian) analysis:

    For a layer k, the mean partial pathlength <L_k> equals dA/dmu_a,k  (A = -ln R).
    The perturbation-MC term (dA/dmu_s)_k is already saved by monte_carlo.py.

so any small change in a layer's absorption or scattering maps to a change in
absorbance without new simulations:

    dA = sum_k <L_k> * d(mu_a,k)  +  sum_k (dA/dmu_s)_k * d(mu_s,k)

Questions answered
------------------
1. DC (diffuse reflectance) channel: how big is the glucose signal compared with
   realistic physiological changes that are NOT glucose (dermal water, dermal blood
   volume, dermal scattering)? Each confounder is converted to the glucose change
   that would produce the same absorbance change -> "glucose-equivalent error" in
   mg/dL. Photon shot noise is converted the same way.
2. Pulsatile (PPG/AC) channel: if glucose is read from the heartbeat-synchronous
   part of the signal, how much does 1 mmol/L change the AC amplitude, and how does
   that compare with AC measurement noise and with oxygen-saturation changes?
3. How much do the conclusions depend on the assumed glucose-scattering coefficient
   (dmu_s/dCg = -0.29 %/mmol/L in main.py)?

The wavelength / detector-spacing ranking produced here (lowest root-sum-square
glucose-equivalent error) is a physically interpretable replacement for the
weighted WPI score: every term is in mg/dL, so no weights have to be chosen.

Run from the MCeXtreme_Sim folder:   python src/analysis/physio_confounders.py
Outputs go to results/physio_analysis/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]          # .../MCeXtreme_Sim
# usage: python src/analysis/physio_confounders.py [results | results_extended]
RES_ROOT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")
RES = RES_ROOT / "mean_metrics"
OUT = RES_ROOT / "physio_analysis"
MGDL_PER_MMOL = 18.016

# ---------------------------------------------------------------------------
# ASSUMPTIONS (all editable; every one is reported in the output header)
# ---------------------------------------------------------------------------
P = {
    # glucose optical effects
    "dmus_dCg_rel": -0.0029,      # relative change in dermal mu_s per mmol/L (value used in main.py)
    "g_tissue": 0.9,              # anisotropy used in layer_calc.py for 1000-1700 nm
    # DC confounders (dermis), magnitudes of a "typical" non-glucose change
    "d_water_abs": 0.01,          # +1 % absolute dermal water volume fraction (hydration)
    "d_blood_abs": 0.001,         # +0.1 % absolute dermal blood volume fraction (vasomotion)
    "derm_SO2": 0.60,             # dermal (mixed) blood oxygen saturation
    "d_mus_rel": 0.01,            # +1 % dermal scattering (temperature, hydration, pressure)
    # photon shot noise (DC channel)
    "P_source_W": 5e-3,           # emitted optical power
    "det_area_mm2": 1.0,          # photodiode active area
    "responsivity_A_W": 0.9,      # InGaAs, 1000-1700 nm
    "T_avg_s": 10.0,              # averaging time per glucose reading
    "ring_halfwidth_mm": 0.1,     # ring detectors in monte_carlo.py are r +/- 0.1 mm
    # pulsatile channel
    "art_SO2": 0.97,              # arterial saturation of the pulsing blood
    "d_SO2": 0.02,                # 2 % saturation change between readings
    "eps_AC_rel": 0.01,           # 1 % relative uncertainty of the averaged AC amplitude
    "isf_water_frac": 1.0,        # pulsing blood displaces interstitial fluid ~ water
    "d_blood_water_abs": 0.005,   # +0.5 % absolute water fraction of blood (hematocrit / plasma shifts)
    # instrument
    "drift_rel": 5e-4,            # 0.05 % intensity drift between readings (LED / electronics), per channel
}
Q_E = 1.602176634e-19


def load_series(csv, col, wl):
    d = pd.read_csv(ROOT / csv, comment="#")
    x = d.iloc[:, 0].to_numpy(float)
    y = d[col].to_numpy(float)
    o = np.argsort(x)
    return pd.Series(np.interp(wl, x[o], y[o]), index=wl)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = {k: pd.read_csv(RES / f"Metric_Mean_{k}.csv", index_col="wavelength_nm")
         for k in ["mean_path_derm", "mean_path_sub", "derm_dA_dmu_s", "dA_dCg_total", "power_dB"]}
    wl = m["mean_path_derm"].index.to_numpy()
    sds = list(m["mean_path_derm"].columns)
    r_mm = np.array([float(s.split("_")[1].replace("mm", "")) for s in sds])

    # spectra on the simulation grid (mm^-1)
    mua_w = load_series("Water_Absorption_Data_Public/water_segelstein1981_400_2000nm.csv", "mu_a_water_mm-1", wl)
    mua_hbo2 = load_series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_oxy_mm-1", wl)
    mua_hb = load_series("Blood_Absorption_Data/Bosschaart_blood_absorption_2014.csv", "mu_a_deoxy_mm-1", wl)
    inputs = RES_ROOT / "inputs_per_wavelength.csv"
    if inputs.exists():   # written by main_extended.py (covers < 1000 nm too)
        inp = pd.read_csv(inputs, index_col="wavelength_nm").reindex(wl)
        dmua_dCg, mus_derm = inp["dmu_a_dCg_mm-1_per_mmol"], inp["derm_mu_s_mm-1"]
    else:                 # original main.py results (1000-1700 nm)
        dmua_dCg = load_series("Glucose_Absorption_Data/ge2024_dmu_a_dCg.csv", "dmu_a_dCg_mm_inv_per_mmol_L", wl)
        musp_derm = load_series("Exp_Scattering/iino_dermis_reduced_scattering.csv", "dermis_mu_s_prime_mm^-1", wl)
        mus_derm = musp_derm / (1 - P["g_tissue"])

    L = m["mean_path_derm"]                          # dA/dmu_a (dermis), mm
    dAdmus = m["derm_dA_dmu_s"]                      # dA/dmu_s (dermis), mm

    # ---------------- glucose sensitivity (ln-absorbance per mmol/L) ----------------
    S_abs = L.mul(dmua_dCg, axis=0)
    S_sca = dAdmus.mul(mus_derm * P["dmus_dCg_rel"], axis=0)
    S_tot = S_abs + S_sca
    check = (S_tot - m["dA_dCg_total"]).abs().max().max()
    print(f"reproduces main.py dA/dCg_total to within {check:.2e}")

    # ---------------- DC confounders: absorbance change ----------------
    mua_blood_derm = P["derm_SO2"] * mua_hbo2 + (1 - P["derm_SO2"]) * mua_hb
    dA = {
        "water +1%": L.mul(mua_w * P["d_water_abs"], axis=0),
        "blood vol +0.1%": L.mul((mua_blood_derm - P["isf_water_frac"] * mua_w) * P["d_blood_abs"], axis=0),
        "scattering +1%": dAdmus.mul(mus_derm * P["d_mus_rel"], axis=0),
    }
    # shot noise: ring R -> reflectance per mm^2 -> photodiode of given area
    R_ring = 10 ** (m["power_dB"] / 10)
    ring_area = 2 * np.pi * r_mm * (2 * P["ring_halfwidth_mm"])
    P_det = R_ring / ring_area * P["det_area_mm2"] * P["P_source_W"]
    I_ph = P_det * P["responsivity_A_W"]
    B = 1 / (2 * P["T_avg_s"])
    dA["shot noise"] = np.sqrt(2 * Q_E * I_ph * B) / I_ph   # relative intensity noise = dA (ln)
    dA["instrument drift 0.05%"] = L * 0 + P["drift_rel"]

    def to_mgdl(dA_df, S):
        return (dA_df.abs() / S.abs()) * MGDL_PER_MMOL

    eq_tot = {k: to_mgdl(v, S_tot) for k, v in dA.items()}
    eq_abs = {k: to_mgdl(v, S_abs) for k, v in dA.items()}
    rss_tot = np.sqrt(sum(v ** 2 for v in eq_tot.values()))
    rss_abs = np.sqrt(sum(v ** 2 for v in eq_abs.values()))

    # ---------------- pulsatile (AC) channel ----------------
    mua_art = P["art_SO2"] * mua_hbo2 + (1 - P["art_SO2"]) * mua_hb
    contrast = mua_art - P["isf_water_frac"] * mua_w       # what makes the pulse visible
    ac_rel_per_mmol = dmua_dCg / contrast                    # relative AC change per mmol/L (geometry cancels)
    ac_rel_SO2 = (mua_hbo2 - mua_hb) * P["d_SO2"] / contrast
    ac = pd.DataFrame({
        "mua_water": mua_w, "mua_art_blood": mua_art, "pulse_contrast_mm-1": contrast,
        "AC_rel_change_per_mmol_%": 100 * ac_rel_per_mmol,
        "eq_err_AC_noise_mgdl": MGDL_PER_MMOL * P["eps_AC_rel"] / ac_rel_per_mmol.abs(),
        "eq_err_SO2_mgdl": MGDL_PER_MMOL * ac_rel_SO2.abs() / ac_rel_per_mmol.abs(),
    })
    ac_rel_bw = mua_w * P["d_blood_water_abs"] / contrast
    ac["eq_err_blood_water_mgdl"] = MGDL_PER_MMOL * ac_rel_bw.abs() / ac_rel_per_mmol.abs()
    ac["eq_err_AC_rss_mgdl"] = np.sqrt(ac.eq_err_AC_noise_mgdl ** 2 + ac.eq_err_SO2_mgdl ** 2 + ac.eq_err_blood_water_mgdl ** 2)
    ac.index.name = "wavelength_nm"

    # ---------------- sensitivity to the assumed glucose-scattering coefficient ----------------
    scan = {}
    for f in [0, 0.5, 1, 2]:
        S = S_abs + S_sca * f
        eq = {k: to_mgdl(v, S) for k, v in dA.items()}
        rss = np.sqrt(sum(v ** 2 for v in eq.values()))
        best = rss.stack().idxmin()
        scan[f"{f}x"] = {"best_wavelength_nm": best[0], "best_SDS": best[1], "best_rss_mgdl": float(rss.stack().min()),
                         "median_rss_mgdl": float(rss.stack().median())}

    # ---------------- net analyte signal (multi-channel) ----------------
    # Channels = every (wavelength, SDS). Confounder Jacobian K: absorbance change per unit of
    # each confounder in every channel. Glucose vector g. Whatever part of g lies in span(K)
    # cannot be told apart from confounders by ANY linear combination of channels; the rest is
    # the net analyte signal (Lorber 1986). With independent per-channel noise sigma, the best
    # achievable glucose error is sigma / ||NAS||.
    mua_blood_derm_dSO2 = (mua_hbo2 - mua_hb)
    K_cols = {
        "water": L.mul(mua_w, axis=0),
        "blood_volume": L.mul(mua_blood_derm - P["isf_water_frac"] * mua_w, axis=0),
        "scattering": dAdmus.mul(mus_derm, axis=0),
        "SO2": L.mul(mua_blood_derm_dSO2 * 0.01, axis=0),
        "offset_drift": L * 0 + 1.0,
    }

    def nas_error(g_df, wl_sel):
        rows = g_df.index.isin(wl_sel)
        g = g_df[rows].to_numpy().ravel()
        K = np.column_stack([v[rows].to_numpy().ravel() for v in K_cols.values()])
        proj = K @ np.linalg.lstsq(K, g, rcond=None)[0]
        nas = g - proj
        n = np.linalg.norm(nas)
        return (P["drift_rel"] / n * MGDL_PER_MMOL if n > 0 else np.inf), n / np.linalg.norm(g)

    nas_rows = []
    for label, g_df in [("glucose total (abs + scattering)", S_tot), ("glucose absorption only", S_abs)]:
        err_all, sel_all = nas_error(g_df, wl)
        nas_rows.append({"glucose_model": label, "wavelengths": "all 71 x 5 SDS",
                         "selectivity_NAS/|g|": sel_all, "best_eq_err_mgdl": err_all})
        # 1 wavelength = 5 channels = no more than the 5 confounders -> start with the best PAIR
        # (exhaustive), then add wavelengths greedily. All 5 detector rings are used per wavelength.
        pairs = [(a, b) for i, a in enumerate(wl) for b in wl[i + 1:]]
        chosen = list(min(pairs, key=lambda ab: nas_error(g_df, list(ab))[0]))
        for k in range(2, 7):
            if k > 2:
                best = min((w for w in wl if w not in chosen), key=lambda w: nas_error(g_df, chosen + [w])[0])
                chosen.append(best)
            e, sel = nas_error(g_df, chosen)
            nas_rows.append({"glucose_model": label, "wavelengths": ",".join(map(str, sorted(chosen))),
                             "selectivity_NAS/|g|": sel, "best_eq_err_mgdl": e})
    nas_df = pd.DataFrame(nas_rows)
    nas_df.to_csv(OUT / "net_analyte_signal.csv", index=False)

    # ---------------- save tables ----------------
    def long(df, name):
        s = df.stack(); s.index.names = ["wavelength_nm", "SDS"]; return s.rename(name)
    T = pd.concat([long(S_tot, "S_total_lnA_per_mmol"), long(S_abs, "S_abs_lnA_per_mmol"),
                   long(S_sca, "S_scat_lnA_per_mmol"), long(P_det * 1e6, "P_det_uW")]
                  + [long(v, f"eqerr_mgdl_{k}") for k, v in eq_tot.items()]
                  + [long(rss_tot, "eqerr_RSS_mgdl"), long(rss_abs, "eqerr_RSS_absorption_only_mgdl")], axis=1)
    T.to_csv(OUT / "dc_glucose_equivalent_errors.csv")
    ac.to_csv(OUT / "ac_pulsatile_analysis.csv")
    rank = T.sort_values("eqerr_RSS_mgdl").head(15)
    rank.to_csv(OUT / "ranking_top15.csv")
    pd.DataFrame(scan).T.to_csv(OUT / "scattering_assumption_scan.csv")
    with open(OUT / "assumptions.txt", "w") as f:
        for k, v in P.items():
            f.write(f"{k} = {v}\n")

    # ---------------- figures ----------------
    s0 = sds[0]
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), layout="constrained")
    ax[0].semilogy(wl, S_tot[s0].abs(), "k", lw=2, label="glucose 1 mmol/L (total)")
    ax[0].semilogy(wl, S_abs[s0].abs(), "k--", label="glucose 1 mmol/L (absorption only)")
    for k, v in dA.items():
        ax[0].semilogy(wl, v[s0].abs() if hasattr(v, "columns") else v, label=k)
    ax[0].set(title=f"DC absorbance change, {s0}", xlabel="Wavelength (nm)", ylabel="|ΔA| (ln units)")
    ax[0].legend(fontsize=7)
    im = ax[1].imshow(np.log10(rss_tot.T.to_numpy()), aspect="auto", origin="lower", cmap="viridis_r",
                      extent=[wl[0], wl[-1], -0.5, len(sds) - 0.5])
    ax[1].set_yticks(range(len(sds)), sds)
    ax[1].set(title="Glucose-equivalent error, RSS of confounders\nlog10(mg/dL) — lower is better",
              xlabel="Wavelength (nm)")
    fig.colorbar(im, ax=ax[1])
    ax[2].semilogy(wl, ac.eq_err_AC_rss_mgdl, color="tab:orange", lw=2, label="AC channel, all confounders (RSS)")
    ax[2].semilogy(wl, ac.eq_err_blood_water_mgdl, color="tab:orange", lw=0.8, ls="--", label="  of which: blood water +0.5 %")
    ax[2].semilogy(wl, ac.eq_err_AC_noise_mgdl.where(ac.eq_err_AC_noise_mgdl > 0), color="tab:orange", lw=0.8, ls=":",
                   label="  of which: 1 % AC amplitude noise")
    ax[2].semilogy(wl, rss_tot.min(axis=1), "k", lw=2, label="DC channel, best SDS (RSS)")
    ax[2].axhline(20, color="r", ls=":", label="~20 mg/dL (Clarke A border at 100 mg/dL)")
    ax[2].set(title="Pulsatile (AC) vs DC channel", xlabel="Wavelength (nm)", ylabel="Glucose-equivalent error (mg/dL)")
    ax[2].legend(fontsize=7)
    for label, mk in [("glucose absorption only", "o-"), ("glucose total (abs + scattering)", "s--")]:
        d = nas_df[(nas_df.glucose_model == label) & (nas_df.wavelengths != "all 71 x 5 SDS")]
        nwl = d.wavelengths.str.count(",") + 1
        ax[3].semilogy(nwl, d.best_eq_err_mgdl.clip(upper=1e5), mk, label=label)
    allv = nas_df[nas_df.wavelengths == "all 71 x 5 SDS"].best_eq_err_mgdl.iloc[0]
    ax[3].axhline(allv, color="gray", ls=":", label=f"all 71 wavelengths ({allv:.1f} mg/dL, idealised)")
    ax[3].axhline(20, color="r", ls=":")
    ax[3].set(title="Multi-wavelength separation (net analyte signal)\nconfounders removed, 0.05 % noise/channel",
              xlabel="Number of wavelengths (5 detector rings each)", ylabel="Best glucose-equivalent error (mg/dL)",
              ylim=(1, 2e5))
    ax[3].legend(fontsize=7)
    fig.savefig(OUT / "physio_confounders.png", dpi=180)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nTop wavelength/SDS combinations (DC channel):")
        print(rank[["S_total_lnA_per_mmol", "S_abs_lnA_per_mmol", "P_det_uW"] +
                   [c for c in rank.columns if c.startswith("eqerr")]].round(4).head(8))
        print("\nScattering-coefficient assumption scan:")
        print(pd.DataFrame(scan).T)
        print("\nAC channel at selected wavelengths:")
        print(ac.loc[[w for w in [1000, 1100, 1300, 1550, 1600, 1650, 1700] if w in ac.index]].round(3))
        print("\nNet analyte signal (multi-wavelength, all 5 rings, per-channel noise = drift):")
        print(nas_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
