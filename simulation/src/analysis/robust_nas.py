"""Robust multi-wavelength (net analyte signal) search.

The greedy search in physio_confounders.py starts from the best wavelength PAIR, but with two (or three)
wavelengths glucose cannot be separated at all (rank argument), so every pair has an essentially infinite
error and the 'best' pair is decided by floating-point noise. This script instead searches ALL sets of
four wavelengths exhaustively (using the additivity of the normal equations over wavelengths), then adds
the fifth and sixth wavelength greedily. Separation for 2 and 3 wavelengths is judged by rank.
Usage (from the simulation folder): python src/analysis/robust_nas.py results_extended results_tissue2
Output: <last folder>/physio_analysis/robust_nas.csv
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_tissue_models as CT

P, MG = CT.P, CT.MG


def jacobians(res, fat_weight=0.0, rng=None):
    m, wl, inp, sp = CT.load(res)
    if rng is not None:      # perturb path lengths and scattering derivatives by their batch standard errors
        for k in ["mean_path_derm", "derm_dA_dmu_s", "mean_path_sub", "sub_dA_dmu_s"]:
            se = pd.read_csv(Path(res) / "error_metrics" / f"Metric_Error_{k}.csv", index_col="wavelength_nm").reindex(m[k].index)
            m[k] = m[k] + rng.standard_normal(m[k].shape) * se.to_numpy()
    L, dAs = m["mean_path_derm"], m["derm_dA_dmu_s"]
    dmua, mus = inp["dmu_a_dCg_mm-1_per_mmol"], inp["derm_mu_s_mm-1"]
    g = L.mul(dmua, axis=0) + dAs.mul(mus * P["dmus_dCg_rel"], axis=0)
    mua_bd = P["derm_SO2"] * sp["o"] + (1 - P["derm_SO2"]) * sp["d"]
    K = [L.mul(sp["w"], axis=0), L.mul(mua_bd - P["isf_water_frac"] * sp["w"], axis=0), dAs.mul(mus, axis=0),
         L.mul((sp["o"] - sp["d"]) * 0.01, axis=0), L * 0 + 1.0]
    if fat_weight:
        Ls, sub_s = m["mean_path_sub"], m["sub_dA_dmu_s"]
        g = g + fat_weight * (Ls.mul(dmua, axis=0) + sub_s.mul(inp["sub_mu_s_mm-1"] * P["dmus_dCg_rel"], axis=0))
        K += [Ls.mul(sp["w"], axis=0), Ls.mul(mua_bd - P["isf_water_frac"] * sp["w"], axis=0),
              sub_s.mul(inp["sub_mu_s_mm-1"], axis=0)]
    G = g.to_numpy()                                         # (n_wl, 5)
    Kt = np.stack([k.to_numpy() for k in K], axis=-1)        # (n_wl, 5, n_conf)
    Kt = Kt / np.linalg.norm(Kt.reshape(-1, Kt.shape[-1]), axis=0)   # column scaling (does not change NAS)
    return wl, G, Kt


def search(wl, G, Kt, chunk=200000, min_gap=0):
    n, c = len(wl), Kt.shape[-1]
    KK = np.einsum("wsi,wsj->wij", Kt, Kt)                   # per-wavelength contributions
    Kg = np.einsum("wsi,ws->wi", Kt, G)
    gg = (G ** 2).sum(1)

    def err(sets):
        sets = np.asarray(sets)
        A = KK[sets].sum(1); b = Kg[sets].sum(1); q = gg[sets].sum(1)
        try:
            x = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            x = np.array([np.linalg.lstsq(a, bb, rcond=None)[0] for a, bb in zip(A, b)])
        nas2 = q - (b * x).sum(1)
        rel = nas2 / q
        e = np.where(rel > 1e-10, P["drift_rel"] / np.sqrt(np.clip(nas2, 1e-300, None)) * MG, np.inf)
        return e

    best_e, best_s = np.inf, None
    ok = lambda c: all(abs(wl[a] - wl[b]) >= min_gap for a, b in itertools.combinations(c, 2))
    combos = (c for c in itertools.combinations(range(n), 4) if ok(c))
    while True:
        block = list(itertools.islice(combos, chunk))
        if not block:
            break
        e = err(block); i = int(np.argmin(e))
        if e[i] < best_e:
            best_e, best_s = float(e[i]), list(block[i])
    out = {4: (best_e, list(best_s))}
    chosen = list(best_s)
    for k in (5, 6):
        cand = [w for w in range(n) if w not in chosen and all(abs(wl[w] - wl[c]) >= min_gap for c in chosen)]
        e = err([chosen + [w] for w in cand]); j = int(np.argmin(e))
        chosen = chosen + [cand[j]]; out[k] = (float(e[j]), list(chosen))
    out["all"] = (float(err([list(range(n))])[0]), list(range(n)))
    # 2 and 3 wavelengths: largest fraction of |g|^2 that lies outside span(K) over ALL pairs / triples
    def max_outside(k):
        best = 0.0
        it = itertools.combinations(range(n), k)
        while True:
            block = np.array(list(itertools.islice(it, chunk)))
            if not len(block):
                return best
            A = KK[block].sum(1); bb = Kg[block].sum(1); q = gg[block].sum(1)
            x = np.array([np.linalg.lstsq(a_, b_, rcond=None)[0] for a_, b_ in zip(A, bb)]) if k < 3 else \
                np.linalg.lstsq(A[0], bb[0], rcond=None)[0][None] * 0
            if k == 3:
                x = np.stack([np.linalg.lstsq(a_, b_, rcond=None)[0] for a_, b_ in zip(A, bb)])
            r = (q - (bb * x).sum(1)) / q
            best = max(best, float(np.nanmax(r)))
    out["max_rel_sv_2"] = max_outside(2); out["max_rel_sv_3"] = max_outside(3)
    return out


def set_error(G, Kt, idx):
    g = G[idx].reshape(-1); K = Kt[idx].reshape(-1, Kt.shape[-1])
    r = g - K @ np.linalg.lstsq(K, g, rcond=None)[0]
    n = np.linalg.norm(r)
    return P["drift_rel"] / n * MG if n > 1e-5 * np.linalg.norm(g) else np.inf


def main(n_draws=200):
    rows = []
    cases = [("A: paper model", sys.argv[1], 0.0), ("B: continuous model", sys.argv[2], 0.0),
             ("B + glucose in fat", sys.argv[2], 0.2184 / 0.65), ("B + glucose in fat (full strength)", sys.argv[2], 1.0)]
    for name, res, fw in cases:
        wl, G, Kt = jacobians(res, fw)
        rng = np.random.default_rng(0)
        draws = [jacobians(res, fw, rng)[1:] for _ in range(n_draws)]
        for gap in (0, 30, 50):
            o = search(wl, G, Kt, min_gap=gap)
            for k in (4, 5, 6):
                idx = sorted(o[k][1])
                d = np.array([set_error(Gd, Kd, idx) for Gd, Kd in draws]); d = d[np.isfinite(d)]
                rows.append({"model": name, "min_spacing_nm": gap, "n_wavelengths": k, "error_mgdl": o[k][0],
                             "mc_median": np.median(d) if len(d) else np.inf,
                             "mc_95": f"{np.percentile(d, 2.5):.1f}-{np.percentile(d, 97.5):.1f}" if len(d) else "inf",
                             "wavelengths": ",".join(str(int(wl[i])) for i in idx)})
            print(name, gap, "done", flush=True)
        rows.append({"model": name, "min_spacing_nm": "-", "n_wavelengths": "all", "error_mgdl": o["all"][0]})
        rows.append({"model": name, "min_spacing_nm": "-", "n_wavelengths": "2 (max share outside span)", "error_mgdl": o["max_rel_sv_2"]})
        rows.append({"model": name, "min_spacing_nm": "-", "n_wavelengths": "3 (max share outside span)", "error_mgdl": o["max_rel_sv_3"]})
    T = pd.DataFrame(rows)
    T.to_csv(Path(sys.argv[2]) / "physio_analysis" / "robust_nas.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 45, "display.max_rows", 100):
        print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
