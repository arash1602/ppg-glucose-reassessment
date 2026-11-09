"""Step 21 - pretrained PPG foundation models under the same person-level evaluation.

Question: does a large pretrained PPG encoder find glucose information that our task-specific
networks missed? Two open models are tested (encoders re-implemented in fm_models.py, released
weights loaded by parameter name):
  * PaPaGei-S   (57,000 h of clinical and wearable PPG, incl. VitalDB/MIMIC; 10 s at 125 Hz)
  * Pulse-PPG   (field-collected wrist PPG; 50 Hz)
Inputs are the same 10-s segments used by the 10-s ResNet34 (step 02), z-scored and resampled.
Folds, readings and metrics are identical to the other steps (person-level 5-fold, seed 42).

  python 21_foundation_models.py download     # weights -> DATA_DIR/fm_weights (once)
  python 21_foundation_models.py embed        # per-reading mean embeddings (GPU, ~20-40 min)
  python 21_foundation_models.py probe        # frozen embeddings: ridge / GBM, clinical, calibrated, MUST (CPU)
  python 21_foundation_models.py finetune     # PaPaGei fine-tuned end to end (GPU, ~1-2 h)
  python 21_foundation_models.py report       # one table with paired comparisons

Out: results/foundation_models.csv, results/foundation_models_must.csv, results/preds/fm_*.csv
"""
import argparse
import importlib
import json
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import config as C
import metrics as M

MS = importlib.import_module("10_multisignal_eval")      # load(), gbm(), paired(), feature lists
WDIR = C.DATA_DIR / "fm_weights"
WEIGHTS = {
    "papagei": ("https://zenodo.org/records/13983110/files/papagei_s.pt?download=1", "papagei_s.pt", None),
    "pulseppg": ("https://zenodo.org/records/17345536/files/pulseppg_model_weights.zip?download=1",
                 "pulseppg/experiments/out/pulseppg/checkpoint_best.pkl", "net"),
}
NAMES = {"papagei": "PaPaGei-S", "pulseppg": "Pulse-PPG"}
DEMO = ["age", "sex_m", "bmi"]
NPC = 32                      # PCA components of the embedding for tree models (fitted inside each training fold)
PRED = C.RESULTS_DIR / "preds"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------- download
def download():
    WDIR.mkdir(parents=True, exist_ok=True)
    for key, (url, rel, _) in WEIGHTS.items():
        target = WDIR / rel
        if target.exists():
            log(key, "weights present"); continue
        tmp = WDIR / (key + ".download")
        log("downloading", key, "...")
        urllib.request.urlretrieve(url, tmp)
        if url.split("?")[0].endswith(".zip"):
            with zipfile.ZipFile(tmp) as z:
                z.extract(rel, WDIR)
            tmp.unlink()
        else:
            tmp.rename(target)
        log(key, "->", target)


def encoder(key, device):
    import fm_models as F
    m = F.papagei_s() if key == "papagei" else F.pulseppg()
    url, rel, sub = WEIGHTS[key]
    missing = F.load(m, WDIR / rel, key=sub)
    if missing:
        raise RuntimeError(f"{key}: {len(missing)} encoder parameters not found in the released weights: {missing[:5]}")
    return m.to(device)


def prep(x, key):
    """x: (n, 1000) segments at 100 Hz -> z-scored, resampled to the model's rate, float32."""
    x = np.asarray(x, np.float64)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    up, down = (5, 4) if key == "papagei" else (1, 2)
    return resample_poly(x, up, down, axis=1).astype(np.float32)


# -------------------------------------------------------------------------------------------- embed
def embed_array(model, X, key, device, bs=512):
    import torch
    out = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(prep(X[i:i + bs], key)).unsqueeze(1).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def must_segments():
    """MUST recordings as one 10-s segment each, processed exactly like the VitalDB segments."""
    import ppg
    MF = importlib.import_module("11_must_external_features")
    import glob, re, scipy.io
    from pathlib import Path
    key = lambda f: re.sub(r"^\D*", "", Path(f).stem)
    mats = {key(f): f for f in glob.glob(str(C.MUST_DIR / "**" / "*.mat"), recursive=True)}
    T = pd.read_csv(C.RESULTS_DIR / "must_features.csv", dtype={"file": str}) if (C.RESULTS_DIR / "must_features.csv").exists() \
        else MF.must_table()
    S = []
    for f in T["file"]:
        x = np.asarray(scipy.io.loadmat(mats[f])["signal"], float).ravel()
        fs = int(round(len(x) / 10.0))
        xf = ppg.bandpass(resample_poly(x, C.FS, fs))
        n = C.FS * C.SEG10_S
        xf = xf[:n] if len(xf) >= n else np.pad(xf, (0, n - len(xf)), mode="edge")
        S.append(ppg.minmax_pm1(xf))
    return T, np.stack(S).astype(np.float32)


def embed(models):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    S = np.load(C.PROC_DIR / "segs.npy", mmap_mode="r")
    srid = np.load(C.PROC_DIR / "seg_rid.npy")
    order = np.argsort(srid, kind="stable")
    rids, starts = np.unique(srid[order], return_index=True)
    for key in models:
        m = encoder(key, device)
        E = np.full((len(R), m.dim), np.nan, np.float32)
        pos = pd.Series(np.arange(len(R)), index=R["rid"].to_numpy())
        t0, chunk = time.time(), 20000
        acc = np.zeros((len(rids), m.dim)); cnt = np.zeros(len(rids))
        grp = np.searchsorted(rids, srid)
        for i in range(0, len(S), chunk):
            e = embed_array(m, S[i:i + chunk], key, device)
            np.add.at(acc, grp[i:i + chunk], e); np.add.at(cnt, grp[i:i + chunk], 1)
            log(f"{NAMES[key]}: {min(i + chunk, len(S))}/{len(S)} segments ({time.time() - t0:.0f} s)")
        E[pos.reindex(rids).to_numpy()] = (acc / cnt[:, None]).astype(np.float32)
        np.savez(C.PROC_DIR / f"fm_emb_{key}.npz", rid=R["rid"].to_numpy(), emb=E)
        log(f"{NAMES[key]}: {np.isfinite(E[:, 0]).sum()} of {len(R)} readings embedded, dim {m.dim}")
        if C.MUST_DIR.exists():
            T, SM = must_segments()
            np.savez(C.PROC_DIR / f"fm_must_{key}.npz", files=T["file"].astype(str).to_numpy().astype("U"), emb=embed_array(m, SM, key, device))
            log(f"{NAMES[key]}: MUST {len(T)} recordings embedded")


# -------------------------------------------------------------------------------------------- probe
def load_emb(key, R):
    z = np.load(C.PROC_DIR / f"fm_emb_{key}.npz")
    return pd.DataFrame(z["emb"], index=z["rid"]).reindex(R["rid"]).to_numpy(np.float64)


def fill(E, tr):
    mu = np.nanmean(E[tr], 0)
    return np.where(np.isfinite(E), E, mu)


def ridge_fit(X, y, groups):
    """ridge with the penalty chosen by person-grouped inner CV (no person in both inner train and validation)"""
    best, sc = None, StandardScaler().fit(X)
    Xs = sc.transform(X)
    for a in [1, 10, 100, 1e3, 1e4, 1e5]:
        err = 0.0
        for itr, iva in GroupKFold(4).split(Xs, y, groups):
            p = Ridge(alpha=a).fit(Xs[itr], y[itr]).predict(Xs[iva])
            err += np.sum((p - y[iva]) ** 2)
        if best is None or err < best[0]:
            best = (err, a)
    return sc, Ridge(alpha=best[1]).fit(Xs, y), best[1]


def pcs(E, tr):
    """in-fold standardise + PCA of the embedding (fitted on training rows only)"""
    E = fill(E, tr)
    sc = StandardScaler().fit(E[tr])
    p = PCA(NPC, random_state=C.SEED).fit(sc.transform(E[tr]))
    return p.transform(sc.transform(E)), sc, p


def cv_ridge(P, E, extra_cols=()):
    out = np.full(len(P), np.nan); alphas = []
    y = P["glucose"].to_numpy(float)
    D = P[list(extra_cols)].astype(float).to_numpy() if extra_cols else np.zeros((len(P), 0))
    for k in range(C.N_FOLDS):
        tr, te = (P.fold != k).to_numpy(), (P.fold == k).to_numpy()
        Dk = np.where(np.isfinite(D), D, np.nanmean(D[tr], 0)) if D.shape[1] else D
        X = np.hstack([fill(E, tr), Dk])
        sc, m, a = ridge_fit(X[tr], y[tr], P.loc[tr, "subjectid"].to_numpy())
        out[te] = m.predict(sc.transform(X[te])); alphas.append(a)
    return out, alphas


def cv_gbm(P, E, cols, E_ref=None, base=None):
    """GBM on tabular columns + in-fold PCA of the embedding (and of its change since calibration)."""
    cols = MS.usable(P, cols)
    out = np.full(len(P), np.nan)
    y = P["glucose"].to_numpy(float) - (P[base].to_numpy(float) if base else 0)
    for k in range(C.N_FOLDS):
        tr, te = (P.fold != k).to_numpy(), (P.fold == k).to_numpy()
        Z, sc, p = pcs(E, tr)
        blocks = [P[cols].astype(float).to_numpy(), Z]
        if E_ref is not None:                           # change of the embedding since the calibration reading
            Zr = p.transform(sc.transform(fill(E_ref, tr)))
            blocks.append(Z - Zr)
        X = np.hstack(blocks)
        m = MS.gbm().fit(X[tr], y[tr])
        out[te] = m.predict(X[te]) + (P[base].to_numpy(float)[te] if base else 0)
    return out


def ref_preds(P, files):
    """out-of-fold predictions saved by earlier steps, aligned to P by rid"""
    got = {}
    for name, f in files.items():
        fp = PRED / f
        if fp.exists():
            d = pd.read_csv(fp)
            got[name] = d.set_index(d["rid"].astype(int))["pred"].reindex(P["rid"].astype(int)).to_numpy(float)
    return got


def rows_for(P, preds, comps, tag, extra=None):
    preds = {k: v for k, v in preds.items() if np.isfinite(v).all()}
    names = list(preds)
    ci, dci = MS.paired(P, preds, [c for c in comps if c[0] in preds and c[1] in preds])
    rows = []
    for k in names:
        D = P[["rid", "subjectid", "caseid", "dt", "fold"]].assign(y=P["glucose"].to_numpy(), pred=preds[k])
        if D.pred.isna().any():
            continue
        m = M.all_metrics(D)
        r = {"setting": tag, "model": k, "rmse": round(m["rmse"], 2), "rmse_ci": f"{ci[k][0]:.1f}-{ci[k][1]:.1f}",
             "mae": round(m["mae"], 2), "mard": round(m["mard"], 2), "r2": round(m["r2"], 3),
             "pearson_r": round(m["pearson_r"], 3), "clarke_A": round(m["clarke_A"], 1), "within_person_r": round(m["ws_r"], 3)}
        for a, b in comps:
            if a == k and (a, b) in dci:
                d = np.sqrt(np.mean((P.glucose - preds[a]) ** 2)) - np.sqrt(np.mean((P.glucose - preds[b]) ** 2))
                r[f"vs {b}"] = f"{d:+.2f} [{dci[(a, b)][0]:+.2f}, {dci[(a, b)][1]:+.2f}]"
        if extra and k in extra:
            r.update(extra[k])
        rows.append(r)
        slug = "".join(ch if ch.isalnum() else "_" for ch in k.lower())[:50]
        D.to_csv(PRED / f"fm_{tag}_{slug}.csv", index=False)
    return rows


def probe(models):
    PRED.mkdir(exist_ok=True)
    R = MS.load()
    log(f"{len(R)} readings, {R.subjectid.nunique()} people")
    rows = []
    # ---------------- calibration-free -----------------
    preds = ref_preds(R, {"constant (training mean)": "baseline_mean_subject.csv",
                          "age, sex, BMI": "baseline_demo_subject.csv",
                          "ResNet34-1D, 1-s beats": "resnet34_beat_subject.csv",
                          "ResNet34-1D, 10-s segments": "resnet34_seg10_subject.csv",
                          "multi-beat CNN + age, sex, BMI": "improve_deepset_demo.csv",
                          "PPG features (GBM)": "baseline_ppg_gbm_subject.csv",
                          "PPG features + age, sex, BMI (GBM)": "baseline_ppg_demo_gbm_subject.csv"})
    comps, extra = [], {}
    for key in models:
        E = load_emb(key, R)
        n = NAMES[key]
        log(f"{n}: {np.isfinite(E[:, 0]).mean():.1%} of readings have an embedding")
        a = f"{n} embedding (ridge)"
        preds[a], al = cv_ridge(R, E); extra[a] = {"ridge_alpha": "/".join(f"{x:g}" for x in al)}
        b = f"{n} embedding + age, sex, BMI (ridge)"
        preds[b], al = cv_ridge(R, E, DEMO); extra[b] = {"ridge_alpha": "/".join(f"{x:g}" for x in al)}
        c = f"{n} embedding + age, sex, BMI (GBM)"
        preds[c] = cv_gbm(R, E, DEMO)
        for x in (a, b, c):
            comps += [(x, "age, sex, BMI"), (x, "ResNet34-1D, 1-s beats")]
        comps += [(b, "multi-beat CNN + age, sex, BMI"), (c, "PPG features + age, sex, BMI (GBM)")]
        log(f"{n}: frozen-embedding models done")
    ft = {f"PaPaGei-S fine-tuned{s}": f"fm_finetune_papagei{t}.csv" for s, t in [("", ""), (" + age, sex, BMI", "_demo")]}
    preds.update(ref_preds(R, ft))
    for x in ft:
        comps += [(x, "age, sex, BMI"), (x, "ResNet34-1D, 1-s beats"), (x, "multi-beat CNN + age, sex, BMI")]
    rows += rows_for(R, preds, comps, "free", extra)

    # ---------------- with clinical context (as Table 7) -----------------
    T1, T2 = MS.TIERS[1], MS.TIERS[2]
    cp = ref_preds(R, {"clinical + time (T1)": "ms_free_t1___time_in_surgery.csv",
                       "clinical + time + PPG features (T2)": "ms_free_t2___ppg.csv"})
    cc = []
    for key in models:
        E = load_emb(key, R); n = NAMES[key]
        a = f"clinical + time + {n} (GBM)"
        cp[a] = cv_gbm(R, E, T1[1])
        b = f"clinical + time + PPG features + {n} (GBM)"
        cp[b] = cv_gbm(R, E, T2[1])
        cc += [(a, "clinical + time (T1)"), (a, "clinical + time + PPG features (T2)"), (b, "clinical + time + PPG features (T2)")]
        log(f"{n}: clinical-context models done")
    rows += rows_for(R, cp, cc, "clinical")

    # ---------------- one calibration reading (as Table 9) -----------------
    S = R.sort_values(["subjectid", "caseid", "dt"])
    first_idx = S.groupby("subjectid").head(1).index
    first = S.loc[first_idx].set_index("subjectid")
    L = S.drop(first_idx).copy()
    ref = first.reindex(L["subjectid"])
    L["ref_glucose"] = ref["glucose"].to_numpy()
    same = ref["caseid"].to_numpy() == L["caseid"].to_numpy()
    L["elapsed_min"] = np.where(same, (L["dt"].to_numpy() - ref["dt"].to_numpy()) / 60, np.nan)
    L["same_case"] = same.astype(float)
    for c in MS.PPG:
        L[f"d_{c}"] = L[c].to_numpy() - ref[c].to_numpy()
    ref_rid = ref["rid"].to_numpy()
    L = L.reset_index(drop=True)
    base = ["ref_glucose", "elapsed_min", "same_case"] + MS.CLIN + MS.TIME
    kp = ref_preds(L, {"persistence (no model)": "ms_calib_persistence__no_model_.csv",
                       "calibration + time + clinical (C0)": "ms_calib_c0_fingerstick___time___clinical.csv",
                       "C0 + PPG features and change (C1)": "ms_calib_c1___ppg_change.csv"})
    kc = []
    for key in models:
        z = np.load(C.PROC_DIR / f"fm_emb_{key}.npz"); emb = pd.DataFrame(z["emb"], index=z["rid"])
        E = emb.reindex(L["rid"]).to_numpy(np.float64); Er = emb.reindex(ref_rid).to_numpy(np.float64)
        n = NAMES[key]
        a = f"C0 + {n} and its change (GBM)"
        kp[a] = cv_gbm(L, E, base, E_ref=Er, base="ref_glucose")
        kc += [(a, "calibration + time + clinical (C0)"), (a, "C0 + PPG features and change (C1)")]
        log(f"{n}: calibrated models done")
    rows += rows_for(L, kp, kc, "calibrated")

    T = pd.DataFrame(rows)
    T.to_csv(C.RESULTS_DIR / "foundation_models.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(T.drop(columns=[c for c in T.columns if c in ("mae", "pearson_r", "within_person_r")]).to_string(index=False))
    must(models, R)


def must(models, R):
    rows = []
    for key in models:
        f = C.PROC_DIR / f"fm_must_{key}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        T = pd.read_csv(C.RESULTS_DIR / "must_features.csv", dtype={"file": str}).set_index("file").loc[z["files"].astype(str)]
        E = load_emb(key, R); ok = np.isfinite(E[:, 0])
        y = R["glucose"].to_numpy(float)[ok]; g = R["subjectid"].to_numpy()[ok]
        for name, D, Dm in [("embedding", np.zeros((ok.sum(), 0)), np.zeros((len(T), 0))),
                            ("embedding + age, sex, BMI", R.loc[ok, DEMO].astype(float).to_numpy(), T[DEMO].astype(float).to_numpy())]:
            D = np.where(np.isfinite(D), D, np.nanmean(D, 0)) if D.shape[1] else D
            X = np.hstack([E[ok], D]); sc, m, a = ridge_fit(X, y, g)
            p = m.predict(sc.transform(np.hstack([z["emb"], Dm])))
            mm = M.basic(T["glucose"], p); mm.update(M.clarke(T["glucose"], p))
            rows.append({"model": f"{NAMES[key]} {name} (ridge, trained on all VitalDB)", "rmse": round(mm["rmse"], 2),
                         "mard": round(mm["mard"], 2), "r2": round(mm["r2"], 3), "pearson_r": round(mm["pearson_r"], 3),
                         "clarke_A": round(mm["clarke_A"], 1), "clarke_AB": round(mm["clarke_A"] + mm["clarke_B"], 1),
                         "pred_range": f"{p.min():.0f}-{p.max():.0f}"})
    for t, f in [("", "fm_finetune_papagei_must.csv"), (" + age, sex, BMI", "fm_finetune_papagei_demo_must.csv")]:
        if (PRED / f).exists():
            d = pd.read_csv(PRED / f)
            mm = M.basic(d["y"], d["pred"]); mm.update(M.clarke(d["y"], d["pred"]))
            rows.append({"model": f"PaPaGei-S fine-tuned{t} (trained on all VitalDB)", "rmse": round(mm["rmse"], 2),
                         "mard": round(mm["mard"], 2), "r2": round(mm["r2"], 3), "pearson_r": round(mm["pearson_r"], 3),
                         "clarke_A": round(mm["clarke_A"], 1), "clarke_AB": round(mm["clarke_A"] + mm["clarke_B"], 1),
                         "pred_range": f"{d.pred.min():.0f}-{d.pred.max():.0f}"})
    if rows:
        V = R["glucose"].mean()
        T = pd.read_csv(C.RESULTS_DIR / "must_features.csv")
        mm = M.basic(T["glucose"], np.full(len(T), V)); mm.update(M.clarke(T["glucose"], np.full(len(T), V)))
        rows.append({"model": "constant: VitalDB mean", "rmse": round(mm["rmse"], 2), "mard": round(mm["mard"], 2),
                     "r2": round(mm["r2"], 3), "pearson_r": np.nan, "clarke_A": round(mm["clarke_A"], 1),
                     "clarke_AB": round(mm["clarke_A"] + mm["clarke_B"], 1), "pred_range": f"{V:.0f}"})
        M_ = pd.DataFrame(rows); M_.to_csv(C.RESULTS_DIR / "foundation_models_must.csv", index=False)
        print("\n=== MUST (external) ===\n" + M_.to_string(index=False))


# ----------------------------------------------------------------------------------------- finetune
def finetune(with_demo=False, epochs=8, per_reading=6, seed=0, bs=128):
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    import data
    R["fold"] = data.subject_folds(R)
    S = np.load(C.PROC_DIR / "segs.npy")                      # ~350 MB in memory (float16)
    srid = np.load(C.PROC_DIR / "seg_rid.npy")
    pos = pd.Series(np.arange(len(R)), index=R["rid"].to_numpy())
    seg_row = pos.reindex(srid).to_numpy()
    by_row = pd.Series(np.arange(len(srid))).groupby(seg_row).apply(np.array).reindex(range(len(R)))
    y_all = R["glucose"].to_numpy(np.float32)
    tag = "_demo" if with_demo else ""

    class Head(nn.Module):
        def __init__(self, enc, nd):
            super().__init__()
            self.enc = enc
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(enc.dim + nd, 1))

        def forward(self, x, d):
            return self.head(torch.cat([self.enc(x), d], 1)).squeeze(1)

    def demo_for(rows, ref_rows):
        if not with_demo:
            return np.zeros((len(rows), 0), np.float32)
        D = R[DEMO].astype(float); mu, sd = D.iloc[ref_rows].mean(), D.iloc[ref_rows].std().replace(0, 1)
        return ((D.iloc[rows].fillna(mu) - mu) / sd).to_numpy(np.float32)

    def predict(model, rows, ref_rows, X=None, D=None):
        model.eval(); out = np.full(len(rows), np.nan)
        Dm = demo_for(rows, ref_rows) if D is None else D
        with torch.inference_mode():
            if X is not None:                                  # one segment per row (MUST)
                for i in range(0, len(rows), 256):
                    xb = torch.from_numpy(prep(X[i:i + 256], "papagei")).unsqueeze(1).to(device)
                    out[i:i + 256] = model(xb, torch.from_numpy(Dm[i:i + 256]).to(device)).cpu().numpy()
                return out
            idx = np.concatenate([by_row[r] for r in rows]); own = np.concatenate([np.full(len(by_row[r]), j) for j, r in enumerate(rows)])
            p = np.empty(len(idx))
            for i in range(0, len(idx), 512):
                xb = torch.from_numpy(prep(S[idx[i:i + 512]], "papagei")).unsqueeze(1).to(device)
                p[i:i + 512] = model(xb, torch.from_numpy(Dm[own[i:i + 512]]).to(device)).cpu().numpy()
            return pd.Series(p).groupby(own).median().reindex(range(len(rows))).to_numpy()

    def train(tr_rows, va_rows, n_epochs, ref_rows):
        enc = encoder("papagei", device)
        model = Head(enc, len(DEMO) if with_demo else 0).to(device)
        mu, sd = y_all[tr_rows].mean(), y_all[tr_rows].std()
        opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 1e-4},
                                 {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=1e-2)
        Dtr = demo_for(tr_rows, ref_rows); best = (np.inf, None, 0); hist = []
        for ep in range(n_epochs):
            model.train()
            pick = [(rng.choice(by_row[r], min(per_reading, len(by_row[r])), replace=False), j) for j, r in enumerate(tr_rows)]
            idx = np.concatenate([p for p, _ in pick]); own = np.concatenate([np.full(len(p), j) for p, j in pick])
            perm = rng.permutation(len(idx)); tl = 0.0
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                xb = torch.from_numpy(prep(S[np.sort(idx[b])], "papagei")).unsqueeze(1).to(device)
                ob = own[b][np.argsort(idx[b])]
                yb = torch.from_numpy((y_all[tr_rows][ob] - mu) / sd).to(device)
                loss = nn.functional.huber_loss(model(xb, torch.from_numpy(Dtr[ob]).to(device)), yb, delta=1.0)
                opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss) * len(b)
            if va_rows is not None:
                pv = predict(model, va_rows, ref_rows) * sd + mu
                rm = float(np.sqrt(np.mean((pv - y_all[va_rows]) ** 2)))
                hist.append(rm)
                if rm < best[0]:
                    best = (rm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
                log(f"  epoch {ep + 1}: train loss {tl / len(perm):.4f}, validation RMSE {rm:.2f}")
            else:
                log(f"  epoch {ep + 1}: train loss {tl / len(perm):.4f}")
        if best[1] is not None:
            model.load_state_dict(best[1])
        return model, mu, sd, best[2], hist

    out = np.full(len(R), np.nan); best_epochs = []; info = []
    for k in range(C.N_FOLDS):
        tr = np.where(R.fold != k)[0]; te = np.where(R.fold == k)[0]
        subs = R.subjectid.iloc[tr].unique(); rng.shuffle(subs)
        va_s = set(subs[: max(1, len(subs) // 10)])
        va = tr[R.subjectid.iloc[tr].isin(va_s).to_numpy()]; trn = tr[~R.subjectid.iloc[tr].isin(va_s).to_numpy()]
        log(f"fold {k + 1}/{C.N_FOLDS}: {len(trn)} train / {len(va)} validation / {len(te)} test readings")
        model, mu, sd, be, hist = train(trn, va, epochs, tr)
        out[te] = predict(model, te, tr) * sd + mu; best_epochs.append(be); info.append({"fold": k, "best_epoch": be, "val_rmse": hist})
        log(f"fold {k + 1}: test RMSE {np.sqrt(np.mean((out[te] - y_all[te]) ** 2)):.2f} (best epoch {be})")
    D = R[["rid", "subjectid", "caseid", "dt", "fold"]].assign(y=y_all, pred=out)
    PRED.mkdir(exist_ok=True)
    D.to_csv(PRED / f"fm_finetune_papagei{tag}.csv", index=False)
    m = M.basic(D.y, D.pred)
    log(f"PaPaGei-S fine-tuned{tag}: RMSE {m['rmse']:.2f}, R2 {m['r2']:.3f}")
    json.dump({"with_demo": with_demo, "epochs": epochs, "per_reading": per_reading, "seed": seed, "folds": info},
              open(C.RESULTS_DIR / f"fm_finetune_papagei{tag}.json", "w"), indent=1)
    if C.MUST_DIR.exists():                                   # external test: train on all VitalDB, fixed epochs
        n_ep = int(np.median(best_epochs))
        log(f"MUST: training on all VitalDB readings for {n_ep} epochs")
        allr = np.arange(len(R))
        model, mu, sd, _, _ = train(allr, None, n_ep, allr)
        T, SM = must_segments()
        Dm = np.zeros((len(T), 0), np.float32)
        if with_demo:
            Dv = R[DEMO].astype(float); mu_d, sd_d = Dv.mean(), Dv.std().replace(0, 1)
            Dm = ((T[DEMO].astype(float).fillna(mu_d) - mu_d) / sd_d).to_numpy(np.float32)
        p = predict(model, np.arange(len(T)), allr, X=SM, D=Dm) * sd + mu
        pd.DataFrame({"file": T["file"], "subjectid": T["subjectid"], "y": T["glucose"], "pred": p}).to_csv(
            PRED / f"fm_finetune_papagei{tag}_must.csv", index=False)
        log(f"MUST: RMSE {np.sqrt(np.mean((p - T.glucose) ** 2)):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "embed", "probe", "finetune", "report"])
    ap.add_argument("--models", default="papagei,pulseppg")
    ap.add_argument("--demo", action="store_true", help="finetune: add age, sex, BMI to the regression head")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--per_reading", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    if a.step == "download":
        download()
    elif a.step == "embed":
        embed(models)
    elif a.step in ("probe", "report"):
        probe(models)
    else:
        finetune(with_demo=a.demo, epochs=a.epochs, per_reading=a.per_reading, seed=a.seed)


if __name__ == "__main__":
    main()
