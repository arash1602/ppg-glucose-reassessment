"""Step 8 - improvement round, evaluated with the SAME person-level folds (no leakage).

A. Calibration-free, multi-beat "deep-set" model (ESP32-sized):
   TinyCNN encoder applied to every beat of a reading -> mean + std of the beat embeddings
   -> + heart-rate variability features (+ optionally age/sex/BMI) -> small MLP -> glucose.
   One prediction per reading from ALL its beats, instead of the median of per-beat guesses.

B. Calibrated tracking (one fingerstick, then PPG):
   For each person, the first reading is the calibration; later readings are predicted.
     persistence      : glucose stays at the calibration value                (no PPG)
     no_ppg_gbm       : calibration value + elapsed time + age/sex/BMI           (no PPG)  <- control
     ppg_delta_gbm    : the above + change in PPG features since calibration
                        + change in the deep-set prediction since calibration (out-of-fold)
   ppg_delta_gbm must beat no_ppg_gbm for the PPG to have added information.

Run: python 08_improve.py [--epochs 40] [--skip-deepset]
Outputs: results/preds/improve_*.csv, results/runs.jsonl, results/improvement_summary.csv
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

import config as C
import data
import metrics as M

HRV = ["hr", "ibi_sd", "rmssd"]
PPG_FEATS = ["hr", "ibi_sd", "rmssd", "amp_raw", "raw_sd", "t_rise", "width50", "area_ratio", "max_slope",
             "min_slope", "apg_ba", "apg_ea", "dicrotic_ratio", "skew", "kurt", "accept_frac", "cos_mean"]


# ------------------------------------------------------------------ A. deep-set model
class DeepSet(nn.Module):
    """TinyCNN beat encoder (same layers as models.TinyCNN) + mean/std pooling over beats."""

    def __init__(self, n_extra, ch=(16, 24, 32, 48)):
        super().__init__()
        layers, c_in = [], 1
        for c, k in zip(ch, (7, 5, 5, 3)):
            layers += [nn.Conv1d(c_in, c, k, 2, k // 2, bias=False), nn.BatchNorm1d(c), nn.ReLU()]
            c_in = c
        self.enc = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * c_in + n_extra, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, beats, mask, extra):
        # beats [B, K, 100], mask [B, K] (1 = real beat), extra [B, n_extra]
        b, k, L = beats.shape
        z = F.adaptive_avg_pool1d(self.enc(beats.reshape(b * k, 1, L)), 1).reshape(b, k, -1)
        m = mask.unsqueeze(-1)
        n = m.sum(1).clamp(min=1)
        mu = (z * m).sum(1) / n
        sd = (((z - mu.unsqueeze(1)) ** 2) * m).sum(1).div(n).clamp(min=1e-6).sqrt()
        return self.head(torch.cat([mu, sd, extra], 1)).squeeze(1)


def make_batch(X, idx_of, rids, K, rng, train):
    B = np.zeros((len(rids), K, C.BEAT_LEN), np.float32)
    Mk = np.zeros((len(rids), K), np.float32)
    for i, r in enumerate(rids):
        ix = idx_of[r]
        if train:
            take = rng.choice(ix, K, replace=len(ix) < K)
        else:
            take = ix[:K]
        take = np.sort(take)
        B[i, :len(take)] = X[take]
        Mk[i, :len(take)] = 1
    return torch.from_numpy(B), torch.from_numpy(Mk)


def run_deepset(R, X, rid, folds, use_demo, epochs, dev, rng):
    order = np.argsort(rid, kind="stable")
    starts = np.searchsorted(rid[order], R["rid"].to_numpy())
    ends = np.searchsorted(rid[order], R["rid"].to_numpy(), side="right")
    idx_of = {r: order[s:e] for r, s, e in zip(R["rid"], starts, ends)}
    cols = HRV + (data.DEMO_COLS if use_demo else [])
    pred = np.full(len(R), np.nan)
    rid_arr = R["rid"].to_numpy()
    y_all = R["glucose"].to_numpy(np.float32)
    for k in range(C.N_FOLDS):
        te = np.where(folds == k)[0]
        trva = np.where(folds != k)[0]
        subs = R.iloc[trva]["subjectid"].unique()
        vsub = set(rng.choice(subs, max(1, int(0.15 * len(subs))), replace=False))
        isva = R.iloc[trva]["subjectid"].isin(vsub).to_numpy()
        tr, va = trva[~isva], trva[isva]
        Etr = R.iloc[tr][cols].astype(float)
        mu_e, sd_e = Etr.mean(), Etr.std().replace(0, 1)
        E = ((R[cols].astype(float) - mu_e) / sd_e).fillna(0).to_numpy(np.float32)
        mu, sd = float(y_all[tr].mean()), float(y_all[tr].std())
        ys = (y_all - mu) / sd
        torch.manual_seed(int(rng.integers(1e9)))
        net = DeepSet(len(cols)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0

        def predict(ix):
            net.eval(); out = []
            with torch.no_grad():
                for s in range(0, len(ix), 64):
                    b = ix[s:s + 64]
                    xb, mb = make_batch(X, idx_of, rid_arr[b], C.MAX_BEATS_PER_READING, rng, False)
                    out.append(net(xb.to(dev), mb.to(dev), torch.from_numpy(E[b]).to(dev)).cpu().numpy())
            return np.concatenate(out)

        for ep in range(epochs):
            net.train()
            perm = rng.permutation(tr)
            for s in range(0, len(perm), 64):
                b = perm[s:s + 64]
                xb, mb = make_batch(X, idx_of, rid_arr[b], 16, rng, True)
                xb = xb * (1 + 0.05 * torch.randn(len(b), 1, 1)) + 0.02 * torch.randn_like(xb)
                loss = F.mse_loss(net(xb.to(dev), mb.to(dev), torch.from_numpy(E[b]).to(dev)),
                                  torch.from_numpy(ys[b]).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            vl = float(np.mean((predict(va) - ys[va]) ** 2))
            print(f"    fold {k} epoch {ep+1:2d}  val {vl:.4f}", flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {n: v.detach().cpu().clone() for n, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
        net.load_state_dict(best_state)
        pred[te] = predict(te) * sd + mu
        mdir = C.DATA_DIR / "models"; mdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best_state, "mu": mu, "sd": sd, "extra_cols": cols,
                    "extra_mu": mu_e.to_dict(), "extra_sd": sd_e.to_dict()},
                   mdir / f"deepset{'_demo' if use_demo else ''}_fold{k}.pt")
    return pred


# ------------------------------------------------------------------ B. calibrated tracking
def calibration_pairs(R):
    """first reading of each person = calibration; every later reading = target."""
    R = R.sort_values(["subjectid", "caseid", "dt"])
    first = R.groupby("subjectid").head(1).set_index("subjectid")
    later = R[~R.index.isin(R.groupby("subjectid").head(1).index)].copy()
    ref = first.reindex(later["subjectid"])
    later["ref_glucose"] = ref["glucose"].to_numpy()
    later["elapsed_s"] = np.where(ref["caseid"].to_numpy() == later["caseid"].to_numpy(),
                                  later["dt"].to_numpy() - ref["dt"].to_numpy(), np.nan)  # NaN = other surgery
    later["same_case"] = (ref["caseid"].to_numpy() == later["caseid"].to_numpy()).astype(float)
    for c in PPG_FEATS + ["ds_pred"]:
        if c in later:
            later[f"d_{c}"] = later[c].to_numpy() - ref[c].to_numpy()
    return later


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def cv_gbm(P, cols, target_delta=True):
    out = np.full(len(P), np.nan)
    for k in range(C.N_FOLDS):
        tr, te = P.fold != k, P.fold == k
        y = P.loc[tr, "glucose"] - (P.loc[tr, "ref_glucose"] if target_delta else 0)
        m = gbm().fit(P.loc[tr, cols], y)
        out[te.to_numpy()] = m.predict(P.loc[te, cols]) + (P.loc[te, "ref_glucose"].to_numpy() if target_delta else 0)
    return out


def record(name, P, ycol="glucose", pcol="pred"):
    D = P.rename(columns={ycol: "y", pcol: "pred"})[["rid", "subjectid", "caseid", "dt", "y", "pred", "fold"]].dropna(subset=["pred"])
    out = C.RESULTS_DIR / "preds"; out.mkdir(exist_ok=True)
    D.to_csv(out / f"{name}.csv", index=False)
    m = M.all_metrics(D)
    ci = M.bootstrap_ci(D, M.all_metrics, n_boot=300)
    with open(C.RESULTS_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps({"run": name, "per_reading": m, "ci": ci}) + "\n")
    print(f"{name:34s} n={m['n']:5d} RMSE {m['rmse']:5.1f} [{ci['rmse_ci'][0]:.1f}-{ci['rmse_ci'][1]:.1f}]  MARD {m['mard']:4.1f}%  "
          f"R2 {m['r2']:+.3f}  ClarkeA {m['clarke_A']:4.1f}%  within-person r {m['ws_r']:+.2f}")
    return {"run": name, **m, **{k: f"{v[0]:.2f}-{v[1]:.2f}" for k, v in ci.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--skip-deepset", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(C.SEED)
    R, X, rid = data.load("beat")
    R = R.reset_index(drop=True)
    folds = data.subject_folds(R)
    R["fold"] = folds
    rows = []
    t0 = time.time()

    # ---- A. calibration-free
    if not a.skip_deepset:
        for use_demo in (False, True):
            name = "improve_deepset" + ("_demo" if use_demo else "")
            print(f"== {name} on {dev}")
            R["ds_pred" if use_demo else "ds_pred_nodemo"] = run_deepset(R, X, rid, folds, use_demo, a.epochs, dev, rng)
            rows.append(record(name, R, pcol="ds_pred" if use_demo else "ds_pred_nodemo"))
    if "ds_pred" not in R:
        R["ds_pred"] = np.nan

    # ---- B. calibrated tracking (first reading of each person = fingerstick)
    P = calibration_pairs(R)
    P["persist"] = P["ref_glucose"]
    rows.append(record("improve_calib_persistence", P, pcol="persist"))
    base_cols = ["ref_glucose", "elapsed_s", "same_case"] + data.DEMO_COLS
    P["no_ppg"] = cv_gbm(P, base_cols)
    rows.append(record("improve_calib_no_ppg_gbm", P, pcol="no_ppg"))
    ppg_cols = base_cols + PPG_FEATS + [f"d_{c}" for c in PPG_FEATS]
    P["ppg_delta"] = cv_gbm(P, ppg_cols)
    rows.append(record("improve_calib_ppg_delta_gbm", P, pcol="ppg_delta"))
    if P["ds_pred"].notna().any():
        P["ppg_delta_ds"] = cv_gbm(P, ppg_cols + ["ds_pred", "d_ds_pred"])
        rows.append(record("improve_calib_ppg_delta_deepset_gbm", P, pcol="ppg_delta_ds"))

    S = pd.DataFrame(rows)
    S.to_csv(C.RESULTS_DIR / "improvement_summary.csv", index=False)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> results/improvement_summary.csv")
    print("Calibrated rows: the PPG helps only if ppg_delta beats no_ppg_gbm (compare RMSE CIs).")


if __name__ == "__main__":
    main()
