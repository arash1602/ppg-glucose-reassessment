"""Step 3 - train/evaluate a deep model under a given split protocol.

  --split subject : 5-fold, folds by PERSON (subjectid). The honest protocol.
  --split leaky   : replicates the original paper - samples (beats / 10-s segments)
                    shuffled into 70/15/15 regardless of person or reading.
  --input beat|seg10, --model resnet34|tiny, --demo (late-fuse age/sex/BMI)

Predictions are produced per sample and aggregated to one value per glucose
reading (median). Metrics are reported per reading (and per sample, for
comparison with the paper). Output: results/preds/<run>.csv + results/runs.jsonl

Example: python 03_train_eval.py --model resnet34 --input beat --split subject
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
import data
import metrics as M
import models


def batches(idx, bs, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def to_t(X, ids, dev):
    return torch.from_numpy(np.asarray(X[np.sort(ids)], dtype=np.float32)).unsqueeze(1).to(dev), np.sort(ids)


def predict(model, X, ids, D_s, dev, bs=1024):
    model.eval()
    out = np.empty(len(ids), np.float32)
    ids = np.sort(ids)
    with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs]
            xb = torch.from_numpy(np.asarray(X[b], dtype=np.float32)).unsqueeze(1).to(dev)
            db = torch.from_numpy(D_s[b]).to(dev) if D_s is not None else None
            out[i:i + len(b)] = model(xb, db).float().squeeze(1).cpu().numpy()
    return ids, out


def sample_per_reading(rid, k, rng):
    """pick at most k samples per reading for one epoch (balances readings, speeds training)."""
    order = rng.permutation(len(rid))
    r = rid[order]
    _, first = np.unique(r, return_index=True)
    # rank of each sample within its reading after shuffling
    srt = np.argsort(r, kind="stable")
    rank = np.empty(len(r), int)
    counts = np.bincount(r)[np.unique(r)]
    rank[srt] = np.concatenate([np.arange(c) for c in counts])
    return order[rank < k]


def train_one(args, X, y_s, D_s, tr_idx, va_idx, rid, dev, rng):
    torch.manual_seed(int(rng.integers(1e9)))
    model = models.build(args.model, n_demo=3 if args.demo else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    loss_fn = nn.MSELoss()
    best, best_state, bad = np.inf, None, 0
    yt = torch.from_numpy(y_s)
    for ep in range(args.epochs):
        model.train()
        ep_idx = tr_idx[sample_per_reading(rid[tr_idx], args.per_reading, rng)] if args.per_reading else tr_idx
        tl, n = 0.0, 0
        for b in batches(ep_idx, args.bs, True, rng):
            b = np.sort(b)
            xb = torch.from_numpy(np.asarray(X[b], dtype=np.float32)).unsqueeze(1).to(dev)
            if args.augment:
                xb = xb * (1 + 0.05 * torch.randn(len(b), 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)
            db = torch.from_numpy(D_s[b]).to(dev) if D_s is not None else None
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                pred = model(xb, db).float().squeeze(1)
                loss = loss_fn(pred, yt[b].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(b); n += len(b)
        vi, vp = predict(model, X, va_idx, D_s, dev)
        vloss = float(np.mean((vp - y_s[vi]) ** 2))
        sched.step(vloss)
        print(f"    epoch {ep+1:3d}  train {tl/n:.4f}  val {vloss:.4f}", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet34", choices=["resnet34", "tiny"])
    ap.add_argument("--input", default="beat", choices=["beat", "seg10"])
    ap.add_argument("--split", default="subject", choices=["subject", "leaky"])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--per_reading", type=int, default=32, help="samples per reading per epoch (0=all)")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--folds", type=str, default="", help="comma list to run a subset, e.g. 0,1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=None, help="training seed (folds stay fixed)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(C.SEED if args.seed is None else args.seed)
    R, X, rid = data.load(args.input)
    R = R.set_index("rid", drop=False).sort_index()
    y = R.loc[rid, "glucose"].to_numpy(np.float32)
    run = f"{args.model}_{args.input}_{args.split}" + ("_demo" if args.demo else "") + (f"_{args.tag}" if args.tag else "")
    print(f"== {run} on {dev}: {len(R)} readings, {len(X)} samples, "
          f"{models.n_params(models.build(args.model, 3 if args.demo else 0)):,} params")

    if args.split == "subject":
        rfold = data.subject_folds(R.reset_index(drop=True))
        fold_of_reading = pd.Series(rfold, index=R.index)
        sfold = fold_of_reading.loc[rid].to_numpy()
        fold_list = range(C.N_FOLDS)
    else:  # paper protocol: shuffle samples, 70/15/15
        perm = rng.permutation(len(X))
        sfold = np.full(len(X), -1)
        n_tr, n_va = int(0.70 * len(X)), int(0.15 * len(X))
        sfold[perm[:n_tr]] = 2  # train
        sfold[perm[n_tr:n_tr + n_va]] = 1  # val
        sfold[perm[n_tr + n_va:]] = 0  # test  -> "fold 0"
        fold_list = [0]
    if args.folds:
        fold_list = [int(f) for f in args.folds.split(",")]

    pred_rows, sample_rows = [], []
    t0 = time.time()
    for k in fold_list:
        if args.split == "subject":
            te = np.where(sfold == k)[0]
            trva = np.where(sfold != k)[0]
            # validation = 15% of the training SUBJECTS
            subs = R.loc[np.unique(rid[trva]), "subjectid"].unique()
            vsub = set(rng.choice(subs, max(1, int(0.15 * len(subs))), replace=False))
            is_va = R.loc[rid[trva], "subjectid"].isin(vsub).to_numpy()
            tr, va = trva[~is_va], trva[is_va]
        else:
            tr, va, te = np.where(sfold == 2)[0], np.where(sfold == 1)[0], np.where(sfold == 0)[0]
        mu, sd = float(y[tr].mean()), float(y[tr].std())
        y_s = ((y - mu) / sd).astype(np.float32)
        D_s = None
        if args.demo:
            train_readings = R.loc[np.unique(rid[tr])]
            D_s = data.demo_matrix(R.loc[rid], ref=train_readings)
        print(f"  fold {k}: train {len(tr)} / val {len(va)} / test {len(te)} samples")
        model = train_one(args, X, y_s, D_s, tr, va, rid, dev, rng)
        ti, tp = predict(model, X, te, D_s, dev)
        tp = tp * sd + mu
        s = pd.DataFrame({"rid": rid[ti], "pred": tp, "fold": k})
        sample_rows.append(s)
        mdir = C.DATA_DIR / "models"   # outside the project folder (ResNet34 checkpoints are ~85 MB)
        mdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "args": vars(args)},
                   mdir / f"{run}_fold{k}.pt")
    S = pd.concat(sample_rows)
    S["y"] = R.loc[S.rid, "glucose"].to_numpy()
    S["subjectid"] = R.loc[S.rid, "subjectid"].to_numpy()
    P = S.groupby("rid").agg(pred=("pred", "median"), fold=("fold", "first"), n=("pred", "size")).reset_index()
    P = P.merge(R[["rid", "subjectid", "caseid", "dt", "glucose"]].reset_index(drop=True).rename(columns={"glucose": "y"}),
                on="rid", how="left")
    out = C.RESULTS_DIR / "preds"
    out.mkdir(exist_ok=True)
    P.to_csv(out / f"{run}.csv", index=False)
    m_read = M.all_metrics(P)
    m_samp = M.basic(S["y"], S["pred"])
    m_samp.update(M.clarke(S["y"], S["pred"]))
    rec = {"run": run, **vars(args), "minutes": (time.time() - t0) / 60,
           "per_reading": m_read, "per_sample": m_samp}
    with open(C.RESULTS_DIR / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({"run": run, "per_reading": m_read, "per_sample_rmse": m_samp["rmse"]}, indent=1))


if __name__ == "__main__":
    main()
