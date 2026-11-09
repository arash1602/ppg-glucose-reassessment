"""Accuracy metrics, Clarke error grid, subject-level bootstrap CIs."""
import numpy as np
import pandas as pd


def basic(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 1e-9 and np.std(y) > 1e-9 else np.nan
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "mard": float(np.mean(np.abs(e) / y) * 100), "r2": float(1 - ss_res / ss_tot),
            "pearson_r": float(r), "bias": float(np.mean(e)), "pred_sd": float(np.std(p))}


def clarke_zone(ref, pred):
    """Clarke et al. 1987 zones (same boundary definitions the paper describes)."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or \
       (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
        return "D"
    return "B"


def clarke(y, p):
    z = pd.Series([clarke_zone(a, b) for a, b in zip(y, p)])
    frac = z.value_counts(normalize=True).reindex(list("ABCDE"), fill_value=0) * 100
    return {f"clarke_{k}": float(v) for k, v in frac.items()}


def within_subject(df, ycol="y", pcol="pred", gcol="subjectid"):
    """Does the model track changes INSIDE a person?  Uses subjects with >=2 readings.
    Returns correlation of (y - subject mean) vs (pred - subject mean) and
    the fraction of consecutive changes whose direction is predicted correctly."""
    d = df.groupby(gcol).filter(lambda g: len(g) >= 2).copy()
    if d.empty:
        return {"ws_n_subjects": 0, "ws_r": np.nan, "ws_trend_acc": np.nan}
    d["yc"] = d[ycol] - d.groupby(gcol)[ycol].transform("mean")
    d["pc"] = d[pcol] - d.groupby(gcol)[pcol].transform("mean")
    r = np.corrcoef(d.yc, d.pc)[0, 1] if d.pc.std() > 1e-9 else np.nan
    sort_cols = [gcol] + (["dt"] if "dt" in d.columns else [])
    d = d.sort_values(sort_cols)
    dy = d.groupby(gcol)[ycol].diff()
    dp = d.groupby(gcol)[pcol].diff()
    m = dy.abs() >= 10  # only count clinically noticeable changes (>=10 mg/dL)
    # a predicted change of exactly 0 carries no direction information -> counts as a coin flip (0.5)
    hit = np.where(dp[m] == 0, 0.5, (np.sign(dy[m]) == np.sign(dp[m])).astype(float))
    acc = float(np.mean(hit)) if m.any() else np.nan
    return {"ws_n_subjects": int(d[gcol].nunique()), "ws_r": float(r), "ws_trend_acc": acc}


def all_metrics(df, ycol="y", pcol="pred"):
    out = basic(df[ycol], df[pcol])
    out.update(clarke(df[ycol], df[pcol]))
    out.update(within_subject(df, ycol, pcol))
    return out


def bootstrap_ci(df, fn, gcol="subjectid", n_boot=1000, seed=0, keys=("rmse", "mae", "mard", "clarke_A", "r2")):
    """Resample SUBJECTS (not readings) - readings from one person are not independent."""
    rng = np.random.default_rng(seed)
    groups = df[gcol].unique()
    idx = {g: np.where(df[gcol].values == g)[0] for g in groups}
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        gs = rng.choice(groups, len(groups), replace=True)
        rows = np.concatenate([idx[g] for g in gs])
        m = fn(df.iloc[rows])
        for k in keys:
            vals[k].append(m[k])
    return {f"{k}_ci": (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}
