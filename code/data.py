"""Load processed arrays and build person-level folds (shared by all experiments)."""
import numpy as np
import pandas as pd

import config as C

DEMO_COLS = ["age", "sex_m", "bmi"]


def load(kind="beat"):
    R = pd.read_csv(C.PROC_DIR / "readings.csv")
    if kind == "beat":
        X = np.load(C.PROC_DIR / "beats.npy", mmap_mode="r")
        rid = np.load(C.PROC_DIR / "beat_rid.npy")
    elif kind == "seg10":
        X = np.load(C.PROC_DIR / "segs.npy", mmap_mode="r")
        rid = np.load(C.PROC_DIR / "seg_rid.npy")
    else:
        raise ValueError(kind)
    return R, X, rid


def subject_folds(R, k=C.N_FOLDS, seed=C.SEED):
    """Every subject (person) appears in exactly one test fold - no person leaks
    between training and testing. Cached so all models use identical folds."""
    f = C.PROC_DIR / f"folds_subject_k{k}_s{seed}.csv"
    if f.exists():
        return pd.read_csv(f).set_index("rid")["fold"].reindex(R["rid"]).to_numpy()
    rng = np.random.default_rng(seed)
    subs = R["subjectid"].unique()
    rng.shuffle(subs)
    fold_of = {s: i % k for i, s in enumerate(subs)}
    folds = R["subjectid"].map(fold_of).to_numpy()
    pd.DataFrame({"rid": R["rid"], "fold": folds}).to_csv(f, index=False)
    return folds


def demo_matrix(R, ref=None):
    """standardised age / sex / BMI; missing -> training mean. ref = training rows."""
    D = R[DEMO_COLS].astype(float).copy()
    ref = D if ref is None else ref[DEMO_COLS].astype(float)
    mu, sd = ref.mean(), ref.std().replace(0, 1)
    return ((D.fillna(mu) - mu) / sd).to_numpy(np.float32)
