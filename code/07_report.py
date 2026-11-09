"""Step 7 - collect every result into results/summary.md, a CSV table with
subject-level bootstrap 95% CIs, and Clarke error-grid figures."""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import metrics as M


def clarke_plot(P, title, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(P.y, P.pred, s=4, alpha=0.35, color="#1f4e79", edgecolors="none")
    ax.plot([0, 400], [0, 400], "k:", lw=0.8)
    for xs, ys in [([0, 175/3, 400/1.2], [70, 70, 400]), ([70, 70], [84, 400]),
                   ([0, 70], [180, 180]), ([70, 290], [180, 400]), ([70, 70], [0, 56]),
                   ([70, 400], [56, 320]), ([180, 180], [0, 70]), ([180, 400], [70, 70]),
                   ([240, 240], [70, 180]), ([240, 400], [180, 180]), ([130, 180], [0, 70])]:
        ax.plot(xs, ys, "k-", lw=0.8)
    for t, x, yv in [("A", 30, 15), ("B", 370, 260), ("B", 280, 370), ("C", 160, 370), ("C", 160, 15),
                     ("D", 30, 140), ("D", 370, 120), ("E", 30, 370), ("E", 370, 15)]:
        ax.text(x, yv, t, fontsize=12)
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel("Reference glucose (mg/dL)"); ax.set_ylabel("Predicted glucose (mg/dL)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    figs = C.RESULTS_DIR / "figures"; figs.mkdir(exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(C.RESULTS_DIR / "preds" / "*.csv"))):
        name = Path(f).stem
        if name.startswith("must_"):
            continue
        P = pd.read_csv(f).dropna(subset=["pred"])
        m = M.all_metrics(P)
        ci = M.bootstrap_ci(P, M.all_metrics, n_boot=300)
        rows.append({"run": name, **m, **{k: f"{v[0]:.1f}-{v[1]:.1f}" for k, v in ci.items()}})
        clarke_plot(P, f"{name}\nRMSE {m['rmse']:.1f}  Clarke A {m['clarke_A']:.1f}%", figs / f"clarke_{name}.png")
    T = pd.DataFrame(rows)
    T.to_csv(C.RESULTS_DIR / "summary_table.csv", index=False)
    cols = ["run", "n", "rmse", "rmse_ci", "mae", "mard", "r2", "pearson_r", "clarke_A", "clarke_B",
            "ws_r", "ws_trend_acc", "pred_sd"]
    cols = [c for c in cols if c in T.columns]
    md = ["# Results summary (per glucose reading)\n",
          "CIs: 95% bootstrap over subjects. ws_r = within-subject correlation (does the model follow "
          "changes inside one person?). pred_sd = spread of predictions (near 0 = model predicts a constant).\n",
          T[cols].round(3).to_markdown(index=False)]
    if (C.RESULTS_DIR / "personalisation.csv").exists():
        Q = pd.read_csv(C.RESULTS_DIR / "personalisation.csv")
        md += ["\n\n# Personalised (first reading used as calibration)\n",
               Q[["source", "strategy", "n", "n_subjects", "rmse", "mae", "mard", "clarke_A"]].round(2).to_markdown(index=False)]
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
