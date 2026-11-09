"""All figures for the paper, from the result files in data/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pathlib import Path as _P
D = str(_P(__file__).resolve().parent / "data") + "/"
OUT = str(_P(__file__).resolve().parent / "figs") + "/"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def tag(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight")
    fig.savefig(OUT + name + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1: study overview
def fig1():
    from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
    S = pd.read_csv(D + "summary_table.csv").set_index("run")
    nas = pd.read_csv(D + "ext/net_analyte_signal.csv")
    dc = pd.read_csv(D + "ext/dc_glucose_equivalent_errors.csv")
    FW, FH, YU = 7.1, 4.75, 66.0
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, YU); ax.axis("off")
    EDGE, MUTE, BG = "#d4d3ce", "#6b6a66", "#fbfbfa"
    cols = [11.0, 41.0, 71.0]; CW = 27.6
    rows = [44.0, 23.5, 3.0]; CH = 18.0
    acc = [C["blue"], C["magenta"], C["aqua"]]

    def card(x, y, title, note, col):
        ax.add_patch(FancyBboxPatch((x, y), CW, CH, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=EDGE, lw=0.8))
        ax.add_patch(Rectangle((x + 1.2, y + CH - 2.55), 0.55, 1.5, fc=col, ec="none"))
        ax.text(x + 2.2, y + CH - 1.8, title, ha="left", va="center", fontsize=7.0, fontweight="bold", color=INK)
        ax.text(x + 1.3, y + 1.6, note, ha="left", va="center", fontsize=5.7, color=MUTE)

    def inset(x, y, w, h):
        a_ = fig.add_axes([x / 100, y / YU, w / 100, h / YU])
        a_.tick_params(labelsize=5.3, length=1.8, pad=1.2, colors=INK2)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_linewidth(0.6)
        a_.grid(False); a_.set_facecolor("none")
        return a_

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color="#9a9893", lw=1.0,
                                                                     shrinkA=0, shrinkB=0, mutation_scale=8))

    # column headers and row labels
    for x_, lab in zip(cols, ["INPUT", "ANALYSIS", "FINDING"]):
        ax.text(x_ + 0.4, 64.0, lab, ha="left", va="center", fontsize=6.6, color=MUTE, fontweight="bold")
    for (n_, nm), y, col in zip([("1", "Evaluation"), ("2", "Physics"), ("3", "Embedded")], rows, acc):
        ax.add_patch(FancyBboxPatch((0.8, y), 96.8 + 1.6, CH, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=col + "0d", ec="none", zorder=0))
        ax.add_patch(plt.Circle((5.6, y + CH / 2 + 2.0), 1.9, fc=col, ec="none"))
        ax.text(5.6, y + CH / 2 + 1.95, n_, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
        ax.text(5.6, y + CH / 2 - 1.6, nm, ha="center", va="center", fontsize=6.2, color=INK, fontweight="bold")
    for y in rows:
        for k in range(2):
            arrow(cols[k] + CW + 0.3, cols[k + 1] - 0.3, y + CH / 2)

    gy = lambda y: y + 3.2            # graphic bottom
    GH = CH - 7.0                     # graphic height

    # ===================== row 1 =====================
    y = rows[0]
    card(cols[0], y, "VitalDB + MUST", "7,224 labs, 2,991 patients  |  MUST: 67 recordings", acc[0])
    z = np.load(D + "case1.npz"); xx = z["ppg"][0][9500:9800].astype(float)
    a = inset(cols[0] + 1.5, gy(y) + 0.8, CW - 3.0, GH - 1.0)
    a.plot(np.arange(300) / 100, xx, color=C["blue"], lw=0.9); a.axis("off")
    a.text(0, 1.02, "finger PPG, 3 s (case 1)", transform=a.transAxes, fontsize=5.3, color=MUTE, va="bottom")
    card(cols[1], y, "Patient-wise evaluation", "5-fold by patient  |  bootstrap CIs over patients", acc[0])
    a = inset(cols[1] + 1.5, gy(y), CW - 3.0, GH); a.axis("off"); a.set_xlim(0, 24); a.set_ylim(0, 6.2)
    pal = ["#8fb8ea", "#f2a07b", "#8ed6b6", "#f3c96a", "#c3b4f0"]
    rng = np.random.default_rng(3)
    test_leaky = set(rng.choice(20, 4, replace=False).tolist())
    for r_, (lab, ttest) in enumerate([("random", test_leaky), ("by person", set(range(16, 20)))]):
        yy = 3.6 - r_ * 3.0
        a.text(0, yy + 0.55, lab, fontsize=5.4, va="center", color=INK)
        for k in range(20):
            person = k // 4
            a.add_patch(Rectangle((5.0 + k * 0.92, yy), 0.78, 1.1, fc=pal[person], ec="none"))
            if k in ttest:
                a.add_patch(Rectangle((5.0 + k * 0.92, yy), 0.78, 1.1, fc="none", ec=INK, lw=0.9))
    a.text(5.0, 5.7, "colour = patient,  outlined = test segment", fontsize=5.2, color=MUTE, va="center")
    card(cols[2], y, "Split matters", "split by person, the network \u2248 age, sex and BMI", acc[0])
    a = inset(cols[2] + 3.2, gy(y) + 1.6, 12.0, GH - 1.8)
    v = [S.loc["resnet34_beat_leaky", "r2"], S.loc["resnet34_beat_subject", "r2"], S.loc["baseline_demo_subject", "r2"]]
    a.bar([0, 1, 2], v, color=[C["orange"], C["blue"], "#a9a8a3"], width=0.6)
    for k_, vv in enumerate(v):
        a.text(k_, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=5.3)
    a.set_xticks([0, 1, 2], ["random", "person", "demog."]); a.set_yticks([0, 0.1, 0.2]); a.set_ylim(0, 0.28)
    a.set_ylabel("R\u00b2", fontsize=5.5, labelpad=1)
    ax.text(cols[2] + 17.3, gy(y) + 6.6, "R\u00b2", fontsize=8.0, color=INK, ha="left", va="center")
    ax.text(cols[2] + 17.3, gy(y) + 3.7, "0.23 \u2192 0.04", fontsize=9.0, color=acc[0], ha="left", va="center", fontweight="bold")

    # ===================== row 2 =====================
    y = rows[1]
    card(cols[0], y, "Published tissue optics", "skin, blood, water and glucose spectra", acc[1])
    a = inset(cols[0] + 1.5, gy(y), CW - 3.0, GH); a.set_xlim(0, 24); a.set_ylim(0, 8.4); a.axis("off")
    for (yb, hb, colr, lab) in [(6.0, 0.6, "#e7c3a2", "epidermis 0.1 mm"), (3.6, 2.4, "#f1d1d4", "dermis 1 mm"), (0.0, 3.6, "#fbf0d6", "fat 10 mm")]:
        a.add_patch(Rectangle((0, yb), 14, hb, fc=colr, ec="none"))
        a.text(14.6, yb + hb / 2, lab, fontsize=5.2, va="center", color=MUTE)
    th = np.linspace(0, np.pi, 60)
    a.plot(4.0 + 3.5 * (1 - np.cos(th)), 6.6 - 4.3 * np.sin(th), color=C["red"], lw=1.0, ls=(0, (2.2, 1.2)))
    a.add_patch(Polygon([[3.4, 8.1], [4.6, 8.1], [4.0, 6.7]], fc=INK2, ec="none"))
    for k in range(5):
        a.add_patch(Rectangle((10.3 + k * 0.7, 6.65), 0.45, 0.75, fc=INK2, ec="none"))
    a.text(4.0, 8.35, "source", fontsize=5.0, color=MUTE, ha="center", va="bottom")
    a.text(11.7, 8.35, "5 detectors", fontsize=5.0, color=MUTE, ha="center", va="bottom")
    card(cols[1], y, "Monte Carlo skin model", "glucose-equivalent error of all confounders", acc[1])
    a = inset(cols[1] + 4.2, gy(y) + 1.7, CW - 6.5, GH - 1.7)
    piv = dc.pivot(index="SDS", columns="wavelength_nm", values="eqerr_RSS_mgdl").sort_index()
    wv = piv.columns.values.astype(float); ed = np.concatenate([[wv[0] - 5], (wv[1:] + wv[:-1]) / 2, [wv[-1] + 5]])
    a.pcolormesh(ed, np.arange(6), np.log10(piv.values), cmap="viridis_r", vmin=1.8, vmax=4.0, shading="flat", rasterized=True)
    a.axvline(1000, color="white", lw=0.7, ls="--")
    a.set_yticks([0.5, 4.5], ["1.7", "2.9"]); a.set_ylabel("mm", fontsize=5.3, labelpad=1)
    a.set_xticks([700, 1000, 1300, 1600], ["700", "1000", "1300", "1600 nm"])
    card(cols[2], y, "Optics limit", "best error vs wavelengths; one alone: 71 mg/dL", acc[1])
    a = inset(cols[2] + 3.2, gy(y) + 1.7, 12.0, GH - 1.9)
    rob = pd.read_csv(D + "ext/robust_nas.csv")
    q = rob[(rob.model == "A: paper model") & (rob.min_spacing_nm == "30")]
    a.plot([2, 3] + list(q.n_wavelengths.astype(int)), [1e5, 1e5] + list(q.error_mgdl), color=C["magenta"], marker="o", ms=2.6, lw=1.1)
    a.set_yscale("log"); a.set_ylim(5, 3e5); a.set_yticks([10, 1e3, 1e5]); a.set_xticks([2, 3, 4, 5, 6])
    a.set_xticks([2, 3, 4, 5, 6], ["2\u03bb", "3\u03bb", "4\u03bb", "5\u03bb", "6\u03bb"]); a.set_ylabel("mg/dL", fontsize=5.3, labelpad=1)
    ax.text(cols[2] + 17.3, gy(y) + 6.6, "model needs", fontsize=7.0, color=INK, ha="left", va="center")
    ax.text(cols[2] + 17.3, gy(y) + 3.7, "\u2265 4 NIR \u03bb", fontsize=9.0, color=acc[1], ha="left", va="center", fontweight="bold")

    # ===================== row 3 =====================
    y = rows[2]
    card(cols[0], y, "ESP32-S3 microcontroller", "240 MHz  |  512 kB SRAM  |  no ML library", acc[2])
    a = inset(cols[0] + 1.5, gy(y), CW - 3.0, GH); a.set_xlim(0, 24); a.set_ylim(0, 8.4); a.axis("off")
    a.add_patch(FancyBboxPatch((7.5, 1.4), 9, 5.6, boxstyle="round,pad=0,rounding_size=0.4", fc="#3d3d3a", ec="none"))
    for k in range(8):
        xp = 8.2 + k * 1.09
        a.plot([xp, xp], [0.3, 1.4], color="#a9a8a3", lw=1.0); a.plot([xp, xp], [7.0, 8.1], color="#a9a8a3", lw=1.0)
    a.text(12, 4.2, "ESP32-S3", fontsize=6.0, color="white", ha="center", va="center", fontweight="bold")
    card(cols[1], y, "Port to plain C", "PC check + hardware-in-the-loop over USB", acc[2])
    a = inset(cols[1] + 1.5, gy(y), CW - 3.0, GH); a.set_xlim(0, 24); a.set_ylim(0, 8.4); a.axis("off")
    steps = [("filter +\nbeats", "275 ms"), ("15\nfeatures", "33 ms"), ("300\ntrees", "3 ms")]
    for k, (lab, tm) in enumerate(steps):
        x0 = 0.6 + k * 8.0
        a.add_patch(FancyBboxPatch((x0, 3.0), 6.0, 4.2, boxstyle="round,pad=0,rounding_size=0.5", fc=acc[2] + "22", ec=acc[2], lw=0.8))
        a.text(x0 + 3.0, 5.1, lab, fontsize=5.6, ha="center", va="center", linespacing=1.1)
        a.text(x0 + 3.0, 1.6, tm, fontsize=5.6, ha="center", va="center", color=MUTE)
        if k < 2:
            a.annotate("", xy=(x0 + 7.8, 5.1), xytext=(x0 + 6.2, 5.1), arrowprops=dict(arrowstyle="-|>", color="#9a9893", lw=0.8, mutation_scale=6))
    card(cols[2], y, "Fast and exact", "per minute of PPG; identical to Python", acc[2])
    a = inset(cols[2] + 5.0, gy(y) + 1.4, 10.2, GH - 1.4)
    a.barh([2, 1, 0], [6.4, 1.555, 0.311], color=["#a9a8a3", C["violet"], acc[2]], height=0.62)
    a.set_yticks([2, 1, 0], ["STM32\nResNet", "ESP32\nCNN", "ESP32\ntrees"]); a.set_xticks([0, 3, 6], ["0", "3 s", "6 s"]); a.set_xlim(0, 7)
    ax.text(cols[2] + 17.3, gy(y) + 6.6, "latency", fontsize=7.0, color=INK, ha="left", va="center")
    ax.text(cols[2] + 17.3, gy(y) + 3.7, "0.3 s", fontsize=9.0, color=acc[2], ha="left", va="center", fontweight="bold")
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- Figure 2: leakage
def fig2():
    S = pd.read_csv(D + "summary_table.csv").set_index("run")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.25, 1]})
    models = [("ResNet34-1D\n1-s beats", "resnet34_beat"), ("ResNet34-1D\n10-s segments", "resnet34_seg10"),
              ("TinyCNN\n1-s beats", "tiny_beat")]
    x = np.arange(len(models)); w = 0.36
    lk = [S.loc[k + "_leaky", "r2"] for _, k in models]
    ps = [S.loc[k + "_subject", "r2"] for _, k in models]
    a.bar(x - w / 2 - 0.01, lk, w, color=C["orange"], label="Random segment split (original)")
    a.bar(x + w / 2 + 0.01, ps, w, color=C["blue"], label="Person-level split (corrected)")
    for xi, v in zip(x - w / 2, lk): a.text(xi, v + 0.006, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    for xi, v in zip(x + w / 2, ps): a.text(xi, v + 0.006, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    demo = S.loc["baseline_demo_subject", "r2"]
    a.axhline(demo, color=INK2, ls="--", lw=1, label=f"Age, sex, BMI only (R\u00b2 {demo:.2f})")
    a.set_xticks(x, [m for m, _ in models]); a.set_ylabel("R² (per reading)"); a.set_ylim(0, 0.5); a.set_yticks([0, 0.1, 0.2, 0.3])
    a.legend(loc="upper left", fontsize=6.6); a.grid(axis="x", visible=False)
    a.set_title("Same network, same data, different split"); tag(a, "a")

    rows = [("Training-set mean\n(constant)", "baseline_mean_subject"),
            ("ResNet34-1D\nperson split", "resnet34_beat_subject"),
            ("ResNet34-1D\nsegment split", "resnet34_beat_leaky")]
    y = np.arange(len(rows))[::-1]
    A = np.array([S.loc[k, "clarke_A"] for _, k in rows]); B = np.array([S.loc[k, "clarke_B"] for _, k in rows])
    b.barh(y, A, 0.55, color=C["blue"], label="Zone A")
    b.barh(y, B, 0.55, left=A + 0.4, color=C["aqua"], label="Zone B")
    for yi, aa, bb in zip(y, A, B):
        b.text(aa / 2, yi, f"{aa:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        b.text(aa + bb + 1.5, yi, f"A+B {aa + bb:.1f}%", va="center", fontsize=7, color=INK)
    b.set_yticks(y, [r for r, _ in rows]); b.set_xlim(0, 125); b.set_xticks([0, 25, 50, 75, 100])
    b.set_xlabel("Readings in Clarke zone (%)"); b.grid(axis="y", visible=False)
    b.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, -0.42))
    b.set_title("Clarke A+B cannot separate them"); tag(b, "b")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2_leakage")


# ---------------------------------------------------------------- Figure 3: incremental signals
def fig3():
    T = pd.read_csv(D + "multisignal_tiers.csv")
    K = pd.read_csv(D + "multisignal_calibrated.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9))

    def panel(ax, df, labels, hl, title, xlim):
        y = np.arange(len(df))[::-1]
        lo = df.rmse_ci.str.split("-").str[0].astype(float); hi = df.rmse_ci.str.split("-").str[1].astype(float)
        cols = [C["orange"] if i in hl else C["blue"] for i in range(len(df))]
        ax.hlines(y, lo, hi, color=cols, lw=1.6)
        ax.scatter(df.rmse, y, s=26, color=cols, zorder=3, edgecolor="white", linewidth=0.8)
        for yi, r, d in zip(y, df.rmse, df.delta_rmse_vs_previous):
            ax.text(xlim[1] - 0.1, yi, ("" if pd.isna(d) else d.split(" ")[0]), ha="right", va="center",
                    fontsize=6.8, color=INK if not pd.isna(d) else INK2)
        ax.set_yticks(y, labels); ax.set_xlim(*xlim); ax.set_xlabel("RMSE, mg/dL (95% CI over patients)")
        ax.grid(axis="y", visible=False); ax.set_title(title)

    T6 = T.iloc[:6]
    panel(a, T6, ["Clinical (age, sex, BMI,\ndiabetes, ASA, emergency)", "+ time since\nanaesthesia start",
                  "+ PPG features", "+ ECG and\narterial pressure", "+ temperature, SpO₂,\nEtCO₂, BIS",
                  "+ drug infusions"], {2}, "No calibration (7,224 readings)", (28.5, 39.5))
    tag(a, "a")
    panel(b, K, ["Glucose assumed\nunchanged", "Reference + time\n+ clinical", "+ PPG change", "+ ECG and arterial\nchange",
                 "+ monitor and drug\nchange"], {2}, "One reference value (4,233 readings)", (25, 49))
    tag(b, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_signals")


# ---------------------------------------------------------------- Figure 4: optical simulation
def fig4():
    d = pd.read_csv(D + "ext/dc_glucose_equivalent_errors.csv")
    ac = pd.read_csv(D + "ext/ac_pulsatile_analysis.csv").replace(np.inf, np.nan)
    nas = pd.read_csv(D + "ext/net_analyte_signal.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.8))
    a, b, c, e = axs.ravel()

    def oxi(ax_, txt=True, y=None):
        ax_.axvspan(650, 1000, color="#efeeea", zorder=0, lw=0)
        for w_ in (660, 940):
            ax_.axvline(w_, color=INK2, lw=0.6, ls=":", zorder=1)
        if txt:
            ax_.text(825, y, "oximeter\nrange", ha="center", va="center", fontsize=6.0, color=INK2)

    s = d[d.SDS == "SDS_1.7mm"].sort_values("wavelength_nm")
    wl = s.wavelength_nm.values
    tot = s.S_total_lnA_per_mmol.abs().values
    absn = s.S_abs_lnA_per_mmol.abs().replace(0, np.nan).values
    # confounder absorbance changes = eq. error (mg/dL) x |S_total| / 18 (back to ln units per change)
    def dA(col):
        return (s[col] * s.S_total_lnA_per_mmol.abs() / 18.016).values
    a.plot(wl, tot, color=INK, lw=2, label="glucose 1 mmol/L (total)")
    a.plot(wl, absn, color=INK, lw=1.4, ls="--", label="glucose, absorption only")
    a.plot(wl, dA("eqerr_mgdl_water +1%"), color=C["blue"], label="water +1%")
    a.plot(wl, dA("eqerr_mgdl_blood vol +0.1%"), color=C["orange"], label="blood volume +0.1%")
    a.plot(wl, dA("eqerr_mgdl_scattering +1%"), color=C["aqua"], label="scattering +1%")
    a.plot(wl, dA("eqerr_mgdl_instrument drift 0.05%"), color=C["violet"], label="drift 0.05%")
    a.set_yscale("log"); a.set_ylim(1e-6, 3); a.set_xlabel("Wavelength (nm)"); a.set_xlim(650, 1710); oxi(a, y=2e-6)
    a.set_ylabel("|ΔA| (ln units), 1.7 mm detector")
    a.legend(loc="upper left", ncol=2, fontsize=6.0); a.set_ylim(1e-6, 30); a.set_title("Glucose signal vs everyday changes"); tag(a, "a")

    piv = d.pivot(index="SDS", columns="wavelength_nm", values="eqerr_RSS_mgdl").sort_index(ascending=False)
    wv = piv.columns.values.astype(float)
    edges = np.concatenate([[wv[0] - 5], (wv[1:] + wv[:-1]) / 2, [wv[-1] + 5]])
    im = b.pcolormesh(edges, np.arange(len(piv) + 1) - 0.5, np.log10(piv.values[::-1]), cmap="viridis_r", vmin=1.8, vmax=4.0,
                      shading="flat", rasterized=True)
    b.axvline(1000, color="white", lw=1.0, ls="--")
    b.text(825, 4.75, "oximeter range", ha="center", va="top", fontsize=6.0, color="white", fontweight="bold")
    b.set_xlim(650, 1710)
    b.set_yticks(range(len(piv)), [i.replace("SDS_", "").replace("mm", " mm") for i in piv.index[::-1]])
    b.set_ylabel("Source-detector distance"); b.set_xlabel("Wavelength (nm)"); b.grid(False)
    cb = fig.colorbar(im, ax=b, pad=0.02, fraction=0.05)
    cb.set_ticks([2, 3, 4], labels=["100", "1,000", "10,000"]); cb.set_label("mg/dL", fontsize=7)
    best = d.loc[d.eqerr_RSS_mgdl.idxmin()]
    b.scatter([best.wavelength_nm], [list(piv.index[::-1]).index(best.SDS)], marker="o", s=40, facecolor="none",
              edgecolor="white", lw=1.5)
    b.annotate(f"best: {best.eqerr_RSS_mgdl:.0f} mg/dL\n{int(best.wavelength_nm)} nm, {best.SDS[4:].replace('mm', ' mm')}",
               (best.wavelength_nm, list(piv.index[::-1]).index(best.SDS)), xytext=(1230, 3.3), fontsize=6.6,
               color="white", fontweight="bold", va="center",
               arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    b.set_title("Glucose-equivalent error, one wavelength"); tag(b, "b")

    dcbest = d.groupby("wavelength_nm").eqerr_RSS_mgdl.min()
    c.plot(ac.wavelength_nm, ac.eq_err_AC_rss_mgdl, color=C["orange"], label="pulsatile (PPG) channel")
    c.plot(ac.wavelength_nm, ac.eq_err_blood_water_mgdl, color=C["orange"], lw=1, ls="--",
           label="  of which blood water +0.5%")
    c.plot(dcbest.index, dcbest.values, color=INK, label="steady (DC) channel, best ring")
    c.axhline(20, color=C["red"], lw=1, ls=":"); c.text(1700, 23, "20 mg/dL", ha="right", fontsize=6.6, color=C["red"])
    c.set_xlim(650, 1710); oxi(c, y=3e6); c.text(825, 2e4, "glucose absorption\nnot modelled:\nno pulsatile\nsignal", ha="center", va="center", fontsize=5.8, color=C["orange"]); c.set_yscale("log"); c.set_ylim(10, 3e8); c.set_xlabel("Wavelength (nm)"); c.set_ylabel("Glucose-equivalent error (mg/dL)")
    c.legend(loc="upper right", fontsize=6.4); c.set_title("Pulsatile vs steady channel"); tag(c, "c")

    rob = pd.read_csv(D + "ext/robust_nas.csv")
    for model, col, mk, ls, lab in [("A: paper model", C["blue"], "o", "-", "tissue model of the paper"),
                                    ("B: continuous model", C["orange"], "s", "-", "continuous tissue model"),
                                    ("B + glucose in fat", C["aqua"], "^", "--", "continuous, glucose also in fat")]:
        q = rob[(rob.model == model) & (rob.min_spacing_nm == "30")]
        e.plot([2, 3] + list(q.n_wavelengths.astype(int)), [1e5, 1e5] + list(q.error_mgdl), color=col, marker=mk, ms=4.5,
               ls=ls, lw=1.3, label=lab)
    allv = float(rob[(rob.model == "A: paper model") & (rob.n_wavelengths == "all")].error_mgdl.iloc[0])
    e.axhline(allv, color=INK2, ls="--", lw=1); e.text(6.05, allv * 1.25, f"all 78 wavelengths: {allv:.1f}", ha="right",
                                                      fontsize=6.6, color=INK2)
    e.axhline(20, color=C["red"], lw=1, ls=":")
    e.set_yscale("log"); e.set_ylim(0.3, 2e5); e.set_xticks([2, 3, 4, 5, 6])
    e.set_xlabel("Number of wavelengths (5 detector rings each)"); e.set_ylabel("Best glucose-equivalent error (mg/dL)")
    e.text(2.05, 0.45, "2-3 wavelengths: not identifiable\n(>400 mg/dL with glucose in fat; clipped)", fontsize=6.0, color=INK2)
    e.legend(loc="upper right", fontsize=6.6); e.set_title("Combining wavelengths (net analyte signal)"); tag(e, "d")
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    save(fig, "fig4_simulation")


# ---------------------------------------------------------------- Figure 5: embedded
def fig5():
    H = pd.read_csv(D + "esp32_hil_avg.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw={"width_ratios": [1.35, 1]})
    rows = [("Original: ResNet34\non STM32H7 (reported)", [("inference", 6400, C["red"])]),
            ("Per-beat TinyCNN\non ESP32-S3", [("filter + beats", 283.4, C["blue"]), ("CNN, 84 beats", 1271.2, C["violet"])]),
            ("Device model\non ESP32-S3 (this work)", [("filter + beats", 274.7, C["blue"]), ("features", 32.7, C["aqua"]),
                                                       ("300 trees", 3.1, C["yellow"])])]
    seen = set()
    for i, (lab, parts) in enumerate(rows[::-1]):
        left = 0
        for name, v, col in parts:
            a.barh(i, v, 0.55, left=left, color=col, label=None if name in seen else name, edgecolor="white", lw=0.8)
            seen.add(name); left += v
        a.text(left + 60, i, f"{left:,.0f} ms", va="center", fontsize=7, color=INK, fontweight="bold")
    a.text(330, -0.42, "275 ms filtering/beat extraction; 33 ms features; 3 ms tree inference (300 trees)", fontsize=6.0, color=INK2, va="center")
    a.set_yticks(range(3), [r for r, _ in rows[::-1]]); a.set_xlim(0, 7600); a.set_ylim(-0.65, 2.45)
    a.set_xlabel("Time per 60-s window (ms)"); a.grid(axis="y", visible=False)
    a.legend(loc="upper right", fontsize=6.5, bbox_to_anchor=(1.0, 0.72)); a.set_title("Latency"); tag(a, "a")

    for cal, col, lab in [(False, C["blue"], "calibration-free (20)"), (True, C["orange"], "calibrated (10)")]:
        q = H[H.calibrated == cal]
        p = q.python_cal if cal else q.python_free; g = q.esp32_cal if cal else q.esp32_free
        b.scatter(p, g, s=22, color=col, edgecolor="white", lw=0.6, label=lab, zorder=3)
    lim = [85, 215]; b.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=1)
    b.set_xlim(lim); b.set_ylim(lim); b.set_aspect("equal")
    b.set_xlabel("Python estimate (mg/dL)"); b.set_ylabel("ESP32-S3 estimate (mg/dL)")
    b.text(90, 200, "max |difference| = 0 mg/dL\n30/30 recordings", fontsize=6.8, color=INK)
    b.legend(loc="lower right"); b.set_title("Hardware-in-the-loop agreement"); tag(b, "b")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5_embedded")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
    print("ok")


# ---------------------------------------------------------------- Figure: signal processing + model schematics
def fig_signal():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
    import ppg as P
    z = np.load(D + "case1.npz")
    x = z["ppg"][0].astype(float)
    xf = P.bandpass(x)
    beats, peaks, q = P.extract_beats(xf)
    xz = P.robust_z(xf)
    allp = P.detect_peaks(xz)
    fig = plt.figure(figsize=(7.1, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.28)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); m = None
    t0 = 9500; n = 1000; t = np.arange(n) / 100
    seg = xz[t0:t0 + n]
    raw = x[t0:t0 + n]; raw = (raw - np.median(raw)) / (np.percentile(raw, 95) - np.percentile(raw, 5)) * 2 + 4.2
    a.plot(t, raw, color=INK2, lw=0.9, label="PPG, 100 Hz (scaled)")
    a.plot(t, seg, color=C["blue"], lw=1.2, label="band-passed, robust z")
    pk = allp[(allp >= t0) & (allp < t0 + n)] - t0
    acc = set((peaks[(peaks >= t0) & (peaks < t0 + n)] - t0).tolist())
    ok = np.array([p in acc for p in pk], bool)
    a.scatter(pk[ok] / 100, seg[pk[ok]], s=16, zorder=3, marker="o", color=C["aqua"], edgecolor="white", lw=0.5,
              label="peak of accepted 1-s beat")
    a.scatter(pk[~ok] / 100, seg[pk[~ok]], s=30, zorder=3, marker="X", color=C["red"], edgecolor="white", lw=0.5,
              label="peak, beat rejected (template)")
    a.set_xlabel("Time (s)"); a.set_yticks([]); a.set_xlim(0, 10); a.set_ylim(-3.2, 7.4)
    a.legend(loc="upper right", ncol=2, fontsize=6.2); a.set_title("10 s of VitalDB case 1 (glucose 154 mg/dL)"); tag(a, "a")
    tb = np.arange(100) / 100 - 0.5
    sel = np.random.default_rng(0).choice(len(beats), 60, replace=False)
    for bt in beats[sel]:
        b.plot(tb, bt, color=C["blue"], lw=0.5, alpha=0.3)
    b.plot(tb, np.median(beats, 0), color=INK, lw=1.8)
    b.set_xlabel("Time from peak (s)"); b.set_ylabel("Scaled amplitude"); b.set_xlim(-0.5, 0.5)
    b.set_title(f"Accepted beats (black: median of {q['n_beats']:,})"); tag(b, "b")

    save(fig, "fig_signal")


# ---------------------------------------------------------------- Figure: predicted vs reference with Clarke grid
def clarke_lines(ax):
    L = [([0, 400], [0, 400], ":"), ([0, 175 / 3], [70, 70], "-"), ([175 / 3, 400 / 1.2], [70, 400], "-"),
         ([70, 70], [84, 400], "-"), ([0, 70], [180, 180], "-"), ([70, 290], [180, 400], "-"), ([70, 70], [0, 56], "-"),
         ([70, 400], [56, 320], "-"), ([180, 180], [0, 70], "-"), ([180, 400], [70, 70], "-"), ([240, 240], [70, 180], "-"),
         ([240, 400], [180, 180], "-"), ([130, 180], [0, 70], "-")]
    for xs, ys, ls in L:
        ax.plot(xs, ys, color=INK2, lw=0.7, ls=ls, zorder=1)
    for s_, (xx, yy) in {"A": (30, 15), "B": (370, 260), "B ": (280, 370), "C": (160, 370), "C ": (160, 15),
                         "D": (30, 140), "D ": (370, 120), "E": (30, 370), "E ": (370, 15)}.items():
        ax.text(xx, yy, s_.strip(), fontsize=7, color=INK2, ha="center", va="center", fontweight="bold")


def fig_scatter():
    S = pd.read_csv(D + "summary_table.csv").set_index("run")
    T = pd.read_csv(D + "multisignal_calibrated.csv")
    runs = [("resnet34_beat_leaky.csv", "ResNet34-1D, segment split", C["orange"], S.loc["resnet34_beat_leaky"]),
            ("resnet34_beat_subject.csv", "ResNet34-1D, person split", C["blue"], S.loc["resnet34_beat_subject"]),
            ("ms_calib_c3___monitor___drugs_change.csv", "Best calibrated model, person split", C["aqua"], None)]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75))
    for ax, (f, title, col, row) in zip(axs, runs):
        d = pd.read_csv(D + f)
        y, p = d.y.values, d.pred.values
        ax.scatter(y, p, s=3, color=col, alpha=0.35, lw=0, zorder=2, rasterized=True)
        clarke_lines(ax)
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        rmse = np.sqrt(np.mean((y - p) ** 2))
        ax.text(200, 418, f"n = {len(y):,}   RMSE {rmse:.1f} mg/dL   R\u00b2 {r2:.3f}", ha="center", va="bottom", fontsize=6.4, color=INK2)
        ax.set_xlim(0, 400); ax.set_ylim(0, 400); ax.set_aspect("equal"); ax.set_xticks([0, 100, 200, 300, 400])
        ax.set_yticks([0, 100, 200, 300, 400]); ax.grid(False)
        ax.set_xlabel("Laboratory glucose (mg/dL)")
        ax.set_title(title, fontsize=7.8, pad=14)
    axs[0].set_ylabel("Predicted glucose (mg/dL)")
    for ax, s_ in zip(axs, "abc"):
        tag(ax, s_)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_scatter")


if __name__ == "__main__":
    fig_signal()
    fig_scatter()
    print("extra ok")


# ---------------------------------------------------------------- Figure: detailed architectures
def slab(ax, x, y0, w, h, col, d=0.18, alpha=1.0, lw=0.8):
    """pseudo-3D feature-map slab; front face bottom-left at (x, y0)"""
    from matplotlib.patches import Polygon, Rectangle
    ax.add_patch(Polygon([[x, y0 + h], [x + d, y0 + h + d], [x + w + d, y0 + h + d], [x + w, y0 + h]],
                         closed=True, fc=col, ec=INK2, lw=lw, alpha=0.55 * alpha))
    ax.add_patch(Polygon([[x + w, y0], [x + w + d, y0 + d], [x + w + d, y0 + h + d], [x + w, y0 + h]],
                         closed=True, fc=col, ec=INK2, lw=lw, alpha=0.75 * alpha))
    ax.add_patch(Rectangle((x, y0), w, h, fc=col, ec=INK2, lw=lw, alpha=alpha))


def arr(ax, x0, y0, x1, y1, col=INK2, lw=0.9):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                                                   shrinkA=0, shrinkB=0, mutation_scale=7))


def node_col(ax, x, yc, n, r, col, gap):
    ys = yc + (np.arange(n) - (n - 1) / 2) * gap
    for yy in ys:
        ax.add_patch(plt.Circle((x, yy), r, fc="white", ec=col, lw=0.9, zorder=3))
    return ys


def fig_arch():
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(7.1, 9.0))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.3, 0.62, 0.95, 0.95, 0.95], hspace=0.32)
    A, B, Cx, Dx, E = [fig.add_subplot(gs[k]) for k in range(5)]
    for ax in (A, B, Cx, Dx, E):
        ax.axis("off"); ax.set_xlim(0, 30)
    tt = np.linspace(0, 1, 100)
    wave = np.exp(-((tt - 0.5) / 0.09) ** 2) - 0.35 * np.exp(-((tt - 0.75) / 0.06) ** 2)
    LAB = dict(ha="center", va="top", fontsize=5.9, linespacing=1.15)

    def box(ax, x0, y0, w, h, txt, col, fs=6.4):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=col + "33", ec=col, lw=1.0))
        ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=fs, linespacing=1.15)

    def mlp(ax, x, yc, labs, col, n=(6, 4, 1), gap=0.42, dx=1.5, ytxt=None):
        cols = [node_col(ax, x + i * dx, yc, k, 0.15, col, gap) for i, k in enumerate(n)]
        for i in range(len(cols) - 1):
            for a_ in cols[i]:
                for b_ in cols[i + 1]:
                    ax.plot([x + i * dx, x + (i + 1) * dx], [a_, b_], color=GRID, lw=0.5, zorder=1)
        for i, l in enumerate(labs):
            ax.text(x + i * dx, ytxt, l, **LAB)

    # ---------- (a) ResNet34-1D
    A.set_ylim(-1.6, 7.3); tag(A, "a")
    A.set_title("ResNet34-1D: 7.38 M parameters (architecture of the original study)", loc="left")
    yl = -0.3; mid = 2.65
    A.plot(0.4 + tt * 1.6, mid + wave * 1.1, color=C["blue"], lw=1.4)
    A.text(1.2, yl, "Input\n1 \u00d7 100", **LAB)
    arr(A, 2.15, mid, 2.65, mid)
    items = [("Stem conv", 64, 100, 1, "#9aa9b8", None),
             ("Stage 1 \u00d73", 64, 100, 3, "#b7d3f2", "75 k\u2020"),
             ("Stage 2 \u00d74", 128, 50, 4, "#7fb0e8", "379 k"),
             ("Stage 3 \u00d76", 256, 25, 6, "#3f87d9", "2.30 M"),
             ("Stage 4 \u00d73", 512, 13, 3, "#1d5aa8", "4.46 M")]
    slot = 3.0; x0 = 2.8
    for k, (lab, c, L, n, col, pp) in enumerate(items):
        h = 0.6 + 0.45 * np.log2(c); w = 0.25 + 0.009 * L; m = min(n, 3)
        tot = w + (m - 1) * 0.28
        xs = x0 + k * slot + (2.2 - tot) / 2
        for q in range(m):
            slab(A, xs + q * 0.28, mid - h / 2, w, h, col, d=0.2)
        xc = x0 + k * slot + 1.1
        A.text(xc, yl, f"{lab}\n{c} \u00d7 {L}", **LAB)
        if pp:
            A.text(xc + 0.1, mid + h / 2 + 0.35, pp, ha="center", va="bottom", fontsize=6.0, color=INK2, style="italic")
        arr(A, x0 + k * slot + 2.35, mid, x0 + (k + 1) * slot - 0.1, mid)
    xg = x0 + 5 * slot
    slab(A, xg + 0.4, mid - 1.6, 0.22, 3.2, "#b9c3cc", d=0.12)
    A.text(xg + 0.55, yl, "GAP\n512", **LAB)
    arr(A, xg + 1.0, mid, xg + 1.6, mid)
    mlp(A, xg + 2.0, mid, ["FC\n256", "FC\n128", "out"],
        C["violet"], n=(7, 5, 1), gap=0.45, dx=1.75, ytxt=yl)
    A.text(xg + 2.0, 6.5, "+ age, sex, BMI\n(optional)", ha="center", va="center", fontsize=6.0, color=C["violet"])
    arr(A, xg + 2.0, 6.0, xg + 2.0, 4.3, col=C["violet"])

    # ---------- (b) residual block
    B.set_ylim(-0.4, 3.6); tag(B, "b")
    B.set_title("Residual basic block (16 in total: 3 + 4 + 6 + 3)", loc="left")
    yb = 1.0
    B.text(1.0, yb, r"$C_{in}\times L$", ha="center", va="center", fontsize=7)
    arr(B, 2.0, yb, 2.6, yb)
    ops = [("Conv k3/s", "#f3a35c"), ("BatchNorm", "#59b359"), ("ReLU", "#59b359"),
           ("Conv k3/1", "#f3a35c"), ("BatchNorm", "#59b359")]
    bx = 2.7
    for lab, col in ops:
        box(B, bx, yb - 0.6, 2.5, 1.2, lab, col, fs=6.0)
        arr(B, bx + 2.55, yb, bx + 3.0, yb); bx += 3.1
    B.add_patch(plt.Circle((bx + 0.35, yb), 0.33, fc="white", ec=INK, lw=1.0, zorder=4))
    B.plot([bx + 0.17, bx + 0.53], [yb, yb], color=INK, lw=1.0, zorder=5); B.plot([bx + 0.35, bx + 0.35], [yb - 0.18, yb + 0.18], color=INK, lw=1.0, zorder=5)
    arr(B, bx + 0.72, yb, bx + 1.2, yb)
    box(B, bx + 1.25, yb - 0.6, 1.9, 1.2, "ReLU", "#59b359")
    arr(B, bx + 3.2, yb, bx + 3.7, yb)
    B.text(bx + 5.2, yb, r"$C_{out}\times L/s$", ha="center", va="center", fontsize=7)
    B.plot([2.3, 2.3, bx + 0.35, bx + 0.35], [yb, 2.75, 2.75, yb + 0.7], color=INK2, lw=1.0)
    arr(B, bx + 0.35, yb + 0.75, bx + 0.35, yb + 0.34, col=INK2)
    B.text((2.3 + bx) / 2, 2.85, "shortcut: identity, or Conv1d k1/2 + BatchNorm in the first block of stages 2-4",
           ha="center", va="bottom", fontsize=6.0, color=INK2)

    # ---------- (c) TinyCNN
    Cx.set_ylim(-1.2, 4.9); tag(Cx, "c")
    Cx.set_title("TinyCNN: 12.3 k parameters (per-beat model sized for the microcontroller)", loc="left")
    mid = 2.1; yl = -0.15
    Cx.plot(0.4 + tt * 1.6, mid + wave * 1.0, color=C["blue"], lw=1.4)
    Cx.text(1.2, yl, "Input\n1 \u00d7 100", **LAB)
    arr(Cx, 2.15, mid, 3.05, mid)
    slot = 4.1; x0 = 3.2
    for k, (c, L, kk) in enumerate([(16, 50, 7), (24, 25, 5), (32, 13, 5), (48, 7, 3)]):
        h = 0.5 + 0.5 * np.log2(c); w = 0.3 + 0.025 * L
        xs = x0 + k * slot + (2.2 - w) / 2
        slab(Cx, xs, mid - h / 2, w, h, ["#b7d3f2", "#7fb0e8", "#3f87d9", "#1d5aa8"][k], d=0.2)
        Cx.text(x0 + k * slot + 1.1, yl, f"Conv k{kk}/2\n{c} \u00d7 {L}", **LAB)
        arr(Cx, x0 + k * slot + 2.35, mid, x0 + (k + 1) * slot - 0.1, mid)
    xg = x0 + 4 * slot
    slab(Cx, xg + 0.4, mid - 1.2, 0.22, 2.4, "#b9c3cc", d=0.12)
    Cx.text(xg + 0.55, yl, "GAP\n48", **LAB)
    arr(Cx, xg + 1.0, mid, xg + 2.0, mid)
    mlp(Cx, xg + 2.4, mid, ["FC\n32", "out"], C["violet"], n=(6, 1), gap=0.42, dx=2.6, ytxt=yl)

    # ---------- (d) multi-beat CNN
    Dx.set_ylim(-1.2, 4.5); tag(Dx, "d")
    mid = 1.3
    Dx.set_title("Multi-beat CNN: 13.9 k parameters (one estimate from all beats of a reading)", loc="left")
    for q in range(5):
        Dx.plot(0.3 + tt * 1.6 + q * 0.15, 0.3 + q * 0.5 + wave * 0.55, color=C["blue"], lw=0.9, alpha=0.45 + 0.11 * q)
    Dx.text(1.4, yl, "K \u2264 120 beats\nK \u00d7 100", **LAB)
    arr(Dx, 2.6, mid, 3.8, mid)
    box(Dx, 3.9, mid - 0.9, 4.2, 1.8, "Shared TinyCNN\nencoder", C["blue"], fs=6.2)
    arr(Dx, 8.2, mid, 9.4, mid)
    for q in range(5):
        Dx.add_patch(Rectangle((9.6 + q * 0.08, 0.15 + q * 0.45), 1.5, 0.3, fc=C["violet"] + "44", ec=C["violet"], lw=0.6))
    Dx.text(10.5, yl, "K \u00d7 48", **LAB)
    arr(Dx, 11.6, mid, 12.8, mid)
    box(Dx, 12.9, mid - 0.9, 3.4, 1.8, "Mean + SD\nover beats", "#7f8c99", fs=6.2)
    Dx.text(14.6, yl, "96", **LAB)
    arr(Dx, 16.4, mid, 17.6, mid)
    mlp(Dx, 18.2, mid, ["FC\n32", "out"], C["violet"], n=(6, 1), gap=0.42, dx=2.6, ytxt=yl)
    Dx.text(18.2, 3.95, "+ age, sex, BMI (optional)", ha="center", va="center", fontsize=6.0, color=C["violet"])
    arr(Dx, 18.2, 3.7, 18.2, 2.55, col=C["violet"])

    # ---------- (e) device model
    E.set_ylim(-1.9, 4.2); tag(E, "e")
    E.set_title("Device model on the ESP32-S3: gradient-boosted trees in plain C (8,700 nodes per model)", loc="left")
    tw = np.linspace(0, 8, 800); ww = np.sin(2 * np.pi * 1.2 * tw) ** 3 + 0.2 * np.sin(2 * np.pi * 2.4 * tw + 1)
    E.plot(0.3 + tw / 8 * 2.4, mid + ww * 0.45, color=C["blue"], lw=0.8)
    E.text(1.5, yl, "60 s PPG\n6,000 samples", **LAB)
    arr(E, 2.85, mid, 3.35, mid)
    for k, lab in enumerate(["Band-pass\nfilter", "Peaks +\ntemplate", "15 beat\nfeatures"]):
        box(E, 3.45 + k * 3.0, mid - 0.95, 2.4, 1.9, lab, C["aqua"], fs=6.2)
        arr(E, 5.9 + k * 3.0, mid, 6.4 + k * 3.0, mid)
    box(E, 12.45, mid - 0.95, 2.6, 1.9, "19 inputs\n(36 if\ncalibrated)", C["violet"], fs=6.0)
    E.text(13.75, 3.95, "+ age, sex, BMI, diabetes (+ calibration value, minutes since it, feature changes)",
           ha="center", va="center", fontsize=6.0, color=C["violet"])
    arr(E, 13.75, 3.7, 13.75, mid + 1.0, col=C["violet"])
    arr(E, 15.1, mid, 15.6, mid)

    def tree(x0_, y0_, s_=1.0):
        pts = [(0, 0), (-0.6, -0.65), (0.6, -0.65), (-0.9, -1.3), (-0.3, -1.3), (0.3, -1.3), (0.9, -1.3)]
        for a_, b_ in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
            E.plot([x0_ + pts[a_][0], x0_ + pts[b_][0]], [y0_ + pts[a_][1], y0_ + pts[b_][1]], color=INK2, lw=0.6)
        for q, (px, py) in enumerate(pts):
            E.add_patch(plt.Circle((x0_ + px, y0_ + py), 0.14, fc=C["yellow"] if q >= 3 else "white",
                                   ec=INK2, lw=0.6, zorder=3))
    for xx in (16.8, 19.0, 22.6):
        tree(xx, mid + 0.65)
    E.text(20.8, mid, "\u2026", ha="center", va="center", fontsize=12)
    E.text(19.7, yl, "300 trees, \u2264 15 leaves each", **LAB)
    arr(E, 23.7, mid, 24.3, mid)
    E.add_patch(plt.Circle((24.75, mid), 0.4, fc="white", ec=INK, lw=1.0))
    E.text(24.75, mid, "\u03a3", ha="center", va="center", fontsize=9, color=INK)
    arr(E, 25.2, mid, 25.7, mid)
    E.text(27.9, mid, "glucose\n(or change since\ncalibration)", ha="center", va="center", fontsize=6.0)
    save(fig, "fig_arch")


if __name__ == "__main__":
    fig_arch()
    print("arch ok")


# ---------------------------------------------------------------- Figure: Bland-Altman
def fig_ba():
    runs = [("resnet34_beat_subject.csv", "ResNet34-1D, person split", C["blue"]),
            ("ms_free_t2___ppg.csv", "Clinical + time + PPG features", C["violet"]),
            ("ms_calib_c3___monitor___drugs_change.csv", "Calibrated, all signals", C["aqua"])]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=True)
    for ax, (f, title, col) in zip(axs, runs):
        d = pd.read_csv(D + f)
        ref, p = d.y.values, d.pred.values
        m, diff = (ref + p) / 2, p - ref
        bias, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(m, diff, s=3, color=col, alpha=0.3, lw=0, rasterized=True)
        for v, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
            ax.axhline(v, color=INK, lw=0.9, ls=ls)
        ax.text(318, bias + 1.96 * sd + 4, f"+1.96 SD: {bias + 1.96 * sd:.0f}", ha="right", va="bottom", fontsize=6.3)
        ax.text(318, bias - 1.96 * sd - 4, f"−1.96 SD: {bias - 1.96 * sd:.0f}", ha="right", va="top", fontsize=6.3)
        ax.text(318, bias + 4, f"bias {bias:+.1f}", ha="right", va="bottom", fontsize=6.3)
        # trend of the difference with the reference value (regression to the mean)
        b1, b0 = np.polyfit(ref, diff, 1)
        ax.text(62, -235, f"difference vs reference: slope {b1:.2f}", fontsize=6.1, color=INK2)
        ax.set_xlim(55, 320); ax.set_ylim(-250, 150); ax.grid(True)
        ax.set_xlabel("Mean of estimate and lab (mg/dL)"); ax.set_title(title, fontsize=7.8)
    axs[0].set_ylabel("Estimate − lab (mg/dL)")
    for ax, s_ in zip(axs, "abc"):
        tag(ax, s_)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_ba")


if __name__ == "__main__":
    fig_ba()
    print("ba ok")


# ---------------------------------------------------------------- Figure: participant flow
def fig_flow():
    """CONSORT-style participant flow: VitalDB development cohort (left) and MUST external test (right)."""
    from matplotlib.patches import FancyBboxPatch
    W, H = 710, 420                                  # 1 unit = 0.01 inch
    fig = plt.figure(figsize=(W / 100, H / 100))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    GREYE, GREYF, ARW = "#8a8984", "#f4f3f0", "#6b6a66"

    def box(x, y, w, h, title, sub=None, fc="white", ec=GREYE, lw=0.8, bold=False, tc=INK, fs=7.0, sfs=6.2):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=5", fc=fc, ec=ec, lw=lw))
        if sub:
            ax.text(x + w / 2, y + 6.5, title, ha="center", va="center", fontsize=fs, color=tc,
                    fontweight="bold" if bold else "normal")
            ax.text(x + w / 2, y - 8, sub, ha="center", va="center", fontsize=sfs, color=INK2)
        else:
            ax.text(x + w / 2, y, title, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3,
                    fontweight="bold" if bold else "normal")

    def down(x, y0, y1):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=ARW, lw=0.9, mutation_scale=8, shrinkA=0, shrinkB=0))

    def side(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARW, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))

    # column headers
    ax.text(62, 408, "a   Development cohort: VitalDB", fontsize=8.5, fontweight="bold", va="top", color=INK)
    ax.text(538, 408, "b   External test: MUST", fontsize=8.5, fontweight="bold", va="top", color=INK)
    ax.plot([62, 512], [390, 390], color=GRID, lw=0.8); ax.plot([538, 708], [390, 390], color=GRID, lw=0.8)

    # stage labels (left gutter)
    ys = [356, 278, 200, 122, 40]
    for y, s in zip(ys, ["Source", "Glucose\nlabels", "PPG\nwindow", "Beat\nquality", "Evaluation"]):
        ax.text(2, y, s, fontsize=6.4, color=INK2, va="center", ha="left", fontweight="bold", linespacing=1.15)

    # main VitalDB column
    MX, MW, MH = 62, 270, 46
    cx = MX + MW / 2
    main = [("VitalDB: 6,388 surgical cases, 6,090 patients", "all cases; the \u20186,388 subjects\u2019 of the original study"),
            ("5,091 cases, 4,818 patients", "35,358 laboratory glucose values"),
            ("3,085 cases with a complete \u00b18-min PPG window", "7,289 windows (< 50% missing samples)"),
            ("Analysed: 7,224 readings", "3,078 cases, 2,991 patients; glucose 120.7 \u00b1 35.7 mg/dL")]
    for k, ((t, sub), y) in enumerate(zip(main, ys[:4])):
        last = k == 3
        box(MX, y, MW, MH, t, sub, fc=C["blue"] + "1c" if last else GREYF, ec=C["blue"] if last else GREYE,
            lw=1.1 if last else 0.8, bold=k in (0, 3))
        down(cx, y - MH / 2, ys[k + 1] + (MH / 2 if k < 3 else 30))

    # exclusions
    EX, EW = 372, 140
    excl = ["1,297 cases without\na glucose lab",
            "2,006 cases: no lab with a full\n\u00b18-min PPG window around it",
            "65 windows with\n< 10 accepted beats"]
    for k, t in enumerate(excl):
        ym = (ys[k] + ys[k + 1]) / 2
        ax.plot([cx, EX - 18], [ym, ym], color=ARW, lw=0.8); side(EX - 18, EX, ym)
        ax.add_patch(FancyBboxPatch((EX, ym - 22), EW, 44, boxstyle="round,pad=0,rounding_size=5",
                                    fc="white", ec="#bdbcb6", lw=0.8))
        ax.text(EX + EW / 2, ym + 12.5, "Excluded", ha="center", va="center", fontsize=6.0, color=C["red"], fontweight="bold")
        ax.text(EX + EW / 2, ym - 5, t, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.2)

    # evaluation box (person-level CV)
    ax.add_patch(FancyBboxPatch((MX, ys[4] - 30), 450, 60, boxstyle="round,pad=0,rounding_size=5",
                                fc="white", ec=C["blue"], lw=0.8, ls=(0, (3, 2))))
    ax.text(MX + 225, ys[4] + 13, "Person-level 5-fold cross-validation (folds by patient ID, fixed seed)",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    fx0, fw, gap = MX + 14, 58, 6
    for i in range(5):
        x = fx0 + i * (fw + gap)
        ax.add_patch(FancyBboxPatch((x, ys[4] - 19), fw, 17, boxstyle="round,pad=0,rounding_size=3",
                                    fc=C["blue"] + ("55" if i == 0 else "18"), ec="none"))
        ax.text(x + fw / 2, ys[4] - 10.5, "test fold" if i == 0 else f"fold {i + 1}", ha="center", va="center",
                fontsize=5.9, color=INK if i == 0 else INK2, fontweight="bold" if i == 0 else "normal")
    ax.text(fx0 + 5 * (fw + gap) + 8, ys[4] - 10.5, "a patient\u2019s readings stay\nin one fold; each fold\nis the test set once",
            ha="left", va="center", fontsize=6.0, color=INK2, linespacing=1.2)

    # MUST column
    UX, UW = 538, 170
    ux = UX + UW / 2
    ux_box = [(356, "68 PPG signal files", "fingertip PPG, about 10 s each"),
              (239, "67 recordings, 23 people", "all-recordings test"),
              (122, "48 recordings, 22 people", "matched test, every model")]
    for k, (y, t, sub) in enumerate(ux_box):
        last = k == 2
        box(UX, y, UW, MH, t, sub, fc=C["aqua"] + ("1c" if last else "0e"), ec=C["aqua"], lw=1.1 if last else 0.8,
            bold=k == 0, sfs=6.0)
    for (y0, y1, t) in [(356, 239, "1 file without a\nglucose label"),
                         (239, 122, "19 with < 3\naccepted beats")]:
        down(ux, y0 - MH / 2, y1 + MH / 2)
        ym = (y0 + y1) / 2
        ax.text(ux + 8, ym + 12, "Excluded", fontsize=6.0, color=C["red"], fontweight="bold", va="center", ha="left")
        ax.text(ux + 8, ym - 5, t, fontsize=6.0, color=INK2, va="center", ha="left", linespacing=1.2)
    ax.text(ux, 62, "Models trained on VitalDB only;\nMUST never used for training\nor model selection",
            ha="center", va="center", fontsize=6.0, color=INK2, linespacing=1.25, style="italic")
    save(fig, "fig_flow")


if __name__ == "__main__":
    fig_flow()
    print("flow ok")
