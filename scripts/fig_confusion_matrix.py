"""
Generate publication-quality confusion matrix figures for SIGGRAPH poster.

Produces:
  - fig_cm_4way.{pdf,png}         original 4x4 confusion (LogReg, 5-fold CV)
  - fig_cm_3way_merged.{pdf,png}  3x3 view with T0 and T1 merged into a
    single "non-suggestive" class. NOTE: the 3x3 counts are AGGREGATED
    from the 4x4 matrix above (the 4-way model's predictions, regrouped).
    They are not the output of a classifier trained directly on the merged
    3-class split, which would score higher because it would not waste
    capacity distinguishing T0 from T1. For the camera-ready, regenerate
    this figure from a real `--merge-tier1-into-tier0` run.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 4-way counts (LogReg, 5-fold CV, 1,075 per class) -- paper-verified, bit-exact.
COUNTS_4WAY = np.array([
    [711, 126, 107, 131],  # Tier 0
    [182, 489, 260, 144],  # Tier 1
    [130, 215, 572, 158],  # Tier 2
    [211,  96,  77, 691],  # Tier 3
])

LABELS_4WAY = ["Tier 0\n(Normal)", "Tier 1\n(Artistic)",
               "Tier 2\n(Suggestive)", "Tier 3\n(Explicit)"]
HIGHLIGHT_4WAY = [(1, 2), (2, 1)]  # T1<->T2 confusion pair

# Aggregate T0+T1 rows and columns into a single non-suggestive class.
COUNTS_3WAY = np.array([
    [COUNTS_4WAY[0:2, 0:2].sum(), COUNTS_4WAY[0:2, 2].sum(), COUNTS_4WAY[0:2, 3].sum()],
    [COUNTS_4WAY[2,   0:2].sum(), COUNTS_4WAY[2, 2],         COUNTS_4WAY[2, 3]],
    [COUNTS_4WAY[3,   0:2].sum(), COUNTS_4WAY[3, 2],         COUNTS_4WAY[3, 3]],
])

LABELS_3WAY = ["Tier 0+1\n(Non-suggestive)",
               "Tier 2\n(Suggestive)",
               "Tier 3\n(Explicit)"]
HIGHLIGHT_3WAY = []

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})


def plot_cm(counts, labels, highlights, out_pdf, out_png, figsize=(5.4, 4.6)):
    n = counts.shape[0]
    row_sums = counts.sum(axis=1, keepdims=True)
    prop = counts / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(prop, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=0, linespacing=0.95)
    ax.set_yticklabels(labels, rotation=0, linespacing=0.95)
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold", labelpad=6)
    ax.set_ylabel("Actual", fontsize=10, fontweight="bold", labelpad=6)
    ax.tick_params(axis="x", which="both", length=0, pad=3)
    ax.tick_params(axis="y", which="both", length=0, pad=3)

    for i in range(n):
        for j in range(n):
            count = counts[i, j]
            pct = prop[i, j] * 100
            text_color = "white" if prop[i, j] > 0.5 else "#222222"
            weight = "bold" if i == j else "normal"
            ax.text(j, i - 0.12, f"{count}", ha="center", va="center",
                    color=text_color, fontsize=9.5, fontweight=weight)
            ax.text(j, i + 0.18, f"({pct:.0f}%)", ha="center", va="center",
                    color=text_color, fontsize=8, fontweight=weight)

    for (ri, ci) in highlights:
        rect = Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                         fill=False, edgecolor="#C44E52", linewidth=2.0, zorder=5)
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    acc = np.trace(counts) / counts.sum()
    print(f"[+] Saved {out_pdf}")
    print(f"[+] Saved {out_png}")
    print(f"    Diagonal accuracy: {acc:.4f}")


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plot_cm(COUNTS_4WAY, LABELS_4WAY, HIGHLIGHT_4WAY,
        os.path.join(OUT_DIR, "fig_cm_4way.pdf"),
        os.path.join(OUT_DIR, "fig_cm_4way.png"))

plot_cm(COUNTS_3WAY, LABELS_3WAY, HIGHLIGHT_3WAY,
        os.path.join(OUT_DIR, "fig_cm_3way_merged.pdf"),
        os.path.join(OUT_DIR, "fig_cm_3way_merged.png"))
