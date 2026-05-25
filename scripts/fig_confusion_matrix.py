"""
Generate publication-quality 4-way confusion matrix figure for SIGGRAPH poster.

Produces fig_cm_4way.pdf and fig_cm_4way.png at ~4x4 inches.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Raw counts (LogReg, 5-fold CV, 1,075 per class)
COUNTS = np.array([
    [711, 126, 107, 131],  # Tier 0
    [182, 489, 260, 144],  # Tier 1
    [130, 215, 572, 158],  # Tier 2
    [211,  96,  77, 691],  # Tier 3
])

# Row-normalized proportions
row_sums = COUNTS.sum(axis=1, keepdims=True)
PROP = COUNTS / row_sums

LABELS = ["Tier 0\n(Normal)", "Tier 1\n(Artistic)",
          "Tier 2\n(Suggestive)", "Tier 3\n(Explicit)"]

# Set font sizes for ~4x4 inch figure at ~0.45 column width
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

fig, ax = plt.subplots(figsize=(5.4, 4.6))

# Heatmap with Blues colormap, row-normalized values drive color
im = ax.imshow(PROP, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

# Ticks and labels
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(LABELS, rotation=0, linespacing=0.95)
ax.set_yticklabels(LABELS, rotation=0, linespacing=0.95)
ax.set_xlabel("Predicted", fontsize=10, fontweight="bold", labelpad=6)
ax.set_ylabel("Actual", fontsize=10, fontweight="bold", labelpad=6)

# Move x-axis label to bottom (default) and ensure tick marks are outside
ax.tick_params(axis="x", which="both", length=0, pad=3)
ax.tick_params(axis="y", which="both", length=0, pad=3)

# Cell annotations: count on top, (pct%) below; diagonal bolded
for i in range(4):
    for j in range(4):
        count = COUNTS[i, j]
        pct = PROP[i, j] * 100
        # Pick readable text color
        text_color = "white" if PROP[i, j] > 0.5 else "#222222"
        is_diagonal = (i == j)
        weight = "bold" if is_diagonal else "normal"
        ax.text(
            j, i - 0.12, f"{count}",
            ha="center", va="center",
            color=text_color, fontsize=9.5, fontweight=weight,
        )
        ax.text(
            j, i + 0.18, f"({pct:.0f}%)",
            ha="center", va="center",
            color=text_color, fontsize=8, fontweight=weight,
        )

# Highlight the T1<->T2 confusion pair with a subtle red border
for (ri, ci) in [(1, 2), (2, 1)]:  # T1->T2 and T2->T1
    rect = Rectangle(
        (ci - 0.5, ri - 0.5), 1, 1,
        fill=False, edgecolor="#C44E52", linewidth=2.0, zorder=5,
    )
    ax.add_patch(rect)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized proportion", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Clean up spines
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(OUT_DIR, "fig_cm_4way.pdf")
out_png = os.path.join(OUT_DIR, "fig_cm_4way.png")

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"[+] Saved {out_pdf}")
print(f"[+] Saved {out_png}")
