#!/usr/bin/env python3
"""
Generate bar charts for Figure 2c.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent.parent.parent / "binomial_test/EMBOSS/binomial_dataset2_loople0_indTrue_emboss.txt"
#RESULTS = Path(__file__).parent.parent.parent / "binomial_test/PalndromeAnalyzer/binomial_dataset2_loople0_indTrue_PA.txt"
#RESULTS = Path(__file__).parent.parent.parent / "binomial_test/EMBOSS/binomial_dataset3_loople0_indTrue_emboss.txt"
OUT_DIR = Path(__file__).parent

# parse results.txt
data = {}
with open(RESULTS) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("structure") or line.startswith("---"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        stype, seltype = parts[0], parts[1]
        pvalue = float(parts[6])
        log10p = float(parts[7])
        data.setdefault(stype, []).append((seltype, pvalue, log10p))

# plot settings
SIGNIFICANCE = -np.log10(0.05)   # dashed line at α = 0.05

COLORS = {
    "most_stable": "#4C72B0",
    "greedy":      "#DD8452",
    "max_cov":     "#55A868",
    "min_cov":     "#C44E52",
    "all":         "#8172B2",
}
LABELS = {
    "most_stable": "Most stable",
    "greedy":      "Greedy",
    "max_cov":     "Max coverage",
    "min_cov":     "Min coverage",
    "all":         "All",
}
TITLE_MAP = {
    "hairpin": "Full hairpin",
    "spacer":  "Loop",
    "tc_end":  "TC at 3' loop end",
    "c_end":   "C at 3' loop end",
}

for stype, rows in data.items():
    sel_types = [r[0] for r in rows]
    pvalues   = [r[1] for r in rows]
    log10ps   = [r[2] for r in rows]
    colors    = [COLORS[s] for s in sel_types]
    x         = np.arange(len(sel_types))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    bars = ax.bar(x, log10ps, color=colors, width=0.55, edgecolor="white",
                  linewidth=0.8, zorder=3)

    # significance threshold line
    ax.axhline(SIGNIFICANCE, color="#D62728", linestyle="--", linewidth=1.2,
               zorder=4, label=f"α = 0.05  (–log₁₀p = {SIGNIFICANCE:.2f})")

    # p-value labels on bars
    for bar, pval in zip(bars, pvalues):
        label = f"{pval:.4f}" if pval >= 1e-3 else f"{pval:.2e}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                label,
                ha="center", va="bottom", fontsize=8.5, color="black")

    #ax.set_xticks(x)
    #ax.set_xticklabels([LABELS[s] for s in sel_types], rotation=25, ha="right",fontsize=10)
    ax.set_xticklabels([])
    #ax.set_ylabel("–log₁₀(p-value)", fontsize=11)
    #ax.set_title(TITLE_MAP.get(stype, stype), fontsize=13, fontweight="bold", pad=8)
    ax.set_ylim(0, max(log10ps) * 1.22 + 0.3)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    #ax.legend(fontsize=8.5, framealpha=0.7)

    fig.tight_layout()
    #out = OUT_DIR / f"barchart_{stype}_dataset3_loople0_indTrue_emboss.png"
    out = OUT_DIR / f"barchart_{stype}_dataset2_loople0_indTrue_emboss.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

print("Done.")
