#!/usr/bin/env python3
"""
loop_boundary_energy: mutated vs unmutated, all (stem,loop) shapes pooled.

Source: ml/predictions_omniscient-panda-747.csv (all rows are loop-boundary
hairpins, in_loop_boundary==1). Groups by any_mutation (0/1), ignoring
loop_boundary_stem_length / loop_boundary_length.

Output: loop_boundary_energy_boxplot_pooled.pdf / .png,
        loop_boundary_energy_pooled_pvalues.tsv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu, ttest_ind, permutation_test

IN_CSV  = Path(__file__).parent / "predictions_omniscient-panda-747.csv"
OUT_DIR = Path(__file__).parent

df = pd.read_csv(IN_CSV)
df = df[df["in_loop_boundary"] == 1]
print(f"Loaded {len(df)} loop-boundary hairpins")

mut = df.loc[df.any_mutation == 1, "loop_boundary_energy"].dropna().tolist()
unm = df.loc[df.any_mutation == 0, "loop_boundary_energy"].dropna().tolist()
print(f"  mutated:   n={len(mut)}  mean={np.mean(mut):.3f}  median={np.median(mut):.3f}")
print(f"  unmutated: n={len(unm)}  mean={np.mean(unm):.3f}  median={np.median(unm):.3f}")

def mean_diff(x, y):
    return np.mean(x) - np.mean(y)

mwu_p  = mannwhitneyu(mut, unm, alternative="two-sided").pvalue
tt_p   = ttest_ind(mut, unm, equal_var=False, alternative="two-sided").pvalue
perm_p = permutation_test(
    (np.array(mut), np.array(unm)), mean_diff,
    permutation_type="independent", alternative="two-sided",
    n_resamples=9999, random_state=42,
).pvalue

def fmt_p(p):
    return f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"

print(f"  Mann-Whitney {fmt_p(mwu_p)}   Welch t-test {fmt_p(tt_p)}   permutation {fmt_p(perm_p)}")

# ── plot ───────────────────────────────────────────────────────────────────
COLOR_MUT = "#D74050"
COLOR_UNM = "#3389C0"

fig, ax = plt.subplots(figsize=(4.5, 5.5))
groups = [mut, unm]
labels = [f"Mutated\n(n={len(mut)})", f"Unmutated\n(n={len(unm)})"]
colors = [COLOR_MUT, COLOR_UNM]

bp = ax.boxplot(
    groups, patch_artist=True, widths=0.45,
    medianprops=dict(color="black", linewidth=2),
    boxprops=dict(linewidth=1.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    showfliers=False,
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

rng = np.random.default_rng(42)
for xi, (gs, color) in enumerate(zip(groups, colors), start=1):
    jitter = rng.uniform(-0.14, 0.14, size=len(gs))
    ax.scatter(xi + jitter, gs, color=color, alpha=0.85, s=40, zorder=3,
               edgecolors="white", linewidths=0.4)
for xi, gs in enumerate(groups, start=1):
    ax.scatter(xi, np.mean(gs), marker="D", color="black", s=50, zorder=4,
               label="mean" if xi == 1 else None)

ax.set_xticks([1, 2])
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("loop_boundary_energy  (kcal/mol, ΔG)", fontsize=12)
ax.set_title("All (stem, loop) shapes pooled", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", fontsize=9, frameon=False)

annot = f"MWU {fmt_p(mwu_p)}\nWelch t {fmt_p(tt_p)}\nperm {fmt_p(perm_p)}"
ax.text(0.03, 0.03, annot, transform=ax.transAxes, fontsize=9,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(OUT_DIR / f"loop_boundary_energy_boxplot_pooled.{ext}", dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: loop_boundary_energy_boxplot_pooled.pdf / .png")

# ── stats table ────────────────────────────────────────────────────────────
out_tsv = OUT_DIR / "loop_boundary_energy_pooled_pvalues.tsv"
with open(out_tsv, "w") as f:
    f.write("\t".join(["n_mut", "n_unm", "mean_mut", "mean_unm", "mean_diff",
                        "median_mut", "median_unm", "mwu_p", "tt_p", "perm_p"]) + "\n")
    f.write("\t".join(str(v) for v in [
        len(mut), len(unm), np.mean(mut), np.mean(unm), np.mean(mut) - np.mean(unm),
        np.median(mut), np.median(unm), mwu_p, tt_p, perm_p,
    ]) + "\n")
print(f"Saved: {out_tsv.name}")
