"""
Test 8: Heatmaps of -log10(p-value) for structure_type=ct_end,
hairpin_selection_type=most_stable across a grid of (stem_max, loop_length).

4 panels — 2x2 grid of (stem_min, mismatches) combinations:
  Panel 1 (top-left):      stem_min=6, mismatches=1  [default]
  Panel 2 (top-right):    stem_min=6, mismatches=0
  Panel 3 (bottom-left):   stem_min=5, mismatches=0
  Panel 4 (bottom-right): stem_min=4, mismatches=0

X axis: loop_length in [2, 5, 8, 11, 14, 17, 20]
Y axis: stem_max    in [stem_min .. stem_min+6]
"""

import sys
import os
import io
import contextlib
from pathlib import Path

# paths
BINOMIAL_DIR = Path(__file__).parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))

# Must chdir before importing load/hairpin_groups — they open config.yaml and
# nn_coefs/ via relative paths.
os.chdir(BINOMIAL_DIR)

# import binomial modules
# import load first: sets genome, mutations_list, mut_cnt as module globals.
# hairpin_groups does "from load import *" so it picks up those globals.
import load
from emboss import get_palindrome
from hairpin_groups import calculate_pval_groups
from all_hairpins import calculate_pval_all

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# grid parameters
LOOP_VALUES = [2, 5, 8, 11, 14, 17, 20]

def stem_max_values(stem_min):
    """Y-axis range: stem_min .. stem_min+6 (7 values)."""
    return list(range(stem_min, stem_min + 7))
STRUCTURE_TYPE  = "c_end"
SELECTION_TYPE  = "greedy"

# 5 panels: (stem_min, mismatches)
PANELS = [
    (6, 1, "stem_min=6, mm=1"),
    (6, 0, "stem_min=6, mm=0"),
    (5, 0, "stem_min=5, mm=0"),
    (4, 0, "stem_min=4, mm=0"),
]

# compute heatmap for one (stem_min, mismatches) combination
def compute_heatmap(stem_min, num_mismatches, label):
    """Return (n_rows x n_cols) array of -log10(p-value)."""
    s_max_values = stem_max_values(stem_min)
    print(f"\n{'='*60}")
    print(f"Panel: {label}\
    ")
    print(f"{'stem_max':>8}  {'loop':>5}  {'-log10p':>10}")
    print("-" * 32)
    matrix = np.full((len(s_max_values), len(LOOP_VALUES)), np.nan)
    for r, stem_max in enumerate(s_max_values):
        for c, loop_len in enumerate(LOOP_VALUES):
            # run EMBOSS + create PA file if needed
            pa_file = get_palindrome(stem_min, stem_max, loop_len, num_mismatches)
            hairpins = load.load_hairpins(str(pa_file))
            if not hairpins:
                print(f"{stem_max:>8}  {loop_len:>5}  {'no hairpins':>10}")
                matrix[r, c] = 0.0
                continue
            # suppress verbose output from calculate_pval_groups
            with contextlib.redirect_stdout(io.StringIO()):
                if SELECTION_TYPE == "all":
                    log10p, pvalue, total_real_hits, targets_p, targets_in_struct, all_targets = calculate_pval_all(hairpins, STRUCTURE_TYPE)                
                else:
                    log10p, pvalue, total_real_hits, targets_p, targets_in_struct, all_targets = calculate_pval_groups(hairpins, SELECTION_TYPE, STRUCTURE_TYPE)
            print(f"{stem_max:>8}  {loop_len:>5}  {log10p:>10.4f}")
            matrix[r, c] = log10p
    return matrix

# run all panels
matrices    = []
y_labels_per_panel = []
for stem_min, mm, label in PANELS:
    mat = compute_heatmap(stem_min, mm, label)
    matrices.append(mat)
    y_labels_per_panel.append([str(v) for v in stem_max_values(stem_min)])

# plot
SIG_THRESH = 1.3   # -log10(0.05)

vmin = 0.0
vmax = max(np.nanmax(m) for m in matrices)
vmax = math.ceil(vmax) if not math.isinf(vmax) else 50

# Custom colormap: white → medium grey for non-significant (0 → 1.3),
# yellow → orange → dark red for significant (1.3 → vmax).
n_nonsig = 128   # must equal n_sig: TwoSlopeNorm maps vcenter to 0.5 in colormap
n_sig    = 128
colors_nonsig = plt.cm.Greys(np.linspace(0.0, 0.55, n_nonsig))  # white → grey
colors_sig    = plt.cm.YlOrRd(np.linspace(0.2,  1.0, n_sig))    # yellow → dark red
cmap_custom   = LinearSegmentedColormap.from_list(
    "sig_cmap", np.vstack([colors_nonsig, colors_sig])
)
norm = TwoSlopeNorm(vmin=vmin, vcenter=SIG_THRESH, vmax=vmax)

x_labels = [str(v) for v in LOOP_VALUES]

out_dir = Path(__file__).parent

def draw_panel(ax, mat, y_labels, label):
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap=cmap_custom, norm=norm)
    ax.set_xticks(range(len(LOOP_VALUES)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Maximum loop length (nt)", fontsize=10)
    ax.set_ylabel("Maximum stem length (bp)", fontsize=10)
    #ax.set_title(label, fontsize=11, fontweight="bold")
    for r in range(len(y_labels)):
        for c in range(len(LOOP_VALUES)):
            val = mat[r, c]
            if not np.isnan(val):
                text_color = "white" if val > SIG_THRESH + (vmax - SIG_THRESH) * 0.5 else "black"
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)
    cbar = ax.get_figure().colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("−log₁₀(p-value)", fontsize=9, loc='top')
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.axhline(y=SIG_THRESH, color="black", linewidth=1.2, linestyle="--")
    cbar.ax.text(1.5, SIG_THRESH, "p=0.05", va="center", fontsize=7.5, color="black")

# combined 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, mat, (stem_min, mm, label), y_labels in zip(axes, matrices, PANELS, y_labels_per_panel):
    draw_panel(ax, mat, y_labels, label)

#fig.suptitle(f"Binomial test significance ({STRUCTURE_TYPE}, {SELECTION_TYPE})\n"
#             "Color: −log₁₀(p-value)",
#             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(out_dir / "heatmap_pvalue.pdf", dpi=150, bbox_inches="tight")
plt.savefig(out_dir / "heatmap_pvalue.png", dpi=600, bbox_inches="tight")
plt.close()
print("Plot saved: heatmap_pvalue.pdf / .png")

# individual figures
panel_filenames = [
    "heatmap_stem6_mm1",
    "heatmap_stem6_mm0",
    "heatmap_stem5_mm0",
    "heatmap_stem4_mm0",
]

for mat, (stem_min, mm, label), y_labels, fname in zip(matrices, PANELS, y_labels_per_panel, panel_filenames):
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_panel(ax, mat, y_labels, label)
    #fig.suptitle(f"Binomial test significance ({STRUCTURE_TYPE}, {SELECTION_TYPE})\n"
    #             "Color: −log₁₀(p-value)",
    #             fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / f"{fname}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / f"{fname}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {fname}.pdf / .png")
