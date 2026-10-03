"""
Layout: 5-row x 4-column grid of heatmaps.
  rows    -> selection_type: most_stable, greedy, max_cov, min_cov, all
  columns -> structure_type: hairpin, spacer, ct_end, c_end

Each heatmap:
  X axis: loop length in [2..10]
  Y axis: stem length in [4..12]
  Cell color: -log10(p-value)
  Cell text:  "hits/targets" on line 1, -log10(p) on line 2
"""

import sys
import os
import io
import contextlib
import math
import re
import pickle
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# paths
BINOMIAL_DIR = Path(__file__).parent.parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))
os.chdir(BINOMIAL_DIR)

import load
from emboss import get_palindrome
from all_hairpins import calculate_pval_all
from hairpin_groups import calculate_pval_groups
from hairpin_groups import greedy_choose, most_stable, min_coverage
from functions import find_groups, max_coverage, max_coverage_spacer

# command-line arguments
parser = argparse.ArgumentParser(description="Test 16: select-first-then-filter heatmaps.")
parser.add_argument("--mm", type=int, default=0,
                    help="Number of mismatches allowed in stem (default: 0)")
args = parser.parse_args()
MM = args.mm
print(f"Running with mismatches = {MM}")

# broad detection parameters
DET_STEM_MIN = 4
DET_STEM_MAX = 12
DET_GAPLIMIT = 20

# grid parameters
LOOP_VALUES     = [2, 3, 4, 5, 6, 7, 8, 9, 10]
STEM_VALUES     = [4, 5, 6, 7, 8, 9, 10, 11, 12]

STRUCTURE_TYPES = ["hairpin", "spacer", "tc_end", "c_end"]
SELECTION_TYPES = ["most_stable", "greedy", "max_cov", "min_cov", "all"]

# non-extendable-stem filter
_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
_genome_seq = load.genome.sequence

def is_extendable(h):
    """True if flanking nucleotides outside the stem are Watson-Crick
    complementary, i.e. the stem could be extended by one more base pair."""
    if h.start == 0 or h.end >= len(_genome_seq) - 1:
        return False
    left  = _genome_seq[h.start - 1].upper()
    right = _genome_seq[h.end + 1].upper()
    return _COMPLEMENT.get(left) == right

_GENOME_DENOM = {
    "hairpin": len(re.findall("TC", _genome_seq)) + len(re.findall("GA", _genome_seq)),
    "spacer":  len(re.findall("TC", _genome_seq)) + len(re.findall("GA", _genome_seq)),
    "tc_end":  len(re.findall("TC", _genome_seq)) + len(re.findall("GA", _genome_seq)),
    "c_end":   len(re.findall("C",  _genome_seq)) + len(re.findall("G",  _genome_seq)),
}
_MUT_CNT = {
    "hairpin": load.mut_cnt_APOBEC,
    "spacer": load.mut_cnt_APOBEC,
    "tc_end": load.mut_cnt_APOBEC,
    "c_end": load.mut_cnt_APOBEC_ext
} 

# hairpin detection
print(f"Detecting hairpins broadly "
      f"(stem={DET_STEM_MIN}-{DET_STEM_MAX}, gaplimit={DET_GAPLIMIT}, mm={MM})...")
pa_file = get_palindrome(DET_STEM_MIN, DET_STEM_MAX, DET_GAPLIMIT, MM)
all_hairpins = load.load_hairpins(str(pa_file))
print(f"  total hairpins found: {len(all_hairpins)}")

broad_pool = [h for h in all_hairpins if not is_extendable(h)]
print(f"  non-extendable hairpins: {len(broad_pool)}")

# selection fixed length stem/loop hairpins
def select_pool(sel_type, struct_type, pool):
    """Run the selection method on the full broad pool. Returns the chosen
    non-overlapping hairpins (or the full pool if sel_type == 'all')."""
    if sel_type == "all":
        return list(pool)
    groups = find_groups(list(pool))
    chosen = []
    for group in groups:
        if struct_type == "hairpin":
            if sel_type == "greedy":
                choice = greedy_choose(group)
            elif sel_type == "max_cov":
                choice = max_coverage(group)
            elif sel_type == "most_stable":
                choice = most_stable(group)
            elif sel_type == "min_cov":
                choice = min_coverage(group)
        else:  # spacer, ct_end, c_end
            if sel_type == "greedy":
                choice = greedy_choose(group)
            elif sel_type == "max_cov":
                choice = max_coverage_spacer(group)
            elif sel_type == "most_stable":
                choice = most_stable(group)
            elif sel_type == "min_cov":
                choice = min_coverage(group)
        chosen += choice
    chosen.sort(key=lambda x: x.start)
    return chosen

print("\nApplying selection on the broad pool...")
selected_pools = {}
for sel_type in SELECTION_TYPES:
    for struct_type in STRUCTURE_TYPES:
        with contextlib.redirect_stdout(io.StringIO()):
            sel = select_pool(sel_type, struct_type, broad_pool)
        selected_pools[(sel_type, struct_type)] = sel
        print(f"  [{sel_type}/{struct_type}]: {len(sel)} hairpins selected")

print("\nComputing per-cell p-values...")
# key: (sel_type, struct_type, stem_len, loop_len)
# val: (log10p, hits, targets_in_struct, targets_p, pvalue)
results = {}

for sel_type in SELECTION_TYPES:
    for struct_type in STRUCTURE_TYPES:
        pool = selected_pools[(sel_type, struct_type)]
        print(f"\n[{sel_type} / {struct_type}]")
        for stem_len in STEM_VALUES:
            for loop_len in LOOP_VALUES:
                cell = [h for h in pool
                        if h.stem_length == stem_len and h.spacer_length == loop_len]
                if not cell:
                    results[(sel_type, struct_type, stem_len, loop_len)] = (0.0, 0, 0, 0.0, 1.0)
                    continue
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    if sel_type == "all":
                        log10p, pvalue, hits, targets_p, targets_in_struct, all_targets = calculate_pval_all(cell, struct_type)
                    else:
                        log10p, pvalue, hits, targets_p, targets_in_struct, all_targets = calculate_pval_groups(cell, sel_type, struct_type)
                captured = buf.getvalue()
                results[(sel_type, struct_type, stem_len, loop_len)] = (
                    log10p, hits, targets_in_struct, targets_p, pvalue
                )
                print(f"  stem={stem_len} loop={loop_len}: "
                      f"hits={hits}, tgt={targets_in_struct}, "
                      f"p={pvalue:.5f}, -log10p={log10p:.3f}")

# panel-level combined stats
from scipy.stats import binomtest as _binomtest
panel_stats = {}
for _sel in SELECTION_TYPES:
    for _str in STRUCTURE_TYPES:
        _non_empty = [
            results[(_sel, _str, sl, ll)]
            for sl in STEM_VALUES for ll in LOOP_VALUES
            if results[(_sel, _str, sl, ll)][2] > 0
        ]
        _hits       = sum(v[1] for v in _non_empty)
        _targets    = sum(v[2] for v in _non_empty)
        _gdenom     = _GENOME_DENOM[_str]
        _p0         = min(_targets / _gdenom, 1.0) if _gdenom > 0 else 0.0
        _k          = min(_hits, _MUT_CNT[_str])
        if _p0 > 0:
            _pv = _binomtest(_k, _MUT_CNT[_str], _p0, alternative='greater').pvalue
        else:
            _pv = 1.0
        _lp = -math.log10(_pv) if _pv > 0 else float('inf')
        panel_stats[(_sel, _str)] = (_hits, _targets, _p0, _lp)

# save results for standalone plotting
out_dir = Path(__file__).parent
with open(out_dir / f"results_mm{MM}.pkl", "wb") as f:
    pickle.dump({
        "results":         results,
        "panel_stats":     panel_stats,
        "LOOP_VALUES":     LOOP_VALUES,
        "STEM_VALUES":     STEM_VALUES,
        "STRUCTURE_TYPES": STRUCTURE_TYPES,
        "SELECTION_TYPES": SELECTION_TYPES,
        "DET_STEM_MIN":    DET_STEM_MIN,
        "DET_STEM_MAX":    DET_STEM_MAX,
        "DET_GAPLIMIT":    DET_GAPLIMIT,
    }, f)
print(f"\nResults saved to results_mm{MM}.pkl")

# list mutated hairpins for significant cells (p < 0.05)
from collections import Counter
SIG_P = 0.05
print(f"\nListing mutated hairpins for significant cells (p < {SIG_P})...")

_mut_list_full_APOBEC = load.mutations_list_APOBEC   # 0-based, with multiplicity
_mut_list_full_APOBEC_ext = load.mutations_list_APOBEC_ext
_mut_counts_APOBEC = Counter(_mut_list_full_APOBEC)
_mut_counts_APOBEC_ext = Counter(_mut_list_full_APOBEC_ext)
_mut_set_APOBEC = set(_mut_counts_APOBEC)
_mut_set_APOBEC_ext = set(_mut_counts_APOBEC_ext)
_mut_sorted_APOBEC = sorted(_mut_set_APOBEC)
_mut_sorted_APOBEC_ext = sorted(_mut_set_APOBEC_ext)

def hairpin_mutations(h, struct_type):
    """Return sorted list of mutation positions (0-based, unique) that hit
    this hairpin under struct_type (test16 definitions)."""
    hits = []
    if struct_type == "hairpin":
        hits = [m for m in _mut_sorted_APOBEC if h.start <= m <= h.end]
    elif struct_type == "spacer":
        l, r = h.spacer_index
        hits = [m for m in _mut_sorted_APOBEC if l <= m <= r]
    elif struct_type == "tc_end":
        if h.spacer_length < 2:
            return []
        l, r = h.spacer_index
        if r in _mut_set_APOBEC and _genome_seq[r-1:r+1].upper() == "TC":
            hits.append(r)
        if l in _mut_set_APOBEC and _genome_seq[l:l+2].upper() == "GA":
            hits.append(l)
    elif struct_type == "c_end":
        if h.spacer_length < 1:
            return []
        l, r = h.spacer_index
        if r in _mut_set_APOBEC_ext and _genome_seq[r].upper() == "C":
            hits.append(r)
        if l in _mut_set_APOBEC_ext and _genome_seq[l].upper() == "G":
            hits.append(l)
    return sorted(set(hits))

rows = []
for (sel_type, struct_type, stem_len, loop_len), (log10p, hits_cnt, tgt, tp, pvalue) in results.items():
    if pvalue >= SIG_P:
        continue
    pool = selected_pools[(sel_type, struct_type)]
    cell = [h for h in pool
            if h.stem_length == stem_len and h.spacer_length == loop_len]
    for h in cell:
        muts = hairpin_mutations(h, struct_type)
        if struct_type == "c_end":
            mut_sum = sum(_mut_counts_APOBEC_ext[m] for m in muts)
        else:
            mut_sum = sum(_mut_counts_APOBEC[m] for m in muts)
        if not muts:
            continue
        rows.append({
            "sel_type":              sel_type,
            "struct_type":           struct_type,
            "stem_len":              stem_len,
            "loop_len":              loop_len,
            "cell_pvalue":           f"{pvalue:.6g}",
            "cell_log10p":           f"{log10p:.4f}",
            "hairpin_start_0b":      h.start,
            "hairpin_end_0b":        h.end,
            "stem_length":           h.stem_length,
            "spacer_length":         h.spacer_length,
            "score":                 h.score,
            "sequence":              h.sequence,
            "n_mut_positions":       len(muts),
            "n_mut_observations":    mut_sum,
            "mutation_positions_0b": ",".join(str(m) for m in muts),
        })

out_tsv = out_dir / f"mutated_hairpins_mm{MM}.tsv"
with open(out_tsv, "w") as f:
    if rows:
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    else:
        f.write("# No significant cells with mutated hairpins (p < 0.05)\n")
print(f"Saved: {out_tsv.name}  ({len(rows)} mutated hairpins across significant cells)")

# colormap (shared across all panels)
SIG_THRESH = 1.3  # -log10(0.05)

finite_vals = [v[0] for v in results.values() if not math.isinf(v[0])]
vmin = 0.0
vmax = math.ceil(max(finite_vals)) if finite_vals else 5

n_nonsig      = 128
n_sig         = 128
colors_nonsig = plt.cm.Greys(np.linspace(0.0, 0.55, n_nonsig))
colors_sig    = plt.cm.YlOrRd(np.linspace(0.2,  1.0, n_sig))
cmap_custom   = LinearSegmentedColormap.from_list(
    "sig_cmap", np.vstack([colors_nonsig, colors_sig])
)
norm = TwoSlopeNorm(vmin=vmin, vcenter=SIG_THRESH, vmax=vmax)

x_labels = [str(v) for v in LOOP_VALUES]
y_labels = [str(v) for v in STEM_VALUES]

struct_labels = {
    "hairpin": "Whole structure",
    "spacer": "Hairpin loop",
    "tc_end": "TC at the 3' loop end",
    "c_end": "C at the 3' loop end"
}
selection_labels = {
    "most_stable": "Most stable structure",
    "greedy": "Greedy selection",
    "max_cov": "Maximum coverage",
    "min_cov": "Minimum coverage",
    "all": "All structures"
}

# draw one heatmap into an axes
def draw_panel(ax, fig, sel_type, struct_type, fontsize_tick=7, fontsize_ann=5):
    n_rows = len(STEM_VALUES)
    n_cols = len(LOOP_VALUES)

    mat = np.zeros((n_rows, n_cols))
    for r, stem_len in enumerate(STEM_VALUES):
        for c, loop_len in enumerate(LOOP_VALUES):
            log10p = results[(sel_type, struct_type, stem_len, loop_len)][0]
            mat[r, c] = min(log10p, vmax)

    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap_custom, norm=norm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels, fontsize=fontsize_tick)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=fontsize_tick)
    ax.set_xlabel("Loop length (nt)", fontsize=fontsize_tick)
    ax.set_ylabel("Stem length (bp)", fontsize=fontsize_tick)
    ph, pt, pp0, plp = panel_stats[(sel_type, struct_type)]
    plp_str = f"{plp:.2f}" if not math.isinf(plp) else "inf"
    ax.set_title(
        f"{struct_labels[struct_type]} / {selection_labels[sel_type]}", #\n"
     #   f"All: {ph}/{pt}  p₀={pp0:.3g}  −log₁₀p={plp_str}",
        fontsize=fontsize_tick + 1 #, fontweight="bold"
    )

    for r, stem_len in enumerate(STEM_VALUES):
        for c, loop_len in enumerate(LOOP_VALUES):
            log10p, hits, targets_in_struct, targets_p, pvalue = results[
                (sel_type, struct_type, stem_len, loop_len)
            ]
            log10p_str = f"{log10p:.2f}" if not math.isinf(log10p) else "inf"
            ann = f"{hits}/{targets_in_struct}\n{log10p_str}"
            text_color = (
                "white"
                if log10p > SIG_THRESH + (vmax - SIG_THRESH) * 0.5
                else "black"
            )
            ax.text(
                c, r, ann,
                ha="center", va="center",
                fontsize=fontsize_ann, fontweight="bold",
                color=text_color, linespacing=1.2,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("−log₁₀(p)", fontsize=fontsize_tick - 1, loc="top")
    cbar.ax.tick_params(labelsize=fontsize_tick - 1)
    cbar.ax.axhline(y=SIG_THRESH, color="black", linewidth=1.0, linestyle="--")
    cbar.ax.text(1.5, SIG_THRESH, "p=0.05", va="center",
                 fontsize=fontsize_tick - 2, color="black")

# combined 5x4 figure
fig, axes = plt.subplots(
    len(SELECTION_TYPES), len(STRUCTURE_TYPES),
    figsize=(21, 22),
)
for row_i, sel_type in enumerate(SELECTION_TYPES):
    for col_i, struct_type in enumerate(STRUCTURE_TYPES):
        draw_panel(axes[row_i, col_i], fig, sel_type, struct_type)

#fig.suptitle(
#    f"Binomial test — select-first then filter (mm={MM})\n"
#    f"Detection: stem={DET_STEM_MIN}-{DET_STEM_MAX}, gaplimit={DET_GAPLIMIT}    "
#    f"Cell: hits/targets | −log₁₀(p)    Color: −log₁₀(p-value)",
#    fontsize=12, y=1.002,
#)
plt.tight_layout()
plt.savefig(out_dir / f"heatmap_all_mm{MM}.pdf", dpi=150, bbox_inches="tight")
plt.savefig(out_dir / f"heatmap_all_mm{MM}.png", dpi=600, bbox_inches="tight")
plt.close()
print(f"\nSaved: heatmap_all_mm{MM}.pdf / .png")

# individual figures (one per panel)
for sel_type in SELECTION_TYPES:
    for struct_type in STRUCTURE_TYPES:
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_panel(ax, fig, sel_type, struct_type, fontsize_tick=9, fontsize_ann=7)
        #fig.suptitle(
        #    f"Binomial test: {struct_type} / {sel_type}  (mm={MM}, select-first)\n"
        #    "Cell: hits/targets | −log₁₀(p)    Color: −log₁₀(p-value)",
        #    fontsize=10, y=1.01,
        #)
        plt.tight_layout()
        fname = f"heatmap_mm{MM}_{sel_type}_{struct_type}"
        plt.savefig(out_dir / f"{fname}.pdf", dpi=150, bbox_inches="tight")
        plt.savefig(out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}.pdf / .png")
