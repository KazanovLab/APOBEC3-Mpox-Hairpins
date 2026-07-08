"""

  1. Detect hairpins ONCE with broad parameters:
       stem_min=4, stem_max=12, gaplimit=20, mm=MM
  2. Apply the non-extendable-stem filter to the full pool.
  3. Apply the selection method on the full filtered pool — so the selection
     competes among hairpins of *different* shapes.
  4. Post-filter the selected hairpins by exact stem_length and spacer_length
     for each (stem_len, loop_len) cell and compute the exclusive p-value.

Structure types (exclusive, no overlap):
  tc_end          — TC/GA at spacer boundary
  c_end_excl      — C/G at spacer boundary NOT in TC/GA context
  spacer_interior — TC/GA in interior spacer positions [l+1, r-1]
  stem            — TC/GA in stem arm positions only

Denominators:
  tc_end, spacer_interior, stem  → total TC+GA positions in genome
  c_end_excl                     → total (C not in TC) + (G not in GA) in genome

Layout: 5-row × 4-column grid of heatmaps.
  rows    → selection_type: most_stable, greedy, max_cov, min_cov, all
  columns → structure_type: tc_end, c_end_excl, spacer_interior, stem
"""

import sys
import os
import re
import math
import pickle
import argparse
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import binomtest

# paths
BINOMIAL_DIR = Path(__file__).parent.parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))
os.chdir(BINOMIAL_DIR)

import load
from emboss import get_palindrome
from functions import find_groups, max_coverage

# command-line arguments
parser = argparse.ArgumentParser(description="Test 17: exclusive select-first heatmaps.")
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

STRUCTURE_TYPES = ["tc_end", "c_end_excl", "spacer_interior", "stem"]
SELECTION_TYPES = ["most_stable", "greedy", "max_cov", "min_cov", "all"]

# genome / mutation data (loaded via load module)
_genome_seq = load.genome.sequence
_mut_list_APOBEC = load.mutations_list_APOBEC   # 0-based, with multiplicity
_mut_list_APOBEC_ext = load.mutations_list_APOBEC_ext
_mut_cnt_APOBEC    = load.mut_cnt_APOBEC
_mut_cnt_APOBEC_ext = load.mut_cnt_APOBEC_ext

# pre-compute genome-wide target sets (done once)
print("Pre-computing genome-wide target sets...")
_tc_pos   = set(m.end() - 1 for m in re.finditer("TC", _genome_seq))   # C of TC
_ga_pos   = set(m.start()   for m in re.finditer("GA", _genome_seq))   # G of GA
_all_c    = set(m.start()   for m in re.finditer("C",  _genome_seq))
_all_g    = set(m.start()   for m in re.finditer("G",  _genome_seq))
_c_not_tc = _all_c - _tc_pos   # C not preceded by T
_g_not_ga = _all_g - _ga_pos   # G not followed by A

_denom_tc_ga    = len(_tc_pos | _ga_pos)
_denom_c_not_tc = len(_c_not_tc | _g_not_ga)
print(f"  TC+GA positions in genome    : {_denom_tc_ga}")
print(f"  C_not_TC+G_not_GA in genome  : {_denom_c_not_tc}")

# non-extendable-stem filter
_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def is_extendable(h):
    """True if flanking nucleotides outside the stem are WC complementary."""
    if h.start == 0 or h.end >= len(_genome_seq) - 1:
        return False
    left  = _genome_seq[h.start - 1].upper()
    right = _genome_seq[h.end + 1].upper()
    return _COMPLEMENT.get(left) == right

# selection helpers
_priority_is_max = False  # matches load.priority_is_max

def _most_stable(group):
    k = -1 if _priority_is_max else 1
    return [min(group, key=lambda x: (x.score * k, k * x.length, x.start))]

def _min_coverage(group):
    return [min(group, key=lambda x: (x.length, x.score, x.start))]

def _greedy_choose(hairpins_group, taken=None, skip_sort=False):
    if taken is None:
        taken = []
    hairpins_group_copy = deepcopy(hairpins_group)
    taken_copy = taken.copy()

    if not skip_sort:
        k = -1 if _priority_is_max else 1
        hairpins_group_copy.sort(key=lambda x: x.score * k)
        n = len(hairpins_group_copy)
        i = 0
        equal_groups = []
        while i < n:
            current_group = [hairpins_group_copy[i]]
            j = i + 1
            while j < n and hairpins_group_copy[i].score == hairpins_group_copy[j].score:
                current_group.append(hairpins_group_copy[j])
                j += 1
            current_group.sort(key=lambda x: x.length * -1)
            equal_groups.append(current_group)
            i = j
        hairpins_group_copy = equal_groups

    for p in range(len(hairpins_group_copy)):
        equal_group = hairpins_group_copy[p]
        if equal_group == []:
            continue
        usable = [h for h in equal_group
                  if all(h.can_exist(j) for j in taken_copy)]
        n = len(usable)
        if n == 0:
            continue
        elif n == 1:
            taken_copy.append(usable[0])
        elif n <= 11:
            variants = []
            for hairpin in usable:
                new_hg = deepcopy(hairpins_group_copy)
                new_hg[p] = [h for h in new_hg[p] if h is not hairpin]
                temp = _greedy_choose(new_hg[p:], taken_copy + [hairpin], True)
                variants.append(temp)
            return max(variants, key=lambda x: len(x))
        else:
            hairpin = usable[0]
            new_hg = deepcopy(hairpins_group_copy)
            new_hg[p] = [h for h in new_hg[p] if h is not hairpin]
            return _greedy_choose(new_hg[p:], taken_copy + [hairpin], True)

    return taken_copy


def select_hairpins(hairpins, sel_type):
    """Return a list of (non-overlapping) hairpins according to sel_type.
    """
    if sel_type == "all":
        return list(hairpins)

    groups = find_groups(list(hairpins))
    selected = []
    for group in groups:
        if sel_type == "most_stable":
            choice = _most_stable(group)
        elif sel_type == "greedy":
            choice = _greedy_choose(group)
        elif sel_type == "max_cov":
            choice = max_coverage(group)
        elif sel_type == "min_cov":
            choice = _min_coverage(group)
        else:
            choice = _most_stable(group)
        selected.extend(choice)
    return selected

# -- exclusive target computation (identical to test11) ------------------------
def compute_exclusive(hairpins, struct_type):
    """
    Return (hits, n_struct_targets, targets_p) for the given exclusive struct_type.

    hits        — number of mutation observations (with multiplicity) in struct_targets
    n_struct    — number of distinct genomic positions that are structure targets
    targets_p   — n_struct / genome_denominator
    """
    if struct_type == "tc_end":
        struct_targets = set()
        for h in hairpins:
            if h.spacer_length < 2:
                continue
            l, r = h.spacer_index
            if r in _tc_pos:
                struct_targets.add(r)
            if l in _ga_pos:
                struct_targets.add(l)
        targets_p = len(struct_targets) / _denom_tc_ga if _denom_tc_ga else 0.0

    elif struct_type == "c_end_excl":
        struct_targets = set()
        for h in hairpins:
            if h.spacer_length < 1:
                continue
            l, r = h.spacer_index
            if r in _c_not_tc:
                struct_targets.add(r)
            if l in _g_not_ga:
                struct_targets.add(l)
        targets_p = len(struct_targets) / _denom_c_not_tc if _denom_c_not_tc else 0.0

    elif struct_type == "spacer_interior":
        struct_targets = set()
        for h in hairpins:
            l, r = h.spacer_index
            for pos in range(l + 1, r):   # interior: [l+1, r-1]
                if pos in _tc_pos or pos in _ga_pos:
                    struct_targets.add(pos)
        targets_p = len(struct_targets) / _denom_tc_ga if _denom_tc_ga else 0.0

    elif struct_type == "stem":
        struct_targets = set()
        for h in hairpins:
            ls, le = h.stem_indexes[0]      # left arm
            rs, re_ = h.stem_indexes[1]     # right arm
            for pos in range(ls, le + 1):
                if pos in _tc_pos or pos in _ga_pos:
                    struct_targets.add(pos)
            for pos in range(rs, re_ + 1):
                if pos in _tc_pos or pos in _ga_pos:
                    struct_targets.add(pos)
        targets_p = len(struct_targets) / _denom_tc_ga if _denom_tc_ga else 0.0

    else:
        raise ValueError(f"Unknown struct_type: {struct_type}")

    if struct_type == "c_end_excl":
        hits = sum(1 for m in _mut_list_APOBEC_ext if m in struct_targets)
    else:
        hits = sum(1 for m in _mut_list_APOBEC if m in struct_targets)

    return hits, len(struct_targets), targets_p

# step 1: broad hairpin detection
print(f"\nDetecting hairpins broadly "
      f"(stem={DET_STEM_MIN}-{DET_STEM_MAX}, gaplimit={DET_GAPLIMIT}, mm={MM})...")
pa_file = get_palindrome(DET_STEM_MIN, DET_STEM_MAX, DET_GAPLIMIT, MM)
all_hairpins = load.load_hairpins(str(pa_file))
print(f"  total hairpins found: {len(all_hairpins)}")

broad_pool = [h for h in all_hairpins if not is_extendable(h)]
print(f"  non-extendable hairpins: {len(broad_pool)}")

# step 2: apply selection on the full broad pool
# Selection is struct_type-agnostic in test11 → cache one pool per sel_type.
print("\nApplying selection on the broad pool...")
selected_pools = {}
for sel_type in SELECTION_TYPES:
    sel = select_hairpins(broad_pool, sel_type)
    selected_pools[sel_type] = sel
    print(f"  [{sel_type}]: {len(sel)} hairpins selected")

# step 3: per-cell post-filter + exclusive p-value 
# key: (sel_type, struct_type, stem_len, loop_len)
# val: (log10p, hits, n_struct, targets_p, pvalue)
print("\nComputing per-cell p-values...")
results = {}

for sel_type in SELECTION_TYPES:
    pool = selected_pools[sel_type]
    for struct_type in STRUCTURE_TYPES:
        print(f"\n[{sel_type} / {struct_type}]")
        for stem_len in STEM_VALUES:
            for loop_len in LOOP_VALUES:
                cell = [h for h in pool
                        if h.stem_length == stem_len and h.spacer_length == loop_len]
                if not cell:
                    results[(sel_type, struct_type, stem_len, loop_len)] = (0.0, 0, 0, 0.0, 1.0)
                    continue

                hits, n_struct, targets_p = compute_exclusive(cell, struct_type)

                if targets_p > 0:
                    if struct_type == "c_end_excl":
                        pvalue = binomtest(hits, _mut_cnt_APOBEC_ext, targets_p,
                                       alternative='greater').pvalue
                    else:
                        pvalue = binomtest(hits, _mut_cnt_APOBEC, targets_p,
                                       alternative='greater').pvalue
                else:
                    pvalue = 1.0

                log10p = -math.log10(pvalue) if pvalue > 0 else float("inf")
                results[(sel_type, struct_type, stem_len, loop_len)] = (
                    log10p, hits, n_struct, targets_p, pvalue
                )
                print(f"  stem={stem_len} loop={loop_len}: "
                      f"hits={hits}, tgt={n_struct}, "
                      f"p={pvalue:.5f}, -log10p={log10p:.3f}")

# panel-level combined stats
_GENOME_DENOM = {
    "tc_end":          _denom_tc_ga,
    "c_end_excl":      _denom_c_not_tc,
    "spacer_interior": _denom_tc_ga,
    "stem":            _denom_tc_ga,
}
panel_stats = {}
for _sel in SELECTION_TYPES:
    for _str in STRUCTURE_TYPES:
        if _str == "c_end_excl":
            mcnt = _mut_cnt_APOBEC_ext
        else:
            mcnt = _mut_cnt_APOBEC
        _non_empty = [
            results[(_sel, _str, sl, ll)]
            for sl in STEM_VALUES for ll in LOOP_VALUES
            if results[(_sel, _str, sl, ll)][2] > 0
        ]
        _hits    = sum(v[1] for v in _non_empty)
        _targets = sum(v[2] for v in _non_empty)
        _gdenom  = _GENOME_DENOM[_str]
        _p0      = min(_targets / _gdenom, 1.0) if _gdenom > 0 else 0.0
        _k       = min(_hits, mcnt)
        if _p0 > 0:
            _pv = binomtest(_k, mcnt, _p0, alternative='greater').pvalue
        else:
            _pv = 1.0
        _lp = -math.log10(_pv) if _pv > 0 else float('inf')
        panel_stats[(_sel, _str)] = (_hits, _targets, _p0, _lp)

# save results
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

_mut_counts_APOBEC = Counter(_mut_list_APOBEC)
_mut_set_APOBEC    = set(_mut_counts_APOBEC)
_mut_counts_APOBEC_ext = Counter(_mut_list_APOBEC_ext)
_mut_set_APOBEC_ext    = set(_mut_counts_APOBEC_ext)

def hairpin_mutations_excl(h, struct_type):
    """Return sorted list of mutation positions (0-based, unique) that hit
    this hairpin under struct_type (test17 exclusive definitions)."""
    hits = []
    if struct_type == "tc_end":
        if h.spacer_length < 2:
            return []
        l, r = h.spacer_index
        if r in _tc_pos and r in _mut_set_APOBEC:
            hits.append(r)
        if l in _ga_pos and l in _mut_set_APOBEC:
            hits.append(l)
    elif struct_type == "c_end_excl":
        if h.spacer_length < 1:
            return []
        l, r = h.spacer_index
        if r in _c_not_tc and r in _mut_set_APOBEC_ext:
            hits.append(r)
        if l in _g_not_ga and l in _mut_set_APOBEC_ext:
            hits.append(l)
    elif struct_type == "spacer_interior":
        l, r = h.spacer_index
        for pos in range(l + 1, r):
            if (pos in _tc_pos or pos in _ga_pos) and pos in _mut_set_APOBEC:
                hits.append(pos)
    elif struct_type == "stem":
        ls, le = h.stem_indexes[0]
        rs, re_ = h.stem_indexes[1]
        for pos in range(ls, le + 1):
            if (pos in _tc_pos or pos in _ga_pos) and pos in _mut_set_APOBEC:
                hits.append(pos)
        for pos in range(rs, re_ + 1):
            if (pos in _tc_pos or pos in _ga_pos) and pos in _mut_set_APOBEC:
                hits.append(pos)
    return sorted(set(hits))

rows = []
for (sel_type, struct_type, stem_len, loop_len), (log10p, hits_cnt, n_struct, tp, pvalue) in results.items():
    if pvalue >= SIG_P:
        continue
    pool = selected_pools[sel_type]
    cell = [h for h in pool
            if h.stem_length == stem_len and h.spacer_length == loop_len]
    for h in cell:
        muts = hairpin_mutations_excl(h, struct_type)
        if struct_type == "c_end_excl":
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
    "stem": "Stem",
    "spacer_interior": "Loop excluding TC at the 3' end",
    "tc_end": "TC at the 3' loop end",
    "c_end_excl": "C at the 3' loop end excluding TC"
}
selection_labels = {
    "most_stable": "Most stable hairpin",
    "greedy": "Greedy selection",
    "max_cov": "Maximum coverage",
    "min_cov": "Minimum coverage",
    "all": "All hairpins"
}

# helper: draw one heatmap into an axes
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
        f"{struct_labels[struct_type]} / {selection_labels[sel_type]}",
    #    f"All: {ph}/{pt}  p₀={pp0:.3g}  −log₁₀p={plp_str}",
        fontsize=fontsize_tick + 1 #, fontweight="bold"
    )

    for r, stem_len in enumerate(STEM_VALUES):
        for c, loop_len in enumerate(LOOP_VALUES):
            log10p, hits, n_struct, targets_p, pvalue = results[
                (sel_type, struct_type, stem_len, loop_len)
            ]
            log10p_str = f"{log10p:.2f}" if not math.isinf(log10p) else "inf"
            ann = f"{hits}/{n_struct}\n{log10p_str}"
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

# -- combined 5x4 figure -------------------------------------------------------
fig, axes = plt.subplots(
    len(SELECTION_TYPES), len(STRUCTURE_TYPES),
    figsize=(18, 22),
)
for row_i, sel_type in enumerate(SELECTION_TYPES):
    for col_i, struct_type in enumerate(STRUCTURE_TYPES):
        draw_panel(axes[row_i, col_i], fig, sel_type, struct_type)

#fig.suptitle(
#    f"Binomial test — exclusive, select-first (mm={MM})\n"
#    f"Detection: stem={DET_STEM_MIN}-{DET_STEM_MAX}, gaplimit={DET_GAPLIMIT}    "
#    f"Cell: hits/targets | −log₁₀(p)    Color: −log₁₀(p-value)",
#    fontsize=12, y=1.002,
#)
plt.tight_layout()
plt.savefig(out_dir / f"heatmap_all_mm{MM}.pdf", dpi=150, bbox_inches="tight")
plt.savefig(out_dir / f"heatmap_all_mm{MM}.png", dpi=600, bbox_inches="tight")
plt.close()
print(f"\nSaved: heatmap_all_mm{MM}.pdf / .png")

# -- individual figures (one per panel) ----------------------------------------
for sel_type in SELECTION_TYPES:
    for struct_type in STRUCTURE_TYPES:
        fig, ax = plt.subplots(figsize=(6, 5))
        draw_panel(ax, fig, sel_type, struct_type, fontsize_tick=9, fontsize_ann=7)
        #fig.suptitle(
        #    f"Binomial test: {struct_type} / {sel_type}  (mm={MM}, exclusive, select-first)\n"
        #    "Cell: hits/targets | −log₁₀(p)    Color: −log₁₀(p-value)",
        #    fontsize=10, y=1.01,
        #)
        plt.tight_layout()
        fname = f"heatmap_mm{MM}_{sel_type}_{struct_type}"
        plt.savefig(out_dir / f"{fname}.pdf", dpi=150, bbox_inches="tight")
        plt.savefig(out_dir / f"{fname}.png", dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}.pdf / .png")
