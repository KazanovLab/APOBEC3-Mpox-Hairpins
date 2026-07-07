"""
Trinucleotide-context bar charts — mutated vs unmutated boundary contexts —
across the three mutation datasets in hairpins_loop3_or_4_stem_ge5_mm0.tsv.

For each hairpin with a TC at the loop's 3' boundary, the "3rd nucleotide
after TC" (T[C]x) is the genomic base immediately to the right of that C.
For a GA boundary, the 3rd nucleotide is the reverse-complement of the base
immediately to the left of the G (because GA on the genome ↔ TC on the other
strand). Each hairpin contributes up to two trinucleotide observations
(one for tc_3p and/or one for ga_5p).

Mutated counts are weighted by the boundary mutation observation count from
the chosen dataset; unmutated counts are 1 per hairpin.

Layout: 3 rows (datasets) × 6 cols (5 shapes + "All combined"). Per panel:
grouped bar chart (mutated vs unmutated proportions across {A, C, G, T}),
with raw counts annotated on each bar and Fisher's exact test "G vs rest"
in the corner.

Hit definition: loop boundary only (mutation at C-of-TC or G-of-GA at the loop
boundary).

Unwound hairpins (is_unwound == 1) are excluded.

Output: trinuc_3datasets_boundary.pdf / .png
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency, permutation_test

# CLI
_DATASET_ALIASES = {
    "1": "dataset1", "dataset1": "dataset1",
    "2": "dataset2_west_africa", "dataset2": "dataset2_west_africa",
    "dataset2_west_africa": "dataset2_west_africa",
    "3": "dataset3_gisaid", "dataset3": "dataset3_gisaid",
    "dataset3_gisaid": "dataset3_gisaid",
    "all": "all",
}
parser = argparse.ArgumentParser(description="Trinucleotide boundary bar charts — choose one dataset row or all three.")
parser.add_argument("--dataset", default="2",
                    choices=list(_DATASET_ALIASES.keys()),
                    help="Which dataset row(s) to plot. Default: 2 (dataset2_west_africa).")
args = parser.parse_args()
DATASET_SEL = _DATASET_ALIASES[args.dataset]

# paths
TSV          = Path(__file__).parent.parent.parent.parent / "hairpins/hairpins_loop3_or_4_stem_ge5_mm0.tsv"
DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "datasets"
GENOME_FNA   = Path(__file__).parent.parent.parent.parent / "package/input/mpox_genome_seq.fna"

DATASETS = [
    ("dataset1",              DATASETS_DIR / "dataset1.substitution_table.tsv"),
    ("dataset2_west_africa",  DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv"),
    ("dataset3_gisaid",       DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv"),
]
SHAPES = [(5, 3), (6, 3), (7, 3), (5, 4), (6, 4)]
NUCS   = ["A", "C", "G", "T"]
COLORS = {"Mutated": "#D74050", "Unmutated": "#3389C0"}
_COMP  = {"A": "T", "T": "A", "G": "C", "C": "G"}

def rc(base):
    return _COMP.get(base.upper(), "N")

# load genome (needed to compute the 3rd-nucleotide context)
with open(GENOME_FNA) as f:
    f.readline()
    GENOME = "".join(line.strip() for line in f).upper()
print(f"Loaded genome ({len(GENOME)} bp)")

# per-position mutation multiplicities
def load_per_position_counts(path):
    sub = pd.read_csv(path, sep="\t")
    sub = sub[sub["is_APOBEC"] == True]
    return Counter(dict(zip(sub["pos_ref"].astype(int),
                            sub["independent_events"].astype(int))))

DATASET_COUNTS = {tag: load_per_position_counts(path) for tag, path in DATASETS}
for tag, ctr in DATASET_COUNTS.items():
    print(f"  {tag}: {len(ctr)} positions, {sum(ctr.values())} observations")

# load hairpin TSV
pos_cols = {f"mutation_positions_1b__{tag}": str for tag, _ in DATASETS}
df = pd.read_csv(TSV, sep="\t", dtype=pos_cols, keep_default_na=False)
n_total = len(df)
df = df[df["is_unwound"] == 0].reset_index(drop=True)
print(f"Loaded {n_total} hairpins; kept {len(df)} after dropping is_unwound==1")

def parse_pos(s):
    return [int(p) for p in s.split(",")] if s else []
for tag, _ in DATASETS:
    df[f"_pos_list__{tag}"] = df[f"mutation_positions_1b__{tag}"].apply(parse_pos)

# per-hairpin boundary trinucleotide contexts
def trinuc_contexts(row):
    """Return list of (third_nt, boundary_pos_1b, side) tuples for this hairpin.
       side ∈ {'tc', 'ga'}. The 3rd nucleotide is the base AFTER C for TC,
       and the reverse-complement of the base BEFORE G for GA."""
    out = []
    loop_l = int(row["loop_start_1b"])
    loop_r = int(row["loop_end_1b"])
    if int(row["tc_3p"]) == 1:
        # boundary C is at loop_r (1-based); 3rd nt is at loop_r + 1
        if loop_r + 1 <= len(GENOME):
            out.append((GENOME[loop_r], loop_r, "tc"))   # GENOME is 0-indexed; loop_r 1-based -> index loop_r
    if int(row["ga_5p"]) == 1:
        # boundary G is at loop_l (1-based); 3rd nt is rc(base at loop_l - 1)
        if loop_l - 1 >= 1:
            out.append((rc(GENOME[loop_l - 2]), loop_l, "ga"))
    return out

df["_trinuc_ctx"] = df.apply(trinuc_contexts, axis=1)

# trinucleotide counters
def trinuc_counts(sub_df, dataset_tag):
    """Return (mut_counts, unm_counts) — defaultdict(int) over {A,C,G,T}.
       Mutated counts are weighted by per-position multiplicity from the dataset.
       A hairpin can contribute to both buckets if one loop boundary is mutated and
       the other is not."""
    counter = DATASET_COUNTS[dataset_tag]
    mut = defaultdict(int)
    unm = defaultdict(int)
    for _, row in sub_df.iterrows():
        for third, bpos, _side in row["_trinuc_ctx"]:
            n = counter.get(bpos, 0)
            if n > 0:
                mut[third] += n
            else:
                unm[third] += 1
    return mut, unm

# rendering
def draw_panel(ax, mut, unm, title):
    x = np.arange(len(NUCS))
    width = 0.35
    for offset, (label, ctr) in enumerate([("Mutated", mut), ("Unmutated", unm)]):
        counts = np.array([ctr.get(n, 0) for n in NUCS])
        total = counts.sum()
        props = counts / total if total > 0 else counts
        bars = ax.bar(x + (offset - 0.5) * width, props, width,
                      label=label, color=COLORS[label], alpha=1)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        str(int(cnt)), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"TC{n}" for n in NUCS], fontsize=11)
    ax.set_xlabel("Trinucleotide context", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7)

# select dataset rows for the figure
if DATASET_SEL == "all":
    DATASETS_TO_PLOT = DATASETS
    fig_tag = "all"
else:
    DATASETS_TO_PLOT = [(t, p) for (t, p) in DATASETS if t == DATASET_SEL]
    fig_tag = DATASET_SEL

# N rows × (len(SHAPES)+1) cols figure
n_cols, n_rows = len(SHAPES) + 1, len(DATASETS_TO_PLOT)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
if n_rows == 1:
    axes = np.array([axes])

for ri, (ds, _) in enumerate(DATASETS_TO_PLOT):
    for ci, (sl, ll) in enumerate(SHAPES):
        sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
        mut, unm = trinuc_counts(sub, ds)
        draw_panel(axes[ri, ci], mut, unm, f"{ds}\nstem={sl}, loop={ll}")
    mut, unm = trinuc_counts(df, ds)
    draw_panel(axes[ri, n_cols - 1], mut, unm, f"{ds}\nAll combined")

plt.tight_layout()

out_stem = f"trinuc_{fig_tag}_boundary"
for ext in ("pdf", "png"):
    plt.savefig( f"{out_stem}.{ext}", dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {out_stem}.pdf / .png")

# p-value tables (3 TSVs, one per dataset)
# Tests answer: "is G enriched as the 3rd nt in mutated vs unmutated boundaries?"
# (G vs rest, 2x2 contingency)
def trinuc_stats(mut, unm):
    g_mut = mut.get("G", 0); ng_mut = sum(mut.get(n, 0) for n in NUCS if n != "G")
    g_unm = unm.get("G", 0); ng_unm = sum(unm.get(n, 0) for n in NUCS if n != "G")
    n_mut_total = g_mut + ng_mut
    n_unm_total = g_unm + ng_unm
    out = {
        "g_mut": g_mut, "g_unm": g_unm,
        "n_mut_total": n_mut_total, "n_unm_total": n_unm_total,
        "prop_g_mut": g_mut / n_mut_total if n_mut_total else float("nan"),
        "prop_g_unm": g_unm / n_unm_total if n_unm_total else float("nan"),
        "fisher_p": None, "chi2_p": None, "perm_p": None,
    }
    table = [[g_mut, ng_mut], [g_unm, ng_unm]]
    if n_mut_total > 0 and n_unm_total > 0:
        _, out["fisher_p"] = fisher_exact(table, alternative="two-sided")
        row_sums = [sum(r) for r in table]
        col_sums = [sum(c) for c in zip(*table)]
        if all(row_sums) and all(col_sums):
            try:
                _, p_chi2, _, _ = chi2_contingency(table)
                out["chi2_p"] = float(p_chi2)
            except ValueError:
                pass
    if n_mut_total >= 2 and n_unm_total >= 2:
        mut_arr = np.concatenate([np.ones(g_mut), np.zeros(ng_mut)])
        unm_arr = np.concatenate([np.ones(g_unm), np.zeros(ng_unm)])
        def _propG_diff(x, y): return float(np.mean(x) - np.mean(y))
        perm = permutation_test((mut_arr, unm_arr), _propG_diff,
                                permutation_type="independent",
                                alternative="two-sided",
                                n_resamples=9999, random_state=42)
        out["perm_p"] = float(perm.pvalue)
    return out

cols = ["dataset", "shape", "n_mut_total", "n_unm_total",
        "g_mut", "g_unm", "prop_g_mut", "prop_g_unm", "prop_g_diff",
        "fisher_p", "chi2_p", "perm_p"]

def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if isinstance(v, float): return f"{v:.6g}"
    return str(v)

def _emit(f, ds, shape_label, st):
    if (not np.isnan(st["prop_g_mut"])) and (not np.isnan(st["prop_g_unm"])):
        prop_g_diff = st["prop_g_mut"] - st["prop_g_unm"]
    else:
        prop_g_diff = float("nan")
    f.write("\t".join(fmt(v) for v in (
        ds, shape_label,
        st["n_mut_total"], st["n_unm_total"],
        st["g_mut"], st["g_unm"],
        st["prop_g_mut"], st["prop_g_unm"], prop_g_diff,
        st["fisher_p"], st["chi2_p"], st["perm_p"],
    )) + "\n")

out_tsv = "trinuc_pvalues.tsv"
with open(out_tsv, "w") as f:
    f.write("\t".join(cols) + "\n")
    for ds, _ in DATASETS:
        for sl, ll in SHAPES:
            sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
            mut, unm = trinuc_counts(sub, ds)
            _emit(f, ds, f"stem={sl},loop={ll}", trinuc_stats(mut, unm))
        mut, unm = trinuc_counts(df, ds)
        _emit(f, ds, "all_combined", trinuc_stats(mut, unm))
print(f"Saved: {out_tsv}")
