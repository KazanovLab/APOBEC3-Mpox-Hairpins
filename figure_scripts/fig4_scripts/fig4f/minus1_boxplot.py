"""
Trinucleotide-context bar charts for the -1 position — mutated vs unmutated
boundary contexts — across the three mutation datasets in
hairpins_loop3_or_4_stem_ge5_mm0.tsv.

Companion to fig4e/trinuc_boxplot.py, which does the same for the nucleotide
3' of the edited C (the TCx context). Here the nucleotide 5' of the T is taken
instead, i.e. the N of NTC.

For a hairpin with a TC at the loop's 3' boundary the edited C sits at
loop_end_1b and the T immediately before it, so the -1 base is two positions
to the left of the C. For a GA boundary the motif is read on the opposite
strand: the C corresponds to loop_start_1b and the T to loop_start_1b + 1, so
the -1 base is the reverse complement of the genomic base at loop_start_1b + 2.
Each hairpin contributes up to two observations (one for tc_3p and/or ga_5p).

Mutated counts are weighted by the boundary mutation observation count from the
chosen dataset; unmutated counts are 1 per hairpin.

Layout: N rows (datasets) x 6 cols (5 shapes + "All combined"). Per panel a
grouped bar chart of mutated vs unmutated proportions across {A, C, G, T}, with
raw counts annotated on each bar.

Two contingency tests are reported per panel. "G vs rest" keeps the fig4e
comparison for symmetry. "Y vs R" (pyrimidine C/T against purine A/G) is the
contrast that matters at this position: it separates the YTCA and RTCA contexts
that distinguish APOBEC3A-like from APOBEC3B-like activity, and G-vs-rest has no
particular meaning here.

Unwound hairpins (is_unwound == 1) are excluded.

Outputs are written next to this script.
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
parser = argparse.ArgumentParser(
    description="-1 nucleotide boundary bar charts — choose one dataset row or all three.")
parser.add_argument("--dataset", default="2", choices=list(_DATASET_ALIASES.keys()),
                    help="Which dataset row(s) to plot. Default: 2 (dataset2_west_africa).")
args = parser.parse_args()
DATASET_SEL = _DATASET_ALIASES[args.dataset]

# paths
HERE         = Path(__file__).parent
ROOT         = HERE.parent.parent.parent
TSV          = ROOT / "hairpins/hairpins_loop3_or_4_stem_ge5_mm0.tsv"
DATASETS_DIR = ROOT / "datasets"
GENOME_FNA   = ROOT / "package/input/mpox_genome_seq.fna"

DATASETS = [
    ("dataset1",              DATASETS_DIR / "dataset1.substitution_table.tsv"),
    ("dataset2_west_africa",  DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv"),
    ("dataset3_gisaid",       DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv"),
]
SHAPES = [(5, 3), (6, 3), (7, 3), (5, 4), (6, 4)]
NUCS   = ["A", "C", "G", "T"]
PYRIMIDINES = ["C", "T"]
COLORS = {"Mutated": "#D74050", "Unmutated": "#3389C0"}
_COMP  = {"A": "T", "T": "A", "G": "C", "C": "G"}


def rc(base):
    return _COMP.get(base.upper(), "N")


# load genome (needed to compute the -1 context)
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


# per-hairpin boundary -1 contexts
def minus1_contexts(row):
    """Return list of (minus1_nt, boundary_pos_1b, side) tuples for this hairpin.
       side in {'tc', 'ga'}. GENOME is 0-indexed, coordinates in the TSV are 1-based."""
    out = []
    loop_l = int(row["loop_start_1b"])
    loop_r = int(row["loop_end_1b"])
    if int(row["tc_3p"]) == 1:
        # C at loop_r, T at loop_r - 1, so the -1 base is at loop_r - 2 (1-based)
        if loop_r - 2 >= 1:
            out.append((GENOME[loop_r - 3], loop_r, "tc"))
    if int(row["ga_5p"]) == 1:
        # read on the other strand: C at loop_l, T at loop_l + 1, so the -1 base
        # is the reverse complement of the genomic base at loop_l + 2 (1-based)
        if loop_l + 2 <= len(GENOME):
            out.append((rc(GENOME[loop_l + 1]), loop_l, "ga"))
    return out


# the coordinate convention is easy to get wrong by one, so check the motif
# itself rather than trusting the arithmetic
bad = 0
for _, row in df.iterrows():
    lo, hi = int(row["loop_start_1b"]), int(row["loop_end_1b"])
    if int(row["tc_3p"]) == 1 and GENOME[hi - 2: hi] != "TC":
        bad += 1
    if int(row["ga_5p"]) == 1 and GENOME[lo - 1: lo + 1] != "GA":
        bad += 1
assert bad == 0, f"{bad} boundaries do not carry the expected TC/GA motif"
print("Motif check passed: every flagged boundary carries TC (tc_3p) or GA (ga_5p)")

df["_minus1_ctx"] = df.apply(minus1_contexts, axis=1)


def minus1_counts(sub_df, dataset_tag):
    """Return (mut_counts, unm_counts) — defaultdict(int) over {A,C,G,T}.
       Mutated counts are weighted by per-position multiplicity from the dataset.
       A hairpin can contribute to both buckets if one loop boundary is mutated
       and the other is not."""
    counter = DATASET_COUNTS[dataset_tag]
    mut = defaultdict(int)
    unm = defaultdict(int)
    for _, row in sub_df.iterrows():
        for minus1, bpos, _side in row["_minus1_ctx"]:
            n = counter.get(bpos, 0)
            if n > 0:
                mut[minus1] += n
            else:
                unm[minus1] += 1
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
    ax.set_xticklabels([f"{n}TC" for n in NUCS], fontsize=11)
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

n_cols, n_rows = len(SHAPES) + 1, len(DATASETS_TO_PLOT)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
if n_rows == 1:
    axes = np.array([axes])

for ri, (ds, _) in enumerate(DATASETS_TO_PLOT):
    for ci, (sl, ll) in enumerate(SHAPES):
        sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
        mut, unm = minus1_counts(sub, ds)
        draw_panel(axes[ri, ci], mut, unm, f"{ds}\nstem={sl}, loop={ll}")
    mut, unm = minus1_counts(df, ds)
    draw_panel(axes[ri, n_cols - 1], mut, unm, f"{ds}\nAll combined")

plt.tight_layout()

out_stem = HERE / f"minus1_{fig_tag}_boundary"
for ext in ("pdf", "png"):
    plt.savefig(f"{out_stem}.{ext}", dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {out_stem.name}.pdf / .png")


# p-value tables
# Two 2x2 contrasts per panel:
#   "G_vs_rest"  — same test as fig4e, kept so the two figures are comparable
#   "Y_vs_R"     — pyrimidine (C/T) against purine (A/G), the YTC/RTC distinction
#                  that is the meaningful one at the -1 position
def contingency_stats(mut, unm, group):
    """group: list of bases forming the first cell of the 2x2 table."""
    a_mut = sum(mut.get(n, 0) for n in group)
    b_mut = sum(mut.get(n, 0) for n in NUCS if n not in group)
    a_unm = sum(unm.get(n, 0) for n in group)
    b_unm = sum(unm.get(n, 0) for n in NUCS if n not in group)
    n_mut_total, n_unm_total = a_mut + b_mut, a_unm + b_unm
    out = {
        "group": "".join(group),
        "a_mut": a_mut, "a_unm": a_unm,
        "n_mut_total": n_mut_total, "n_unm_total": n_unm_total,
        "prop_a_mut": a_mut / n_mut_total if n_mut_total else float("nan"),
        "prop_a_unm": a_unm / n_unm_total if n_unm_total else float("nan"),
        "fisher_p": None, "chi2_p": None, "perm_p": None,
    }
    table = [[a_mut, b_mut], [a_unm, b_unm]]
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
        mut_arr = np.concatenate([np.ones(a_mut), np.zeros(b_mut)])
        unm_arr = np.concatenate([np.ones(a_unm), np.zeros(b_unm)])

        def _prop_diff(x, y):
            return float(np.mean(x) - np.mean(y))

        perm = permutation_test((mut_arr, unm_arr), _prop_diff,
                                permutation_type="independent",
                                alternative="two-sided",
                                n_resamples=9999, random_state=42)
        out["perm_p"] = float(perm.pvalue)
    return out


cols = ["dataset", "shape", "test", "group", "n_mut_total", "n_unm_total",
        "a_mut", "a_unm", "prop_a_mut", "prop_a_unm", "prop_a_diff",
        "fisher_p", "chi2_p", "perm_p"]

TESTS = [("G_vs_rest", ["G"]), ("Y_vs_R", PYRIMIDINES)]


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _emit(f, ds, shape_label, test_name, st):
    if (not np.isnan(st["prop_a_mut"])) and (not np.isnan(st["prop_a_unm"])):
        prop_a_diff = st["prop_a_mut"] - st["prop_a_unm"]
    else:
        prop_a_diff = float("nan")
    f.write("\t".join(fmt(v) for v in (
        ds, shape_label, test_name, st["group"],
        st["n_mut_total"], st["n_unm_total"],
        st["a_mut"], st["a_unm"],
        st["prop_a_mut"], st["prop_a_unm"], prop_a_diff,
        st["fisher_p"], st["chi2_p"], st["perm_p"],
    )) + "\n")


out_tsv = HERE / "minus1_pvalues.tsv"
with open(out_tsv, "w") as f:
    f.write("\t".join(cols) + "\n")
    for ds, _ in DATASETS:
        for sl, ll in SHAPES:
            sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
            mut, unm = minus1_counts(sub, ds)
            for test_name, group in TESTS:
                _emit(f, ds, f"stem={sl},loop={ll}", test_name,
                      contingency_stats(mut, unm, group))
        mut, unm = minus1_counts(df, ds)
        for test_name, group in TESTS:
            _emit(f, ds, "all_combined", test_name, contingency_stats(mut, unm, group))
print(f"Saved: {out_tsv.name}")
