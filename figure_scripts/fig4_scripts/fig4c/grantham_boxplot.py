"""
Grantham-score boxplots — mutated vs unmutated boundary positions —
across the three mutation datasets in hairpins_loop3_or_4_stem_ge5_mm0.tsv.

For each hairpin we annotate its boundary substitution(s):
  - TC at 3' end of loop  -> C->T at position loop_end_1b
  - GA at 5' end of loop  -> G->A at position loop_start_1b
Each boundary is classified as intergenic / synonymous / nonsynonymous against
the mpox GTF, and a Grantham distance is assigned to nonsynonymous
amino-acid changes. Synonymous + intergenic -> 0. Nonsense (AA -> *) and
stop-loss (* -> AA) substitutions are scored as STOP_GRANTHAM (default 215,
the matrix maximum). A hairpin with two boundaries contributes the max of its
two Grantham scores (one row per hairpin).

Hit definition (mutated vs unmutated): loop boundary only — mutation observation
must hit C-of-TC or G-of-GA at the loop boundary.

Layout: 3 rows (datasets) × 6 cols (5 shapes + "All combined").
Per panel: boxplot mutated vs unmutated, strip overlay, mean diamond,
annotation with Mann-Whitney p, Welch t-test p, permutation p.

Unwound hairpins (is_unwound == 1) are excluded.

Output: grantham_3datasets_boundary.pdf / .png
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu, ttest_ind, permutation_test

# CLI
_DATASET_ALIASES = {
    "1": "dataset1", "dataset1": "dataset1",
    "2": "dataset2_west_africa", "dataset2": "dataset2_west_africa",
    "dataset2_west_africa": "dataset2_west_africa",
    "3": "dataset3_gisaid", "dataset3": "dataset3_gisaid",
    "dataset3_gisaid": "dataset3_gisaid",
    "all": "all",
}
parser = argparse.ArgumentParser(description="Grantham boundary boxplots — choose one dataset row or all three.")
parser.add_argument("--dataset", default="2",
                    choices=list(_DATASET_ALIASES.keys()),
                    help="Which dataset row(s) to plot. Default: 2 (dataset2_west_africa).")
args = parser.parse_args()
DATASET_SEL = _DATASET_ALIASES[args.dataset]

# paths
TSV          = Path(__file__).parent.parent.parent.parent / "hairpins/hairpins_loop3_or_4_stem_ge5_mm0.tsv"
DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "datasets"
GENOME_FNA   = Path(__file__).parent.parent.parent.parent / "package/input/mpox_genome_seq.fna"
GTF          = Path(__file__).parent.parent.parent.parent / "package/input/GCF_014621545.1_ASM1462154v1_genomic.gtf"

DATASETS = [
    ("dataset1",              DATASETS_DIR / "dataset1.substitution_table.tsv"),
    ("dataset2_west_africa",  DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv"),
    ("dataset3_gisaid",       DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv"),
]
SHAPES = [(5, 3), (6, 3), (7, 3), (5, 4), (6, 4)]

# load genome (1-based access via [pos - 1])
with open(GENOME_FNA) as f:
    f.readline()
    GENOME = "".join(line.strip() for line in f).upper()
print(f"Loaded genome ({len(GENOME)} bp)")

_COMP_TR = str.maketrans("ACGT", "TGCA")
def rc(s): return s.translate(_COMP_TR)[::-1]

# codon table
CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

# parse GTF: CDS features
cds_list = []
with open(GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.rstrip('\n').split('\t')
        if len(fields) < 9 or fields[2] != 'CDS':
            continue
        start = int(fields[3]); end = int(fields[4])
        strand = fields[6]; phase = int(fields[7])
        attrs = fields[8]
        m = re.search(r'gene "([^"]+)"', attrs)
        gene = m.group(1) if m else re.search(r'locus_tag "([^"]+)"', attrs).group(1)
        cds_list.append((start, end, strand, gene, phase))
cds_list.sort(key=lambda x: x[0])
print(f"Loaded {len(cds_list)} CDS features from GTF")

# Grantham
_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_GRANTHAM_RAW = [
    [  0,112,111,126,195, 91,107, 60, 86, 94,102,106, 84,113, 27, 99, 58,148,112, 64],
    [112,  0, 86, 96,180, 43, 54,125, 29, 97,102, 26, 91, 97,103,110, 71,101, 77, 96],
    [111, 86,  0, 23,139, 46, 42, 80, 68,149,153, 94,142,158, 91, 46, 65,174,143,133],
    [126, 96, 23,  0,154, 61, 45, 94, 81,168,172,101,160,177,108, 65, 85,181,160,152],
    [195,180,139,154,  0,154,170,159,174,198,198,202,196,205,169,112,149,215,194,192],
    [ 91, 43, 46, 61,154,  0, 29, 87, 24,109,113, 53,101,116, 76, 68, 42,130, 99, 96],
    [107, 54, 42, 45,170, 29,  0, 98, 40,134,138, 56,126,140, 93, 80, 65,152,122,121],
    [ 60,125, 80, 94,159, 87, 98,  0, 98,135,138,127,127,153, 42, 56, 59,184,147,109],
    [ 86, 29, 68, 81,174, 24, 40, 98,  0, 94, 99, 32, 87,100, 77, 89, 47,115, 83, 84],
    [ 94, 97,149,168,198,109,134,135, 94,  0,  5,102, 10, 21, 95,142, 89, 61, 33, 29],
    [102,102,153,172,198,113,138,138, 99,  5,  0,107, 15, 22, 98,145, 92, 61, 36, 32],
    [106, 26, 94,101,202, 53, 56,127, 32,102,107,  0, 95,102,103,121, 78,110, 85, 97],
    [ 84, 91,142,160,196,101,126,127, 87, 10, 15, 95,  0, 28, 87,135, 81, 67, 36, 21],
    [113, 97,158,177,205,116,140,153,100, 21, 22,102, 28,  0,114,155,103, 40, 22, 50],
    [ 27,103, 91,108,169, 76, 93, 42, 77, 95, 98,103, 87,114,  0, 74, 38,147,110, 68],
    [ 99,110, 46, 65,112, 68, 80, 56, 89,142,145,121,135,155, 74,  0, 58,177,144,124],
    [ 58, 71, 65, 85,149, 42, 65, 59, 47, 89, 92, 78, 81,103, 38, 58,  0,128, 92, 69],
    [148,101,174,181,215,130,152,184,115, 61, 61,110, 67, 40,147,177,128,  0, 37, 88],
    [112, 77,143,160,194, 99,122,147, 83, 33, 36, 85, 36, 22,110,144, 92, 37,  0, 55],
    [ 64, 96,133,152,192, 96,121,109, 84, 29, 32, 97, 21, 50, 68,124, 69, 88, 55,  0],
]
_IDX = {aa: i for i, aa in enumerate(_AA_ORDER)}
GRANTHAM = {(a1, a2): _GRANTHAM_RAW[_IDX[a1]][_IDX[a2]]
            for a1 in _AA_ORDER for a2 in _AA_ORDER}

# Score used for substitutions that create or destroy a stop codon
# (nonsense AA->*, or stop-loss *->AA). 
STOP_GRANTHAM = 215

# annotation helpers (from test12/annotate_mutations.py)
def overlapping_cds(pos_1):
    return [c for c in cds_list if c[0] <= pos_1 <= c[1]]

def codon_consequence(pos_1, mut_base, cds):
    start, end, strand, gene, phase = cds
    if strand == '+':
        pos_in_cds  = pos_1 - start
        codon_index = (pos_in_cds + phase) // 3
        codon_start = start + codon_index * 3 - phase
        orig_codon  = GENOME[codon_start - 1 : codon_start + 2]
        local       = pos_1 - codon_start
        mut_codon   = orig_codon[:local] + mut_base + orig_codon[local + 1:]
    else:
        pos_in_cds   = end - pos_1
        codon_index  = (pos_in_cds + phase) // 3
        codon_right  = end - codon_index * 3 + phase
        orig_codon_p = GENOME[codon_right - 3 : codon_right]
        orig_codon   = rc(orig_codon_p)
        local_p      = pos_1 - (codon_right - 2)
        mut_codon_p  = orig_codon_p[:local_p] + mut_base + orig_codon_p[local_p + 1:]
        mut_codon    = rc(mut_codon_p)
    return CODON_TABLE.get(orig_codon, '?'), CODON_TABLE.get(mut_codon, '?')

def grantham_for_position(pos_1, mut_base):
    """Return the Grantham score for a single substitution.
       0 for intergenic or synonymous; the max across overlapping CDSs for
       nonsynonymous AA<->AA substitutions; STOP_GRANTHAM for nonsense
       (AA->*) or stop-loss (*->AA); 0 when annotation fails for any other
       reason (e.g. codon containing 'N')."""
    hits = overlapping_cds(pos_1)
    if not hits:
        return 0   # intergenic
    scores = []
    for cds in hits:
        wt, mut = codon_consequence(pos_1, mut_base, cds)
        if wt == mut:
            scores.append(0)                          # synonymous (incl. stop→stop)
        elif wt in _IDX and mut in _IDX:
            scores.append(GRANTHAM[(wt, mut)])        # standard AA → AA
        elif wt == '*' or mut == '*':
            scores.append(STOP_GRANTHAM)              # nonsense or stop-loss
        else:
            scores.append(0)                          # truly unknown (e.g. N in codon)
    return max(scores)

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

# per-hairpin Grantham score
def hairpin_grantham(row):
    loop_l = int(row["loop_start_1b"])
    loop_r = int(row["loop_end_1b"])
    scores = []
    if int(row["tc_3p"]) == 1:
        scores.append(grantham_for_position(loop_r, 'T'))   # C->T at 1-based loop_r
    if int(row["ga_5p"]) == 1:
        scores.append(grantham_for_position(loop_l, 'A'))   # G->A at 1-based loop_l
    return max(scores) if scores else 0

df["grantham"] = df.apply(hairpin_grantham, axis=1)

# loop boundary mutated/not for each dataset
def boundary_observations(row, dataset_tag):
    positions = row[f"_pos_list__{dataset_tag}"]
    if not positions:
        return 0
    loop_l = int(row["loop_start_1b"]); loop_r = int(row["loop_end_1b"])
    allowed = set()
    if int(row["tc_3p"]) == 1: allowed.add(loop_r)
    if int(row["ga_5p"]) == 1: allowed.add(loop_l)
    keep = [p for p in positions if p in allowed]
    ctr = DATASET_COUNTS[dataset_tag]
    return sum(ctr.get(p, 0) for p in keep)

# stat helpers
def mean_diff(x, y): return np.mean(x) - np.mean(y)

def run_tests(mut_gs, unm_gs):
    res = dict(n_mut=len(mut_gs), n_unm=len(unm_gs),
               mean_mut=float(np.mean(mut_gs)) if mut_gs else float('nan'),
               mean_unm=float(np.mean(unm_gs)) if unm_gs else float('nan'),
               mwu_p=None, tt_p=None, perm_p=None)
    if len(mut_gs) < 2 or len(unm_gs) < 2:
        return res
    res['mwu_p']  = float(mannwhitneyu(mut_gs, unm_gs,
                                       alternative='two-sided').pvalue)
    res['tt_p']   = float(ttest_ind(mut_gs, unm_gs, equal_var=False,
                                    alternative='two-sided').pvalue)
    res['perm_p'] = float(permutation_test(
        (np.array(mut_gs), np.array(unm_gs)), mean_diff,
        permutation_type='independent', alternative='two-sided',
        n_resamples=9999, random_state=42).pvalue)
    return res

def fmt_p(p):
    if p is None: return "n/a"
    if p < 0.001: return f"p={p:.2e}"
    return f"p={p:.3f}"

# rendering
COLOR_MUT = "#D74050"
COLOR_UNM = "#3389C0"

def draw_panel(ax, sub, dataset_tag, title):
    obs = sub.apply(lambda r: boundary_observations(r, dataset_tag), axis=1)
    mut_mask = obs > 0
    mut_gs = sub.loc[mut_mask, "grantham"].tolist()
    unm_gs = sub.loc[~mut_mask, "grantham"].tolist()
    stats = run_tests(mut_gs, unm_gs)

    groups = [mut_gs, unm_gs]
    labels = ["Mutated", "Unmutated"]
    colors = [COLOR_MUT, COLOR_UNM]

    bp = ax.boxplot(
        [g if g else [np.nan] for g in groups],
        patch_artist=True, widths=0.45,
        medianprops=dict(color="black", linewidth=2),
        showfliers=False,
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7) #0.45)

    rng = np.random.default_rng(42)
    for xi, (gs, color) in enumerate(zip(groups, colors), start=1):
        if gs:
            jitter = rng.uniform(-0.14, 0.14, size=len(gs))
            ax.scatter(xi + jitter, gs, color=color, alpha=0.85,
                       s=40, zorder=3, edgecolors="white", linewidths=0.4)
    for xi, gs in enumerate(groups, start=1):
        if gs:
            ax.scatter(xi, np.mean(gs), marker="D", color="black",
                       s=35, zorder=4)

    ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Hairpins", fontsize=12)
    ax.set_ylabel("Grantham score", fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.grid(axis='y', linestyle='--', alpha=0.35)

# select dataset rows for the figure
if DATASET_SEL == "all":
    DATASETS_TO_PLOT = DATASETS
    fig_tag = "all"
else:
    DATASETS_TO_PLOT = [(t, p) for (t, p) in DATASETS if t == DATASET_SEL]
    fig_tag = DATASET_SEL

# N rows × (len(SHAPES) + 1) figure
n_cols, n_rows = len(SHAPES) + 1, len(DATASETS_TO_PLOT)
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.5 * n_cols, 4.5 * n_rows))
if n_rows == 1:
    axes = np.array([axes])

for ri, (ds, _) in enumerate(DATASETS_TO_PLOT):
    for ci, (sl, ll) in enumerate(SHAPES):
        sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
        draw_panel(axes[ri, ci], sub, ds, f"{ds}\nstem={sl}, loop={ll}")
    draw_panel(axes[ri, n_cols - 1], df, ds, f"{ds}\nAll combined")

plt.tight_layout()

out_stem = f"grantham_{fig_tag}_boundary"
for ext in ("pdf", "png"):
    plt.savefig( f"{out_stem}.{ext}", dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {out_stem}.pdf / .png")

# p-value tables (3 TSVs, one per dataset)
def collect_stats(sub, dataset_tag):
    obs = sub.apply(lambda r: boundary_observations(r, dataset_tag), axis=1)
    mut_mask = obs > 0
    mut_gs = sub.loc[mut_mask, "grantham"].tolist()
    unm_gs = sub.loc[~mut_mask, "grantham"].tolist()
    return run_tests(mut_gs, unm_gs)

cols = ["dataset", "shape", "n_mut", "n_unm",
        "mean_mut", "mean_unm", "mean_diff",
        "mwu_p", "tt_p", "perm_p"]

def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if isinstance(v, float): return f"{v:.6g}"
    return str(v)

def _emit(f, ds, shape_label, st):
    if (st["mean_mut"] is not None and st["mean_unm"] is not None
            and not np.isnan(st["mean_mut"]) and not np.isnan(st["mean_unm"])):
        mean_diff = st["mean_mut"] - st["mean_unm"]
    else:
        mean_diff = float("nan")
    f.write("\t".join(fmt(v) for v in (
        ds, shape_label,
        st["n_mut"], st["n_unm"],
        st["mean_mut"], st["mean_unm"], mean_diff,
        st["mwu_p"], st["tt_p"], st["perm_p"],
    )) + "\n")

out_tsv = "grantham_pvalues.tsv"
with open(out_tsv, "w") as f:
    f.write("\t".join(cols) + "\n")
    for ds, _ in DATASETS:
        for sl, ll in SHAPES:
            sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
            _emit(f, ds, f"stem={sl},loop={ll}", collect_stats(sub, ds))
        _emit(f, ds, "all_combined", collect_stats(df, ds))
print(f"Saved: {out_tsv}")
