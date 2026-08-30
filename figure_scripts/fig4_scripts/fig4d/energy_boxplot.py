"""
Energy (ΔG) boxplots — mutated vs unmutated hairpins — across the three
mutation datasets in hairpins_loop3_or_4_stem_ge5_mm{MM}.tsv.

Hit definition: mutation at the loop boundary
(C of TC at loop 3' end, or G of GA at loop 5' end).

Layout: 3 rows (datasets) × 6 cols (5 shapes + "All combined").
Per panel: boxplot mutated vs unmutated, strip overlay (marker size ∝ obs count),
annotation with Δmean, Mann-Whitney p, Welch t-test p, permutation p.

Output: energy_boxplot_3datasets_boundary.pdf / .png

Unwound hairpins (is_unwound == 1) are excluded.
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
parser = argparse.ArgumentParser(description="Energy (ΔG) boundary boxplots — choose one dataset row or all three.")
parser.add_argument("--dataset", default="2",
                    choices=list(_DATASET_ALIASES.keys()),
                    help="Which dataset row(s) to plot. Default: 2 (dataset2_west_africa).")
parser.add_argument("--energy_type", default="hairpin",
                    choices=["cruciform", "hairpin"],
                    help="Which energy column to plot: 'cruciform' (cruciform_energy) or 'hairpin' (pin_energy). Default: hairpin.")
args = parser.parse_args()
DATASET_SEL = _DATASET_ALIASES[args.dataset]
ENERGY_COL = {"cruciform": "cruciform_energy", "hairpin": "pin_energy"}[args.energy_type]

# inputs
TSV          = Path(__file__).parent.parent.parent.parent / "hairpins/hairpins_loop3_or_4_stem_ge5_mm0.tsv"
DATASETS_DIR = Path(__file__).parent.parent.parent.parent / "datasets"

DATASETS = [
    ("dataset1",              DATASETS_DIR / "dataset1.substitution_table.tsv"),
    ("dataset2_west_africa",  DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv"),
    ("dataset3_gisaid",       DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv"),
]
SHAPES = [(5, 3), (6, 3), (7, 3), (5, 4), (6, 4)]

# per-position multiplicity per dataset (1-based positions)
def load_per_position_counts(tsv_path):
    """Return Counter mapping 1-based position -> mutation multiplicity (APOBEC strict)."""
    sub = pd.read_csv(tsv_path, sep="\t")
    sub = sub[sub["is_APOBEC"] == True]
    return Counter(dict(zip(sub["pos_ref"].astype(int),
                            sub["independent_events"].astype(int))))

DATASET_COUNTS = {tag: load_per_position_counts(path) for tag, path in DATASETS}
for tag, ctr in DATASET_COUNTS.items():
    print(f"  {tag}: {len(ctr)} positions, {sum(ctr.values())} total observations")

# stat helpers
def mean_diff_stat(x, y):
    return np.mean(x) - np.mean(y)

def run_stats(mut_scores, unm_scores):
    n_m, n_u = len(mut_scores), len(unm_scores)
    out = {"n_mut": n_m, "n_unm": n_u,
           "mean_diff": None, "mw_p": None, "mw_stat": None,
           "tt_p": None, "tt_stat": None, "perm_p": None, "rbc": None}
    if n_m == 0 or n_u == 0:
        return out
    m, u = np.asarray(mut_scores), np.asarray(unm_scores)
    out["mean_diff"] = float(m.mean() - u.mean())
    if n_m >= 2 and n_u >= 2:
        mw = mannwhitneyu(m, u, alternative="two-sided")
        out["mw_stat"] = float(mw.statistic); out["mw_p"] = float(mw.pvalue)
        out["rbc"]     = float(1 - 2 * mw.statistic / (n_m * n_u))
        tt = ttest_ind(m, u, equal_var=False, alternative="two-sided")
        out["tt_stat"] = float(tt.statistic); out["tt_p"] = float(tt.pvalue)
        perm = permutation_test((m, u), mean_diff_stat,
                                permutation_type="independent",
                                alternative="two-sided",
                                n_resamples=9999, random_state=42)
        out["perm_p"] = float(perm.pvalue)
    return out

def fmt_p(p):
    if p is None: return "n/a"
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.4f}"

# load TSV
pos_cols = {f"mutation_positions_1b__{tag}": str for tag, _ in DATASETS}
df = pd.read_csv(TSV, sep="\t", dtype=pos_cols, keep_default_na=False)
n_total = len(df)
df = df[df["is_unwound"] == 0].reset_index(drop=True)
print(f"Loaded {n_total} hairpins from {TSV.name}; "
      f"kept {len(df)} after filtering out is_unwound==1")

# Parse the mutation-positions string per row once.
def parse_pos(s):
    if s == "" or s is None:
        return []
    return [int(p) for p in s.split(",")]

for tag, _ in DATASETS:
    df[f"_pos_list__{tag}"] = df[f"mutation_positions_1b__{tag}"].apply(parse_pos)

# boundary hit count
def hit_obs(row, dataset_tag):
    """Return observation count of mutations at the spacer boundary
    (C of TC at loop 3' end, or G of GA at loop 5' end)."""
    positions = row[f"_pos_list__{dataset_tag}"]
    if not positions:
        return 0
    loop_l, loop_r = int(row["loop_start_1b"]), int(row["loop_end_1b"])
    allowed = set()
    if int(row["tc_3p"]) == 1: allowed.add(loop_r)
    if int(row["ga_5p"]) == 1: allowed.add(loop_l)
    keep = [p for p in positions if p in allowed]
    counter = DATASET_COUNTS[dataset_tag]
    return sum(counter.get(p, 0) for p in keep)

# boxplot rendering
def draw_panel(ax, sub, dataset_tag, title):
    obs = sub.apply(lambda r: hit_obs(r, dataset_tag), axis=1)
    mut_mask = obs > 0
    mut_df, unm_df = sub[mut_mask], sub[~mut_mask]
    mut_obs = obs[mut_mask]

    mut_scores = [e for e, n in zip(mut_df[ENERGY_COL].values, mut_obs.values)
                  for _ in range(int(n))]
    unm_scores = unm_df[ENERGY_COL].tolist()

    n_mut_hp, n_mut_obs = len(mut_df), int(mut_obs.sum())
    n_unm_hp = len(unm_df)

    colors = ["#D74050", "#3389C0"]
    labels = ["Mutated", "Unmutated"]

    data = [mut_scores or [0.0], unm_scores or [0.0]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    rng = np.random.default_rng(0)
    for (_, row), n in zip(mut_df.iterrows(), mut_obs):
        jitter = rng.uniform(-0.12, 0.12)
        ax.scatter(1 + jitter, row[ENERGY_COL],
                   color=colors[0], alpha=0.85,
                   s=20 + 25 * int(n), zorder=3, edgecolors="white")
    for sc in unm_scores:
        jitter = rng.uniform(-0.12, 0.12)
        ax.scatter(2 + jitter, sc, color=colors[1], alpha=0.65, s=20, zorder=3, edgecolors="white")

    ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Hairpins", fontsize=12)
    ax.set_ylabel(f"{args.energy_type.capitalize()} energy (ΔG)", fontsize=12)

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
        draw_panel(axes[ri, ci], sub, ds, f"{ds}\nstem={sl}, loop={ll}")
    draw_panel(axes[ri, n_cols - 1], df, ds, f"{ds}\nAll combined")

plt.tight_layout()
out_stem = f"energy_boxplot_{fig_tag}_boundary_{args.energy_type}"
for ext in ("pdf", "png"):
    plt.savefig(f"{out_stem}.{ext}", dpi=600, bbox_inches="tight")
plt.close()
print(f"Saved: {out_stem}.pdf / .png")

# p-value tables (3 TSVs, one per dataset)
def collect_stats(sub, dataset_tag):
    obs = sub.apply(lambda r: hit_obs(r, dataset_tag), axis=1)
    mut_mask = obs > 0
    mut_df, unm_df = sub[mut_mask], sub[~mut_mask]
    mut_obs = obs[mut_mask]
    mut_scores = [e for e, n in zip(mut_df[ENERGY_COL].values, mut_obs.values)
                  for _ in range(int(n))]
    unm_scores = unm_df[ENERGY_COL].tolist()
    st = run_stats(mut_scores, unm_scores)
    st["n_mut_hp"]  = int(len(mut_df))
    st["n_unm_hp"]  = int(len(unm_df))
    st["n_mut_obs"] = int(mut_obs.sum())
    st["mean_mut"]  = float(np.mean(mut_scores)) if mut_scores else float("nan")
    st["mean_unm"]  = float(np.mean(unm_scores)) if unm_scores else float("nan")
    return st

cols = ["dataset", "shape", "n_mut_hp", "n_unm_hp", "n_mut_obs",
        "mean_mut", "mean_unm", "mean_diff",
        "mw_p", "tt_p", "perm_p"]

def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

out_tsv = f"energy_pvalues_{args.energy_type}.tsv"
with open(out_tsv, "w") as f:
    f.write("\t".join(cols) + "\n")
    for ds, _ in DATASETS:
        for sl, ll in SHAPES:
            sub = df[(df["target_stem_length"] == sl) & (df["target_loop_length"] == ll)]
            st = collect_stats(sub, ds)
            f.write("\t".join(fmt(v) for v in (
                ds, f"stem={sl},loop={ll}",
                st["n_mut_hp"], st["n_unm_hp"], st["n_mut_obs"],
                st["mean_mut"], st["mean_unm"], st["mean_diff"],
                st["mw_p"], st["tt_p"], st["perm_p"],
            )) + "\n")
        st = collect_stats(df, ds)
        f.write("\t".join(fmt(v) for v in (
            ds, "all_combined",
            st["n_mut_hp"], st["n_unm_hp"], st["n_mut_obs"],
            st["mean_mut"], st["mean_unm"], st["mean_diff"],
            st["mw_p"], st["tt_p"], st["perm_p"],
        )) + "\n")
print(f"Saved: {out_tsv}")
