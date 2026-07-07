"""
Null distribution of significant cells via shuffled-genome permutation

For each iteration:
  1. Shuffle the real mpox genome (random permutation of all bases).
     Preserves exact mononucleotide composition.
  2. Run the test pipeline on the shuffled genome:
       - broad EMBOSS detection (stem 4-12, gaplimit 20, mm=MM)
       - selection (most_stable / greedy / max_cov / min_cov / all)
       - per-cell exclusive p-value (tc_end / c_end_excl / spacer_interior / stem)
       - cells span stem in [4..12], loop in [2..10]  (9 x 9 = 81 per panel)
  3. Mutation positions stay fixed at the observed mpox positions.
  4. Count cells with p < 0.05 across all 5 sel_types x 4 struct_types x 81 cells
     (1620 cells per iteration).

Default: 1000 iterations, multiprocessing across all cores.

Output:
  - null_sig_counts_mm{MM}.tsv   per-iteration significant cell count
                                  with breakdown by (sel_type, struct_type)
  - null_histogram_mm{MM}.pdf/png  histogram of total sig-cell counts
                                   with real-genome value marked
"""

import sys
import os
import io
import re
import math
import random
import pickle
import argparse
import tempfile
import contextlib
import subprocess
from pathlib import Path
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import binomtest

# paths
BINOMIAL_DIR = Path(__file__).parent.parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))
os.chdir(BINOMIAL_DIR)

import load
from hairpins import HairpinList
from functions import find_groups, max_coverage

# CLI
parser = argparse.ArgumentParser(description="null distribution via shuffled genome (loop range 2..10).")
parser.add_argument("--mm",      type=int, default=0,   help="Mismatches in stem (default: 0)")
parser.add_argument("--n-iter",  type=int, default=1000, help="Number of iterations (default: 100)")
parser.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Parallel workers (default: cpu_count)")
parser.add_argument("--seed",    type=int, default=12345, help="Master RNG seed")
args = parser.parse_args()
MM       = args.mm
N_ITER   = args.n_iter
WORKERS  = args.workers
MASTER_S = args.seed
print(f"mm={MM}, n_iter={N_ITER}, workers={WORKERS}, master_seed={MASTER_S}")

# shared parameters
DET_STEM_MIN = 4
DET_STEM_MAX = 12
DET_GAPLIMIT = 10

LOOP_VALUES     = [2, 3, 4, 5, 6, 7, 8, 9, 10]
STEM_VALUES     = [4, 5, 6, 7, 8, 9, 10, 11, 12]
STRUCTURE_TYPES = ["tc_end", "c_end_excl", "spacer_interior", "stem"]
SELECTION_TYPES = ["most_stable", "greedy", "max_cov", "min_cov", "all"]

SIG_P = 0.05

# real-genome reference data
REAL_GENOME = load.genome.sequence
GENOME_LEN  = len(REAL_GENOME)
MUT_LIST_APOBEC    = list(load.mutations_list_APOBEC)   # 0-based, with multiplicity
MUT_CNT_APOBEC     = load.mut_cnt_APOBEC
MUT_LIST_APOBEC_EXT    = list(load.mutations_list_APOBEC_ext)
MUT_CNT_APOBEC_EXT     = load.mut_cnt_APOBEC_ext
print(f"Real genome length: {GENOME_LEN}")
print(f"Mutations APOBEC-signature (with multiplicity): {MUT_CNT_APOBEC}")
print(f"Mutations extended APOBEC-signature (with multiplicity): {MUT_CNT_APOBEC_EXT}")

# per-iteration worker
_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def _is_extendable(h, genome_seq):
    if h.start == 0 or h.end >= len(genome_seq) - 1:
        return False
    left  = genome_seq[h.start - 1].upper()
    right = genome_seq[h.end + 1].upper()
    return _COMPLEMENT.get(left) == right

# selection helpers (copied from test19, no shared mutable state)
def _most_stable(group):
    return [min(group, key=lambda x: (x.score, x.length, x.start))]

def _min_coverage(group):
    return [min(group, key=lambda x: (x.length, x.score, x.start))]

def _greedy_choose(hairpins_group, taken=None, skip_sort=False):
    if taken is None:
        taken = []
    hairpins_group_copy = deepcopy(hairpins_group)
    taken_copy = taken.copy()

    if not skip_sort:
        hairpins_group_copy.sort(key=lambda x: x.score)
        n = len(hairpins_group_copy)
        i = 0
        equal_groups = []
        while i < n:
            current = [hairpins_group_copy[i]]
            j = i + 1
            while j < n and hairpins_group_copy[i].score == hairpins_group_copy[j].score:
                current.append(hairpins_group_copy[j])
                j += 1
            current.sort(key=lambda x: x.length * -1)
            equal_groups.append(current)
            i = j
        hairpins_group_copy = equal_groups

    for p in range(len(hairpins_group_copy)):
        equal_group = hairpins_group_copy[p]
        if not equal_group:
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
                variants.append(_greedy_choose(new_hg[p:], taken_copy + [hairpin], True))
            return max(variants, key=lambda x: len(x))
        else:
            hairpin = usable[0]
            new_hg = deepcopy(hairpins_group_copy)
            new_hg[p] = [h for h in new_hg[p] if h is not hairpin]
            return _greedy_choose(new_hg[p:], taken_copy + [hairpin], True)
    return taken_copy

def _select(hairpins, sel_type):
    if sel_type == "all":
        return list(hairpins)
    groups = find_groups(list(hairpins))
    selected = []
    for group in groups:
        if   sel_type == "most_stable":  choice = _most_stable(group)
        elif sel_type == "greedy":       choice = _greedy_choose(group)
        elif sel_type == "max_cov":      choice = max_coverage(group)
        elif sel_type == "min_cov":      choice = _min_coverage(group)
        else:                            choice = _most_stable(group)
        selected.extend(choice)
    return selected

def _compute_exclusive(hairpins, struct_type, tc_pos, ga_pos, c_not_tc, g_not_ga,
                       denom_tc_ga, denom_c_not_tc, mut_list_full_APOBEC, mut_list_full_APOBEC_ext):
    if struct_type == "tc_end":
        struct_targets = set()
        for h in hairpins:
            if h.spacer_length < 2:
                continue
            l, r = h.spacer_index
            if r in tc_pos: struct_targets.add(r)
            if l in ga_pos: struct_targets.add(l)
        targets_p = len(struct_targets) / denom_tc_ga if denom_tc_ga else 0.0
    elif struct_type == "c_end_excl":
        struct_targets = set()
        for h in hairpins:
            if h.spacer_length < 1:
                continue
            l, r = h.spacer_index
            if r in c_not_tc: struct_targets.add(r)
            if l in g_not_ga: struct_targets.add(l)
        targets_p = len(struct_targets) / denom_c_not_tc if denom_c_not_tc else 0.0
    elif struct_type == "spacer_interior":
        struct_targets = set()
        for h in hairpins:
            l, r = h.spacer_index
            for pos in range(l + 1, r):
                if pos in tc_pos or pos in ga_pos:
                    struct_targets.add(pos)
        targets_p = len(struct_targets) / denom_tc_ga if denom_tc_ga else 0.0
    elif struct_type == "stem":
        struct_targets = set()
        for h in hairpins:
            ls, le = h.stem_indexes[0]
            rs, re_ = h.stem_indexes[1]
            for pos in range(ls, le + 1):
                if pos in tc_pos or pos in ga_pos:
                    struct_targets.add(pos)
            for pos in range(rs, re_ + 1):
                if pos in tc_pos or pos in ga_pos:
                    struct_targets.add(pos)
        targets_p = len(struct_targets) / denom_tc_ga if denom_tc_ga else 0.0
    else:
        return 0, 0, 0.0
    if struct_type == "c_end_excl":
        hits = sum(1 for m in mut_list_full_APOBEC_ext if m in struct_targets)
    else:
        hits = sum(1 for m in mut_list_full_APOBEC if m in struct_targets)
    return hits, len(struct_targets), targets_p

def _emboss_palindrome(genome_seq, stem_min, stem_max, gaplimit, mm, workdir):
    """Run EMBOSS palindrome on an arbitrary genome string. Returns hairpin list."""
    fasta = workdir / "genome.fna"
    with open(fasta, "w") as f:
        f.write(">random\n")
        for i in range(0, len(genome_seq), 80):
            f.write(genome_seq[i:i+80] + "\n")
    emboss_out = workdir / "emboss.txt"
    pa_out     = workdir / "pa.txt"
    subprocess.run(
        ["palindrome",
         "-sequence",      str(fasta),
         "-minpallen",     str(stem_min),
         "-maxpallen",     str(stem_max),
         "-gaplimit",      str(gaplimit),
         "-nummismatches", str(mm),
         "-outfile",       str(emboss_out),
         "-overlap"],
        check=True, capture_output=True,
    )
    pins = HairpinList.from_emboss(str(emboss_out), genome_seq)
    pins.recalculate_energy()
    pins.to_palindrome_analyzer(str(pa_out))
    return load.load_hairpins(str(pa_out))

_SIG_THRESH = 1.3  # -log10(0.05)

def _save_iter_heatmap(results, panel_stats, denom_for, iter_idx, mm, out_pdf):
    """Save the 5x4 combined heatmap for one iteration (mirrors test19 layout)."""
    finite = [v[0] for v in results.values() if not math.isinf(v[0])]
    vmin = 0.0
    vmax = math.ceil(max(finite)) if finite else 5
    if vmax <= _SIG_THRESH:
        vmax = _SIG_THRESH + 1

    colors_nonsig = plt.cm.Greys(np.linspace(0.0, 0.55, 128))
    colors_sig    = plt.cm.YlOrRd(np.linspace(0.2,  1.0, 128))
    cmap_custom   = LinearSegmentedColormap.from_list(
        "sig_cmap", np.vstack([colors_nonsig, colors_sig])
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=_SIG_THRESH, vmax=vmax)

    x_labels = [str(v) for v in LOOP_VALUES]
    y_labels = [str(v) for v in STEM_VALUES]
    n_rows, n_cols = len(STEM_VALUES), len(LOOP_VALUES)

    fontsize_tick, fontsize_ann = 7, 5
    fig, axes = plt.subplots(len(SELECTION_TYPES), len(STRUCTURE_TYPES),
                             figsize=(18, 22))
    for ri, sel_type in enumerate(SELECTION_TYPES):
        for ci, struct_type in enumerate(STRUCTURE_TYPES):
            ax = axes[ri, ci]
            mat = np.zeros((n_rows, n_cols))
            for r, stem_len in enumerate(STEM_VALUES):
                for c, loop_len in enumerate(LOOP_VALUES):
                    lp = results[(sel_type, struct_type, stem_len, loop_len)][0]
                    mat[r, c] = min(lp, vmax)
            im = ax.imshow(mat, aspect="auto", origin="lower",
                           cmap=cmap_custom, norm=norm)
            ax.set_xticks(range(n_cols)); ax.set_xticklabels(x_labels, fontsize=fontsize_tick)
            ax.set_yticks(range(n_rows)); ax.set_yticklabels(y_labels, fontsize=fontsize_tick)
            ax.set_xlabel("Loop length (nt)", fontsize=fontsize_tick)
            ax.set_ylabel("Stem length (bp)", fontsize=fontsize_tick)
            ph, pt, pp0, plp = panel_stats[(sel_type, struct_type)]
            plp_str = f"{plp:.2f}" if not math.isinf(plp) else "inf"
            ax.set_title(
                f"{struct_type} / {sel_type}\n"
                f"All: {ph}/{pt}  p₀={pp0:.3g}  −log₁₀p={plp_str}",
                fontsize=fontsize_tick + 1, fontweight="bold"
            )
            for r, stem_len in enumerate(STEM_VALUES):
                for c, loop_len in enumerate(LOOP_VALUES):
                    lp, hits, n_struct, _, pv = results[
                        (sel_type, struct_type, stem_len, loop_len)
                    ]
                    lp_str = f"{lp:.2f}" if not math.isinf(lp) else "inf"
                    ann = f"{hits}/{n_struct}\n{lp_str}"
                    text_color = ("white"
                                  if lp > _SIG_THRESH + (vmax - _SIG_THRESH) * 0.5
                                  else "black")
                    ax.text(c, r, ann, ha="center", va="center",
                            fontsize=fontsize_ann, fontweight="bold",
                            color=text_color, linespacing=1.2)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("−log₁₀(p)", fontsize=fontsize_tick - 1)
            cbar.ax.tick_params(labelsize=fontsize_tick - 1)
            cbar.ax.axhline(y=_SIG_THRESH, color="black", linewidth=1.0, linestyle="--")
            cbar.ax.text(2.6, _SIG_THRESH, "p=0.05", va="center",
                         fontsize=fontsize_tick - 2, color="black")
    fig.suptitle(
        f"Null iteration {iter_idx} — shuffled genome (mm={mm}, loop≤10)\n"
        f"Detection: stem={DET_STEM_MIN}-{DET_STEM_MAX}, gaplimit={DET_GAPLIMIT}    "
        f"Cell: hits/targets | −log₁₀(p)    Color: −log₁₀(p-value)",
        fontsize=12, y=1.002,
    )
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_iteration(payload):
    """Run one null iteration. payload = (iter_idx, seed, mm, real_genome,
    mut_list_full, mut_cnt, iter_dir)."""
    (iter_idx, seed, mm, real_genome, mut_list_full_APOBEC, mut_cnt_APOBEC, mut_list_full_APOBEC_ext, mut_cnt_APOBEC_ext, iter_dir) = payload
    rng = random.Random(seed)
    bases = list(real_genome)
    rng.shuffle(bases)
    genome = "".join(bases)

    # target sets for the shuffled genome
    tc_pos    = set(m.end() - 1 for m in re.finditer("TC", genome))
    ga_pos    = set(m.start()   for m in re.finditer("GA", genome))
    all_c     = set(m.start()   for m in re.finditer("C",  genome))
    all_g     = set(m.start()   for m in re.finditer("G",  genome))
    c_not_tc  = all_c - tc_pos
    g_not_ga  = all_g - ga_pos
    denom_tc_ga    = len(tc_pos | ga_pos)
    denom_c_not_tc = len(c_not_tc | g_not_ga)
    denom_for = {
        "tc_end":          denom_tc_ga,
        "c_end_excl":      denom_c_not_tc,
        "spacer_interior": denom_tc_ga,
        "stem":            denom_tc_ga,
    }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        all_hairpins = _emboss_palindrome(
            genome, DET_STEM_MIN, DET_STEM_MAX, DET_GAPLIMIT, mm, workdir
        )

    broad_pool = [h for h in all_hairpins if not _is_extendable(h, genome)]
    n_total      = len(all_hairpins)
    n_nonextend  = len(broad_pool)

    # full results dict: (sel_type, struct_type, stem_len, loop_len)
    #                  -> (log10p, hits, n_struct, targets_p, pvalue)
    results = {}
    sig_by_panel = {(s, t): 0 for s in SELECTION_TYPES for t in STRUCTURE_TYPES}
    total_sig = 0

    for sel_type in SELECTION_TYPES:
        pool = _select(broad_pool, sel_type)
        for struct_type in STRUCTURE_TYPES:
            n_sig = 0
            for stem_len in STEM_VALUES:
                for loop_len in LOOP_VALUES:
                    cell = [h for h in pool
                            if h.stem_length == stem_len
                            and h.spacer_length == loop_len]
                    if not cell:
                        results[(sel_type, struct_type, stem_len, loop_len)] = (
                            0.0, 0, 0, 0.0, 1.0
                        )
                        continue
                    hits, n_struct, targets_p = _compute_exclusive(
                        cell, struct_type,
                        tc_pos, ga_pos, c_not_tc, g_not_ga,
                        denom_tc_ga, denom_c_not_tc, mut_list_full_APOBEC, mut_list_full_APOBEC_ext
                    )
                    if struct_type == "c_end_excl":
                        if targets_p > 0:
                            pv = binomtest(hits, mut_cnt_APOBEC_ext, targets_p,
                                       alternative='greater').pvalue
                        else:
                            pv = 1.0
                    else:
                        if targets_p > 0:
                            pv = binomtest(hits, mut_cnt_APOBEC, targets_p,
                                       alternative='greater').pvalue
                        else:
                            pv = 1.0
                    lp = -math.log10(pv) if pv > 0 else float('inf')
                    results[(sel_type, struct_type, stem_len, loop_len)] = (
                        lp, hits, n_struct, targets_p, pv
                    )
                    if pv < SIG_P:
                        n_sig += 1
            sig_by_panel[(sel_type, struct_type)] = n_sig
            total_sig += n_sig

    # panel-level combined stats 
    panel_stats = {}
    for s in SELECTION_TYPES:
        for t in STRUCTURE_TYPES:
            if t == "c_end_excl":
                mutcnt = mut_cnt_APOBEC_ext
            else:
                mutcnt = mut_cnt_APOBEC
            non_empty = [results[(s, t, sl, ll)]
                         for sl in STEM_VALUES for ll in LOOP_VALUES
                         if results[(s, t, sl, ll)][2] > 0]
            ph = sum(v[1] for v in non_empty)
            pt = sum(v[2] for v in non_empty)
            gd = denom_for[t]
            p0 = min(pt / gd, 1.0) if gd > 0 else 0.0
            k  = min(ph, mutcnt)
            if p0 > 0:
                pv = binomtest(k, mutcnt, p0, alternative='greater').pvalue
            else:
                pv = 1.0
            lp = -math.log10(pv) if pv > 0 else float('inf')
            panel_stats[(s, t)] = (ph, pt, p0, lp)

    # save heatmap for this iteration
    out_pdf = Path(iter_dir) / f"iter_{iter_idx:04d}_mm{mm}.pdf"
    _save_iter_heatmap(results, panel_stats, denom_for, iter_idx, mm, out_pdf)

    return {
        "iter":         iter_idx,
        "seed":         seed,
        "total_hairpins":     n_total,
        "nonextendable":      n_nonextend,
        "total_sig_cells":    total_sig,
        "sig_by_panel":       sig_by_panel,
    }

# main
def main():
    out_dir  = Path(__file__).parent
    iter_dir = out_dir / f"iterations_mm{MM}"
    iter_dir.mkdir(exist_ok=True)
    print(f"Per-iteration PDFs will be saved to: {iter_dir}")

    master_rng = random.Random(MASTER_S)
    payloads = []
    for i in range(N_ITER):
        seed = master_rng.randrange(2**31)
        payloads.append((i, seed, MM, REAL_GENOME, MUT_LIST_APOBEC, MUT_CNT_APOBEC, MUT_LIST_APOBEC_EXT, MUT_CNT_APOBEC_EXT, str(iter_dir)))
    results = []
    print(f"\nRunning {N_ITER} iterations on {WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_iteration, p) for p in payloads]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"  [{done:>4}/{N_ITER}] iter={r['iter']:>4}  "
                  f"hairpins={r['total_hairpins']:>6}  "
                  f"sig_cells={r['total_sig_cells']:>4}", flush=True)

    results.sort(key=lambda x: x["iter"])

    # write per-iteration TSV
    panel_cols = [f"{s}__{t}" for s in SELECTION_TYPES for t in STRUCTURE_TYPES]
    tsv_path = out_dir / f"null_sig_counts_mm{MM}.tsv"
    with open(tsv_path, "w") as f:
        cols = ["iter", "seed", "total_hairpins", "nonextendable",
                "total_sig_cells"] + panel_cols
        f.write("\t".join(cols) + "\n")
        for r in results:
            row = [r["iter"], r["seed"], r["total_hairpins"],
                   r["nonextendable"], r["total_sig_cells"]]
            for s in SELECTION_TYPES:
                for t in STRUCTURE_TYPES:
                    row.append(r["sig_by_panel"][(s, t)])
            f.write("\t".join(str(v) for v in row) + "\n")
    print(f"\nSaved: {tsv_path.name}")

    # save raw 
    with open(out_dir / f"null_results_mm{MM}.pkl", "wb") as f:
        pickle.dump({
            "results":           results,
            "N_ITER":            N_ITER,
            "MM":                MM,
            "LOOP_VALUES":       LOOP_VALUES,
            "STEM_VALUES":       STEM_VALUES,
            "STRUCTURE_TYPES":   STRUCTURE_TYPES,
            "SELECTION_TYPES":   SELECTION_TYPES,
            "DET_STEM_MIN":      DET_STEM_MIN,
            "DET_STEM_MAX":      DET_STEM_MAX,
            "DET_GAPLIMIT":      DET_GAPLIMIT,
            "GENOME_LEN":        GENOME_LEN,
            "MUT_CNT_APOBEC":   MUT_CNT_APOBEC,
            "MUT_CNT_APOBEC_EXT":   MUT_CNT_APOBEC_EXT,
            "MASTER_SEED":       MASTER_S,
        }, f)

    # summary stats
    counts = np.array([r["total_sig_cells"] for r in results])
    total_cells = (len(SELECTION_TYPES) * len(STRUCTURE_TYPES)
                   * len(STEM_VALUES) * len(LOOP_VALUES))
    print(f"\nTotal cells per iteration: {total_cells}")
    print(f"Null distribution of total significant cells (p<{SIG_P}):")
    print(f"  mean   = {counts.mean():.2f}")
    print(f"  median = {np.median(counts):.2f}")
    print(f"  stdev  = {counts.std(ddof=1):.2f}")
    print(f"  min    = {counts.min()}")
    print(f"  max    = {counts.max()}")
    print(f"  95 %tl = {np.percentile(counts, 95):.1f}")
    print(f"  99 %tl = {np.percentile(counts, 99):.1f}")

    # histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_edges = np.arange(int(counts.min()), int(counts.max()) + 2)
    ax.hist(counts, bins=bin_edges, #bins=max(10, min(40, len(set(counts)))),
            color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(counts.mean(), color="red", linestyle="--",
               label=f"mean={counts.mean():.1f}")
    ax.set_xlabel(f"Number of significant cells per genome (p < {SIG_P})")
    ax.set_ylabel("Number of shuffled genomes")
    ax.set_title(
        f"Null distribution under shuffled genome (n={N_ITER}, mm={MM})\n"
        f"Total cells per panel grid = {total_cells} "
        f"(5 sel × 4 struct × {len(STEM_VALUES)} stem × {len(LOOP_VALUES)} loop)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"null_histogram_mm{MM}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(out_dir / f"null_histogram_mm{MM}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: null_histogram_mm{MM}.pdf / .png")

if __name__ == "__main__":
    main()
