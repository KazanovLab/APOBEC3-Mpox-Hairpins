"""
Extract all hairpins with loop length 3 or 4 and any stem
length ≥ 5 bp, including unwound representations. Run across three mutation
datasets.

A hairpin can be reported by EMBOSS as
(target_stem + k, target_loop - 2*k) for k = 0, 1, 2, ... — k outermost stem
base pairs treated as un-paired ("unwound"). For target_loop in {3, 4}:
  - target_loop=3 supports k=0 (loop=3 detection) and k=1 (loop=1 detection)
  - target_loop=4 supports k=0 (loop=4), k=1 (loop=2), and k=2 (loop=0,
    i.e. perfect IRs with no loop)

Constraints applied:
  - target_stem ≥ 5
  - non-extendable-stem filter on the detected hairpin
  - TC at the 3' end of the loop OR GA at the 5' end

Pipeline:
  1. Broad EMBOSS detection: stem_min=5, stem_max=30, gaplimit=4, mm=MM.
     (gaplimit=4 covers all needed detection loop lengths: 0,1,2,3,4.)
     Run once - the hairpin pool is independent of the mutation dataset.
  2. Non-extendable-stem filter (flanks not Watson-Crick complementary).
  3. For each hairpin matching one of the allowed (detected_stem, detected_loop)
     shapes, build a row with its (target_stem, target_loop) view
     and apply the TC 3' / GA 5' filter on the loop.
  4. For each of the three mutation datasets, annotate the matching hairpins
     with their per-dataset mutation hits and write a separate TSV.

Output: hairpins_loop3_or_4_stem_ge5_mm{MM}.tsv  — one file with shared
        hairpin columns and per-dataset mutation columns (suffixed by tag).
"""

import sys
import os
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

# paths 
PACKAGE_DIR = Path(__file__).parent.parent / "package"
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
sys.path.insert(0, str(PACKAGE_DIR))
os.chdir(PACKAGE_DIR)

import load
from emboss import get_palindrome

# Bypass the looplen filter
load.params["filter_looplen_lessequal"] = -1

DATASETS = [
    ("dataset1",              DATASETS_DIR / "dataset1.substitution_table.tsv"),
    ("dataset2_west_africa",  DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv"),
    ("dataset3_gisaid",       DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv"),
]

# CLI
parser = argparse.ArgumentParser(description="Test 24: extract loop=3/4 hairpins, any stem > 5, including unwound.")
parser.add_argument("--mm", type=int, default=0,
                    help="Mismatches in stem (default: 0)")
args = parser.parse_args()
MM = args.mm
print(f"Running with mismatches = {MM}")

# detection params
DET_STEM_MIN = 5     # target_stem >= 5
DET_STEM_MAX = 30    # loop upper bound
DET_GAPLIMIT = 4     # covers detection loops 0,1,2,3,4

TARGET_STEM_MIN = 5  # "stem length ≥ 5"
TARGET_LOOPS    = (3, 4)
MAX_K           = {3: 1, 4: 2}   # max unwinding steps per target loop

# genome
_genome_seq = load.genome.sequence

# non-extendable-stem filter
_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

def is_extendable(h):
    if h.start == 0 or h.end >= len(_genome_seq) - 1:
        return False
    left  = _genome_seq[h.start - 1].upper()
    right = _genome_seq[h.end + 1].upper()
    return _COMPLEMENT.get(left) == right

# per-dataset mutation loader
def load_mutations(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["is_APOBEC"] == True]
    mut_list = []
    for _, row in df.iterrows():
        pos = int(row["pos_ref"]) - 1            # 1-based -> 0-based
        count = int(row["independent_events"])
        mut_list.extend([pos] * count)
    counts = Counter(mut_list)
    sorted_unique = sorted(counts)
    return mut_list, counts, sorted_unique

def hairpin_mutations(h, mut_sorted):
    return [m for m in mut_sorted if h.start <= m <= h.end]

# hairpin detection 
print(f"Hairpin detection (stem={DET_STEM_MIN}-{DET_STEM_MAX}, "
      f"gaplimit={DET_GAPLIMIT}, mm={MM})...")
pa_file = get_palindrome(DET_STEM_MIN, DET_STEM_MAX, DET_GAPLIMIT, MM)
all_hairpins = load.load_hairpins(str(pa_file))
print(f"  total hairpins: {len(all_hairpins)}")

non_ext = [h for h in all_hairpins if not is_extendable(h)]
print(f"  non-extendable: {len(non_ext)}")

# A detected hairpin matches if there exists a (target_loop, k) such that:
#   target_stem  = detected_stem - k     and  target_stem >= TARGET_STEM_MIN
#   target_loop  = detected_loop + 2*k   and  target_loop in TARGET_LOOPS
def conceptual_view(h):
    s, l = h.stem_length, h.spacer_length
    for L in TARGET_LOOPS:
        if l > L:
            continue
        k = (L - l) // 2
        if (L - l) % 2 != 0:
            continue
        if k > MAX_K[L]:
            continue
        target_stem = s - k
        if target_stem < TARGET_STEM_MIN:
            continue
        ul = h.stem_indexes[0][1] - (k - 1) if k > 0 else h.spacer_index[0]
        ur = h.stem_indexes[1][0] + (k - 1) if k > 0 else h.spacer_index[1]
        return target_stem, L, k, ul, ur
    return None

rows = []
counts_by_target = Counter()
counts_by_k      = Counter()
for h in non_ext:
    view = conceptual_view(h)
    if view is None:
        continue
    target_stem, target_loop, k, ul, ur = view
    loop_5p = _genome_seq[ul:ul + 2].upper()
    loop_3p = _genome_seq[ur - 1:ur + 1].upper()
    # TC at 3' end of conceptual loop OR GA at 5' end
    if loop_3p != "TC" and loop_5p != "GA":
        continue
    rows.append({
        "h": h,
        "target_stem": target_stem, "target_loop": target_loop, "k": k,
        "loop_l": ul, "loop_r": ur,
        "loop_5p": loop_5p, "loop_3p": loop_3p,
    })
    counts_by_target[(target_stem, target_loop)] += 1
    counts_by_k[k] += 1

print(f"\nHairpins with target_loop ∈ {TARGET_LOOPS}, target_stem ≥ {TARGET_STEM_MIN}, "
      f"TC@3' or GA@5' on conceptual loop: {len(rows)}")
print(f"  by unwinding depth: " +
      ", ".join(f"k={k}: {counts_by_k[k]}" for k in sorted(counts_by_k)))
print("\nBreakdown by (target_stem, target_loop):")
for (s, l) in sorted(counts_by_target):
    print(f"  stem={s:>2} loop={l}: {counts_by_target[(s, l)]}")

rows.sort(key=lambda r: (r["target_stem"], r["target_loop"], r["h"].start))

# load mutations for each dataset
dataset_muts = [] 
for tag, tsv_path in DATASETS:
    print(f"\n— dataset: {tag} ({tsv_path.name})")
    mut_list, mut_counts, mut_sorted = load_mutations(tsv_path)
    print(f"  APOBEC mutations (with multiplicity): {len(mut_list)} "
          f"({len(mut_sorted)} unique positions)")
    dataset_muts.append((tag, mut_counts, mut_sorted))

# save one combined TSV: hairpin columns + per-dataset mutation columns
out_dir = Path(__file__).parent
shared_cols = [
    "target_stem_length", "target_loop_length",
    "detected_stem_length", "detected_loop_length",
    "unwound_steps",
    "is_unwound",
    "hairpin_start_1b", "hairpin_end_1b",
    "loop_start_1b", "loop_end_1b",
    "hairpin_length", "energy",
    "sequence",
    "loop_5p_dinuc", "loop_3p_dinuc", "tc_3p", "ga_5p",
]
per_dataset_cols = ["n_mut_positions", "n_mut_observations", "mutation_positions_1b"]
cols = list(shared_cols)
for tag, _, _ in dataset_muts:
    cols.extend(f"{c}__{tag}" for c in per_dataset_cols)

out_tsv = out_dir / f"hairpins_loop3_or_4_stem_ge5_mm{MM}.tsv"
with open(out_tsv, "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in rows:
        h = r["h"]
        shared_vals = [
            r["target_stem"], r["target_loop"],
            h.stem_length, h.spacer_length,
            r["k"], int(r["k"] > 0),
            h.start + 1, h.end + 1,
            r["loop_l"] + 1, r["loop_r"] + 1,
            h.length, h.score,
            h.sequence,
            r["loop_5p"], r["loop_3p"],
            int(r["loop_3p"] == "TC"), int(r["loop_5p"] == "GA"),
        ]
        per_dataset_vals = []
        for tag, mut_counts, mut_sorted in dataset_muts:
            muts = hairpin_mutations(h, mut_sorted)
            per_dataset_vals.extend([
                len(muts),
                sum(mut_counts[m] for m in muts),
                ",".join(str(m + 1) for m in muts),
            ])
        f.write("\t".join(str(v) for v in shared_vals + per_dataset_vals) + "\n")
print(f"\nSaved: {out_tsv.name}  ({len(rows)} hairpins, "
      f"{len(dataset_muts)} datasets as columns)")
