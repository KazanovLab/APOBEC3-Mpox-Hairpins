"""
Number of hairpins produced by EMBOSS palindrome vs stem_max_length.

Loops stem_max_length from 30 to 50, all other parameters taken from
binomial/config.yaml. For each value:
  - Method 1: count hairpins via HairpinList.from_emboss()  (hairpins.py)
  - Method 2: convert to PA file, then count via load_hairpins() (load.py)
If counts differ, a diagnostic message is printed but plotting proceeds
using Method 1 counts.
"""

import sys
import os
from pathlib import Path

# paths
BINOMIAL_DIR = Path(__file__).parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))

import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def _genome_coverage_pct(hairpin_list, genome_length):
    """Fraction of genome covered by hairpins (merged intervals), as percent."""
    if not hairpin_list:
        return 0.0
    intervals = sorted((h.Start, h.End) for h in hairpin_list)
    covered = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s < cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    covered += cur_e - cur_s
    return covered * 100 / genome_length

# emboss.py / hairpins.py open config.yaml and nn_coefs/ via relative paths,
# so we must run from inside the binomial directory.
os.chdir(BINOMIAL_DIR)

# load config
with open("config.yaml") as f:
    params = yaml.safe_load(f)

GENOME_PATH    = params["genome_path"]
STEM_MIN       = params["stem_min_length"]
LOOP_LENGTH    = params["loop_length"]
NUM_MISMATCHES = params["number_mismatches"]

# genome sequence
with open(GENOME_PATH) as f:
    f.readline()                                  # skip FASTA header
    GENOME_SEQ = f.read().replace("\n", "")

# imports from package dir
from emboss   import get_palindrome
from hairpins import HairpinList

# Inline load_hairpins + its Hairpin class from load.py to avoid
# the module-level side effects in load.py (genome loading, mutations, etc.)
import re

class _LoadHairpin:
    """Hairpin class copied from load.py for PA-format reading."""
    def __init__(self, l_s_m, position, score, palindrome):
        lsm = list(map(int, l_s_m.split("-")))
        self.spacer_length = lsm[1]
        self.stem_length   = lsm[0]
        self.miss          = lsm[2]
        self.length        = self.spacer_length + self.stem_length * 2
        self.position      = position
        self.score         = score
        self.palindrome    = palindrome

def _load_hairpins(pa_file_path):
    """load_hairpins() from load.py, using _LoadHairpin."""
    hairpins = []
    path = str(pa_file_path)
    if not os.path.exists(path):
        print(f"  [load_hairpins] File not found: {path}")
        return None
    with open(path) as fh:
        fh.readline()                             # skip header
        for line in fh:
            parts = line.split()
            h = _LoadHairpin(parts[0], int(parts[1]), float(parts[2]),
                             " ".join(parts[3:]))
            hairpins.append(h)
    return hairpins

# main loop
stem_max_values = list(range(6, 51))
emboss_dir      = BINOMIAL_DIR / "emboss"

GENOME_LENGTH = len(GENOME_SEQ)

def run_series(stem_min, num_mismatches):
    """Run the full stem_max loop for given stem_min and mismatch count.
    Returns (counts, coverages) using method 1, cross-checked with method 2."""
    counts    = []
    coverages = []
    label     = f"stem_min={stem_min}, mismatches={num_mismatches}"
    print(f"\n── {label} ──")
    print(f"{'stem_max':>8}  {'method1':>9}  {'method2':>9}  {'match':>5}  {'cov%':>7}")
    print("-" * 52)
    for stem_max in stem_max_values:
        pa_file     = get_palindrome(stem_min, stem_max, LOOP_LENGTH, num_mismatches)
        tag         = f"mm{num_mismatches}_stemin{stem_min}_stemax{stem_max}_spacer{LOOP_LENGTH}"
        emboss_file = emboss_dir / f"emboss_{tag}.txt"

        pins_m1 = HairpinList.from_emboss(str(emboss_file), GENOME_SEQ)
        n_m1    = len(pins_m1)
        cov     = _genome_coverage_pct(pins_m1, GENOME_LENGTH)

        pins_m2 = _load_hairpins(pa_file)
        if pins_m2 is None:
            n_m2, match = None, "N/A"
        else:
            n_m2  = len(pins_m2)
            match = "OK" if n_m1 == n_m2 else "DIFF"

        if match == "DIFF":
            print(f"  [DIAGNOSTIC] stem_max={stem_max}: "
                  f"method1={n_m1} hairpins vs method2={n_m2} hairpins")

        print(f"{stem_max:>8}  {n_m1:>9}  {str(n_m2):>9}  {match:>5}  {cov:>6.2f}%")
        counts.append(n_m1)
        coverages.append(cov)
    return counts, coverages

counts_min6_mm1, coverages_min6_mm1 = run_series(STEM_MIN, NUM_MISMATCHES)

# plot
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(stem_max_values, counts_min6_mm1,
        marker="o", markersize=4, linewidth=1.5, color="#3389C0")

ax2 = ax.twinx()
ax2.plot(stem_max_values, coverages_min6_mm1,
         marker="s", markersize=4, linewidth=1.5, color="#D74050", linestyle="--")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Genome coverage (%)", fontsize=12, color="#D74050")
ax2.tick_params(axis="y", labelcolor="#D74050", labelsize=10)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

ax.set_xlabel("Maximum stem length (bp)", fontsize=12)
ax.set_ylabel("Number of hairpins", fontsize=12, color="#3389C0")
#ax.set_title("EMBOSS palindrome hairpin count vs maximum stem length\n"
#             f"(stem_min={STEM_MIN}, loop={LOOP_LENGTH}, mismatches={NUM_MISMATCHES})",
#              fontsize=11)

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()

out_dir = Path(__file__).parent
plt.savefig(out_dir / "hairpins_vs_stem_max.pdf", dpi=150)
plt.savefig(out_dir / "hairpins_vs_stem_max.png", dpi=600)
print("\nPlot saved: hairpins_vs_stem_max.pdf / .png")
