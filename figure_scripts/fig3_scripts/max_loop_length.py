"""
Number of hairpins produced by EMBOSS palindrome vs loop_length.

Loops loop_length from 2 to 50, all other parameters taken from
binomial/config.yaml. For each value, hairpins are counted by parsing
the EMBOSS output file directly (no energy calculation), which avoids
the nn_energy() limitation of loops <= 30 nt.
"""

import sys
import os
import subprocess
from pathlib import Path

# paths 
BINOMIAL_DIR = Path(__file__).parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))

import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.chdir(BINOMIAL_DIR)

# load config
with open("config.yaml") as f:
    params = yaml.safe_load(f)

GENOME_PATH    = params["genome_path"]
STEM_MIN       = params["stem_min_length"]
STEM_MAX       = params["stem_max_length"]
NUM_MISMATCHES = params["number_mismatches"]

# genome length
with open(GENOME_PATH) as f:
    f.readline()
    GENOME_LENGTH = len(f.read().replace("\n", ""))

# parse EMBOSS output: count hairpins and collect intervals
def parse_emboss(filepath):
    """Return (count, intervals) from an EMBOSS palindrome output file.
    intervals: list of (start, end) tuples, 0-based with end exclusive."""
    with open(filepath) as f:
        lines = f.readlines()
    intervals = []
    for i in range(12, len(lines), 4):
        if lines[i].strip():
            pos = int(lines[i].split()[0])
            end = int(lines[i + 2].split()[0])
            intervals.append((pos - 1, end))   # 0-based, end exclusive
    return len(intervals), intervals

def _genome_coverage_pct(intervals, genome_length):
    """Fraction of genome covered by merged intervals, as percent."""
    if not intervals:
        return 0.0
    sorted_ivs = sorted(intervals)
    covered = 0
    cur_s, cur_e = sorted_ivs[0]
    for s, e in sorted_ivs[1:]:
        if s < cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    covered += cur_e - cur_s
    return covered * 100 / genome_length

# run EMBOSS and count
loop_values = list(range(2, 51))
emboss_dir  = BINOMIAL_DIR / "emboss"
emboss_dir.mkdir(exist_ok=True)

print(f"Parameters: stem_min={STEM_MIN}, stem_max={STEM_MAX}, mismatches={NUM_MISMATCHES}")
print(f"{'loop_len':>8}  {'count':>9}  {'cov%':>7}")
print("-" * 32)

counts    = []
coverages = []
for loop_len in loop_values:
    tag         = f"mm{NUM_MISMATCHES}_stemin{STEM_MIN}_stemax{STEM_MAX}_spacer{loop_len}"
    emboss_file = emboss_dir / f"emboss_{tag}.txt"

    if not emboss_file.exists():
        subprocess.run(
            [
                "palindrome",
                "-sequence",      str(GENOME_PATH),
                "-minpallen",     str(STEM_MIN),
                "-maxpallen",     str(STEM_MAX),
                "-gaplimit",      str(loop_len),
                "-nummismatches", str(NUM_MISMATCHES),
                "-outfile",       str(emboss_file),
                "-overlap",
            ],
            check=True, capture_output=True,
        )

    n, intervals = parse_emboss(str(emboss_file))
    cov = _genome_coverage_pct(intervals, GENOME_LENGTH)
    print(f"{loop_len:>8}  {n:>9}  {cov:>6.2f}%")
    counts.append(n)
    coverages.append(cov)

# plot
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(loop_values, counts,
        marker="o", markersize=4, linewidth=1.5, color="#3389C0")

ax2 = ax.twinx()
ax2.plot(loop_values, coverages,
         marker="s", markersize=4, linewidth=1.5, color="#D74050", linestyle="--")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Genome coverage (%)", fontsize=12, color="#D74050")
ax2.tick_params(axis="y", labelcolor="#D74050", labelsize=10)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

ax.set_xlabel("Maximum loop length (nt)", fontsize=12)
ax.set_ylabel("Number of hairpins", fontsize=12, color="#3389C0")
#ax.set_title("EMBOSS palindrome hairpin count vs maximum loop length\n"
#             f"(stem_min={STEM_MIN}, stem_max={STEM_MAX}, mismatches={NUM_MISMATCHES})",
#             fontsize=11)

ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()

out_dir = Path(__file__).parent
plt.savefig(out_dir / "hairpins_vs_loop_len.pdf", dpi=150)
plt.savefig(out_dir / "hairpins_vs_loop_len.png", dpi=600)
print("\nPlot saved: hairpins_vs_loop_len.pdf / .png")
