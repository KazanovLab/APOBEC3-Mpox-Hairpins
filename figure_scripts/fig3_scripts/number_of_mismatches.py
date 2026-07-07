"""
Number of hairpins and genome coverage for mismatches = 0, 1, 2.

All other parameters taken from binomial/config.yaml (stem_min=6, stem_max=30,
loop=10). Results presented as a bar chart with a secondary y-axis for genome
coverage percentage.
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
import numpy as np

os.chdir(BINOMIAL_DIR)

# load config
with open("config.yaml") as f:
    params = yaml.safe_load(f)

GENOME_PATH = params["genome_path"]
STEM_MIN    = params["stem_min_length"]
STEM_MAX    = params["stem_max_length"]
LOOP_LENGTH = params["loop_length"]

# genome length
with open(GENOME_PATH) as f:
    f.readline()
    GENOME_LENGTH = len(f.read().replace("\n", ""))

# parse EMBOSS output
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

def genome_coverage_pct(intervals, genome_length):
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

# main loop
mismatch_values = [0, 1, 2]
emboss_dir      = BINOMIAL_DIR / "emboss"
emboss_dir.mkdir(exist_ok=True)

print(f"Fixed: stem_min={STEM_MIN}, stem_max={STEM_MAX}, loop={LOOP_LENGTH}")
print(f"{'mismatches':>10}  {'count':>9}  {'cov%':>7}")
print("-" * 35)

counts    = []
coverages = []
for mm in mismatch_values:
    tag         = f"mm{mm}_stemin{STEM_MIN}_stemax{STEM_MAX}_spacer{LOOP_LENGTH}"
    emboss_file = emboss_dir / f"emboss_{tag}.txt"

    if not emboss_file.exists():
        subprocess.run(
            [
                "palindrome",
                "-sequence",      str(GENOME_PATH),
                "-minpallen",     str(STEM_MIN),
                "-maxpallen",     str(STEM_MAX),
                "-gaplimit",      str(LOOP_LENGTH),
                "-nummismatches", str(mm),
                "-outfile",       str(emboss_file),
                "-overlap",
            ],
            check=True, capture_output=True,
        )

    n, intervals = parse_emboss(str(emboss_file))
    cov = genome_coverage_pct(intervals, GENOME_LENGTH)
    print(f"{mm:>10}  {n:>9,}  {cov:>6.2f}%")
    counts.append(n)
    coverages.append(cov)

# plot
x      = np.arange(len(mismatch_values))
labels = [f"{v}" for v in mismatch_values]
bar_w  = 0.45
color_count = "#3389C0"
color_cov   = "#D74050"

fig, ax = plt.subplots(figsize=(6, 4))
ax2 = ax.twinx()

bars = ax.bar(x, counts, width=bar_w, color=color_count, alpha=0.85,
              label="Hairpin count")
ax2.bar(x + bar_w, coverages, width=bar_w, color=color_cov, alpha=0.85, 
        label="Genome coverage (%)")

for bar, n in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
            f"{n:,}", ha="center", va="bottom", fontsize=9, color=color_count)
for xi, cov in zip(x + bar_w, coverages):
    ax2.text(xi, cov + 0.8, f"{cov:.1f}%",
             ha="center", va="bottom", fontsize=9, color=color_cov)

ax.set_xticks(x + bar_w / 2)
ax.set_xticklabels(labels, fontsize=11)
ax.set_xlabel("Number of mismatches", fontsize=12)
ax.set_ylabel("Number of hairpins", fontsize=12, color=color_count)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.tick_params(axis="y", labelsize=10)
ax.set_ylim(0, max(counts) * 1.15)

ax2.set_ylim(0, 100)
ax2.set_ylabel("Genome coverage (%)", fontsize=12, color=color_cov)
ax2.tick_params(axis="y", labelcolor=color_cov, labelsize=10)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

#ax.set_title(f"Hairpin count and genome coverage vs number of mismatches\n"
#             f"(stem_min={STEM_MIN}, stem_max={STEM_MAX}, loop={LOOP_LENGTH})",
#             fontsize=11)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper left")

ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

out_dir = Path(__file__).parent
plt.savefig(out_dir / "hairpins_vs_mismatches.pdf", dpi=150)
plt.savefig(out_dir / "hairpins_vs_mismatches.png", dpi=600)
print("\nPlot saved: hairpins_vs_mismatches.pdf / .png")
