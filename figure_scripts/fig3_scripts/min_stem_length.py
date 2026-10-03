"""
Structure count and genome coverage vs minimum stem length
for both mismatch=0 and mismatch=1 on the same plot.

All other parameters taken from binomial/config.yaml (stem_max=30, loop=10).
"""

import sys
import os
import subprocess
from pathlib import Path

BINOMIAL_DIR = Path(__file__).parent.parent.parent / "package"
sys.path.insert(0, str(BINOMIAL_DIR))

import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

os.chdir(BINOMIAL_DIR)

with open("config.yaml") as f:
    params = yaml.safe_load(f)

GENOME_PATH = params["genome_path"]
STEM_MAX    = params["stem_max_length"]
LOOP_LENGTH = params["loop_length"]

with open(GENOME_PATH) as f:
    f.readline()
    GENOME_LENGTH = len(f.read().replace("\n", ""))


def parse_emboss(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    intervals = []
    for i in range(12, len(lines), 4):
        if lines[i].strip():
            pos = int(lines[i].split()[0])
            end = int(lines[i + 2].split()[0])
            intervals.append((pos - 1, end))
    return len(intervals), intervals


def genome_coverage_pct(intervals, genome_length):
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


stem_min_values  = [3, 4, 5, 6]
mismatch_values  = [0, 1]
# mm=1 with stem_min=3 is excluded (impractically large)
mm_stem_excluded = {(1, 3)}
emboss_dir       = BINOMIAL_DIR / "emboss"
emboss_dir.mkdir(exist_ok=True)

# results[mm][stem_min] = (count, coverage)
results = {}
for mm in mismatch_values:
    results[mm] = {}
    print(f"\nmismatches={mm}  (stem_max={STEM_MAX}, loop={LOOP_LENGTH})")
    print(f"{'stem_min':>8}  {'count':>9}  {'cov%':>7}")
    print("-" * 32)
    for stem_min in stem_min_values:
        if (mm, stem_min) in mm_stem_excluded:
            continue
        tag         = f"mm{mm}_stemin{stem_min}_stemax{STEM_MAX}_spacer{LOOP_LENGTH}"
        emboss_file = emboss_dir / f"emboss_{tag}.txt"
        if not emboss_file.exists():
            subprocess.run(
                [
                    "palindrome",
                    "-sequence",      str(GENOME_PATH),
                    "-minpallen",     str(stem_min),
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
        print(f"{stem_min:>8}  {n:>9,}  {cov:>6.2f}%")
        results[mm][stem_min] = (n, cov)

# plot
n_groups  = len(stem_min_values)
x         = np.arange(n_groups)
bar_w     = 0.20

# colours: count (blue family), coverage (red family)
colors = {
    (0, "count"): "#9AD499",
    (1, "count"): "#3389C0",
    (0, "cov"):   "#FC8D58",
    (1, "cov"):   "#D74050",
}

offsets = {
    (0, "count"): -1.5 * bar_w,
    (1, "count"): -0.5 * bar_w,
    (0, "cov"):    0.5 * bar_w,
    (1, "cov"):    1.5 * bar_w,
}

fig, ax = plt.subplots(figsize=(9, 5))
ax2 = ax.twinx()

all_counts = [
    results[mm][s][0]
    for mm in mismatch_values
    for s in stem_min_values
    if (mm, s) not in mm_stem_excluded
]

for mm in mismatch_values:
    stem_mins_mm = [s for s in stem_min_values if (mm, s) not in mm_stem_excluded]
    xi_idx       = [stem_min_values.index(s) for s in stem_mins_mm]
    counts    = [results[mm][s][0] for s in stem_mins_mm]
    coverages = [results[mm][s][1] for s in stem_mins_mm]

    bars = ax.bar(
        np.array(xi_idx) + offsets[(mm, "count")], counts, width=bar_w,
        color=colors[(mm, "count")], alpha=0.88,
        label=f"Structure no., mism.={mm}",
    )
    ax2.bar(
        np.array(xi_idx) + offsets[(mm, "cov")], coverages, width=bar_w,
        color=colors[(mm, "cov")], alpha=0.88,
        label=f"Genome cov., mism.={mm}",
    )

    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(all_counts) * 0.02,
            f"{n:,}", ha="center", va="bottom", fontsize=7,
            color=colors[(mm, "count")],
        )
    for xi, cov in zip(np.array(xi_idx) + offsets[(mm, "cov")], coverages):
        ax2.text(
            xi, cov + 0.5, f"{cov:.1f}%",
            ha="center", va="bottom", fontsize=7,
            color=colors[(mm, "cov")],
        )

ax.set_xticks(x)
ax.set_xticklabels([f"{v}" for v in stem_min_values], fontsize=11)
ax.set_ylabel("Number of structures", fontsize=12, color="#3389C0")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.tick_params(axis="y", labelsize=10)
ax.set_ylim(0, max(all_counts) * 1.18)
ax.set_xlabel("Minimum stem length (bp)")

ax2.set_ylim(0, 100)
ax2.set_ylabel("Genome coverage (%)", fontsize=12, color="#D74050")
ax2.tick_params(axis="y", labelcolor="#D74050", labelsize=10)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

#ax.set_title(
#    f"Hairpin count and genome coverage vs minimum stem length\n"
#    f"(stem_max={STEM_MAX}, loop={LOOP_LENGTH}, mismatch=0 vs 1)",
#    fontsize=11,
#)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

out_dir = Path(__file__).parent
plt.savefig(out_dir / "hairpins_vs_stem_min_mm.pdf", dpi=150)
plt.savefig(out_dir / "hairpins_vs_stem_min_mm.png", dpi=600)
print("\nPlot saved: hairpins_vs_stem_min_mm.pdf / .png")
