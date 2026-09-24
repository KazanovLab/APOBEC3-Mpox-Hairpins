import os
import sys
import numpy as np
import pandas as pd
import logomaker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Enrichment logo restricted to one 3-nt motif, 
# e.g. TCA: log2(p_mutated / p_not_mutated) per position and base, 
# computed among sites that carry that motif.
# The background is the non-mutated sites of the same motif.

MOTIF = "TCA"
IN_TSV = "ml_table_all.tsv"
RESULTS_DIR = "results_sequence_logo"
TARGET = "any_mutation"
WINDOW_HALF = 10
CENTER_IDX = 50
BASES = ["A", "C", "G", "T"]
ANCHOR_HEIGHT = 0.55
os.makedirs(RESULTS_DIR, exist_ok=True)

motif = (sys.argv[1] if len(sys.argv) > 1 else MOTIF).upper()
if not motif.startswith("TC"):
    raise ValueError(f"motif must start with TC (every site in the table is a "
                     f"TC/GA site), got {motif!r}")
if not set(motif) <= set(BASES):
    raise ValueError(f"motif must use only {''.join(BASES)}, got {motif!r}")

df = pd.read_csv(IN_TSV, sep='\t')
context = df["context_pm50"].str.upper()

keep = pd.Series(True, index=df.index)
for i, base in enumerate(motif[2:], start=1):
    keep &= context.str[CENTER_IDX + i] == base
df, context = df[keep].reset_index(drop=True), context[keep].reset_index(drop=True)

seqs = context.str[CENTER_IDX - WINDOW_HALF: CENTER_IDX + WINDOW_HALF + 1]
mut = df[TARGET].values.astype(bool)
positions = np.arange(-WINDOW_HALF, WINDOW_HALF + 1)
print(f"motif {motif}: {len(df)} sites, mutated {int(mut.sum())}, "
      f"not mutated {int((~mut).sum())} ({mut.mean():.1%} mutated)")

seq_arr = np.array([list(s) for s in seqs])


def freq_matrix(rows):
    counts = np.zeros((len(positions), 4))
    for i in range(len(positions)):
        col = seq_arr[rows, i]
        for b, base in enumerate(BASES):
            counts[i, b] = (col == base).sum()
    return pd.DataFrame(counts, index=positions, columns=BASES)


counts_mut = freq_matrix(mut)
counts_unmut = freq_matrix(~mut)
p_mut = counts_mut.div(counts_mut.sum(axis=1), axis=0)
p_unmut = counts_unmut.div(counts_unmut.sum(axis=1), axis=0)

enrichment = np.log2((p_mut + 1e-9) / (p_unmut + 1e-9))

ANCHOR = {i - 1: base for i, base in enumerate(motif)}

anchor_matrix = pd.DataFrame(0.0, index=positions, columns=BASES)
for pos, base in ANCHOR.items():
    anchor_matrix.loc[pos, base] = ANCHOR_HEIGHT

fig, ax = plt.subplots(figsize=(10, 4))

logomaker.Logo(anchor_matrix, ax=ax, color_scheme={b: "#b0b0b0" for b in BASES},
               alpha=0.65, zorder=1)
logomaker.Logo(enrichment, ax=ax, color_scheme="classic", zorder=2)
ax.axhline(0, color="black", linewidth=0.6)
ax.text((min(ANCHOR) + max(ANCHOR)) / 2, ANCHOR_HEIGHT + 0.06,
        f"invariant {motif}\n(all sites in subset)", ha="center",
        va="bottom", fontsize=7.5, color="#808080", linespacing=1.3)
ax.set_ylabel("log2(mutated / not mutated)")
ax.set_xlabel("Relative position")
#ax.set_title(f"Enrichment logo: {int(mut.sum())} mutated {motif} sites scored "
#             f"against non-mutated {motif} sites")
ax.set_xticks(positions)

fig.tight_layout()
out_png = os.path.join(RESULTS_DIR, f"sequence_logo_{motif}.png")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
plt.close(fig)

out_csv = os.path.join(RESULTS_DIR, f"enrichment_log2_{motif}_vs_not.csv")
enrichment.round(4).to_csv(out_csv)
print(f"\nlog2 enrichment at the positions of interest:")
print(enrichment.loc[[1, 2, 3, 4, 5]].round(3).to_string())
print(f"\nsaved to {out_png}")
