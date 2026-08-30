import os
import numpy as np
import pandas as pd
import logomaker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Enrichment logo around the mutated position (any_mutation == 1):
# log2(p_mutated / p_not_mutated) per position and base, i.e. the mutated sites
# scored against the non-mutated TC/GA sites as background.

IN_TSV = "ml_table_all.tsv"
RESULTS_DIR = "results_sequence_logo"
TARGET = "any_mutation"
WINDOW_HALF = 10
BASES = ["A", "C", "G", "T"]
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(IN_TSV, sep='\t')
seqs = df["context_pm50"].str[50 - WINDOW_HALF: 50 + WINDOW_HALF + 1].str.upper()
mut = df[TARGET].values.astype(bool)
positions = np.arange(-WINDOW_HALF, WINDOW_HALF + 1)
print(f"mutated sites: {mut.sum()}, not mutated: {(~mut).sum()}")

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

# The TC at -1/0 is invariant by construction
ANCHOR = {-1: "T", 0: "C"}
ANCHOR_HEIGHT = 0.55

anchor_matrix = pd.DataFrame(0.0, index=positions, columns=BASES)
for pos, base in ANCHOR.items():
    anchor_matrix.loc[pos, base] = ANCHOR_HEIGHT

fig, ax = plt.subplots(figsize=(10, 4))

logomaker.Logo(anchor_matrix, ax=ax, color_scheme={b: "#b0b0b0" for b in BASES},
               alpha=0.65, zorder=1)
logomaker.Logo(enrichment, ax=ax, color_scheme="classic", zorder=2)
ax.axhline(0, color="black", linewidth=0.6)
ax.text(-0.5, ANCHOR_HEIGHT + 0.06, "invariant TC\n(all sites)", ha="center",
        va="bottom", fontsize=7.5, color="#808080", linespacing=1.3)
ax.set_ylabel("log2(mutated / not mutated)")
ax.set_xlabel("Relative position")
#ax.set_title(f"Enrichment logo: {int(mut.sum())} mutated sites scored against "
#             "non-mutated TC/GA sites")
ax.set_xticks(positions)
#for x in (2, 3):
#    ax.axvspan(x - 0.5, x + 0.5, color="gold", alpha=0.15, zorder=0)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "sequence_logo_mutated.png"), dpi=600, bbox_inches="tight")
plt.close(fig)

enrichment.round(4).to_csv(os.path.join(RESULTS_DIR, "enrichment_log2_mutated_vs_not.csv"))
print("\nlog2 enrichment at the positions of interest:")
print(enrichment.loc[[1, 2, 3, 4, 5]].round(3).to_string())
print(f"\nsaved to {RESULTS_DIR}/sequence_logo_mutated.png")
