import os
from pathlib import Path

import pandas as pd

# Adds the protein-stability predictions from ddG_results.tsv to the ML table.
# ddG_results.tsv scores the amino-acid substitution caused by the C->T (or G->A)
# edit at each motif site: ddG from the FoldX with AlphaFold models, 
# and the two DDGun baselines (sequence-only and structure-aware). 
# Stability change is only defined for a substitution that changes the protein, 
# so synonymous, intergenic, nonsense and in-stop-codon sites carry no score.

REPO = Path(__file__).resolve().parent.parent
ML_TSV = REPO / "ml" / "ml_table_all.tsv"
DDG_TSV = Path(__file__).resolve().parent / "ddG_results.tsv"
OUT_DIR = Path(__file__).resolve().parent / "ml"
OUT_TSV = OUT_DIR / "ml_table_all_with_ddG.tsv"
OUT_DIR.mkdir(exist_ok=True)

DDG_COLS = ["ddG", "ddgun_seq", "ddgun_3d"]
KEY = "position"

ml = pd.read_csv(ML_TSV, sep='\t')
ddg = pd.read_csv(DDG_TSV, sep='\t')
print(f"ml_table_all.tsv : {ml.shape[0]} rows, {ml.shape[1]} columns")
print(f"ddG_results.tsv  : {ddg.shape[0]} rows")

assert not ddg[KEY].duplicated().any(), "duplicate positions in ddG_results.tsv"
missing = set(ml[KEY]) - set(ddg[KEY])
assert not missing, f"{len(missing)} positions have no ddG row, e.g. {sorted(missing)[:5]}"

check = ml[[KEY, "nucleotide"]].merge(ddg[[KEY, "ref_nt"]], on=KEY)
mismatch = (check.nucleotide != check.ref_nt).sum()
assert mismatch == 0, f"{mismatch} positions disagree on the reference base"

for col in DDG_COLS:
    ddg[col] = pd.to_numeric(ddg[col], errors="coerce")

out = ml.merge(ddg[[KEY] + DDG_COLS], on=KEY, how="left", validate="one_to_one")
assert len(out) == len(ml), "merge changed the number of rows"

out.to_csv(OUT_TSV, sep='\t', index=False)

print(f"\nadded {len(DDG_COLS)} columns -> {out.shape[1]} total")
for col in DDG_COLS:
    s = out[col]
    print(f"  {col:10s} {s.notna().sum():6d} scored ({s.notna().mean():5.1%}), "
          f"range {s.min():7.3f} .. {s.max():6.3f}, mean {s.mean():6.3f}")
print("\nrows without a score, by reason (status column of ddG_results.tsv):")
print(ddg.loc[ddg[DDG_COLS].isna().all(axis=1), "status"].value_counts().to_string())
print(f"\nsaved {OUT_TSV.name} to {OUT_TSV.parent}")
