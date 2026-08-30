import numpy as np
import pandas as pd

# Creates matched-pairs dataset: for each of the hotspot targets at long-stem & short-loop hairpins
# (loop_boundary_length==3 & stem in {5,6,7}, or ==4 & stem in {5,6}), takes one control row that has the
# exact same other features (grantham, third_nt, minus1_nt) but located outside any loop boundary (in_loop_boundary == 0), 
# sampled without replacement. This holds the non-hairpin confounders fixed by construction and isolates what the
# loop-boundary features add on top.

RNG_SEED = 1
OUT_TSV = "matched_pairs_dataset.tsv"

df = pd.read_csv("ml_table_all.tsv", sep='\t')

core_mask = (((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 7)) |
             ((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 6)) |
             ((df.loop_boundary_length == 3) & (df.loop_boundary_stem_length == 5)) |
             ((df.loop_boundary_length == 4) & (df.loop_boundary_stem_length == 6)) |
             ((df.loop_boundary_length == 4) & (df.loop_boundary_stem_length == 5)))
core = df.loc[core_mask].sort_values("position").reset_index(drop=True)
print(f"core rows: {len(core)}")

control_pool = df.loc[(~core_mask) & (df.in_loop_boundary == 0)].copy()
print(f"control pool (no loop boundary at all): {len(control_pool)}")

rng = np.random.default_rng(RNG_SEED)
used_positions = set()
matched_rows = []
unmatched = []
for _, case in core.iterrows():
    cands = control_pool[
        (control_pool.grantham == case.grantham) &
        (control_pool.third_nt == case.third_nt) &
        (control_pool.minus1_nt == case.minus1_nt) &
        (~control_pool.position.isin(used_positions))
    ]
    if len(cands) == 0:
        unmatched.append(case.position)
        continue
    pick = cands.iloc[rng.integers(len(cands))]
    used_positions.add(pick.position)
    matched_rows.append({"pair_id": case.position, "group": "case", **case.to_dict()})
    matched_rows.append({"pair_id": case.position, "group": "control", **pick.to_dict()})

if unmatched:
    print(f"WARNING: {len(unmatched)} core rows could not be matched: {unmatched}")

matched = pd.DataFrame(matched_rows)
matched = pd.get_dummies(matched, columns=["third_nt", "minus1_nt"], dtype="uint8")
LOOP_BOUNDARY_COLS = ["loop_boundary_length", "loop_boundary_stem_length", "loop_boundary_pin_energy"]
matched[LOOP_BOUNDARY_COLS] = matched[LOOP_BOUNDARY_COLS].fillna(0.0)

n_case, n_ctrl = (matched.group == "case").sum(), (matched.group == "control").sum()
case_pos = matched.loc[matched.group == "case", "any_mutation"].sum()
ctrl_pos = matched.loc[matched.group == "control", "any_mutation"].sum()
print(f"matched dataset: {len(matched)} rows ({n_case} case / {n_ctrl} control)")
print(f"  case positives:    {case_pos}/{n_case} ({case_pos/n_case:.1%})")
print(f"  control positives: {ctrl_pos}/{n_ctrl} ({ctrl_pos/n_ctrl:.1%})")

id_cols = ["pair_id", "group", "position", "nucleotide", "context_pm50"]
matched[id_cols + [c for c in matched.columns if c not in id_cols]].to_csv(OUT_TSV, sep="\t", index=False)
print(f"saved matched dataset to {OUT_TSV}")
