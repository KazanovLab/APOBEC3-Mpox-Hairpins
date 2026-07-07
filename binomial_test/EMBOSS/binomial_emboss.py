#!/usr/bin/env python3
"""
Sweep all combinations of structure_type and hairpin_selection_type for EMBOSS hairpin set.

Run:
    python binomial_emboss.py
"""
import sys
import os
import io
import math
import contextlib
import re
from itertools import product
from pathlib import Path

OUT_PATH     = Path(__file__).parent / "binomial_results_emboss.txt"
PACKAGE_DIR = Path(__file__).parent.parent.parent / "package"

# set up path and working directory
os.chdir(PACKAGE_DIR)
sys.path.insert(0, str(PACKAGE_DIR))

import load
import functions
import all_hairpins as ah
import hairpin_groups as hg

print(f"Genome length : {load.genome.length:,} bp")
print(f"Mutations APOBEC    : {load.mut_cnt_APOBEC:,}")
print(f"Mutations APOBEC extended   : {load.mut_cnt_APOBEC_ext:,}")
print(f"Hairpins      : {len(load.all_hairpins_list):,}")
print()

# sweep parameters
STRUCTURE_TYPES = ["hairpin", "spacer", "tc_end", "c_end"]
SELECTION_TYPES = ["most_stable", "greedy", "max_cov", "min_cov", "all"]

# run all combinations
results = []
total = len(STRUCTURE_TYPES) * len(SELECTION_TYPES)
done  = 0

for structure_type, selection_type in product(STRUCTURE_TYPES, SELECTION_TYPES):
    done += 1
    print(f"[{done:>2}/{total}] structure={structure_type:<8}  selection={selection_type}", flush=True)

    # patch the module-level hit_type that hit_or_not() reads at call time
    functions.hit_type = structure_type

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if selection_type == "all":
            log10p, pvalue, hits, frac, targets_in_struct, all_targets = ah.calculate_pval_all(load.all_hairpins_list, structure_type)
        else:
            log10p, pvalue, hits, frac, targets_in_struct, all_targets = hg.calculate_pval_groups(load.all_hairpins_list, selection_type, structure_type)
    captured = buf.getvalue()
    print(captured, end="")   # echo to terminal so user can follow progress

    results.append(dict(
        structure_type=structure_type,
        selection_type=selection_type,
        mut_cnt_APOBEC=load.mut_cnt_APOBEC,
        mut_cnt_APOBEC_ext=load.mut_cnt_APOBEC_ext,
        hits=hits,
        frac_pct=frac,
        pvalue=pvalue,
        log10p=log10p,
    ))
    log10p_str = "inf" if math.isinf(log10p) else f"{log10p:.4f}"
    print(f"   -log10(p) = {log10p_str}\n")

# format table
col_w = dict(structure_type=14, selection_type=12, mut_cnt_APOBEC=8, mut_cnt_APOBEC_ext=8, hits=6,
             frac_pct=14, pvalue=14, log10p=14)

header = (f"{'structure_type':<{col_w['structure_type']}}  "
          f"{'selection_type':<{col_w['selection_type']}}  "
          f"{'mut_cnt_APOBEC':>{col_w['mut_cnt_APOBEC']}}  "
          f"{'mut_cnt_APOBEC_ext':>{col_w['mut_cnt_APOBEC_ext']}}  "
          f"{'hits':>{col_w['hits']}}  "
          f"{'frac_targets_%':>{col_w['frac_pct']}}  "
          f"{'p_value':>{col_w['pvalue']}}  "
          f"{'-log10(p)':>{col_w['log10p']}}")
sep = "-" * len(header)

lines = [header, sep]
for r in results:
    log10p_str = ("inf" if math.isinf(r["log10p"]) else
                  "NA"  if math.isnan(r["log10p"]) else
                  f"{r['log10p']:.4f}")
    pval_str   = ("NA" if math.isnan(r["pvalue"]) else
                  f"{r['pvalue']:.6f}" if r["pvalue"] >= 1e-3 else
                  f"{r['pvalue']:.6e}")
    hits_str   = "ERR" if (r["hits"] is None or (isinstance(r["hits"], float) and math.isnan(r["hits"]))) else str(int(r["hits"]))
    frac_str   = "NA"  if math.isnan(r["frac_pct"]) else f"{r['frac_pct']:.4f}"

    lines.append(
        f"{r['structure_type']:<{col_w['structure_type']}}  "
        f"{r['selection_type']:<{col_w['selection_type']}}  "
        f"{r['mut_cnt_APOBEC']:>{col_w['mut_cnt_APOBEC']}}  "
        f"{r['mut_cnt_APOBEC_ext']:>{col_w['mut_cnt_APOBEC_ext']}}  "
        f"{hits_str:>{col_w['hits']}}  "
        f"{frac_str:>{col_w['frac_pct']}}  "
        f"{pval_str:>{col_w['pvalue']}}  "
        f"{log10p_str:>{col_w['log10p']}}"
    )
    # blank line between structure type blocks
    if r is results[-1] or results[results.index(r)+1]["structure_type"] != r["structure_type"]:
        lines.append("")

table_str = "\n".join(lines)
print("\n" + table_str)

with open(OUT_PATH, "w") as f:
    f.write(table_str + "\n")
print(f"Results saved → {OUT_PATH}")
