#!/usr/bin/env python3
"""
Build a per-motif ML feature table for every TC / GA (APOBEC-signature) site in
the mpox reference genome (NC_063383.1).

One row per motif occurrence:
  - a "TC" motif is anchored on the C nucleotide (the edited base, C->T);
  - a "GA" motif is anchored on the G nucleotide (edited base on the opposite
    strand, read there as C->T, i.e. G->A on the reference).

Columns
  1  position                 1-based position of the C (TC) or G (GA)
  2  nucleotide               C or G
  3  context_pm50             +/-50 nt window around the position (101 nt);
                              reverse-complemented when the anchor is G, so the
                              motif is always presented in TC orientation.
                              All bases lowercased except the central TC motif.
  4  grantham                 Grantham score of the APOBEC substitution
                              (C->T for TC, G->A for GA): 0 for intergenic /
                              synonymous, matrix value for missense, 215 for
                              nonsense / stop-loss
  5  third_nt                 third base of the T-C-N triplet (base 3' of the C
                              in TC orientation): genome[C+1] for TC,
                              complement(genome[G-1]) for GA

  Hairpin features (one output file per overlap-resolution strategy):
  6  in_stem                  1 if the position lies in a hairpin stem
  7  stem_length              stem length (bp); for the "all" strategy the max
                              stem length over all overlapping hairpins covering
                              the position; 0 if not in a stem
  8  stem_loop_length         loop length of the hairpin whose stem was used for
                              stem_length (i.e. the one giving the max); 0 if not
                              in a stem
  9  stem_energy              NN free energy of that same hairpin; blank if not
                              in a stem
  Loop positions are split by role, per covering hairpin: "boundary" (the C of
  a TC at the loop's 3' end, or the G of a GA at the loop's 5' end -- the same
  convention as structure_type="tc_end" / load.Genome.end_targets, which
  excludes loops shorter than 2 nt) vs. "other" (every other loop position). A
  position can be boundary for one overlapping hairpin and other for another,
  so the two roles are tracked independently, each with its own min-loop-length
  selection (mirroring stem_length's max selection).

  10 in_loop_boundary          1 if the position is a TC-3'/GA-5' loop boundary
                              for some covering hairpin
  11 loop_boundary_length      loop length (nt) of the hairpin achieving that
                              min, among boundary-qualifying hairpins; 0 if none
  12 loop_boundary_stem_length stem length of that same hairpin; 0 if none
  13 loop_boundary_energy      NN free energy of that same hairpin; blank if none
  14 in_loop_other             1 if the position is a non-boundary loop position
                              for some covering hairpin
  15 loop_other_length         loop length (nt) of the hairpin achieving that
                              min, among other-qualifying hairpins; 0 if none
  16 loop_other_stem_length    stem length of that same hairpin; 0 if none
  17 loop_other_energy         NN free energy of that same hairpin; blank if none

  Mutation features (three datasets):
  18 ds1_mutation             1 if the APOBEC substitution is observed here in
                              dataset1
  19 ds1_independent_events   number of independent events (0 if none)
  20 ds2_mutation             dataset2_west_africa
  21 ds2_independent_events
  22 ds3_mutation             dataset3_gisaid
  23 ds3_independent_events
  24 any_mutation             1 if the APOBEC substitution is observed in any of
                              the three datasets
  25 total_independent_events sum of independent events across all three datasets

Hairpin prediction: EMBOSS palindrome, mismatches=0, stem_min=4,
stem_max=30 (repo default), loop_max=20. Five overlap-resolution strategies,
one TSV each: most_stable, greedy, max_cov, min_cov, all.

Output: ml/ml_table_<strategy>.tsv
"""

import sys
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
REPO         = Path(__file__).resolve().parent.parent
PACKAGE      = REPO / "package"
DATASETS_DIR = REPO / "datasets"
GTF          = PACKAGE / "input" / "GCF_014621545.1_ASM1462154v1_genomic.gtf"
OUT_DIR      = Path(__file__).resolve().parent

# import the package modules; they read config.yaml / nn_coefs/ via relative
# paths, so we must chdir into package/ first.
sys.path.insert(0, str(PACKAGE))
os.chdir(PACKAGE)

import load
# keep every detected hairpin, including loop-length-0 perfect palindromes
load.params["filter_looplen_lessequal"] = -1
from emboss import get_palindrome
from functions import find_groups, max_coverage
from hairpin_groups import greedy_choose, most_stable, min_coverage

GENOME = load.genome.sequence.upper()
N = len(GENOME)
print(f"Genome: {load.genome.name.strip()}  length={N}")

_COMP = str.maketrans("ACGT", "TGCA")
def rc(s):   return s.translate(_COMP)[::-1]
def comp(b): return b.translate(_COMP)

# ── Grantham scoring (codon logic + matrix copied from fig4c) ─────────────────
CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_GRANTHAM_RAW = [
    [  0,112,111,126,195, 91,107, 60, 86, 94,102,106, 84,113, 27, 99, 58,148,112, 64],
    [112,  0, 86, 96,180, 43, 54,125, 29, 97,102, 26, 91, 97,103,110, 71,101, 77, 96],
    [111, 86,  0, 23,139, 46, 42, 80, 68,149,153, 94,142,158, 91, 46, 65,174,143,133],
    [126, 96, 23,  0,154, 61, 45, 94, 81,168,172,101,160,177,108, 65, 85,181,160,152],
    [195,180,139,154,  0,154,170,159,174,198,198,202,196,205,169,112,149,215,194,192],
    [ 91, 43, 46, 61,154,  0, 29, 87, 24,109,113, 53,101,116, 76, 68, 42,130, 99, 96],
    [107, 54, 42, 45,170, 29,  0, 98, 40,134,138, 56,126,140, 93, 80, 65,152,122,121],
    [ 60,125, 80, 94,159, 87, 98,  0, 98,135,138,127,127,153, 42, 56, 59,184,147,109],
    [ 86, 29, 68, 81,174, 24, 40, 98,  0, 94, 99, 32, 87,100, 77, 89, 47,115, 83, 84],
    [ 94, 97,149,168,198,109,134,135, 94,  0,  5,102, 10, 21, 95,142, 89, 61, 33, 29],
    [102,102,153,172,198,113,138,138, 99,  5,  0,107, 15, 22, 98,145, 92, 61, 36, 32],
    [106, 26, 94,101,202, 53, 56,127, 32,102,107,  0, 95,102,103,121, 78,110, 85, 97],
    [ 84, 91,142,160,196,101,126,127, 87, 10, 15, 95,  0, 28, 87,135, 81, 67, 36, 21],
    [113, 97,158,177,205,116,140,153,100, 21, 22,102, 28,  0,114,155,103, 40, 22, 50],
    [ 27,103, 91,108,169, 76, 93, 42, 77, 95, 98,103, 87,114,  0, 74, 38,147,110, 68],
    [ 99,110, 46, 65,112, 68, 80, 56, 89,142,145,121,135,155, 74,  0, 58,177,144,124],
    [ 58, 71, 65, 85,149, 42, 65, 59, 47, 89, 92, 78, 81,103, 38, 58,  0,128, 92, 69],
    [148,101,174,181,215,130,152,184,115, 61, 61,110, 67, 40,147,177,128,  0, 37, 88],
    [112, 77,143,160,194, 99,122,147, 83, 33, 36, 85, 36, 22,110,144, 92, 37,  0, 55],
    [ 64, 96,133,152,192, 96,121,109, 84, 29, 32, 97, 21, 50, 68,124, 69, 88, 55,  0],
]
_IDX = {aa: i for i, aa in enumerate(_AA_ORDER)}
GRANTHAM = {(a1, a2): _GRANTHAM_RAW[_IDX[a1]][_IDX[a2]]
            for a1 in _AA_ORDER for a2 in _AA_ORDER}
STOP_GRANTHAM = 215

cds_list = []
with open(GTF) as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.rstrip('\n').split('\t')
        if len(fields) < 9 or fields[2] != 'CDS':
            continue
        start = int(fields[3]); end = int(fields[4])
        strand = fields[6]; phase = int(fields[7])
        attrs = fields[8]
        m = re.search(r'gene "([^"]+)"', attrs)
        gene = m.group(1) if m else re.search(r'locus_tag "([^"]+)"', attrs).group(1)
        cds_list.append((start, end, strand, gene, phase))
cds_list.sort(key=lambda x: x[0])
print(f"Loaded {len(cds_list)} CDS features from GTF")

def overlapping_cds(pos_1):
    return [c for c in cds_list if c[0] <= pos_1 <= c[1]]

def codon_consequence(pos_1, mut_base, cds):
    start, end, strand, gene, phase = cds
    if strand == '+':
        pos_in_cds  = pos_1 - start
        codon_index = (pos_in_cds + phase) // 3
        codon_start = start + codon_index * 3 - phase
        orig_codon  = GENOME[codon_start - 1 : codon_start + 2]
        local       = pos_1 - codon_start
        mut_codon   = orig_codon[:local] + mut_base + orig_codon[local + 1:]
    else:
        pos_in_cds   = end - pos_1
        codon_index  = (pos_in_cds + phase) // 3
        codon_right  = end - codon_index * 3 + phase
        orig_codon_p = GENOME[codon_right - 3 : codon_right]
        orig_codon   = rc(orig_codon_p)
        local_p      = pos_1 - (codon_right - 2)
        mut_codon_p  = orig_codon_p[:local_p] + mut_base + orig_codon_p[local_p + 1:]
        mut_codon    = rc(mut_codon_p)
    return CODON_TABLE.get(orig_codon, '?'), CODON_TABLE.get(mut_codon, '?')

def grantham_for_position(pos_1, mut_base):
    hits = overlapping_cds(pos_1)
    if not hits:
        return 0
    scores = []
    for cds in hits:
        wt, mut = codon_consequence(pos_1, mut_base, cds)
        if wt == mut:
            scores.append(0)
        elif wt in _IDX and mut in _IDX:
            scores.append(GRANTHAM[(wt, mut)])
        elif wt == '*' or mut == '*':
            scores.append(STOP_GRANTHAM)
        else:
            scores.append(0)
    return max(scores)

# ── motif sites ──────────────────────────────────────────────────────────────
# TC -> anchor on C (edited base);  GA -> anchor on G (edited base on opp strand)
c_sites = [(m.end() - 1, 'C') for m in re.finditer("TC", GENOME)]
g_sites = [(m.start(), 'G') for m in re.finditer("GA", GENOME)]
sites = sorted(c_sites + g_sites, key=lambda x: x[0])   # 0-based positions
print(f"Motif sites: {len(c_sites)} TC (C-anchored) + {len(g_sites)} GA "
      f"(G-anchored) = {len(sites)}")

def context_pm50(pos0, base):
    """101-nt window centered on pos0, N-padded at genome edges; RC when base==G.
    Everything is lowercased except the central TC motif (indices 49-50), which
    is kept uppercase."""
    left  = max(0, pos0 - 50)
    right = min(N, pos0 + 51)
    seq = GENOME[left:right]
    seq = "N" * (50 - (pos0 - left)) + seq + "N" * (50 - (right - 1 - pos0))
    if base == 'G':
        seq = rc(seq)
    seq = seq.lower()
    return seq[:49] + "TC" + seq[51:]   # centered T(49) C(50) uppercase

def third_nt(pos0, base):
    if base == 'C':
        return GENOME[pos0 + 1] if pos0 + 1 < N else 'N'
    else:  # G: complement of the base 5' of G  ==  3rd base of TC triplet in RC frame
        return comp(GENOME[pos0 - 1]) if pos0 - 1 >= 0 else 'N'

# strategy-independent columns (positions 1-5 + mutation columns)
print("Computing motif-level features (context / grantham / third_nt)...")
base_rows = []
for pos0, base in sites:
    pos1 = pos0 + 1
    mut_base = 'T' if base == 'C' else 'A'
    base_rows.append({
        "position":     pos1,
        "nucleotide":   base,
        "context_pm50": context_pm50(pos0, base),
        "grantham":     grantham_for_position(pos1, mut_base),
        "third_nt":     third_nt(pos0, base),
        "_pos0":        pos0,
    })

# ── mutation datasets ────────────────────────────────────────────────────────
DATASETS = [
    ("ds1", DATASETS_DIR / "dataset1.substitution_table.tsv"),
    ("ds2", DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv"),
    ("ds3", DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv"),
]

def load_mut(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df["is_APOBEC"] == True]
    return dict(zip(df["pos_ref"].astype(int),
                    df["independent_events"].astype(int)))

mut_maps = {}
for tag, path in DATASETS:
    mut_maps[tag] = load_mut(path)
    print(f"  {tag}: {len(mut_maps[tag])} APOBEC-mutated positions, "
          f"{sum(mut_maps[tag].values())} independent events")

for row in base_rows:
    p = row["position"]
    total_events = 0
    any_mut = 0
    for tag, _ in DATASETS:
        events = mut_maps[tag].get(p, 0)
        row[f"{tag}_mutation"] = int(events > 0)
        row[f"{tag}_independent_events"] = events
        total_events += events
        any_mut |= int(events > 0)
    row["any_mutation"] = any_mut
    row["total_independent_events"] = total_events

# ── hairpin prediction ───────────────────────────────────────────────────────
STEM_MIN, STEM_MAX, LOOP_MAX, MM = 4, 30, 20, 0
print(f"\nHairpin detection: EMBOSS palindrome stem={STEM_MIN}-{STEM_MAX}, "
      f"loop_max={LOOP_MAX}, mismatches={MM}")
pa_file = get_palindrome(STEM_MIN, STEM_MAX, LOOP_MAX, MM)
hairpins = load.load_hairpins(str(pa_file))
print(f"  detected hairpins: {len(hairpins)}")

def select(strategy):
    if strategy == "all":
        return list(hairpins)
    groups = find_groups(hairpins)   # re-sorts hairpins in place; groups share objs
    fn = {"most_stable": most_stable, "greedy": greedy_choose,
          "max_cov": max_coverage, "min_cov": min_coverage}[strategy]
    chosen = []
    for g in groups:
        chosen += fn(list(g))
    return chosen

def mark(selected):
    """Per-position hairpin features.

    stem_len   : max stem length over hairpins whose stem covers the position
    stem_loop  : loop length of the hairpin that achieves that stem max
    stem_en    : energy of the hairpin that achieves that stem max

    Loop positions are split into two disjoint roles per covering hairpin:
      - "boundary": the position is the C of a TC dinucleotide at the loop's
        3' end, or the G of a GA dinucleotide at the loop's 5' end (the exact
        convention used elsewhere in this repo for structure_type="tc_end",
        see load.Genome.end_targets). Loops shorter than 2 nt never qualify,
        matching that convention.
      - "other": every other position covered by the hairpin's loop.
    A position can be "boundary" for one overlapping hairpin and "other" for
    another, so the two roles are tracked independently:
      loopb_len / loopb_stem / loopb_en : min loop length (+ companions) over
                                           hairpins where this is a boundary pos
      loopo_len / loopo_stem / loopo_en : min loop length (+ companions) over
                                           hairpins where this is a non-boundary
                                           loop position

    For single-selection strategies at most one hairpin covers a position, so
    the companion values are simply that hairpin's loop / stem / energy. Ties in
    the max/min are resolved first-hairpin-seen. Energy is NaN where the
    position doesn't fall in that role.
    """
    BIG = np.iinfo(np.int32).max
    stem_len  = np.zeros(N, dtype=np.int32)
    stem_loop = np.zeros(N, dtype=np.int32)
    stem_en   = np.full(N, np.nan, dtype=np.float64)

    loopb_len  = np.full(N, BIG, dtype=np.int32)
    loopb_stem = np.zeros(N, dtype=np.int32)
    loopb_en   = np.full(N, np.nan, dtype=np.float64)
    loopo_len  = np.full(N, BIG, dtype=np.int32)
    loopo_stem = np.zeros(N, dtype=np.int32)
    loopo_en   = np.full(N, np.nan, dtype=np.float64)

    for h in selected:
        for a, b in h.stem_indexes:                       # 0-based inclusive
            seg  = stem_len[a:b + 1]
            segl = stem_loop[a:b + 1]
            sege = stem_en[a:b + 1]
            mask = h.stem_length > seg                    # strictly greater -> new max
            seg[mask]  = h.stem_length
            segl[mask] = h.spacer_length
            sege[mask] = h.score

        l, r = h.spacer_index                             # 0-based inclusive
        if r < l:
            continue                                      # no loop (spacer_length 0)

        boundary = set()
        if h.spacer_length >= 2:
            if GENOME[r - 1:r + 1] == "TC":
                boundary.add(r)
            if GENOME[l:l + 2] == "GA":
                boundary.add(l)

        for pos0 in boundary:
            if h.spacer_length < loopb_len[pos0]:
                loopb_len[pos0]  = h.spacer_length
                loopb_stem[pos0] = h.stem_length
                loopb_en[pos0]   = h.score

        for pos0 in range(l, r + 1):
            if pos0 in boundary:
                continue
            if h.spacer_length < loopo_len[pos0]:
                loopo_len[pos0]  = h.spacer_length
                loopo_stem[pos0] = h.stem_length
                loopo_en[pos0]   = h.score

    loopb_len[loopb_len == BIG] = 0
    loopo_len[loopo_len == BIG] = 0
    return (stem_len, stem_loop, stem_en,
            loopb_len, loopb_stem, loopb_en,
            loopo_len, loopo_stem, loopo_en)

# ── assemble & write one table per strategy ──────────────────────────────────
STRATEGIES = ["most_stable", "greedy", "max_cov", "min_cov", "all"]
COLUMNS = [
    "position", "nucleotide", "context_pm50", "grantham", "third_nt",
    "in_stem", "stem_length", "stem_loop_length", "stem_energy",
    "in_loop_boundary", "loop_boundary_length",
    "loop_boundary_stem_length", "loop_boundary_energy",
    "in_loop_other", "loop_other_length",
    "loop_other_stem_length", "loop_other_energy",
    "ds1_mutation", "ds1_independent_events",
    "ds2_mutation", "ds2_independent_events",
    "ds3_mutation", "ds3_independent_events",
    "any_mutation", "total_independent_events",
]

for strategy in STRATEGIES:
    selected = select(strategy)
    (stem_len, stem_loop, stem_en,
     loopb_len, loopb_stem, loopb_en,
     loopo_len, loopo_stem, loopo_en) = mark(selected)
    out_rows = []
    for row in base_rows:
        p0 = row["_pos0"]
        sl = int(stem_len[p0]); lb = int(loopb_len[p0]); lo = int(loopo_len[p0])
        out_rows.append({
            "position":     row["position"],
            "nucleotide":   row["nucleotide"],
            "context_pm50": row["context_pm50"],
            "grantham":     row["grantham"],
            "third_nt":     row["third_nt"],
            "in_stem":          int(sl > 0),
            "stem_length":      sl,
            "stem_loop_length": int(stem_loop[p0]),
            "stem_energy":      stem_en[p0],
            "in_loop_boundary":          int(lb > 0),
            "loop_boundary_length":      lb,
            "loop_boundary_stem_length": int(loopb_stem[p0]),
            "loop_boundary_energy":      loopb_en[p0],
            "in_loop_other":             int(lo > 0),
            "loop_other_length":         lo,
            "loop_other_stem_length":    int(loopo_stem[p0]),
            "loop_other_energy":         loopo_en[p0],
            "ds1_mutation":            row["ds1_mutation"],
            "ds1_independent_events":  row["ds1_independent_events"],
            "ds2_mutation":            row["ds2_mutation"],
            "ds2_independent_events":  row["ds2_independent_events"],
            "ds3_mutation":            row["ds3_mutation"],
            "ds3_independent_events":  row["ds3_independent_events"],
            "any_mutation":             row["any_mutation"],
            "total_independent_events": row["total_independent_events"],
        })
    out_df = pd.DataFrame(out_rows, columns=COLUMNS)
    out_path = OUT_DIR / f"ml_table_{strategy}.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    n_stem = int((out_df["in_stem"] == 1).sum())
    n_loopb = int((out_df["in_loop_boundary"] == 1).sum())
    n_loopo = int((out_df["in_loop_other"] == 1).sum())
    print(f"  [{strategy:11s}] {len(selected):5d} hairpins  ->  {out_path.name}  "
          f"(rows={len(out_df)}, in_stem={n_stem}, "
          f"in_loop_boundary={n_loopb}, in_loop_other={n_loopo})")

print("\nDone.")
