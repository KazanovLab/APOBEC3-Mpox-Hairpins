#!/usr/bin/env python3
"""Build a per-position substitution table from a multiple sequence alignment.

For every position of the reference genome the script counts, across all the
other aligned sequences, how many substitutions to A/C/G/T and how many gaps
occur, and flags whether the reference context corresponds to an APOBEC3
mutation signature.

APOBEC signature        : TC -> TT  (C, with a 5' T)   or  GA -> AA  (G, with a 3' A)
APOBEC extended signature: VC -> VT  (C, with a 5' A/C/G) or  GB -> AB  (G, with a 3' C/G/T)
(V = A/C/G, B = C/G/T -- so the extended class excludes the strict signature.)

Usage:
    python build_substitution_table.py <alignment.aln> <reference.fna> [-o out.tsv]
"""

import argparse
import sys

GAP_CHARS = {"-", ".", "~"}
BASES = ("A", "C", "G", "T")
V = {"A", "C", "G"}   # IUPAC V: not T
B = {"C", "G", "T"}   # IUPAC B: not A


def read_fasta(path):
    """Return a list of (name, sequence) tuples."""
    records = []
    name = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("alignment", help="multi-FASTA alignment file")
    ap.add_argument("reference", help="reference genome FASTA (single sequence)")
    ap.add_argument("-o", "--output", help="output TSV file (default: stdout)")
    args = ap.parse_args()

    # reference genome
    ref_records = read_fasta(args.reference)
    if len(ref_records) != 1:
        sys.exit(f"expected exactly one sequence in {args.reference}, "
                 f"got {len(ref_records)}")
    ref_seq = ref_records[0][1].upper()

    # alignment
    aln = read_fasta(args.alignment)
    if not aln:
        sys.exit("alignment file contains no sequences")

    aln_len = len(aln[0][1])
    for name, seq in aln:
        if len(seq) != aln_len:
            sys.exit(f"sequence '{name}' length {len(seq)} != alignment "
                     f"length {aln_len} (alignment is not rectangular)")

    # locate the reference row inside the alignment by matching the
    # ungapped sequence to the reference genome
    ref_idx = None
    for i, (name, seq) in enumerate(aln):
        ungapped = "".join(c for c in seq.upper() if c not in GAP_CHARS)
        if ungapped == ref_seq:
            ref_idx = i
            break
    if ref_idx is None:
        sys.exit("could not find the reference genome among the aligned "
                 "sequences (no ungapped sequence matches the reference)")

    ref_aln = aln[ref_idx][1].upper()
    others = [seq.upper() for i, (name, seq) in enumerate(aln) if i != ref_idx]
    sys.stderr.write(
        f"reference '{aln[ref_idx][0]}' found; "
        f"comparing against {len(others)} other sequences "
        f"over {aln_len} alignment columns\n")

    # walk the alignment columns
    out = open(args.output, "w") if args.output else sys.stdout
    header = ["pos_ref", "pos_aln", "ref", "alt_A", "alt_C", "alt_G", "alt_T",
              "gaps", "is_APOBEC", "is_APOBEC_ext"]
    out.write("\t".join(header) + "\n")

    pos_ref = 0
    written = 0
    for col in range(aln_len):
        ref_base = ref_aln[col]
        if ref_base in GAP_CHARS:
            continue  # insertion relative to the reference -> no ref position
        pos_ref += 1

        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        gaps = 0
        for seq in others:
            c = seq[col]
            if c in GAP_CHARS:
                gaps += 1
            elif c in counts and c != ref_base:
                counts[c] += 1
            # match to reference or ambiguous base (N etc.) -> not counted

        # keep only positions with at least one substitution
        if counts["A"] + counts["C"] + counts["G"] + counts["T"] == 0:
            continue

        # APOBEC context, evaluated on the ungapped reference genome
        prev_base = ref_seq[pos_ref - 2] if pos_ref >= 2 else ""
        next_base = ref_seq[pos_ref] if pos_ref < len(ref_seq) else ""

        # APOBEC flags require both the reference context AND that the
        # corresponding signature substitution (C->T or G->A) was observed.
        is_apobec = False
        is_apobec_ext = False
        if ref_base == "C" and counts["T"] > 0:
            if prev_base == "T":
                is_apobec = True
            elif prev_base in V:
                is_apobec_ext = True
        elif ref_base == "G" and counts["A"] > 0:
            if next_base == "A":
                is_apobec = True
            elif next_base in B:
                is_apobec_ext = True

        row = [pos_ref, col + 1, ref_base,
               counts["A"], counts["C"], counts["G"], counts["T"], gaps,
               "True" if is_apobec else "False",
               "True" if is_apobec_ext else "False"]
        out.write("\t".join(str(x) for x in row) + "\n")
        written += 1

    if args.output:
        out.close()
    sys.stderr.write(f"wrote {written} substituted positions "
                     f"(of {pos_ref} reference positions)\n")


if __name__ == "__main__":
    main()
