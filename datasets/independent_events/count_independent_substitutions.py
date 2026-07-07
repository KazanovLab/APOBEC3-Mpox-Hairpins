#!/usr/bin/env python3
"""
Count independent substitution events per alignment site using
IQ-TREE ancestral state reconstruction.

For each site, traverses the phylogenetic tree and counts the number
of branches where the parent state differs from the child state.
This gives the number of independent (phylogenetically distinct)
substitution events at each position.

Usage:
    python count_independent_substitutions.py \
        --tree iqtree/mpox_new_data.aln.treefile \
        --aln iqtree/mpox_new_data.aln \
        --state iqtree/mpox_new_data.aln.state \
        --output substitution_counts.tsv
"""

import argparse
import sys
import numpy as np
from Bio import Phylo, SeqIO
from io import StringIO


def parse_tree_edges(tree_file):
    """Parse Newick tree and return list of (parent_name, child_name) edges.

    IQ-TREE names internal nodes as Node1, Node2, etc. in the .state file.
    Bio.Phylo reads these as clade.name for labeled internal nodes.
    For the root (which may be unlabeled), we need to handle it specially.
    """
    tree = Phylo.read(tree_file, "newick")

    edges = []
    # Assign names to internal nodes if not already named
    # IQ-TREE uses Node1, Node2, ... for internal nodes in the .state file
    # The tree file also has these labels as internal node names

    # Collect all edges as (parent_name, child_name)
    def traverse(clade, parent_name=None):
        name = clade.name if clade.name else None
        if parent_name is not None and name is not None:
            edges.append((parent_name, name))
        for child in clade.clades:
            traverse(child, name)

    traverse(tree.root)
    return edges


def read_tip_states(aln_file):
    """Read alignment and return dict {seq_name: numpy_array_of_chars}."""
    tip_states = {}
    for record in SeqIO.parse(aln_file, "fasta"):
        tip_states[record.id] = np.array(list(str(record.seq).upper()))
    return tip_states


def read_internal_states(state_file, nsites):
    """Read IQ-TREE .state file and return dict {node_name: numpy_array_of_states}.

    The .state file has columns: Node, Site, State, p_A, p_C, p_G, p_T
    We only need Node, Site, State (the ML ancestral state).
    """
    print("Reading ancestral states (this may take a minute)...", file=sys.stderr)

    node_states = {}

    with open(state_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('Node\t'):
                continue
            parts = line.rstrip('\n').split('\t')
            node_name = parts[0]
            site_idx = int(parts[1]) - 1  # 1-based to 0-based
            state = parts[2]

            if node_name not in node_states:
                node_states[node_name] = np.empty(nsites, dtype='U1')
            node_states[node_name][site_idx] = state

    print(f"  Read states for {len(node_states)} internal nodes", file=sys.stderr)
    return node_states


def count_substitutions(edges, all_states, nsites):
    """For each site, count branches where parent state != child state.

    Skips branches where either parent or child has a gap (-) or
    ambiguous state (N, R, Y, S, W, K, M, B, D, H, V).
    """
    VALID = set('ACGT')
    counts = np.zeros(nsites, dtype=np.int32)

    # Also track substitution types per site
    sub_details = [[] for _ in range(nsites)]

    n_edges = len(edges)
    skipped = 0

    for i, (parent_name, child_name) in enumerate(edges):
        if (i + 1) % 50 == 0:
            print(f"  Processing edge {i+1}/{n_edges}...", file=sys.stderr)

        if parent_name not in all_states:
            print(f"  WARNING: parent '{parent_name}' not found in states, skipping edge", file=sys.stderr)
            skipped += 1
            continue
        if child_name not in all_states:
            print(f"  WARNING: child '{child_name}' not found in states, skipping edge", file=sys.stderr)
            skipped += 1
            continue

        p_states = all_states[parent_name]
        c_states = all_states[child_name]

        for site in range(nsites):
            ps = p_states[site]
            cs = c_states[site]
            if ps in VALID and cs in VALID and ps != cs:
                counts[site] += 1
                sub_details[site].append(f"{ps}>{cs}")

    if skipped:
        print(f"  WARNING: {skipped} edges skipped due to missing node states", file=sys.stderr)

    return counts, sub_details


def main():
    parser = argparse.ArgumentParser(
        description="Count independent substitution events per alignment site"
    )
    parser.add_argument("--tree", required=True, help="IQ-TREE treefile (Newick)")
    parser.add_argument("--aln", required=True, help="Alignment file (FASTA)")
    parser.add_argument("--state", required=True, help="IQ-TREE .state file from -asr")
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument("--min-events", type=int, default=0,
                        help="Only output sites with >= this many events (default: 0 = all)")
    args = parser.parse_args()

    # Step 1: Parse tree edges
    print("Parsing tree...", file=sys.stderr)
    edges = parse_tree_edges(args.tree)
    print(f"  Found {len(edges)} edges in tree", file=sys.stderr)

    # Step 2: Read tip sequences
    print("Reading alignment...", file=sys.stderr)
    tip_states = read_tip_states(args.aln)
    nsites = len(next(iter(tip_states.values())))
    print(f"  {len(tip_states)} sequences, {nsites} sites", file=sys.stderr)

    # Step 3: Read internal node states
    internal_states = read_internal_states(args.state, nsites)

    # Merge all states into one dict
    all_states = {}
    all_states.update(internal_states)
    all_states.update(tip_states)
    print(f"  Total nodes with states: {len(all_states)}", file=sys.stderr)

    # Step 4: Count substitutions per site
    print("Counting independent substitution events...", file=sys.stderr)
    counts, sub_details = count_substitutions(edges, all_states, nsites)

    # Step 5: Write output
    print(f"Writing output to {args.output}...", file=sys.stderr)
    n_written = 0
    with open(args.output, 'w') as out:
        out.write("site\tindependent_events\tsubstitutions\n")
        for i in range(nsites):
            if counts[i] >= args.min_events:
                subs = ",".join(sub_details[i]) if sub_details[i] else "."
                out.write(f"{i+1}\t{counts[i]}\t{subs}\n")
                n_written += 1

    # Summary
    variable_sites = np.sum(counts > 0)
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Total sites: {nsites}", file=sys.stderr)
    print(f"  Sites with >= 1 substitution event: {variable_sites}", file=sys.stderr)
    print(f"  Sites written to output: {n_written}", file=sys.stderr)
    print(f"  Max independent events at a single site: {np.max(counts)}", file=sys.stderr)

    # Distribution
    print(f"\nDistribution of independent events per site:", file=sys.stderr)
    for n_events in range(0, min(int(np.max(counts)) + 1, 20)):
        n_sites = np.sum(counts == n_events)
        if n_sites > 0:
            print(f"  {n_events} events: {n_sites} sites", file=sys.stderr)
    if np.max(counts) >= 20:
        n_sites = np.sum(counts >= 20)
        print(f"  >=20 events: {n_sites} sites", file=sys.stderr)


if __name__ == "__main__":
    main()
