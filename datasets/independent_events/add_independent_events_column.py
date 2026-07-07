#!/usr/bin/env python3
"""
Add an `independent_events` column to substitution tables using counts
from the per-site independent substitution files.

Matches rows by `pos_aln` (in the substitution table) to `site`
(in the independent events file). The new column is appended as the
last column of each TSV.

Usage:
    python add_independent_events_column.py
"""

import csv
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent
EVENTS_DIR = Path(__file__).resolve().parent

PAIRS = [
    (
        DATASETS_DIR / "dataset2_west_africa.substitution_table.tsv",
        EVENTS_DIR / "dataset2_independent_substitutions.txt",
    ),
    (
        DATASETS_DIR / "dataset3_gisaid.substitution_table.tsv",
        EVENTS_DIR / "dataset3_independent_substitutions.txt",
    ),
]


def load_events(events_path):
    """Return dict {site (int): independent_events (str)}."""
    events = {}
    with open(events_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            events[int(row["site"])] = row["independent_events"]
    return events


def add_column(tsv_path, events_path):
    events = load_events(events_path)

    with open(tsv_path) as f:
        rows = list(csv.reader(f, delimiter="\t"))

    header = rows[0]
    if "independent_events" in header:
        idx = header.index("independent_events")
        header = header[:idx] + header[idx + 1 :]
        rows = [header] + [r[:idx] + r[idx + 1 :] for r in rows[1:]]

    pos_aln_idx = header.index("pos_aln")
    new_header = header + ["independent_events"]

    missing = 0
    new_rows = [new_header]
    for row in rows[1:]:
        pos_aln = int(row[pos_aln_idx])
        ie = events.get(pos_aln)
        if ie is None:
            missing += 1
            ie = ""
        new_rows.append(row + [ie])

    with open(tsv_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(new_rows)

    print(f"{tsv_path.name}: {len(new_rows) - 1} rows, {missing} missing")


def main():
    for tsv_path, events_path in PAIRS:
        add_column(tsv_path, events_path)


if __name__ == "__main__":
    main()
