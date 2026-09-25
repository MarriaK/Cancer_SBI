#!/usr/bin/env python
"""Tabulate every run of the 2026-09-24 repair campaign from its metrics_summary.csv.

    python utilities/campaign_summary.py --results ../results/2026-09-24 [--markdown]

One row per run directory that holds a metrics_summary.csv; the published models
(results/metrics_summary.csv) are prepended for reference. Nothing is typed by
hand: every number is read from the files the analyze jobs wrote.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

COLS = [
    ("model", "model", "s"),
    ("n_cases", "n", "d"),
    ("mean_true_r2", "R2", ".3f"),
    ("mean_pearson_r2", "r2", ".3f"),
    ("n_arms_ks_reject_fdr05", "SBC", "d"),
    ("mean_contraction", "contr", ".2f"),
    ("coverage_95", "cov95", ".3f"),
    ("pooled_std_z", "std_z", ".2f"),
    ("mean_log_prob_true", "logp", ".2f"),
]


def run_sort_key(name: str) -> tuple:
    """R0 < R1 < ... < D0 < AT0 < H0; a run's seeds (s1, s2), ensembles and
    recalibrated copies (rc) sort directly after their base run."""
    m = re.match(r"([A-Z]+)(\d+)(.*)", name)
    if not m:
        return (9, 0, name)
    head, num, rest = m.groups()
    order = {"R": 0, "D": 1, "AT": 2, "H": 3}
    return (order.get(head, 9), int(num), rest)


def read_row(path: Path) -> dict:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1]


def fmt(row: dict, key: str, spec: str) -> str:
    v = row.get(key, "")
    if spec == "s":
        return str(v)
    try:
        return format(int(float(v)) if spec == "d" else float(v), spec)
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="../results/2026-09-24")
    ap.add_argument("--published", default="../results/published/metrics_summary.csv")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.published):
        with open(args.published) as fh:
            for r in csv.DictReader(fh):
                rows.append(("published-" + r["model"], r))
    for d in sorted(Path(args.results).iterdir(), key=lambda p: run_sort_key(p.name)):
        f = d / "metrics_summary.csv"
        if f.exists():
            rows.append((d.name, read_row(f)))

    header = ["run"] + [c[1] for c in COLS]
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for name, r in rows:
            print("| " + " | ".join([name] + [fmt(r, k, s) for k, _, s in COLS]) + " |")
    else:
        print(f"{'run':16s}" + "".join(f"{h:>9s}" for h in header[1:]))
        for name, r in rows:
            print(f"{name:16s}" + "".join(f"{fmt(r, k, s):>9s}" for k, _, s in COLS))


if __name__ == "__main__":
    main()
