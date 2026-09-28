#!/usr/bin/env python
"""Check that every simulation in a run directory is complete.

Stage 2 of the three-stage data pipeline:

    dataSimulator_MPI.py  ->  update_checker.py  ->  fixed_dataSimulation.py
      (simulate)                (this script)          (re-run what failed)

Writes ``failed_simulations.txt``, one simulation name per line, which is
exactly the format ``fixed_dataSimulation.py --failed_list`` expects, plus a
detailed report naming what was wrong with each.

    python update_checker.py --root ~/cancer/data/Guassian_Normal/simulation_outputs
    python update_checker.py --root <dir> --out-dir reports --ntrials 25

Exit status is 0 when every simulation is complete and 1 otherwise, so it can
gate the next step of a job script:

    python update_checker.py --root "$DATA" || sbatch fixed_dataSimulation.sh

A simulation is complete when it holds ``parameters.pkl``, exactly the expected
numbered trial directories, and each of those holds every required file with a
non-zero size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

EXPECTED_NUM_TRIALS = 25
REQUIRED_SIM_FILES = ["parameters.pkl"]
REQUIRED_TRIAL_FILES = ["results.pkl", "CNratios_all.pkl.gz", "sim.log"]


def sim_sort_key(name: str) -> tuple:
    """Sort sim10 after sim9, not after sim1.

    Plain lexical sorting gives sim1, sim10, sim100, sim1000, sim1001, ... which
    makes a report over 3,600 simulations very hard to read.
    """
    digits = "".join(c for c in name if c.isdigit())
    return (0, int(digits)) if digits else (1, 0), name


def is_trial_folder(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def check_file(path: Path, label: str, problems: List[str]) -> None:
    """Require the file to exist AND to be non-empty.

    Existence alone is not enough here: this project has a directory of
    simulations whose run files are all present and all zero bytes
    (`archive/Guassian_Normal_parameters_only`). A checker that only tests
    existence reports those as complete, and the failure then surfaces much
    later as an unreadable pickle in the middle of a training job.
    """
    if not path.exists():
        problems.append(f"Missing {label}: {path.name}")
    elif path.stat().st_size == 0:
        problems.append(f"Empty (0 bytes) {label}: {path.name}")


def check_one_sim(sim_dir: Path, ntrials: int) -> Dict:
    problems: List[str] = []

    for fname in REQUIRED_SIM_FILES:
        check_file(sim_dir / fname, "sim-level file", problems)

    found = {int(p.name) for p in sim_dir.iterdir() if is_trial_folder(p)}
    expected = set(range(1, ntrials + 1))

    if missing := sorted(expected - found):
        problems.append(f"Missing trial folders: {missing}")
    if extra := sorted(found - expected):
        problems.append(f"Unexpected extra trial folders: {extra}")

    for i in sorted(expected & found):
        trial_dir = sim_dir / str(i)
        for fname in REQUIRED_TRIAL_FILES:
            check_file(trial_dir / fname, f"file in trial {i}", problems)

    return {"sim_name": sim_dir.name, "ok": not problems, "problems": problems}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root", type=Path, required=True,
        help="Directory holding the sim<N>/ folders, e.g. "
             "~/cancer/data/Guassian_Normal/simulation_outputs",
    )
    p.add_argument(
        "--ntrials", type=int, default=EXPECTED_NUM_TRIALS,
        help=f"Replicate runs expected per simulation (default {EXPECTED_NUM_TRIALS}).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("."),
        help="Where to write the two report files (default: the current directory).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Print the summary only, not the per-simulation problems.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    root = args.root.expanduser()
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 2

    sim_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("sim")),
        key=lambda p: sim_sort_key(p.name),
    )
    if not sim_dirs:
        print(f"error: no sim*/ directories under {root}", file=sys.stderr)
        return 2

    reports = []
    for n, sim_dir in enumerate(sim_dirs, 1):
        reports.append(check_one_sim(sim_dir, args.ntrials))
        # 3,600 simulations over a network filesystem is minutes of silence.
        if n % 200 == 0 or n == len(sim_dirs):
            print(f"  checked {n}/{len(sim_dirs)}", file=sys.stderr, flush=True)

    failed = [r for r in reports if not r["ok"]]

    print("=" * 70)
    print(f"Root: {root}")
    print(f"Total sim folders checked: {len(sim_dirs)}")
    print(f"Complete simulations:      {len(sim_dirs) - len(failed)}")
    print(f"Incomplete simulations:    {len(failed)}")
    print("=" * 70)

    if failed and not args.quiet:
        print("\nIncomplete simulations:\n")
        for rep in failed:
            print(f"{rep['sim_name']}:")
            for problem in rep["problems"]:
                print(f"  - {problem}")
            print()

    args.out_dir.expanduser().mkdir(parents=True, exist_ok=True)
    names_path = args.out_dir.expanduser() / "failed_simulations.txt"
    detail_path = args.out_dir.expanduser() / "failed_simulations_detailed.txt"

    # Written even when empty, so a downstream --failed_list always has a file
    # to read and an empty run is distinguishable from a run that never happened.
    names_path.write_text("".join(r["sim_name"] + "\n" for r in failed))
    detail_path.write_text(
        "".join(
            f"{r['sim_name']}\n" + "".join(f"  - {p}\n" for p in r["problems"]) + "\n"
            for r in failed
        )
    )

    print("Saved:")
    print(f"  {names_path}")
    print(f"  {detail_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
