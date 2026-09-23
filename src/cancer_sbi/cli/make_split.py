"""Create the fixed train/test split over simulations.

Replaces ``SetTransformer_NPE/data_preprocessing.py``, a nine-line top-level
script that read a hard-coded simulation directory and wrote
``train_test_split.pkl`` into whatever directory it was run from.

Example:
    Recreate the split the published runs used::

        python -m cancer_sbi.cli.make_split \\
            --data-root /path/to/simulation_outputs \\
            --out train_test_split.pkl

The split is deterministic: the same simulation directory and the same
``--random-state`` always produce the same lists, because
``sklearn.model_selection.train_test_split`` is seeded. That is why the pickle
could be committed and shared between the three model folders.

**Overwriting an existing split invalidates every model already trained against
it**, so this command refuses to do that unless you pass ``--force``. The
original script overwrote silently.

Heavy imports happen inside :func:`main` so that ``--help`` works without numpy
or scikit-learn.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from cancer_sbi.cli import DATA_ROOT_ENV, add_data_root_argument, require_path

#: What the original script called its output, in the directory it was run from.
DEFAULT_SPLIT_FILENAME = "train_test_split.pkl"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for ``python -m cancer_sbi.cli.make_split``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cancer_sbi.cli.make_split",
        description=(
            "Split the simulations into a training and a test set and pickle "
            "the two lists of simulation names. Run this once; train and "
            "evaluate then point at the file it writes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_root_argument(parser)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_SPLIT_FILENAME),
        help=(
            "Where to write the split pickle. A relative path is resolved "
            "against the directory you run the command from; the absolute path "
            "actually written is printed."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of simulations held out. The published split used 0.2.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=123,
        help=(
            "Seed for scikit-learn's shuffle. The published split used 123; "
            "changing it produces a different, incompatible split."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite --out if it already exists. Without this the command "
            "stops, because replacing a split file silently invalidates every "
            "checkpoint trained against it."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the split command.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 when the output exists and
        ``--force`` was not given.
    """
    args = build_parser().parse_args(argv)

    # Local imports: see the module docstring.
    from cancer_sbi.data.splits import create_split, save_split

    data_root = require_path(args.data_root, "--data-root", DATA_ROOT_ENV)
    out_path = Path(args.out)

    if out_path.exists() and not args.force:
        print(
            f"error: {out_path.resolve()} already exists. Every model trained "
            f"against it would become incomparable if it changed. Pass --force "
            f"if you really mean to replace it."
        )
        return 1

    train_ids, test_ids = create_split(
        data_root, test_size=args.test_size, random_state=args.random_state
    )
    written = save_split(out_path, train_ids, test_ids)

    print(
        f"Saved split to {written.resolve()}\n"
        f"  {len(train_ids)} training sims, {len(test_ids)} test sims "
        f"(test_size={args.test_size}, random_state={args.random_state})"
    )
    print(
        "Note: the two dataset classes filter these lists differently, so the "
        "number of sims a model actually trains on is smaller than the number "
        "above -- see 'Known issues' in docs/REFACTOR_NOTES.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
