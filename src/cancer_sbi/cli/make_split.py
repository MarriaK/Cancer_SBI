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

``--add-val`` does something different and safer: it reads an existing split and
carves a seeded early-stopping set out of its ``train_ids``, leaving
``test_ids`` byte-identical. It never redraws the train/test partition --
``MODEL_IMPROVEMENT_PLAN.md`` P0.2 is explicit that redrawing it would make
every published number incomparable::

        python -m cancer_sbi.cli.make_split --add-val \\
            --frac 0.1 --seed 20260924 \\
            --in data/train_test_split.pkl \\
            --out data/train_val_test_split.pkl

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
        "--add-val",
        action="store_true",
        help=(
            "Do not draw a new split. Read --in, carve --frac of its train_ids "
            "into val_ids with --seed, and write the three-key split to --out. "
            "test_ids is copied through unchanged."
        ),
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=None,
        help="Existing split pickle to carve from. Required with --add-val.",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.1,
        help="Fraction of train_ids moved into val_ids by --add-val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260924,
        help=(
            "Seed for the --add-val carve. The same input file and the same "
            "seed always give the same val_ids. Deliberately NOT the same knob "
            "as --random-state, which seeds scikit-learn's train/test shuffle: "
            "the two seed different operations, and --random-state has to keep "
            "its published value of 123 while this one does not."
        ),
    )
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
            "Seed for scikit-learn's train/test shuffle, used when building a "
            "split from scratch. The published split used 123; changing it "
            "produces a different, incompatible split. The --add-val carve has "
            "its own seed, --seed, because it is a different operation."
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
    from cancer_sbi.data.splits import carve_val_ids, create_split, load_split, save_split

    out_path = Path(args.out)

    if args.add_val:
        if args.in_path is None:
            print("error: --add-val requires --in (the split to carve from).")
            return 1
        in_path = Path(args.in_path)
        if not in_path.exists():
            print(f"error: --in {in_path.resolve()} does not exist.")
            return 1
        if out_path.exists() and not args.force:
            print(
                f"error: {out_path.resolve()} already exists. Every model "
                f"trained against it would become incomparable if it changed. "
                f"Pass --force if you really mean to replace it."
            )
            return 1

        split = load_split(in_path)
        if "val_ids" in split:
            print(
                f"error: {in_path.resolve()} already has val_ids. Carving "
                f"again would take the new val set out of an already reduced "
                f"train set."
            )
            return 1

        old_train = split["train_ids"]
        test_ids = split["test_ids"]
        train_ids, val_ids = carve_val_ids(old_train, frac=args.frac, seed=args.seed)

        # test_ids is passed straight through, not recomputed: the whole point
        # of --add-val is that the reported test set does not move.
        written = save_split(out_path, train_ids, test_ids, val_ids=val_ids)
        print(
            f"Saved three-way split to {written.resolve()}\n"
            f"  {len(train_ids)} training sims, {len(val_ids)} validation sims, "
            f"{len(test_ids)} test sims "
            f"(carved from {len(old_train)} train ids, frac={args.frac}, "
            f"seed={args.seed})"
        )
        print("  test_ids copied unchanged from " + str(in_path.resolve()))
        return 0

    data_root = require_path(args.data_root, "--data-root", DATA_ROOT_ENV)

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
