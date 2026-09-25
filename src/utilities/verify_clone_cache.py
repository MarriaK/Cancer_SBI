"""Gate 1 of the three cache gates: bit-exactness.

``MODEL_IMPROVEMENT_PLAN.md`` §5 step 3: *"20 random sims x 3 trials recomputed
through the live code path and compared with ``np.array_equal`` -- exact, not
``allclose``. One differing value and the cache is not the data the published
numbers came from."*

Three assertions, and a non-zero exit on any of them:

1. every sampled ``(sim, trial)`` recomputed through the live
   :meth:`~cancer_sbi.data.clone_sets.CNASimsDataset._load_and_process_trial`
   is ``np.array_equal`` to the cached slice,
2. every sampled sim's cached theta row equals its ``parameters.pkl[2:]``,
3. the manifest's recorded source hash of
   :func:`~cancer_sbi.data.clone_sets.top_frequent_rows_tensor` matches the live
   function.

The sims sampled are drawn from those ``--split`` names -- pooling
``train_ids``, ``val_ids`` and ``test_ids``, since one cache feeds all three
loaders -- so the gate checks the rows the run will actually read.

Example::

    python utilities/verify_clone_cache.py \\
        --root ../data/Guassian_Normal/simulation_outputs \\
        --split ../data/train_test_split.pkl \\
        --cache ../data/cache/clone_top100_v1 \\
        --n-sims 20 --n-trials 3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancer_sbi.data.clone_sets import (  # noqa: E402
    CACHE_MANIFEST_FILENAME,
    CACHE_TRIAL_COUNTS_FILENAME,
    CACHE_SIM_IDS_FILENAME,
    CACHE_THETA_FILENAME,
    CACHE_X_FILENAME,
    CNASimsDataset,
    load_pickle,
    top_frequent_rows_source_sha1,
)
from cancer_sbi.data.splits import load_split  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for this script.
    """
    parser = argparse.ArgumentParser(
        prog="verify_clone_cache.py",
        description="Check a clone cache against the live loading code, exactly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path, help="Directory of sim*/.")
    parser.add_argument(
        "--split",
        required=True,
        type=Path,
        help=(
            "Split pickle. The spot-check is restricted to the sims it names, "
            "so a cache can be verified against the split it will actually be "
            "trained on rather than against an arbitrary 20 of its rows."
        ),
    )
    parser.add_argument("--cache", required=True, type=Path, help="Cache directory.")
    parser.add_argument("--n-sims", type=int, default=20, help="Sims to sample.")
    parser.add_argument("--n-trials", type=int, default=3, help="Trials per sampled sim.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the sampling.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Verify the cache.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        0 when every check passes, 1 on the first kind of mismatch found. The
        exit code is the contract: this runs before GPU time is spent.
    """
    args = build_parser().parse_args(argv)

    cache_dir = Path(args.cache).resolve()
    root = str(Path(args.root).resolve())
    failures: List[str] = []

    with (cache_dir / CACHE_MANIFEST_FILENAME).open() as handle:
        manifest = json.load(handle)

    live_hash = top_frequent_rows_source_sha1()
    cached_hash = manifest.get("top_frequent_rows_tensor_sha1")
    if cached_hash != live_hash:
        failures.append(
            f"manifest hash {cached_hash!r} != live top_frequent_rows_tensor "
            f"hash {live_hash!r}"
        )
    else:
        print(f"[verify] manifest source hash matches live function: {live_hash}")

    sim_ids = [str(name) for name in np.load(cache_dir / CACHE_SIM_IDS_FILENAME)]
    rows = {name: idx for idx, name in enumerate(sim_ids)}
    # A manifest with no min_trials key predates the flag: it holds complete
    # sims only, i.e. the rule num_trials describes.
    cache_min_trials = manifest.get("min_trials")
    counts_path = cache_dir / CACHE_TRIAL_COUNTS_FILENAME
    trial_counts = np.load(counts_path) if counts_path.exists() else None
    if (cache_min_trials is None) != (trial_counts is None):
        failures.append(
            f"manifest min_trials={cache_min_trials!r} but "
            f"{CACHE_TRIAL_COUNTS_FILENAME} is "
            f"{'absent' if trial_counts is None else 'present'} -- a partial "
            f"cache needs both, a complete-only cache neither"
        )
    if cache_min_trials is not None:
        print(f"[verify] cache is a PARTIAL cache: min_trials={cache_min_trials}")

    x_mm = np.load(cache_dir / CACHE_X_FILENAME, mmap_mode="r")
    theta_mm = np.load(cache_dir / CACHE_THETA_FILENAME, mmap_mode="r")
    print(f"[verify] cache holds {len(sim_ids)} sims, X shape {x_mm.shape}")

    # The split decides which rows are worth checking: a cache is only ever used
    # for the sims some split names, and a row no split reaches is not what the
    # published numbers came from. Every key is pooled (train, val and test),
    # because one cache serves all three loaders.
    split = load_split(args.split)
    wanted = {
        str(name)
        for key in ("train_ids", "val_ids", "test_ids")
        for name in (split.get(key) if split.get(key) is not None else [])
    }
    in_split = [name for name in sim_ids if name in wanted]
    missing_from_cache = sorted(wanted - set(sim_ids))
    print(
        f"[verify] split {args.split} names {len(wanted)} sims; "
        f"{len(in_split)} of them are in the cache"
    )
    if missing_from_cache:
        print(
            f"[verify] note: {len(missing_from_cache)} split sims are not cached "
            f"(the builder drops sims below its trial bar), e.g. "
            f"{', '.join(missing_from_cache[:5])}"
        )
    if not in_split:
        failures.append(
            f"no sim named by {args.split} is in the cache -- this cache was "
            f"built for a different split"
        )
        in_split = sim_ids

    rng = np.random.default_rng(args.seed)
    n_sims = min(args.n_sims, len(in_split))
    sampled = [in_split[i] for i in rng.choice(len(in_split), size=n_sims, replace=False)]

    # The live path, with no cache attached: this is the code the published
    # numbers came from, and the only thing the cache is allowed to equal.
    dataset = CNASimsDataset(
        root,
        num_trials_per_sim=int(manifest["num_trials"]),
        top_k=int(manifest["top_k"]),
        sim_ids=sampled,
        min_trials=cache_min_trials,
    )
    by_name = {os.path.basename(item["sim_dir"]): item for item in dataset.items}

    n_trial_checks = 0
    for name in sampled:
        item = by_name.get(name)
        if item is None:
            failures.append(f"{name} is in the cache but the live dataset dropped it")
            continue
        row = rows[name]
        avail = item["available_trials"][: dataset.num_trials]
        slots = rng.choice(len(avail), size=min(args.n_trials, len(avail)), replace=False)
        for slot in slots:
            live = dataset._load_and_process_trial(item["sim_dir"], avail[int(slot)])
            live_np = live.numpy()
            cached = np.asarray(x_mm[row, int(slot)])
            n_trial_checks += 1
            if not np.array_equal(live_np, cached):
                n_diff = int((live_np != cached).sum())
                failures.append(
                    f"X mismatch at {name} (row {row}) trial slot {int(slot)}: "
                    f"{n_diff} of {live_np.size} values differ"
                )

        # A partial sim's unwritten slots are the one thing a value-by-value
        # comparison of the real trials cannot see: a cache that forgot to
        # NaN-fill them would serve zero clone rows, which the encoders read as
        # real clones of frequency 0 rather than as an absent replicate.
        if trial_counts is not None:
            if int(trial_counts[row]) != len(avail):
                failures.append(
                    f"trial_counts mismatch at {name} (row {row}): cache says "
                    f"{int(trial_counts[row])}, the live dataset has {len(avail)}"
                )
            pad = np.asarray(x_mm[row, len(avail):])
            if pad.size and not np.isnan(pad).all():
                failures.append(
                    f"padding not NaN at {name} (row {row}): slots "
                    f"{len(avail)}..{dataset.num_trials - 1} must be all NaN"
                )

        params = load_pickle(os.path.join(item["sim_dir"], dataset.params_filename))
        theta_live = np.asarray(params[2:], dtype=np.float32)
        if not np.array_equal(theta_live, np.asarray(theta_mm[row])):
            failures.append(f"theta mismatch at {name} (row {row})")

    print(
        f"[verify] checked {n_sims} sims, {n_trial_checks} trials, "
        f"{n_sims} theta rows with np.array_equal"
    )

    if failures:
        print(f"[verify] FAILED ({len(failures)} problem(s)):")
        for line in failures[:20]:
            print(f"  - {line}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print("[verify] OK: cache is bit-identical to the live code path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
