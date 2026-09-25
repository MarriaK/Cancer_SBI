"""Build the preprocessed clone-set cache.

``MODEL_IMPROVEMENT_PLAN.md`` §5 step 3 / ``CODEBASE_IMPROVEMENT_PLAN.md`` A2:
loading one sim means opening 25 gzipped files and running an expensive
lexicographic row sort, which is where ~100% of an epoch goes. This writes the
result of that work once, as float32 ``(n_sims, num_trials, top_k, 45)``.

**Nothing here reimplements the transform.** The rows are produced by
:meth:`cancer_sbi.data.clone_sets.CNASimsDataset._load_and_process_trial`, which
calls the live :func:`~cancer_sbi.data.clone_sets.top_frequent_rows_tensor`, and
the sim filter is the dataset's own scan -- including trap 10, "a sim with even
one missing trial file is dropped entirely". Bit-identity with the live loader
is therefore by construction rather than by luck, and the manifest records
``sha1(inspect.getsource(top_frequent_rows_tensor))`` so that editing that
function invalidates the cache instead of silently serving stale tensors.

Example::

    python utilities/build_clone_cache.py \\
        --root ../data/Guassian_Normal/simulation_outputs \\
        --split ../data/train_test_split.pkl \\
        --out ../data/cache/clone_top100_v1 \\
        --workers 6 --top-k 100

Run the verifier (``utilities/verify_clone_cache.py``) against the result before
any GPU time is spent on it.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow "python utilities/build_clone_cache.py" from src/ as well as -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancer_sbi.data.clone_sets import (  # noqa: E402
    CACHE_MANIFEST_FILENAME,
    CACHE_TRIAL_COUNTS_FILENAME,
    CACHE_SIM_IDS_FILENAME,
    CACHE_THETA_FILENAME,
    CACHE_X_FILENAME,
    CNASimsDataset,
    discover_sim_trials,
    top_frequent_rows_source_sha1,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for this script.
    """
    parser = argparse.ArgumentParser(
        prog="build_clone_cache.py",
        description=(
            "Precompute the clone-set tensors for every simulation in a split "
            "and write them as a float32 memmap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, type=Path, help="Directory of sim*/.")
    parser.add_argument(
        "--split",
        required=True,
        type=Path,
        help=(
            "Split pickle. Every id it names (train, val and test) is cached, "
            "so one cache serves all three loaders."
        ),
    )
    parser.add_argument("--out", required=True, type=Path, help="Cache directory.")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes.")
    parser.add_argument("--top-k", type=int, default=100, help="Clones kept per trial.")
    parser.add_argument(
        "--num-trials", type=int, default=25, help="Trials per sim (the T dimension)."
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=None,
        help=(
            "Cache sims with at least this many complete trial files instead "
            "of only the complete ones (trap 10). The missing slots are stored "
            "as NaN and the real count per sim is written to "
            "trial_counts.npy. Default: unset, i.e. complete sims only, which "
            "is byte-identical to a cache built before this flag existed."
        ),
    )
    parser.add_argument(
        "--start-method",
        choices=["fork", "spawn", "forkserver"],
        default="fork",
        help="multiprocessing start method. Use spawn if fork misbehaves.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing cache directory's files.",
    )
    return parser


#: Set once per worker process by :func:`_init_worker`.
_WORKER: Dict[str, object] = {}


def _init_worker(
    root: str,
    top_k: int,
    num_trials: int,
    out_dir: str,
    sim_names: Sequence[str],
    min_trials: Optional[int] = None,
) -> None:
    """Build this worker's dataset view and open the output memmaps.

    Args:
        root: Directory of ``sim*/``.
        top_k: Clones kept per trial.
        num_trials: Trials per sim.
        out_dir: Cache directory holding the already-created ``.npy`` files.
        sim_names: The sims this worker is responsible for. The dataset is
            restricted to them so the constructor does not re-read every
            ``parameters.pkl`` in the tree once per worker.
        min_trials: Passed straight to the dataset, so this worker keeps the
            same sims the parent's scan kept. ``None`` is the complete-only
            rule.
    """
    dataset = CNASimsDataset(
        root,
        num_trials_per_sim=num_trials,
        top_k=top_k,
        sim_ids=list(sim_names),
        min_trials=min_trials,
    )
    _WORKER["dataset"] = dataset
    _WORKER["by_name"] = {
        os.path.basename(item["sim_dir"]): item for item in dataset.items
    }
    _WORKER["x"] = np.lib.format.open_memmap(
        os.path.join(out_dir, CACHE_X_FILENAME), mode="r+"
    )
    _WORKER["theta"] = np.lib.format.open_memmap(
        os.path.join(out_dir, CACHE_THETA_FILENAME), mode="r+"
    )
    counts_path = os.path.join(out_dir, CACHE_TRIAL_COUNTS_FILENAME)
    # Same disjoint-row discipline as X and theta: one row per sim, and no two
    # workers own the same sim.
    _WORKER["counts"] = (
        np.lib.format.open_memmap(counts_path, mode="r+")
        if os.path.exists(counts_path)
        else None
    )


def _fill_rows(task: Tuple[str, int]) -> Tuple[str, int]:
    """Fill one cache row from one simulation.

    Args:
        task: ``(sim_name, row)`` -- the sim to read and the row index to write.
            Rows are assigned up front and never shared, so no two workers ever
            touch the same bytes of the memmap.

    Returns:
        The task it was given, so the parent can count completions.

    Raises:
        KeyError: If this worker's dataset does not hold ``sim_name``.
    """
    sim_name, row = task
    dataset: CNASimsDataset = _WORKER["dataset"]  # type: ignore[assignment]
    item = _WORKER["by_name"][sim_name]  # type: ignore[index]
    x_mm = _WORKER["x"]
    theta_mm = _WORKER["theta"]
    counts_mm = _WORKER.get("counts")

    sim_dir = item["sim_dir"]
    avail = item["available_trials"][: dataset.num_trials]
    for slot, trial_idx in enumerate(avail):
        # The live code path, not a copy of it.
        x_mm[row, slot] = dataset._load_and_process_trial(sim_dir, trial_idx).numpy()
    # Slots len(avail)..num_trials-1 are left exactly as main() pre-filled them,
    # i.e. NaN -- the same sentinel the uncached __getitem__ writes. With the
    # complete-only rule there are none, so nothing is left unwritten.
    if counts_mm is not None:
        counts_mm[row] = len(avail)
    theta_mm[row] = item["y"].numpy()
    return task


def main(argv: Optional[List[str]] = None) -> int:
    """Build the cache.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        0 on success, 1 on a refusal (existing cache without ``--force``).
    """
    args = build_parser().parse_args(argv)
    from cancer_sbi.data.splits import load_split

    root = str(Path(args.root).resolve())
    out_dir = Path(args.out).resolve()

    split = load_split(args.split)
    wanted: List[str] = []
    for key in ("train_ids", "val_ids", "test_ids"):
        if key in split:
            wanted.extend(str(name) for name in split[key])
    wanted_set = set(wanted)

    # Gate 2 of the three cache gates: the sim count must be visible, not
    # assumed. 888 on the laptop, 3,600 on the cluster.
    discovered = discover_sim_trials(root)
    print(f"[build_clone_cache] simulations discovered under {root}: {len(discovered)}")
    print(f"[build_clone_cache] simulation ids named by {args.split}: {len(wanted_set)}")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"error: {out_dir} is not empty. Pass --force to overwrite.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    # One full scan in the parent, which is where the "all num_trials trials
    # present" filter (trap 10) is applied -- by the dataset, not by this file.
    scan = CNASimsDataset(
        root,
        num_trials_per_sim=args.num_trials,
        top_k=args.top_k,
        sim_ids=sorted(wanted_set),
        min_trials=args.min_trials,
    )
    sim_names = [os.path.basename(item["sim_dir"]) for item in scan.items]
    n_sims = len(sim_names)
    print(
        f"[build_clone_cache] simulations surviving the dataset's filters: {n_sims} "
        f"(dropped {len(wanted_set) - n_sims} of the split's ids)"
    )

    x_path = out_dir / CACHE_X_FILENAME
    theta_path = out_dir / CACHE_THETA_FILENAME
    x_shape = (n_sims, args.num_trials, args.top_k, 45)
    x_mm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=x_shape
    )
    theta_mm = np.lib.format.open_memmap(
        theta_path, mode="w+", dtype=np.float32, shape=(n_sims, 44)
    )
    if args.min_trials is not None:
        # open_memmap zero-fills, and a zero clone row is a *legal* padded clone
        # (trap 8's first sentinel), so a partial sim's unwritten trial slots
        # have to be set to the trial-level sentinel before the workers start.
        # Skipped entirely with the complete-only rule, where every slot of
        # every row is overwritten -- which is what keeps that cache
        # byte-identical to one built before this flag existed.
        x_mm[:] = np.nan
        x_mm.flush()
    counts_mm = None
    if args.min_trials is not None:
        counts_mm = np.lib.format.open_memmap(
            out_dir / CACHE_TRIAL_COUNTS_FILENAME,
            mode="w+",
            dtype=np.int32,
            shape=(n_sims,),
        )
        del counts_mm
    del x_mm, theta_mm  # workers reopen their own handles

    np.save(out_dir / CACHE_SIM_IDS_FILENAME, np.array(sim_names))

    tasks = list(zip(sim_names, range(n_sims)))
    n_workers = max(1, int(args.workers))
    # Contiguous chunks: each worker's dataset then covers the sims it is asked
    # for and nothing else.
    chunks = [sim_names[i::n_workers] for i in range(n_workers)]

    started = time.time()
    if n_workers == 1:
        _run_chunk(
            (
                root,
                args.top_k,
                args.num_trials,
                str(out_dir),
                sim_names,
                tasks,
                args.min_trials,
            )
        )
    else:
        by_chunk = {name: idx for idx, chunk in enumerate(chunks) for name in chunk}
        jobs = []
        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            chunk_tasks = [t for t in tasks if by_chunk[t[0]] == idx]
            jobs.append(
                (
                    root,
                    args.top_k,
                    args.num_trials,
                    str(out_dir),
                    chunk,
                    chunk_tasks,
                    args.min_trials,
                )
            )
        ctx = mp.get_context(args.start_method)
        with ctx.Pool(processes=len(jobs)) as pool:
            for done in pool.imap_unordered(_run_chunk, jobs):
                print(f"[build_clone_cache] chunk done: {done} sims")
    elapsed = time.time() - started

    manifest = {
        "top_k": int(args.top_k),
        "num_trials": int(args.num_trials),
        "root": root,
        "split": str(Path(args.split).resolve()),
        "n_sims": int(n_sims),
        "n_discovered": int(len(discovered)),
        "x_shape": list(x_shape),
        "theta_shape": [int(n_sims), 44],
        "dtype": "float32",
        "build_seconds": round(elapsed, 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "top_frequent_rows_tensor_sha1": top_frequent_rows_source_sha1(),
    }
    if args.min_trials is not None:
        # Written only when the flag is given, so a complete-only cache's
        # manifest is unchanged. A reader that finds no key falls back to
        # num_trials, which is exactly the rule such a cache was built under.
        manifest["min_trials"] = int(args.min_trials)
    with (out_dir / CACHE_MANIFEST_FILENAME).open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    size_mb = (x_path.stat().st_size + theta_path.stat().st_size) / 1e6
    print(
        f"[build_clone_cache] wrote {n_sims} sims to {out_dir} "
        f"({size_mb:.1f} MB) in {elapsed:.1f}s using {n_workers} worker(s)"
    )
    print("[build_clone_cache] now run utilities/verify_clone_cache.py against it.")
    return 0


def _run_chunk(
    job: Tuple[
        str, int, int, str, Sequence[str], Sequence[Tuple[str, int]], Optional[int]
    ]
) -> int:
    """Worker entry point: initialise this chunk's dataset, then fill its rows.

    Args:
        job: ``(root, top_k, num_trials, out_dir, chunk, tasks, min_trials)``.
            ``chunk`` is
            the sim names this worker owns and ``tasks`` the ``(sim_name, row)``
            pairs for exactly those sims -- disjoint from every other chunk's,
            which is what makes concurrent writes into one ``open_memmap`` safe.

    Returns:
        How many rows this worker wrote.
    """
    root, top_k, num_trials, out_dir, chunk, tasks, min_trials = job
    _init_worker(root, top_k, num_trials, out_dir, chunk, min_trials)
    for task in tasks:
        _fill_rows(task)
    _WORKER["x"].flush()  # type: ignore[union-attr]
    _WORKER["theta"].flush()  # type: ignore[union-attr]
    if _WORKER.get("counts") is not None:
        _WORKER["counts"].flush()  # type: ignore[union-attr]
    return len(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
