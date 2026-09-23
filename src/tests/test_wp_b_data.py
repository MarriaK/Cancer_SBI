"""Work package B: split carving, loader plumbing and the clone cache.

Run from ``src/``::

    python -m pytest tests/test_wp_b_data.py -q

The clone-cache and loader tests need the local simulation tree. They are
skipped, not failed, when it is absent, so the file still runs on the cluster's
login node.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cancer_sbi.data.clone_sets import (  # noqa: E402
    CACHE_MANIFEST_FILENAME,
    CACHE_X_FILENAME,
    CNASimsDataset,
    top_frequent_rows_source_sha1,
)
from cancer_sbi.data.loaders import (  # noqa: E402
    _worker_kwargs,
    build_clone_set_dataloaders,
    build_dominant_clone_dataloaders,
)
from cancer_sbi.data.splits import carve_val_ids, load_split, save_split  # noqa: E402

DATA_ROOT = SRC.parent / "data" / "Guassian_Normal" / "simulation_outputs"
SPLIT_PATH = SRC.parent / "data" / "train_test_split.pkl"
CACHE_DIR = SRC.parent / "data" / "cache" / "clone_top100_v1"

needs_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir() or not SPLIT_PATH.exists(),
    reason="local simulation tree / split pickle not present",
)
needs_cache = pytest.mark.skipif(
    not (CACHE_DIR / CACHE_MANIFEST_FILENAME).exists(),
    reason="clone cache not built; run utilities/build_clone_cache.py",
)


def _run_make_split(args):
    """Run the make_split CLI in-process.

    Args:
        args: Argument list, without the program name.

    Returns:
        The command's exit code.
    """
    from cancer_sbi.cli.make_split import main

    return main(args)


def _sim_dirs_on_disk(names):
    """Keep only the ids that actually exist under DATA_ROOT.

    Args:
        names: Candidate simulation names.

    Returns:
        A list of the ones with a directory on disk.
    """
    return [n for n in names if (DATA_ROOT / str(n)).is_dir()]


# --------------------------------------------------------------------------
# 1. Three-way split
# --------------------------------------------------------------------------


@needs_data
def test_add_val_keeps_test_ids_identical(tmp_path):
    """--add-val must not move a single test id."""
    out = tmp_path / "three.pkl"
    code = _run_make_split(
        [
            "--add-val",
            "--frac",
            "0.1",
            "--seed",
            "20260924",
            "--in",
            str(SPLIT_PATH),
            "--out",
            str(out),
        ]
    )
    assert code == 0

    before = load_split(SPLIT_PATH)
    after = load_split(out)
    assert list(after["test_ids"]) == list(before["test_ids"])
    # byte-identical, not merely equal as a list
    assert np.asarray(after["test_ids"]).tobytes() == np.asarray(before["test_ids"]).tobytes()


@needs_data
def test_add_val_partitions_the_old_train_ids(tmp_path):
    """train and val are disjoint and together are exactly the old train set."""
    out = tmp_path / "three.pkl"
    assert _run_make_split(
        ["--add-val", "--in", str(SPLIT_PATH), "--out", str(out)]
    ) == 0

    before = load_split(SPLIT_PATH)
    after = load_split(out)
    old_train = set(map(str, before["train_ids"]))
    new_train = set(map(str, after["train_ids"]))
    val = set(map(str, after["val_ids"]))

    assert new_train & val == set()
    assert new_train | val == old_train
    assert len(after["train_ids"]) + len(after["val_ids"]) == len(before["train_ids"])
    assert len(val) == round(0.1 * len(old_train))


@needs_data
def test_add_val_is_deterministic(tmp_path):
    """The same seed gives the same carve; a different seed does not."""
    a, b, c = tmp_path / "a.pkl", tmp_path / "b.pkl", tmp_path / "c.pkl"
    for out, seed in ((a, "20260924"), (b, "20260924"), (c, "7")):
        assert _run_make_split(
            ["--add-val", "--seed", seed, "--in", str(SPLIT_PATH), "--out", str(out)]
        ) == 0

    va = list(map(str, load_split(a)["val_ids"]))
    vb = list(map(str, load_split(b)["val_ids"]))
    vc = list(map(str, load_split(c)["val_ids"]))
    assert va == vb
    assert va != vc


def test_carve_val_ids_rejects_bad_frac():
    """A frac outside (0, 1), or one that would empty a side, raises."""
    ids = [f"sim{i}" for i in range(10)]
    with pytest.raises(ValueError):
        carve_val_ids(ids, frac=0.0)
    with pytest.raises(ValueError):
        carve_val_ids(ids, frac=1.0)
    with pytest.raises(ValueError):
        carve_val_ids(ids, frac=0.001)


def test_two_key_files_still_load(tmp_path):
    """A split written without val_ids loads and simply has no val_ids key."""
    path = tmp_path / "two.pkl"
    save_split(path, np.array(["sim1", "sim2"]), np.array(["sim3"]))
    split = load_split(path)
    assert set(split) == {"train_ids", "test_ids"}
    assert list(split["train_ids"]) == ["sim1", "sim2"]


def test_save_split_writes_val_ids_when_given(tmp_path):
    """save_split carries val_ids through when it is given one."""
    path = tmp_path / "three.pkl"
    save_split(path, ["sim1"], ["sim3"], val_ids=["sim2"])
    split = load_split(path)
    assert list(split["val_ids"]) == ["sim2"]


@needs_data
def test_add_val_refuses_to_carve_twice(tmp_path):
    """Carving a file that already has val_ids is refused, not silently redone."""
    first, second = tmp_path / "a.pkl", tmp_path / "b.pkl"
    assert _run_make_split(
        ["--add-val", "--in", str(SPLIT_PATH), "--out", str(first)]
    ) == 0
    assert _run_make_split(
        ["--add-val", "--in", str(first), "--out", str(second)]
    ) == 1


# --------------------------------------------------------------------------
# 2. Loaders
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_ids():
    """A few sim ids that exist on disk, split into train / val / test."""
    if not DATA_ROOT.is_dir():
        pytest.skip("no local simulation tree")
    names = sorted(
        (d for d in os.listdir(DATA_ROOT) if d.startswith("sim")),
        key=lambda n: int(n[3:]),
    )[:9]
    return names[:3], names[3:6], names[6:9]


@needs_data
def test_clone_set_builder_returns_three_and_no_val(small_ids):
    """Without val_ids the middle element of the 3-tuple is None."""
    train, _, test = small_ids
    loaders = build_clone_set_dataloaders(str(DATA_ROOT), train, test, batch_size=2)
    assert len(loaders) == 3
    train_loader, val_loader, test_loader = loaders
    assert val_loader is None
    assert len(train_loader.dataset) > 0 and len(test_loader.dataset) > 0


@needs_data
def test_dominant_builder_returns_three_and_no_val(small_ids):
    """The dominant-clone builder has the same arity and the same None rule."""
    train, val, test = small_ids
    t, v, te = build_dominant_clone_dataloaders(str(DATA_ROOT), train, test, batch_size=2)
    assert v is None
    t2, v2, te2 = build_dominant_clone_dataloaders(
        str(DATA_ROOT), train, test, batch_size=2, val_ids=val
    )
    assert v2 is not None


@needs_data
def test_val_loader_draws_only_from_val_ids(small_ids):
    """With val_ids the validation loader sees those sims and nothing else."""
    train, val, test = small_ids
    _, val_loader, _ = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, batch_size=2, val_ids=val
    )
    assert val_loader is not None
    seen = {os.path.basename(item["sim_dir"]) for item in val_loader.dataset.items}
    assert seen <= set(val)
    assert seen.isdisjoint(set(train) | set(test))


@needs_data
def test_num_workers_is_rng_neutral(small_ids):
    """num_workers=2 gives the same batch order and the same tensors as 0."""
    train, _, test = small_ids
    train = _sim_dirs_on_disk(train)

    def collect(n_workers):
        torch.manual_seed(1234)
        loader, _, _ = build_clone_set_dataloaders(
            str(DATA_ROOT), train, test, batch_size=1, num_workers=n_workers
        )
        return [(x.clone(), m.clone(), y.clone()) for x, m, y in loader]

    a = collect(0)
    b = collect(2)
    assert len(a) == len(b)
    for (xa, ma, ya), (xb, mb, yb) in zip(a, b):
        assert torch.equal(xa, xb)
        assert torch.equal(ma, mb)
        assert torch.equal(ya, yb)


@needs_data
def test_num_workers_reaches_the_dataloader(small_ids):
    """The value must actually be handed to ``DataLoader``, at every site."""
    train, val, test = small_ids
    t0, v0, te0 = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, batch_size=2, val_ids=val
    )
    t2, v2, te2 = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, batch_size=2, val_ids=val, num_workers=2
    )
    assert (t0.num_workers, v0.num_workers, te0.num_workers) == (0, 0, 0)
    assert (t2.num_workers, v2.num_workers, te2.num_workers) == (2, 2, 2)


@needs_data
def test_num_workers_is_rng_neutral_across_two_epochs(small_ids):
    """Two epochs, not one: this is what ``persistent_workers`` broke.

    With ``persistent_workers=True`` each worker survives into epoch 2 carrying
    the RNG state epoch 1 left it in, instead of the per-epoch seed torch
    derives from the main generator, so a seeded ``num_workers=2`` run diverged
    from the identical ``num_workers=0`` run from the second epoch onwards. One
    epoch of comparison could not see it.
    """
    train, _, test = small_ids
    train = _sim_dirs_on_disk(train)

    def collect(n_workers, n_epochs=2):
        torch.manual_seed(1234)
        loader, _, _ = build_clone_set_dataloaders(
            str(DATA_ROOT), train, test, batch_size=1, num_workers=n_workers
        )
        return [
            [(x.clone(), m.clone(), y.clone()) for x, m, y in loader]
            for _ in range(n_epochs)
        ]

    a = collect(0)
    b = collect(2)
    assert len(a) == len(b) == 2
    for epoch, (ea, eb) in enumerate(zip(a, b)):
        assert len(ea) == len(eb) > 0, epoch
        for (xa, ma, ya), (xb, mb, yb) in zip(ea, eb):
            assert torch.equal(xa, xb), f"epoch {epoch + 1}"
            assert torch.equal(ma, mb), f"epoch {epoch + 1}"
            assert torch.equal(ya, yb), f"epoch {epoch + 1}"

    # The shuffle really does re-order between epochs, so the check above is
    # comparing two different orders and not two copies of one.
    assert [y.tolist() for _, _, y in a[0]] != [y.tolist() for _, _, y in a[1]]


@pytest.mark.parametrize("bad", [-1, -8])
def test_worker_kwargs_rejects_negative_num_workers(bad):
    """A negative count is a typo, not a request; it must not reach torch."""
    with pytest.raises(ValueError, match="num_workers must be >= 0"):
        _worker_kwargs(bad)


def test_worker_kwargs_does_not_set_persistent_workers():
    """Removed 2026-09-24: it breaks RNG neutrality from epoch 2."""
    assert _worker_kwargs(0) == {"num_workers": 0}
    assert _worker_kwargs(4) == {"num_workers": 4}


# --------------------------------------------------------------------------
# 3. The clone cache
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    """Build a 5-sim cache with the real builder and return (dir, sim names)."""
    if not DATA_ROOT.is_dir():
        pytest.skip("no local simulation tree")
    work = tmp_path_factory.mktemp("clone_cache")

    names = sorted(
        (d for d in os.listdir(DATA_ROOT) if d.startswith("sim")),
        key=lambda n: int(n[3:]),
    )
    # The builder filters to sims with all 25 trials, so offer a few extra.
    candidates = names[:12]
    split_path = work / "split.pkl"
    save_split(split_path, np.array(candidates[:8]), np.array(candidates[8:12]))

    out = work / "cache"
    cmd = [
        sys.executable,
        str(SRC / "utilities" / "build_clone_cache.py"),
        "--root",
        str(DATA_ROOT),
        "--split",
        str(split_path),
        "--out",
        str(out),
        "--workers",
        "2",
        "--top-k",
        "100",
        "--force",
    ]
    proc = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "simulations discovered" in proc.stdout  # the sim-count gate
    cached = [str(n) for n in np.load(out / "sim_ids.npy")]
    if len(cached) < 5:
        pytest.skip("fewer than 5 complete sims available locally")
    return out, cached, split_path


@needs_data
def test_cached_getitem_matches_uncached(tiny_cache):
    """The cache must be bit-identical to the live path, item by item."""
    cache_dir, cached, _ = tiny_cache
    sims = cached[:5]

    plain = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=sims)
    fast = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=sims, cache_dir=str(cache_dir))
    assert len(plain) == len(fast)

    for i in range(len(plain)):
        xa, ma, ya = plain[i]
        xb, mb, yb = fast[i]
        assert torch.equal(xa, xb)
        assert torch.equal(ma, mb)
        assert torch.equal(ya, yb)


@needs_data
def test_manifest_hash_mismatch_raises(tiny_cache, tmp_path):
    """A cache built by a different top_frequent_rows_tensor must not load."""
    cache_dir, cached, _ = tiny_cache
    corrupt = tmp_path / "hash_cache"
    shutil.copytree(cache_dir, corrupt)

    manifest_path = corrupt / CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest["top_frequent_rows_tensor_sha1"] == top_frequent_rows_source_sha1()
    manifest["top_frequent_rows_tensor_sha1"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="different version of"):
        CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=cached[:2], cache_dir=str(corrupt))


@needs_data
def test_sim_missing_from_cache_raises_at_construction(tiny_cache, tmp_path):
    """A cache that does not hold every sim served is an error, not a fallback.

    Both ``sim_ids.npy`` and the arrays lose the last sim, so this trips the
    "missing sims" gate and not the row-count one.
    """
    cache_dir, cached, _ = tiny_cache
    trimmed = tmp_path / "trimmed_cache"
    shutil.copytree(cache_dir, trimmed)
    np.save(trimmed / "sim_ids.npy", np.array(cached[:-1]))
    for name in ("X.npy", "theta.npy"):
        arr = np.load(trimmed / name)
        np.save(trimmed / name, arr[:-1])

    with pytest.raises(ValueError, match="missing 1 of") as excinfo:
        CNASimsDataset(
            str(DATA_ROOT), top_k=100, sim_ids=cached, cache_dir=str(trimmed)
        )
    # The message has to name the sims, or the rebuild is a guessing game.
    assert cached[-1] in str(excinfo.value)


@needs_data
def test_cache_row_count_mismatch_raises(tiny_cache, tmp_path):
    """``sim_ids.npy`` and ``X.npy`` disagreeing means every row is mislabelled."""
    cache_dir, cached, _ = tiny_cache
    skewed = tmp_path / "skewed_cache"
    shutil.copytree(cache_dir, skewed)
    np.save(skewed / "sim_ids.npy", np.array(cached[:-1]))

    with pytest.raises(ValueError, match="sim_ids.npy names"):
        CNASimsDataset(
            str(DATA_ROOT), top_k=100, sim_ids=cached[:2], cache_dir=str(skewed)
        )


@needs_data
def test_cache_built_from_another_root_raises(tiny_cache, tmp_path):
    """A cache from a different tree holds different tensors under the same names."""
    cache_dir, cached, _ = tiny_cache
    foreign = tmp_path / "foreign_cache"
    shutil.copytree(cache_dir, foreign)
    manifest_path = foreign / CACHE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["root"] = "/somewhere/else/simulation_outputs"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="was built from root"):
        CNASimsDataset(
            str(DATA_ROOT), top_k=100, sim_ids=cached[:2], cache_dir=str(foreign)
        )


@needs_data
def test_cache_root_comparison_is_by_resolved_path(tiny_cache):
    """An unresolved but equivalent root must still be accepted."""
    cache_dir, cached, _ = tiny_cache
    awkward = str(DATA_ROOT.parent / "." / DATA_ROOT.name)
    ds = CNASimsDataset(
        awkward, top_k=100, sim_ids=cached[:2], cache_dir=str(cache_dir)
    )
    assert len(ds) == 2


@needs_data
def test_getitem_cached_still_guards_an_unknown_sim(tiny_cache):
    """The per-item guard stays, behind the construction-time gate."""
    cache_dir, cached, _ = tiny_cache
    ds = CNASimsDataset(
        str(DATA_ROOT), top_k=100, sim_ids=cached[:2], cache_dir=str(cache_dir)
    )
    with pytest.raises(KeyError, match="not in the clone cache"):
        ds._getitem_cached(os.path.join(str(DATA_ROOT), "sim_does_not_exist"))


@needs_data
def test_top_k_mismatch_raises(tiny_cache):
    """A cache built for a different top_k must not be silently reshaped."""
    cache_dir, cached, _ = tiny_cache
    with pytest.raises(ValueError, match="top_k"):
        CNASimsDataset(
            str(DATA_ROOT), top_k=50, sim_ids=cached[:2], cache_dir=str(cache_dir)
        )


# --------------------------------------------------------------------------
# 4. The verifier
# --------------------------------------------------------------------------


def _run_verifier(cache_dir, split_path, n_sims=3, n_trials=2):
    """Run verify_clone_cache.py as a subprocess.

    Args:
        cache_dir: The cache to check.
        split_path: The split it was built from.
        n_sims: Sims to sample.
        n_trials: Trials per sim.

    Returns:
        The completed process.
    """
    cmd = [
        sys.executable,
        str(SRC / "utilities" / "verify_clone_cache.py"),
        "--root",
        str(DATA_ROOT),
        "--split",
        str(split_path),
        "--cache",
        str(cache_dir),
        "--n-sims",
        str(n_sims),
        "--n-trials",
        str(n_trials),
    ]
    return subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True)


@needs_data
def test_verifier_passes_on_a_good_cache(tiny_cache):
    """A freshly built cache verifies clean."""
    cache_dir, _, split_path = tiny_cache
    proc = _run_verifier(cache_dir, split_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "bit-identical" in proc.stdout


@needs_data
def test_verifier_restricts_the_spot_check_to_the_split(tiny_cache):
    """``--split`` is used, not merely accepted."""
    cache_dir, _, split_path = tiny_cache
    proc = _run_verifier(cache_dir, split_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "names" in proc.stdout and "of them are in the cache" in proc.stdout


@needs_data
def test_verifier_fails_when_the_split_names_no_cached_sim(tiny_cache, tmp_path):
    """A cache built for another split is a failure, not a silent pass."""
    cache_dir, _, _ = tiny_cache
    foreign = tmp_path / "foreign_split.pkl"
    save_split(foreign, np.array(["simZZZ1"]), np.array(["simZZZ2"]))

    proc = _run_verifier(cache_dir, foreign, n_sims=2, n_trials=1)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "built for a different split" in proc.stdout


@needs_data
def test_verifier_catches_one_corrupted_value(tiny_cache, tmp_path):
    """One flipped float anywhere in the sampled region fails the gate."""
    cache_dir, cached, split_path = tiny_cache
    corrupt = tmp_path / "corrupt_cache"
    shutil.copytree(cache_dir, corrupt)

    x = np.lib.format.open_memmap(corrupt / CACHE_X_FILENAME, mode="r+")
    # Corrupt every sim's first trial's first value, so any sample hits it.
    x[:, :, 0, 0] += np.float32(1.0)
    x.flush()
    del x

    proc = _run_verifier(corrupt, split_path, n_sims=3, n_trials=2)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAILED" in proc.stdout
    assert "X mismatch" in proc.stdout


def test_make_split_keeps_its_two_seeds_separate():
    """``--seed`` and ``--random-state`` are different knobs, not a rename.

    ``--random-state`` seeds scikit-learn's train/test shuffle and must stay at
    the published 123; ``--seed`` seeds the ``--add-val`` carve. Collapsing them
    into one flag would either move the published split or tie the carve to it.
    """
    from cancer_sbi.cli.make_split import build_parser

    args = build_parser().parse_args([])
    assert args.random_state == 123
    assert args.seed == 20260924

    args = build_parser().parse_args(["--seed", "7"])
    assert args.seed == 7 and args.random_state == 123
