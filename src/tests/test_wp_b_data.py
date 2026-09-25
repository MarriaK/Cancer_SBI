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
from cancer_sbi.data.splits import (  # noqa: E402
    carve_val_ids,
    complete_sim_ids,
    load_split,
    save_split,
)

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


# ---------------------------------------------------------------------------
# require_all_trials: the DominantClone path opting into the clone-set sim set.
#
# Trap 10 is the reason the three models' numbers were not comparable:
# CNASimsDataset drops a sim that is missing even one trial file, while
# SimulationDataset NaN-pads it and keeps it. ``complete_sim_ids`` is the one
# definition of the clone-set rule, and the first test below is what keeps it
# honest -- if it ever drifts from CNASimsDataset by a single sim, the "same
# simulations" claim is false.
# ---------------------------------------------------------------------------


def _mixed_ids(n_complete=6, n_incomplete=3):
    """Sim names that exist on disk: some complete, some missing trial files."""
    names = sorted(
        (d for d in os.listdir(DATA_ROOT) if d.startswith("sim")),
        key=lambda n: int(n[3:]),
    )
    complete = set(complete_sim_ids(str(DATA_ROOT), names))
    good = [n for n in names if n in complete][:n_complete]
    bad = [n for n in names if n not in complete][:n_incomplete]
    if len(good) < n_complete or len(bad) < n_incomplete:
        pytest.skip("local tree has no mix of complete and incomplete sims")
    # Interleaved, so an implementation that only ever drops a suffix fails.
    return sorted(good + bad, key=lambda n: int(n[3:])), good


@needs_data
def test_complete_sim_ids_matches_the_clone_set_dataset():
    """The rule is *exactly* CNASimsDataset's, not a re-derivation of it."""
    ids, _ = _mixed_ids(n_complete=12, n_incomplete=6)
    dataset = CNASimsDataset(str(DATA_ROOT), sim_ids=ids)
    kept_by_dataset = {os.path.basename(item["sim_dir"]) for item in dataset.items}
    assert set(complete_sim_ids(str(DATA_ROOT), ids)) == kept_by_dataset
    # And the filter really does something on this tree.
    assert len(kept_by_dataset) < len(ids)


@needs_data
def test_complete_sim_ids_keeps_the_input_order():
    ids, _ = _mixed_ids()
    kept = complete_sim_ids(str(DATA_ROOT), ids)
    assert kept == [name for name in ids if name in set(kept)]


@needs_data
def test_complete_sim_ids_drops_an_id_with_no_directory(tmp_path):
    """An id that is not on disk fails the parameters check, it does not raise."""
    ids, good = _mixed_ids()
    assert complete_sim_ids(str(DATA_ROOT), ["sim999999"]) == []
    assert complete_sim_ids(str(DATA_ROOT), good) == good


@needs_data
def test_dominant_builder_default_keeps_the_published_sim_set():
    """Off by default: trap 10 stands, incomplete sims are NaN-padded and kept."""
    ids, _ = _mixed_ids()
    train, _v, test = build_dominant_clone_dataloaders(
        str(DATA_ROOT), ids, ids, batch_size=2
    )
    assert len(train.dataset.sim_dirs) == len(ids)
    assert len(test.dataset.sim_dirs) == len(ids)


@needs_data
def test_dominant_builder_require_all_trials_restricts_every_partition(capsys):
    """Each of train/val/test is filtered, and each reports its own counts."""
    ids, good = _mixed_ids()
    train, val, test = build_dominant_clone_dataloaders(
        str(DATA_ROOT), ids, ids, batch_size=2, val_ids=ids, require_all_trials=True
    )
    for loader in (train, val, test):
        assert list(loader.dataset.sim_dirs) == good

    out = capsys.readouterr().out
    for partition in ("train", "val", "test"):
        assert (
            f"[dominant_clone] require_all_trials: {partition} "
            f"{len(ids)} -> {len(good)}" in out
        ), out


@needs_data
def test_dominant_builder_restricted_set_is_the_clone_set_models_set():
    """The point of the flag: the two families end up on the same sims."""
    ids, _ = _mixed_ids()
    _t, _v, test = build_dominant_clone_dataloaders(
        str(DATA_ROOT), ids, ids, batch_size=2, require_all_trials=True
    )
    clone_set = CNASimsDataset(str(DATA_ROOT), sim_ids=ids)
    assert set(test.dataset.sim_dirs) == {
        os.path.basename(item["sim_dir"]) for item in clone_set.items
    }


# --------------------------------------------------------------------------
# 7. Trial subsampling (matrix 3): training only, fresh each epoch, seeded
# --------------------------------------------------------------------------


@needs_data
def test_trial_subsample_shapes_the_item(small_ids):
    """K trials instead of 25, with a mask of matching length."""
    train, _, _ = small_ids
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train, trial_subsample=4)
    x, mask, y = ds[0]
    assert x.shape == (4, 100, 45)
    assert mask.shape == (4,)
    assert bool(mask.all())
    assert y.shape == (44,)
    # A chosen trial is a real one, so nothing in the item is NaN-padded.
    assert not bool(torch.isnan(x).any())


@needs_data
def test_trial_subsample_of_all_25_is_the_published_item(small_ids):
    """K >= num_trials has nothing to choose, so it must not change a thing."""
    train, _, _ = small_ids
    plain = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train)
    full = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train, trial_subsample=25)
    # Recorded as asked for, but inert -- the flag is not silently dropped.
    assert full.trial_subsample == 25
    assert full._subsample_k is None
    for i in range(len(plain)):
        xa, ma, ya = plain[i]
        xb, mb, yb = full[i]
        assert torch.equal(xa, xb) and torch.equal(ma, mb) and torch.equal(ya, yb)


@needs_data
def test_trial_subsample_of_25_leaves_the_encoder_output_bitwise_equal(small_ids):
    """The end the identity is for: the same context vector, bit for bit."""
    from cancer_sbi.config import get_preset
    from cancer_sbi.training.trainer import build_embedding_net

    train, _, _ = small_ids
    plain = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train)
    full = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train, trial_subsample=25)

    torch.manual_seed(1234)
    net = build_embedding_net(get_preset("clonemlp").encoder, "cpu")
    net.eval()
    with torch.no_grad():
        out_plain = net(plain[0][0].unsqueeze(0))
        out_full = net(full[0][0].unsqueeze(0))
    assert torch.equal(out_plain, out_full)


@needs_data
def test_trial_subsample_pooling_accepts_any_k(small_ids):
    """models/trials.py means over the T dimension, so K=4 is as good as 25."""
    from cancer_sbi.config import get_preset
    from cancer_sbi.training.trainer import build_embedding_net

    train, _, _ = small_ids
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train, trial_subsample=4)

    torch.manual_seed(1234)
    net = build_embedding_net(get_preset("clonemlp").encoder, "cpu")
    net.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = net(ds[0][0].unsqueeze(0))
    assert out.shape == (1, get_preset("clonemlp").encoder.trials_output_dim)
    assert bool(torch.isfinite(out).all())


@needs_data
def test_trial_subsample_is_reproducible_from_the_global_seed(small_ids):
    """Two seeded runs draw the same subsets; a different seed does not."""
    train, _, _ = small_ids
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train, trial_subsample=6)

    def draw(seed):
        torch.manual_seed(seed)
        return [ds._draw_trial_indices() for _ in range(6)]

    assert draw(20260924) == draw(20260924)
    assert draw(1) != draw(2)
    # A fresh draw per call, i.e. per epoch -- not a fixed function of the item.
    torch.manual_seed(7)
    repeated = [ds._draw_trial_indices() for _ in range(6)]
    assert len({tuple(d) for d in repeated}) > 1


@needs_data
def test_trial_subsample_indices_are_distinct_and_in_range(small_ids):
    train, _, _ = small_ids
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=train, trial_subsample=6)
    torch.manual_seed(3)
    for _ in range(20):
        picks = ds._draw_trial_indices()
        assert len(picks) == len(set(picks)) == 6
        assert picks == sorted(picks)
        assert all(0 <= p < 25 for p in picks)


@pytest.mark.parametrize("bad", [0, -1])
def test_trial_subsample_rejects_a_non_positive_k(bad):
    with pytest.raises(ValueError, match="trial_subsample"):
        CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=["sim1"], trial_subsample=bad)


@needs_data
@needs_cache
def test_trial_subsample_agrees_between_the_cached_and_uncached_paths(tiny_cache):
    """Same seed, same subset, same tensors -- whichever path served them."""
    cache_dir, cached, _ = tiny_cache
    sims = cached[:3]

    plain = CNASimsDataset(
        str(DATA_ROOT), top_k=100, sim_ids=sims, trial_subsample=5
    )
    fast = CNASimsDataset(
        str(DATA_ROOT),
        top_k=100,
        sim_ids=sims,
        cache_dir=str(cache_dir),
        trial_subsample=5,
    )
    for i in range(len(plain)):
        torch.manual_seed(20260924)
        xa, ma, ya = plain[i]
        torch.manual_seed(20260924)
        xb, mb, yb = fast[i]
        assert xa.shape == xb.shape == (5, 100, 45)
        assert torch.equal(xa, xb), i
        assert torch.equal(ma, mb) and torch.equal(ya, yb)


@needs_data
def test_clone_set_builder_subsamples_training_only(small_ids):
    """Validation and test are the published condition: all 25 trials."""
    train, val, test = small_ids
    train_loader, val_loader, test_loader = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, val_ids=val, batch_size=2, trial_subsample=16
    )
    assert train_loader.dataset.trial_subsample == 16
    assert val_loader.dataset.trial_subsample is None
    assert test_loader.dataset.trial_subsample is None

    torch.manual_seed(0)
    assert next(iter(train_loader))[0].shape[1] == 16
    assert next(iter(val_loader))[0].shape[1] == 25
    assert next(iter(test_loader))[0].shape[1] == 25


@needs_data
def test_clone_set_builder_defaults_to_no_subsampling(small_ids):
    """The published default: every loader serves all 25 trials."""
    train, val, test = small_ids
    loaders = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, val_ids=val, batch_size=2
    )
    for loader in loaders:
        assert loader.dataset.trial_subsample is None
        assert next(iter(loader))[0].shape[1] == 25


# --------------------------------------------------------------------------
# 6. min_trials: training on sims with fewer than 25 replicates (matrix 7)
# --------------------------------------------------------------------------


def _trials_on_disk(name):
    """Count a sim's present ``CNratios_all.pkl.gz`` files, straight from disk."""
    return sum(
        (DATA_ROOT / name / str(t) / "CNratios_all.pkl.gz").exists()
        for t in range(1, 26)
    )


def _has_params(name):
    """Whether ``parameters.pkl`` is readable, the other half of the filter."""
    from cancer_sbi.data.clone_sets import load_pickle

    try:
        load_pickle(str(DATA_ROOT / name / "parameters.pkl"))
    except Exception:  # noqa: BLE001 - mirrors CNASimsDataset.drop_missing
        return False
    return True


def _sim_names():
    """Every ``sim*`` directory name under the local tree, numerically sorted."""
    return sorted(
        (d for d in os.listdir(DATA_ROOT) if d.startswith("sim")),
        key=lambda n: int(n[3:]),
    )


def _kept(ds):
    """The sim names a dataset kept."""
    return {os.path.basename(item["sim_dir"]) for item in ds.items}


@needs_data
def test_min_trials_none_is_the_published_sim_set():
    """The default must still be trap 10: exactly ``complete_sim_ids``."""
    ids = _sim_names()[:120]
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=ids)
    assert ds.min_trials is None
    assert _kept(ds) == set(complete_sim_ids(str(DATA_ROOT), ids))


@needs_data
def test_min_trials_adds_exactly_the_sims_with_enough_files():
    """The kept set is counted independently, off the filesystem."""
    ids = _sim_names()[:120]
    k = 5
    relaxed = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=ids, min_trials=k)
    assert relaxed.min_trials == k

    # Independent count: enough trial files AND a readable parameters.pkl, which
    # is the other half of the dataset's own filter.
    expected = {
        name for name in ids if _trials_on_disk(name) >= k and _has_params(name)
    }
    assert _kept(relaxed) == expected
    assert _kept(relaxed) >= set(complete_sim_ids(str(DATA_ROOT), ids))
    # And it really did add something the published rule drops.
    assert any(_trials_on_disk(name) < 25 for name in expected)


@needs_data
def test_min_trials_item_is_nan_padded_with_a_matching_mask():
    """A partial sim's real trials come first; the rest are NaN and masked out."""
    ids = _sim_names()[:200]
    partial = [n for n in ids if 5 <= _trials_on_disk(n) < 25 and _has_params(n)]
    if not partial:
        pytest.skip("no partial sim in the local tree")
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=partial[:2], min_trials=5)

    for i, item in enumerate(ds.items):
        n = _trials_on_disk(os.path.basename(item["sim_dir"]))
        x, mask, _ = ds[i]
        assert x.shape == (25, 100, 45)
        assert mask.shape == (25,)
        assert int(mask.sum()) == n
        assert bool(mask[:n].all()) and not bool(mask[n:].any())
        assert not torch.isnan(x[:n]).any()
        assert bool(torch.isnan(x[n:]).all())


@needs_data
@pytest.mark.parametrize("bad", [0, -1, 26])
def test_min_trials_rejects_a_bar_outside_the_trial_count(bad):
    with pytest.raises(ValueError, match="min_trials"):
        CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=["sim1"], min_trials=bad)


@needs_data
def test_min_trials_reaches_train_and_val_but_never_test():
    """The published test set is the whole point: it must not move."""
    ids = _sim_names()[:200]
    partial = [n for n in ids if 5 <= _trials_on_disk(n) < 25 and _has_params(n)]
    if len(partial) < 2:
        pytest.skip("fewer than two partial sims in the local tree")
    complete = complete_sim_ids(str(DATA_ROOT), ids)
    # Every partition is offered both kinds, so "test did not move" is a real
    # assertion rather than a partition with nothing to drop.
    train = complete[:6] + partial[:1]
    val = complete[6:10] + partial[1:2]
    test = complete[10:16] + partial[:1]

    train_loader, val_loader, test_loader = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, val_ids=val, batch_size=2, min_trials=5
    )
    assert train_loader.dataset.min_trials == 5
    assert val_loader.dataset.min_trials == 5
    assert test_loader.dataset.min_trials is None
    assert _kept(train_loader.dataset) == set(train)
    assert _kept(val_loader.dataset) == set(val)
    # The test dataset keeps the published rule, so the partial sim is gone.
    assert _kept(test_loader.dataset) == set(complete_sim_ids(str(DATA_ROOT), test))

    default = build_clone_set_dataloaders(
        str(DATA_ROOT), train, test, val_ids=val, batch_size=2
    )
    for loader in default:
        assert loader.dataset.min_trials is None


@needs_data
@pytest.mark.parametrize("model", ["armtoken", "cloneatt", "clonemlp"])
def test_every_encoder_gives_a_finite_context_on_a_partial_item(model):
    """NaN slots must be masked, not propagated, by all three encoders."""
    from cancer_sbi.config import get_preset
    from cancer_sbi.training.trainer import build_embedding_net

    ids = _sim_names()[:200]
    partial = [n for n in ids if 5 <= _trials_on_disk(n) < 25 and _has_params(n)]
    if not partial:
        pytest.skip("no partial sim in the local tree")
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=partial[:1], min_trials=5)
    x, _, _ = ds[0]

    net = build_embedding_net(get_preset(model).encoder, "cpu").eval()
    with torch.no_grad():
        out = net(x.unsqueeze(0))
    assert torch.isfinite(out).all(), model


@needs_data
def test_armtoken_ignores_the_padded_slots_exactly():
    """20 real trials + 5 NaN slots must embed to the same thing as the 20."""
    from cancer_sbi.config import get_preset
    from cancer_sbi.training.trainer import build_embedding_net

    ids = _sim_names()[:120]
    complete = complete_sim_ids(str(DATA_ROOT), ids)
    if not complete:
        pytest.skip("no complete sim in the local tree")
    ds = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=complete[:1])
    x, _, _ = ds[0]

    twenty = x[:20]
    padded = x.clone()
    padded[20:] = float("nan")

    net = build_embedding_net(get_preset("armtoken").encoder, "cpu").eval()
    with torch.no_grad():
        a = net(twenty.unsqueeze(0))
        b = net(padded.unsqueeze(0))
    assert torch.allclose(a, b, atol=0, rtol=0), (a - b).abs().max()


# --- the partial cache -----------------------------------------------------


@pytest.fixture(scope="module")
def partial_cache(tmp_path_factory):
    """Build a --min-trials 2 cache over a few partial sims and a few complete."""
    if not DATA_ROOT.is_dir():
        pytest.skip("no local simulation tree")
    ids = _sim_names()[:200]
    partial = [n for n in ids if 2 <= _trials_on_disk(n) < 25 and _has_params(n)]
    complete = complete_sim_ids(str(DATA_ROOT), ids)
    if not partial or len(complete) < 4:
        pytest.skip("local tree has no partial sims to cache")
    chosen = complete[:4] + partial[:2]

    work = tmp_path_factory.mktemp("partial_clone_cache")
    split_path = work / "split.pkl"
    save_split(split_path, np.array(chosen[:4]), np.array(chosen[4:]))
    out = work / "cache"
    proc = subprocess.run(
        [
            sys.executable,
            str(SRC / "utilities" / "build_clone_cache.py"),
            "--root", str(DATA_ROOT),
            "--split", str(split_path),
            "--out", str(out),
            "--workers", "2",
            "--top-k", "100",
            "--min-trials", "2",
            "--force",
        ],
        cwd=str(SRC), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out, chosen, split_path


@needs_data
def test_partial_cache_manifest_and_trial_counts(partial_cache):
    cache_dir, chosen, _ = partial_cache
    manifest = json.loads((cache_dir / CACHE_MANIFEST_FILENAME).read_text())
    assert manifest["min_trials"] == 2

    names = [str(n) for n in np.load(cache_dir / "sim_ids.npy")]
    counts = np.load(cache_dir / "trial_counts.npy")
    assert set(names) == set(chosen)
    assert len(counts) == len(names)
    for name, count in zip(names, counts):
        assert int(count) == _trials_on_disk(name), name


@needs_data
def test_partial_cache_item_equals_the_uncached_item(partial_cache):
    """Values, mask and NaN pattern, all three."""
    cache_dir, chosen, _ = partial_cache
    plain = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=chosen, min_trials=2)
    fast = CNASimsDataset(
        str(DATA_ROOT), top_k=100, sim_ids=chosen, min_trials=2, cache_dir=str(cache_dir)
    )
    assert len(plain) == len(fast) == len(chosen)
    for i in range(len(plain)):
        xa, ma, ya = plain[i]
        xb, mb, yb = fast[i]
        assert torch.equal(torch.isnan(xa), torch.isnan(xb))
        assert torch.equal(xa[~torch.isnan(xa)], xb[~torch.isnan(xb)])
        assert torch.equal(ma, mb)
        assert torch.equal(ya, yb)


@needs_data
def test_a_complete_only_cache_refuses_a_relaxed_dataset(tiny_cache, partial_cache):
    """It simply does not hold the partial sims, and the message names them."""
    cache_dir, cached, _ = tiny_cache
    _, chosen, _ = partial_cache
    partial = [n for n in chosen if _trials_on_disk(n) < 25]
    with pytest.raises(ValueError) as excinfo:
        CNASimsDataset(
            str(DATA_ROOT),
            top_k=100,
            sim_ids=cached[:2] + partial[:1],
            min_trials=2,
            cache_dir=str(cache_dir),
        )
    message = str(excinfo.value)
    assert partial[0] in message
    assert "min_trials=25" in message


@needs_data
def test_verify_clone_cache_passes_on_a_partial_cache(partial_cache):
    cache_dir, _, split_path = partial_cache
    proc = subprocess.run(
        [
            sys.executable,
            str(SRC / "utilities" / "verify_clone_cache.py"),
            "--root", str(DATA_ROOT),
            "--split", str(split_path),
            "--cache", str(cache_dir),
            "--n-sims", "6",
            "--n-trials", "2",
        ],
        cwd=str(SRC), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PARTIAL cache: min_trials=2" in proc.stdout


@needs_data
def test_the_complete_only_cache_path_is_unchanged(tiny_cache):
    """min_trials=None + the published cache: still all-True masks, same values."""
    cache_dir, cached, _ = tiny_cache
    assert not (cache_dir / "trial_counts.npy").exists()
    manifest = json.loads((cache_dir / CACHE_MANIFEST_FILENAME).read_text())
    assert "min_trials" not in manifest

    plain = CNASimsDataset(str(DATA_ROOT), top_k=100, sim_ids=cached[:3])
    fast = CNASimsDataset(
        str(DATA_ROOT), top_k=100, sim_ids=cached[:3], cache_dir=str(cache_dir)
    )
    for i in range(len(plain)):
        xa, ma, ya = plain[i]
        xb, mb, yb = fast[i]
        assert torch.equal(xa, xb) and torch.equal(ma, mb) and torch.equal(ya, yb)
        assert bool(mb.all())
