"""Integration seams between the four repair packages (2026-09-24).

Run from ``src/``::

    python -m pytest tests/test_integration_seams.py -q

Four seams, one section each:

1. ``cli/evaluate.py`` and ``evaluation/sample_posteriors.py`` must speak the
   NEW split/loader contract: ``load_split`` returns a dict and both loader
   builders return ``(train, val_or_None, test)``. Evaluation keeps scoring the
   TEST ids.
2. ``--cache-dir`` must reach the loaders: CLI flag -> ``DataConfig.cache_dir``
   -> ``build_clone_set_dataloaders(cache_dir=...)``, and the cached batches
   must be bit-identical to the uncached ones.
3. The post-construction re-seed in ``cli/train.py`` must not clobber the RNG
   state a resume just restored.
4. Reading a cached item must not emit the "NumPy array is not writable"
   UserWarning.

Tests needing the local simulation tree or the built clone cache are skipped,
not failed, when those are absent.
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cancer_sbi.cli.train import build_config, build_parser  # noqa: E402
from cancer_sbi.config import OptimConfig, TrainConfig  # noqa: E402
from cancer_sbi.data.clone_sets import CACHE_MANIFEST_FILENAME  # noqa: E402
from cancer_sbi.data.loaders import build_clone_set_dataloaders  # noqa: E402
from cancer_sbi.training import checkpoints  # noqa: E402
from cancer_sbi.training.trainer import Trainer, seed_everything  # noqa: E402

DATA_ROOT = SRC.parent / "data" / "Guassian_Normal" / "simulation_outputs"
SPLIT_PATH = SRC.parent / "data" / "train_test_split.pkl"
CACHE_DIR = SRC.parent / "data" / "cache" / "clone_top100_v1"
CACHE_IDS = CACHE_DIR / "sim_ids.npy"

needs_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="local simulation tree not present"
)
needs_cache = pytest.mark.skipif(
    not (CACHE_DIR / CACHE_MANIFEST_FILENAME).exists(),
    reason="clone cache not built; run utilities/build_clone_cache.py",
)


def _cached_sim_ids(n):
    """The first ``n`` sim ids that the built cache holds."""
    return [str(s) for s in np.load(CACHE_IDS, allow_pickle=True)[:n]]


def _tiny_two_key_split(tmp_path, train_ids, test_ids):
    """Write a 2-key (no ``val_ids``) split pickle, as the old files are.

    Returns:
        Path to the pickle.
    """
    path = tmp_path / "tiny_split.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {"train_ids": np.array(train_ids), "test_ids": np.array(test_ids)}, handle
        )
    return path


# ---------------------------------------------------------------------------
# Seam 1: the two evaluation callers and the new split/loader contract.
# ---------------------------------------------------------------------------


class _StopAtModelBuild(Exception):
    """Raised in place of ``build_training_components`` to end the run early.

    Sampling needs a trained checkpoint, which does not exist locally, so the
    tests below exercise everything up to the model build -- which is exactly
    where the broken split/loader unpacking lived -- and stop there, carrying
    the loaders out for inspection.
    """

    def __init__(self, train_loader):
        super().__init__("stopped at model build")
        self.train_loader = train_loader


def _stub_checkpoint(tmp_path, model="clonemlp", **flow_or_encoder):
    """Write a checkpoint carrying only an ``effective_config``.

    Enough for :func:`cancer_sbi.evaluation.posterior.resolve_eval_config`,
    which reads that key and nothing else.

    Returns:
        Path to the ``.pt``.
    """
    from cancer_sbi.config import config_to_dict, get_preset

    effective = config_to_dict(get_preset(model))
    for block, values in flow_or_encoder.items():
        effective[block].update(values)
    path = tmp_path / f"stub_{model}.pt"
    torch.save({"effective_config": effective}, path)
    return path


def _patch_model_build(monkeypatch, holder):
    """Make ``build_training_components`` raise, recording the train loader."""
    import cancer_sbi.training.trainer as trainer_mod

    def _boom(cfg, train_loader, **kwargs):
        holder["train_loader"] = train_loader
        raise _StopAtModelBuild(train_loader)

    monkeypatch.setattr(trainer_mod, "build_training_components", _boom)


@needs_data
@needs_cache
def test_sample_posteriors_uses_dict_split_and_three_tuple_loaders(
    tmp_path, monkeypatch, capsys
):
    """``sample_posteriors.main`` must survive the dict split and 3-tuple loaders.

    FAILS on the old ``train_ids, test_ids = load_split(...)`` with
    "too many values to unpack" (the dict has 2 keys here but the tuple unpack
    of the 3-tuple loader result raises regardless). PASSES once both use the
    new contract, and the held-out set still comes from ``test_ids``.
    """
    import cancer_sbi.evaluation.sample_posteriors as sp

    ids = _cached_sim_ids(4)
    train_ids, test_ids = ids[:2], ids[2:]
    split_path = _tiny_two_key_split(tmp_path, train_ids, test_ids)
    # A checkpoint has to exist and be readable: since 2026-09-24 the
    # architecture is resolved from it *before* the model is built, so the run
    # stops at a missing file rather than at the patched build below.
    ckpt_path = _stub_checkpoint(tmp_path)

    holder = {}
    _patch_model_build(monkeypatch, holder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample_posteriors",
            "--model", "clonemlp",
            "--data-root", str(DATA_ROOT),
            "--split", str(split_path),
            "--ckpt", str(ckpt_path),
            "--out-dir", str(tmp_path / "out"),
            "--limit", "2",
            "--device", "cpu",
        ],
    )

    with pytest.raises(_StopAtModelBuild):
        sp.main()

    out = capsys.readouterr().out
    assert "split: 2 train ids / 2 test ids" in out
    # The held-out set is the TEST ids, not a validation slice.
    assert "held-out cases: 2" in out
    assert len(holder["train_loader"].dataset) == 2


@needs_data
@needs_cache
def test_evaluate_cli_uses_dict_split_and_three_tuple_loaders(
    tmp_path, monkeypatch, capsys
):
    """Same seam, in ``cli/evaluate.py``. Same failure before the fix."""
    from cancer_sbi.cli import evaluate as ev

    ids = _cached_sim_ids(4)
    train_ids, test_ids = ids[:2], ids[2:]
    split_path = _tiny_two_key_split(tmp_path, train_ids, test_ids)

    holder = {}
    _patch_model_build(monkeypatch, holder)

    with pytest.raises(_StopAtModelBuild):
        ev.main(
            [
                "--model", "clonemlp",
                "--data-root", str(DATA_ROOT),
                "--split", str(split_path),
                "--run-dir", str(tmp_path / "run"),
                "--out-dir", str(tmp_path / "out"),
                "--limit-cases", "2",
                "--device", "cpu",
            ]
        )

    out = capsys.readouterr().out
    # n_cases comes from test_ids: 2 sims x 1 case each.
    assert "x_all shape: (2," in out
    assert len(holder["train_loader"].dataset) == 2


# ---------------------------------------------------------------------------
# Seam 2: --cache-dir has to reach the loaders.
# ---------------------------------------------------------------------------


def test_build_config_carries_cache_dir():
    """``--cache-dir X`` must land on ``cfg.data.cache_dir``."""
    from cancer_sbi.config import PRESETS

    preset = PRESETS["clonemlp"]
    args = build_parser().parse_args(["--model", "clonemlp", "--cache-dir", "X"])
    cfg = build_config(args, preset, Path("/d"), Path("/s.pkl"), Path("/c"))
    assert cfg.data.cache_dir == "X"

    # Omitting it keeps the published behaviour: no cache.
    args = build_parser().parse_args(["--model", "clonemlp"])
    cfg = build_config(args, preset, Path("/d"), Path("/s.pkl"), Path("/c"))
    assert cfg.data.cache_dir is None


@needs_data
@needs_cache
def test_cli_code_path_with_cache_dir_matches_uncached_batches(tmp_path):
    """End to end: the CLI's own config -> loader call, cached vs uncached.

    Builds the loaders exactly as ``cli/train.py`` does -- ``build_config`` with
    ``--cache-dir``, then ``build_clone_set_dataloaders(..., cache_dir=...)`` --
    and compares every batch with the uncached path.
    """
    from cancer_sbi.config import PRESETS

    ids = _cached_sim_ids(3)

    args = build_parser().parse_args(
        ["--model", "clonemlp", "--cache-dir", str(CACHE_DIR), "--batch-size", "2"]
    )
    cfg = build_config(args, PRESETS["clonemlp"], DATA_ROOT, tmp_path / "s.pkl", tmp_path)
    assert cfg.data.cache_dir == str(CACHE_DIR)

    def _loaders(cache_dir):
        return build_clone_set_dataloaders(
            root_dir=str(DATA_ROOT),
            train_ids=ids,
            val_ids=None,
            test_ids=ids,
            top_k=cfg.data.top_k,
            batch_size=cfg.data.batch_size,
            pin_memory=cfg.data.pin_memory,
            num_workers=cfg.data.num_workers,
            cache_dir=cache_dir,
        )

    _, cached_val, cached_test = _loaders(cfg.data.cache_dir)
    _, plain_val, plain_test = _loaders(None)
    assert cached_val is None and plain_val is None  # no val_ids -> no val loader

    cached_batches = list(cached_test)
    plain_batches = list(plain_test)
    assert len(cached_batches) == len(plain_batches) > 0
    for (xc, mc, yc), (xp, mp, yp) in zip(cached_batches, plain_batches):
        assert torch.equal(xc, xp)
        assert torch.equal(mc, mp)
        assert torch.equal(yc, yp)


# ---------------------------------------------------------------------------
# Seam 3: the post-construction re-seed must not fight the resume restore.
# ---------------------------------------------------------------------------

SEED = 20260924


def _tiny_trainer(ckpt_dir):
    """A Trainer around a 3->2 Linear with no data, pointed at ``ckpt_dir``."""
    return Trainer(
        density_estimator=nn.Linear(3, 2),
        train_loader=[],
        val_loader=[],
        optim_cfg=OptimConfig(
            use_param_groups=False,
            learning_rate=1e-3,
            learning_rate_is_used=True,
            grad_clip=None,
        ),
        train_cfg=TrainConfig(max_epochs=0),
        dataset="clone_sets",
        device="cpu",
        ckpt_dir=ckpt_dir,
    )


def _cli_reseed(trainer, seed):
    """The exact branch ``cli/train.py`` runs after constructing the Trainer."""
    if not trainer.resumed:
        seed_everything(seed)


def test_fresh_run_reseeds_after_construction(tmp_path):
    """No checkpoint -> ``resumed`` is False and the re-seed happens."""
    torch.manual_seed(SEED)
    expected = torch.get_rng_state().clone()

    torch.manual_seed(SEED + 1)  # some other state going in
    trainer = _tiny_trainer(tmp_path / "ckpt")
    assert trainer.resumed is False

    _cli_reseed(trainer, SEED)
    assert torch.equal(torch.get_rng_state(), expected)


def test_resumed_run_keeps_the_checkpoints_rng_state(tmp_path):
    """A checkpoint is present -> its saved RNG state survives construction.

    FAILS on the unconditional ``seed_everything(cfg.train.seed)``: the state
    after the re-seed was the seed's, so a resumed run replayed the stream.
    """
    ckpt_dir = tmp_path / "ckpt"

    # Write latest.pt carrying a distinctive RNG state. The seed is set AFTER the
    # stand-in is built, because building it draws from the same generator.
    writer = _tiny_trainer(ckpt_dir)
    torch.manual_seed(999_111)
    saved_rng = torch.get_rng_state().clone()
    payload = checkpoints.build_checkpoint(
        epoch=3,
        density_estimator=writer.density_estimator,
        optimizer=writer.optimizer,
        best_val_loss=0.5,
        best_model_state_dict=None,
        history={"training_loss": [], "validation_loss": []},
        epochs_since_last_improvement=0,
    )
    assert torch.equal(payload["rng_state"], saved_rng)
    checkpoints.save_checkpoint(ckpt_dir, 3, payload)

    # Now construct over it from a different state entirely.
    torch.manual_seed(SEED + 7)
    trainer = _tiny_trainer(ckpt_dir)
    assert trainer.resumed is True
    assert trainer.epoch == 3

    _cli_reseed(trainer, SEED)
    after = torch.get_rng_state().clone()

    # The state is the checkpoint's, not the seed's.
    assert torch.equal(after, saved_rng)
    torch.manual_seed(SEED)
    assert not torch.equal(after, torch.get_rng_state())


# ---------------------------------------------------------------------------
# Seam 4: no "not writable" warning when reading a cached item.
# ---------------------------------------------------------------------------


@needs_data
@needs_cache
def test_cached_item_is_writable_and_warning_free():
    """The mmap slice used to produce a read-only tensor and a UserWarning."""
    from cancer_sbi.data.clone_sets import CNASimsDataset

    dataset = CNASimsDataset(
        str(DATA_ROOT), top_k=100, sim_ids=_cached_sim_ids(2), cache_dir=str(CACHE_DIR)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x_trials, trial_mask, y = dataset[0]
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "not writable" in m], messages

    # Checked BEFORE the write below, and by address rather than by flags.
    # Writing into a tensor that wraps a read-only mmap does not raise in
    # Python: it walks off into the mapping and kills the interpreter with a
    # bus error, which reports as a crashed run rather than as a failing test.
    # torch does not propagate the array's read-only flag either
    # (``from_numpy(ro).numpy().flags.writeable`` is True), so the flags say
    # nothing -- what distinguishes the copy from the view is whether the
    # tensor's storage lies inside the memmap.
    for name, tensor, mm in (
        ("x_trials", x_trials, dataset._cache_x),
        ("y", y, dataset._cache_theta),
    ):
        start = mm.ctypes.data
        assert not (
            start <= tensor.data_ptr() < start + mm.nbytes
        ), f"{name} aliases the read-only mmap instead of being a copy"

    # And then it really is writable.
    x_trials[0, 0, 0] = 1.0
    y[0] = 1.0
    assert trial_mask.all()
