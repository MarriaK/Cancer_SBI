"""Work package A: the argument-to-config path, seeding, and two trainer repairs.

Run from ``src/``::

    python -m pytest tests/test_wp_a_train_cli.py -q

Why these tests and not preset-level ones: both bugs repaired here
(``cli/train.py:226``'s unconditional ``seed=args.seed`` and ``:229``'s
``replace(preset, data=..., train=...)`` that never replaced ``flow``) lived in
the mapping from *flags* to config, not in the presets. A test that imported
``get_preset`` and asserted on it would have passed while every run ignored the
flags -- see docs/CODEBASE_IMPROVEMENT_PLAN.md, "Testing", item 1.

Two of these are regression tests in the strict sense -- they fail on the code
as it stood before this work package:

* :func:`test_snapshot_best_state_does_not_alias_live_weights` fails on the old
  ``{k: v.detach().cpu() ...}``, which shares storage with the live parameters
  on a CPU-only run.
* :func:`test_reload_best_always_with_no_best_snapshot_does_not_raise` fails on
  the old unguarded ``load_state_dict(self.best_model_state_dict)``.
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from cancer_sbi.cli.train import build_config, build_parser
from cancer_sbi.config import OptimConfig, TrainConfig, get_preset
from cancer_sbi.training.trainer import Trainer, build_embedding_net, seed_everything

# tests/ -> src/ -> code/
CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = CODE_ROOT / "data" / "Guassian_Normal" / "simulation_outputs"


def config_from_argv(argv, model="clonemlp"):
    """Parse ``argv`` and fold it into ``model``'s preset, as ``main`` does."""
    args = build_parser().parse_args(["--model", model] + list(argv))
    preset = get_preset(model)
    return build_config(
        args,
        preset,
        data_root=Path("/nonexistent/data"),
        split_path=Path("/nonexistent/split.pkl"),
        ckpt_dir=Path("/nonexistent/ckpt"),
    )


# ---------------------------------------------------------------------------
# 1. Absence of every new flag leaves the published values in place.
# ---------------------------------------------------------------------------


def test_new_flags_absent_keeps_published_defaults():
    cfg = config_from_argv([])
    assert cfg.flow.z_score_x == "none"
    assert cfg.encoder.input_space == "log2"
    assert cfg.encoder.freq_renorm is False
    assert cfg.data.num_workers == 0
    # The published runs were unseeded and the preset says so.
    assert cfg.train.seed is None


def test_new_flags_absent_keeps_published_defaults_for_every_preset():
    # dominantclone is the one preset whose flow already whitens (trap 1); the
    # flags must not flatten that to a single tree-wide value.
    expected = {"clonemlp": "none", "cloneatt": "none", "dominantclone": "structured"}
    for model, z in expected.items():
        cfg = config_from_argv([], model=model)
        assert cfg.flow.z_score_x == z, model
        assert cfg.encoder.input_space == "log2", model
        assert cfg.encoder.freq_renorm is False, model
        assert cfg.data.num_workers == 0, model


# ---------------------------------------------------------------------------
# 2. Each new flag is accepted and reaches the config.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "structured", "independent"])
def test_z_score_x_flag_reaches_the_flow_block(mode):
    cfg = config_from_argv(["--z-score-x", mode])
    assert cfg.flow.z_score_x == mode
    # The rest of the flow block is untouched -- this is a one-value change.
    assert cfg.flow.z_score_y == get_preset("clonemlp").flow.z_score_y
    assert cfg.flow.hidden_features == 50


def test_z_score_x_flag_is_the_regression_for_line_229():
    """``replace(preset, data=..., train=...)`` never replaced ``flow``.

    Before this work package the assertion below was false for every value of
    the flag, because the flag did not exist and the ``flow`` block was carried
    through unchanged. R0-vs-R1 turns on exactly this.
    """
    assert get_preset("clonemlp").flow.z_score_x == "none"
    assert config_from_argv(["--z-score-x", "structured"]).flow.z_score_x == "structured"


@pytest.mark.parametrize("space", ["log2", "copy"])
def test_input_space_flag_reaches_the_encoder(space):
    assert config_from_argv(["--input-space", space]).encoder.input_space == space


def test_freq_renorm_flag_reaches_the_encoder():
    cfg = config_from_argv(["--freq-renorm"], model="cloneatt")
    assert cfg.encoder.freq_renorm is True


@pytest.mark.parametrize("n", [0, 4, 8])
def test_num_workers_flag_reaches_the_data_config(n):
    assert config_from_argv(["--num-workers", str(n)]).data.num_workers == n


def test_deterministic_flag_parses_and_is_off_by_default():
    parser = build_parser()
    assert parser.parse_args(["--model", "clonemlp"]).deterministic is False
    assert parser.parse_args(["--model", "clonemlp", "--deterministic"]).deterministic is True


# ---------------------------------------------------------------------------
# 3. Seed plumbing (cli/train.py:226).
# ---------------------------------------------------------------------------


def test_seed_falls_back_to_the_preset_when_the_flag_is_absent():
    """The old line was the bare ``seed=args.seed``, which clobbered the preset.

    A preset carrying a seed is constructed with ``replace`` rather than taken
    off the shelf, because none of the three published presets sets one -- that
    is the point of trap 21.
    """
    preset = get_preset("clonemlp")
    seeded = replace(preset, train=replace(preset.train, seed=4242))
    args = build_parser().parse_args(["--model", "clonemlp"])
    cfg = build_config(
        args, seeded, Path("/d"), Path("/s.pkl"), Path("/c")
    )
    assert cfg.train.seed == 4242


def test_seed_flag_wins_over_the_preset():
    preset = get_preset("clonemlp")
    seeded = replace(preset, train=replace(preset.train, seed=4242))
    args = build_parser().parse_args(["--model", "clonemlp", "--seed", "7"])
    cfg = build_config(args, seeded, Path("/d"), Path("/s.pkl"), Path("/c"))
    assert cfg.train.seed == 7
    # And on an unseeded preset, which is what the real presets are.
    assert config_from_argv(["--seed", "7"]).train.seed == 7


def test_seed_everything_seeds_python_random():
    """``seed_everything`` left ``random`` untouched; now it does not."""
    import random

    seed_everything(123)
    first = [random.random() for _ in range(3)]
    seed_everything(123)
    assert [random.random() for _ in range(3)] == first

    # seed=None must still leave every RNG alone (trap 21 -- the published runs).
    random.seed(999)
    expected = random.random()
    random.seed(999)
    seed_everything(None)
    assert random.random() == expected


# ---------------------------------------------------------------------------
# 4. Trainer repair: the best-state snapshot must be a real copy on CPU.
# ---------------------------------------------------------------------------


def _trainer_with(module, reload_best):
    """A Trainer around ``module`` with no data, no resume and no optimiser tricks.

    ``dominantclone``'s OptimConfig is used because it is the single-group one,
    so no embedding net has to be invented to split parameters by identity.
    """
    return Trainer(
        density_estimator=module,
        train_loader=[],
        val_loader=[],
        optim_cfg=OptimConfig(
            use_param_groups=False,
            learning_rate=1e-3,
            learning_rate_is_used=True,
            grad_clip=None,
        ),
        train_cfg=TrainConfig(max_epochs=0, reload_best=reload_best),
        dataset="clone_sets",
        device="cpu",
        ckpt_dir=None,
    )


def test_snapshot_best_state_does_not_alias_live_weights(tmp_path, monkeypatch):
    """The CPU path returned tensors sharing storage with the live parameters.

    FAILS on the pre-WP-A ``{k: v.detach().cpu() for ...}``: mutating the live
    weight below also changed the "snapshot", so ``load_state_dict(best)``
    restored the last epoch. PASSES with ``.detach().cpu().clone()``.
    """
    monkeypatch.chdir(tmp_path)
    module = nn.Linear(3, 2)
    trainer = _trainer_with(module, reload_best="never")
    assert trainer.quirks.best_state_on_device is False, "this must exercise the CPU path"

    snapshot = trainer._snapshot_best_state()
    before = snapshot["weight"].clone()

    with torch.no_grad():
        module.weight.add_(1.0)

    assert torch.equal(snapshot["weight"], before), "snapshot moved with the live weight"
    assert not torch.equal(snapshot["weight"], module.weight.detach())
    assert snapshot["weight"].data_ptr() != module.weight.data_ptr()


def test_snapshot_on_device_path_is_still_a_deepcopy(tmp_path, monkeypatch):
    """CloneAtt's quirk (deepcopy on the model's device) is unchanged."""
    monkeypatch.chdir(tmp_path)
    module = nn.Linear(3, 2)
    trainer = _trainer_with(module, reload_best="on_early_stop")
    assert trainer.quirks.best_state_on_device is True

    snapshot = trainer._snapshot_best_state()
    before = snapshot["weight"].clone()
    with torch.no_grad():
        module.weight.add_(1.0)
    assert torch.equal(snapshot["weight"], before)


# ---------------------------------------------------------------------------
# 5. Trainer repair: reload_best="always" with nothing to reload.
# ---------------------------------------------------------------------------


def test_reload_best_always_with_no_best_snapshot_does_not_raise(tmp_path, monkeypatch, capsys):
    """A run that never improved has no snapshot; the old code raised on it.

    ``max_epochs=0`` reproduces that state without any data: the epoch loop
    never runs, ``best_model_state_dict`` stays ``None``, and control falls
    straight into the ``reload_best == "always"`` tail.

    FAILS on the pre-WP-A unguarded
    ``self.density_estimator.load_state_dict(self.best_model_state_dict)``
    (TypeError / AttributeError on ``None``). PASSES with the guard.
    """
    monkeypatch.chdir(tmp_path)
    module = nn.Linear(3, 2)
    trainer = _trainer_with(module, reload_best="always")
    assert trainer.best_model_state_dict is None

    returned = trainer.train()

    assert returned is module
    assert "no best snapshot" in capsys.readouterr().out


def test_reload_best_always_still_restores_when_a_snapshot_exists(tmp_path, monkeypatch):
    """The guard must not turn the restore itself into a no-op."""
    monkeypatch.chdir(tmp_path)
    module = nn.Linear(3, 2)
    trainer = _trainer_with(module, reload_best="always")
    trainer.best_model_state_dict = trainer._snapshot_best_state()
    best_weight = trainer.best_model_state_dict["weight"].clone()

    with torch.no_grad():
        module.weight.add_(5.0)
    trainer.train()

    assert torch.equal(module.weight.detach(), best_weight)


# ---------------------------------------------------------------------------
# 6. End-to-end smoke: the clonemlp encoder, built through build_embedding_net,
#    on a real batch from the local 888-sim subsample.
# ---------------------------------------------------------------------------


def _wp_c_kwargs_present():
    """Is WP-C's constructor contract merged yet?"""
    import inspect

    from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding
    from cancer_sbi.models.set_transformer import CloneSetEmbedding

    mlp = inspect.signature(BaselineCloneEmbedding.__init__).parameters
    att = inspect.signature(CloneSetEmbedding.__init__).parameters
    missing = []
    if "input_space" not in mlp:
        missing.append("BaselineCloneEmbedding(input_space=...)")
    if "freq_renorm" not in att:
        missing.append("CloneSetEmbedding(freq_renorm=...)")
    return missing


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"local subsample missing at {DATA_ROOT}")
def test_clonemlp_embedding_net_forward_on_a_real_batch():
    """Published defaults through ``build_embedding_net``; output must be finite."""
    missing = _wp_c_kwargs_present()
    if missing:
        pytest.skip(
            "WP-C not merged yet: build_embedding_net passes "
            + ", ".join(missing)
            + ", which the current constructors do not accept."
        )

    from cancer_sbi.data.clone_sets import CNASimsDataset

    preset = get_preset("clonemlp")
    assert preset.encoder.input_space == "log2"
    assert preset.encoder.freq_renorm is False

    sim_ids = sorted(p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim"))[:2]
    dataset = CNASimsDataset(str(DATA_ROOT), top_k=preset.data.top_k, sim_ids=sim_ids)
    if len(dataset) < 2:
        pytest.skip("fewer than 2 usable sims among the two sampled (trap 10 filtering)")

    x0, _, _ = dataset[0]
    x1, _, _ = dataset[1]
    batch = torch.stack([x0, x1])                      # (2, 25, 100, 45)
    assert batch.shape[0] == 2 and batch.shape[-1] == 45

    net = build_embedding_net(preset.encoder, device="cpu")
    net.eval()
    with torch.no_grad():
        out = net(batch)

    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. cli/train.py main(): which loader reaches the Trainer as `val_loader`.
#
# This is the repair of MODEL_IMPROVEMENT_PLAN.md §5 step 1 -- `*/main.py:33`
# passed the TEST loader as the validation loader, so best.pt was selected on
# the sims the score is reported on. The fix lives in main(), between the
# builder call and the Trainer construction, and nothing above this section
# reaches it: build_config never sees a split file, and the loader tests never
# see main(). Two mutants it must kill:
#   * `if val_loader is None:` -> `if True:`   (always fall back to test)
#   * dropping `val_ids=` from the builder call (no validation loader is built)
# ---------------------------------------------------------------------------

SPLIT_SIM_COUNT = 6


class _StopAtTrainer(Exception):
    """Raised in place of ``Trainer(...)``, carrying the keyword arguments."""

    def __init__(self, kwargs):
        super().__init__("stopped at Trainer construction")
        self.kwargs = kwargs


def _sim_names(n):
    """The first ``n`` sim directories on disk, in numeric order."""
    names = sorted(
        (p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim")),
        key=lambda name: int(name[3:]),
    )
    return names[:n]


def _write_split(path, **id_lists):
    """Write a split pickle from ``{key: [sim names]}``."""
    import pickle

    import numpy as np

    with Path(path).open("wb") as handle:
        pickle.dump({k: np.array(v) for k, v in id_lists.items()}, handle)
    return path


def _loader_sim_names(loader):
    """The sim names a clone-set loader's dataset will serve."""
    import os

    return {os.path.basename(item["sim_dir"]) for item in loader.dataset.items}


def _run_main_to_trainer(monkeypatch, tmp_path, split_path):
    """Drive ``train.main()`` up to ``Trainer(...)`` and return its kwargs.

    ``build_training_components`` is stubbed out with a 3->2 Linear: the flow
    this test is not about takes seconds to build from a real batch, and the
    Trainer stub never touches it.
    """
    import cancer_sbi.training.trainer as trainer_mod
    from cancer_sbi.cli import train as train_cli

    def _components(cfg, train_loader, **kwargs):
        return trainer_mod.TrainingComponents(
            embedding_net=None,
            density_estimator=nn.Linear(3, 2),
            example_batch=None,
        )

    def _trainer(**kwargs):
        raise _StopAtTrainer(kwargs)

    monkeypatch.setattr(trainer_mod, "build_training_components", _components)
    monkeypatch.setattr(trainer_mod, "Trainer", _trainer)

    with pytest.raises(_StopAtTrainer) as excinfo:
        train_cli.main(
            [
                "--model", "clonemlp",
                "--data-root", str(DATA_ROOT),
                "--split", str(split_path),
                "--out", str(tmp_path / "run"),
                "--device", "cpu",
                "--batch-size", "2",
            ]
        )
    return excinfo.value.kwargs


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"local subsample missing at {DATA_ROOT}")
def test_three_key_split_gives_the_trainer_a_real_validation_loader(
    tmp_path, monkeypatch
):
    """The Trainer's val_loader covers exactly ``val_ids`` -- not the test set."""
    names = _sim_names(SPLIT_SIM_COUNT)
    train_ids, val_ids, test_ids = names[:3], names[3:4], names[4:]
    split_path = _write_split(
        tmp_path / "three.pkl",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    kwargs = _run_main_to_trainer(monkeypatch, tmp_path, split_path)
    val_loader = kwargs["val_loader"]
    train_loader = kwargs["train_loader"]

    assert val_loader is not None
    assert _loader_sim_names(val_loader) == set(val_ids)
    # The two mutants land here: a val_loader that is the test loader, or a
    # val_loader that was never built and fell back to it.
    assert _loader_sim_names(val_loader).isdisjoint(set(test_ids))
    assert _loader_sim_names(val_loader).isdisjoint(set(train_ids))
    assert val_loader is not train_loader
    # Early stopping must not shuffle, and must see every val sim.
    assert len(val_loader.dataset) == len(val_ids)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"local subsample missing at {DATA_ROOT}")
def test_three_key_split_does_not_warn_about_the_test_set(
    tmp_path, monkeypatch, capsys
):
    """With a real validation set the historical warning must be silent."""
    names = _sim_names(SPLIT_SIM_COUNT)
    split_path = _write_split(
        tmp_path / "three.pkl",
        train_ids=names[:3],
        val_ids=names[3:4],
        test_ids=names[4:],
    )
    _run_main_to_trainer(monkeypatch, tmp_path, split_path)

    out = capsys.readouterr().out
    assert "early stopping runs on the" not in out
    assert "Split: 3 train sims, 1 val sims, 2 test sims" in out


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"local subsample missing at {DATA_ROOT}")
def test_two_key_split_falls_back_to_the_test_loader_and_says_so(
    tmp_path, monkeypatch, capsys
):
    """The historical behaviour is kept for old splits -- loudly."""
    names = _sim_names(SPLIT_SIM_COUNT)
    train_ids, test_ids = names[:4], names[4:]
    split_path = _write_split(
        tmp_path / "two.pkl", train_ids=train_ids, test_ids=test_ids
    )

    kwargs = _run_main_to_trainer(monkeypatch, tmp_path, split_path)
    val_loader = kwargs["val_loader"]

    assert val_loader is not None
    assert _loader_sim_names(val_loader) == set(test_ids)

    out = capsys.readouterr().out
    assert "[warn]" in out
    assert "val_ids" in out
    assert "TEST set" in out


# ---------------------------------------------------------------------------
# 8. The three "flag does nothing for this preset" warnings.
#
# Each flag is model-specific, and accepting one silently for the wrong model
# is how a run gets launched believing it carries a repair it does not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, model, needle",
    [
        (["--input-space", "copy"], "cloneatt", "--input-space"),
        (["--input-space", "copy"], "dominantclone", "--input-space"),
        (["--freq-renorm"], "clonemlp", "--freq-renorm"),
        (["--freq-renorm"], "dominantclone", "--freq-renorm"),
        (["--top-k", "50"], "dominantclone", "--top-k"),
    ],
)
def test_flag_that_the_preset_ignores_is_warned_about(argv, model, needle, capsys):
    config_from_argv(argv, model=model)
    out = capsys.readouterr().out
    assert f"[warn] {needle} is not used by {model}" in out


@pytest.mark.parametrize(
    "argv, model",
    [
        (["--input-space", "copy"], "clonemlp"),
        (["--freq-renorm"], "cloneatt"),
        (["--top-k", "50"], "clonemlp"),
    ],
)
def test_the_preset_that_does_use_the_flag_is_not_warned_about(argv, model, capsys):
    config_from_argv(argv, model=model)
    assert "[warn]" not in capsys.readouterr().out


def test_an_ignored_flag_does_not_change_the_config():
    """The warning is not the only thing that must happen: nothing else may."""
    plain = config_from_argv([], model="cloneatt")
    warned = config_from_argv(["--input-space", "copy"], model="cloneatt")
    # `input_space` is only read by the mlp encoder, so cloneatt's encoder
    # block must otherwise be the preset's.
    assert warned.encoder.kind == plain.encoder.kind
    assert warned.flow == plain.flow
    assert config_from_argv(["--top-k", "50"], model="dominantclone").data.top_k is None


# ---------------------------------------------------------------------------
# 9. --deterministic: the environment variable and the torch call.
#
# CUBLAS_WORKSPACE_CONFIG is read by cuBLAS when its handle is created, so it
# has to be set before torch is imported, let alone before CUDA starts; a value
# set afterwards is ignored while use_deterministic_algorithms(True) still
# demands it. No GPU is touched here -- torch is monkeypatched.
# ---------------------------------------------------------------------------


def test_deterministic_sets_cublas_before_the_torch_import_in_the_source():
    """Source order is the guarantee; a runtime check cannot see it."""
    import inspect

    from cancer_sbi.cli import train as train_cli

    source = inspect.getsource(train_cli.main)
    env_line = source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]')
    import_line = source.index("import torch")
    assert env_line < import_line, "the env var must be set before torch is imported"


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"local subsample missing at {DATA_ROOT}")
def test_deterministic_flag_sets_the_env_and_calls_torch(tmp_path, monkeypatch):
    """Both halves, in one run of ``main()``."""
    import cancer_sbi.training.trainer as trainer_mod

    from cancer_sbi.cli import train as train_cli

    calls = []
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda flag: calls.append(flag)
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    names = _sim_names(4)
    split_path = _write_split(
        tmp_path / "s.pkl", train_ids=names[:2], test_ids=names[2:]
    )

    def _boom(cfg, train_loader, **kwargs):
        raise _StopAtTrainer({})

    monkeypatch.setattr(trainer_mod, "build_training_components", _boom)

    with pytest.raises(_StopAtTrainer):
        train_cli.main(
            [
                "--model", "clonemlp",
                "--data-root", str(DATA_ROOT),
                "--split", str(split_path),
                "--out", str(tmp_path / "run"),
                "--device", "cpu",
                "--batch-size", "2",
                "--deterministic",
            ]
        )

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == [True]


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"local subsample missing at {DATA_ROOT}")
def test_without_deterministic_neither_happens(tmp_path, monkeypatch):
    """Off by default, which is what the published runs did."""
    import cancer_sbi.training.trainer as trainer_mod

    from cancer_sbi.cli import train as train_cli

    calls = []
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda flag: calls.append(flag)
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    names = _sim_names(4)
    split_path = _write_split(
        tmp_path / "s.pkl", train_ids=names[:2], test_ids=names[2:]
    )

    def _boom(cfg, train_loader, **kwargs):
        raise _StopAtTrainer({})

    monkeypatch.setattr(trainer_mod, "build_training_components", _boom)

    with pytest.raises(_StopAtTrainer):
        train_cli.main(
            [
                "--model", "clonemlp",
                "--data-root", str(DATA_ROOT),
                "--split", str(split_path),
                "--out", str(tmp_path / "run"),
                "--device", "cpu",
                "--batch-size", "2",
            ]
        )

    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert calls == []
