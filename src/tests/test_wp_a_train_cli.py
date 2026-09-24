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
    assert cfg.data.require_all_trials is False
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
        assert cfg.data.require_all_trials is False, model


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


def test_require_all_trials_flag_reaches_the_data_config():
    """DominantClone's opt-in to the clone-set models' sim set (trap 10)."""
    assert config_from_argv([], model="dominantclone").data.require_all_trials is False
    cfg = config_from_argv(["--require-all-trials"], model="dominantclone")
    assert cfg.data.require_all_trials is True
    # A one-value change: nothing else in the data block moves.
    assert cfg.data.dataset == "dominant_clone"
    assert cfg.data.num_trials == 25 and cfg.data.top_k is None


def test_deterministic_flag_parses_and_is_off_by_default():
    parser = build_parser()
    assert parser.parse_args(["--model", "clonemlp"]).deterministic is False
    assert parser.parse_args(["--model", "clonemlp", "--deterministic"]).deterministic is True


# ---------------------------------------------------------------------------
# 2b. Matrix 2's switches (2026-09-24): the same rule, one level further in.
#
# R4 left CloneAtt at R^2=0.025 with a renormalised multiply that still shrinks
# every token to ~1/K, an encoder lr of 1e-4 and no dropout; R2 reached 0.415
# and overfits from epoch ~15. These six flags are those two findings, and each
# one has to survive the trip flag -> config -> checkpoint -> rebuilt network.
# ---------------------------------------------------------------------------


def test_matrix_two_flags_absent_keeps_published_defaults():
    for model in ("clonemlp", "cloneatt", "dominantclone"):
        cfg = config_from_argv([], model=model)
        preset = get_preset(model)
        assert cfg.encoder.freq_mode == "weight", model
        assert cfg.encoder.attn_ln is False, model
        assert cfg.encoder.attn_dropout_active is False, model
        assert cfg.encoder.dropout == preset.encoder.dropout, model
        assert cfg.optim.embed_lr == preset.optim.embed_lr, model
        assert cfg.optim.embed_weight_decay == preset.optim.embed_weight_decay, model
        assert cfg.optim.flow_weight_decay == preset.optim.flow_weight_decay, model
        # Trap 5's documentary field is untouched by the wired one.
        assert cfg.encoder.layer_norm_in_attention is False, model


@pytest.mark.parametrize("mode", ["weight", "feature"])
def test_freq_mode_flag_reaches_the_encoder(mode):
    cfg = config_from_argv(["--freq-mode", mode], model="cloneatt")
    assert cfg.encoder.freq_mode == mode


def test_attn_ln_flag_reaches_the_encoder_without_touching_the_record():
    cfg = config_from_argv(["--attn-ln"], model="cloneatt")
    assert cfg.encoder.attn_ln is True
    # `layer_norm_in_attention` records what the PUBLISHED model did and is read
    # by verify_refactor.py:282; the switch must not rewrite that history.
    assert cfg.encoder.layer_norm_in_attention is False


def test_encoder_dropout_flag_reaches_the_mlp_encoder():
    cfg = config_from_argv(["--encoder-dropout", "0.35"], model="clonemlp")
    assert cfg.encoder.dropout == 0.35
    # clonemlp already builds nn.Dropout from it (trap 4 is cloneatt's).
    assert cfg.encoder.encoder_dropout_is_used is True
    assert cfg.encoder.attn_dropout_active is False


def test_encoder_dropout_flag_also_activates_it_for_cloneatt():
    """Trap 4: setting the probability alone would change nothing at all."""
    assert get_preset("cloneatt").encoder.dropout == 0.2
    assert get_preset("cloneatt").encoder.attn_dropout_active is False
    cfg = config_from_argv(["--encoder-dropout", "0.1"], model="cloneatt")
    assert cfg.encoder.dropout == 0.1
    assert cfg.encoder.attn_dropout_active is True


@pytest.mark.parametrize(
    "flag, field, value",
    [
        ("--embed-lr", "embed_lr", 5e-4),
        ("--embed-weight-decay", "embed_weight_decay", 1e-4),
        ("--flow-weight-decay", "flow_weight_decay", 1e-3),
    ],
)
@pytest.mark.parametrize("model", ["clonemlp", "cloneatt"])
def test_optimiser_overrides_reach_the_optim_block(flag, field, value, model):
    cfg = config_from_argv([flag, str(value)], model=model)
    assert getattr(cfg.optim, field) == value
    # A one-value change: trap 2's structure and trap 3's dead rate stay put.
    assert cfg.optim.use_param_groups is True
    assert cfg.optim.learning_rate_is_used is False


def test_optimiser_overrides_are_not_recorded_for_the_one_group_model():
    """Trap 2: dominantclone reads neither group field, so neither may be set."""
    plain = config_from_argv([], model="dominantclone")
    cfg = config_from_argv(
        ["--embed-lr", "5e-4", "--embed-weight-decay", "1e-4",
         "--flow-weight-decay", "1e-3"],
        model="dominantclone",
    )
    assert cfg.optim == plain.optim
    assert cfg.optim.embed_lr is None and cfg.optim.flow_weight_decay is None


def test_the_optimiser_overrides_actually_reach_the_parameter_groups():
    """build_optimizer reads the fields; this is the proof it reads the new ones."""
    from cancer_sbi.training.trainer import build_optimizer

    cfg = config_from_argv(
        ["--embed-lr", "5e-4", "--embed-weight-decay", "1e-4",
         "--flow-weight-decay", "1e-3"],
        model="clonemlp",
    ).optim
    embed = nn.Linear(4, 3)
    flow = nn.Sequential(embed, nn.Linear(3, 2))
    groups = build_optimizer(flow, embed, cfg).param_groups

    # Group order is fixed by build_optimizer: flow first, embedding second.
    assert groups[0]["lr"] == cfg.flow_lr == 1e-3
    assert groups[0]["weight_decay"] == 1e-3
    assert groups[1]["lr"] == 5e-4
    assert groups[1]["weight_decay"] == 1e-4
    # The split is by object identity, not by name (see build_optimizer).
    assert len(groups[1]["params"]) == len(list(embed.parameters()))


def test_freq_mode_feature_and_freq_renorm_are_refused_together():
    """'feature' removes the multiply, so there is nothing left to renormalise."""
    with pytest.raises(ValueError, match="freq-renorm"):
        config_from_argv(["--freq-mode", "feature", "--freq-renorm"], model="cloneatt")
    # Either one alone is fine.
    assert config_from_argv(["--freq-mode", "feature"], model="cloneatt").encoder.freq_mode == "feature"
    assert config_from_argv(["--freq-renorm"], model="cloneatt").encoder.freq_renorm is True


def test_input_space_now_reaches_the_set_transformer_too():
    """Matrix 2 item 1: the flag stopped being clonemlp-only."""
    cfg = config_from_argv(["--input-space", "copy"], model="cloneatt")
    assert cfg.encoder.input_space == "copy"
    assert cfg.encoder.kind == "attention"


def test_matrix_two_flags_ride_into_the_effective_config():
    """Round trip: flags -> preset -> checkpoint dict -> preset again."""
    from cancer_sbi.cli.train import effective_config_payload
    from cancer_sbi.config import preset_from_effective_config

    argv = [
        "--input-space", "copy", "--freq-mode", "feature", "--attn-ln",
        "--encoder-dropout", "0.1", "--embed-lr", "5e-4",
        "--flow-weight-decay", "1e-3",
    ]
    args = build_parser().parse_args(["--model", "cloneatt"] + argv)
    cfg = config_from_argv(argv, model="cloneatt")
    payload = effective_config_payload(cfg, args, Path("/d"), Path("/s.pkl"))

    assert payload["encoder"]["freq_mode"] == "feature"
    assert payload["encoder"]["attn_ln"] is True
    assert payload["encoder"]["attn_dropout_active"] is True
    assert payload["encoder"]["dropout"] == 0.1
    assert payload["encoder"]["input_space"] == "copy"
    assert payload["optim"]["embed_lr"] == 5e-4
    assert payload["optim"]["flow_weight_decay"] == 1e-3
    assert payload["cli_flags"]["freq_mode"] == "feature"
    assert payload["cli_flags"]["attn_ln"] is True
    assert payload["cli_flags"]["encoder_dropout"] == 0.1

    # What evaluation rebuilds from. The optim block is deliberately NOT part of
    # preset_from_effective_config (it decides no shape), so only the encoder
    # and flow blocks are asserted here.
    rebuilt = preset_from_effective_config(payload)
    assert rebuilt.encoder.freq_mode == "feature"
    assert rebuilt.encoder.attn_ln is True
    assert rebuilt.encoder.attn_dropout_active is True
    assert rebuilt.encoder.input_space == "copy"
    assert rebuilt.encoder.dropout == 0.1


def test_an_old_checkpoint_without_the_new_keys_still_rebuilds():
    """Every checkpoint on the cluster predates these fields."""
    from cancer_sbi.config import preset_from_effective_config

    stored = {"model": "cloneatt", "encoder": {"kind": "attention"}, "flow": {}}
    rebuilt = preset_from_effective_config(stored)
    assert rebuilt.encoder.freq_mode == "weight"
    assert rebuilt.encoder.attn_ln is False
    assert rebuilt.encoder.attn_dropout_active is False


def test_build_embedding_net_forwards_the_new_encoder_fields():
    """The seam evaluation goes through; a dropped argument is silent otherwise."""
    cfg = config_from_argv(
        ["--input-space", "copy", "--freq-mode", "feature", "--attn-ln",
         "--encoder-dropout", "0.1"],
        model="cloneatt",
    ).encoder
    encoder = build_embedding_net(cfg, device="cpu").trial_encoder

    assert encoder.freq_mode == "feature"
    assert encoder.input_space == "copy"
    assert encoder.attn_ln is True
    assert encoder.attn_dropout is not None and encoder.attn_dropout.p == 0.1
    assert encoder.input_proj.in_features == 45
    assert any(isinstance(m, nn.LayerNorm) for m in encoder.modules())

    # And the default still builds the published network.
    published = build_embedding_net(get_preset("cloneatt").encoder, "cpu").trial_encoder
    assert published.input_proj.in_features == 44
    assert published.attn_dropout is None
    assert not any(isinstance(m, nn.LayerNorm) for m in published.modules())


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
        # --input-space is no longer cloneatt-only-ignored: matrix 2 wired the
        # same copy-space transform into the set transformer, so dominantclone
        # is the only preset left that ignores it.
        (["--input-space", "copy"], "dominantclone", "--input-space"),
        (["--freq-renorm"], "clonemlp", "--freq-renorm"),
        (["--freq-renorm"], "dominantclone", "--freq-renorm"),
        (["--top-k", "50"], "dominantclone", "--top-k"),
        (["--require-all-trials"], "clonemlp", "--require-all-trials"),
        (["--require-all-trials"], "cloneatt", "--require-all-trials"),
        # Matrix 2: the three attention-only switches and the three
        # two-group-optimiser overrides. --freq-mode has since stopped being
        # one of them: matrix 3 gave clonemlp its own "feature" mode (a 45th
        # log10 input column), so dominantclone is the only preset left that
        # ignores it -- see test_freq_mode_no_longer_warns_for_clonemlp.
        (["--freq-mode", "feature"], "dominantclone", "--freq-mode"),
        (["--attn-ln"], "clonemlp", "--attn-ln"),
        (["--attn-ln"], "dominantclone", "--attn-ln"),
        (["--encoder-dropout", "0.1"], "dominantclone", "--encoder-dropout"),
        (["--embed-lr", "5e-4"], "dominantclone", "--embed-lr"),
        (["--embed-weight-decay", "1e-4"], "dominantclone", "--embed-weight-decay"),
        (["--flow-weight-decay", "1e-3"], "dominantclone", "--flow-weight-decay"),
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
        # Matrix 2 made this one a real switch for cloneatt too.
        (["--input-space", "copy"], "cloneatt"),
        (["--freq-renorm"], "cloneatt"),
        (["--top-k", "50"], "clonemlp"),
        (["--require-all-trials"], "dominantclone"),
        (["--freq-mode", "feature"], "cloneatt"),
        (["--attn-ln"], "cloneatt"),
        (["--encoder-dropout", "0.1"], "clonemlp"),
        (["--encoder-dropout", "0.1"], "cloneatt"),
        (["--embed-lr", "5e-4"], "clonemlp"),
        (["--flow-weight-decay", "1e-3"], "cloneatt"),
    ],
)
def test_the_preset_that_does_use_the_flag_is_not_warned_about(argv, model, capsys):
    config_from_argv(argv, model=model)
    assert "[warn]" not in capsys.readouterr().out


def test_an_ignored_flag_does_not_change_the_config():
    """The warning is not the only thing that must happen: nothing else may."""
    plain = config_from_argv([], model="dominantclone")
    warned = config_from_argv(["--input-space", "copy"], model="dominantclone")
    # DeepSet reads no `input_space`, so its encoder block must be the preset's
    # -- a recorded value would ride into the checkpoint describing a transform
    # nothing applied.
    assert warned.encoder == plain.encoder
    assert warned.flow == plain.flow
    assert config_from_argv(["--top-k", "50"], model="dominantclone").data.top_k is None
    # The clone-set path already applies the rule, so the flag must not land in
    # its config (and hence not in its checkpoint) pretending it was applied.
    assert (
        config_from_argv(["--require-all-trials"], model="cloneatt").data.require_all_trials
        is False
    )


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


# ---------------------------------------------------------------------------
# 7. Matrix 3: --flow-dropout, --flow-num-transforms, --trial-subsample, and
#    --freq-mode for clonemlp.
# ---------------------------------------------------------------------------


def test_matrix_three_flags_absent_keeps_published_defaults():
    """Omit all three and every preset keeps the run it produced before."""
    for model in ("clonemlp", "cloneatt", "dominantclone"):
        cfg = config_from_argv([], model=model)
        # 0.2, from */main.py:35 -- the published value, not 0.
        assert cfg.flow.dropout_probability == 0.2, model
        assert cfg.flow.num_transforms == 5, model
        assert cfg.flow.hidden_features == 50, model
        assert cfg.data.trial_subsample is None, model
        assert cfg.encoder.freq_mode == "weight", model


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "dominantclone"])
def test_flow_dropout_reaches_every_preset(model):
    cfg = config_from_argv(["--flow-dropout", "0.1"], model=model)
    assert cfg.flow.dropout_probability == 0.1
    # Nothing else in the flow block moves with it.
    assert cfg.flow.num_transforms == 5
    assert cfg.flow.hidden_features == 50


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "dominantclone"])
def test_flow_num_transforms_reaches_every_preset(model):
    cfg = config_from_argv(["--flow-num-transforms", "3"], model=model)
    assert cfg.flow.num_transforms == 3
    assert cfg.flow.dropout_probability == 0.2
    # hidden_features has no flag on purpose: 50 is the published width.
    assert cfg.flow.hidden_features == 50


def test_flow_overrides_compose_with_z_score_x():
    """All three land in one FlowConfig; none of them drops the others."""
    cfg = config_from_argv(
        ["--z-score-x", "structured", "--flow-dropout", "0.1",
         "--flow-num-transforms", "3"]
    )
    assert cfg.flow.z_score_x == "structured"
    assert cfg.flow.dropout_probability == 0.1
    assert cfg.flow.num_transforms == 3


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt"])
def test_trial_subsample_reaches_the_clone_set_presets(model):
    cfg = config_from_argv(["--trial-subsample", "16"], model=model)
    assert cfg.data.trial_subsample == 16


def test_trial_subsample_is_warned_and_ignored_for_dominantclone(capsys):
    """SimulationDataset NaN-pads instead of dropping; see cli/train.py."""
    cfg = config_from_argv(["--trial-subsample", "16"], model="dominantclone")
    assert cfg.data.trial_subsample is None
    assert "--trial-subsample is not used by dominantclone" in capsys.readouterr().out


def test_freq_mode_feature_now_reaches_clonemlp():
    """Matrix 3 item 4: the flag stopped being cloneatt-only."""
    cfg = config_from_argv(["--freq-mode", "feature"], model="clonemlp")
    assert cfg.encoder.freq_mode == "feature"
    # The raw-frequency column stays off: it is a different column, and the
    # published default.
    assert cfg.encoder.include_freq_in_mlp is False


def test_freq_mode_no_longer_warns_for_clonemlp(capsys):
    config_from_argv(["--freq-mode", "feature"], model="clonemlp")
    assert "--freq-mode is not used" not in capsys.readouterr().out


def test_freq_mode_still_warns_for_dominantclone(capsys):
    cfg = config_from_argv(["--freq-mode", "feature"], model="dominantclone")
    assert cfg.encoder.freq_mode == "weight"
    assert "--freq-mode is not used by dominantclone" in capsys.readouterr().out


def test_clonemlp_feature_mode_builds_a_45_input_mlp():
    """The flag has to reach the layer, not only the dataclass."""
    cfg = config_from_argv(["--freq-mode", "feature"], model="clonemlp")
    encoder = build_embedding_net(cfg.encoder, device="cpu").trial_encoder
    assert encoder.freq_mode == "feature"
    assert encoder.mlp[0].in_features == 45
    # And the published build is still 44 wide.
    published = build_embedding_net(get_preset("clonemlp").encoder, "cpu").trial_encoder
    assert published.mlp[0].in_features == 44


def test_matrix_three_flags_ride_into_the_effective_config():
    """Round trip: flags -> preset -> checkpoint dict -> preset again."""
    from cancer_sbi.cli.train import effective_config_payload
    from cancer_sbi.config import preset_from_effective_config

    argv = [
        "--input-space", "copy", "--freq-mode", "feature",
        "--flow-dropout", "0.1", "--flow-num-transforms", "3",
        "--trial-subsample", "16",
    ]
    args = build_parser().parse_args(["--model", "clonemlp"] + argv)
    cfg = config_from_argv(argv, model="clonemlp")
    payload = effective_config_payload(cfg, args, Path("/d"), Path("/s.pkl"))

    assert payload["flow"]["dropout_probability"] == 0.1
    assert payload["flow"]["num_transforms"] == 3
    assert payload["encoder"]["freq_mode"] == "feature"
    assert payload["data"]["trial_subsample"] == 16
    assert payload["cli_flags"]["flow_dropout"] == 0.1
    assert payload["cli_flags"]["flow_num_transforms"] == 3
    assert payload["cli_flags"]["trial_subsample"] == 16

    # What evaluation rebuilds: the flow's shape and the encoder's input width.
    # `data` is deliberately not part of preset_from_effective_config, so the
    # rebuilt test loader keeps all 25 trials -- which is the intent.
    rebuilt = preset_from_effective_config(payload)
    assert rebuilt.flow.dropout_probability == 0.1
    assert rebuilt.flow.num_transforms == 3
    assert rebuilt.encoder.freq_mode == "feature"
    assert rebuilt.data.trial_subsample is None


def test_a_checkpoint_without_the_matrix_three_keys_still_rebuilds():
    """Every checkpoint on the cluster predates these fields."""
    from cancer_sbi.config import preset_from_effective_config

    rebuilt = preset_from_effective_config(
        {
            "model": "clonemlp",
            "flow": {"z_score_x": "structured"},
            "encoder": {"kind": "mlp"},
        }
    )
    assert rebuilt.flow.dropout_probability == 0.2
    assert rebuilt.flow.num_transforms == 5
    assert rebuilt.encoder.freq_mode == "weight"


# ---------------------------------------------------------------------------
# 10. Matrix 4: --attn-scale, --tail-bound, --trial-pool and the three
#     capacity knobs.
# ---------------------------------------------------------------------------


def test_matrix_four_flags_absent_keeps_published_defaults():
    """Omit all six and every preset keeps the run it produced before."""
    for model in ("clonemlp", "cloneatt", "dominantclone"):
        cfg = config_from_argv([], model=model)
        assert cfg.encoder.attn_scale == "published", model
        assert cfg.encoder.trial_pool == "mean", model
        assert cfg.flow.tail_bound == 3.0, model
    published = config_from_argv([], model="cloneatt").encoder
    assert (published.d_model, published.n_heads, published.num_inducing) == (
        128,
        8,
        32,
    )


def test_attn_scale_flag_reaches_the_encoder():
    cfg = config_from_argv(["--attn-scale", "standard"], model="cloneatt")
    assert cfg.encoder.attn_scale == "standard"
    # Trap 6's documentary neighbour is untouched: attn_scale is the wired one.
    assert cfg.encoder.layer_norm_in_attention is False


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "dominantclone"])
def test_tail_bound_reaches_every_preset(model):
    cfg = config_from_argv(["--tail-bound", "5"], model=model)
    assert cfg.flow.tail_bound == 5.0
    # Nothing else in the flow block moves with it.
    assert cfg.flow.num_transforms == 5
    assert cfg.flow.hidden_features == 50


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt"])
def test_trial_pool_reaches_the_clone_set_presets(model):
    cfg = config_from_argv(["--trial-pool", "attention"], model=model)
    assert cfg.encoder.trial_pool == "attention"


def test_capacity_knobs_reach_the_encoder():
    cfg = config_from_argv(
        ["--d-model", "256", "--n-heads", "4", "--num-inducing", "64"],
        model="cloneatt",
    )
    assert (cfg.encoder.d_model, cfg.encoder.n_heads, cfg.encoder.num_inducing) == (
        256,
        4,
        64,
    )


def test_d_model_reaches_clonemlp_too():
    """BaselineCloneEmbedding takes d_model as well; only DeepSet does not."""
    cfg = config_from_argv(["--d-model", "64"], model="clonemlp")
    assert cfg.encoder.d_model == 64
    assert build_embedding_net(cfg.encoder, "cpu").trial_encoder.d_model == 64


def test_indivisible_d_model_and_n_heads_is_refused():
    """MAB's integer split would silently drop the remainder of every token."""
    with pytest.raises(ValueError, match="not divisible"):
        config_from_argv(["--d-model", "100", "--n-heads", "8"], model="cloneatt")
    with pytest.raises(ValueError, match="not divisible"):
        config_from_argv(["--n-heads", "12"], model="cloneatt")


def test_trial_pool_attention_checks_the_pooling_head_count():
    """The pooling PMA has the constraint too, on the heads it will use."""
    with pytest.raises(ValueError, match="trial-pool attention"):
        config_from_argv(
            ["--trial-pool", "attention", "--d-model", "100"], model="clonemlp"
        )


@pytest.mark.parametrize(
    "argv, model, needle",
    [
        (["--attn-scale", "standard"], "clonemlp", "--attn-scale"),
        (["--attn-scale", "standard"], "dominantclone", "--attn-scale"),
        (["--trial-pool", "attention"], "dominantclone", "--trial-pool"),
        (["--d-model", "64"], "dominantclone", "--d-model"),
        (["--n-heads", "4"], "clonemlp", "--n-heads"),
        (["--n-heads", "4"], "dominantclone", "--n-heads"),
        (["--num-inducing", "64"], "clonemlp", "--num-inducing"),
        (["--num-inducing", "64"], "dominantclone", "--num-inducing"),
    ],
)
def test_matrix_four_flag_the_preset_ignores_is_warned_about(
    argv, model, needle, capsys
):
    cfg = config_from_argv(argv, model=model)
    assert f"[warn] {needle} is not used by {model}" in capsys.readouterr().out
    # And nothing is recorded: a value in the checkpoint would describe a
    # network nothing built.
    assert cfg.encoder == get_preset(model).encoder


@pytest.mark.parametrize(
    "argv, model",
    [
        (["--attn-scale", "standard"], "cloneatt"),
        (["--trial-pool", "attention"], "cloneatt"),
        (["--trial-pool", "attention"], "clonemlp"),
        (["--d-model", "64"], "cloneatt"),
        (["--d-model", "64"], "clonemlp"),
        (["--n-heads", "4"], "cloneatt"),
        (["--num-inducing", "64"], "cloneatt"),
        # --tail-bound is read by every preset's flow, so it never warns.
        (["--tail-bound", "5"], "cloneatt"),
        (["--tail-bound", "5"], "dominantclone"),
    ],
)
def test_matrix_four_flag_the_preset_uses_is_not_warned_about(argv, model, capsys):
    config_from_argv(argv, model=model)
    assert "[warn]" not in capsys.readouterr().out


def test_matrix_four_switches_reach_the_built_modules():
    """The flags have to reach the layers, not only the dataclass."""
    cfg = config_from_argv(
        [
            "--attn-scale", "standard",
            "--trial-pool", "attention",
            "--d-model", "64",
            "--n-heads", "4",
            "--num-inducing", "8",
        ],
        model="cloneatt",
    )
    net = build_embedding_net(cfg.encoder, "cpu")
    encoder = net.trial_encoder
    assert encoder.attn_scale == "standard"
    assert encoder.d_model == 64
    assert encoder.layers[0].mab0.num_heads == 4
    assert encoder.layers[0].I.shape == (1, 8, 64)
    # The pooling PMA follows the encoder, and the context width does not move.
    assert net.trial_pool == "attention"
    assert net.pool_attn.pma.mab.attn_scale == "standard"
    assert net.pool_attn.pma.mab.num_heads == 4
    assert net.pool_attn.output_dim == get_preset("cloneatt").encoder.trials_output_dim


def test_capacity_knobs_change_the_parameter_count():
    """R21/R23 are bigger networks, R22 is the same one differently split."""

    def _count(argv):
        cfg = config_from_argv(argv, model="cloneatt")
        return sum(p.numel() for p in build_embedding_net(cfg.encoder, "cpu").parameters())

    published = _count([])
    assert _count(["--d-model", "256"]) > published
    assert _count(["--num-inducing", "64"]) > published
    assert _count(["--n-heads", "4"]) == published
    # Attention pooling replaces sbi's mean with a PMA plus the same MLP.
    assert _count(["--trial-pool", "attention"]) > published


def test_matrix_four_flags_ride_into_the_effective_config():
    """Round trip: flags -> preset -> checkpoint dict -> preset again."""
    from cancer_sbi.cli.train import effective_config_payload
    from cancer_sbi.config import preset_from_effective_config

    argv = [
        "--attn-scale", "standard",
        "--tail-bound", "5",
        "--trial-pool", "attention",
        "--d-model", "256",
        "--n-heads", "4",
        "--num-inducing", "64",
    ]
    args = build_parser().parse_args(["--model", "cloneatt"] + argv)
    cfg = config_from_argv(argv, model="cloneatt")
    payload = effective_config_payload(cfg, args, Path("/d"), Path("/s.pkl"))

    assert payload["encoder"]["attn_scale"] == "standard"
    assert payload["encoder"]["trial_pool"] == "attention"
    assert payload["encoder"]["d_model"] == 256
    assert payload["encoder"]["n_heads"] == 4
    assert payload["encoder"]["num_inducing"] == 64
    assert payload["flow"]["tail_bound"] == 5.0
    assert payload["flow"]["hidden_features"] == 50, "must stay 50"
    for flag, value in (
        ("attn_scale", "standard"), ("tail_bound", 5.0), ("trial_pool", "attention"),
        ("d_model", 256), ("n_heads", 4), ("num_inducing", 64),
    ):
        assert payload["cli_flags"][flag] == value

    rebuilt = preset_from_effective_config(payload)
    assert rebuilt.encoder.attn_scale == "standard"
    assert rebuilt.encoder.trial_pool == "attention"
    assert rebuilt.encoder.d_model == 256
    assert rebuilt.encoder.n_heads == 4
    assert rebuilt.encoder.num_inducing == 64
    assert rebuilt.flow.tail_bound == 5.0


def test_a_checkpoint_without_the_matrix_four_keys_still_rebuilds():
    """Every checkpoint on the cluster predates these fields."""
    from cancer_sbi.config import preset_from_effective_config

    rebuilt = preset_from_effective_config(
        {
            "model": "cloneatt",
            "flow": {"z_score_x": "structured"},
            "encoder": {"kind": "attention"},
        }
    )
    assert rebuilt.encoder.attn_scale == "published"
    assert rebuilt.encoder.trial_pool == "mean"
    assert rebuilt.flow.tail_bound == 3.0


# ---------------------------------------------------------------------------
# 13. Matrix 5: armtoken's flags, and the five that do nothing for it.
#
# ArmToken reads `attn_ln`, `attn_scale`, `n_heads`, `trial_pool`,
# `input_space` and `dropout` like CloneAtt, and reads NOTHING of
# `freq_mode`, `freq_renorm`, `d_model` or `num_inducing` -- the moments
# renormalise their own weights, the output width is 44 * d_arm + d_global,
# and its ISABs are sized by the separate --arm-num-inducing. Accepting any of
# the four silently is how a run gets launched believing it carries a switch
# it does not.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, needle",
    [
        (["--freq-mode", "feature"], "--freq-mode"),
        (["--freq-renorm"], "--freq-renorm"),
        (["--d-model", "64"], "--d-model"),
        (["--num-inducing", "64"], "--num-inducing"),
        (["--require-all-trials"], "--require-all-trials"),
    ],
)
def test_matrix_five_flag_armtoken_ignores_is_warned_about(argv, needle, capsys):
    cfg = config_from_argv(argv, model="armtoken")
    assert f"[warn] {needle} is not used by armtoken" in capsys.readouterr().out
    # And nothing is recorded: a value in the checkpoint would describe a
    # network nothing built.
    assert cfg.encoder == get_preset("armtoken").encoder
    assert cfg.data.require_all_trials is False


@pytest.mark.parametrize(
    "argv",
    [
        ["--input-space", "log2"],
        ["--attn-ln"],
        ["--encoder-dropout", "0.1"],
        ["--trial-pool", "attention"],
        ["--attn-scale", "standard"],
        ["--n-heads", "8"],
        ["--arm-layers", "0"],
        ["--d-arm", "4"],
        ["--arm-num-inducing", "8"],
        ["--z-score-x", "independent"],
        ["--tail-bound", "5"],
        ["--flow-dropout", "0.1"],
        ["--flow-num-transforms", "5"],
        ["--embed-lr", "1e-3"],
        ["--embed-weight-decay", "1e-4"],
        ["--flow-weight-decay", "1e-3"],
        ["--trial-subsample", "16"],
        ["--seed", "1"],
        ["--cache-dir", "/tmp/cache"],
    ],
)
def test_matrix_five_flag_armtoken_uses_is_not_warned_about(argv, capsys):
    config_from_argv(argv, model="armtoken")
    assert "[warn]" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, needle",
    [
        (["--arm-layers", "2"], "--arm-layers"),
        (["--d-arm", "4"], "--d-arm"),
        (["--arm-num-inducing", "8"], "--arm-num-inducing"),
    ],
)
@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "dominantclone"])
def test_armtoken_only_flags_warn_for_the_other_models(argv, needle, model, capsys):
    cfg = config_from_argv(argv, model=model)
    assert f"[warn] {needle} is not used by {model}" in capsys.readouterr().out
    assert cfg.encoder == get_preset(model).encoder


def test_armtoken_flags_reach_the_built_module():
    """The flags have to reach the layers, not only the dataclass."""
    cfg = config_from_argv(
        [
            "--trial-pool", "attention",
            "--attn-scale", "standard",
            "--arm-layers", "2",
            "--d-arm", "4",
            "--arm-num-inducing", "8",
            "--n-heads", "8",
            "--encoder-dropout", "0.1",
            "--input-space", "log2",
        ],
        model="armtoken",
    )
    net = build_embedding_net(cfg.encoder, "cpu")
    assert net.trial_pool == "attention"
    assert net.attn_scale == "standard"
    assert len(net.arm_layers) == 2
    assert net.arm_layers[0].I.shape == (1, 8, 64)
    assert net.arm_layers[0].mab0.num_heads == 8
    assert net.arm_head.out_features == 4
    assert net.input_space == "log2"
    # --encoder-dropout must wire the layers up as well as set the probability,
    # the same trap-4 shape CloneAtt has.
    assert net.arm_dropout is not None and net.arm_dropout.p == 0.1
    assert net.d_model == 44 * 4 + 64


def test_d_arm_over_the_cap_is_refused():
    """44 * 16 + 64 = 768 would make the flow's first layer the whole network."""
    with pytest.raises(ValueError, match="512"):
        config_from_argv(["--d-arm", "16"], model="armtoken")
    with pytest.raises(ValueError, match="--d-arm"):
        config_from_argv(["--d-arm", "16"], model="armtoken")
    # The published value is inside it, and one step under the cap is allowed.
    assert config_from_argv(["--d-arm", "10"], model="armtoken").encoder.d_arm == 10


def test_indivisible_d_token_and_n_heads_is_refused_for_armtoken():
    with pytest.raises(ValueError, match="does not divide"):
        config_from_argv(["--n-heads", "7"], model="armtoken")


def test_matrix_five_flags_ride_into_the_effective_config():
    """Round trip: flags -> preset -> checkpoint dict -> preset again."""
    from cancer_sbi.cli.train import effective_config_payload
    from cancer_sbi.config import preset_from_effective_config

    argv = ["--arm-layers", "0", "--d-arm", "4", "--arm-num-inducing", "8",
            "--trial-pool", "attention"]
    args = build_parser().parse_args(["--model", "armtoken"] + argv)
    cfg = config_from_argv(argv, model="armtoken")
    payload = effective_config_payload(cfg, args, Path("/d"), Path("/s.pkl"))

    assert payload["model"] == "armtoken"
    assert payload["encoder"]["kind"] == "armtoken"
    assert payload["encoder"]["n_arm_layers"] == 0
    assert payload["encoder"]["d_arm"] == 4
    assert payload["encoder"]["arm_num_inducing"] == 8
    assert payload["encoder"]["trial_pool"] == "attention"
    assert payload["flow"]["hidden_features"] == 50, "must stay 50"
    for flag, value in (
        ("arm_layers", 0), ("d_arm", 4), ("arm_num_inducing", 8),
        ("trial_pool", "attention"),
    ):
        assert payload["cli_flags"][flag] == value

    rebuilt = preset_from_effective_config(payload)
    assert rebuilt.encoder.n_arm_layers == 0
    assert rebuilt.encoder.d_arm == 4
    assert rebuilt.encoder.arm_num_inducing == 8
    assert rebuilt.encoder.trial_pool == "attention"


def test_armtoken_is_a_model_choice():
    assert "armtoken" in build_parser().parse_args(
        ["--model", "armtoken"]
    ).model
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "armtokens"])


# --- matrix 7: --min-trials ------------------------------------------------


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "armtoken", "hybrid"])
def test_min_trials_reaches_the_clone_set_presets(model):
    cfg = config_from_argv(["--min-trials", "5"], model=model)
    assert cfg.data.min_trials == 5


@pytest.mark.parametrize(
    "model", ["clonemlp", "cloneatt", "armtoken", "hybrid", "dominantclone"]
)
def test_min_trials_defaults_to_the_published_rule(model):
    assert config_from_argv([], model=model).data.min_trials is None


def test_min_trials_is_warned_and_ignored_for_dominantclone(capsys):
    """SimulationDataset NaN-pads already; there is no bar to lower."""
    cfg = config_from_argv(["--min-trials", "5"], model="dominantclone")
    assert cfg.data.min_trials is None
    assert "--min-trials is not used by dominantclone" in capsys.readouterr().out


def test_min_trials_is_recorded_in_the_effective_config():
    from cancer_sbi.cli.train import effective_config_payload

    argv = ["--min-trials", "5"]
    args = build_parser().parse_args(["--model", "armtoken"] + argv)
    cfg = config_from_argv(argv, model="armtoken")
    payload = effective_config_payload(cfg, args, Path("/d"), Path("/s.pkl"))
    assert payload["data"]["min_trials"] == 5
    assert payload["cli_flags"]["min_trials"] == 5

    plain = build_parser().parse_args(["--model", "armtoken"])
    plain_payload = effective_config_payload(
        config_from_argv([], model="armtoken"), plain, Path("/d"), Path("/s.pkl")
    )
    assert plain_payload["data"]["min_trials"] is None
    assert plain_payload["cli_flags"]["min_trials"] is None


# ---------------------------------------------------------------------------
# 15. Matrix 8: --arm-feature-norm and --arm-context-norm.
#
# Both are read by the armtoken encoder and by the hybrid's ARM branch, and by
# nothing else. Both default to AT0's behaviour, which builds no module at all,
# so the first test here is the one that matters: absence must leave the
# published values in place on every preset.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model", ["clonemlp", "cloneatt", "dominantclone", "armtoken", "hybrid"]
)
def test_matrix_eight_flags_absent_keeps_the_at0_behaviour(model):
    cfg = config_from_argv([], model=model)
    assert cfg.encoder.arm_feature_norm == "none"
    assert cfg.encoder.arm_context_norm is False


@pytest.mark.parametrize("model", ["armtoken", "hybrid"])
@pytest.mark.parametrize("norm", ["layernorm", "batchnorm", "none"])
def test_arm_feature_norm_reaches_the_arm_branch_presets(model, norm):
    cfg = config_from_argv(["--arm-feature-norm", norm], model=model)
    assert cfg.encoder.arm_feature_norm == norm


@pytest.mark.parametrize("model", ["armtoken", "hybrid"])
def test_arm_context_norm_reaches_the_arm_branch_presets(model):
    cfg = config_from_argv(["--arm-context-norm"], model=model)
    assert cfg.encoder.arm_context_norm is True


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "dominantclone"])
def test_arm_feature_norm_is_warned_and_ignored_elsewhere(model, capsys):
    """A value on a preset with no arm branch would describe nothing built."""
    cfg = config_from_argv(["--arm-feature-norm", "layernorm"], model=model)
    assert cfg.encoder.arm_feature_norm == "none"
    out = capsys.readouterr().out
    assert f"--arm-feature-norm is not used by {model}" in out


@pytest.mark.parametrize("model", ["clonemlp", "cloneatt", "dominantclone"])
def test_arm_context_norm_is_warned_and_ignored_elsewhere(model, capsys):
    cfg = config_from_argv(["--arm-context-norm"], model=model)
    assert cfg.encoder.arm_context_norm is False
    out = capsys.readouterr().out
    assert f"--arm-context-norm is not used by {model}" in out


@pytest.mark.parametrize("model", ["armtoken", "hybrid"])
def test_the_arm_branch_presets_warn_about_neither(model, capsys):
    config_from_argv(
        ["--arm-feature-norm", "batchnorm", "--arm-context-norm"], model=model
    )
    out = capsys.readouterr().out
    assert "--arm-feature-norm" not in out and "--arm-context-norm" not in out


def test_an_unknown_arm_feature_norm_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--model", "armtoken", "--arm-feature-norm", "groupnorm"]
        )


def test_matrix_eight_flags_reach_the_effective_config():
    from cancer_sbi.cli.train import effective_config_payload

    argv = ["--arm-feature-norm", "batchnorm", "--arm-context-norm"]
    args = build_parser().parse_args(["--model", "armtoken"] + argv)
    cfg = config_from_argv(argv, model="armtoken")
    payload = effective_config_payload(cfg, args, Path("/d"), Path("/s.pkl"))
    assert payload["encoder"]["arm_feature_norm"] == "batchnorm"
    assert payload["encoder"]["arm_context_norm"] is True
    assert payload["cli_flags"]["arm_feature_norm"] == "batchnorm"
    assert payload["cli_flags"]["arm_context_norm"] is True

    plain = build_parser().parse_args(["--model", "armtoken"])
    plain_payload = effective_config_payload(
        config_from_argv([], model="armtoken"), plain, Path("/d"), Path("/s.pkl")
    )
    assert plain_payload["encoder"]["arm_feature_norm"] == "none"
    assert plain_payload["encoder"]["arm_context_norm"] is False
    assert plain_payload["cli_flags"]["arm_feature_norm"] is None
    assert plain_payload["cli_flags"]["arm_context_norm"] is False


def test_the_flags_really_build_the_modules():
    from cancer_sbi.training.trainer import build_embedding_net

    cfg = config_from_argv(
        ["--arm-feature-norm", "layernorm", "--arm-context-norm"], model="armtoken"
    )
    net = build_embedding_net(cfg.encoder, "cpu")
    assert net.arm_norm is not None and net.context_norm is not None
    # ...and their absence really does not.
    plain = build_embedding_net(config_from_argv([], model="armtoken").encoder, "cpu")
    assert plain.arm_norm is None and plain.context_norm is None


def test_a_pre_matrix_eight_checkpoint_rebuilds_as_at0():
    """No such keys, and an explicit None, both have to mean 'AT0'."""
    from dataclasses import replace

    from cancer_sbi.config import preset_from_effective_config
    from cancer_sbi.training.trainer import build_arm_token_encoder

    rebuilt = preset_from_effective_config(
        {"model": "armtoken", "flow": {}, "encoder": {"kind": "armtoken"}}
    )
    assert rebuilt.encoder.arm_feature_norm == "none"
    assert rebuilt.encoder.arm_context_norm is False

    # A snapshot that carries an explicit None -- what an older tree's unused
    # optional fields look like -- must not reach nn.LayerNorm(None).
    nulled = replace(
        get_preset("armtoken").encoder, arm_feature_norm=None, arm_context_norm=None
    )
    net = build_arm_token_encoder(nulled, "cpu")
    assert net.arm_norm is None and net.context_norm is None


def test_input_space_log2_is_accepted_by_armtoken_without_a_warning(capsys):
    """AT11: the preset's copy space, put back to the stored log2 values.

    Already true before matrix 8 -- arm_moments takes input_space and the
    warn-and-ignore gate is `!= "deepset"` -- and pinned here because AT11 is
    the one row of the matrix that undoes a preset default instead of adding
    to it, so a silent ignore would make it a duplicate of AT0.
    """
    cfg = config_from_argv(["--input-space", "log2"], model="armtoken")
    assert cfg.encoder.input_space == "log2"
    assert "--input-space" not in capsys.readouterr().out
    assert get_preset("armtoken").encoder.input_space == "copy"
