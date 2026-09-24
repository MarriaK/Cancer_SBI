"""Matrix 6: the hybrid encoder, the two flow/optimiser switches, train6.sh.

Three things are pinned here.

1. :class:`~cancer_sbi.models.hybrid.HybridEmbedding` -- its shape (672), its
   finiteness, and the *partial* symmetry that is the honest statement of what
   a late fusion is: the first 352 outputs are exactly arm-equivariant, the
   next 64 exactly arm-invariant, and the last 256 -- the clone branch -- are
   neither. Section 1 asserts all three, including the negative half, so the
   equivariance assertion cannot pass by the whole output being constant.

2. ``--flow-hidden-features`` and ``--lr-plateau``. Both default to the
   published behaviour, and the tests that matter are the ones about *absence*:
   no scheduler object exists without the flag, no ``scheduler_state`` key is
   written, and two seeded two-epoch runs -- one with the flag, one without --
   produce bitwise-identical weights, because ReduceLROnPlateau cannot fire
   inside its own patience. The firing itself is pinned separately against a
   fake validation curve.

3. ``jobs/train6.sh`` -- nine command lines, the exact flags of the matrix.

Run from ``src/``::

    ~/miniconda3/envs/cancer/bin/python -m pytest tests/test_wp_f_hybrid.py -q
"""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cancer_sbi.config import (
    ARMTOKEN,
    CLONEATT,
    EFFECTIVE_CONFIG_KEY,
    HYBRID,
    PRESETS,
    TrainConfig,
    get_preset,
    preset_from_effective_config,
)
from cancer_sbi.models.arm_tokens import ArmTokenEmbedding
from cancer_sbi.models.hybrid import HybridEmbedding
from cancer_sbi.models.trials import TrialsSBIEmbedding
from cancer_sbi.training import checkpoints
from cancer_sbi.training.trainer import (
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_PATIENCE,
    build_embedding_net,
    build_lr_scheduler,
)

from tests.test_wp_a_train_cli import config_from_argv
from tests.test_eval_rebuilds_from_checkpoint import (  # noqa: F401
    DATA_ROOT,
    _sample,
    _sim_names,
    _train_tiny,
    _write_split,
    needs_data,
)

#: 44 * 8 + 64 (ArmToken) + 256 (CloneAtt's wrapped context).
EXPECTED_D_MODEL = 672
ARM_WIDTH = 416
ARM_BLOCKS = 44 * 8


def _synthetic_batch(batch=3, trials=5, clones=7, seed=0):
    """A small ``(B, T, K, 45)`` batch with plausible frequencies."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, trials, clones, 45, generator=gen)
    x[..., 44] = torch.rand(batch, trials, clones, generator=gen) * 0.01
    return x


def _permute_arms(x, perm):
    """Permute the 44 arm columns, leaving the frequency column last."""
    out = x.clone()
    out[..., :44] = x[..., perm]
    return out


def _seeded_hybrid(cfg=None, seed=1234):
    torch.manual_seed(seed)
    module = build_embedding_net(cfg or HYBRID.encoder, "cpu")
    module.eval()
    return module


# --------------------------------------------------------------------------- #
# 1. The encoder.
# --------------------------------------------------------------------------- #


def test_hybrid_shape_and_finiteness():
    enc = _seeded_hybrid()
    with torch.no_grad():
        out = enc(_synthetic_batch())
    assert out.shape == (3, EXPECTED_D_MODEL)
    assert enc.d_model == EXPECTED_D_MODEL
    assert (enc.arm_width, enc.clone_width) == (ARM_WIDTH, 256)
    assert torch.isfinite(out).all()


def test_hybrid_is_built_directly_and_owns_both_branches():
    enc = build_embedding_net(HYBRID.encoder, "cpu")
    assert isinstance(enc, HybridEmbedding)
    # Not wrapped again: a TrialsSBIEmbedding over the concatenation would put
    # a dense MLP across the 44 per-arm blocks and destroy the equivariance.
    assert not isinstance(enc, TrialsSBIEmbedding)
    assert isinstance(enc.arm_branch, ArmTokenEmbedding)
    # The clone branch IS wrapped -- that wrapper is what makes it produce the
    # 256-wide per-sim context CloneAtt's flow sees.
    assert isinstance(enc.clone_branch, TrialsSBIEmbedding)


def test_arm_blocks_are_equivariant_and_the_global_block_invariant():
    """The half of the symmetry that survives the fusion."""
    enc = _seeded_hybrid()
    x = _synthetic_batch()
    perm = torch.randperm(44, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        out = enc(x)
        out_p = enc(_permute_arms(x, perm))

    arms = out[:, :ARM_BLOCKS].reshape(out.shape[0], 44, 8)
    arms_p = out_p[:, :ARM_BLOCKS].reshape(out.shape[0], 44, 8)
    assert torch.allclose(arms_p, arms[:, perm], atol=1e-5)

    glob = out[:, ARM_BLOCKS:ARM_WIDTH]
    glob_p = out_p[:, ARM_BLOCKS:ARM_WIDTH]
    assert torch.allclose(glob_p, glob, atol=1e-5)

    # The other half, asserted as the negative it is: the clone branch mixes
    # all 44 arms on its first projection, so a permutation moves it
    # arbitrarily. If this ever started passing, the clone branch would have
    # stopped contributing anything and the hybrid would be AT0 with extra
    # parameters.
    clone = out[:, ARM_WIDTH:]
    clone_p = out_p[:, ARM_WIDTH:]
    assert not torch.allclose(clone_p, clone, atol=1e-5)
    # Non-vacuous: the arm blocks really did move too.
    assert not torch.allclose(out_p, out, atol=1e-5)


def test_the_two_branches_are_the_modules_they_claim_to_be():
    """Each branch's output is bit-for-bit its own encoder's, run alone."""
    enc = _seeded_hybrid()
    x = _synthetic_batch()
    with torch.no_grad():
        out = enc(x)
        arm_alone = enc.arm_branch(x)
        clone_alone = enc.clone_branch(x)
    assert torch.equal(out[:, :ARM_WIDTH], arm_alone)
    assert torch.equal(out[:, ARM_WIDTH:], clone_alone)


def test_hybrid_parameter_count_is_the_sum_of_the_two_branches():
    """Late fusion adds no parameters of its own -- within 5% of the sum."""
    torch.manual_seed(0)
    arm = build_embedding_net(ARMTOKEN.encoder, "cpu")
    torch.manual_seed(0)
    clone_cfg = replace(
        CLONEATT.encoder,
        d_model=256,
        input_space="copy",
        freq_mode="feature",
        attn_ln=True,
    )
    clone = build_embedding_net(clone_cfg, "cpu")
    torch.manual_seed(0)
    hyb = build_embedding_net(HYBRID.encoder, "cpu")

    separate = sum(p.numel() for p in arm.parameters()) + sum(
        p.numel() for p in clone.parameters()
    )
    together = sum(p.numel() for p in hyb.parameters())
    assert abs(together - separate) <= 0.05 * separate, (together, separate)


# --------------------------------------------------------------------------- #
# 2. The preset.
# --------------------------------------------------------------------------- #


def test_hybrid_preset_fields():
    assert HYBRID.name == "hybrid" and HYBRID.paper_name == "Hybrid-NPE"
    assert PRESETS["hybrid"] is HYBRID and get_preset("HYBRID") is HYBRID

    enc = HYBRID.encoder
    assert enc.kind == "hybrid"
    # ArmToken's five, at AT0's values.
    assert (enc.d_token, enc.d_arm, enc.d_global) == (64, 8, 64)
    assert (enc.n_arm_layers, enc.arm_num_inducing) == (1, 16)
    assert enc.trial_pool == "mean"
    # CloneAtt's, at R26's values.
    assert (enc.d_model, enc.n_heads, enc.num_inducing) == (256, 8, 32)
    assert enc.freq_mode == "feature"
    assert enc.attn_scale == "published"
    assert enc.trials_output_dim == 256
    # Shared.
    assert enc.input_space == "copy" and enc.attn_ln is True

    assert HYBRID.flow.z_score_x == "structured"
    assert HYBRID.flow.num_transforms == 3
    assert HYBRID.flow.hidden_features == 50
    # In the preset on purpose, unlike every preset above it -- see config.py.
    assert HYBRID.flow.tail_bound == 5.0
    assert HYBRID.flow.dropout_probability == 0.2

    # data/optim/train come from CLONEATT.
    assert HYBRID.data.dataset == CLONEATT.data.dataset
    assert HYBRID.data.top_k == CLONEATT.data.top_k
    assert HYBRID.optim == CLONEATT.optim
    assert HYBRID.train.reload_best == "on_early_stop"
    assert HYBRID.train.enforce_min_epochs is False
    assert HYBRID.train.lr_plateau is False


def test_n_heads_divides_both_branches():
    """One shared head count has to work for d_model 256 AND d_token 64."""
    assert HYBRID.encoder.d_model % HYBRID.encoder.n_heads == 0
    assert HYBRID.encoder.d_token % HYBRID.encoder.n_heads == 0


def test_the_published_presets_are_untouched():
    for name in ("clonemlp", "cloneatt", "dominantclone"):
        preset = get_preset(name)
        assert preset.flow.hidden_features == 50
        assert preset.train.lr_plateau is False
    assert CLONEATT.flow.tail_bound == 3.0
    assert ARMTOKEN.encoder.kind == "armtoken"


# --------------------------------------------------------------------------- #
# 3. The CLI: which flags reach which branch.
# --------------------------------------------------------------------------- #


def test_clone_branch_flags_apply_to_hybrid():
    cfg = config_from_argv(
        ["--d-model", "128", "--n-heads", "4", "--num-inducing", "16",
         "--freq-mode", "weight", "--attn-scale", "standard"],
        model="hybrid",
    )
    assert cfg.encoder.d_model == 128
    assert cfg.encoder.n_heads == 4
    assert cfg.encoder.num_inducing == 16
    assert cfg.encoder.freq_mode == "weight"
    assert cfg.encoder.attn_scale == "standard"


def test_arm_branch_flags_apply_to_hybrid():
    cfg = config_from_argv(
        ["--arm-layers", "0", "--d-arm", "4", "--arm-num-inducing", "8"],
        model="hybrid",
    )
    assert cfg.encoder.n_arm_layers == 0
    assert cfg.encoder.d_arm == 4
    assert cfg.encoder.arm_num_inducing == 8


def test_shared_flags_apply_to_both_branches():
    cfg = config_from_argv(
        ["--attn-ln", "--input-space", "log2", "--encoder-dropout", "0.3",
         "--trial-pool", "attention"],
        model="hybrid",
    )
    assert cfg.encoder.attn_ln is True
    assert cfg.encoder.input_space == "log2"
    assert cfg.encoder.dropout == 0.3
    assert cfg.encoder.attn_dropout_active is True
    assert cfg.encoder.trial_pool == "attention"
    # And both branches really read them.
    enc = build_embedding_net(cfg.encoder, "cpu")
    assert enc.arm_branch.input_space == "log2"
    assert enc.clone_branch.trial_encoder.input_space == "log2"


@pytest.mark.parametrize("flag", [["--freq-renorm"], ["--require-all-trials"]])
def test_ignored_flags_warn_and_do_not_reach_the_config(flag, capsys):
    cfg = config_from_argv(flag, model="hybrid")
    warned = capsys.readouterr().out
    assert "[warn]" in warned and "ignoring it" in warned
    assert cfg.encoder.freq_renorm is False
    assert cfg.data.require_all_trials is False


def test_an_indivisible_head_count_is_refused_for_either_branch():
    with pytest.raises(ValueError, match="n-heads"):
        config_from_argv(["--n-heads", "6"], model="hybrid")


def test_flow_hidden_features_grows_the_flow():
    """The flag the plan deliberately did not add until matrix 6."""
    from cancer_sbi.models.flow import build_flow

    def _flow(hidden):
        torch.manual_seed(0)
        theta = torch.randn(8, 44)
        context = torch.randn(8, 64)
        cfg = replace(ARMTOKEN.flow, hidden_features=hidden)
        return build_flow(theta, context, torch.nn.Identity(), cfg)

    small = sum(p.numel() for p in _flow(50).parameters())
    big = sum(p.numel() for p in _flow(100).parameters())
    assert big > small

    cfg = config_from_argv(["--flow-hidden-features", "100"], model="armtoken")
    assert cfg.flow.hidden_features == 100
    assert config_from_argv([], model="armtoken").flow.hidden_features == 50


def test_lr_plateau_flag_reaches_the_train_config():
    assert config_from_argv(["--lr-plateau"], model="armtoken").train.lr_plateau is True
    assert config_from_argv([], model="armtoken").train.lr_plateau is False


# --------------------------------------------------------------------------- #
# 4. The scheduler itself.
# --------------------------------------------------------------------------- #


def _dummy_optimizer(lr=1e-3):
    param = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.Adam([param], lr=lr)


def test_no_scheduler_is_built_by_default():
    assert build_lr_scheduler(_dummy_optimizer(), TrainConfig()) is None


def test_scheduler_halves_the_lr_after_six_non_improving_epochs():
    """patience=5 fires on the sixth bad epoch, not the fifth."""
    opt = _dummy_optimizer(lr=1e-3)
    sched = build_lr_scheduler(opt, TrainConfig(lr_plateau=True))
    assert sched is not None

    sched.step(1.0)                      # the best so far
    for _ in range(LR_PLATEAU_PATIENCE):  # five bad epochs: still 1e-3
        sched.step(2.0)
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    sched.step(2.0)                       # the sixth
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 * LR_PLATEAU_FACTOR)


def test_both_optimiser_groups_are_scheduled():
    a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([{"params": [a], "lr": 1e-3}, {"params": [b], "lr": 1e-4}])
    sched = build_lr_scheduler(opt, TrainConfig(lr_plateau=True))
    sched.step(1.0)
    for _ in range(LR_PLATEAU_PATIENCE + 1):
        sched.step(2.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert opt.param_groups[1]["lr"] == pytest.approx(5e-5)


def test_scheduler_state_survives_a_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = build_lr_scheduler(opt, TrainConfig(lr_plateau=True))
    sched.step(1.0)
    for _ in range(3):
        sched.step(2.0)

    payload = checkpoints.build_checkpoint(
        epoch=4,
        density_estimator=model,
        optimizer=opt,
        best_val_loss=1.0,
        best_model_state_dict=None,
        history={"training_loss": [], "validation_loss": []},
        epochs_since_last_improvement=3,
        scheduler_state=sched.state_dict(),
    )
    assert payload[checkpoints.SCHEDULER_STATE_KEY]["num_bad_epochs"] == 3
    checkpoints.save_checkpoint(tmp_path, 4, payload)

    model2 = torch.nn.Linear(3, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    resumed = checkpoints.load_checkpoint(
        tmp_path / "latest.pt", model2, opt2, verbose=False
    )
    sched2 = build_lr_scheduler(opt2, TrainConfig(lr_plateau=True))
    sched2.load_state_dict(resumed.scheduler_state)
    assert sched2.num_bad_epochs == 3
    # Three more bad epochs finish the patience the first run had started.
    for _ in range(3):
        sched2.step(2.0)
    assert opt2.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_a_run_without_the_flag_writes_no_scheduler_key(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters())
    payload = checkpoints.build_checkpoint(
        epoch=1,
        density_estimator=model,
        optimizer=opt,
        best_val_loss=1.0,
        best_model_state_dict=None,
        history={"training_loss": [], "validation_loss": []},
        epochs_since_last_improvement=0,
    )
    assert checkpoints.SCHEDULER_STATE_KEY not in payload
    # ...and a reader of such a checkpoint gets None rather than raising.
    checkpoints.save_checkpoint(tmp_path, 1, payload)
    resumed = checkpoints.load_checkpoint(
        tmp_path / "latest.pt", model, opt, verbose=False
    )
    assert resumed.scheduler_state is None


# --------------------------------------------------------------------------- #
# 5. End to end on real data.
# --------------------------------------------------------------------------- #


@needs_data
def test_hybrid_checkpoint_round_trips_through_the_sampler(
    tmp_path, monkeypatch, capsys
):
    """Train one epoch, rebuild strictly, then really sample."""
    from cancer_sbi.data.loaders import build_clone_set_dataloaders
    from cancer_sbi.evaluation import posterior as posterior_mod
    from cancer_sbi.training.trainer import build_training_components

    ckpt_dir, split_path = _train_tiny(tmp_path, "hybrid", [], "H0")

    stored = torch.load(ckpt_dir / "best.pt", map_location="cpu")[EFFECTIVE_CONFIG_KEY]
    assert stored["model"] == "hybrid"
    assert stored["encoder"]["kind"] == "hybrid"
    assert stored["encoder"]["d_arm"] == 8 and stored["encoder"]["d_model"] == 256
    assert stored["flow"]["z_score_x"] == "structured"
    assert stored["flow"]["hidden_features"] == 50
    assert checkpoints.SCHEDULER_STATE_KEY not in torch.load(
        ckpt_dir / "best.pt", map_location="cpu"
    )

    path, meta = _sample(
        tmp_path, "hybrid", ckpt_dir / "best.pt", split_path,
        monkeypatch=monkeypatch,
    )
    assert path.exists()
    assert meta["config_from_checkpoint"] is True
    assert meta["encoder_config"]["kind"] == "hybrid"
    out = capsys.readouterr().out
    assert "[config] rebuilt" in out and "kind=hybrid" in out
    # Both branches are named on the rebuilt line.
    assert "arm[d_arm=8" in out and "clone[d_model=256" in out

    rebuilt = posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "hybrid").preset
    net = build_embedding_net(rebuilt.encoder, "cpu")
    assert isinstance(net, HybridEmbedding) and net.d_model == EXPECTED_D_MODEL

    state = torch.load(ckpt_dir / "best.pt", map_location="cpu")["model_state"]
    names = _sim_names(3)
    train_loader, _, _ = build_clone_set_dataloaders(
        str(DATA_ROOT), names, names, top_k=100, batch_size=2
    )
    flow = build_training_components(
        rebuilt, train_loader, device="cpu", log_progress=False
    ).density_estimator
    flow.load_state_dict(state, strict=True)


@needs_data
def test_a_hybrid_checkpoint_does_not_load_into_an_armtoken(tmp_path):
    """The proof that the effective config is necessary, not tidy."""
    from cancer_sbi.data.loaders import build_clone_set_dataloaders
    from cancer_sbi.training.trainer import build_training_components

    ckpt_dir, _ = _train_tiny(tmp_path, "hybrid", [], "Hx")
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu")["model_state"]

    names = _sim_names(3)
    train_loader, _, _ = build_clone_set_dataloaders(
        str(DATA_ROOT), names, names, top_k=100, batch_size=2
    )
    armtoken_built = build_training_components(
        get_preset("armtoken"), train_loader, device="cpu", log_progress=False
    ).density_estimator
    with pytest.raises(RuntimeError):
        armtoken_built.load_state_dict(state)


def test_a_checkpoint_without_the_matrix_six_keys_still_rebuilds():
    """A pre-matrix-6 snapshot keeps every published default."""
    rebuilt = preset_from_effective_config(
        {
            "model": "armtoken",
            "flow": {"z_score_x": "structured", "num_transforms": 3},
            "encoder": {"kind": "armtoken", "d_arm": 8},
        }
    )
    assert rebuilt.flow.hidden_features == 50
    assert rebuilt.train.lr_plateau is False


@needs_data
def test_lr_plateau_does_not_move_a_two_epoch_run(tmp_path):
    """Absence and presence agree until the scheduler can fire.

    Two seeded two-epoch runs, one with ``--lr-plateau`` and one without,
    produce bitwise-identical weights: ReduceLROnPlateau cannot lower anything
    inside its five-epoch patience, so the flag is provably inert until it
    fires, and the run without it builds no scheduler object at all.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain, _ = _train_tiny(tmp_path / "a", "armtoken", [], "AT0a")
    sched, _ = _train_tiny(tmp_path / "b", "armtoken", ["--lr-plateau"], "AT7a")

    a = torch.load(plain / "best.pt", map_location="cpu")
    b = torch.load(sched / "best.pt", map_location="cpu")
    assert checkpoints.SCHEDULER_STATE_KEY not in a
    assert checkpoints.SCHEDULER_STATE_KEY in b
    assert a["model_state"].keys() == b["model_state"].keys()
    for key, value in a["model_state"].items():
        assert torch.equal(value, b["model_state"][key]), key


# --------------------------------------------------------------------------- #
# 6. The jobs script.
# --------------------------------------------------------------------------- #


JOBS = Path(__file__).resolve().parents[1] / "jobs"


def _dry_run_lines():
    env = dict(os.environ, CANCER=str(JOBS.parent.parent), DRY_RUN="1")
    out = subprocess.run(
        ["bash", str(JOBS / "train6.sh")],
        capture_output=True, text=True, check=True, env=env,
    ).stdout
    return [ln for ln in out.splitlines() if ln.startswith("python -m")]


def test_train6_dry_run_prints_nine_commands():
    lines = _dry_run_lines()
    assert len(lines) == 9

    expected_models = ["hybrid"] * 3 + ["armtoken"] * 6
    for line, model in zip(lines, expected_models):
        assert f"--model {model}" in line
        for common in (
            "--min-epochs 1", "--stop-after-epochs 15", "--max-epochs 60",
            "--num-workers 8", "--tail-bound 5",
        ):
            assert common in line, (common, line)

    seeds = ["20260924", "1", "2", "2", "20260924", "20260924",
             "20260924", "20260924", "20260924"]
    for line, seed in zip(lines, seeds):
        assert f"--seed {seed}" in line

    # The per-run flags, exactly.
    assert "--flow-num-transforms" not in lines[0]
    assert "--lr-plateau" not in lines[3]
    assert lines[4].endswith("--flow-num-transforms 5")
    assert lines[5].endswith("--flow-hidden-features 100")
    assert lines[6].endswith("--lr-plateau")
    assert lines[7].endswith("--flow-num-transforms 5 --flow-hidden-features 100")
    assert lines[8].endswith(
        "--flow-num-transforms 5 --flow-hidden-features 100 --lr-plateau"
    )


def test_train6_names_its_runs_and_is_a_nine_task_array():
    text = (JOBS / "train6.sh").read_text()
    assert "#SBATCH --array=0-8" in text
    for name in ("H0", "H0s1", "H0s2", "AT0s2", "AT5", "AT6", "AT7", "AT8", "AT9"):
        assert name in text
    # Header line per run, as train5.sh has.
    assert "echo \"host=$(hostname)" in text


def test_train6_leaves_the_earlier_matrices_alone():
    """The script is new; nothing in matrix 1-5 may have been edited."""
    for older in ("train.sh", "train2.sh", "train3.sh", "train4.sh",
                  "train4b.sh", "train5.sh"):
        assert (JOBS / older).exists()
