"""Work package E: the ArmToken encoder, and the property it exists for.

``ArmTokenEmbedding`` is the first encoder in this project whose architecture is
tied to the 44 per-arm coefficients the flow has to predict. Its headline claim
is an exact symmetry, not a score:

    permuting the 44 arm columns of the input permutes the 44 per-arm blocks of
    the output, and leaves the global block alone.

Sections 1-2 pin that, with a negative control (``CloneSetEmbedding`` under the
same permutation) so the assertion cannot pass vacuously. Section 3 pins the
moments themselves against hand arithmetic, section 4 the defaults, the wiring
and the proof that ``build_embedding_net`` does **not** wrap this module in
``TrialsSBIEmbedding``, and section 5 that the three published presets and
CloneAtt's forward pass are untouched.

Run from ``src/``::

    ~/miniconda3/envs/cancer/bin/python -m pytest tests/test_wp_e_armtoken.py -q
"""

import math
from pathlib import Path

import pytest
import torch

from cancer_sbi.config import ARMTOKEN, CLONEATT, get_preset
from cancer_sbi.models.arm_tokens import (
    COPY_CUT_DEEP_LOSS,
    LOG2_CUT_DEEP_LOSS,
    MAX_CONTEXT_WIDTH,
    ArmTokenEmbedding,
    arm_moments,
)
from cancer_sbi.models.set_transformer import CloneSetEmbedding
from cancer_sbi.models.trials import TrialsSBIEmbedding
from cancer_sbi.training.trainer import build_embedding_net

from tests.test_wp_c_encoders import SENTINEL, _load_pre_edit_module  # noqa: F401

CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = CODE_ROOT / "data" / "Guassian_Normal" / "simulation_outputs"

#: 44 * 8 + 64. The width every published ArmToken checkpoint's flow is built on.
EXPECTED_D_MODEL = 416


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_batch_4d():
    """A real ``(B, T, K, 45)`` batch: 2 sims x 25 trials, top_k=100.

    ``ArmTokenEmbedding`` is the whole embedding net, so unlike the clone
    encoders of work package C it is fed the un-flattened item the dataset
    yields.
    """
    if not DATA_ROOT.is_dir():
        pytest.skip(f"local simulation subsample not found at {DATA_ROOT}")

    from torch.utils.data import DataLoader

    from cancer_sbi.data.clone_sets import CNASimsDataset, discover_sim_trials

    sim_names = sorted(
        Path(p).name for p in discover_sim_trials(str(DATA_ROOT), r"^sim\d+$")
    )[:2]
    ds = CNASimsDataset(
        root_dir=str(DATA_ROOT),
        num_trials_per_sim=25,
        top_k=100,
        sim_ids=sim_names,
    )
    if len(ds) < 2:
        pytest.skip("fewer than 2 usable sims in the local subsample")

    x, _mask, _theta = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
    assert tuple(x.shape) == (2, 25, 100, 45)
    return x.contiguous()


def _synthetic_batch(batch=3, trials=5, clones=7, seed=0):
    """A small ``(B, T, K, 45)`` batch with plausible frequencies."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, trials, clones, 45, generator=gen)
    x[..., 44] = torch.rand(batch, trials, clones, generator=gen) * 0.01
    return x


def _seeded(factory, seed: int = 1234):
    """Build a module under a fixed seed and put it in eval mode."""
    torch.manual_seed(seed)
    module = factory()
    module.eval()
    return module


def _permute_arms(x, perm):
    """Permute the 44 arm columns, leaving the frequency column last."""
    out = x.clone()
    out[..., :44] = x[..., perm]
    return out


def _blocks(out, d_arm=8, n_arms=44):
    """Split a ``(B, d_model)`` output into ``(B, 44, d_arm)`` and the global tail."""
    return out[:, : n_arms * d_arm].reshape(out.shape[0], n_arms, d_arm), out[
        :, n_arms * d_arm :
    ]


# --------------------------------------------------------------------------- #
# 1. The headline: arm equivariance.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("trial_pool", ["mean", "attention"])
@pytest.mark.parametrize("n_arm_layers", [0, 1, 2])
def test_per_arm_blocks_are_equivariant(trial_pool, n_arm_layers):
    """Permute the arms in, and the 44 blocks come out permuted the same way."""
    enc = _seeded(
        lambda: ArmTokenEmbedding(
            trial_pool=trial_pool, n_arm_layers=n_arm_layers
        )
    )
    x = _synthetic_batch()
    perm = torch.randperm(44, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        out = enc(x)
        out_p = enc(_permute_arms(x, perm))

    arms, glob = _blocks(out)
    arms_p, glob_p = _blocks(out_p)

    assert torch.allclose(arms_p, arms[:, perm], atol=1e-5)
    # The global block is invariant: a PMA over the arms plus two per-sim
    # frequency scalars, neither of which can carry arm order.
    assert torch.allclose(glob_p, glob, atol=1e-5)
    # Non-vacuous: the flat outputs really are different tensors, so the
    # assertion above is a permutation and not "nothing moved".
    assert not torch.allclose(out_p, out, atol=1e-5)


def test_clone_set_embedding_is_not_arm_equivariant():
    """The negative control: the published encoder has no such property."""
    enc = _seeded(lambda: CloneSetEmbedding(input_space="copy"))
    x = _synthetic_batch(batch=2, trials=1, clones=7).reshape(2, 7, 45)
    perm = torch.randperm(44, generator=torch.Generator().manual_seed(7))

    with torch.no_grad():
        out = enc(x)
        out_p = enc(_permute_arms(x, perm))

    # Not merely "not permuted" -- CloneSetEmbedding's output has no per-arm
    # structure at all, so the only thing that can be asserted is that the
    # permutation changed it. It is the *absence* of a block structure that
    # makes this encoder unable to have the property above.
    assert not torch.allclose(out_p, out, atol=1e-5)
    assert out.shape == (2, enc.d_model)


def test_equivariance_holds_in_train_mode_with_dropout_inactive():
    """train() must not quietly break it -- no module is stochastic by default."""
    enc = _seeded(lambda: ArmTokenEmbedding())
    enc.train()
    assert enc.arm_dropout is None, "dropout is opt-in; train() must be exact"

    x = _synthetic_batch()
    perm = torch.randperm(44, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        arms, glob = _blocks(enc(x))
        arms_p, glob_p = _blocks(enc(_permute_arms(x, perm)))
    assert torch.allclose(arms_p, arms[:, perm], atol=1e-5)
    assert torch.allclose(glob_p, glob, atol=1e-5)


def test_equivariance_holds_on_a_real_batch(real_batch_4d):
    """The synthetic batches above are not the data the model is trained on."""
    enc = _seeded(lambda: ArmTokenEmbedding())
    perm = torch.randperm(44, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        arms, glob = _blocks(enc(real_batch_4d))
        arms_p, glob_p = _blocks(enc(_permute_arms(real_batch_4d, perm)))
    assert torch.allclose(arms_p, arms[:, perm], atol=1e-5)
    assert torch.allclose(glob_p, glob, atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. The other two invariances: trials and clones are sets.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("trial_pool", ["mean", "attention"])
def test_trial_order_does_not_matter(trial_pool):
    enc = _seeded(lambda: ArmTokenEmbedding(trial_pool=trial_pool))
    x = _synthetic_batch()
    perm = torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x[:, perm]), atol=1e-5)


@pytest.mark.parametrize("trial_pool", ["mean", "attention"])
def test_clone_order_does_not_matter(trial_pool):
    enc = _seeded(lambda: ArmTokenEmbedding(trial_pool=trial_pool))
    x = _synthetic_batch()
    perm = torch.randperm(x.shape[2], generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(x[:, :, perm]), atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. Shapes, finiteness and the moments themselves.
# --------------------------------------------------------------------------- #


def test_shape_and_d_model(real_batch_4d):
    enc = _seeded(lambda: ArmTokenEmbedding())
    assert enc.d_model == EXPECTED_D_MODEL
    with torch.no_grad():
        out = enc(real_batch_4d)
    assert out.shape == (real_batch_4d.shape[0], EXPECTED_D_MODEL)
    assert torch.isfinite(out).all()


def test_arm_moments_shape_on_a_real_batch(real_batch_4d):
    moments, trial_valid, glob = arm_moments(real_batch_4d)
    assert moments.shape == (real_batch_4d.shape[0], 25, 44, 8)
    assert trial_valid.shape == (real_batch_4d.shape[0], 25)
    assert glob.shape == (real_batch_4d.shape[0], 25, 2)
    assert torch.isfinite(moments).all() and torch.isfinite(glob).all()
    # The three fractions are weighted counts of a renormalised distribution.
    fractions = moments[..., 2:5]
    assert fractions.min() >= -1e-6
    assert fractions.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("trial_pool", ["mean", "attention"])
def test_no_nan_leaves_the_module_on_an_all_nan_trial(trial_pool):
    """An unfilled trial slot is NaN on disk and must be zeroed, not propagated."""
    enc = _seeded(lambda: ArmTokenEmbedding(trial_pool=trial_pool))
    x = _synthetic_batch()
    x[0, 1] = float("nan")          # a whole trial slot
    x[1, 0, 3:] = float("nan")      # clone padding inside a real trial
    moments, trial_valid, glob = arm_moments(x)
    assert torch.isfinite(moments).all() and torch.isfinite(glob).all()
    assert trial_valid[0, 1].item() is False
    assert torch.equal(moments[0, 1], torch.zeros_like(moments[0, 1]))
    assert torch.equal(glob[0, 1], torch.zeros_like(glob[0, 1]))
    with torch.no_grad():
        assert torch.isfinite(enc(x)).all()


@pytest.mark.parametrize("trial_pool", ["mean", "attention"])
def test_every_trial_invalid_is_still_finite(trial_pool):
    """A sim with no usable trial at all: the masked means must not divide by 0."""
    enc = _seeded(lambda: ArmTokenEmbedding(trial_pool=trial_pool))
    x = _synthetic_batch()
    x[0] = float("nan")
    with torch.no_grad():
        assert torch.isfinite(enc(x)).all()


@pytest.mark.parametrize("trial_pool", ["mean", "attention"])
def test_all_zero_frequency_trial_is_finite(trial_pool):
    """Every kept clone carries frequency 0: the weights have nothing to normalise."""
    enc = _seeded(lambda: ArmTokenEmbedding(trial_pool=trial_pool))
    x = _synthetic_batch()
    x[0, 0, :, 44] = 0.0
    moments, _, glob = arm_moments(x)
    assert torch.isfinite(moments).all() and torch.isfinite(glob).all()
    # The weighted statistics are all zero -- nothing is known about the mass.
    assert torch.allclose(moments[0, 0, :, 0], torch.zeros(44))
    assert torch.allclose(moments[0, 0, :, 2], torch.zeros(44))
    # ...but the presence statistics are not: the clones are still there.
    assert not torch.allclose(moments[0, 0, :, 5], torch.zeros(44))
    # log10 of the 1e-6 floor.
    assert torch.allclose(glob[0, 0], torch.full((2,), -6.0), atol=1e-6)
    with torch.no_grad():
        assert torch.isfinite(enc(x)).all()


def test_moment_zero_is_the_weighted_mean_by_hand():
    """A two-clone set, arithmetic done on paper."""
    x = torch.zeros(1, 1, 2, 45)
    # Copy space: 0.0 -> 0.0 and 1.0 -> (clamp(2**2) - 2) / 2 = 1.0.
    x[0, 0, 0, :44] = 0.0
    x[0, 0, 1, :44] = 1.0
    x[0, 0, :, 44] = torch.tensor([0.003, 0.001])

    moments, _, glob = arm_moments(x, input_space="copy")
    w1 = 0.001 / 0.004
    assert torch.allclose(moments[0, 0, :, 0], torch.full((44,), w1), atol=1e-6)
    # sd of a two-point distribution: sqrt(w0 * w1) * |x1 - x0|.
    expected_sd = math.sqrt((1 - w1) * w1)
    assert torch.allclose(moments[0, 0, :, 1], torch.full((44,), expected_sd), atol=1e-5)
    # Fractions: nothing lost, everything above +0.25 is the 1.0 clone.
    assert torch.allclose(moments[0, 0, :, 2], torch.zeros(44), atol=1e-6)
    assert torch.allclose(moments[0, 0, :, 3], torch.full((44,), w1), atol=1e-6)
    assert torch.allclose(moments[0, 0, :, 5], torch.ones(44), atol=1e-6)   # max
    assert torch.allclose(moments[0, 0, :, 6], torch.zeros(44), atol=1e-6)  # min
    # The dominant clone is the 0.003 one, whose copy number is 0.
    assert torch.allclose(moments[0, 0, :, 7], torch.zeros(44), atol=1e-6)
    assert torch.allclose(
        glob[0, 0], torch.tensor([math.log10(0.004), math.log10(0.003)]), atol=1e-6
    )


def test_sentinel_column_is_deeply_lost():
    """-10.966 is "arm completely lost"; it must read as a deep loss."""
    x = torch.zeros(1, 1, 1, 45)
    x[0, 0, 0, :44] = SENTINEL
    x[0, 0, 0, 44] = 0.5

    moments, _, _ = arm_moments(x, input_space="copy")
    # (clamp(2 ** -9.966, 0, 8) - 2) / 2 == -0.99950..., i.e. -1.0 to 4 s.f.
    # Exactly -1.0 is the limit, not the value: the clamp's lower end is 0 and
    # 2 ** -9.966 is ~1e-3 above it.
    assert torch.allclose(moments[0, 0, :, 0], torch.full((44,), -1.0), atol=1e-3)
    assert moments[0, 0, 0, 0].item() < COPY_CUT_DEEP_LOSS
    # ...and the deep-loss fraction is the whole population.
    assert torch.allclose(moments[0, 0, :, 4], torch.ones(44), atol=1e-6)


def test_log2_cut_points_ask_the_same_question():
    """The three thresholds move with the space, so the fractions do not."""
    # -2.0 is the exact log2 pre-image of the -0.75 copy-space cut.
    assert math.isclose(LOG2_CUT_DEEP_LOSS, -2.0, abs_tol=1e-12)

    x = _synthetic_batch()
    copy_m, _, _ = arm_moments(x, input_space="copy")
    log2_m, _, _ = arm_moments(x, input_space="log2")
    # Moments 2-4 are threshold counts and are therefore identical; 0, 1, 5-7
    # are values and are not, because the two spaces are different scales.
    assert torch.allclose(copy_m[..., 2:5], log2_m[..., 2:5], atol=1e-6)
    assert not torch.allclose(copy_m[..., 0], log2_m[..., 0], atol=1e-3)


def test_input_space_is_validated():
    with pytest.raises(ValueError, match="input_space"):
        arm_moments(_synthetic_batch(), input_space="copies")
    with pytest.raises(ValueError, match="input_space"):
        ArmTokenEmbedding(input_space="copies")


# --------------------------------------------------------------------------- #
# 4. Defaults, wiring and determinism.
# --------------------------------------------------------------------------- #


def test_armtoken_preset_fields():
    enc = ARMTOKEN.encoder
    assert (ARMTOKEN.name, ARMTOKEN.paper_name) == ("armtoken", "ArmToken-NPE")
    assert enc.kind == "armtoken"
    assert enc.in_dim == 45
    assert (enc.d_token, enc.d_arm, enc.d_global) == (64, 8, 64)
    assert (enc.n_arm_layers, enc.arm_num_inducing, enc.n_heads) == (1, 16, 4)
    assert (enc.trial_pool, enc.input_space) == ("mean", "copy")
    assert enc.attn_ln is True
    assert enc.dropout == 0.2
    assert enc.attn_dropout_active is False
    assert enc.attn_scale == "published"
    # It is the embedding net itself; there is no wrapper to configure.
    assert enc.trials_aggregation_fn is None
    assert enc.trials_num_hiddens is None
    assert enc.trials_num_layers is None
    assert enc.trials_output_dim is None
    assert enc.trials_aggregation_dim is None
    # `d_model` is CloneMLP's and CloneAtt's per-trial width and means nothing
    # here -- the output width is 44 * d_arm + d_global.
    assert enc.d_model is None
    # The flow block: matrices 2-4's findings, and the width that never moves.
    assert ARMTOKEN.flow.z_score_x == "structured"
    assert ARMTOKEN.flow.z_score_y == CLONEATT.flow.z_score_y
    assert ARMTOKEN.flow.num_transforms == 3
    assert ARMTOKEN.flow.dropout_probability == 0.2
    assert ARMTOKEN.flow.hidden_features == 50, "must stay 50"
    # Data, optim and train are CloneAtt's, so a matrix-5 run is read against
    # matrix 4 on everything except the encoder.
    assert ARMTOKEN.data.top_k == CLONEATT.data.top_k
    assert ARMTOKEN.data.batch_size == CLONEATT.data.batch_size
    assert ARMTOKEN.optim == CLONEATT.optim
    assert ARMTOKEN.train.reload_best == CLONEATT.train.reload_best
    assert ARMTOKEN.train.enforce_min_epochs == CLONEATT.train.enforce_min_epochs


def test_build_embedding_net_returns_the_bare_module():
    """NOT wrapped: TrialsSBIEmbedding's MLP would blend the per-arm blocks."""
    net = build_embedding_net(ARMTOKEN.encoder, "cpu")
    assert isinstance(net, ArmTokenEmbedding)
    assert not isinstance(net, TrialsSBIEmbedding)
    assert net.d_model == EXPECTED_D_MODEL
    assert net.trial_pool == "mean"
    assert net.input_space == "copy"
    assert net.attn_ln is True
    assert net.attn_scale == "published"
    assert net.arm_dropout is None
    assert len(net.arm_layers) == 1
    assert net.arm_layers[0].I.shape == (1, 16, 64)
    assert net.arm_head.out_features == 8
    assert net.global_head.out_features == 64
    # The mean path has no moment projection or trial PMA at all: an unused
    # submodule would put keys in every checkpoint that nothing reads.
    assert net.moment_proj is None and net.trial_pma is None


def test_attention_pool_path_builds_its_two_extra_modules():
    from dataclasses import replace

    cfg = replace(ARMTOKEN.encoder, trial_pool="attention")
    net = build_embedding_net(cfg, "cpu")
    assert net.moment_proj is not None and net.trial_pma is not None
    assert net.moment_proj.in_features == 8
    assert net.arm_mlp[0].in_features == 64
    # The mean path's MLP sees [mean, sd] of the 8 moments instead.
    assert build_embedding_net(ARMTOKEN.encoder, "cpu").arm_mlp[0].in_features == 16


def test_parameter_count_is_small():
    net = build_embedding_net(ARMTOKEN.encoder, "cpu")
    total = sum(p.numel() for p in net.parameters())
    assert 55_000 <= total <= 70_000, total
    cloneatt = sum(
        p.numel()
        for p in build_embedding_net(get_preset("cloneatt").encoder, "cpu").parameters()
    )
    assert total < 2 * cloneatt


def test_seeded_construction_is_deterministic():
    a = _seeded(lambda: ArmTokenEmbedding())
    b = _seeded(lambda: ArmTokenEmbedding())
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)
    x = _synthetic_batch()
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_the_context_cap_is_enforced_in_the_module():
    with pytest.raises(ValueError, match=str(MAX_CONTEXT_WIDTH)):
        ArmTokenEmbedding(d_arm=16)
    # The published pair is comfortably inside it.
    assert 44 * 8 + 64 <= MAX_CONTEXT_WIDTH


def test_indivisible_d_token_and_n_heads_is_refused():
    with pytest.raises(ValueError, match="not divisible"):
        ArmTokenEmbedding(d_token=66, n_heads=4)


def test_trial_pool_is_validated():
    with pytest.raises(ValueError, match="trial_pool"):
        ArmTokenEmbedding(trial_pool="max")


def test_dropout_is_opt_in():
    assert ArmTokenEmbedding().arm_dropout is None
    enc = ArmTokenEmbedding(attn_dropout_active=True, dropout=0.3)
    assert enc.arm_dropout is not None and enc.arm_dropout.p == 0.3


def test_arm_layers_zero_has_no_isab():
    net = ArmTokenEmbedding(n_arm_layers=0)
    assert len(net.arm_layers) == 0
    with torch.no_grad():
        assert net(_synthetic_batch()).shape == (3, EXPECTED_D_MODEL)


# --------------------------------------------------------------------------- #
# 5. Nothing published moved.
# --------------------------------------------------------------------------- #


def test_the_three_published_presets_are_untouched():
    """Adding a fourth preset must not have edited the other three."""
    assert get_preset("clonemlp").encoder.kind == "mlp"
    assert get_preset("cloneatt").encoder.kind == "attention"
    assert get_preset("dominantclone").encoder.kind == "deepset"
    for name in ("clonemlp", "cloneatt", "dominantclone"):
        enc = get_preset(name).encoder
        # The five new fields must be None on every preset that is not ArmToken:
        # a value there would describe a network nothing builds.
        assert enc.d_token is None
        assert enc.d_arm is None
        assert enc.d_global is None
        assert enc.n_arm_layers is None
        assert enc.arm_num_inducing is None
    assert get_preset("cloneatt").flow.z_score_x == "none"
    assert get_preset("cloneatt").encoder.attn_ln is False   # trap 5


def test_cloneatt_forward_is_byte_identical_to_head(tmp_path):
    """The pre-edit module, imported from HEAD and run on the same input."""
    old = _load_pre_edit_module(
        "src/cancer_sbi/models/set_transformer.py",
        "_wpe_old_set_transformer",
        tmp_path,
    )
    cfg = get_preset("cloneatt").encoder
    kwargs = dict(
        in_dim=cfg.in_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_layers=cfg.num_layers,
        num_inducing=cfg.num_inducing,
        dropout=cfg.dropout,
    )
    new_enc = _seeded(lambda: CloneSetEmbedding(**kwargs))
    old_enc = _seeded(lambda: old.CloneSetEmbedding(**kwargs))

    x = _synthetic_batch(batch=2, trials=1, clones=9).reshape(2, 9, 45)
    with torch.no_grad():
        assert torch.equal(new_enc(x), old_enc(x))
