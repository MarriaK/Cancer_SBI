"""Work package C: the two encoder repairs, and proof they are opt-in.

Covers the two constructor flags added to the clone-set encoders:

* ``BaselineCloneEmbedding(input_space=...)`` -- repair T2 / run R2, the
  log2-to-copy-space conversion of the 44 CNA columns
  (``docs/MODEL_IMPROVEMENT_PLAN.md`` §5 step 5b).
* ``CloneSetEmbedding(freq_renorm=...)`` -- repair R4, renormalising the masked
  clone frequencies to sum to 1 before the token multiply (§5 step 5c). ``ln``
  stays ``False``; this deliberately tests only one half of the architecture
  review's T3.

The equivalence tests do not *assume* the defaults are untouched: they fetch the
pre-edit modules with ``git show HEAD:<path>``, import them under throwaway
names, and compare parameters and a seeded forward pass on a real batch from the
local ``data/Guassian_Normal/simulation_outputs`` subsample.

Run from ``src/``::

    ~/miniconda3/envs/cancer/bin/python -m pytest tests/test_wp_c_encoders.py -v
"""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cancer_sbi.models import set_transformer as new_st
from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding
from cancer_sbi.models.set_transformer import CloneSetEmbedding

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "Guassian_Normal" / "simulation_outputs"

SENTINEL = -10.966  # "arm completely lost" in log2 space


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _load_pre_edit_module(rel_path: str, mod_name: str, tmp_dir: Path):
    """Import the HEAD version of a model module under a throwaway name."""
    src = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = tmp_dir / f"{mod_name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def old_mlp_mod(tmp_path_factory):
    return _load_pre_edit_module(
        "src/cancer_sbi/models/mlp_encoder.py",
        "_wpc_old_mlp_encoder",
        tmp_path_factory.mktemp("pre_edit_mlp"),
    )


@pytest.fixture(scope="module")
def old_st_mod(tmp_path_factory):
    return _load_pre_edit_module(
        "src/cancer_sbi/models/set_transformer.py",
        "_wpc_old_set_transformer",
        tmp_path_factory.mktemp("pre_edit_st"),
    )


@pytest.fixture(scope="module")
def real_batch():
    """A real ``(B, K, 45)`` batch: 2 sims x 25 trials, top_k=100, flattened.

    The flattening mirrors ``TrialsSBIEmbedding.forward``
    (``models/trials.py:119``), which is the only thing that ever calls these
    encoders.
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
    b, t, k, f = x.shape
    assert (b, t, k, f) == (2, 25, 100, 45)
    return x.reshape(b * t, k, f).contiguous()


def _seeded(factory, seed: int = 1234):
    """Build a module under a fixed seed and put it in eval mode."""
    torch.manual_seed(seed)
    module = factory()
    module.eval()
    return module


def _param_spec(module):
    return {name: tuple(p.shape) for name, p in module.named_parameters()}


# --------------------------------------------------------------------------- #
# 1. Defaults are byte-identical to the pre-edit code
# --------------------------------------------------------------------------- #


def test_mlp_default_matches_pre_edit(old_mlp_mod, real_batch):
    old = _seeded(old_mlp_mod.BaselineCloneEmbedding)
    new = _seeded(BaselineCloneEmbedding)

    assert _param_spec(new) == _param_spec(old)
    for name, p in new.named_parameters():
        assert torch.equal(p, dict(old.named_parameters())[name]), name

    with torch.no_grad():
        out_old = old(real_batch)
        out_new = new(real_batch)

    assert out_new.shape == out_old.shape
    assert torch.equal(out_new, out_old), "default forward pass is not bitwise equal"


def test_mlp_default_flag_value():
    assert BaselineCloneEmbedding().input_space == "log2"
    with pytest.raises(ValueError):
        BaselineCloneEmbedding(input_space="copies")


def test_set_transformer_default_matches_pre_edit(old_st_mod, real_batch):
    old = _seeded(old_st_mod.CloneSetEmbedding)
    new = _seeded(CloneSetEmbedding)

    assert _param_spec(new) == _param_spec(old)
    for name, p in new.named_parameters():
        assert torch.equal(p, dict(old.named_parameters())[name]), name

    with torch.no_grad():
        out_old = old(real_batch)
        out_new = new(real_batch)

    assert out_new.shape == out_old.shape
    assert torch.equal(out_new, out_old), "default forward pass is not bitwise equal"


def test_set_transformer_defaults_unchanged():
    enc = CloneSetEmbedding()
    assert enc.freq_renorm is False
    # ln stays False everywhere: trap 5 means the LayerNorms are simply absent.
    for layer in enc.layers:
        for mab in (layer.mab0, layer.mab1):
            assert getattr(mab, "ln0", None) is None
            assert getattr(mab, "ln1", None) is None
    assert getattr(enc.pma.mab, "ln0", None) is None
    assert getattr(enc.pma.mab, "ln1", None) is None
    # in_dim is still ignored and the projection still hard-codes 44 inputs.
    assert CloneSetEmbedding(in_dim=99).input_proj.in_features == 44


# --------------------------------------------------------------------------- #
# 2. input_space="copy"  (repair T2 / run R2)
# --------------------------------------------------------------------------- #


def _capture_mlp_input(encoder, x):
    """Return the tensor the per-clone MLP actually receives."""
    seen = {}

    def hook(_mod, args):
        seen["x"] = args[0].detach().clone()

    handle = encoder.mlp.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            encoder(x)
    finally:
        handle.remove()
    return seen["x"]


def test_copy_space_maps_the_three_reference_values():
    enc = _seeded(lambda: BaselineCloneEmbedding(input_space="copy"))

    x = torch.zeros(1, 3, 45)
    x[0, 0, :44] = SENTINEL
    x[0, 1, :44] = 0.0
    x[0, 2, :44] = 2.0
    x[0, :, 44] = torch.tensor([0.5, 0.3, 0.2])

    feats = _capture_mlp_input(enc, x)
    assert feats.shape == (1, 3, 44)

    # -10.966 -> (2**(-9.966) - 2) / 2 = -0.99950..., i.e. -1.0 to 4 sig figs.
    assert feats[0, 0].allclose(torch.full((44,), -1.0), atol=1e-3), feats[0, 0, 0]
    assert torch.equal(feats[0, 1], torch.zeros(44))
    assert feats[0, 2].allclose(torch.full((44,), 3.0), atol=1e-6)

    # Exact arithmetic, for the record: 2**(-9.966) is ~1e-3, so the sentinel
    # lands just above -1.0, and 2**3 = 8 is at the clamp so 2.0 -> 3.0 exactly.
    assert math.isclose(feats[0, 0, 0].item(), (2.0 ** -9.966 - 2.0) / 2.0, abs_tol=1e-6)
    assert -1.0 < feats[0, 0, 0].item() < -0.999
    assert math.isclose(feats[0, 2, 0].item(), 3.0, rel_tol=0, abs_tol=1e-6)


def test_copy_space_leaves_the_frequency_column_alone():
    """The frequency column is untouched: prove it via include_freq_in_mlp."""
    freqs = torch.tensor([0.5, 0.3, 0.2])
    x = torch.zeros(1, 3, 45)
    x[0, :, :44] = SENTINEL
    x[0, :, 44] = freqs

    enc = _seeded(
        lambda: BaselineCloneEmbedding(input_space="copy", include_freq_in_mlp=True)
    )
    feats = _capture_mlp_input(enc, x)
    assert feats.shape == (1, 3, 45)
    assert torch.equal(feats[0, :, 44], freqs)
    assert feats[0, :, :44].allclose(torch.full((3, 44), -1.0), atol=1e-3)


def test_copy_space_is_finite_and_bounded_on_a_real_batch(real_batch):
    enc = _seeded(lambda: BaselineCloneEmbedding(input_space="copy"))
    feats = _capture_mlp_input(enc, real_batch)
    assert torch.isfinite(feats).all()
    # (clamp(2**(x+1), 0, 8) - 2) / 2 lands in [-1, 3] by construction.
    assert feats.min() >= -1.0
    assert feats.max() <= 3.0 + 1e-5

    with torch.no_grad():
        out = enc(real_batch)
    assert out.shape == (real_batch.shape[0], enc.d_model)
    assert torch.isfinite(out).all()


def test_copy_space_actually_changes_the_output(real_batch):
    a = _seeded(lambda: BaselineCloneEmbedding(input_space="log2"))
    b = _seeded(lambda: BaselineCloneEmbedding(input_space="copy"))
    with torch.no_grad():
        assert not torch.equal(a(real_batch), b(real_batch))


# --------------------------------------------------------------------------- #
# 3. freq_renorm=True  (repair R4)
# --------------------------------------------------------------------------- #


def _capture_tokens(encoder, x):
    """Return the tokens handed to the first ISAB, i.e. after the freq multiply."""
    seen = {}

    def hook(_mod, args):
        seen["h"] = args[0].detach().clone()

    handle = encoder.layers[0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            encoder(x)
    finally:
        handle.remove()
    return seen["h"]


def test_freq_renorm_weights_are_a_distribution_and_keep_the_ratio():
    """Two clones at 0.001 and 0.003 scale 1:3, and the weights sum to 1."""
    x = torch.zeros(1, 2, 45)
    x[0, :, :44] = 1.0  # identical features, so the tokens differ only in scale
    x[0, :, 44] = torch.tensor([0.001, 0.003])

    enc = _seeded(lambda: CloneSetEmbedding(freq_renorm=True))
    h = _capture_tokens(enc, x)

    raw = _seeded(lambda: CloneSetEmbedding(freq_renorm=False))
    h_raw = _capture_tokens(raw, x)

    # Ratio between the two clones' token scales is 1:3 either way.
    for tokens in (h, h_raw):
        ratio = tokens[0, 1] / tokens[0, 0]
        assert ratio.allclose(torch.full_like(ratio, 3.0), atol=1e-4)

    # Renormalised weights sum to 1: recover them against the unscaled token.
    unscaled = _seeded(lambda: CloneSetEmbedding(freq_renorm=True, freq_as_weight=False))
    base = _capture_tokens(unscaled, x)
    weights = (h[0, :, 0] / base[0, :, 0])
    assert weights.allclose(torch.tensor([0.25, 0.75]), atol=1e-5)
    assert math.isclose(weights.sum().item(), 1.0, abs_tol=1e-6)

    # The published path leaves the raw sub-1% frequencies in place.
    raw_weights = h_raw[0, :, 0] / base[0, :, 0]
    assert raw_weights.allclose(torch.tensor([0.001, 0.003]), atol=1e-7)
    assert raw_weights.sum().item() < 0.01


def test_freq_renorm_gives_padded_rows_weight_zero():
    """NaN (padded) rows stay at weight 0 and do not enter the normaliser."""
    x = torch.zeros(1, 4, 45)
    x[0, :, :44] = 1.0
    x[0, :, 44] = torch.tensor([0.001, 0.003, 0.0, 0.0])
    x[0, 2:, :] = float("nan")  # last two rows are padded trials/clones

    enc = _seeded(lambda: CloneSetEmbedding(freq_renorm=True))
    h = _capture_tokens(enc, x)
    base = _capture_tokens(
        _seeded(lambda: CloneSetEmbedding(freq_renorm=True, freq_as_weight=False)), x
    )

    weights = h[0, :, 0] / base[0, :, 0]
    assert torch.equal(weights[2:], torch.zeros(2))
    # The two valid clones still carry the whole unit of mass.
    assert weights[:2].allclose(torch.tensor([0.25, 0.75]), atol=1e-5)
    assert math.isclose(weights.sum().item(), 1.0, abs_tol=1e-6)


def test_freq_renorm_all_zero_frequencies_do_not_divide_by_zero():
    x = torch.zeros(1, 3, 45)
    x[0, :, :44] = 1.0  # frequencies all 0
    enc = _seeded(lambda: CloneSetEmbedding(freq_renorm=True))
    with torch.no_grad():
        out = enc(x)
    assert torch.isfinite(out).all()


def test_freq_renorm_is_finite_on_a_real_batch(real_batch):
    enc = _seeded(lambda: CloneSetEmbedding(freq_renorm=True))
    with torch.no_grad():
        out = enc(real_batch)
    assert out.shape == (real_batch.shape[0], enc.d_model)
    assert torch.isfinite(out).all()
    assert not torch.equal(out, _seeded(CloneSetEmbedding)(real_batch))


# --- the point of R4: the attention logits stop being flat ------------------ #


class _SoftmaxSpy:
    """Proxies the module-level ``torch`` name and records softmax inputs."""

    def __init__(self):
        self.logits = []

    def __getattr__(self, name):
        return getattr(torch, name)

    def softmax(self, inp, dim):
        self.logits.append(inp.detach().clone())
        return torch.softmax(inp, dim)


def _attention_logit_spread(encoder, x, monkeypatch):
    """Mean sd of the first MAB's attention logits over the key axis."""
    spy = _SoftmaxSpy()
    with monkeypatch.context() as mp:
        mp.setattr(new_st, "torch", spy)
        with torch.no_grad():
            encoder(x)
    assert spy.logits, "no softmax call was intercepted"
    return spy.logits[0].std(dim=-1).mean().item()


def test_freq_renorm_widens_the_attention_logit_spread(real_batch, monkeypatch, capsys):
    off = _seeded(lambda: CloneSetEmbedding(freq_renorm=False))
    on = _seeded(lambda: CloneSetEmbedding(freq_renorm=True))

    spread_off = _attention_logit_spread(off, real_batch, monkeypatch)
    spread_on = _attention_logit_spread(on, real_batch, monkeypatch)

    with capsys.disabled():
        print(
            f"\n[R4] attention logit spread: freq_renorm=False {spread_off:.6e}"
            f"  freq_renorm=True {spread_on:.6e}"
            f"  (ratio {spread_on / spread_off:.1f}x)"
        )

    assert spread_on > spread_off


# --------------------------------------------------------------------------- #
# 4. Matrix 2: freq_mode, attn_ln, attn_dropout_active and copy space in the
#    set transformer (runs R5-R8).
#
# R4 proved that keeping the multiply is the problem, not its normalisation:
# even renormalised, 100 clones share one unit of mass, so a token is still at
# ~1/100 of its scale and there is no LayerNorm to rescale it. These switches
# take the multiply out, put the LayerNorms in, and make trap 4's dropout real.
# --------------------------------------------------------------------------- #


def test_set_transformer_matrix_two_defaults():
    """Every new constructor argument defaults to the published behaviour."""
    enc = CloneSetEmbedding()
    assert enc.freq_mode == "weight"
    assert enc.input_space == "log2"
    assert enc.attn_ln is False
    assert enc.attn_dropout_active is False
    assert enc.attn_dropout is None
    # The published projection still takes the 44 CNA columns only.
    assert enc.input_proj.in_features == 44

    with pytest.raises(ValueError):
        CloneSetEmbedding(freq_mode="multiply")
    with pytest.raises(ValueError):
        CloneSetEmbedding(input_space="copies")


def test_freq_mode_feature_widens_the_projection_and_changes_the_output(real_batch):
    """45 inputs, the 45th being log10(freq) -- and it is not a no-op."""
    weight = _seeded(CloneSetEmbedding)
    feature = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))

    assert weight.input_proj.in_features == 44
    assert feature.input_proj.in_features == 45

    with torch.no_grad():
        out_w = weight(real_batch)
        out_f = feature(real_batch)

    assert out_f.shape == out_w.shape
    assert torch.isfinite(out_f).all()
    assert not torch.equal(out_f, out_w)


def test_feature_mode_does_not_scale_the_tokens_with_the_frequency(real_batch):
    """The R4 finding, pinned: in "weight" mode the frequency IS the token scale.

    Multiplying every frequency by 10 multiplies every "weight"-mode token by
    exactly 10. In "feature" mode it shifts one input column by
    log10(10) = 1 and leaves the token magnitude where it was -- which is the
    whole reason the mode exists. It is *not* a bitwise no-op (the frequency
    still informs the token, which is the other half of the point); the exact
    invariance is proved in the next test.
    """
    scaled = real_batch.clone()
    scaled[..., 44] = scaled[..., 44] * 10.0

    weight = _seeded(CloneSetEmbedding)
    tokens_w = _capture_tokens(weight, real_batch)
    tokens_w10 = _capture_tokens(weight, scaled)
    assert torch.allclose(tokens_w10, tokens_w * 10.0, atol=1e-5)

    feature = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))
    tokens_f = _capture_tokens(feature, real_batch)
    tokens_f10 = _capture_tokens(feature, scaled)
    norm, norm10 = tokens_f.norm().item(), tokens_f10.norm().item()
    assert 0.5 < norm10 / norm < 2.0, (norm, norm10)
    # ... whereas the multiply moved the token norm by a clean decade.
    assert tokens_w10.norm().item() / tokens_w.norm().item() > 9.0


def test_feature_mode_frequency_enters_only_through_the_45th_column(real_batch):
    """Zero that column's weight and the frequency stops mattering, exactly."""
    feature = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))
    with torch.no_grad():
        feature.input_proj.weight[:, 44] = 0.0

    scaled = real_batch.clone()
    scaled[..., 44] = scaled[..., 44] * 10.0

    with torch.no_grad():
        assert torch.equal(feature(real_batch), feature(scaled))


def test_feature_mode_keeps_padded_rows_out_of_the_attention(real_batch):
    """A padded row's log10(0 -> 1e-6) = -6 is an ordinary number, not padding.

    In "weight" mode the masked-to-zero frequency multiply is what zeroes those
    tokens; with no multiply left the zeroing has to be explicit, or the
    padding would enter attention as data.
    """
    x = torch.zeros(1, 4, 45)
    x[0, :, :44] = 1.0
    x[0, :, 44] = torch.tensor([0.01, 0.02, 0.0, 0.0])
    x[0, 2:, :] = float("nan")

    enc = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))
    tokens = _capture_tokens(enc, x)

    assert torch.equal(tokens[0, 2:], torch.zeros_like(tokens[0, 2:]))
    assert not torch.equal(tokens[0, :2], torch.zeros_like(tokens[0, :2]))


def test_feature_mode_clamps_a_zero_frequency_to_the_log_floor():
    """log10(0) is -inf; the clamp at 1e-6 is what keeps the tokens finite."""
    x = torch.zeros(1, 3, 45)  # every frequency 0, no padding
    enc = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))
    with torch.no_grad():
        out = enc(x)
    assert torch.isfinite(out).all()

    tokens = _capture_tokens(enc, x)
    # -6 through the 45th column, plus the bias: exactly what a -6 input gives.
    expected = enc.input_proj(torch.cat([torch.zeros(44), torch.tensor([-6.0])]))
    assert torch.allclose(tokens[0, 0], expected, atol=1e-6)


def test_copy_space_in_the_set_transformer_equals_the_mlp_formula():
    """One repair, one formula: the two encoders must not drift apart."""
    feats = torch.linspace(-12.0, 3.0, steps=44).reshape(1, 1, 44)
    x = torch.zeros(1, 1, 45)
    x[..., :44] = feats
    x[..., 44] = 0.5

    enc = _seeded(lambda: CloneSetEmbedding(input_space="copy"))
    tokens = _capture_tokens(enc, x)

    mlp_formula = (torch.clamp(2.0 ** (feats + 1.0), 0, 8) - 2.0) / 2.0
    expected = enc.input_proj(mlp_formula) * 0.5  # freq_as_weight, "weight" mode
    assert torch.equal(tokens, expected)

    # And the sentinel lands on ~-1.0 here as it does in the MLP encoder,
    # instead of the 15-sd outlier -10.966 it is in log2 space.
    sentinel = torch.full((1, 1, 44), SENTINEL)
    assert torch.allclose(
        (torch.clamp(2.0 ** (sentinel + 1.0), 0, 8) - 2.0) / 2.0,
        torch.full((1, 1, 44), -1.0),
        atol=1e-3,
    )


def test_copy_space_leaves_the_set_transformers_frequency_alone(real_batch):
    """The 45th column is a frequency, not a log2 ratio."""
    enc = _seeded(lambda: CloneSetEmbedding(freq_mode="feature", input_space="copy"))
    plain = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))

    x = real_batch[:4].clone()
    tokens_copy = _capture_tokens(enc, x)
    tokens_log2 = _capture_tokens(plain, x)
    # Same weights, different CNA columns, identical frequency column: the two
    # differ, and the difference is entirely explained by the 44 features.
    assert not torch.equal(tokens_copy, tokens_log2)
    feats = torch.nan_to_num(x, nan=0.0)[..., :44]
    freq = torch.log10(torch.nan_to_num(x, nan=0.0)[..., 44].clamp_min(1e-6))
    converted = (torch.clamp(2.0 ** (feats + 1.0), 0, 8) - 2.0) / 2.0
    valid = (~torch.isnan(x).all(dim=-1)).unsqueeze(-1).float()
    expected = enc.input_proj(torch.cat([converted, freq.unsqueeze(-1)], dim=-1)) * valid
    assert torch.allclose(tokens_copy, expected, atol=1e-6)


def test_attn_ln_builds_layer_norms_and_default_builds_none():
    """Trap 5 counted: 0 LayerNorms by default, 14 with the switch."""
    from torch import nn

    default = CloneSetEmbedding()
    on = CloneSetEmbedding(attn_ln=True)

    n_default = sum(isinstance(m, nn.LayerNorm) for m in default.modules())
    n_on = sum(isinstance(m, nn.LayerNorm) for m in on.modules())
    assert n_default == 0
    # 3 ISABs x 2 MABs x 2 norms, plus the PMA's MAB's 2.
    assert n_on == 14

    for layer in on.layers:
        for mab in (layer.mab0, layer.mab1):
            assert isinstance(mab.ln0, nn.LayerNorm)
            assert isinstance(mab.ln1, nn.LayerNorm)
    assert isinstance(on.pma.mab.ln0, nn.LayerNorm)


def test_attn_ln_state_dict_does_not_load_into_a_default_encoder():
    """Why evaluation MUST rebuild from the checkpoint and not from the preset."""
    on = _seeded(lambda: CloneSetEmbedding(attn_ln=True))
    default = _seeded(CloneSetEmbedding)

    with pytest.raises(RuntimeError):
        default.load_state_dict(on.state_dict())
    # The other direction is just as broken: the LayerNorms have no source.
    with pytest.raises(RuntimeError):
        on.load_state_dict(default.state_dict())


def test_feature_mode_state_dict_does_not_load_into_a_default_encoder():
    """The 44 -> 45 projection is a shape change, so this one is loud too."""
    feature = _seeded(lambda: CloneSetEmbedding(freq_mode="feature"))
    with pytest.raises(RuntimeError):
        _seeded(CloneSetEmbedding).load_state_dict(feature.state_dict())


def test_attn_dropout_is_off_by_default_and_deterministic_in_train_mode(real_batch):
    """Trap 4 intact: the published encoder ignores `dropout` even in train()."""
    enc = _seeded(lambda: CloneSetEmbedding(dropout=0.5))
    enc.train()
    with torch.no_grad():
        assert torch.equal(enc(real_batch), enc(real_batch))


def test_attn_dropout_active_fires_in_train_mode_only(real_batch):
    enc = _seeded(lambda: CloneSetEmbedding(dropout=0.5, attn_dropout_active=True))

    enc.train()
    with torch.no_grad():
        first, second = enc(real_batch), enc(real_batch)
    assert not torch.equal(first, second), "dropout did not fire in train mode"

    enc.eval()
    with torch.no_grad():
        assert torch.equal(enc(real_batch), enc(real_batch))


def test_attn_dropout_adds_no_parameters(real_batch):
    """nn.Dropout is stateless, so R8 changes the state_dict not at all."""
    off = _seeded(lambda: CloneSetEmbedding(dropout=0.1))
    on = _seeded(lambda: CloneSetEmbedding(dropout=0.1, attn_dropout_active=True))
    assert _param_spec(on) == _param_spec(off)
    on.eval()
    with torch.no_grad():
        # Same weights, and eval-mode dropout is the identity.
        assert torch.equal(on(real_batch), off(real_batch))


def test_the_r5_stack_runs_end_to_end(real_batch):
    """R5-R8's encoder, all four switches on, on a real batch."""
    enc = _seeded(
        lambda: CloneSetEmbedding(
            input_space="copy",
            freq_mode="feature",
            attn_ln=True,
            dropout=0.1,
            attn_dropout_active=True,
        )
    )
    with torch.no_grad():
        out = enc(real_batch)
    assert out.shape == (real_batch.shape[0], enc.d_model)
    assert torch.isfinite(out).all()
    # The LayerNorms have something to work with: R4's complaint was that the
    # pooled embedding sat at ~1e-3 of the scale the flow expects.
    assert out.abs().mean().item() > 1e-2
