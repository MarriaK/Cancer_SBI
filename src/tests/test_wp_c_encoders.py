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
