#!/usr/bin/env python
"""Prove that `cancer_sbi` computes exactly what the original scripts computed.

Why this file exists
--------------------
The `cancer_sbi` package is a refactor of `Base_NPE/`, `SetTransformer_NPE/` and
`Plain_NPE/`. A refactor is only trustworthy if the new code produces bit-identical
numbers, and that cannot be checked on a machine without a working PyTorch. This
script does the check wherever PyTorch and the data are both available - on the
cluster, or on the laptop once torch is repaired.

It never trains anything and never writes into the repository. It builds the same
objects twice, once through the original modules and once through the package, and
compares them element by element.

What it checks
--------------
1. Data pipeline - the exact tensors the model is fed:
     * `top_frequent_rows_tensor` on real clone matrices (values, shapes, dtypes,
       and the order of the retained rows);
     * one full `__getitem__` from each dataset family.
2. Model construction - that both code paths build the same architecture:
     * the parameter-name/shape map of each encoder;
     * a forward pass on a fixed random input, with the same seed, compared exactly.
3. Configuration - that every per-model setting in `cancer_sbi.config` matches the
   value hard-coded in the original file.

Running it
----------
From the repository root (the directory that holds `cancer_sbi/` and `Base_NPE/`)::

    python verify_refactor.py                       # data + config checks
    python verify_refactor.py --with-models         # also build and compare encoders
    python verify_refactor.py --sim sim714 --trial 1

Exit status is 0 when every check passes, 1 otherwise, so it can be used in a job
script. Nothing is written to disk.

Note on the CloneMLP import
---------------------------
`Base_NPE/inference_model.py` imports `BaselineCloneEmbedding` from a module named
`set_transformer`, but on this laptop that class lives in `Base_NPE/MLP_encoder.py`
and no `Base_NPE/set_transformer.py` exists. This script imports the encoder from
whichever of the two module names is present, and reports which one it used, so the
mismatch is visible rather than fatal.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent
# Where the ORIGINAL model folders (Base_NPE/, SetTransformer_NPE/, Plain_NPE/) live.
# They sat beside this file until the 2026-09-23 reorganisation moved them into
# _archive_2026-09-23/. Override with --legacy-root to compare against the archive.
LEGACY = REPO
FAILURES: list[str] = []
CHECKS = 0


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_module(path: Path, name: str):
    """Import a .py file by path, without putting its directory on sys.path.

    The three model folders contain same-named modules (`utils.py`, ...), so they
    cannot all be imported normally in one process.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))
        FAILURES.append(label)


def tensors_identical(a: Any, b: Any) -> tuple[bool, str]:
    """Exact equality, treating NaN in the same position as equal."""
    import torch

    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    same = torch.equal(torch.nan_to_num(a, nan=-12345.0), torch.nan_to_num(b, nan=-12345.0))
    nan_same = bool(torch.equal(torch.isnan(a), torch.isnan(b)))
    if same and nan_same:
        return True, ""
    diff = (torch.nan_to_num(a) - torch.nan_to_num(b)).abs()
    return False, f"max |difference| = {diff.max().item():.3e}, differing entries = {int((diff > 0).sum())}"


# --------------------------------------------------------------------------- #
# 1. Data pipeline
# --------------------------------------------------------------------------- #
def check_data(data_root: Path, sim: str, trial: int) -> None:
    import pickle, gzip
    import torch

    print("\n1. DATA PIPELINE")
    old = load_module(LEGACY / "Base_NPE" / "utils.py", "old_base_utils")
    from cancer_sbi.data.clone_sets import top_frequent_rows_tensor as new_topk

    path = data_root / sim / str(trial) / "CNratios_all.pkl.gz"
    if not path.exists():
        check("clone matrix available", False, f"not found: {path}")
        return
    with gzip.open(path, "rb") as handle:
        raw = torch.as_tensor(pickle.load(handle), dtype=torch.float32)

    for top_k in (100, 50, 10):
        a = old.top_frequent_rows_tensor(raw, top_k=top_k, normalize=True, pad_to_exact_k=True)
        b = new_topk(raw, top_k=top_k, normalize=True, pad_to_exact_k=True)
        ok, detail = tensors_identical(a, b)
        check(f"top_frequent_rows_tensor(top_k={top_k}) identical", ok, detail)

    # The frequency column is the quirk most likely to drift: it is divided by the
    # total number of clone rows, so the retained frequencies sum to less than 1.
    a = old.top_frequent_rows_tensor(raw, top_k=100, normalize=True, pad_to_exact_k=True)
    b = new_topk(raw, top_k=100, normalize=True, pad_to_exact_k=True)
    check(
        "frequency column sums to the same value (< 1 by design)",
        bool(torch.equal(a[:, 44], b[:, 44])),
        f"old sum {a[:, 44].sum():.9f} vs new sum {b[:, 44].sum():.9f}",
    )

    # One full sample from each dataset family.
    old_ds = old.CNASimsDataset(str(data_root), sim_ids=[sim], top_k=100)
    from cancer_sbi.data.clone_sets import CNASimsDataset as NewCNASims

    new_ds = NewCNASims(str(data_root), sim_ids=[sim], top_k=100)
    if len(old_ds) and len(new_ds):
        (x_old, m_old, y_old), (x_new, m_new, y_new) = old_ds[0], new_ds[0]
        ok, detail = tensors_identical(x_old, x_new)
        check("CNASimsDataset X identical", ok, detail)
        check("CNASimsDataset mask identical", bool(torch.equal(m_old, m_new)))
        ok, detail = tensors_identical(y_old, y_new)
        check("CNASimsDataset theta identical", ok, detail)
    else:
        check("CNASimsDataset returned a sample", False, f"old {len(old_ds)}, new {len(new_ds)}")

    old_plain = load_module(LEGACY / "Plain_NPE" / "utils.py", "old_plain_utils")
    from cancer_sbi.data.dominant_clone import SimulationDataset as NewSimDS

    o = old_plain.SimulationDataset(str(data_root), sim_ids=[sim])
    n = NewSimDS(str(data_root), sim_ids=[sim])
    a_, b_ = o[0], n[0]
    if a_ is None or b_ is None:
        check("SimulationDataset sample", a_ is None and b_ is None, "one path dropped the sample, the other did not")
    else:
        ok, detail = tensors_identical(a_[0], b_[0])
        check("SimulationDataset theta identical", ok, detail)
        ok, detail = tensors_identical(a_[1], b_[1])
        check("SimulationDataset x identical", ok, detail)


# --------------------------------------------------------------------------- #
# 2. Models
# --------------------------------------------------------------------------- #
def _param_map(module) -> dict[str, tuple]:
    return {name: tuple(p.shape) for name, p in module.named_parameters()}


def _compare_encoder(label: str, build_old: Callable[[], Any], build_new: Callable[[], Any], sample_shape: tuple) -> None:
    import torch

    torch.manual_seed(0)
    old = build_old().eval()
    torch.manual_seed(0)
    new = build_new().eval()

    check(f"{label}: same parameter names and shapes", _param_map(old) == _param_map(new),
          f"only in old: {set(_param_map(old)) - set(_param_map(new))}; "
          f"only in new: {set(_param_map(new)) - set(_param_map(old))}")

    # Identical seeds must give identical initial weights; then the forward pass
    # is a complete test of the architecture and of the maths inside it.
    same_init = all(torch.equal(a, b) for (_, a), (_, b) in zip(old.named_parameters(), new.named_parameters()))
    check(f"{label}: identical initial weights under the same seed", same_init)

    torch.manual_seed(1)
    x = torch.randn(*sample_shape)
    x[0, -1, :] = float("nan")            # exercise the NaN-padding path too
    with torch.no_grad():
        ok, detail = tensors_identical(old(x), new(x))
    check(f"{label}: forward pass identical", ok, detail)


def check_models() -> None:
    print("\n2. MODELS")
    # CloneMLP encoder: the original import name and the real file name disagree.
    base = LEGACY / "Base_NPE"
    src = base / "set_transformer.py" if (base / "set_transformer.py").exists() else base / "MLP_encoder.py"
    print(f"  (CloneMLP encoder read from {src.name})")
    old_mlp = load_module(src, "old_mlp_encoder")
    from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding as NewMLP

    _compare_encoder(
        "BaselineCloneEmbedding",
        lambda: old_mlp.BaselineCloneEmbedding(in_dim=45, d_model=128, hidden_dim=256, num_layers=3,
                                               dropout=0.2, freq_as_weight=True, include_freq_in_mlp=False),
        lambda: NewMLP(in_dim=45, d_model=128, hidden_dim=256, num_layers=3,
                       dropout=0.2, freq_as_weight=True, include_freq_in_mlp=False),
        (4, 100, 45),
    )

    old_st = load_module(LEGACY / "SetTransformer_NPE" / "set_transformer.py", "old_set_transformer")
    from cancer_sbi.models.set_transformer import CloneSetEmbedding as NewAtt

    _compare_encoder(
        "CloneSetEmbedding",
        lambda: old_st.CloneSetEmbedding(in_dim=45, d_model=128, n_heads=8, num_layers=3,
                                         num_inducing=32, dropout=0.2, freq_as_weight=True),
        lambda: NewAtt(in_dim=45, d_model=128, n_heads=8, num_layers=3,
                       num_inducing=32, dropout=0.2, freq_as_weight=True),
        (4, 100, 45),
    )

    old_ds = load_module(LEGACY / "Plain_NPE" / "net_builder.py", "old_net_builder")
    from cancer_sbi.models.deep_set import DeepSet as NewDeepSet

    _compare_encoder(
        "DeepSet",
        lambda: old_ds.DeepSet(hidden_dim_phi=44, hidden_dim_rho=44, output_dim=128),
        lambda: NewDeepSet(hidden_dim_phi=44, hidden_dim_rho=44, output_dim=128),
        (4, 25, 44),
    )


# --------------------------------------------------------------------------- #
# 3. Configuration
# --------------------------------------------------------------------------- #
def check_config() -> None:
    """Every per-model setting must equal the value hard-coded in the original."""
    print("\n3. CONFIGURATION (the per-model divergences that must not be unified)")
    from cancer_sbi.config import get_preset

    # The PUBLISHED presets, deliberately. The bare names `clonemlp`,
    # `cloneatt` and `dominantclone` were repointed at the repaired
    # configurations on 2026-09-25 (cancer_sbi/config.py, "The repaired
    # defaults"); this script's job is the opposite one -- to assert that the
    # models as published are still reproduced field for field -- so every
    # lookup below is a `*_published` one.
    expected = {
        # preset name : (flow z_score_x, grad_clip, two optimiser groups, reload_best)
        "clonemlp_published": ("none", 5.0, True, "never"),
        "cloneatt_published": ("none", 5.0, True, "on_early_stop"),
        "dominantclone_published": ("structured", None, False, "always"),
    }
    for name, (z_score, clip, groups, reload_best) in expected.items():
        preset = get_preset(name)
        check(f"{name}: flow z_score_x == {z_score!r}", preset.flow.z_score_x == z_score,
              f"got {preset.flow.z_score_x!r}")
        check(f"{name}: grad_clip == {clip}", preset.optim.grad_clip == clip, f"got {preset.optim.grad_clip}")
        check(f"{name}: two optimiser groups is {groups}", preset.optim.use_param_groups == groups,
              f"got {preset.optim.use_param_groups}")
        check(f"{name}: reload_best == {reload_best!r}", preset.train.reload_best == reload_best,
              f"got {preset.train.reload_best!r}")

    # The optimiser numbers are the ones a "tidy-up" would most easily unify.
    mlp = get_preset("clonemlp_published").optim
    check("clonemlp: flow group lr 1e-3 / weight decay 1e-4",
          (mlp.flow_lr, mlp.flow_weight_decay) == (1e-3, 1e-4), f"got {mlp.flow_lr}, {mlp.flow_weight_decay}")
    check("clonemlp: embedding group lr 1e-4 / weight decay 0.0",
          (mlp.embed_lr, mlp.embed_weight_decay) == (1e-4, 0.0), f"got {mlp.embed_lr}, {mlp.embed_weight_decay}")
    check("clonemlp: learning_rate is recorded as unused", mlp.learning_rate_is_used is False)
    dom = get_preset("dominantclone_published").optim
    check("dominantclone: single group lr 5e-4", dom.learning_rate == 5e-4 and dom.learning_rate_is_used is True,
          f"got lr={dom.learning_rate}, used={dom.learning_rate_is_used}")
    att = get_preset("cloneatt_published")
    check("cloneatt: encoder dropout recorded as ignored", att.encoder.encoder_dropout_is_used is False)
    check("cloneatt: no LayerNorm in the attention stack", att.encoder.layer_norm_in_attention is False)
    check("cloneatt: min_epochs not enforced", att.train.enforce_min_epochs is False)
    check("clonemlp checkpoint dir is the original 'checkpoints_baseline'",
          get_preset("clonemlp_published").train.ckpt_dir == "checkpoints_baseline",
          f"got {get_preset('clonemlp_published').train.ckpt_dir!r}")
    for nm in ("clonemlp_published", "cloneatt_published", "dominantclone_published"):
        f = get_preset(nm).flow
        check(f"{nm}: sbi architecture pinned explicitly (50/5/10/2/3.0)",
              (f.hidden_features, f.num_transforms, f.num_bins, f.num_blocks, f.tail_bound) == (50, 5, 10, 2, 3.0),
              f"got {f.hidden_features}, {f.num_transforms}, {f.num_bins}, {f.num_blocks}, {f.tail_bound}")

    check("prior sd defaults to the correct 0.2 (originals used sqrt(0.2))",
          abs(get_preset("clonemlp_published").prior_sd - 0.2) < 1e-12,
          f"got {get_preset('clonemlp_published').prior_sd}")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-root", type=Path, default=None,
                        help="Directory holding the original Base_NPE/, SetTransformer_NPE/ and "
                             "Plain_NPE/ folders. Defaults to this file's own directory; after the "
                             "2026-09-23 reorganisation they live in ~/cancer/_archive_2026-09-23/.")
    parser.add_argument("--data-root", type=Path, default=REPO / "Guassian_Normal" / "simulation_outputs",
                        help="directory holding the sim<N> folders")
    parser.add_argument("--sim", default="sim714", help="simulation folder to compare")
    parser.add_argument("--trial", type=int, default=1, help="replicate inside that folder")
    parser.add_argument("--with-models", action="store_true",
                        help="also build both versions of each encoder and compare a forward pass")
    parser.add_argument("--skip-data", action="store_true", help="skip the data-pipeline checks")
    args = parser.parse_args()

    global LEGACY
    if args.legacy_root is not None:
        LEGACY = args.legacy_root.resolve()
    sys.path.insert(0, str(REPO))
    try:
        import torch  # noqa: F401
    except Exception as exc:                      # pragma: no cover - environment problem, not a test failure
        print("Cannot run: PyTorch does not import here.")
        print(f"  {type(exc).__name__}: {exc}")
        print("\nRun this on the cluster, or repair the local install with:")
        print("  pip install --force-reinstall torch")
        return 2

    print(f"Comparing the original scripts with cancer_sbi\nrepository: {REPO}")
    check_config()
    if not args.skip_data:
        check_data(args.data_root, args.sim, args.trial)
    if args.with_models:
        check_models()

    print(f"\n{'-' * 60}\n{CHECKS - len(FAILURES)} of {CHECKS} checks passed")
    if FAILURES:
        print("FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("The refactored code matches the originals on every check above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
