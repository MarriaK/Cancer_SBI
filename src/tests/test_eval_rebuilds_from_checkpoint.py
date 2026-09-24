"""Evaluation must rebuild the network the checkpoint was trained with.

Run from ``src/``::

    python -m pytest tests/test_eval_rebuilds_from_checkpoint.py -q

Until 2026-09-24 both evaluation entry points rebuilt the model from
``get_preset(args.model)``, which is only right for a run that used no repair
flag. The three matrix runs each break it a different way:

* **R1** (``--z-score-x structured``) puts a standardising transform inside the
  flow's composite chain, so the checkpoint's ``state_dict`` has keys the
  preset-built network does not -- :func:`test_r1_checkpoint_does_not_load_into_a_preset_built_flow`
  is that failure, pinned.
* **R2** (``--input-space copy``) and **R4** (``--freq-renorm``) change what the
  encoder *computes* without changing a single parameter name, so they load
  silently and report wrong numbers.

So every test here trains a real one-epoch model on a handful of local sims and
then drives the real ``sample_posteriors.main()`` over it.

Skipped, not failed, when the local simulation tree is absent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cancer_sbi.evaluation.sample_posteriors as sp  # noqa: E402
from cancer_sbi.cli import train as train_cli  # noqa: E402
from cancer_sbi.config import EFFECTIVE_CONFIG_KEY, get_preset  # noqa: E402
from cancer_sbi.evaluation import posterior as posterior_mod  # noqa: E402

DATA_ROOT = SRC.parent / "data" / "Guassian_Normal" / "simulation_outputs"

needs_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="local simulation tree not present"
)

#: The three repair runs as (name, model, extra train flags).
RUNS = [
    ("R1", "clonemlp", ["--z-score-x", "structured"]),
    ("R2", "clonemlp", ["--z-score-x", "structured", "--input-space", "copy"]),
    ("R4", "cloneatt", ["--z-score-x", "structured", "--freq-renorm"]),
]


def _sim_names(n):
    """The first ``n`` sim directories on disk, in numeric order."""
    names = sorted(
        (p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim")),
        key=lambda name: int(name[3:]),
    )
    return names[:n]


def _write_split(tmp_path, train_ids, val_ids, test_ids):
    """A three-key split pickle, as ``make_split --add-val`` writes."""
    import pickle

    path = tmp_path / "split.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "train_ids": np.array(train_ids),
                "val_ids": np.array(val_ids),
                "test_ids": np.array(test_ids),
            },
            handle,
        )
    return path


def _train_tiny(tmp_path, model, extra, tag):
    """Train ``model`` for one epoch on six sims and return its checkpoint dir."""
    names = _sim_names(6)
    split_path = _write_split(tmp_path, names[:3], names[3:4], names[4:6])
    ckpt_dir = tmp_path / tag / "checkpoints"
    rc = train_cli.main(
        [
            "--model", model,
            "--data-root", str(DATA_ROOT),
            "--split", str(split_path),
            "--out", str(tmp_path / tag),
            "--ckpt-dir", str(ckpt_dir),
            "--device", "cpu",
            "--batch-size", "2",
            "--max-epochs", "1",
            "--min-epochs", "1",
            "--stop-after-epochs", "1",
            "--seed", "20260924",
            "--no-final-pickle",
            *extra,
        ]
    )
    assert rc == 0
    return ckpt_dir, split_path


def _sample(tmp_path, model, ckpt, split_path, argv_extra=(), monkeypatch=None):
    """Drive the real ``sample_posteriors.main()`` and return (path, meta)."""
    out_dir = tmp_path / "post"
    argv = [
        "sample_posteriors",
        "--model", model,
        "--data-root", str(DATA_ROOT),
        "--split", str(split_path),
        "--ckpt", str(ckpt),
        "--out-dir", str(out_dir),
        "--limit", "1",
        "--num-samples", "32",
        "--device", "cpu",
        *argv_extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    sp.main()
    path = out_dir / sp.output_filename(model, 1, None)
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(str(data["meta_json"]))
        assert data["samples"].shape == (1, 32, 44)
    return path, meta


# ---------------------------------------------------------------------------
# 1. Training writes the effective config.
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("tag, model, extra", RUNS, ids=[r[0] for r in RUNS])
def test_checkpoints_carry_the_effective_config(tmp_path, tag, model, extra):
    """best.pt and latest.pt both record the flags the run was given."""
    ckpt_dir, _ = _train_tiny(tmp_path, model, extra, tag)

    for name in ("best.pt", "latest.pt"):
        stored = torch.load(ckpt_dir / name, map_location="cpu")[EFFECTIVE_CONFIG_KEY]
        assert stored["model"] == model
        assert stored["flow"]["z_score_x"] == "structured"
        assert stored["flow"]["hidden_features"] == 50, "must stay 50"
        assert stored["encoder"]["input_space"] == (
            "copy" if "--input-space" in extra else "log2"
        )
        assert stored["encoder"]["freq_renorm"] is ("--freq-renorm" in extra)
        assert stored["seed"] == 20260924
        assert stored["split_path"].endswith("split.pkl")
        assert stored["cli_flags"]["z_score_x"] == "structured"


# ---------------------------------------------------------------------------
# 2. Evaluation rebuilds from it, for each of the three runs.
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("tag, model, extra", RUNS, ids=[r[0] for r in RUNS])
def test_sample_posteriors_loads_a_repair_checkpoint(
    tmp_path, monkeypatch, tag, model, extra
):
    """The real sampler runs to an .npz, with the right architecture.

    Before the repair this raised for R1 (state-dict key mismatch) and quietly
    produced wrong numbers for R2 and R4.
    """
    ckpt_dir, split_path = _train_tiny(tmp_path, model, extra, tag)

    _, meta = _sample(
        tmp_path, model, ckpt_dir / "best.pt", split_path, monkeypatch=monkeypatch
    )

    # The .npz records what it was drawn from (M1: "record the effective
    # flow/encoder config in the meta").
    assert meta["config_from_checkpoint"] is True
    assert meta["flow_config"]["z_score_x"] == "structured"
    assert meta["encoder_config"]["input_space"] == (
        "copy" if "--input-space" in extra else "log2"
    )
    assert meta["encoder_config"]["freq_renorm"] is ("--freq-renorm" in extra)


@needs_data
def test_r1_checkpoint_does_not_load_into_a_preset_built_flow(tmp_path):
    """The bug M1 fixes, pinned: the preset-built network is a different network.

    This is why evaluation could not simply keep using ``get_preset``.
    """
    from cancer_sbi.data.loaders import build_clone_set_dataloaders
    from cancer_sbi.training.trainer import build_training_components

    ckpt_dir, _ = _train_tiny(tmp_path, "clonemlp", ["--z-score-x", "structured"], "R1x")
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu")["model_state"]

    names = _sim_names(3)
    train_loader, _, _ = build_clone_set_dataloaders(
        str(DATA_ROOT), names, names, top_k=100, batch_size=2
    )
    preset_built = build_training_components(
        get_preset("clonemlp"), train_loader, device="cpu", log_progress=False
    ).density_estimator

    with pytest.raises(RuntimeError):
        preset_built.load_state_dict(state)

    # And the checkpoint's own config does build something that loads.
    rebuilt = build_training_components(
        posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "clonemlp").preset,
        train_loader,
        device="cpu",
        log_progress=False,
    ).density_estimator
    rebuilt.load_state_dict(state)


# ---------------------------------------------------------------------------
# 3. Old checkpoints, and the overrides that describe them.
# ---------------------------------------------------------------------------


@needs_data
def test_old_style_checkpoint_still_loads_with_a_warning(
    tmp_path, monkeypatch, capsys
):
    """A checkpoint with no ``effective_config`` key -- every one on the cluster."""
    ckpt_dir, split_path = _train_tiny(tmp_path, "clonemlp", [], "R0")

    old_style = ckpt_dir / "old_best.pt"
    payload = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    payload.pop(EFFECTIVE_CONFIG_KEY)
    torch.save(payload, old_style)

    _, meta = _sample(
        tmp_path, "clonemlp", old_style, split_path, monkeypatch=monkeypatch
    )
    out = capsys.readouterr().out
    assert "[warn]" in out and "effective_config" in out
    assert meta["config_from_checkpoint"] is False
    # Fell back to the published preset, which is what R0 used anyway.
    assert meta["flow_config"]["z_score_x"] == "none"


@needs_data
def test_overrides_describe_an_old_checkpoint(tmp_path):
    """With no stored config the flags are the only way to say what a run was."""
    ckpt_dir, _ = _train_tiny(tmp_path, "clonemlp", ["--z-score-x", "structured"], "R1o")
    old_style = ckpt_dir / "old_best.pt"
    payload = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    payload.pop(EFFECTIVE_CONFIG_KEY)
    torch.save(payload, old_style)

    resolved = posterior_mod.resolve_eval_config(
        old_style, "clonemlp", z_score_x="structured", input_space="copy"
    )
    assert resolved.from_checkpoint is False
    assert resolved.preset.flow.z_score_x == "structured"
    assert resolved.preset.encoder.input_space == "copy"


@needs_data
@pytest.mark.parametrize(
    "override, needle",
    [
        ({"z_score_x": "none"}, "--z-score-x"),
        ({"input_space": "copy"}, "--input-space"),
        ({"freq_renorm": True}, "--freq-renorm"),
    ],
)
def test_an_override_that_contradicts_the_checkpoint_raises(
    tmp_path, override, needle
):
    """Neither side is trusted silently when they disagree."""
    ckpt_dir, _ = _train_tiny(tmp_path, "clonemlp", ["--z-score-x", "structured"], "R1c")

    with pytest.raises(ValueError, match="contradict"):
        posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "clonemlp", **override)
    try:
        posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "clonemlp", **override)
    except ValueError as exc:
        assert needle in str(exc)


@needs_data
def test_the_checkpoints_model_must_match_the_model_flag(tmp_path):
    """A cloneatt checkpoint evaluated as clonemlp is caught by name, not by shape."""
    ckpt_dir, _ = _train_tiny(tmp_path, "cloneatt", [], "att")

    with pytest.raises(ValueError, match="trained as model 'cloneatt'"):
        posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "clonemlp")


# ---------------------------------------------------------------------------
# 4. cli/evaluate.py goes through the same resolution.
# ---------------------------------------------------------------------------


class _StopAtModelBuild(Exception):
    """Raised in place of ``build_training_components``, carrying its preset."""

    def __init__(self, preset):
        super().__init__("stopped at model build")
        self.preset = preset


@needs_data
@pytest.mark.parametrize("tag, model, extra", RUNS, ids=[r[0] for r in RUNS])
def test_evaluate_cli_rebuilds_from_the_checkpoint(
    tmp_path, monkeypatch, tag, model, extra
):
    """``cli/evaluate.py`` must resolve the architecture before it builds one.

    Checked at the seam rather than by drawing every figure: what changed is
    which preset reaches ``build_training_components``, and the figures below
    it are unchanged and slow.
    """
    import cancer_sbi.training.trainer as trainer_mod

    from cancer_sbi.cli import evaluate as eval_cli

    ckpt_dir, split_path = _train_tiny(tmp_path, model, extra, tag)

    def _boom(preset, train_loader, **kwargs):
        raise _StopAtModelBuild(preset)

    monkeypatch.setattr(trainer_mod, "build_training_components", _boom)

    with pytest.raises(_StopAtModelBuild) as excinfo:
        eval_cli.main(
            [
                "--model", model,
                "--data-root", str(DATA_ROOT),
                "--split", str(split_path),
                "--ckpt", str(ckpt_dir / "best.pt"),
                "--run-dir", str(tmp_path / tag),
                "--out-dir", str(tmp_path / "eval"),
                "--device", "cpu",
                "--limit-cases", "2",
            ]
        )

    preset = excinfo.value.preset
    assert preset.flow.z_score_x == "structured"
    assert preset.flow.hidden_features == 50
    assert preset.encoder.input_space == ("copy" if "--input-space" in extra else "log2")
    assert preset.encoder.freq_renorm is ("--freq-renorm" in extra)
    # Untouched by the repair, and the thing a preset rebuild would have given.
    assert get_preset(model).flow.z_score_x == "none"


@needs_data
def test_evaluate_cli_rejects_a_contradicting_override(tmp_path):
    """The same guard as the sampler, through the other entry point."""
    from cancer_sbi.cli import evaluate as eval_cli

    ckpt_dir, split_path = _train_tiny(
        tmp_path, "clonemlp", ["--z-score-x", "structured"], "R1e"
    )

    with pytest.raises(ValueError, match="contradict"):
        eval_cli.main(
            [
                "--model", "clonemlp",
                "--data-root", str(DATA_ROOT),
                "--split", str(split_path),
                "--ckpt", str(ckpt_dir / "best.pt"),
                "--run-dir", str(tmp_path / "R1e"),
                "--out-dir", str(tmp_path / "eval"),
                "--device", "cpu",
                "--z-score-x", "none",
            ]
        )


# ---------------------------------------------------------------------------
# 5. B12: evaluating a run must not create anything beside it.
#
# `cli/evaluate.py` used to construct a Trainer pointed at
# `run_dir/preset.train.ckpt_dir`, and `Trainer.__init__` creates that
# directory eagerly -- so evaluating a clonemlp run left an empty, wrong
# `checkpoints_baseline/` next to the `checkpoints/` it had just read, and
# printed a [note] about the collision on every single run of the matrix.
# ---------------------------------------------------------------------------


@needs_data
def test_evaluating_a_run_creates_no_checkpoint_directory(tmp_path, capsys):
    """The run directory's listing is unchanged, and nothing is noted.

    A full evaluation, not a seam: the stray directory was created by an
    import-time-invisible side effect of constructing the Trainer, so only
    running the command can show it is gone.
    """
    from cancer_sbi.cli import evaluate as eval_cli

    ckpt_dir, split_path = _train_tiny(tmp_path, "clonemlp", [], "B12")
    run_dir = tmp_path / "B12"
    before = sorted(p.name for p in run_dir.iterdir())
    assert before == ["checkpoints"], before

    rc = eval_cli.main(
        [
            "--model", "clonemlp",
            "--data-root", str(DATA_ROOT),
            "--split", str(split_path),
            "--ckpt", str(ckpt_dir / "best.pt"),
            "--run-dir", str(run_dir),
            # --out-dir elsewhere, so the figures cannot be what keeps the
            # listing unchanged.
            "--out-dir", str(tmp_path / "figures"),
            "--device", "cpu",
            "--limit-cases", "2",
            "--num-posterior-samples", "32",
        ]
    )

    assert rc == 0
    assert sorted(p.name for p in run_dir.iterdir()) == before
    assert not (run_dir / "checkpoints_baseline").exists()
    assert "[note]" not in capsys.readouterr().out


@needs_data
def test_explicit_resume_dir_still_resumes_and_still_notes(tmp_path, capsys):
    """The flag is opt-in now, not gone: asking for a resume still gets one."""
    from cancer_sbi.cli import evaluate as eval_cli

    ckpt_dir, split_path = _train_tiny(tmp_path, "clonemlp", [], "B12b")
    run_dir = tmp_path / "B12b"
    resume_dir = run_dir / "somewhere_else"

    rc = eval_cli.main(
        [
            "--model", "clonemlp",
            "--data-root", str(DATA_ROOT),
            "--split", str(split_path),
            "--ckpt", str(ckpt_dir / "best.pt"),
            "--run-dir", str(run_dir),
            "--resume-dir", str(resume_dir),
            "--out-dir", str(tmp_path / "figures2"),
            "--device", "cpu",
            "--limit-cases", "2",
            "--num-posterior-samples", "32",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[note]" in out and "somewhere_else" in out
    # Only the directory the caller named, and only because they named it.
    assert resume_dir.is_dir()
    assert not (run_dir / "checkpoints_baseline").exists()


# ---------------------------------------------------------------------------
# 6. --require-all-trials: DominantClone on the clone-set models' sim set.
#
# The flag lives in DataConfig, so it rides in the checkpoint's effective
# config for free -- but nothing *used* it at evaluation time until the test
# loader was built from it too. Without that, a run trained on 2,261 sims is
# scored on a test set of 707 instead of 651, and the comparison it exists for
# is wrong again.
# ---------------------------------------------------------------------------


def _mixed_dominant_split(tmp_path):
    """A split whose test set mixes complete and incomplete sims.

    Returns:
        ``(split_path, complete_test_ids, all_test_ids)``.
    """
    from cancer_sbi.data.splits import complete_sim_ids

    names = sorted(
        (p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim")),
        key=lambda name: int(name[3:]),
    )
    complete = complete_sim_ids(str(DATA_ROOT), names)
    incomplete = [
        name
        for name in names
        if name not in set(complete)
        # Only the ones the dominant-clone dataset itself keeps: a sim whose
        # every trial is missing is dropped by both paths and would prove
        # nothing here.
        and _dominant_keeps(name)
    ]
    if len(complete) < 6 or not incomplete:
        pytest.skip("local tree has no mix of complete and incomplete sims")

    test_ids = sorted(complete[4:6] + incomplete[:1], key=lambda n: int(n[3:]))
    path = _write_split(tmp_path, complete[:3], complete[3:4], test_ids)
    return path, complete[4:6], test_ids


def _dominant_keeps(name):
    """True when ``SimulationDataset`` returns a sample for this sim."""
    from cancer_sbi.data.dominant_clone import SimulationDataset

    return SimulationDataset(str(DATA_ROOT), sim_ids=[name])[0] is not None


def _train_tiny_dominant(tmp_path, split_path, extra, tag):
    """One epoch of dominantclone on the given split."""
    ckpt_dir = tmp_path / tag / "checkpoints"
    rc = train_cli.main(
        [
            "--model", "dominantclone",
            "--data-root", str(DATA_ROOT),
            "--split", str(split_path),
            "--out", str(tmp_path / tag),
            "--ckpt-dir", str(ckpt_dir),
            "--device", "cpu",
            "--batch-size", "2",
            "--max-epochs", "1",
            "--min-epochs", "1",
            "--stop-after-epochs", "1",
            "--seed", "20260924",
            "--no-final-pickle",
            *extra,
        ]
    )
    assert rc == 0
    return ckpt_dir


@needs_data
def test_require_all_trials_reaches_the_checkpoints_effective_config(tmp_path):
    """It lives in DataConfig, so config_to_dict carries it -- pinned, not assumed."""
    split_path, _, _ = _mixed_dominant_split(tmp_path)

    on = _train_tiny_dominant(tmp_path, split_path, ["--require-all-trials"], "D0")
    off = _train_tiny_dominant(tmp_path, split_path, [], "Dpub")

    for name in ("best.pt", "latest.pt"):
        stored = torch.load(on / name, map_location="cpu")[EFFECTIVE_CONFIG_KEY]
        assert stored["data"]["require_all_trials"] is True
        assert stored["cli_flags"]["require_all_trials"] is True
    stored = torch.load(off / "best.pt", map_location="cpu")[EFFECTIVE_CONFIG_KEY]
    assert stored["data"]["require_all_trials"] is False


@needs_data
def test_sample_posteriors_uses_the_restricted_test_set(tmp_path, monkeypatch):
    """The whole point: the checkpoint's own flag rebuilds the same test set."""
    split_path, complete_test, all_test = _mixed_dominant_split(tmp_path)
    assert len(all_test) > len(complete_test)

    ckpt_dir = _train_tiny_dominant(tmp_path, split_path, ["--require-all-trials"], "D0")
    _, meta = _sample(
        tmp_path, "dominantclone", ckpt_dir / "best.pt", split_path,
        monkeypatch=monkeypatch,
    )
    assert meta["require_all_trials"] is True
    assert meta["n_cases_available"] == len(complete_test)


@needs_data
def test_sample_posteriors_keeps_the_published_test_set_without_the_flag(
    tmp_path, monkeypatch
):
    """Default off is byte-for-byte the published behaviour: every sim is scored."""
    split_path, complete_test, all_test = _mixed_dominant_split(tmp_path)

    ckpt_dir = _train_tiny_dominant(tmp_path, split_path, [], "Dpub")
    _, meta = _sample(
        tmp_path, "dominantclone", ckpt_dir / "best.pt", split_path,
        monkeypatch=monkeypatch,
    )
    assert meta["require_all_trials"] is False
    assert meta["n_cases_available"] == len(all_test)


@needs_data
def test_require_all_trials_override_contradicting_the_checkpoint_raises(tmp_path):
    """Same rule as the architecture flags: neither side is trusted silently."""
    split_path, _, _ = _mixed_dominant_split(tmp_path)
    ckpt_dir = _train_tiny_dominant(tmp_path, split_path, [], "Dpub")

    with pytest.raises(ValueError, match="contradict") as excinfo:
        posterior_mod.resolve_eval_config(
            ckpt_dir / "best.pt", "dominantclone", require_all_trials=True
        )
    assert "--require-all-trials" in str(excinfo.value)


@needs_data
def test_require_all_trials_override_describes_an_old_checkpoint(tmp_path):
    """A checkpoint that predates the flag carries no key, so the flag is allowed."""
    split_path, _, _ = _mixed_dominant_split(tmp_path)
    ckpt_dir = _train_tiny_dominant(tmp_path, split_path, [], "Dold")

    old_style = ckpt_dir / "old_best.pt"
    payload = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    payload.pop(EFFECTIVE_CONFIG_KEY)
    torch.save(payload, old_style)

    resolved = posterior_mod.resolve_eval_config(
        old_style, "dominantclone", require_all_trials=True
    )
    assert resolved.from_checkpoint is False
    assert resolved.require_all_trials is True


# ---------------------------------------------------------------------------
# 7. Matrix 2 (R5-R8): the CloneAtt switches make a *different network*.
#
# `--attn-ln` adds 14 LayerNorms and `--freq-mode feature` widens the input
# projection from 44 to 45, so a preset-built cloneatt cannot load either one --
# the same failure R1 has in the flow, now in the encoder. `--encoder-dropout`
# is the quiet one: it changes no parameter name at all, and a rebuild that
# forgot it would report numbers from an encoder that never had dropout.
# ---------------------------------------------------------------------------


@needs_data
def test_matrix_two_cloneatt_checkpoint_round_trips_through_the_sampler(
    tmp_path, monkeypatch
):
    """Train R8's encoder for one epoch, then really sample from it."""
    extra = [
        "--z-score-x", "structured",
        "--freq-mode", "feature",
        "--attn-ln",
        "--encoder-dropout", "0.1",
        "--input-space", "copy",
    ]
    ckpt_dir, split_path = _train_tiny(tmp_path, "cloneatt", extra, "R8")

    stored = torch.load(ckpt_dir / "best.pt", map_location="cpu")[EFFECTIVE_CONFIG_KEY]
    assert stored["encoder"]["freq_mode"] == "feature"
    assert stored["encoder"]["attn_ln"] is True
    assert stored["encoder"]["attn_dropout_active"] is True
    assert stored["encoder"]["dropout"] == 0.1
    assert stored["encoder"]["input_space"] == "copy"
    assert stored["flow"]["hidden_features"] == 50, "must stay 50"

    path, meta = _sample(
        tmp_path, "cloneatt", ckpt_dir / "best.pt", split_path,
        monkeypatch=monkeypatch,
    )
    assert path.exists()
    assert meta["config_from_checkpoint"] is True
    assert meta["encoder_config"]["freq_mode"] == "feature"
    assert meta["encoder_config"]["attn_ln"] is True
    assert meta["encoder_config"]["attn_dropout_active"] is True
    assert meta["encoder_config"]["input_space"] == "copy"

    # The network the sampler rebuilt is the one that was trained.
    rebuilt = posterior_mod.resolve_eval_config(
        ckpt_dir / "best.pt", "cloneatt"
    ).preset
    encoder = _build_cloneatt_encoder(rebuilt)
    assert encoder.freq_mode == "feature"
    assert encoder.attn_ln is True
    assert encoder.input_space == "copy"
    assert encoder.attn_dropout is not None and encoder.attn_dropout.p == 0.1
    assert encoder.input_proj.in_features == 45


def _build_cloneatt_encoder(preset):
    """The trial encoder ``build_embedding_net`` makes for this preset."""
    from cancer_sbi.training.trainer import build_embedding_net

    return build_embedding_net(preset.encoder, "cpu").trial_encoder


@needs_data
def test_an_attn_ln_checkpoint_does_not_load_into_a_default_cloneatt(tmp_path):
    """The proof that rebuilding from the checkpoint is necessary, not tidy."""
    from cancer_sbi.data.loaders import build_clone_set_dataloaders
    from cancer_sbi.training.trainer import build_training_components

    ckpt_dir, _ = _train_tiny(tmp_path, "cloneatt", ["--attn-ln"], "R5ln")
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu")["model_state"]

    names = _sim_names(3)
    train_loader, _, _ = build_clone_set_dataloaders(
        str(DATA_ROOT), names, names, top_k=100, batch_size=2
    )
    preset_built = build_training_components(
        get_preset("cloneatt"), train_loader, device="cpu", log_progress=False
    ).density_estimator

    with pytest.raises(RuntimeError):
        preset_built.load_state_dict(state)

    # ... and the checkpoint's own config does build something that loads.
    rebuilt = build_training_components(
        posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "cloneatt").preset,
        train_loader,
        device="cpu",
        log_progress=False,
    ).density_estimator
    rebuilt.load_state_dict(state)


# ---------------------------------------------------------------------------
# 8. Matrix 3: a smaller flow, a 45-input MLP, and an augmented TRAIN loader.
#
# `--flow-num-transforms 3` changes the flow's state_dict shape and
# `--freq-mode feature` changes the MLP's, so both have to be rebuilt from the
# checkpoint -- the same argument as section 7, on the other model.
# `--trial-subsample` is the opposite case: it must reach the training loader
# and must NOT reach the test loader, because 25 trials is the condition every
# reported number is measured under.
# ---------------------------------------------------------------------------


@needs_data
def test_matrix_three_clonemlp_checkpoint_round_trips_through_the_sampler(
    tmp_path, monkeypatch
):
    """Train all four matrix-3 switches for one epoch, then really sample."""
    from cancer_sbi.data import loaders as loaders_mod
    from cancer_sbi.training.trainer import build_embedding_net

    seen = []
    real_builder = loaders_mod.build_clone_set_dataloaders

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("trial_subsample"))
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(loaders_mod, "build_clone_set_dataloaders", _spy)

    extra = [
        "--z-score-x", "structured",
        "--input-space", "copy",
        "--freq-mode", "feature",
        "--flow-dropout", "0.1",
        "--flow-num-transforms", "3",
        "--trial-subsample", "3",
    ]
    ckpt_dir, split_path = _train_tiny(tmp_path, "clonemlp", extra, "M3")

    # Training asked for the augmentation...
    assert seen == [3]

    stored = torch.load(ckpt_dir / "best.pt", map_location="cpu")[EFFECTIVE_CONFIG_KEY]
    assert stored["flow"]["dropout_probability"] == 0.1
    assert stored["flow"]["num_transforms"] == 3
    assert stored["flow"]["hidden_features"] == 50, "must stay 50"
    assert stored["encoder"]["freq_mode"] == "feature"
    assert stored["data"]["trial_subsample"] == 3
    assert stored["cli_flags"]["trial_subsample"] == 3

    seen.clear()
    path, meta = _sample(
        tmp_path, "clonemlp", ckpt_dir / "best.pt", split_path,
        monkeypatch=monkeypatch,
    )
    assert path.exists()
    assert meta["config_from_checkpoint"] is True
    # ...and evaluation did not: the test loader keeps all 25 trials.
    assert seen == [None]

    rebuilt = posterior_mod.resolve_eval_config(
        ckpt_dir / "best.pt", "clonemlp"
    ).preset
    assert rebuilt.flow.num_transforms == 3
    assert rebuilt.flow.dropout_probability == 0.1
    assert rebuilt.data.trial_subsample is None
    encoder = build_embedding_net(rebuilt.encoder, "cpu").trial_encoder
    assert encoder.freq_mode == "feature"
    assert encoder.mlp[0].in_features == 45


@needs_data
def test_a_three_transform_checkpoint_does_not_load_into_a_preset_built_flow(
    tmp_path,
):
    """The proof that the flow size has to come from the checkpoint."""
    from cancer_sbi.data.loaders import build_clone_set_dataloaders
    from cancer_sbi.training.trainer import build_training_components

    ckpt_dir, _ = _train_tiny(
        tmp_path, "clonemlp", ["--flow-num-transforms", "3"], "M3nt"
    )
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu")["model_state"]

    names = _sim_names(3)
    train_loader, _, _ = build_clone_set_dataloaders(
        str(DATA_ROOT), names, names, top_k=100, batch_size=2
    )
    preset_built = build_training_components(
        get_preset("clonemlp"), train_loader, device="cpu", log_progress=False
    ).density_estimator

    with pytest.raises(RuntimeError):
        preset_built.load_state_dict(state)

    rebuilt = build_training_components(
        posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "clonemlp").preset,
        train_loader,
        device="cpu",
        log_progress=False,
    ).density_estimator
    rebuilt.load_state_dict(state)
