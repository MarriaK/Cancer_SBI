"""The 2026-09-25 default change: ``--published``, and what the job scripts run.

Run from ``src/``::

    python -m pytest tests/test_repaired_defaults.py -q

On 2026-09-25 ``--model clonemlp`` (and ``cloneatt``, and ``dominantclone``)
stopped meaning "the model as published" and started meaning the best honest
configuration the 2026-09-24 campaign found -- R2, R26 and D0 respectively. The
published presets are preserved under ``<name>_published`` and are reached with
``--published``.

Three things have to hold for that change to be safe, and each is a section
below:

1. ``--published`` exists on all three entry points, means the same thing on
   each, and is refused for the two models that were never published.
2. A checkpoint records the RESOLVED preset name, so evaluation rebuilds what
   training built rather than what today's defaults say.
3. **Every campaign run in ``jobs/train*.sh`` still produces the config it
   actually produced.** The effective configs of those runs live in
   ``runs/.../checkpoints/best.pt`` on the cluster and cannot be read from
   here, so section 3 reconstructs each one the only other way there is: it
   takes the command line the script's ``DRY_RUN=1`` prints and puts it through
   the real ``build_parser`` / ``build_config``, then asserts the fields the
   run was written against. A missing ``--published`` on any of those rows
   shows up here as a changed field.

Nothing here needs a GPU, a dataset or a checkpoint except the one end-to-end
test at the bottom, which is skipped when the local simulation tree is absent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CODE_ROOT = SRC.parent
JOBS = SRC / "jobs"
DATA_ROOT = CODE_ROOT / "data" / "Guassian_Normal" / "simulation_outputs"

from cancer_sbi.cli.train import build_config, build_parser  # noqa: E402
from cancer_sbi.config import (  # noqa: E402
    PRESETS,
    REPAIRED_PRESETS,
    get_preset,
    published_preset_name,
    resolve_model_name,
)

needs_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="local simulation tree not present"
)


# ---------------------------------------------------------------------------
# 1. --published, on all three entry points.
# ---------------------------------------------------------------------------


def _train_parser_args(argv):
    return build_parser().parse_args(argv)


def test_published_maps_the_three_models_onto_their_twins():
    for name in REPAIRED_PRESETS:
        assert resolve_model_name(name, published=True) == name + "_published"
        assert resolve_model_name(name, published=False) == name
        # Already published: the flag is a no-op, not a double suffix.
        assert resolve_model_name(name + "_published", published=True) == name + "_published"
        assert published_preset_name(name) == name + "_published"


@pytest.mark.parametrize("model", ["armtoken", "hybrid"])
def test_published_is_refused_for_the_models_that_were_never_published(model):
    with pytest.raises(KeyError, match="no published version"):
        resolve_model_name(model, published=True)


def test_the_published_presets_are_in_the_lookup_and_are_their_own_models():
    for name in REPAIRED_PRESETS:
        twin = PRESETS[name + "_published"]
        assert twin.name == name + "_published"
        # paper_name is the manuscript's and does NOT get a suffix.
        assert twin.paper_name == PRESETS[name].paper_name
        assert twin.origin == PRESETS[name].origin


def test_train_cli_accepts_published_and_the_suffixed_model_name():
    parser = build_parser()
    assert parser.parse_args(["--model", "clonemlp"]).published is False
    assert parser.parse_args(["--model", "clonemlp", "--published"]).published is True
    # And the suffixed name is a --model choice in its own right.
    assert parser.parse_args(["--model", "clonemlp_published"]).model == "clonemlp_published"


def test_evaluate_cli_accepts_published_and_the_suffixed_model_name():
    from cancer_sbi.cli.evaluate import build_parser as eval_parser

    parser = eval_parser()
    assert parser.parse_args(["--model", "cloneatt"]).published is False
    assert parser.parse_args(["--model", "cloneatt", "--published"]).published is True
    assert parser.parse_args(["--model", "cloneatt_published"]).model == "cloneatt_published"


def test_sample_posteriors_accepts_published_and_the_suffixed_model_name():
    """--help must keep working without torch, so the parser is built by hand."""
    r = subprocess.run(
        [sys.executable, str(SRC / "cancer_sbi" / "evaluation" / "sample_posteriors.py"),
         "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "--published" in r.stdout
    assert "clonemlp_published" in r.stdout


@pytest.mark.parametrize("model", list(REPAIRED_PRESETS))
def test_published_and_the_suffixed_name_build_the_same_config(model):
    def _cfg(argv):
        args = build_parser().parse_args(argv)
        return build_config(
            args,
            get_preset(resolve_model_name(args.model, args.published)),
            data_root=Path("/nonexistent/data"),
            split_path=Path("/nonexistent/split.pkl"),
            ckpt_dir=Path("/nonexistent/ckpt"),
        )

    via_flag = _cfg(["--model", model, "--published"])
    via_name = _cfg(["--model", model + "_published"])
    assert via_flag == via_name
    assert via_flag.name == model + "_published"
    # ... and it is NOT the repaired default.
    assert via_flag != _cfg(["--model", model])


# ---------------------------------------------------------------------------
# 2. The checkpoint records the RESOLVED name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", list(REPAIRED_PRESETS))
def test_the_effective_config_records_the_resolved_preset_name(model):
    from cancer_sbi.cli.train import effective_config_payload

    for argv, expected in (
        ([], model),
        (["--published"], model + "_published"),
    ):
        args = build_parser().parse_args(["--model", model] + argv)
        cfg = build_config(
            args,
            get_preset(resolve_model_name(args.model, args.published)),
            data_root=Path("/nonexistent/data"),
            split_path=Path("/nonexistent/split.pkl"),
            ckpt_dir=Path("/nonexistent/ckpt"),
        )
        payload = effective_config_payload(
            cfg, args, Path("/nonexistent/data"), Path("/nonexistent/split.pkl")
        )
        assert payload["model"] == expected
        # The flag as typed rides in beside the resolved name, so a checkpoint
        # says both what was asked for and what it got.
        assert payload["cli_flags"]["published"] is bool(argv)
        # It must round-trip: this is what evaluation rebuilds from.
        from cancer_sbi.config import preset_from_effective_config

        assert preset_from_effective_config(payload).name == expected


# ---------------------------------------------------------------------------
# 3. Every campaign run still produces the config it actually produced.
#
# The expected values below are "the published preset, plus the flags the row
# names" -- reasoned from the scripts and from the presets as they stood before
# 2026-09-25, because the runs' own effective configs are on the cluster.
# ---------------------------------------------------------------------------


def _dry_run_commands(script):
    env = dict(os.environ)
    env.update({"DRY_RUN": "1", "CANCER": str(CODE_ROOT)})
    r = subprocess.run(
        ["bash", str(JOBS / script)], capture_output=True, text=True, env=env, timeout=120
    )
    assert r.returncode == 0, r.stderr
    return [ln for ln in r.stdout.splitlines() if ln.startswith("python ")]


def _config_from_command(line):
    """Put one dry-run command line through the real flag-to-config path."""
    argv = line.split()
    assert argv[:3] == ["python", "-m", "cancer_sbi.cli.train"], argv[:3]
    args = build_parser().parse_args(argv[3:])
    return build_config(
        args,
        get_preset(resolve_model_name(args.model, args.published)),
        data_root=Path(args.data_root),
        split_path=Path(args.split),
        ckpt_dir=Path(args.ckpt_dir),
    )


#: script -> {run name: (index into the script's run list, expected fields)}.
#:
#: Ten runs spread over the five scripts: the three models, both clone-set
#: encoders, and each of the five fields the repaired presets changed, so that
#: a missing ``--published`` anywhere shows up on at least one row.
CAMPAIGN_RUNS = {
    "train.sh": {
        # R0 is the control: the published CloneMLP with every code fix and no
        # model change. It names --z-score-x none but nothing else, so a
        # missing --published would give it R2's copy-space encoder.
        "R0": (0, dict(model="clonemlp_published", z_score_x="none", input_space="log2",
                       freq_mode="weight", num_transforms=5, tail_bound=3.0,
                       d_model=128, attn_ln=False, require_all_trials=False)),
        "R1": (1, dict(model="clonemlp_published", z_score_x="structured",
                       input_space="log2", num_transforms=5, tail_bound=3.0,
                       d_model=128, require_all_trials=False)),
        "R2": (2, dict(model="clonemlp_published", z_score_x="structured",
                       input_space="copy", num_transforms=5, tail_bound=3.0,
                       d_model=128, require_all_trials=False)),
        # R4 names --freq-renorm, which the repaired cloneatt refuses outright
        # (its freq_mode is "feature"); with --published the mode is "weight"
        # again and the run is exactly what it was.
        "R4": (3, dict(model="cloneatt_published", z_score_x="structured",
                       input_space="log2", freq_mode="weight", freq_renorm=True,
                       attn_ln=False, num_transforms=5, tail_bound=3.0, d_model=128)),
        # D0 names --require-all-trials explicitly, so it lands on True either
        # way -- but everything else about it must still be the published
        # DeepSet, which is the point of the flag here.
        "D0": (4, dict(model="dominantclone_published", z_score_x="structured",
                       require_all_trials=True, num_transforms=5, tail_bound=3.0)),
    },
    "train2.sh": {
        # R5: frequency as a feature + LayerNorm, and the ONE cloneatt run in
        # the campaign that stayed in log2 space.
        "R5": (3, dict(model="cloneatt_published", z_score_x="structured",
                       input_space="log2", freq_mode="feature", attn_ln=True,
                       num_transforms=5, tail_bound=3.0, d_model=128)),
        "R6": (4, dict(model="cloneatt_published", z_score_x="structured",
                       input_space="copy", freq_mode="feature", attn_ln=True,
                       num_transforms=5, tail_bound=3.0, d_model=128)),
    },
    "train3.sh": {
        # R12: R6 plus the 3-transform flow. tail_bound is still sbi's 3.0 --
        # R18 has not happened yet at this point in the campaign.
        "R12": (9, dict(model="cloneatt_published", z_score_x="structured",
                        input_space="copy", freq_mode="feature", attn_ln=True,
                        num_transforms=3, tail_bound=3.0, d_model=128)),
    },
    "train4.sh": {
        # R18: the tail bound, alone. d_model is still 128.
        "R18": (1, dict(model="cloneatt_published", z_score_x="structured",
                        input_space="copy", freq_mode="feature", attn_ln=True,
                        num_transforms=3, tail_bound=5.0, d_model=128,
                        attn_scale="published", trial_pool="mean")),
    },
    "train4b.sh": {
        # R26: the run the repaired `cloneatt` preset now reproduces.
        "R26": (2, dict(model="cloneatt_published", z_score_x="structured",
                        input_space="copy", freq_mode="feature", attn_ln=True,
                        num_transforms=3, tail_bound=5.0, d_model=256,
                        n_heads=8, num_inducing=32, freq_renorm=False)),
    },
}

_FLOW_FIELDS = {"z_score_x", "num_transforms", "tail_bound"}
_DATA_FIELDS = {"require_all_trials"}


@pytest.mark.parametrize(
    "script, run",
    [(s, r) for s, runs in CAMPAIGN_RUNS.items() for r in runs],
)
def test_each_campaign_run_still_builds_the_config_it_ran_with(script, run):
    index, expected = CAMPAIGN_RUNS[script][run]
    commands = _dry_run_commands(script)
    cfg = _config_from_command(commands[index])
    # The run really is the one this row names.
    assert f"/{run}/checkpoints" in commands[index], commands[index]

    for field, value in expected.items():
        if field == "model":
            got = cfg.name
        elif field in _FLOW_FIELDS:
            got = getattr(cfg.flow, field)
        elif field in _DATA_FIELDS:
            got = getattr(cfg.data, field)
        else:
            got = getattr(cfg.encoder, field)
        assert got == value, f"{script}:{run}.{field}: {got!r} != {value!r}"


def test_r26_from_train4b_is_exactly_the_repaired_cloneatt_preset():
    """The claim the `cloneatt` preset makes, checked against the run itself.

    train4b.sh's R26 row is "the published CloneAtt plus seven flags"; the
    repaired preset is "the published CloneAtt with the same seven fields
    replaced". They must be the same network, or the preset is documenting a
    run that was never made.
    """
    cfg = _config_from_command(_dry_run_commands("train4b.sh")[2])
    repaired = get_preset("cloneatt")
    assert cfg.flow == repaired.flow
    assert cfg.encoder == repaired.encoder
    # The same for R2 and D0, whose rows are shorter.
    r2 = _config_from_command(_dry_run_commands("train.sh")[2])
    assert r2.flow == get_preset("clonemlp").flow
    assert r2.encoder == get_preset("clonemlp").encoder
    d0 = _config_from_command(_dry_run_commands("train.sh")[4])
    assert d0.data.require_all_trials == get_preset("dominantclone").data.require_all_trials
    assert d0.encoder == get_preset("dominantclone").encoder


def test_every_run_in_the_five_campaign_scripts_passes_published():
    """The audit, script by script: no row may have been missed.

    Every run in these five scripts is one of the three published models and
    was written against the published preset defaults, so every row needs the
    flag -- there is no row for which adding it would be wrong.
    """
    for script in ("train.sh", "train2.sh", "train3.sh", "train4.sh", "train4b.sh"):
        commands = _dry_run_commands(script)
        assert commands, script
        for line in commands:
            assert "--published" in line, f"{script}: {line}"
            model = line.split("--model ")[1].split()[0]
            assert model in REPAIRED_PRESETS, f"{script}: {model}"


def test_the_armtoken_scripts_are_unaffected_and_name_their_tail_bound():
    """train5-8 are armtoken/hybrid: no published twin, and no reliance on 3.0.

    They pass ``--tail-bound 5`` on every command line, so the ArmToken
    preset's new 5.0 changes nothing about what they run -- which is why they
    are deliberately left alone.
    """
    for script in ("train5.sh", "train6.sh", "train7.sh", "train8.sh"):
        for line in _dry_run_commands(script):
            assert "--tail-bound 5" in line, f"{script}: {line}"
            assert "--published" not in line, f"{script}: {line}"
            model = line.split("--model ")[1].split()[0]
            assert model in ("armtoken", "hybrid", "cloneatt"), f"{script}: {model}"
            cfg = _config_from_command(line)
            assert cfg.flow.tail_bound == 5.0


def test_jobs_readme_records_the_default_change():
    text = (JOBS / "README.md").read_text()
    assert "2026-09-25" in text
    assert "--published" in text
    # The two things a reader gets wrong otherwise: --published moves no
    # checkpoint (the cluster's published files stay under the bare name), and
    # a --published run's checkpoint needs MODEL=<name>_published downstream.
    assert "selects the preset, not a path" in text
    assert "runs/clonemlp/checkpoints/best.pt" in text
    for script in ("sample.sh", "recalibrate.sh", "ensemble.sh", "finaltest.sh",
                   "treetest.sh"):
        assert script in text, script
    assert "MODEL=<name>_published" in text


# ---------------------------------------------------------------------------
# 4. End to end: a default cloneatt run, trained and then read back.
# ---------------------------------------------------------------------------


@needs_data
def test_a_default_cloneatt_run_rebuilds_as_r26(tmp_path, capsys):
    """One epoch with no flags at all, then the resolver's own report line."""
    import pickle

    import numpy as np

    from cancer_sbi.cli import train as train_cli
    from cancer_sbi.evaluation import posterior as posterior_mod

    names = sorted(
        (p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim")),
        key=lambda n: int(n[3:]),
    )[:6]
    split_path = tmp_path / "split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump(
            {
                "train_ids": np.array(names[:3]),
                "val_ids": np.array(names[3:4]),
                "test_ids": np.array(names[4:6]),
            },
            handle,
        )

    ckpt_dir = tmp_path / "att" / "checkpoints"
    rc = train_cli.main(
        [
            "--model", "cloneatt",          # no repair flags: the new default
            "--data-root", str(DATA_ROOT),
            "--split", str(split_path),
            "--out", str(tmp_path / "att"),
            "--ckpt-dir", str(ckpt_dir),
            "--device", "cpu",
            "--batch-size", "2",
            "--max-epochs", "1",
            "--min-epochs", "1",
            "--stop-after-epochs", "1",
            "--seed", "20260924",
            "--no-final-pickle",
        ]
    )
    assert rc == 0

    capsys.readouterr()  # drop the training log
    resolved = posterior_mod.resolve_eval_config(ckpt_dir / "best.pt", "cloneatt")
    line = [
        ln for ln in capsys.readouterr().out.splitlines()
        if ln.startswith("[config] rebuilt")
    ]
    assert len(line) == 1, "the resolver must say what it rebuilt"
    for fragment in (
        "input_space=copy",
        "freq_mode=feature",
        "attn_ln=True",
        "num_transforms=3",
        "tail_bound=5.0",
        "d_model=256",
        "z_score_x=structured",
    ):
        assert fragment in line[0], (fragment, line[0])

    assert resolved.from_checkpoint is True
    assert resolved.preset.flow == get_preset("cloneatt").flow
    assert resolved.preset.encoder == get_preset("cloneatt").encoder
