"""WP-D: the evaluation output names and the SLURM job scripts.

Two things are checked here, both cheap and both about failures that are silent on
the cluster:

1. ``sample_posteriors.output_filename`` - a ``--limit`` smoke test must not land on
   the real ``posteriors_<model>.npz`` of a full run.
2. Every ``jobs/*.sh`` parses, activates the env the same way, logs to an absolute
   path, and - for the two scripts that take per-run overrides - builds the command
   we think it builds. ``DRY_RUN=1`` makes that testable without conda or SLURM.

Nothing here imports torch: the filename helper is pure, and the CLI's ``--help`` is
run under ``/usr/bin/python3`` precisely to prove the heavy imports stay behind
``parse_args``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
JOBS = SRC / "jobs"
SAMPLE_POSTERIORS = SRC / "cancer_sbi" / "evaluation" / "sample_posteriors.py"

SHELL_SCRIPTS = sorted(JOBS.glob("*.sh"))


def _load_output_filename():
    """Import only the helper, without importing the package (which needs torch)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_wp_d_sp", SAMPLE_POSTERIORS)
    # The module's heavy imports live inside main(), so a plain exec is safe.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.output_filename


def _run(script, env=None, args=()):
    full = dict(os.environ)
    full.update(env or {})
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True, text=True, env=full, timeout=120,
    )


# --------------------------------------------------------------- output filename
@pytest.mark.parametrize(
    "limit, run_tag, expected",
    [
        (None, None, "posteriors_clonemlp.npz"),
        (4, None, "posteriors_clonemlp_limit4.npz"),
        (None, "R1", "posteriors_clonemlp_R1.npz"),
    ],
)
def test_output_filename(limit, run_tag, expected):
    assert _load_output_filename()("clonemlp", limit, run_tag) == expected


def test_output_filename_tag_wins_over_limit():
    fn = _load_output_filename()
    assert fn("cloneatt", 4, "R4") == "posteriors_cloneatt_R4.npz"


def test_sample_posteriors_help_runs_without_torch():
    """--help must work on a machine that cannot import torch (e.g. a login node)."""
    interpreter = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
    r = subprocess.run([interpreter, str(SAMPLE_POSTERIORS), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--run-tag" in r.stdout
    assert "--limit" in r.stdout


# ------------------------------------------------------------------ job scripts
def test_jobs_dir_has_the_expected_scripts():
    names = {p.name for p in SHELL_SCRIPTS}
    assert {"train.sh", "sample.sh", "analyze.sh", "shrink.sh",
            "treetest.sh", "finaltest.sh", "verify.sh"} <= names
    assert (JOBS / "README.md").exists()


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_ld_library_path_export_follows_conda_activate(script):
    """Without this export scipy dies with a GLIBCXX_3.4.30 ImportError."""
    lines = script.read_text().splitlines()
    activate = [i for i, ln in enumerate(lines) if ln.strip().startswith("conda activate")]
    export = [i for i, ln in enumerate(lines) if "LD_LIBRARY_PATH=" in ln and "CONDA_PREFIX" in ln]
    assert activate, "no `conda activate`"
    assert export, "no LD_LIBRARY_PATH export"
    assert min(export) > min(activate), "the export must come after `conda activate`"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_sbatch_log_paths_are_absolute(script):
    """Relative -o/-e paths land wherever sbatch was run from, or nowhere."""
    found = False
    for line in script.read_text().splitlines():
        for flag in ("#SBATCH -o ", "#SBATCH -e "):
            if line.startswith(flag):
                found = True
                assert line[len(flag):].strip().startswith("/"), line
    assert found, "no #SBATCH -o/-e lines"


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_scripts_use_the_cluster_tree(script):
    """There is no code/ level on the cluster, and the data lives under data/."""
    text = script.read_text()
    assert "cancer/code" not in text
    if "Guassian_Normal" in text:
        assert "data/Guassian_Normal" in text


# ------------------------------------------------------------------- train.sh
def test_train_sh_mentions_all_five_runs_and_min_epochs():
    text = (JOBS / "train.sh").read_text()
    for run in ("R0", "R1", "R2", "R4", "D0"):
        assert run in text
    assert "--min-epochs 1" in text, "without it --stop-after-epochs is a no-op (A2)"
    assert "--array=0-4" in text
    assert "-C a100" in text
    assert "general-gpu" in text
    assert "--gres=gpu:1" in text
    # 12 h, not the first draft's 8: 60 epochs at ~28 min/epoch uncached does
    # not fit in eight hours, and a killed array task loses the whole run.
    assert "-t 12:00:00" in text


def test_train_sh_refuses_a_non_empty_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "R1" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_text("not really a checkpoint")
    r = _run(JOBS / "train.sh",
             {"RUNS_ROOT": str(tmp_path), "SLURM_ARRAY_TASK_ID": "1"})
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


#: A ``CANCER`` pointing at this checkout, so the dry run resolves the real
#: three-key split instead of the cluster's ``$HOME/cancer``.
CODE_ROOT = SRC.parent


def _train_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train_dry_run_lines():
    return [
        ln for ln in _train_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


def test_train_sh_is_executable():
    """/home is noexec on the cluster, but the bit still has to be set here."""
    assert os.access(JOBS / "train.sh", os.X_OK)


def test_train_sh_defaults_to_the_three_key_split():
    text = (JOBS / "train.sh").read_text()
    assert "train_val_test_split.pkl" in text
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train_dry_run_reports_the_split_gate_as_ok():
    """The repo's own three-key split passes the gate."""
    out = _train_dry_run().stdout
    assert "split gate: OK" in out, out


def test_train_dry_run_reports_a_two_key_split_as_refused():
    out = _train_dry_run(
        {"CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl")}
    ).stdout
    assert "split gate: would REFUSE" in out, out


def test_train_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    """A real (non-dry) run must stop before it trains on the test set."""
    r = _run(
        JOBS / "train.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "1",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr
    assert "val_ids" in r.stderr
    assert "--add-val" in r.stderr


def test_allow_test_as_val_waives_the_gate(tmp_path):
    """The escape hatch exists, and it is the only thing that opens the gate."""
    r = _run(
        JOBS / "train.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "ALLOW_TEST_AS_VAL": "1",
            "DRY_RUN": "1",
        },
    )
    assert r.returncode == 0, r.stderr
    assert "split gate: WAIVED" in r.stdout


def test_train_dry_run_prints_five_commands():
    cmds = _train_dry_run_lines()
    assert len(cmds) == 5


def test_train_dry_run_flags_per_run():
    r0, r1, r2, r4, d0 = _train_dry_run_lines()

    for cmd in (r0, r1, r2, r4, d0):
        assert "-m cancer_sbi.cli.train" in cmd
        assert "--seed 20260924" in cmd
        assert "--min-epochs 1" in cmd
        assert "--stop-after-epochs 15" in cmd
        assert "--max-epochs 60" in cmd
        assert "--num-workers 8" in cmd
        assert "--ckpt-dir" in cmd and "/runs/2026-09-24/" in cmd
        assert "/checkpoints" in cmd

    assert "--model clonemlp" in r0 and "--z-score-x none" in r0
    assert "--model clonemlp" in r1 and "--z-score-x structured" in r1
    assert "--model clonemlp" in r2 and "--input-space copy" in r2
    assert "--model cloneatt" in r4 and "--freq-renorm" in r4
    assert "--model dominantclone" in d0 and "--require-all-trials" in d0

    # copy space is R2 only; the frequency repair is R4 only
    runs = (r0, r1, r2, r4, d0)
    assert sum("--input-space copy" in c for c in runs) == 1
    assert sum("--freq-renorm" in c for c in runs) == 1
    assert sum("--require-all-trials" in c for c in runs) == 1

    # D0 carries none of the clone-set-only flags, and no --z-score-x at all:
    # the dominantclone preset is already "structured", and naming it here
    # would make a preset default look like a per-run choice.
    for flag in ("--input-space", "--freq-renorm", "--z-score-x"):
        assert flag not in d0, d0

    # each run gets its own checkpoint directory
    dirs = []
    for name, cmd in zip(("R0", "R1", "R2", "R4", "D0"), runs):
        parts = cmd.split()
        d = parts[parts.index("--ckpt-dir") + 1]
        assert d.endswith(f"/{name}/checkpoints")
        dirs.append(d)
    assert len(set(dirs)) == 5


def test_train_dry_run_never_gives_the_clone_cache_to_dominantclone():
    """The cache holds (25, top_k, 45) clone sets; DominantClone reads none."""
    cmds = [
        ln
        for ln in _train_dry_run({"CACHE_DIR": "/some/cache"}).stdout.splitlines()
        if ln.startswith("python ")
    ]
    assert len(cmds) == 5
    assert sum("--cache-dir /some/cache" in c for c in cmds) == 4
    assert "--cache-dir" not in cmds[4], cmds[4]


# ------------------------------------------------------- sample / analyze / shrink
def test_sample_sh_honours_ckpt_and_post():
    r = _run(JOBS / "sample.sh",
             {"CKPT": "/x/best.pt", "POST": "/y", "DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    cmd = r.stdout.strip()
    assert "-m cancer_sbi.evaluation.sample_posteriors" in cmd
    assert "--ckpt /x/best.pt" in cmd
    assert "--out-dir /y" in cmd


def test_sample_sh_passes_run_tag_and_extra_last():
    r = _run(JOBS / "sample.sh",
             {"POST": "/y", "RUN_TAG": "R1", "EXTRA": "--limit 4", "DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    cmd = r.stdout.strip()
    assert "--run-tag R1" in cmd
    # argparse is last-wins, so EXTRA must be able to override what came before
    assert cmd.rstrip().endswith("--limit 4")


def test_sample_sh_model_env_overrides_the_array_index():
    r = _run(JOBS / "sample.sh",
             {"MODEL": "cloneatt", "SLURM_ARRAY_TASK_ID": "0", "POST": "/y", "DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert "--model cloneatt" in r.stdout


def test_sample_sh_refuses_a_ckpt_with_a_multi_task_array():
    r = _run(JOBS / "sample.sh",
             {"CKPT": "/x/best.pt", "SLURM_ARRAY_TASK_ID": "1", "SLURM_ARRAY_TASK_COUNT": "3",
              "POST": "/y", "DRY_RUN": "1"})
    assert r.returncode == 1
    assert "REFUSING" in r.stderr
    # A single-task array with a CKPT is the documented per-run path and must still work.
    ok = _run(JOBS / "sample.sh",
              {"CKPT": "/x/best.pt", "SLURM_ARRAY_TASK_ID": "1", "SLURM_ARRAY_TASK_COUNT": "1",
               "POST": "/y", "DRY_RUN": "1"})
    assert ok.returncode == 0 and "--model cloneatt" in ok.stdout


def test_sample_sh_wall_clock_is_above_the_estimate():
    assert "-t 12:00:00" in (JOBS / "sample.sh").read_text()


@pytest.mark.parametrize("name", ["analyze.sh", "shrink.sh"])
def test_analyze_and_shrink_honour_post_and_out(name):
    r = _run(JOBS / name, {"POST": "/p", "OUT": "/o", "DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert "--in-dir /p" in r.stdout
    assert "--out-dir /o" in r.stdout


@pytest.mark.parametrize("name", ["analyze.sh", "shrink.sh"])
def test_analyze_and_shrink_pass_run_tag_through(name):
    r = _run(JOBS / name, {"POST": "/p", "OUT": "/o", "RUN_TAG": "R1", "DRY_RUN": "1"})
    assert r.returncode == 0, r.stderr
    assert "--run-tag R1" in r.stdout
    r = _run(JOBS / name, {"POST": "/p", "OUT": "/o", "DRY_RUN": "1"})
    assert "--run-tag" not in r.stdout


def test_metric_scripts_find_tagged_posterior_files(tmp_path):
    """sample_posteriors --run-tag R1 writes posteriors_<m>_R1.npz; both metric
    scripts must be able to read exactly that name back, or a matrix run is
    sampled and then never measured."""
    from cancer_sbi.evaluation import poster_metrics
    from cancer_sbi.evaluation.sample_posteriors import output_filename
    tagged = output_filename("clonemlp", None, "R1")
    (tmp_path / tagged).write_bytes(b"")
    assert poster_metrics.posterior_path(str(tmp_path), "clonemlp", "R1") == str(tmp_path / tagged)
    assert os.path.exists(poster_metrics.posterior_path(str(tmp_path), "clonemlp", "R1"))
    assert not os.path.exists(poster_metrics.posterior_path(str(tmp_path), "clonemlp"))


def test_shrink_dry_run_covers_the_poster_build():
    r = _run(JOBS / "shrink.sh", {"POST": "/p", "OUT": "/o", "DRY_RUN": "1"})
    assert any(ln.endswith("--poster") for ln in r.stdout.splitlines())


# ------------------------------------------------------------- the stale scripts
@pytest.mark.parametrize("name", ["treetest.sh", "finaltest.sh"])
def test_evaluation_scripts_are_invoked_as_modules(name):
    text = (JOBS / name).read_text()
    assert "src/evaluation/" not in text, "pre-reorg path"
    assert "cancer_sbi.evaluation.sample_posteriors" in text


def test_finaltest_uses_poster_metrics_not_analyze():
    text = (JOBS / "finaltest.sh").read_text()
    assert "analyze.py" not in text
    assert "cancer_sbi.evaluation.poster_metrics" in text


def test_verify_sh_points_at_the_moved_script_and_the_archive():
    text = (JOBS / "verify.sh").read_text()
    assert "verify_refactor.py" in text
    assert "/src" in text
    assert "_archive_2026-09-23" in text
    assert "--legacy-root" in text
