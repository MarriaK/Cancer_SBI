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
# Only scripts meant for `sbatch` carry a #SBATCH header; helpers that are run with
# `bash` on the login node (chain_eval.sh submits other jobs) are linted for syntax
# only, not for the conda/LD_LIBRARY_PATH preamble or absolute log paths.
SBATCH_SCRIPTS = [p for p in SHELL_SCRIPTS if "#SBATCH" in p.read_text()]


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
    assert {"train.sh", "train2.sh", "train3.sh", "sample.sh", "analyze.sh", "shrink.sh",
            "treetest.sh", "finaltest.sh", "verify.sh"} <= names
    assert (JOBS / "README.md").exists()


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", SBATCH_SCRIPTS, ids=lambda p: p.name)
def test_ld_library_path_export_follows_conda_activate(script):
    """Without this export scipy dies with a GLIBCXX_3.4.30 ImportError."""
    lines = script.read_text().splitlines()
    activate = [i for i, ln in enumerate(lines) if ln.strip().startswith("conda activate")]
    export = [i for i, ln in enumerate(lines) if "LD_LIBRARY_PATH=" in ln and "CONDA_PREFIX" in ln]
    assert activate, "no `conda activate`"
    assert export, "no LD_LIBRARY_PATH export"
    assert min(export) > min(activate), "the export must come after `conda activate`"


@pytest.mark.parametrize("script", SBATCH_SCRIPTS, ids=lambda p: p.name)
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


# ------------------------------------------------------------------ train2.sh
#
# Matrix 2: seven runs, four new switches. The dry run is the only place the
# per-run flag sets can be checked without a GPU, and a wrong flag here is a
# 12-hour array task that answers a question nobody asked.


def _train2_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train2.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train2_dry_run_lines():
    return [
        ln for ln in _train2_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


def _seed_of(cmd):
    """The value of --seed in one dry-run command line."""
    parts = cmd.split()
    return parts[parts.index("--seed") + 1]


def test_train2_sh_header_matches_the_matrix():
    text = (JOBS / "train2.sh").read_text()
    for run in ("R3", "R2s1", "R2s2", "R5", "R6", "R7", "R8"):
        assert run in text
    assert "--array=0-6" in text
    assert "--min-epochs 1" in text
    assert "-C a100" in text
    assert "general-gpu" in text
    assert "--gres=gpu:1" in text
    assert "-t 12:00:00" in text
    # train.sh is the first matrix and must not have been edited into this one.
    assert "--array=0-4" in (JOBS / "train.sh").read_text()


def test_train2_sh_is_executable():
    assert os.access(JOBS / "train2.sh", os.X_OK)


def test_train2_sh_defaults_to_the_three_key_split():
    text = (JOBS / "train2.sh").read_text()
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train2_dry_run_reports_the_split_gate_as_ok():
    assert "split gate: OK" in _train2_dry_run().stdout


def test_train2_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    r = _run(
        JOBS / "train2.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr and "val_ids" in r.stderr


def test_train2_sh_refuses_a_non_empty_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "R5" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_text("not really a checkpoint")
    r = _run(JOBS / "train2.sh",
             {"RUNS_ROOT": str(tmp_path), "SLURM_ARRAY_TASK_ID": "3"})
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


def test_train2_dry_run_prints_seven_commands():
    assert len(_train2_dry_run_lines()) == 7


def test_train2_dry_run_flags_per_run():
    r3, r2s1, r2s2, r5, r6, r7, r8 = _train2_dry_run_lines()
    runs = (r3, r2s1, r2s2, r5, r6, r7, r8)

    for cmd in runs:
        assert "-m cancer_sbi.cli.train" in cmd
        assert "--min-epochs 1" in cmd
        assert "--stop-after-epochs 15" in cmd
        assert "--max-epochs 60" in cmd
        assert "--num-workers 8" in cmd
        assert "--z-score-x structured" in cmd
        assert "--ckpt-dir" in cmd and "/runs/2026-09-24/" in cmd

    # R3: weight decay on both groups, and the matrix seed, not a repeat seed.
    assert "--model clonemlp" in r3
    assert "--input-space copy" in r3
    assert "--flow-weight-decay 1e-3" in r3
    assert "--embed-weight-decay 1e-4" in r3
    # By token, not substring: "--seed 20260924" contains "--seed 2".
    assert _seed_of(r3) == "20260924"

    # R2s1/R2s2: R2 again, only the seed differs.
    for cmd, seed in ((r2s1, "1"), (r2s2, "2")):
        assert "--model clonemlp" in cmd
        assert "--input-space copy" in cmd
        assert _seed_of(cmd) == seed
        for flag in ("--flow-weight-decay", "--embed-weight-decay", "--embed-lr",
                     "--freq-mode", "--attn-ln", "--encoder-dropout"):
            assert flag not in cmd, cmd

    # R5-R8: the CloneAtt ladder, one switch added at a time.
    for cmd in (r5, r6, r7, r8):
        assert "--model cloneatt" in cmd
        assert "--freq-mode feature" in cmd
        assert "--attn-ln" in cmd
        assert _seed_of(cmd) == "20260924"
        # Refused by build_config, so it must never appear beside --freq-mode.
        assert "--freq-renorm" not in cmd

    assert "--input-space" not in r5, "R5 is the one CloneAtt run without T2"
    assert "--input-space copy" in r6
    assert "--embed-lr" not in r6
    assert "--embed-lr 5e-4" in r7 and "--encoder-dropout" not in r7
    # R8 is R7 plus dropout: all four CloneAtt switches at once.
    for flag in ("--freq-mode feature", "--attn-ln", "--input-space copy",
                 "--embed-lr 5e-4", "--encoder-dropout 0.1"):
        assert flag in r8, r8

    # Each run gets its own checkpoint directory.
    dirs = []
    for name, cmd in zip(("R3", "R2s1", "R2s2", "R5", "R6", "R7", "R8"), runs):
        parts = cmd.split()
        d = parts[parts.index("--ckpt-dir") + 1]
        assert d.endswith(f"/{name}/checkpoints")
        dirs.append(d)
    assert len(set(dirs)) == 7


def test_train2_gives_every_run_the_clone_cache():
    """All seven are clone-set models -- there is no DominantClone exception."""
    cmds = [
        ln
        for ln in _train2_dry_run({"CACHE_DIR": "/some/cache"}).stdout.splitlines()
        if ln.startswith("python ")
    ]
    assert len(cmds) == 7
    assert all("--cache-dir /some/cache" in c for c in cmds)


def test_jobs_readme_documents_matrix_two():
    text = (JOBS / "README.md").read_text()
    assert "train2.sh" in text
    assert "Matrix 2" in text
    for run in ("R3", "R2s1", "R2s2", "R5", "R6", "R7", "R8"):
        assert run in text


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


# ------------------------------------------------------------------ train3.sh
#
# Matrix 3: fourteen runs, three new switches plus --freq-mode for clonemlp.
# Same reasoning as train2.sh -- the dry run is the only place the per-run flag
# sets can be checked without a GPU.


def _train3_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train3.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train3_dry_run_lines():
    return [
        ln for ln in _train3_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


RUN3_NAMES = ("R3s1", "R3s2", "R6s1", "R6s2", "D0s1", "D0s2",
              "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16")


def test_train3_sh_header_matches_the_matrix():
    text = (JOBS / "train3.sh").read_text()
    for run in RUN3_NAMES:
        assert run in text
    assert "--array=0-13" in text
    assert "--min-epochs 1" in text
    assert "-C a100" in text
    assert "general-gpu" in text
    assert "--gres=gpu:1" in text
    assert "-t 12:00:00" in text
    # The two earlier matrices must not have been edited into this one.
    assert "--array=0-4" in (JOBS / "train.sh").read_text()
    assert "--array=0-6" in (JOBS / "train2.sh").read_text()


def test_train3_sh_is_executable():
    assert os.access(JOBS / "train3.sh", os.X_OK)


def test_train3_sh_defaults_to_the_three_key_split():
    text = (JOBS / "train3.sh").read_text()
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train3_dry_run_reports_the_split_gate_as_ok():
    assert "split gate: OK" in _train3_dry_run().stdout


def test_train3_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    r = _run(
        JOBS / "train3.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr and "val_ids" in r.stderr


def test_train3_sh_refuses_a_non_empty_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "R11" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_text("not really a checkpoint")
    r = _run(JOBS / "train3.sh",
             {"RUNS_ROOT": str(tmp_path), "SLURM_ARRAY_TASK_ID": "8"})
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


def test_train3_dry_run_prints_fourteen_commands():
    assert len(_train3_dry_run_lines()) == 14


def test_train3_dry_run_flags_per_run():
    lines = _train3_dry_run_lines()
    (r3s1, r3s2, r6s1, r6s2, d0s1, d0s2,
     r9, r10, r11, r12, r13, r14, r15, r16) = lines

    for cmd in lines:
        assert "-m cancer_sbi.cli.train" in cmd
        assert "--min-epochs 1" in cmd
        assert "--stop-after-epochs 15" in cmd
        assert "--max-epochs 60" in cmd
        assert "--num-workers 8" in cmd
        assert "--ckpt-dir" in cmd and "/runs/2026-09-24/" in cmd

    base_r3 = ("--z-score-x structured", "--input-space copy",
               "--flow-weight-decay 1e-3", "--embed-weight-decay 1e-4")
    base_r6 = ("--z-score-x structured", "--input-space copy",
               "--freq-mode feature", "--attn-ln")

    # 0-1: R3 on two seeds, nothing else added.
    for cmd, seed in ((r3s1, "1"), (r3s2, "2")):
        assert "--model clonemlp" in cmd
        for flag in base_r3:
            assert flag in cmd, cmd
        assert _seed_of(cmd) == seed
        for flag in ("--flow-dropout", "--flow-num-transforms",
                     "--trial-subsample", "--freq-mode"):
            assert flag not in cmd, cmd

    # 2-3: R6 on two seeds.
    for cmd, seed in ((r6s1, "1"), (r6s2, "2")):
        assert "--model cloneatt" in cmd
        for flag in base_r6:
            assert flag in cmd, cmd
        assert _seed_of(cmd) == seed
        for flag in ("--flow-dropout", "--flow-num-transforms",
                     "--trial-subsample", "--flow-weight-decay"):
            assert flag not in cmd, cmd

    # 4-5: the DominantClone replicates. No --z-score-x (the preset is already
    # structured) and no cache -- asserted separately below.
    for cmd, seed in ((d0s1, "1"), (d0s2, "2")):
        assert "--model dominantclone" in cmd
        assert "--require-all-trials" in cmd
        assert _seed_of(cmd) == seed
        for flag in ("--z-score-x", "--input-space", "--freq-mode",
                     "--attn-ln", "--trial-subsample"):
            assert flag not in cmd, cmd

    # 6-13: the matrix seed, one switch added to a base each time.
    for cmd in (r9, r10, r11, r12, r13, r14, r15, r16):
        assert _seed_of(cmd) == "20260924"

    assert "--model cloneatt" in r9
    for flag in base_r6 + ("--flow-weight-decay 1e-3", "--embed-weight-decay 1e-4"):
        assert flag in r9, r9

    # R10: BASE_R3 plus the log10-frequency feature, on clonemlp.
    assert "--model clonemlp" in r10
    for flag in base_r3 + ("--freq-mode feature",):
        assert flag in r10, r10
    assert "--attn-ln" not in r10

    # R11/R12/R13: BASE_R6 plus exactly one new switch each.
    for cmd, added in ((r11, "--flow-dropout 0.3"),
                       (r12, "--flow-num-transforms 3"),
                       (r13, "--trial-subsample 16")):
        assert "--model cloneatt" in cmd
        for flag in base_r6:
            assert flag in cmd, cmd
        assert added in cmd, cmd
    assert "--flow-num-transforms" not in r11 and "--trial-subsample" not in r11
    assert "--flow-dropout" not in r12 and "--trial-subsample" not in r12
    assert "--flow-dropout" not in r13 and "--flow-num-transforms" not in r13

    # R14/R15/R16: the same three on BASE_R3 / clonemlp.
    for cmd, added in ((r14, "--flow-dropout 0.3"),
                       (r15, "--trial-subsample 16"),
                       (r16, "--flow-num-transforms 3")):
        assert "--model clonemlp" in cmd
        for flag in base_r3:
            assert flag in cmd, cmd
        assert added in cmd, cmd
        assert "--freq-mode" not in cmd, cmd
    assert "--flow-num-transforms" not in r14 and "--trial-subsample" not in r14
    assert "--flow-dropout" not in r15 and "--flow-num-transforms" not in r15
    assert "--flow-dropout" not in r16 and "--trial-subsample" not in r16

    # The two runs that carry the augmentation, and only those two.
    assert sum("--trial-subsample 16" in c for c in lines) == 2
    assert [c for c in lines if "--trial-subsample 16" in c] == [r13, r15]

    # Each run gets its own checkpoint directory.
    dirs = []
    for name, cmd in zip(RUN3_NAMES, lines):
        parts = cmd.split()
        d = parts[parts.index("--ckpt-dir") + 1]
        assert d.endswith(f"/{name}/checkpoints")
        dirs.append(d)
    assert len(set(dirs)) == 14


def test_train3_never_gives_the_clone_cache_to_dominantclone():
    """The cache holds (25, top_k, 45) clone sets; DominantClone reads none."""
    cmds = [
        ln
        for ln in _train3_dry_run({"CACHE_DIR": "/some/cache"}).stdout.splitlines()
        if ln.startswith("python ")
    ]
    assert len(cmds) == 14
    assert sum("--cache-dir /some/cache" in c for c in cmds) == 12
    for cmd in (cmds[4], cmds[5]):
        assert "--cache-dir" not in cmd, cmd


def test_jobs_readme_documents_matrix_three():
    text = (JOBS / "README.md").read_text()
    assert "train3.sh" in text
    assert "Matrix 3" in text
    for run in RUN3_NAMES:
        assert run in text
    for flag in ("--flow-dropout", "--flow-num-transforms", "--trial-subsample"):
        assert flag in text


# ------------------------------------------------------------------ train4.sh
#
# Matrix 4: twelve runs, all cloneatt, each BASE_R12 plus the one thing it
# tests. Same reasoning as train2.sh and train3.sh -- the dry run is the only
# place the per-run flag sets can be checked without a GPU.


def _train4_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train4.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train4_dry_run_lines():
    return [
        ln for ln in _train4_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


RUN4_NAMES = ("R17", "R18", "R19", "R20", "R20s1", "R21",
              "R22", "R23", "R24", "R12s1", "R12s2", "R25")

#: BASE_R12, the matrix-3 best model every run here is built on.
BASE_R12 = ("--input-space copy", "--freq-mode feature", "--attn-ln",
            "--flow-num-transforms 3")


def test_train4_sh_header_matches_the_matrix():
    text = (JOBS / "train4.sh").read_text()
    for run in RUN4_NAMES:
        assert run in text
    assert "--array=0-11" in text
    assert "--min-epochs 1" in text
    assert "-C a100" in text
    assert "general-gpu" in text
    assert "--gres=gpu:1" in text
    assert "-t 12:00:00" in text
    # The three earlier matrices must not have been edited into this one.
    assert "--array=0-4" in (JOBS / "train.sh").read_text()
    assert "--array=0-6" in (JOBS / "train2.sh").read_text()
    assert "--array=0-13" in (JOBS / "train3.sh").read_text()


def test_train4_sh_is_executable():
    assert os.access(JOBS / "train4.sh", os.X_OK)


def test_train4_sh_defaults_to_the_three_key_split():
    text = (JOBS / "train4.sh").read_text()
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train4_dry_run_reports_the_split_gate_as_ok():
    assert "split gate: OK" in _train4_dry_run().stdout


def test_train4_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    r = _run(
        JOBS / "train4.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr and "val_ids" in r.stderr


def test_train4_sh_refuses_a_non_empty_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "R20" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_text("not really a checkpoint")
    r = _run(JOBS / "train4.sh",
             {"RUNS_ROOT": str(tmp_path), "SLURM_ARRAY_TASK_ID": "3"})
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


def test_train4_dry_run_prints_twelve_commands():
    assert len(_train4_dry_run_lines()) == 12


def test_train4_dry_run_flags_per_run():
    cmds = _train4_dry_run_lines()
    (r17, r18, r19, r20, r20s1, r21, r22, r23, r24, r12s1, r12s2, r25) = cmds

    for name, cmd in zip(RUN4_NAMES, cmds):
        assert "-m cancer_sbi.cli.train" in cmd
        assert "--model cloneatt" in cmd, name
        assert "--min-epochs 1" in cmd
        assert "--stop-after-epochs 15" in cmd
        assert "--max-epochs 60" in cmd
        assert "--num-workers 8" in cmd
        for flag in BASE_R12:
            assert flag in cmd, (name, flag)
        # Refused by build_config, so it must never appear beside --freq-mode.
        assert "--freq-renorm" not in cmd
        # R19 is the one run that moves a base flag; everything else is
        # BASE_R12's structured whitening.
        expected_z = "independent" if name == "R19" else "structured"
        assert f"--z-score-x {expected_z}" in cmd, name

    # The one switch each run adds, and nothing more.
    only = {
        "R17": ["--attn-scale standard"],
        "R18": ["--tail-bound 5"],
        "R19": [],
        "R20": ["--trial-pool attention"],
        "R20s1": ["--trial-pool attention"],
        "R21": ["--d-model 256"],
        "R22": ["--n-heads 4"],
        "R23": ["--num-inducing 64"],
        "R24": ["--d-model 256", "--num-inducing 64"],
        "R12s1": [],
        "R12s2": [],
        "R25": ["--attn-scale standard", "--tail-bound 5", "--trial-pool attention"],
    }
    matrix4_flags = ["--attn-scale", "--tail-bound", "--trial-pool",
                     "--d-model", "--n-heads", "--num-inducing"]
    for name, cmd in zip(RUN4_NAMES, cmds):
        for added in only[name]:
            assert added in cmd, (name, added)
        for flag in matrix4_flags:
            if not any(a.startswith(flag + " ") for a in only[name]):
                assert flag not in cmd, (name, flag)

    # The three seed replicates, by token: "--seed 20260924" contains
    # "--seed 2".
    seeds = {name: _seed_of(cmd) for name, cmd in zip(RUN4_NAMES, cmds)}
    assert seeds["R20s1"] == "1"
    assert seeds["R12s1"] == "1"
    assert seeds["R12s2"] == "2"
    for name in ("R17", "R18", "R19", "R20", "R21", "R22", "R23", "R24", "R25"):
        assert seeds[name] == "20260924", name

    # R20s1 is R20 with a different seed and nothing else; the same for R12s1
    # and R12s2 against each other. Compared on the flags alone: --out and
    # --ckpt-dir carry the run name and are checked separately below.
    def _architecture_flags(cmd):
        parts = cmd.split()
        drop = set()
        for flag in ("--out", "--ckpt-dir", "--seed"):
            i = parts.index(flag)
            drop |= {i, i + 1}
        return [p for i, p in enumerate(parts) if i not in drop]

    assert _architecture_flags(r20) == _architecture_flags(r20s1)
    assert _architecture_flags(r12s1) == _architecture_flags(r12s2)

    # Each run gets its own checkpoint directory.
    dirs = []
    for name, cmd in zip(RUN4_NAMES, cmds):
        parts = cmd.split()
        d = parts[parts.index("--ckpt-dir") + 1]
        assert d.endswith(f"/{name}/checkpoints")
        dirs.append(d)
    assert len(set(dirs)) == 12


def test_train4_gives_every_run_the_clone_cache():
    """All twelve are cloneatt -- there is no DominantClone exception here."""
    cmds = [
        ln
        for ln in _train4_dry_run({"CACHE_DIR": "/some/cache"}).stdout.splitlines()
        if ln.startswith("python ")
    ]
    assert len(cmds) == 12
    assert all("--cache-dir /some/cache" in c for c in cmds)


def test_jobs_readme_documents_matrix_four():
    text = (JOBS / "README.md").read_text()
    assert "train4.sh" in text
    assert "Matrix 4" in text
    for run in RUN4_NAMES:
        assert run in text
    for flag in ("--attn-scale", "--tail-bound", "--trial-pool", "--d-model",
                 "--n-heads", "--num-inducing"):
        assert flag in text


# ------------------------------------------------------------------ train5.sh
#
# Matrix 5: six runs of a fourth model. Same reasoning as train2/3/4.sh -- the
# dry run is the only place the per-run flag sets can be checked without a GPU.


def _train5_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train5.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train5_dry_run_lines():
    return [
        ln for ln in _train5_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


RUN5_NAMES = ("AT0", "AT0s1", "AT1", "AT2", "AT3", "AT4")


def test_train5_sh_header_matches_the_matrix():
    text = (JOBS / "train5.sh").read_text()
    for run in RUN5_NAMES:
        assert run in text
    assert "--array=0-5" in text
    assert "--min-epochs 1" in text
    assert "-C a100" in text
    assert "general-gpu" in text
    assert "--gres=gpu:1" in text
    assert "-t 12:00:00" in text
    # The four earlier matrices must not have been edited into this one.
    assert "--array=0-4" in (JOBS / "train.sh").read_text()
    assert "--array=0-6" in (JOBS / "train2.sh").read_text()
    assert "--array=0-13" in (JOBS / "train3.sh").read_text()


def test_train5_sh_is_executable():
    assert os.access(JOBS / "train5.sh", os.X_OK)


def test_train5_sh_defaults_to_the_three_key_split():
    text = (JOBS / "train5.sh").read_text()
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train5_dry_run_reports_the_split_gate_as_ok():
    assert "split gate: OK" in _train5_dry_run().stdout


def test_train5_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    r = _run(
        JOBS / "train5.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr and "val_ids" in r.stderr


def test_train5_sh_refuses_a_non_empty_checkpoint_dir(tmp_path):
    ckpt = tmp_path / "AT1" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "best.pt").write_text("not really a checkpoint")
    r = _run(JOBS / "train5.sh",
             {"RUNS_ROOT": str(tmp_path), "SLURM_ARRAY_TASK_ID": "2"})
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


def test_train5_dry_run_prints_six_commands():
    assert len(_train5_dry_run_lines()) == 6


def test_train5_dry_run_flags_per_run():
    at0, at0s1, at1, at2, at3, at4 = _train5_dry_run_lines()

    for cmd in (at0, at0s1, at1, at2, at3, at4):
        assert "-m cancer_sbi.cli.train" in cmd
        assert "--model armtoken" in cmd
        assert "--min-epochs 1" in cmd
        # Matrix 4 R18 showed tail_bound 3 clips the well-learned arms; every
        # ArmToken run carries the proven value so it is not confounded.
        assert "--tail-bound 5" in cmd
        assert "--stop-after-epochs 15" in cmd
        assert "--max-epochs 60" in cmd
        assert "--num-workers 8" in cmd
        assert "--ckpt-dir" in cmd and "/runs/2026-09-24/" in cmd
        # The preset already carries matrices 2-4's findings, so naming them
        # here would make a preset default look like a per-run choice.
        for flag in ("--z-score-x", "--input-space", "--attn-ln",
                     "--flow-num-transforms", "--freq-mode", "--d-model",
                     "--num-inducing", "--d-arm", "--arm-num-inducing"):
            assert flag not in cmd, cmd

    # The base and its seed replicate add nothing at all.
    for cmd in (at0, at0s1):
        for flag in ("--trial-pool", "--attn-scale", "--arm-layers",
                     "--embed-lr"):
            assert flag not in cmd, cmd
    assert _seed_of(at0) == "20260924"
    assert _seed_of(at0s1) == "1"

    # One switch each, on the matrix seed.
    for cmd, added in ((at1, "--trial-pool attention"),
                       (at2, "--attn-scale standard"),
                       (at3, "--arm-layers 0"),
                       (at4, "--embed-lr 1e-3")):
        assert added in cmd, cmd
        assert _seed_of(cmd) == "20260924"
    for cmd, absent in (
        (at1, ("--attn-scale", "--arm-layers", "--embed-lr")),
        (at2, ("--trial-pool", "--arm-layers", "--embed-lr")),
        (at3, ("--trial-pool", "--attn-scale", "--embed-lr")),
        (at4, ("--trial-pool", "--attn-scale", "--arm-layers")),
    ):
        for flag in absent:
            assert flag not in cmd, cmd


def test_train5_dry_run_gives_every_run_the_clone_cache():
    """Every run is a clone-set model; there is no DominantClone exception."""
    lines = [
        ln
        for ln in _train5_dry_run({"CACHE_DIR": "/some/cache"}).stdout.splitlines()
        if ln.startswith("python ")
    ]
    assert len(lines) == 6
    assert all("--cache-dir /some/cache" in ln for ln in lines)


def test_jobs_readme_documents_matrix_five():
    text = (JOBS / "README.md").read_text()
    assert "Matrix 5" in text and "train5.sh" in text
    for run in RUN5_NAMES:
        assert run in text
    for flag in ("--trial-pool attention", "--attn-scale standard",
                 "--arm-layers 0", "--embed-lr 1e-3", "--seed 1"):
        assert flag in text


def test_metric_scripts_know_every_preset_model():
    """A posteriors_<model>_<tag>.npz for a model missing from MODEL_ORDER is
    silently skipped and the run is never measured (this happened to ArmToken
    on 2026-09-24). Every preset name must be in both scripts' tables."""
    from cancer_sbi.config import PRESETS
    from cancer_sbi.evaluation import fig_shrinkage, poster_metrics
    for mod in (poster_metrics, fig_shrinkage):
        for name in PRESETS:
            assert name in mod.MODEL_ORDER, (mod.__name__, name)
            assert name in mod.LABEL and name in mod.COLOUR, (mod.__name__, name)


# ------------------------------------------------- train7.sh / prepare_partial.sh
#
# Matrix 7: four runs on the partial-sim cache. Same reasoning as the matrices
# above -- the dry run is the only place the per-run flag sets can be checked
# without a GPU.


def _train7_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train7.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train7_dry_run_lines():
    return [
        ln for ln in _train7_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


RUN7_NAMES = ("AT10", "AT10s1", "R27", "H1")


def test_train7_sh_header_matches_the_matrix():
    text = (JOBS / "train7.sh").read_text()
    for run in RUN7_NAMES:
        assert run in text
    assert "--array=0-3" in text
    assert "-C a100" in text and "general-gpu" in text and "-t 12:00:00" in text
    # The earlier matrices must not have been edited into this one.
    assert "--array=0-5" in (JOBS / "train5.sh").read_text()
    assert "--array=0-8" in (JOBS / "train6.sh").read_text()


def test_train7_sh_is_executable_and_uses_the_three_key_split():
    assert os.access(JOBS / "train7.sh", os.X_OK)
    text = (JOBS / "train7.sh").read_text()
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train7_sh_defaults_to_the_partial_cache():
    text = (JOBS / "train7.sh").read_text()
    assert "CACHE_DIR:-$CANCER/data/cache/clone_top100_partial_v1" in text


def test_train7_dry_run_prints_four_commands_with_the_common_flags():
    lines = _train7_dry_run_lines()
    assert len(lines) == 4
    for line in lines:
        for flag in (
            "--min-epochs 1",
            "--stop-after-epochs 15",
            "--max-epochs 60",
            "--num-workers 8",
            "--tail-bound 5",
            "--min-trials 5",
        ):
            assert flag in line, (flag, line)
        assert "--cache-dir " in line and "clone_top100_partial_v1" in line


def test_train7_dry_run_flags_per_run():
    at10, at10s1, r27, h1 = _train7_dry_run_lines()
    assert "--model armtoken" in at10 and "--seed 20260924" in at10
    assert "--model armtoken" in at10s1 and "--seed 1" in at10s1
    assert "--model hybrid" in h1
    # R27 is R26 plus the partial sims, so all six of R26's flags must be there.
    assert "--model cloneatt" in r27
    for flag in (
        "--z-score-x structured",
        "--input-space copy",
        "--freq-mode feature",
        "--attn-ln",
        "--flow-num-transforms 3",
        "--d-model 256",
    ):
        assert flag in r27, flag
    # ...and nowhere else: they are cloneatt's preset gap, not a matrix default.
    for line in (at10, at10s1, h1):
        assert "--freq-mode" not in line


def test_train7_dry_run_reports_the_split_gate_as_ok():
    assert "split gate: OK" in _train7_dry_run().stdout


def test_train7_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    r = _run(
        JOBS / "train7.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr and "val_ids" in r.stderr


def test_prepare_partial_sh_is_valid_bash_and_executable():
    r = subprocess.run(
        ["bash", "-n", str(JOBS / "prepare_partial.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert os.access(JOBS / "prepare_partial.sh", os.X_OK)


def test_prepare_partial_dry_run_prints_the_build_and_the_verify():
    r = _run(JOBS / "prepare_partial.sh", {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)})
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("python ")]
    assert len(lines) == 2
    build, verify = lines
    assert "build_clone_cache.py" in build
    assert "--min-trials 5" in build
    assert "clone_top100_partial_v1" in build
    assert "verify_clone_cache.py" in verify
    assert "clone_top100_partial_v1" in verify
    # It must NOT carve a split -- prepare.sh owns that, and this one reads it.
    assert "make_split" not in (JOBS / "prepare_partial.sh").read_text()


def test_prepare_sh_is_untouched_and_still_complete_only():
    text = (JOBS / "prepare.sh").read_text()
    assert "--min-trials" not in text
    assert "clone_top100_v1" in text and "clone_top100_partial_v1" not in text


def test_jobs_readme_documents_matrix_seven():
    text = (JOBS / "README.md").read_text()
    assert "Matrix 7" in text and "train7.sh" in text
    for run in RUN7_NAMES:
        assert run in text
    assert "--min-trials 5" in text
    # The rule itself, not only the table.
    assert "test set does not" in text


# ------------------------------------------------------------- train8.sh
#
# Matrix 8: eight armtoken runs, four questions with a seed-1 replicate each.
# The dry run is the only place the per-run flag sets can be checked without a
# GPU, and the two switches build modules, so a wrong flag here is a run that
# silently reproduces AT0.


def _train8_dry_run(env=None):
    full = {"DRY_RUN": "1", "CANCER": str(CODE_ROOT)}
    full.update(env or {})
    r = _run(JOBS / "train8.sh", full)
    assert r.returncode == 0, r.stderr
    return r


def _train8_dry_run_lines():
    return [
        ln for ln in _train8_dry_run().stdout.splitlines() if ln.startswith("python ")
    ]


RUN8_NAMES = ("AT11", "AT11s1", "AT12", "AT12s1", "AT13", "AT13s1", "AT14", "AT14s1")


def test_train8_sh_header_matches_the_matrix():
    text = (JOBS / "train8.sh").read_text()
    for run in RUN8_NAMES:
        assert run in text
    assert "--array=0-7" in text
    assert "-C a100" in text and "general-gpu" in text and "-t 12:00:00" in text
    # Header line per run, as train5.sh has.
    assert 'echo "host=$(hostname)' in text
    # The earlier matrices must not have been edited into this one.
    assert "--array=0-5" in (JOBS / "train5.sh").read_text()
    assert "--array=0-8" in (JOBS / "train6.sh").read_text()
    assert "--array=0-3" in (JOBS / "train7.sh").read_text()


def test_train8_sh_is_executable_and_uses_the_three_key_split():
    assert os.access(JOBS / "train8.sh", os.X_OK)
    text = (JOBS / "train8.sh").read_text()
    assert "CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl" in text


def test_train8_dry_run_prints_eight_commands_with_the_common_flags():
    lines = _train8_dry_run_lines()
    assert len(lines) == 8
    for line in lines:
        for flag in (
            "--model armtoken",
            "--min-epochs 1",
            "--stop-after-epochs 15",
            "--max-epochs 60",
            "--num-workers 8",
            "--tail-bound 5",
        ):
            assert flag in line, (flag, line)
    # Not matrix 7's data switch: this matrix is read against AT0.
    assert not any("--min-trials" in line for line in lines)


def test_train8_dry_run_flags_per_run():
    at11, at11s1, at12, at12s1, at13, at13s1, at14, at14s1 = _train8_dry_run_lines()

    assert at11.endswith("--input-space log2") and "--seed 20260924" in at11
    assert at11s1.endswith("--input-space log2") and "--seed 1" in at11s1

    assert at12.endswith("--arm-feature-norm layernorm") and "--seed 20260924" in at12
    assert at12s1.endswith("--arm-feature-norm layernorm") and "--seed 1" in at12s1

    assert at13.endswith("--arm-feature-norm batchnorm") and "--seed 20260924" in at13
    assert at13s1.endswith("--arm-feature-norm batchnorm") and "--seed 1" in at13s1

    # store_true: the flag is bare, and takes no value.
    assert at14.endswith("--arm-context-norm") and "--seed 20260924" in at14
    assert at14s1.endswith("--arm-context-norm") and "--seed 1" in at14s1

    # Each row is AT0 plus exactly one thing -- no row carries two switches.
    for line in (at11, at11s1):
        assert "--arm-feature-norm" not in line and "--arm-context-norm" not in line
    for line in (at12, at12s1, at13, at13s1, at14, at14s1):
        assert "--input-space" not in line
    for line in (at14, at14s1):
        assert "--arm-feature-norm" not in line


def test_train8_dry_run_reports_the_split_gate_as_ok():
    assert "split gate: OK" in _train8_dry_run().stdout


def test_train8_sh_hard_fails_on_a_split_without_val_ids(tmp_path):
    r = _run(
        JOBS / "train8.sh",
        {
            "CANCER": str(CODE_ROOT),
            "CANCER_SBI_SPLIT": str(CODE_ROOT / "data" / "train_test_split.pkl"),
            "RUNS_ROOT": str(tmp_path),
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSING" in r.stderr and "val_ids" in r.stderr


def test_train8_leaves_the_earlier_matrices_alone():
    for older in ("train.sh", "train2.sh", "train3.sh", "train4.sh",
                  "train4b.sh", "train5.sh", "train6.sh", "train7.sh"):
        assert (JOBS / older).exists()


def test_jobs_readme_documents_matrix_eight():
    text = (JOBS / "README.md").read_text()
    assert "Matrix 8" in text and "train8.sh" in text
    for run in RUN8_NAMES:
        assert run in text
    assert "--arm-feature-norm" in text and "--arm-context-norm" in text
    # The reason sbi's own switch is not the tool for the context.
    assert "z_score_y" in text


# ------------------------------------------------------- analyze.sh runs TARP too
#
# Stage 2 is two commands now: poster_metrics for the tables and figures A/B/C/S1, and
# tarp for the joint calibration test and figure T. The dry run is the only place the
# pair can be checked without conda or a posteriors file.


def _analyze_dry_run(env=None):
    full = {"POST": "/p", "OUT": "/o", "DRY_RUN": "1"}
    full.update(env or {})
    r = _run(JOBS / "analyze.sh", full)
    assert r.returncode == 0, r.stderr
    return [ln for ln in r.stdout.splitlines() if ln.startswith("python ")]


def test_analyze_dry_run_prints_both_stage_two_commands():
    lines = _analyze_dry_run()
    assert len(lines) == 2, lines
    assert lines[0] == "python -m cancer_sbi.evaluation.poster_metrics --in-dir /p --out-dir /o"
    assert lines[1] == "python -m cancer_sbi.evaluation.tarp --in-dir /p --out-dir /o"


def test_analyze_passes_the_run_tag_to_both_commands():
    lines = _analyze_dry_run({"RUN_TAG": "R1"})
    assert len(lines) == 2, lines
    assert all(ln.endswith("--in-dir /p --out-dir /o --run-tag R1") for ln in lines), lines


def test_analyze_actually_invokes_tarp_not_only_in_the_dry_run():
    """The dry run prints TARP; the real branch has to run it as well."""
    text = (JOBS / "analyze.sh").read_text()
    assert text.count("cancer_sbi.evaluation.tarp") == 1, "the command is built once"
    body = text.split('if [ "${DRY_RUN:-0}" = "1" ]', 1)[1].split("exit 0", 1)[1]
    assert '"${TARP[@]}"' in body, "TARP is printed but never executed"


def test_tarp_module_is_importable_and_has_the_cli():
    from cancer_sbi.evaluation import tarp
    assert hasattr(tarp, "main") and hasattr(tarp, "tarp_from_samples")
