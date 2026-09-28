"""Post-hoc posterior recalibration: the affine correction fitted on the validation split.

Most of this runs on tiny synthetic .npz files written with the exact key layout of
``sample_posteriors.py`` - the real ones are ~530 MB and live on the cluster. Nothing here
imports torch except the one end-to-end test at the bottom, which trains clonemlp for a
single epoch on a handful of local sims and is skipped when the simulation tree is absent.

Run from ``src/``::

    python -m pytest tests/test_recalibration.py -q
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cancer_sbi.evaluation import poster_metrics as pm  # noqa: E402
from cancer_sbi.evaluation import recalibrate_posteriors as rc  # noqa: E402
from cancer_sbi.evaluation import sample_posteriors as sp  # noqa: E402

N_ARMS = 44
BIASED_ARM = 3
BIAS = 0.3          # the posterior mean sits 0.3 sd below the truth on that arm
WIDTH = 1.5         # and the posterior is 1.5x too wide

JOBS = SRC / "jobs"
RECALIBRATE_SH = JOBS / "recalibrate.sh"
SAMPLE_SH = JOBS / "sample.sh"
DATA_ROOT = SRC.parent / "data" / "Guassian_Normal" / "simulation_outputs"

needs_data = pytest.mark.skipif(
    not DATA_ROOT.is_dir(), reason="local simulation tree not present"
)


# ------------------------------------------------------------------ synthetic files
def write_run(path, theta, samples, checkpoint="/ckpt/best.pt", partition="test",
              run_tag=None, num_samples=None):
    """One stage-1 .npz, with sample_posteriors.py's keys, dtypes and derived arrays."""
    theta = np.asarray(theta, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    n, s, _ = samples.shape
    meta = {
        "model": "clonemlp", "folder": "Base_NPE", "checkpoint": checkpoint,
        "checkpoints_not_used": [], "checkpoint_epoch": 7, "best_val_loss": -1.0,
        "n_cases": n, "n_cases_available": n, "num_samples": int(num_samples or s),
        "prior_sd": 0.2, "seed": 0, "limit": None, "run_tag": run_tag,
        "partition": partition, "device": "cpu", "data_root": "/data",
        "split_file": "/split.pkl", "n_train_ids": 10, "n_test_ids": n, "n_val_ids": n,
        "flow_config": {}, "encoder_config": {}, "config_from_checkpoint": True,
        "require_all_trials": False, "written_utc": "2026-09-24T00:00:00+00:00",
        "elapsed_s": 1.0, "built_with": "cancer_sbi",
    }
    np.savez_compressed(
        path,
        theta_true=theta,
        samples=samples,
        post_mean=samples.mean(axis=1).astype(np.float32),
        post_std=samples.std(axis=1, ddof=0).astype(np.float32),
        sbc_ranks=(samples < theta[:, None, :]).sum(axis=1).astype(np.int32),
        log_prob_true=np.full(n, 2.0, dtype=np.float32),
        sim_ids=np.array([f"sim{i}" for i in range(n)]),
        meta_json=json.dumps(meta),
    )
    return str(path)


def _miscalibrated(rng, n_cases, n_draws, sd=0.2):
    """A "model" that is honest on every arm but ``BIASED_ARM``.

    Each case has a posterior centred on ``theta + e_i`` for an error ``e_i ~ N(0, sd)`` and a
    posterior sd of ``w_j``. On an honest arm ``w_j = sd``, so ``z = (theta - mean)/std =
    -e_i/sd`` is standard normal: mean 0, sd 1, ranks uniform. On ``BIASED_ARM`` the centre is
    pushed a further ``BIAS * WIDTH * sd`` below the truth and ``w_j = WIDTH * sd``, so

        z = BIAS - e_i / (WIDTH * sd)   ->   mean BIAS, sd 1 / WIDTH

    which is exactly the pair ``(a_j, b_j) = (BIAS, 1/WIDTH)`` the fit must recover.
    """
    theta = rng.normal(0.0, sd, size=(n_cases, N_ARMS))
    e = rng.normal(0.0, sd, size=(n_cases, N_ARMS))          # the posterior-mean error
    width = np.full(N_ARMS, sd)
    offset = np.zeros(N_ARMS)
    width[BIASED_ARM] = WIDTH * sd
    offset[BIASED_ARM] = -BIAS * WIDTH * sd
    centre = theta + e + offset[None, :]
    samples = centre[:, None, :] + width[None, None, :] * rng.normal(
        0.0, 1.0, size=(n_cases, n_draws, N_ARMS))
    return theta.astype(np.float32), samples.astype(np.float32)


@pytest.fixture
def pair(tmp_path):
    """A validation file and a test file from the same (miscalibrated) model."""
    rng = np.random.default_rng(20260924)
    tv, sv = _miscalibrated(rng, 400, 400)
    tt, st = _miscalibrated(rng, 400, 400)
    val = write_run(tmp_path / "posteriors_clonemlp_AT0_val.npz", tv, sv,
                    partition="val", run_tag="AT0")
    test = write_run(tmp_path / "posteriors_clonemlp_AT0.npz", tt, st, run_tag="AT0")
    return val, test


def run_rc(val, test, out_dir, tag="AT0rc", extra=()):
    rc.main(["--val", val, "--test", test, "--out-dir", str(out_dir),
             "--model", "clonemlp", "--run-tag", tag, *extra])
    return os.path.join(str(out_dir), f"posteriors_clonemlp_{tag}.npz")


# ------------------------------------------------------------------ the fit
def test_fit_recovers_the_injected_bias_and_width(pair):
    val, _ = pair
    a, b, z = rc.fit_affine(rc.load_run(val))
    assert z.shape == (400, N_ARMS)
    assert a[BIASED_ARM] == pytest.approx(BIAS, abs=0.08)
    assert b[BIASED_ARM] == pytest.approx(1.0 / WIDTH, abs=0.08)
    # every other arm was honest: a ~ 0, b ~ 1
    others = [j for j in range(N_ARMS) if j != BIASED_ARM]
    assert np.abs(a[others]).max() < 0.20
    assert np.abs(b[others] - 1.0).max() < 0.20


def test_the_correction_makes_validation_z_standard(pair):
    """(z - a)/b has mean 0 and sd 1 on the fitted arm, by construction."""
    val, _ = pair
    a, b, z = rc.fit_affine(rc.load_run(val))
    zc = (z - a[None, :]) / b[None, :]
    assert zc[:, BIASED_ARM].mean() == pytest.approx(0.0, abs=1e-6)
    assert zc[:, BIASED_ARM].std(ddof=0) == pytest.approx(1.0, abs=1e-6)


def test_applying_the_fit_to_the_val_file_itself_standardises_its_z(pair):
    """The same thing again, but through ``apply_affine`` and a real recompute."""
    val, _ = pair
    d = rc.load_run(val)
    a, b, _ = rc.fit_affine(d)
    corrected = rc.apply_affine(d["samples"], d["post_mean"], d["post_std"], a, b)
    z = rc.zscores(d["theta_true"], corrected.mean(axis=1), corrected.std(axis=1, ddof=0))
    assert z[:, BIASED_ARM].mean() == pytest.approx(0.0, abs=1e-3)
    assert z[:, BIASED_ARM].std(ddof=0) == pytest.approx(1.0, abs=1e-3)


# ------------------------------------------------------------------ the test file
def test_test_ranks_become_uniform_on_the_biased_arm(pair, tmp_path):
    before = pm.ks_uniform(pm.load_run(pair[1])["u"][:, BIASED_ARM])
    out = run_rc(*pair, tmp_path / "rc")
    after = pm.ks_uniform(pm.load_run(out)["u"][:, BIASED_ARM])
    assert before[1] < 0.01, "the synthetic model is supposed to be badly miscalibrated"
    assert after[1] > 0.01, f"ranks still non-uniform after recalibration: KS p={after[1]}"


def test_other_arms_are_left_essentially_alone(pair, tmp_path):
    out = np.load(run_rc(*pair, tmp_path / "rc"))
    orig = np.load(pair[1])
    others = [j for j in range(N_ARMS) if j != BIASED_ARM]
    # a_j ~ 0 and b_j ~ 1 there, so the draws move by well under a tenth of a posterior sd
    move = np.abs(out["samples"][:, :, others] - orig["samples"][:, :, others])
    assert (move / (orig["post_std"][:, None, others] + rc.EPS)).max() < 0.6
    assert np.abs(out["post_mean"][:, others] - orig["post_mean"][:, others]).max() < 0.06


def test_derived_arrays_are_recomputed_from_the_corrected_samples(pair, tmp_path):
    out = np.load(run_rc(*pair, tmp_path / "rc"))
    np.testing.assert_allclose(out["post_mean"], out["samples"].mean(axis=1),
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(out["post_std"], out["samples"].std(axis=1, ddof=0),
                               rtol=1e-4, atol=1e-6)
    expect = (out["samples"] < out["theta_true"][:, None, :]).sum(axis=1)
    np.testing.assert_array_equal(out["sbc_ranks"], expect)
    assert out["sbc_ranks"].dtype == np.int32


def test_theta_and_sim_ids_are_copied_unchanged(pair, tmp_path):
    out = np.load(run_rc(*pair, tmp_path / "rc"))
    orig = np.load(pair[1])
    np.testing.assert_array_equal(out["theta_true"], orig["theta_true"])
    np.testing.assert_array_equal(out["sim_ids"], orig["sim_ids"])


def test_apply_affine_matches_the_documented_formula(pair):
    d = rc.load_run(pair[1])
    a = np.linspace(-0.5, 0.5, N_ARMS)
    b = np.linspace(0.5, 1.5, N_ARMS)
    got = rc.apply_affine(d["samples"], d["post_mean"], d["post_std"], a, b)
    j, i = BIASED_ARM, 7
    m, s = d["post_mean"][i, j], d["post_std"][i, j]
    want = m + a[j] * s + b[j] * (d["samples"][i, :, j] - m)
    np.testing.assert_allclose(got[i, :, j], want.astype(np.float32), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ --method shift
def test_shift_leaves_the_widths_untouched(pair, tmp_path):
    out = np.load(run_rc(*pair, tmp_path / "rcs", tag="AT0sh", extra=("--method", "shift")))
    orig = np.load(pair[1])
    np.testing.assert_allclose(out["post_std"], orig["post_std"], rtol=1e-4, atol=1e-6)
    # but the centre did move on the biased arm
    shift = (out["post_mean"] - orig["post_mean"])[:, BIASED_ARM] / orig["post_std"][:, BIASED_ARM]
    assert shift.mean() == pytest.approx(BIAS, abs=0.08)
    assert np.abs(out["post_mean"] - orig["post_mean"]).max() > 0.0


def test_shift_records_b_as_all_ones(pair, tmp_path):
    out = np.load(run_rc(*pair, tmp_path / "rcs", tag="AT0sh", extra=("--method", "shift")),
                  allow_pickle=True)
    rec = json.loads(str(out["meta_json"]))["recalibration"]
    assert rec["method"] == "shift"
    assert rec["b"] == [1.0] * N_ARMS
    assert rec["a"][BIASED_ARM] == pytest.approx(BIAS, abs=0.08)


def test_unknown_method_raises(pair):
    with pytest.raises(ValueError, match="affine"):
        rc.fit_affine(rc.load_run(pair[0]), method="quadratic")


# ------------------------------------------------------------------ meta and guards
def test_meta_carries_the_recalibration_block(pair, tmp_path):
    val, test = pair
    out = np.load(run_rc(val, test, tmp_path / "rc"), allow_pickle=True)
    meta = json.loads(str(out["meta_json"]))
    rec = meta["recalibration"]
    assert rec["method"] == "affine"
    assert rec["fitted_on"] == val
    assert rec["n_val"] == 400
    assert len(rec["a"]) == len(rec["b"]) == N_ARMS
    assert meta["run_tag"] == "AT0rc"
    assert meta["model"] == "clonemlp"
    # inherited from the TEST file, so downstream still divides ranks by the right S
    assert meta["num_samples"] == 400
    assert meta["log_prob_true"] == rc.LOG_PROB_NOTE


def test_log_prob_true_is_nan_and_poster_metrics_tolerates_it(pair, tmp_path):
    out_path = run_rc(*pair, tmp_path / "rc")
    assert np.isnan(np.load(out_path)["log_prob_true"]).all()
    run = pm.load_run(out_path)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = float(np.nanmean(run["log_prob_true"]))
    assert np.isnan(value)          # NaN reported, not an exception


def test_checkpoint_mismatch_is_refused(tmp_path):
    rng = np.random.default_rng(0)
    t, s = _miscalibrated(rng, 5, 8)
    val = write_run(tmp_path / "v.npz", t, s, checkpoint="/ckpt/A.pt", partition="val")
    test = write_run(tmp_path / "t.npz", t, s, checkpoint="/ckpt/B.pt")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        run_rc(val, test, tmp_path / "rc")


def test_force_waives_the_checkpoint_guard(tmp_path, capsys):
    rng = np.random.default_rng(0)
    t, s = _miscalibrated(rng, 20, 30)
    val = write_run(tmp_path / "v.npz", t, s, checkpoint="/ckpt/A.pt", partition="val")
    test = write_run(tmp_path / "t.npz", t, s, checkpoint="/ckpt/B.pt")
    out = run_rc(val, test, tmp_path / "rc", extra=("--force",))
    assert os.path.exists(out)
    assert "checkpoint mismatch" in capsys.readouterr().out
    meta = json.loads(str(np.load(out, allow_pickle=True)["meta_json"]))
    assert meta["recalibration"]["forced_checkpoint_mismatch"] is True


def test_arm_count_mismatch_is_refused(tmp_path):
    t = np.zeros((3, 4), np.float32)
    val = write_run(tmp_path / "v.npz", t, np.zeros((3, 6, 4), np.float32), partition="val")
    t2 = np.zeros((3, 5), np.float32)
    test = write_run(tmp_path / "t.npz", t2, np.zeros((3, 6, 5), np.float32))
    with pytest.raises(ValueError, match="arm count mismatch"):
        run_rc(val, test, tmp_path / "rc")


def test_a_degenerate_validation_spread_leaves_the_width_alone(tmp_path, capsys):
    """One validation case gives std(z) = 0, which would collapse the posterior to a point."""
    rng = np.random.default_rng(1)
    t, s = _miscalibrated(rng, 1, 40)
    val = write_run(tmp_path / "v.npz", t, s, partition="val")
    a, b, _ = rc.fit_affine(rc.load_run(val))
    assert (b == 1.0).all()
    assert "non-positive validation z spread" in capsys.readouterr().out


def test_a_val_file_sampled_on_the_test_partition_warns(tmp_path, capsys):
    rng = np.random.default_rng(2)
    t, s = _miscalibrated(rng, 20, 30)
    val = write_run(tmp_path / "v.npz", t, s, partition="test")
    test = write_run(tmp_path / "t.npz", t, s)
    run_rc(val, test, tmp_path / "rc")
    assert "fitted on the very cases it is scored on" in capsys.readouterr().out


def test_the_run_prints_its_a_b_summary_and_val_mean_abs_z(pair, tmp_path, capsys):
    run_rc(*pair, tmp_path / "rc")
    out = capsys.readouterr().out
    assert "  a   min" in out and "  b   min" in out
    assert "median" in out and "max" in out
    assert "val mean|z|  before" in out and "->  after" in out
    assert "recalibrated  " in out


# ------------------------------------------------------------------ downstream
def test_the_output_loads_through_poster_metrics(pair, tmp_path):
    out_dir = tmp_path / "rc"
    out = run_rc(*pair, out_dir)
    assert pm.posterior_path(str(out_dir), "clonemlp", "AT0rc") == out
    run = pm.load_run(out)
    assert run["u"].shape == (400, N_ARMS)
    assert ((run["u"] > 0) & (run["u"] < 1)).all()
    z = pm.zscores(run)
    assert np.isfinite(z).all()
    assert np.isfinite(pm.coverage_curve(run["u"], np.linspace(0, 1, 11))).all()


def test_the_output_has_exactly_the_stage_one_keys(pair, tmp_path):
    d = np.load(run_rc(*pair, tmp_path / "rc"), allow_pickle=True)
    assert set(d.files) == {"theta_true", "samples", "post_mean", "post_std", "sbc_ranks",
                            "log_prob_true", "sim_ids", "meta_json"}


# ------------------------------------------------------------------ output_filename
@pytest.mark.parametrize("limit, tag, partition, expected", [
    (None, None, "test", "posteriors_clonemlp.npz"),
    (None, None, "val", "posteriors_clonemlp_val.npz"),
    (None, "AT0", "test", "posteriors_clonemlp_AT0.npz"),
    (None, "AT0", "val", "posteriors_clonemlp_AT0_val.npz"),
    (3, None, "val", "posteriors_clonemlp_limit3_val.npz"),
    (3, None, "test", "posteriors_clonemlp_limit3.npz"),
])
def test_output_filename_with_partition(limit, tag, partition, expected):
    assert sp.output_filename("clonemlp", limit, tag, partition) == expected


def test_output_filename_default_partition_is_unchanged():
    assert sp.output_filename("clonemlp", None, "R1") == "posteriors_clonemlp_R1.npz"
    assert sp.output_filename("clonemlp", 4, None) == "posteriors_clonemlp_limit4.npz"


def test_output_filename_rejects_an_unknown_partition():
    with pytest.raises(ValueError, match="partition"):
        sp.output_filename("clonemlp", None, None, "train")


# ------------------------------------------------------------------ jobs/*.sh
def _run_sh(script, env, args=()):
    full = dict(os.environ)
    full.update(env)
    return subprocess.run(["bash", str(script), *args],
                          capture_output=True, text=True, env=full, timeout=120)


def test_sample_sh_dry_run_emits_partition_val():
    r = _run_sh(SAMPLE_SH, {"DRY_RUN": "1", "PARTITION": "val", "POST": "/y",
                            "RUN_TAG": "AT0", "MODEL": "armtoken"})
    assert r.returncode == 0, r.stderr
    assert "--partition val" in r.stdout
    assert "--run-tag AT0" in r.stdout


def test_sample_sh_defaults_to_partition_test():
    r = _run_sh(SAMPLE_SH, {"DRY_RUN": "1", "POST": "/y"})
    assert r.returncode == 0, r.stderr
    assert "--partition test" in r.stdout


def test_recalibrate_sh_dry_run_prints_the_command():
    r = _run_sh(RECALIBRATE_SH, {"DRY_RUN": "1", "VAL": "/v_val.npz", "TEST": "/t.npz",
                                 "OUT": "/o", "MODEL": "armtoken", "RUN_TAG": "AT0rc",
                                 "METHOD": "shift"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == (
        "python -m cancer_sbi.evaluation.recalibrate_posteriors --val /v_val.npz "
        "--test /t.npz --out-dir /o --model armtoken --run-tag AT0rc --method shift")


def test_recalibrate_sh_extra_goes_last():
    r = _run_sh(RECALIBRATE_SH, {"DRY_RUN": "1", "VAL": "/v.npz", "TEST": "/t.npz",
                                 "OUT": "/o", "EXTRA": "--force"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("--force")


def test_recalibrate_sh_refuses_without_val_and_test():
    r = _run_sh(RECALIBRATE_SH, {"DRY_RUN": "1", "OUT": "/o"})
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


def test_recalibrate_sh_parses_and_is_documented():
    assert subprocess.run(["bash", "-n", str(RECALIBRATE_SH)]).returncode == 0
    readme = (JOBS / "README.md").read_text()
    assert "`recalibrate.sh`" in readme
    assert "PARTITION" in readme


# ------------------------------------------------------------------ end to end
@needs_data
def test_train_sample_val_and_test_then_recalibrate(tmp_path):
    """The real pipeline on six sims: train 1 epoch, sample both partitions, recalibrate."""
    import pickle

    from cancer_sbi.cli import train as train_cli

    names = sorted((p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim")),
                   key=lambda s: int(s[3:]))[:6]
    split_path = tmp_path / "split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump({"train_ids": np.array(names[:2]),
                     "val_ids": np.array(names[2:4]),
                     "test_ids": np.array(names[4:6])}, handle)

    ckpt_dir = tmp_path / "AT" / "checkpoints"
    assert train_cli.main([
        "--model", "clonemlp",
        "--data-root", str(DATA_ROOT),
        "--split", str(split_path),
        "--out", str(tmp_path / "AT"),
        "--ckpt-dir", str(ckpt_dir),
        "--device", "cpu", "--batch-size", "2",
        "--max-epochs", "1", "--min-epochs", "1", "--stop-after-epochs", "1",
        "--seed", "20260924", "--no-final-pickle",
    ]) == 0

    post = tmp_path / "post"
    common = ["--model", "clonemlp", "--data-root", str(DATA_ROOT),
              "--split", str(split_path), "--ckpt", str(ckpt_dir / "best.pt"),
              "--out-dir", str(post), "--num-samples", "64", "--device", "cpu",
              "--limit", "3", "--run-tag", "E2E"]
    sp.main(common + ["--partition", "val"])
    sp.main(common + ["--partition", "test"])

    val_path = post / sp.output_filename("clonemlp", 3, "E2E", "val")
    test_path = post / sp.output_filename("clonemlp", 3, "E2E", "test")
    assert val_path.exists() and test_path.exists()
    assert json.loads(str(np.load(val_path, allow_pickle=True)["meta_json"]))["partition"] == "val"

    out = rc.main(["--val", str(val_path), "--test", str(test_path),
                   "--out-dir", str(tmp_path / "rc"), "--model", "clonemlp",
                   "--run-tag", "E2Erc"])
    assert os.path.basename(out) == "posteriors_clonemlp_E2Erc.npz"
    run = pm.load_run(str(out))
    assert run["samples"].shape[2] == N_ARMS
    assert run["meta"]["recalibration"]["n_val"] == run["theta_true"].shape[0]


@needs_data
def test_partition_val_on_a_two_key_split_is_refused(tmp_path):
    import pickle

    names = sorted((p.name for p in DATA_ROOT.iterdir() if p.name.startswith("sim")),
                   key=lambda s: int(s[3:]))[:4]
    split_path = tmp_path / "two_key.pkl"
    with split_path.open("wb") as handle:
        pickle.dump({"train_ids": np.array(names[:2]), "test_ids": np.array(names[2:])}, handle)

    with pytest.raises(SystemExit, match="val_ids"):
        sp.main(["--model", "clonemlp", "--data-root", str(DATA_ROOT),
                 "--split", str(split_path), "--ckpt", "/nonexistent.pt",
                 "--out-dir", str(tmp_path / "out"), "--device", "cpu",
                 "--partition", "val"])
