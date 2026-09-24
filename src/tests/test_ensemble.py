"""The seed ensemble: pooling several stage-1 posterior files into one mixture file.

Everything here runs on tiny synthetic .npz files written with the exact key layout of
``sample_posteriors.py`` - the real ones are ~530 MB and live on the cluster. Nothing
imports torch: ``ensemble_posteriors`` is numpy only, and so is ``poster_metrics``.
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

from cancer_sbi.evaluation import ensemble_posteriors as ens  # noqa: E402
from cancer_sbi.evaluation import poster_metrics as pm  # noqa: E402

N_ARMS = 44
ENSEMBLE_SH = SRC / "jobs" / "ensemble.sh"


def write_member(path, theta, samples, seed, sim_ids=None, run_tag=None, epoch=7):
    """One stage-1 .npz, with sample_posteriors.py's keys, dtypes and derived arrays."""
    theta = np.asarray(theta, dtype=np.float32)
    samples = np.asarray(samples, dtype=np.float32)
    n, s, _ = samples.shape
    meta = {
        "model": "cloneatt", "folder": "SetTransformer_NPE", "checkpoint": f"/ckpt/s{seed}.pt",
        "checkpoints_not_used": [], "checkpoint_epoch": epoch, "best_val_loss": -1.0 - seed,
        "n_cases": n, "n_cases_available": n, "num_samples": s, "prior_sd": 0.2,
        "seed": seed, "limit": None, "run_tag": run_tag, "device": "cpu",
        "data_root": "/data", "split_file": "/split.pkl", "n_train_ids": 10, "n_test_ids": n,
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
        log_prob_true=np.full(n, 1.0 + seed, dtype=np.float32),
        sim_ids=np.array(sim_ids if sim_ids is not None else [f"sim{i}" for i in range(n)]),
        meta_json=json.dumps(meta),
    )
    return str(path)


@pytest.fixture
def members(tmp_path):
    """Three members, 5 cases x 44 arms, 20/20/20 draws, different seeds."""
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.2, size=(5, N_ARMS)).astype(np.float32)
    paths = []
    for seed in (0, 1, 2):
        draws = np.random.default_rng(100 + seed).normal(
            0.0, 0.2, size=(5, 20, N_ARMS)).astype(np.float32)
        paths.append(write_member(tmp_path / f"posteriors_cloneatt_s{seed}.npz",
                                  theta, draws, seed, run_tag=f"s{seed}"))
    return theta, paths


def run_ensemble(inputs, out_dir, tag="R12ens", extra=()):
    ens.main(["--inputs", *inputs, "--out-dir", str(out_dir),
              "--model", "cloneatt", "--run-tag", tag, *extra])
    return os.path.join(str(out_dir), f"posteriors_cloneatt_{tag}.npz")


# ------------------------------------------------------------------ pooling
def test_shapes_and_pooled_count(members, tmp_path, capsys):
    theta, paths = members
    out = run_ensemble(paths, tmp_path / "ens")
    d = np.load(out, allow_pickle=True)
    assert d["samples"].shape == (5, 60, N_ARMS)
    assert d["theta_true"].shape == (5, N_ARMS)
    assert d["post_mean"].shape == d["post_std"].shape == (5, N_ARMS)
    assert d["sbc_ranks"].shape == (5, N_ARMS)
    assert d["sim_ids"].shape == (5,)
    assert d["log_prob_true"].shape == (5,)
    assert set(d.files) == {"theta_true", "samples", "post_mean", "post_std", "sbc_ranks",
                            "log_prob_true", "sim_ids", "meta_json"}
    meta = json.loads(str(d["meta_json"]))
    assert meta["num_samples"] == 60
    assert meta["ensemble_of"] == paths
    assert meta["ensemble_seeds"] == [0, 1, 2]
    assert meta["checkpoint_epoch"] == [7, 7, 7]
    # one summary line per member plus one for the ensemble
    printed = capsys.readouterr().out
    assert printed.count("member  ") == 3
    assert "ensemble  " in printed


def test_pooled_samples_are_the_members_concatenated(members, tmp_path):
    _, paths = members
    out = run_ensemble(paths, tmp_path / "ens")
    pooled = np.load(out)["samples"]
    for k, p in enumerate(paths):
        np.testing.assert_array_equal(pooled[:, k * 20:(k + 1) * 20], np.load(p)["samples"])


def test_post_mean_and_std_come_from_the_pooled_samples(members, tmp_path):
    _, paths = members
    out = np.load(run_ensemble(paths, tmp_path / "ens"))
    np.testing.assert_allclose(out["post_mean"], out["samples"].mean(axis=1), rtol=1e-5)
    np.testing.assert_allclose(out["post_std"], out["samples"].std(axis=1, ddof=0),
                               rtol=1e-5, atol=1e-6)
    # and it is NOT just the mean of the members' means unless the shares are equal
    members_means = np.mean([np.load(p)["post_mean"] for p in paths], axis=0)
    np.testing.assert_allclose(out["post_mean"], members_means, rtol=1e-4, atol=1e-6)


def test_sbc_ranks_on_a_hand_checkable_case(tmp_path):
    """Two members, one arm each of a 2-arm case, with draws chosen by hand.

    theta = [0.0, 10.0]. Member A draws [-2,-1,1,2] on arm 0, member B [-3,3,5,7].
    Pooled arm 0: -2, -1 and -3 are below 0.0, so the rank is 3 of 8. Arm 1: every
    one of the 8 draws is below 10.0, so the rank is 8.
    """
    theta = np.array([[0.0, 10.0]], dtype=np.float32)
    a = np.array([[[-2.0, 0.0], [-1.0, 1.0], [1.0, 2.0], [2.0, 3.0]]], dtype=np.float32)
    b = np.array([[[-3.0, 4.0], [3.0, 5.0], [5.0, 6.0], [7.0, 7.0]]], dtype=np.float32)
    pa = write_member(tmp_path / "a.npz", theta, a, 0)
    pb = write_member(tmp_path / "b.npz", theta, b, 1)
    out = np.load(run_ensemble([pa, pb], tmp_path / "ens"))
    np.testing.assert_array_equal(out["sbc_ranks"], np.array([[3, 8]], dtype=np.int32))
    assert out["sbc_ranks"].dtype == np.int32
    # the +0.5/(S+1) convention lives in poster_metrics and must use the POOLED count
    run = pm.load_run(os.path.join(str(tmp_path / "ens"), "posteriors_cloneatt_R12ens.npz"))
    np.testing.assert_allclose(run["u"], (np.array([[3.0, 8.0]]) + 0.5) / 9.0)


def test_sbc_ranks_match_a_recompute_over_the_pooled_draws(members, tmp_path):
    _, paths = members
    out = np.load(run_ensemble(paths, tmp_path / "ens"))
    expect = (out["samples"] < out["theta_true"][:, None, :]).sum(axis=1)
    np.testing.assert_array_equal(out["sbc_ranks"], expect)


# ------------------------------------------------------------------ guards
def test_mismatched_theta_true_raises(tmp_path):
    theta = np.zeros((2, 3), dtype=np.float32)
    other = theta.copy()
    other[1, 2] = 0.5
    draws = np.zeros((2, 4, 3), dtype=np.float32)
    pa = write_member(tmp_path / "a.npz", theta, draws, 0)
    pb = write_member(tmp_path / "b.npz", other, draws, 1)
    with pytest.raises(ValueError, match="theta_true mismatch"):
        run_ensemble([pa, pb], tmp_path / "ens")


def test_mismatched_theta_shape_raises(tmp_path):
    pa = write_member(tmp_path / "a.npz", np.zeros((2, 3), np.float32),
                      np.zeros((2, 4, 3), np.float32), 0)
    pb = write_member(tmp_path / "b.npz", np.zeros((3, 3), np.float32),
                      np.zeros((3, 4, 3), np.float32), 1)
    with pytest.raises(ValueError, match="theta_true shape mismatch"):
        run_ensemble([pa, pb], tmp_path / "ens")


def test_mismatched_sim_ids_raises(tmp_path):
    theta = np.zeros((2, 3), dtype=np.float32)
    draws = np.zeros((2, 4, 3), dtype=np.float32)
    pa = write_member(tmp_path / "a.npz", theta, draws, 0, sim_ids=["sim1", "sim2"])
    pb = write_member(tmp_path / "b.npz", theta, draws, 1, sim_ids=["sim1", "sim9"])
    with pytest.raises(ValueError, match="sim_ids mismatch"):
        run_ensemble([pa, pb], tmp_path / "ens")


# ------------------------------------------------------------------ subsampling
def test_num_samples_takes_an_equal_share_from_each_member(members, tmp_path):
    _, paths = members
    out = np.load(run_ensemble(paths, tmp_path / "ens", extra=("--num-samples", "12")))
    assert out["samples"].shape == (5, 12, N_ARMS)
    for k, p in enumerate(paths):
        np.testing.assert_array_equal(out["samples"][:, k * 4:(k + 1) * 4],
                                      np.load(p)["samples"][:, :4])
    meta = json.loads(str(out["meta_json"]))
    assert meta["num_samples"] == 12
    assert meta["ensemble_member_draws_used"] == [4, 4, 4]


def test_num_samples_remainder_goes_to_the_first_members():
    assert ens.shares(14, [20, 20, 20]) == [5, 5, 4]
    assert ens.shares(12, [20, 20, 20]) == [4, 4, 4]


def test_num_samples_at_or_above_the_total_keeps_everything(members, tmp_path):
    _, paths = members
    out = np.load(run_ensemble(paths, tmp_path / "ens", extra=("--num-samples", "999")))
    assert out["samples"].shape == (5, 60, N_ARMS)
    assert ens.shares(60, [20, 20, 20]) is None


def test_num_samples_beyond_a_members_draws_raises(tmp_path):
    with pytest.raises(ValueError, match="fewer than its"):
        ens.shares(20, [20, 2])


def test_default_keeps_all_draws(members, tmp_path):
    _, paths = members
    assert np.load(run_ensemble(paths, tmp_path / "ens"))["samples"].shape[1] == 60


# ------------------------------------------------------------------ downstream
def test_poster_metrics_reads_the_ensemble_file(members, tmp_path):
    _, paths = members
    out_dir = tmp_path / "ens"
    out = run_ensemble(paths, out_dir)
    assert pm.posterior_path(str(out_dir), "cloneatt", "R12ens") == out
    run = pm.load_run(out)
    assert run["u"].shape == (5, N_ARMS)
    assert ((run["u"] > 0) & (run["u"] < 1)).all()
    assert run["meta"]["num_samples"] == 60
    # the metric path poster_metrics.main walks, on a 2-arm-wide slice of the real one
    z = pm.zscores(run)
    assert np.isfinite(z).all()
    assert np.isfinite(pm.coverage_curve(run["u"], np.linspace(0, 1, 11))).all()


def test_log_prob_true_is_the_mixture_density(members, tmp_path):
    _, paths = members
    out = np.load(run_ensemble(paths, tmp_path / "ens"))
    lp = np.array([1.0, 2.0, 3.0])
    expect = np.log(np.exp(lp).sum() / 3.0)
    np.testing.assert_allclose(out["log_prob_true"], np.full(5, expect, np.float32), rtol=1e-5)


def test_log_prob_nan_in_one_member_propagates(tmp_path):
    theta = np.zeros((1, 3), dtype=np.float32)
    draws = np.zeros((1, 4, 3), dtype=np.float32)
    pa = write_member(tmp_path / "a.npz", theta, draws, 0)
    pb = write_member(tmp_path / "b.npz", theta, draws, 1)
    d = dict(np.load(pb, allow_pickle=True))
    d["log_prob_true"] = np.array([np.nan], dtype=np.float32)
    np.savez_compressed(pb, **d)
    out = np.load(run_ensemble([pa, pb], tmp_path / "ens"))
    assert np.isnan(out["log_prob_true"]).all()


# ------------------------------------------------------------------ jobs/ensemble.sh
def _run_sh(env, args=()):
    full = dict(os.environ)
    full.update(env)
    return subprocess.run(["bash", str(ENSEMBLE_SH), *args],
                          capture_output=True, text=True, env=full, timeout=120)


def test_ensemble_sh_dry_run_prints_the_command():
    r = _run_sh({"DRY_RUN": "1", "INPUTS": "a.npz b.npz", "OUT": "/o",
                 "MODEL": "cloneatt", "RUN_TAG": "X"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == (
        "python -m cancer_sbi.evaluation.ensemble_posteriors --inputs a.npz b.npz "
        "--out-dir /o --model cloneatt --run-tag X")


def test_ensemble_sh_passes_num_samples_and_extra_last():
    r = _run_sh({"DRY_RUN": "1", "INPUTS": "a.npz", "OUT": "/o", "MODEL": "clonemlp",
                 "RUN_TAG": "X", "NUM_SAMPLES": "5000", "EXTRA": "--num-samples 10"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("--num-samples 5000 --num-samples 10")


def test_ensemble_sh_without_a_run_tag_omits_the_flag():
    r = _run_sh({"DRY_RUN": "1", "INPUTS": "a.npz", "OUT": "/o", "MODEL": "clonemlp"})
    assert r.returncode == 0, r.stderr
    assert "--run-tag" not in r.stdout


def test_ensemble_sh_refuses_empty_inputs():
    r = _run_sh({"DRY_RUN": "1", "OUT": "/o", "MODEL": "clonemlp"})
    assert r.returncode == 1
    assert "INPUTS is empty" in r.stderr


def test_ensemble_sh_header_carries_the_two_line_recipe():
    text = ENSEMBLE_SH.read_text()
    assert "jobs/ensemble.sh" in text
    assert "sbatch jobs/analyze.sh" in text
    assert "#SBATCH -p general\n" in text          # CPU partition, like analyze.sh


def test_jobs_readme_documents_ensemble_sh():
    text = (SRC / "jobs" / "README.md").read_text()
    assert "`ensemble.sh`" in text
