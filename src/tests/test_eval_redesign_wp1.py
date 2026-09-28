"""WP1 of the evaluation redesign: the per-run stage.

Everything here runs on synthetic posteriors built in ``tmp_path``, because no full
posterior file (the one with the 5,000-draw ``samples`` array) exists off the cluster and
the ones that do exist are 573 MB each.

The synthetic posterior that is supposed to pass is built *exchangeably*: a posterior
centre ``mu`` is drawn first, and then the truth and the S draws are drawn i.i.d. around
it. That is the only construction that is genuinely calibrated. Drawing the samples around
the truth instead - the obvious thing to write - makes the truth the centre of its own
posterior, every draw falls below it with probability exactly 1/2, and the SBC rank comes
out Binomial(S, 1/2) rather than uniform. The over-confident twin keeps the same truths and
shrinks only the draws.

What is checked:
  * figure C's simultaneous band accepts the calibrated run on at least 42 of 44 arms,
    rejects the shrunken one, and narrows as n grows;
  * TARP reproduces the diagonal on the calibrated run and falls below it on the shrunken
    one, with a positive ``atc`` either way (it is a distance);
  * ``summary_arrays.npz`` has exactly the contract's keys, shapes and dtypes, and its
    ``u`` is ``load_run``'s ``u`` to the bit;
  * figure B names all 44 arms (read off the matplotlib ``Text`` objects, not the pixels);
  * figure S1 returns 44 panels whose R2 never exceeds r^2;
  * an unrecognised model warns instead of vanishing;
  * ``ARMS`` is ``cancer_sbi.arms.ARM_LABELS`` with the prefix off;
  * both CLIs run end to end on a tiny run directory and write every promised file.
"""
import json
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cancer_sbi.arms import ARM_LABELS                                  # noqa: E402
from cancer_sbi.evaluation import poster_metrics as pm                  # noqa: E402
from cancer_sbi.evaluation import tarp as tp                            # noqa: E402

N_ARMS = 44
PRIOR_SD = 0.2
POST_SD = 0.12


def make_posterior(n, s, d=N_ARMS, shrink=1.0, seed=0, prior_sd=PRIOR_SD, post_sd=POST_SD):
    """(theta_true, samples) for a posterior that is calibrated iff ``shrink`` is 1.

    ``mu`` is the posterior centre; the truth and every draw are i.i.d. N(mu, post_sd)
    around it, so they are exchangeable and both the SBC rank and TARP's coverage value
    are exactly uniform. ``mu``'s own spread is chosen so the marginal truth is
    N(0, prior_sd), the project's prior. ``shrink < 1`` narrows only the draws.
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, np.sqrt(prior_sd ** 2 - post_sd ** 2), (n, d))
    theta = (mu + post_sd * rng.standard_normal((n, d))).astype(np.float32)
    samples = (mu[:, None, :] + shrink * post_sd * rng.standard_normal((n, s, d))).astype(np.float32)
    return theta, samples


def write_stage_one(path, theta, samples, model="clonemlp", run_tag="SYN", **meta_extra):
    """A file in exactly the layout ``sample_posteriors.py`` writes."""
    n, s, _ = samples.shape
    meta = {"model": model, "n_cases": n, "n_cases_available": n, "num_samples": s,
            "prior_sd": PRIOR_SD, "checkpoint_epoch": 7, "run_tag": run_tag,
            "seed": 0, "partition": "test"}
    meta.update(meta_extra)
    np.savez_compressed(
        path,
        theta_true=theta,
        samples=samples,
        post_mean=samples.mean(axis=1).astype(np.float32),
        post_std=samples.std(axis=1, ddof=0).astype(np.float32),
        sbc_ranks=(samples < theta[:, None, :]).sum(axis=1).astype(np.int32),
        log_prob_true=np.full(n, 1.5, dtype=np.float32),
        sim_ids=np.array([f"sim{i}" for i in range(n)]),
        meta_json=json.dumps(meta),
    )
    return str(path)


def ranks_to_u(theta, samples):
    s = samples.shape[1]
    return ((samples < theta[:, None, :]).sum(axis=1) + 0.5) / (s + 1.0)


# --------------------------------------------------------------- 7. the arm table
def test_arms_is_arm_labels_with_the_prefix_off():
    assert len(pm.ARMS) == len(ARM_LABELS) == N_ARMS
    for short, long in zip(pm.ARMS, ARM_LABELS):
        assert long == "chr" + short
    assert pm.ARMS[0] == "1p" and pm.ARMS[-1] == "22q"


# --------------------------------------------------- 1. the simultaneous ECDF band
def test_the_band_shrinks_with_n():
    wide, narrow = pm.ecdf_band(100), pm.ecdf_band(651)
    assert wide > narrow
    assert pm.ecdf_band(651) > pm.ecdf_band(2000)
    # sqrt(n) scaling: the KS statistic's only rate
    assert 0.8 < (wide / narrow) / np.sqrt(651 / 100) < 1.2


def test_the_band_is_seeded_and_near_the_asymptotic_critical_value():
    assert pm.ecdf_band(651, seed=1) == pm.ecdf_band(651, seed=1)
    assert abs(pm.ecdf_band(651) - 1.358 / np.sqrt(651)) < 0.003


def test_a_calibrated_run_stays_inside_the_band_on_at_least_42_arms():
    theta, samples = make_posterior(651, 800, seed=0)
    u = ranks_to_u(theta, samples)
    band = pm.ecdf_band(u.shape[0])
    inside = sum(pm.ks_uniform(u[:, j])[0] <= band for j in range(N_ARMS))
    assert inside >= 42, f"only {inside}/44 arms inside the 95 % band"


def test_an_over_confident_run_leaves_the_band_on_every_arm():
    theta, samples = make_posterior(651, 800, shrink=0.5, seed=0)
    u = ranks_to_u(theta, samples)
    band = pm.ecdf_band(u.shape[0])
    outside = sum(pm.ks_uniform(u[:, j])[0] > band for j in range(N_ARMS))
    assert outside == N_ARMS


# ------------------------------------------------------------------- 5. TARP
@pytest.fixture(scope="module")
def tarp_pair():
    theta, samples = make_posterior(400, 500, seed=11)
    _, narrow = make_posterior(400, 500, shrink=0.35, seed=11)
    return theta, samples, narrow


def test_tarp_reproduces_the_diagonal_on_a_calibrated_posterior(tarp_pair):
    theta, samples, _ = tarp_pair
    alpha, ecp = tp.tarp_from_samples(theta, samples, n_alpha=50, seed=3)
    assert alpha.shape == ecp.shape == (50,)
    assert np.abs(ecp - alpha).max() < 0.05
    assert ecp[0] == 0.0


def test_tarp_falls_below_the_diagonal_when_the_posterior_is_too_narrow(tarp_pair):
    """Below the diagonal everywhere that matters, and a negative signed atc.

    The bulk starts at alpha = 0.15, not at 0: a *severely* narrow posterior makes f
    bimodal - for each tumour either every draw beats the truth or none does - so the 3 %
    of tumours with f = 0 lift the curve above the diagonal for the first few alphas
    before it drops. That crossing is the shape of the failure, not an exception to it.
    """
    theta, _, narrow = tarp_pair
    alpha, ecp = tp.tarp_from_samples(theta, narrow, n_alpha=50, seed=3)
    bulk = (alpha >= 0.15) & (alpha <= 0.95)
    assert (ecp[bulk] < alpha[bulk]).all()
    f = tp.tarp_coverage_values(theta, narrow, seed=3)
    assert tp.tarp_scalars(alpha, ecp, f)["atc_signed"] < 0


def test_atc_is_a_distance_and_the_signed_one_carries_the_direction(tarp_pair):
    theta, samples, narrow = tarp_pair
    alpha, ecp = tp.tarp_from_samples(theta, samples, n_alpha=50, seed=3)
    f = tp.tarp_coverage_values(theta, samples, seed=3)
    good = tp.tarp_scalars(alpha, ecp, f)
    alpha_n, ecp_n = tp.tarp_from_samples(theta, narrow, n_alpha=50, seed=3)
    bad = tp.tarp_scalars(alpha_n, ecp_n, tp.tarp_coverage_values(theta, narrow, seed=3))
    assert good["atc"] >= 0 and bad["atc"] > 0
    assert bad["atc"] > 10 * good["atc"]
    assert bad["atc_signed"] < 0              # sbi's convention: under-dispersed
    assert good["ks_p"] > 0.01 > bad["ks_p"]


def test_coverage_values_are_uniform_for_a_calibrated_posterior(tarp_pair):
    theta, samples, _ = tarp_pair
    f = tp.tarp_coverage_values(theta, samples, seed=3)
    assert f.shape == (400, 1)
    assert 0.0 <= f.min() and f.max() <= 1.0
    assert abs(float(f.mean()) - 0.5) < 0.05


def test_more_reference_points_give_more_coverage_values(tarp_pair):
    theta, samples, _ = tarp_pair
    f = tp.tarp_coverage_values(theta[:40], samples[:40], num_references=3, seed=3)
    assert f.shape == (40, 3)


def test_the_empirical_prior_reference_also_lands_on_the_diagonal(tarp_pair):
    """TARP holds for any reference distribution, so the two must agree when calibrated."""
    theta, samples, _ = tarp_pair
    alpha, ecp = tp.tarp_from_samples(theta, samples, n_alpha=50, reference="prior", seed=3)
    assert np.abs(ecp - alpha).max() < 0.08


def test_the_prior_reference_never_hands_a_tumour_its_own_truth(tarp_pair):
    """Centring the reference on the same simulation's truth breaks TARP's exchangeability:
    the truth is then the nearest point by construction and f collapses to 0."""
    theta, _, _ = tarp_pair
    refs = tp.draw_references(theta, num_references=4, reference="prior", seed=3)
    assert refs.shape == (400, 4, N_ARMS)
    for i in range(theta.shape[0]):
        assert not (refs[i] == theta[i]).all(axis=1).any()


def test_an_unknown_reference_is_refused():
    with pytest.raises(ValueError, match="reference must be"):
        tp.draw_references(np.zeros((4, 2)), reference="truth")


def test_a_constant_scale_cannot_change_the_statistic(tarp_pair):
    """All 44 dimensions share one prior, so dividing every one by 0.2 is a no-op on f."""
    theta, samples, _ = tarp_pair
    raw = tp.tarp_coverage_values(theta[:50], samples[:50], z_score="none", seed=3)
    scaled = tp.tarp_coverage_values(theta[:50], samples[:50], z_score="prior_sd",
                                     prior_sd=PRIOR_SD, seed=3)
    np.testing.assert_allclose(raw, scaled)


def test_the_bootstrap_band_brackets_the_curve(tarp_pair):
    theta, samples, _ = tarp_pair
    alpha = np.linspace(0, 1, 25)
    f = tp.tarp_coverage_values(theta, samples, seed=3)
    ecp = tp.ecp_curve(f, alpha)
    lo, hi = tp.tarp_bootstrap(f, alpha, n_boot=200, seed=3)
    assert (lo <= ecp + 1e-12).all() and (ecp <= hi + 1e-12).all()
    assert (hi - lo)[1:-1].min() > 0


def test_distance_scale_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="z_score must be one of"):
        tp.distance_scale(np.zeros((3, 2)), z_score="whatever")


def test_a_constant_dimension_does_not_divide_by_zero():
    theta = np.zeros((5, 3))
    theta[:, 0] = np.arange(5)
    assert (tp.distance_scale(theta, "theta_sd") > 0).all()


# ------------------------------------------------- 4. the summary_arrays contract
def test_summary_arrays_has_exactly_the_contract(tmp_path):
    theta, samples = make_posterior(30, 40, seed=5)
    path = write_stage_one(tmp_path / "posteriors_clonemlp_SYN.npz", theta, samples)
    run = pm.load_run(path)
    out = tmp_path / "summary_arrays.npz"
    pm.write_summary_arrays(run, out, "clonemlp")

    with np.load(out, allow_pickle=False) as d:
        assert set(d.files) == set(pm.SUMMARY_ARRAY_KEYS)
        assert d["theta_true"].shape == (30, N_ARMS) and d["theta_true"].dtype == np.float32
        assert d["post_mean"].shape == (30, N_ARMS) and d["post_mean"].dtype == np.float32
        assert d["post_std"].shape == (30, N_ARMS) and d["post_std"].dtype == np.float32
        assert d["u"].shape == (30, N_ARMS) and d["u"].dtype == np.float64
        assert d["sim_ids"].shape == (30,) and d["sim_ids"].dtype.kind == "U"
        assert list(d["arms"]) == pm.ARMS
        assert str(d["model"]) == "clonemlp"
        assert str(d["run_tag"]) == "SYN"
        assert json.loads(str(d["meta_json"]))["num_samples"] == 40
        # u is load_run's u, not a second definition of it
        np.testing.assert_array_equal(d["u"], run["u"])
        np.testing.assert_array_equal(d["theta_true"], run["theta_true"])


def test_summary_arrays_run_tag_is_empty_when_meta_has_none(tmp_path):
    theta, samples = make_posterior(6, 8, d=4, seed=5)
    path = write_stage_one(tmp_path / "posteriors_clonemlp.npz", theta, samples, run_tag=None)
    out = tmp_path / "s.npz"
    pm.write_summary_arrays(pm.load_run(path), out, "clonemlp")
    with np.load(out) as d:
        assert str(d["run_tag"]) == ""
        assert list(d["arms"]) == pm.ARMS[:4]


def test_summary_arrays_name_is_plain_for_one_model_and_tagged_for_many():
    assert pm.summary_arrays_name() == "summary_arrays.npz"
    assert pm.summary_arrays_name("cloneatt") == "summary_arrays_cloneatt.npz"


# ------------------------------------------------------- 2. figure B labels them all
def test_fig_b_labels_every_arm(tmp_path):
    theta, samples = make_posterior(40, 30, seed=2)
    path = write_stage_one(tmp_path / "posteriors_clonemlp_SYN.npz", theta, samples)
    run = pm.load_run(path)
    rows = pm.per_arm_table(run)
    base = str(tmp_path / "fig_B")
    # rebuild the figure through the public function, then read its Text objects back
    import matplotlib.pyplot as plt
    pm.fig_contraction({"clonemlp": run}, {"clonemlp": rows}, base)
    assert os.path.exists(base + ".png") and os.path.exists(base + ".pdf")

    fig, ax = plt.subplots()
    contr = np.array([r["contraction"] for r in rows])
    mz = np.array([r["mean_z"] for r in rows])
    texts = pm._label_points(ax, contr, mz, [r["chromosome_arm"] for r in rows],
                             emphasise={0, 1})
    drawn = {t.get_text() for t in texts}
    plt.close(fig)
    assert drawn == set(pm.ARMS), sorted(set(pm.ARMS) - drawn)
    assert len(texts) == N_ARMS and all(t is not None for t in texts)


def test_fig_b_x_range_is_data_driven_and_contains_zero(tmp_path):
    """No reserved space out to 0.45 for a model that never gets past 0.1."""
    theta, samples = make_posterior(40, 30, seed=2)
    path = write_stage_one(tmp_path / "posteriors_clonemlp_SYN.npz", theta, samples)
    run = pm.load_run(path)
    rows = pm.per_arm_table(run)
    for r in rows:                              # force a narrow, low cloud
        r["contraction"] = 0.05 + 0.001 * (rows.index(r) % 5)
    import matplotlib.pyplot as plt
    pm.fig_contraction({"clonemlp": run}, {"clonemlp": rows}, str(tmp_path / "b"))
    fig = plt.figure()                          # the helper closed its own figure
    plt.close(fig)
    lo, hi = min(r["contraction"] for r in rows), max(r["contraction"] for r in rows)
    pad = max(0.09 * (hi - lo), 0.02)
    assert min(lo - pad, 0.0) <= 0.0 <= hi + pad
    assert hi + pad < 0.45


def test_fig_b_canvas_is_one_panel_wide_for_one_model(tmp_path):
    """The audit's geometry bug: a 6.22 in canvas around a 3 in panel."""
    theta, samples = make_posterior(40, 30, seed=2)
    path = write_stage_one(tmp_path / "posteriors_clonemlp_SYN.npz", theta, samples)
    run = pm.load_run(path)
    base = str(tmp_path / "fig_B")
    pm.fig_contraction({"clonemlp": run}, {"clonemlp": pm.per_arm_table(run)}, base)
    import matplotlib.image as mpimg
    width_in = mpimg.imread(base + ".png").shape[1] / 300.0
    assert width_in < pm.PANEL_W_B + 1.2, f"canvas {width_in:.2f} in for one panel"


# ---------------------------------------------------------- 3. figure S1, the grid
def test_fig_s1_returns_44_panels_with_r2_never_above_r_squared(tmp_path):
    theta, samples = make_posterior(120, 60, seed=4)
    path = write_stage_one(tmp_path / "posteriors_clonemlp_SYN.npz", theta, samples)
    run = pm.load_run(path)
    base = str(tmp_path / "fig_S1_recovery_grid")
    stats = pm.fig_recovery_grid(run, base, model="clonemlp")
    assert os.path.exists(base + ".png") and os.path.exists(base + ".pdf")
    assert len(stats) == N_ARMS
    assert [s["chromosome_arm"] for s in stats] == pm.ARMS
    for s in stats:
        # r^2 is the best R2 any affine rescaling could reach, so it is an upper bound
        assert s["true_r2"] <= s["pearson_r"] ** 2 + 1e-9, s
        assert s["n"] == 120
        assert np.isfinite(s["slope"]) and np.isfinite(s["intercept"])


def test_fig_s1_numbers_match_the_per_arm_table(tmp_path):
    theta, samples = make_posterior(80, 40, seed=6)
    path = write_stage_one(tmp_path / "posteriors_clonemlp_SYN.npz", theta, samples)
    run = pm.load_run(path)
    stats = pm.fig_recovery_grid(run, str(tmp_path / "s1"), model="clonemlp")
    table = pm.per_arm_table(run)
    for s, row in zip(stats, table):
        assert s["chromosome_arm"] == row["chromosome_arm"]
        assert abs(s["true_r2"] - row["true_r2"]) < 1e-9
        assert abs(s["pearson_r"] - row["pearson_r"]) < 1e-9


# ---------------------------------------------------- 6. unknown models are loud
def test_unknown_models_finds_the_stranger(tmp_path):
    for name in ("posteriors_clonemlp.npz", "posteriors_armtoken.npz",
                 "posteriors_mystery.npz"):
        (tmp_path / name).write_bytes(b"")
    assert pm.unknown_models(str(tmp_path)) == ["mystery"]


def test_unknown_models_strips_the_run_tag_before_deciding(tmp_path):
    (tmp_path / "posteriors_clonemlp_R1.npz").write_bytes(b"")
    (tmp_path / "posteriors_mystery_R1.npz").write_bytes(b"")
    assert pm.unknown_models(str(tmp_path), "R1") == ["mystery"]
    # a published twin is a known model, tag or no tag
    (tmp_path / "posteriors_cloneatt_published_R1.npz").write_bytes(b"")
    assert pm.unknown_models(str(tmp_path), "R1") == ["mystery"]


def test_main_warns_about_an_unknown_model_instead_of_skipping_it(tmp_path, capsys, monkeypatch):
    theta, samples = make_posterior(20, 30, seed=8)
    write_stage_one(tmp_path / "posteriors_clonemlp.npz", theta, samples, run_tag=None)
    (tmp_path / "posteriors_notamodel.npz").write_bytes(b"")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["poster_metrics", "--in-dir", str(tmp_path), "--out-dir", str(out)])
    pm.main()
    printed = capsys.readouterr().out
    assert "[warn] posteriors_notamodel.npz present but notamodel not in MODEL_ORDER" in printed


# --------------------------------------------------------------- 8. end to end
@pytest.fixture
def tiny_run(tmp_path):
    post = tmp_path / "posteriors"
    post.mkdir()
    theta, samples = make_posterior(60, 40, seed=9)
    write_stage_one(post / "posteriors_clonemlp_SYN.npz", theta, samples)
    return post, tmp_path / "out"


def test_poster_metrics_main_writes_everything(tiny_run, capsys, monkeypatch):
    post, out = tiny_run
    monkeypatch.setattr(sys, "argv",
                        ["poster_metrics", "--in-dir", str(post), "--run-tag", "SYN",
                         "--out-dir", str(out)])
    pm.main()
    capsys.readouterr()
    for name in ("metrics_per_arm.csv", "metrics_summary.csv", "coverage_curve.csv",
                 "summary_arrays.npz",
                 "fig_A_coverage.png", "fig_A_coverage.pdf",
                 "fig_B_contraction_zscore.png", "fig_B_contraction_zscore.pdf",
                 "fig_C_sbc_ecdf.png", "fig_C_sbc_ecdf.pdf",
                 "fig_S1_recovery_grid.png", "fig_S1_recovery_grid.pdf"):
        assert (out / name).exists(), name
    # the CloneMLP-only hack is opt-in now
    assert not (out / "posterior_export.npz").exists()


def test_poster_export_is_written_when_asked_for(tiny_run, capsys, monkeypatch):
    post, out = tiny_run
    monkeypatch.setattr(sys, "argv",
                        ["poster_metrics", "--in-dir", str(post), "--run-tag", "SYN",
                         "--out-dir", str(out), "--poster-export"])
    pm.main()
    capsys.readouterr()
    assert (out / "posterior_export.npz").exists()


def test_two_models_get_one_summary_arrays_file_each(tmp_path, capsys, monkeypatch):
    post, out = tmp_path / "p", tmp_path / "o"
    post.mkdir()
    for model, seed in (("clonemlp", 9), ("cloneatt", 10)):
        theta, samples = make_posterior(40, 30, seed=seed)
        write_stage_one(post / f"posteriors_{model}_SYN.npz", theta, samples, model=model)
    monkeypatch.setattr(sys, "argv",
                        ["poster_metrics", "--in-dir", str(post), "--run-tag", "SYN",
                         "--out-dir", str(out)])
    pm.main()
    capsys.readouterr()
    import glob
    found = sorted(os.path.basename(p) for p in glob.glob(str(out / "summary_arrays*.npz")))
    assert found == ["summary_arrays_cloneatt.npz", "summary_arrays_clonemlp.npz"]
    with np.load(out / "summary_arrays_cloneatt.npz") as d:
        assert str(d["model"]) == "cloneatt"


def test_tarp_main_writes_curve_summary_and_figure(tiny_run, capsys, monkeypatch):
    post, out = tiny_run
    monkeypatch.setattr(sys, "argv",
                        ["tarp", "--in-dir", str(post), "--run-tag", "SYN",
                         "--out-dir", str(out), "--n-alpha", "20", "--n-boot", "40"])
    tp.main()
    capsys.readouterr()
    for name in ("tarp_curve.csv", "tarp_summary.json", "fig_T_tarp.png", "fig_T_tarp.pdf"):
        assert (out / name).exists(), name

    header, *rows = (out / "tarp_curve.csv").read_text().strip().splitlines()
    assert header == "model,alpha,ecp,ecp_lo,ecp_hi"
    assert len(rows) == 20
    assert rows[0].startswith("clonemlp,")

    summary = json.loads((out / "tarp_summary.json").read_text())
    assert isinstance(summary, list) and len(summary) == 1
    entry = summary[0]
    for key in ("model", "run_tag", "atc", "ks_p", "num_sims", "num_samples",
                "num_references", "distance"):
        assert key in entry, key
    assert entry["model"] == "clonemlp" and entry["run_tag"] == "SYN"
    assert entry["num_sims"] == 60 and entry["num_samples"] == 40
    assert entry["num_references"] == 1 and entry["distance"] == "l2"
    assert entry["atc"] >= 0.0


def test_tarp_main_refuses_an_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["tarp", "--in-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="no posteriors"):
        tp.main()


def test_tarp_max_sims_truncates(tiny_run, capsys):
    post, out = tiny_run
    path = str(post / "posteriors_clonemlp_SYN.npz")
    res = tp.run_tarp_on_file(path, "clonemlp", n_alpha=10, n_boot=10, max_sims=15)
    capsys.readouterr()
    assert res["summary"]["num_sims"] == 15
    assert res["f"].shape == (15, 1)


# ------------------------------------------------ review F3: which band is conservative
def test_dkw_lies_above_the_exact_quantile_at_every_n():
    """``dkw_band``'s docstring calls itself conservative; this is what that means.

    DKW's 95 % half-width is 1.3581/sqrt(n), the *asymptotic* KS critical value, and it
    sits above the exact finite-n quantile - so a band drawn from it is a little too wide
    and would keep inside itself arms the title has already counted as rejections. That is
    why figure C draws ``ecdf_band`` instead.
    """
    from scipy.stats import kstwo
    for n in (50, 100, 651, 2000):
        assert pm.dkw_band(n) > kstwo.ppf(0.95, n), n


def test_the_simulated_band_tracks_the_exact_quantile_at_the_run_size():
    from scipy.stats import kstwo
    assert abs(pm.ecdf_band(651) - kstwo.ppf(0.95, 651)) < 1e-4


# ------------------------------- review F4: the drawn curve is the statistic, exactly
@pytest.mark.parametrize("n", [7, 100, 651])
def test_the_drawn_curve_extreme_is_the_ks_statistic(n):
    """An ECDF's largest deviation happens *at* a jump, so the polyline must contain it."""
    rng = np.random.default_rng(0)
    u = rng.random(n)
    xs, ys = pm.ecdf_difference_curve(u)
    assert xs.shape == ys.shape == (2 * n + 2,)
    assert xs[0] == 0.0 and xs[-1] == 1.0 and ys[0] == 0.0 and ys[-1] == 0.0
    assert (np.diff(xs) >= -1e-12).all(), "the polyline must run left to right"
    assert float(np.abs(ys).max()) == pm.ks_uniform(u)[0]


def test_leaving_the_band_and_rejecting_are_the_same_500_arms():
    """The picture and the count in figure C's title have to be one claim, not two."""
    n, arms = 651, 500
    u = np.random.default_rng(0).random((n, arms))
    band = pm.ecdf_band(n)
    outside = {j for j in range(arms)
               if np.abs(pm.ecdf_difference_curve(u[:, j])[1]).max() > band}
    rejects = {j for j in range(arms) if pm.ks_uniform(u[:, j])[1] < 0.05}
    assert rejects, "a 500-arm draw with no rejection at all would test nothing"
    assert outside == rejects, sorted(outside ^ rejects)


def test_the_old_grid_sampling_would_have_disagreed():
    """Locks in why the step form is needed: a 201-point grid understates the extreme."""
    n, arms = 651, 500
    u = np.random.default_rng(0).random((n, arms))
    band, grid = pm.ecdf_band(n), np.linspace(0, 1, 201)
    on_grid = {j for j in range(arms)
               if np.abs(np.searchsorted(np.sort(u[:, j]), grid, side="right") / n
                         - grid).max() > band}
    rejects = {j for j in range(arms) if pm.ks_uniform(u[:, j])[1] < 0.05}
    assert on_grid < rejects, "the grid can only ever miss a rejection, never invent one"
    assert len(rejects - on_grid) >= 1


# ------------------------- review F5: the KS test needs independent reference draws
def test_many_references_do_not_inflate_the_rejection_rate():
    """At R = 10 the ten f values of one tumour are strongly dependent.

    Pooling them would hand ``ks_uniform`` N*R "observations" with N*R/R of the
    information, and a perfectly calibrated posterior would be rejected most of the time.
    ``tarp_scalars`` uses the first reference of each tumour, so the size of the test is
    the nominal 5 %.
    """
    alpha = np.linspace(0, 1, 50)
    rejected = pooled_rejected = 0
    for seed in range(20):
        theta, samples = make_posterior(120, 200, seed=seed)
        f = tp.tarp_coverage_values(theta, samples, num_references=10, seed=seed)
        assert f.shape == (120, 10)
        ecp = tp.ecp_curve(f, alpha)
        rejected += tp.tarp_scalars(alpha, ecp, f)["ks_p"] < 0.05
        pooled_rejected += pm.ks_uniform(f.ravel())[1] < 0.05      # the anti-conservative way
    assert rejected <= 3, f"{rejected}/20 false rejections at R=10"
    assert pooled_rejected > rejected, "the pooled test should be visibly worse"


def test_tarp_scalars_takes_the_first_reference_not_the_pool():
    theta, samples = make_posterior(80, 100, seed=21)
    alpha = np.linspace(0, 1, 30)
    f = tp.tarp_coverage_values(theta, samples, num_references=4, seed=2)
    ecp = tp.ecp_curve(f, alpha)
    got = tp.tarp_scalars(alpha, ecp, f)
    assert got["ks_stat"] == pm.ks_uniform(f[:, 0])[0]
    assert got["ks_stat"] != pm.ks_uniform(f.ravel())[0]
    # a plain 1-D array of coverage values is still accepted
    flat = tp.tarp_scalars(alpha, ecp, f[:, 0])
    assert flat["ks_stat"] == got["ks_stat"]
