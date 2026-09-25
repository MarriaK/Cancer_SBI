"""WP2 of the evaluation redesign: the cross-run stage.

Stage 3 turns the per-run outputs of the twelve best-honest runs into the poster
figures. What is checked here:

1. the seed-family manifest is the single source of truth, and it still names the
   twelve runs the campaign settled on;
2. ``collect_best_honest.py``, now reading that manifest instead of a literal,
   still writes the README table that is in the repository (plus the new TARP
   column);
3. figure E reproduces the prototype's numbers *exactly* -- the prototype
   (``docs/figures/proto_per_arm_r2.py``) is the specification of that figure, so
   any drift in the group-by, the seed set or the reindex shows up as a diff;
4. the calibration panel's simulated band actually separates a uniform PIT from a
   skewed one, and still renders when TARP is missing;
5. the pooled figure D reports the coefficient of determination, which for a
   shrunk posterior mean must be below the squared correlation;
6. ``jobs/best_honest.sh`` builds the commands we think it builds;
7. ``figures.py``'s scatter grid no longer prints r^2 under the name R^2;
8. every reader survives a run directory that holds several models side by side
   (``summary_arrays_<model>.npz``, a ``model`` column, a list of TARP summaries),
   and a run that does *not* hold the model asked for yields NaN and a warning --
   never another encoder's numbers under this encoder's name;
9. TARP is reported the way ``tarp.py`` computes it: ``atc`` unsigned, because it is
   a distance, with the direction taken from ``atc_signed``;
10. the number of held-out simulations in a caption comes from the CSVs, not from
    the manifest.

The tests that need the campaign CSVs skip themselves when the checkout has no
``results/2026-09-24/`` (they are 44-row files, so they are in the repo).
"""
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cancer_sbi.evaluation import best_honest_figures as bhf

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
RESULTS = REPO / "results"
CAMPAIGN = RESULTS / "2026-09-24"
COLLECTED = RESULTS / "best_honest"
PROTO_TABLE = REPO / "docs" / "figures" / "proto_fig_E_table.csv"
MANIFEST = SRC / "cancer_sbi" / "evaluation" / "manifests" / "best_honest_2026-09-24.json"
JOB = SRC / "jobs" / "best_honest.sh"

#: The campaign's twelve runs, headline first per encoder.
EXPECTED_RUNS = ["AT0ens3", "AT0", "AT0s1", "AT0s2", "R26", "R26s1",
                 "R2", "R2s1", "R2s2", "D0", "D0s1", "D0s2"]

#: The prototype's group names, in manifest order.
PROTO_NAMES = ["ArmToken", "CloneAtt", "CloneMLP", "DominantClone"]

needs_campaign = pytest.mark.skipif(
    not (CAMPAIGN / "AT0" / "metrics_per_arm.csv").exists(),
    reason="results/2026-09-24/ is not in this checkout")

ARMS = bhf.ARMS


@pytest.fixture(autouse=True)
def _fresh_module_state():
    """The warn-once set and the npz cache are module globals; keep tests independent."""
    bhf._WARNED.clear()
    bhf._ARRAY_CACHE.clear()
    yield
    bhf._WARNED.clear()
    bhf._ARRAY_CACHE.clear()


def _load_collect():
    """Import utilities/collect_best_honest.py without a package."""
    spec = importlib.util.spec_from_file_location(
        "_wp2_collect", SRC / "utilities" / "collect_best_honest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- manifest
def test_manifest_lists_the_twelve_runs():
    man = bhf.load_manifest(MANIFEST)
    assert bhf.manifest_runs(man) == EXPECTED_RUNS
    assert len(man["encoders"]) == 4
    keys = [e["key"] for e in man["encoders"]]
    assert keys == ["armtoken", "cloneatt", "clonemlp", "dominantclone"]
    # the fields every consumer relies on
    for enc in man["encoders"]:
        assert enc["label"].endswith("-NPE")
        assert enc["members"], enc["key"]
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", enc["colour"])
    assert len({e["colour"] for e in man["encoders"]}) == 4, "two encoders share a colour"


def test_manifest_members_exclude_the_ensemble_headline():
    """Error bars must come from re-seeded single runs, never from a mixture of them."""
    man = bhf.load_manifest(MANIFEST)
    armtoken = next(e for e in man["encoders"] if e["key"] == "armtoken")
    assert armtoken["headline"] == "AT0ens3"
    assert "AT0ens3" not in armtoken["members"]


def test_collect_best_honest_reads_the_manifest():
    """The seed families are no longer a literal in the collector."""
    collect = _load_collect()
    man = bhf.load_manifest(MANIFEST)
    assert collect.load_best(MANIFEST) == {
        e["key"]: (e["headline"], list(e["members"]), e["config"], e["why"])
        for e in man["encoders"]}
    assert not hasattr(collect, "BEST"), "importing the module must not read the manifest"
    source = (SRC / "utilities" / "collect_best_honest.py").read_text()
    assert "AT0s1" not in source, "the run list is hard-coded again"
    # the scale and the ranking are read off the CSVs, not typed into the source
    for typed in ("651 held-out", "5,000 posterior draws", "ArmToken \u226b CloneAtt"):
        assert typed not in source, f"{typed!r} is hard-coded again"
    for name in ("summary_arrays*.npz", "tarp_curve.csv", "tarp_summary.json",
                 "fig_S1_*", "fig_T_*"):
        assert name in collect.COPY_GLOBS


@needs_campaign
def test_collect_best_honest_reproduces_the_readme_table(tmp_path, monkeypatch):
    """Same numbers as the committed README, with the TARP column added."""
    root = tmp_path / "results"
    root.mkdir()
    (root / "2026-09-24").symlink_to(CAMPAIGN, target_is_directory=True)
    collect = _load_collect()
    monkeypatch.setattr(sys, "argv", ["collect_best_honest.py", "--results", str(root)])
    collect.main()

    def rows(text):
        out = []
        for line in text.splitlines():
            if line.startswith("|") and "---" not in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                out.append(cells)
        return out

    def without_tarp(row):
        """The eight columns the committed README had before the two TARP columns."""
        return row[:7] + row[9:] if len(row) == 10 else row

    new = rows((root / "best_honest" / "README.md").read_text())
    old = rows((COLLECTED / "README.md").read_text())
    assert len(new) == len(old) == 5                      # header + four encoders
    assert new[0][7] == "TARP dist." and new[0][8] == "TARP direction"
    for n, o in zip(new[1:], old[1:]):
        assert without_tarp(n) == without_tarp(o), "a number in the table moved"
        assert n[7] == n[8] == "—", "no TARP output in this checkout: both columns dash"
    # the prose is derived: this campaign pools three seeds into AT0ens3, so the draws vary
    text = (root / "best_honest" / "README.md").read_text()
    assert "651 held-out simulations" in text
    assert "5,000-15,000 posterior draws" in text
    assert "by headline true R²: armtoken 0.904 > cloneatt 0.568" in text


# ------------------------------------------------------------------- statistics
def test_bh_reject_matches_a_hand_worked_case():
    # p = 0.001, 0.02, 0.9: thresholds 0.0167, 0.0333, 0.05 -> the first two survive
    assert bhf.bh_reject([0.001, 0.02, 0.9]) == 2
    assert bhf.bh_reject([0.9, 0.8, 0.7]) == 0
    assert bhf.bh_reject([]) == 0
    assert bhf.bh_reject([float("nan"), 0.0001]) == 1


def test_spearman_is_rank_based_and_tolerates_ties():
    x = np.arange(10, dtype=float)
    assert bhf.spearman(x, x ** 3) == pytest.approx(1.0)
    assert bhf.spearman(x, -x) == pytest.approx(-1.0)
    tied = np.array([1.0, 1.0, 2.0, 3.0])
    assert np.isfinite(bhf.spearman(tied, np.array([1.0, 2.0, 3.0, 4.0])))


def test_simultaneous_band_is_seeded_and_near_the_ks_value():
    eps = bhf.simultaneous_band(651, n_draws=300, seed=0)
    assert eps == bhf.simultaneous_band(651, n_draws=300, seed=0)
    # the asymptotic two-sided KS 95 % critical value, 1.358/sqrt(n)
    assert eps == pytest.approx(1.358 / np.sqrt(651), abs=0.006)


def test_the_band_defaults_match_the_per_run_figure_c():
    """Figure C (per run, poster_metrics) and this panel must draw the same band."""
    import inspect

    sig = inspect.signature(bhf.simultaneous_band)
    assert sig.parameters["n_draws"].default == 1000
    assert sig.parameters["seed"].default == 20260925
    assert sig.parameters["alpha"].default == 0.05          # gamma = 0.95
    from cancer_sbi.evaluation import poster_metrics
    if hasattr(poster_metrics, "ecdf_band"):
        # same statistic, same draws, same seed: the two must agree to the last bit,
        # even though neither module imports the other
        assert bhf.simultaneous_band(651) == pytest.approx(poster_metrics.ecdf_band(651),
                                                           abs=1e-12)


# --------------------------------------------------------------------- figure E
@needs_campaign
def test_fig_E_reproduces_the_prototype_numbers():
    man = bhf.load_manifest(MANIFEST)
    proto = {}
    with open(PROTO_TABLE) as fh:
        for row in csv.DictReader(fh):
            proto[row[""]] = row
    for enc, name in zip(man["encoders"], PROTO_NAMES):
        mean, sd, n = bhf.seed_family(CAMPAIGN, enc, "true_r2")
        want_mean = np.array([float(proto[a][f"{name}_mean"]) for a in ARMS])
        want_sd = np.array([float(proto[a][f"{name}_sd"]) for a in ARMS])
        assert np.allclose(mean, want_mean, atol=1e-6), name
        assert np.allclose(sd, want_sd, atol=1e-6), name
        assert set(n) == {len(enc["members"])}


@needs_campaign
def test_dominantclone_learns_the_other_half_of_the_genome():
    """The finding figure E exists to show: its arm ranking is inverted."""
    man = bhf.load_manifest(MANIFEST)
    fam = {e["key"]: bhf.seed_family(CAMPAIGN, e, "true_r2")[0] for e in man["encoders"]}
    assert bhf.spearman(fam["dominantclone"], fam["armtoken"]) < 0
    # the three clone-set encoders agree with each other
    assert bhf.spearman(fam["clonemlp"], fam["cloneatt"]) > 0.5


@needs_campaign
def test_build_writes_the_figure_set_and_the_s2_table(tmp_path):
    out = bhf.build(CAMPAIGN, MANIFEST, tmp_path)
    figures = tmp_path / "figures"
    for stem in ("fig_E_per_arm_r2", "fig_S3_per_arm_coverage"):
        for ext in ("png", "pdf"):
            assert (figures / f"{stem}.{ext}").exists(), stem
    assert (figures / "per_arm_summary.md").exists()
    assert (figures / "headline_table.md").exists()
    # no posterior arrays in the CSV-only checkout: those two panels are skipped
    assert out["fig_calibration"] is None
    assert out["fig_D"] is None

    with open(figures / "per_arm_summary.csv") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    n_enc, n_stat = 4, len(bhf.SUMMARY_STATS)
    assert len(body) == 44
    assert [r[0] for r in body] == ARMS
    assert len(header) == 1 + n_enc * n_stat * 3     # mean, sd and n for each
    assert header[1] == "armtoken_true_r2_mean"
    assert all(len(r) == len(header) for r in body)


@needs_campaign
def test_per_arm_markdown_carries_true_r2_with_its_seed_sd(tmp_path):
    bhf.build(CAMPAIGN, MANIFEST, tmp_path)
    text = (tmp_path / "figures" / "per_arm_summary.md").read_text()
    man = bhf.load_manifest(MANIFEST)
    mean, sd, _ = bhf.seed_family(CAMPAIGN, man["encoders"][0], "true_r2")
    line = next(ln for ln in text.splitlines() if ln.startswith("| 1p |"))
    assert f"{mean[0]:.3f} ± {sd[0]:.3f}" in line
    assert len([ln for ln in text.splitlines() if ln.startswith("| ") and "---" not in ln]) == 45


@needs_campaign
def test_headline_table_rows_equal_the_csv_numbers(tmp_path):
    out = bhf.build(CAMPAIGN, MANIFEST, tmp_path)
    text = (tmp_path / "figures" / "headline_table.md").read_text()
    man = bhf.load_manifest(MANIFEST)
    for enc, row in zip(man["encoders"], out["headline"]):
        with open(CAMPAIGN / enc["headline"] / "metrics_summary.csv") as fh:
            csv_row = list(csv.DictReader(fh))[-1]
        assert row["true_r2_headline"] == pytest.approx(float(csv_row["mean_true_r2"]))
        assert row["log_prob_true"] == pytest.approx(float(csv_row["mean_log_prob_true"]))
        assert row["coverage_95"] == pytest.approx(float(csv_row["coverage_95"]))
        assert row["sbc_fails_headline"] == int(csv_row["n_arms_ks_reject_fdr05"])
        seeds = [float(bhf.summary_row(CAMPAIGN, enc, m)["mean_true_r2"]) for m in enc["members"]]
        assert row["true_r2_mean"] == pytest.approx(float(np.mean(seeds)))
        assert row["true_r2_sd"] == pytest.approx(float(np.std(seeds, ddof=1)))
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {enc['label']} |"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[1] == enc["headline"]
        assert cells[2] == f"{float(csv_row['mean_true_r2']):.3f}"
        assert cells[4] == f"{float(csv_row['mean_log_prob_true']):.1f}"
        assert cells[5] == f"{float(csv_row['coverage_95']):.3f}"
        assert cells[6].startswith(csv_row["n_arms_ks_reject_fdr05"] + " (")
        assert cells[7] == "—"              # no TARP output in this checkout


@needs_campaign
def test_figures_can_be_rebuilt_from_the_collected_encoder_tree(tmp_path):
    """results/best_honest/ nests runs under <encoder>/; both layouts must work."""
    if not (COLLECTED / "armtoken" / "AT0" / "metrics_per_arm.csv").exists():
        pytest.skip("results/best_honest/ is not in this checkout")
    man = bhf.load_manifest(MANIFEST)
    enc = man["encoders"][0]
    from_campaign = bhf.seed_family(CAMPAIGN, enc, "true_r2")[0]
    from_tree = bhf.seed_family(COLLECTED, enc, "true_r2")[0]
    assert np.allclose(from_campaign, from_tree, equal_nan=True)


# ---------------------------------------------------- synthetic per-run outputs
def _write_per_arm(path, model, ks_p, true_r2=0.5, coverage=0.95):
    fields = ["model", "chromosome_arm", "true_r2", "pearson_r2", "rmse", "contraction",
              "coverage_95", "sbc_ks_p"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for j, arm in enumerate(ARMS):
            w.writerow([model, arm, true_r2, true_r2 + 0.02, 0.1, 0.8, coverage, ks_p[j]])


def _write_summary(path, model, r2=0.5, fails=7):
    fields = ["model", "n_cases", "num_samples", "mean_true_r2", "mean_log_prob_true",
              "coverage_95", "n_arms_ks_reject_fdr05"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerow([model, 200, 5000, r2, 42.0, 0.95, fails])


def _write_arrays(path, u, theta=None, mean=None, std=None, model="synth"):
    n, d = u.shape
    rng = np.random.default_rng(3)
    theta = rng.normal(0, 0.2, (n, d)) if theta is None else theta
    mean = 0.6 * theta if mean is None else mean
    std = np.full((n, d), 0.1) if std is None else std
    np.savez_compressed(
        path, theta_true=theta.astype(np.float32), post_mean=mean.astype(np.float32),
        post_std=std.astype(np.float32), u=u.astype(np.float64),
        sim_ids=np.array([f"sim{i}" for i in range(n)]), arms=np.array(ARMS),
        model=np.array(model), run_tag=np.array("T0"),
        meta_json=np.array(json.dumps({"n_cases": n, "num_samples": 5000, "prior_sd": 0.2})))


def _write_tarp(run_directory, models, n_cases=200):
    """WP1's format: a `model` column in the curve, a list of objects in the summary."""
    alpha = np.linspace(0, 1, 21)
    with open(run_directory / "tarp_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "alpha", "ecp", "ecp_lo", "ecp_hi"])
        for i, model in enumerate(models):
            for a in alpha:
                ecp = min(max(a + 0.01 * i * np.sin(np.pi * a), 0.0), 1.0)
                w.writerow([model, a, ecp, max(ecp - 0.02, 0), min(ecp + 0.02, 1)])
    # atc is a distance (mean |gap|, never negative); the side lives in atc_signed,
    # a SUM over the upper half of the alpha grid -- WP1's tarp.tarp_scalars
    (run_directory / "tarp_summary.json").write_text(json.dumps([
        {"model": model, "run_tag": run_directory.name, "atc": 0.004 + 0.001 * i,
         "atc_signed": -0.11 * (i + 1), "ks_stat": 0.02, "ks_p": 0.4,
         "num_sims": n_cases, "num_samples": 5000, "num_references": 100,
         "distance": "l2", "z_score": "theta_sd", "reference": "prior",
         "n_alpha": len(alpha), "n_boot": 100, "seed": 20260925}
        for i, model in enumerate(models)]))


def _synth_campaign(root, u_by_key, with_tarp=(), n_cases=200):
    """A two-encoder campaign: CSVs for every run, arrays only for the headline."""
    encoders = []
    for i, (key, u) in enumerate(u_by_key.items()):
        head, member = f"{key}0", f"{key}1"
        encoders.append({"key": key, "label": f"{key.title()}-NPE", "model": key,
                         "headline": head, "members": [head, member],
                         "config": f"--model {key}", "why": "synthetic",
                         "colour": ["#2a78d6", "#eb6834"][i % 2]})
        rng = np.random.default_rng(11 + i)
        for run, ks in ((head, rng.uniform(size=44)), (member, rng.uniform(size=44))):
            d = root / run
            d.mkdir(parents=True, exist_ok=True)
            _write_per_arm(d / "metrics_per_arm.csv", key, ks)
            _write_summary(d / "metrics_summary.csv", key)
        _write_arrays(root / head / "summary_arrays.npz", u, model=key)
        if key in with_tarp:
            _write_tarp(root / head, [key], n_cases=n_cases)
    man = {"campaign": "synthetic", "n_test_cases": n_cases, "encoders": encoders}
    path = root / "manifest.json"
    path.write_text(json.dumps(man))
    return man, path


def test_calibration_panel_separates_a_uniform_pit_from_a_skewed_one(tmp_path, capsys):
    n = 400
    rng = np.random.default_rng(5)
    uniform = rng.uniform(size=(n, 44))
    skewed = rng.uniform(size=(n, 44)) ** 2          # piles up near 0: over-confident
    man, _ = _synth_campaign(tmp_path, {"good": uniform, "bad": skewed})
    paths, stats = bhf.fig_calibration_panel(tmp_path, man, str(tmp_path / "cal"),
                                             band_draws=200)
    assert paths is not None and paths["png"].exists() and paths["pdf"].exists()

    good, bad = stats["good"], stats["bad"]
    assert good["n_cases"] == n
    assert good["band"] == pytest.approx(1.358 / np.sqrt(n), abs=0.01)
    # a calibrated PIT stays inside its own 95 % band on (nearly) every arm
    assert (good["max_dev"] <= good["band"]).mean() >= 0.9
    # a skewed one leaves it on every arm
    assert (bad["max_dev"] > bad["band"]).all()
    # the FDR count and its seed range come from the per-arm CSVs
    assert good["fdr_headline"] is not None
    lo, hi = good["fdr_range"]
    assert 0 <= lo <= hi <= 44

    out = capsys.readouterr().out
    assert "no tarp_curve.csv" in out, "a missing TARP curve must warn, not raise"


def test_calibration_panel_adds_a_tarp_row_when_the_curve_is_there(tmp_path):
    rng = np.random.default_rng(6)
    u = rng.uniform(size=(150, 44))
    man, _ = _synth_campaign(tmp_path, {"good": u}, with_tarp=("good",))
    paths, stats = bhf.fig_calibration_panel(tmp_path, man, str(tmp_path / "cal"),
                                             band_draws=100)
    assert paths["png"].exists()
    curve, summary = bhf.load_tarp(tmp_path, man["encoders"][0], "good0")
    assert set(curve) == {"alpha", "ecp", "ecp_lo", "ecp_hi"}
    assert summary["atc"] == pytest.approx(0.004)
    assert summary["atc"] >= 0, "atc is a distance, never negative"
    gap, verdict = bhf.tarp_direction(summary)
    # atc_signed -0.11 over the upper half of a 21-point grid = 11 points
    assert gap == pytest.approx(-0.11 / 11)
    assert verdict == "over-confident"


def test_a_run_dir_holding_several_models_is_filtered_down_to_this_one(tmp_path):
    """results/published/ carries three models side by side in one directory."""
    n = 80
    rng = np.random.default_rng(21)
    run = tmp_path / "P0"
    run.mkdir()
    models = ["clonemlp", "cloneatt", "dominantclone"]
    # one CSV per statistic, all three models in it
    arm_fields = ["model", "chromosome_arm", "true_r2", "pearson_r2", "rmse", "contraction",
                  "coverage_95", "sbc_ks_p"]
    with open(run / "metrics_per_arm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(arm_fields)
        for i, model in enumerate(models):
            for arm in ARMS:
                w.writerow([model, arm, 0.1 * (i + 1), 0.2, 0.1, 0.8, 0.95, 0.5])
    with open(run / "metrics_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n_cases", "mean_true_r2", "mean_log_prob_true",
                    "coverage_95", "n_arms_ks_reject_fdr05"])
        for i, model in enumerate(models):
            w.writerow([model, n, 0.1 * (i + 1), 10.0 * (i + 1), 0.95, i])
    # ... and one array bundle per model
    for i, model in enumerate(models):
        _write_arrays(run / f"summary_arrays_{model}.npz",
                      rng.uniform(size=(n, 44)), model=model)
    _write_tarp(run, models, n_cases=n)

    enc = {"key": "cloneatt", "label": "CloneAtt-NPE", "model": "cloneatt", "headline": "P0",
           "members": ["P0"], "config": "-", "why": "-", "colour": "#eb6834"}
    man = {"campaign": "published", "n_test_cases": n, "encoders": [enc]}

    assert bhf.per_arm_values(tmp_path, enc, "P0", "true_r2")[0] == pytest.approx(0.2)
    assert bhf.summary_row(tmp_path, enc, "P0")["mean_log_prob_true"] == "20.0"
    arrays = bhf.load_summary_arrays(tmp_path, enc, "P0")
    assert str(arrays["model"]) == "cloneatt"
    curve, summary = bhf.load_tarp(tmp_path, enc, "P0")
    assert summary["model"] == "cloneatt"
    assert summary["atc"] == pytest.approx(0.005)
    assert curve["alpha"].size == 21, "the other two models' rows must not be in the curve"
    rows = bhf.headline_rows(tmp_path, man)
    assert rows[0]["tarp_atc"] == pytest.approx(0.005)
    assert rows[0]["tarp_gap"] == pytest.approx(-0.22 / 11)
    assert rows[0]["tarp_verdict"] == "over-confident"
    assert rows[0]["true_r2_headline"] == pytest.approx(0.2)


def test_calibration_panel_skips_itself_when_no_arrays_exist(tmp_path, capsys):
    man, _ = _synth_campaign(tmp_path, {"good": np.random.default_rng(1).uniform(size=(50, 44))})
    os.remove(tmp_path / "good0" / "summary_arrays.npz")
    bhf._ARRAY_CACHE.clear()
    bhf._WARNED.clear()
    paths, stats = bhf.fig_calibration_panel(tmp_path, man, str(tmp_path / "cal"))
    assert paths is None and stats == {}
    assert "figure skipped" in capsys.readouterr().out


def test_arrays_with_reordered_arms_are_called_out(tmp_path, capsys):
    """Everything below reads the 44 columns positionally, so a reorder must be loud."""
    u = np.random.default_rng(8).uniform(size=(30, 44))
    man, _ = _synth_campaign(tmp_path, {"good": u})
    path = tmp_path / "good0" / "summary_arrays.npz"
    arrays = dict(np.load(path, allow_pickle=False))
    arrays["arms"] = np.array(list(reversed(ARMS)))
    np.savez_compressed(path, **arrays)
    bhf._ARRAY_CACHE.clear()
    assert bhf.load_summary_arrays(tmp_path, man["encoders"][0], "good0") is not None
    assert "genomic order" in capsys.readouterr().out


def test_a_run_without_this_model_yields_nothing_not_another_models_numbers(tmp_path, capsys):
    """The failure this guards against is one encoder's numbers under another's name."""
    n = 60
    run = tmp_path / "X0"
    run.mkdir()
    rng = np.random.default_rng(31)
    _write_per_arm(run / "metrics_per_arm.csv", "dominantclone", rng.uniform(size=44),
                   true_r2=0.17)
    _write_summary(run / "metrics_summary.csv", "dominantclone", r2=0.17)
    _write_arrays(run / "summary_arrays_dominantclone.npz", rng.uniform(size=(n, 44)),
                  model="dominantclone")
    _write_tarp(run, ["dominantclone"], n_cases=n)

    enc = {"key": "armtoken", "label": "ArmToken-NPE", "model": "armtoken", "headline": "X0",
           "members": ["X0"], "config": "-", "why": "-", "colour": "#2a78d6"}
    man = {"campaign": "x", "n_test_cases": n, "encoders": [enc]}

    values = bhf.per_arm_values(tmp_path, enc, "X0", "true_r2")
    assert np.isnan(values).all(), "0.17 is DominantClone's number, not ArmToken's"
    mean, sd, counts = bhf.seed_family(tmp_path, enc, "true_r2")
    assert np.isnan(mean).all() and (counts == 0).all()
    assert bhf.summary_row(tmp_path, enc, "X0") is None
    assert bhf.load_summary_arrays(tmp_path, enc, "X0") is None
    assert bhf.load_tarp(tmp_path, enc, "X0") == (None, None)
    assert bhf.headline_rows(tmp_path, man) == []

    out = capsys.readouterr().out
    assert "no rows for model armtoken" in out
    assert "found dominantclone" in out
    assert "no summary_arrays for model armtoken" in out
    assert "0.17" not in out

    # and nothing is drawn from an all-NaN family
    assert bhf.fig_E_per_arm_r2(tmp_path, man, str(tmp_path / "figE")) is None
    assert not (tmp_path / "figE.png").exists()


def test_collect_refuses_a_run_that_has_no_row_for_this_model(tmp_path):
    """The collector hard-fails rather than copying another model's row into the table."""
    collect = _load_collect()
    run = tmp_path / "results" / "2026-09-24" / "X0"
    run.mkdir(parents=True)
    _write_summary(run / "metrics_summary.csv", "dominantclone")
    man = {"campaign": "x", "encoders": [
        {"key": "armtoken", "label": "ArmToken-NPE", "model": "armtoken", "headline": "X0",
         "members": ["X0"], "config": "-", "why": "-", "colour": "#2a78d6"}]}
    path = tmp_path / "man.json"
    path.write_text(json.dumps(man))
    with pytest.raises(SystemExit) as excinfo:
        collect.read_summary(run, "armtoken")
    assert "no row for model armtoken" in str(excinfo.value)
    assert "found dominantclone" in str(excinfo.value)
    # the one-model file that predates the column still reads
    plain = tmp_path / "plain"
    plain.mkdir()
    with open(plain / "metrics_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_cases", "mean_true_r2"])
        w.writerow([651, 0.5])
    assert collect.read_summary(plain, "armtoken")["mean_true_r2"] == "0.5"


def test_tarp_is_printed_as_a_distance_plus_a_direction(tmp_path):
    """`atc` is non-negative; the side of the diagonal comes from `atc_signed`."""
    u = np.random.default_rng(41).uniform(size=(100, 44))
    man, man_path = _synth_campaign(tmp_path, {"good": u}, with_tarp=("good",))
    out = bhf.build(tmp_path, man_path, tmp_path / "out", band_draws=100)
    row = out["headline"][0]
    assert row["tarp_atc"] > 0
    assert row["tarp_gap"] == pytest.approx(-0.11 / 11)
    assert row["tarp_verdict"] == "over-confident"

    text = (tmp_path / "out" / "figures" / "headline_table.md").read_text()
    header = next(ln for ln in text.splitlines() if ln.startswith("| Encoder |"))
    assert "TARP dist." in header and "TARP direction" in header
    line = next(ln for ln in text.splitlines() if ln.startswith("| Good-NPE |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[7] == "0.004", "the distance carries no sign"
    assert cells[8] == "-0.010 (over-confident)"


def test_the_case_count_comes_from_the_csvs_not_the_manifest(tmp_path):
    """A figure caption states what the run evaluated, not what somebody typed."""
    u = np.random.default_rng(51).uniform(size=(30, 44))
    man, man_path = _synth_campaign(tmp_path, {"good": u}, n_cases=200)
    man["n_test_cases"] = 999999                       # the manifest is wrong on purpose
    Path(man_path).write_text(json.dumps(man))
    assert bhf.data_n_cases(tmp_path, man) == 200      # _write_summary wrote n_cases=200

    out = bhf.build(tmp_path, man_path, tmp_path / "out", band_draws=50)
    preamble = (tmp_path / "out" / "figures" / "headline_table.md").read_text()
    assert "200 held-out simulations" in preamble
    assert "999999" not in preamble
    assert out["fig_E"] is not None


def test_fig_D_panel_true_r2_is_below_the_squared_correlation(tmp_path):
    n, d = 300, 44
    rng = np.random.default_rng(9)
    theta = rng.normal(0, 0.2, (n, d))
    mean = 0.5 * theta + rng.normal(0, 0.02, (n, d))        # a shrunk posterior mean
    u = rng.uniform(size=(n, d))
    man, _ = _synth_campaign(tmp_path, {"good": u})
    _write_arrays(tmp_path / "good0" / "summary_arrays.npz", u, theta=theta, mean=mean,
                  model="good")
    bhf._ARRAY_CACHE.clear()
    paths, stats = bhf.fig_D_panel(tmp_path, man, str(tmp_path / "figD"))
    assert paths["png"].exists() and paths["pdf"].exists()
    s = stats["good"]
    t, m = theta.ravel(), mean.ravel()
    want = 1.0 - ((t - m) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    assert s["true_r2"] == pytest.approx(want)
    assert s["true_r2"] < s["pearson_r2"], "shrinkage must cost R^2, not r^2"
    assert s["slope"] == pytest.approx(0.5, abs=0.02)
    assert s["n"] == n * d


def test_a_single_seed_encoder_gets_no_error_bars(tmp_path, capsys):
    """One member means no sd: the figure must say so instead of drawing zero bars."""
    u = np.random.default_rng(2).uniform(size=(60, 44))
    man, _ = _synth_campaign(tmp_path, {"good": u})
    man["encoders"][0]["members"] = ["good0"]
    mean, sd, n = bhf.seed_family(tmp_path, man["encoders"][0], "true_r2")
    assert np.isnan(sd).all()
    assert set(n) == {1}
    bhf.fig_E_per_arm_r2(tmp_path, man, str(tmp_path / "figE"))
    assert (tmp_path / "figE.png").exists()


def test_build_degrades_to_the_csv_only_layout(tmp_path):
    """No arrays, no TARP: two figures and every table still come out."""
    root = tmp_path / "campaign"
    man, man_path = _synth_campaign(root, {"good": np.random.default_rng(4).uniform(size=(40, 44))})
    os.remove(root / "good0" / "summary_arrays.npz")
    bhf._ARRAY_CACHE.clear()
    out = bhf.build(root, man_path, tmp_path / "out")
    figs = tmp_path / "out" / "figures"
    assert (figs / "fig_E_per_arm_r2.png").exists()
    assert (figs / "fig_S3_per_arm_coverage.png").exists()
    assert not (figs / "fig_calibration_panel.png").exists()
    assert not (figs / "fig_D_panel.png").exists()
    assert (figs / "headline_table.md").exists()
    assert len(out["headline"]) == 1


# ----------------------------------------------------------------- job script
def _dry_run(tmp_path, **env):
    full = dict(os.environ)
    full.update({"DRY_RUN": "1", "MANIFEST": str(MANIFEST),
                 "RESULTS": str(tmp_path / "results" / "2026-09-24"),
                 "OUT": str(tmp_path / "results" / "best_honest")})
    full.update(env)
    return subprocess.run(["bash", str(JOB)], capture_output=True, text=True,
                          env=full, timeout=120)


def test_best_honest_sh_dry_run_covers_every_run_and_stage(tmp_path):
    r = _dry_run(tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    results = str(tmp_path / "results" / "2026-09-24")
    for run in EXPECTED_RUNS:
        assert f"=== {run}" in out
        for mod in ("poster_metrics", "tarp", "fig_shrinkage"):
            assert (f"python -m cancer_sbi.evaluation.{mod} --in-dir {results}/{run}/posteriors "
                    f"--out-dir {results}/{run} --run-tag {run}") in out, (mod, run)
    # 12 runs x 3 per-run commands
    assert out.count("--run-tag") == len(EXPECTED_RUNS) * 3
    # the two collection commands
    assert f"python utilities/collect_best_honest.py --results {tmp_path / 'results'}" in out
    assert "--campaign 2026-09-24 --out best_honest" in out
    assert (f"python -m cancer_sbi.evaluation.best_honest_figures --results {results}") in out
    assert f"--out-dir {tmp_path / 'results' / 'best_honest'}" in out
    assert "conda activate" not in out, "a dry run must not touch the environment"


def test_best_honest_sh_refuses_an_out_that_is_not_beside_results(tmp_path):
    r = _dry_run(tmp_path, OUT=str(tmp_path / "elsewhere" / "best_honest"))
    assert r.returncode == 1
    assert "REFUSING" in r.stderr


def test_best_honest_sh_refuses_a_missing_manifest(tmp_path):
    r = _dry_run(tmp_path, MANIFEST=str(tmp_path / "nope.json"))
    assert r.returncode == 1
    assert "REFUSING" in r.stderr


def test_best_honest_sh_takes_its_runs_from_the_manifest(tmp_path):
    """Change the manifest, and the job changes with it."""
    man = json.loads(MANIFEST.read_text())
    man["encoders"] = [{"key": "solo", "label": "Solo-NPE", "model": "solo", "headline": "X9",
                        "members": ["X9"], "config": "--model solo", "why": "test",
                        "colour": "#123456"}]
    path = tmp_path / "small.json"
    path.write_text(json.dumps(man))
    r = _dry_run(tmp_path, MANIFEST=str(path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("--run-tag") == 3
    assert "=== X9" in r.stdout
    assert "AT0" not in r.stdout


def test_best_honest_sh_is_executable_and_linted_like_its_siblings():
    assert os.access(JOB, os.X_OK)
    text = JOB.read_text()
    assert "#SBATCH -p general" in text
    assert "#SBATCH -t 02:00:00" in text
    assert "set -euo pipefail" in text
    assert "cancer/code" not in text            # there is no code/ level on the cluster


# ------------------------------------------------------- figures.py grid title
def test_scatter_grid_title_prints_the_true_r2_and_r(tmp_path, monkeypatch):
    """``figures.py:701`` used to label the squared Pearson correlation R^2."""
    from cancer_sbi.evaluation import figures

    rng = np.random.default_rng(12)
    n = 150
    theta = rng.normal(0, 0.2, (n, 44))
    mean = 0.45 * theta + rng.normal(0, 0.03, (n, 44))     # heavy, deliberate shrinkage

    titles = []
    real_set_title = figures.plt.Axes.set_title

    def spy(self, label, *a, **kw):
        titles.append(label)
        return real_set_title(self, label, *a, **kw)

    monkeypatch.setattr(figures.plt.Axes, "set_title", spy)
    # rendering two 4x6 grids at 300 dpi is the slow part and proves nothing here
    monkeypatch.setattr(figures, "_save_png_pdf",
                        lambda fig, out_dir, stem, **kw: {"png": Path(f"{stem}.png"),
                                                          "pdf": Path(f"{stem}.pdf")})
    figures.plot_true_vs_postmean_scatter(theta, mean, tmp_path)

    assert len(titles) == 44
    pattern = re.compile(r"^(?P<arm>\S+),\s+\$R\^2 = (?P<r2>-?\d+\.\d+)\$"
                         r"\s+\(\$r = (?P<r>-?\d+\.\d+)\$\)$")
    for j, title in enumerate(titles):
        m = pattern.match(title)
        assert m, title
        t, p = theta[:, j], mean[:, j]
        want_r2 = 1.0 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        want_r = float(np.corrcoef(t, p)[0, 1])
        assert m["r2"] == f"{want_r2:.2f}", title
        assert m["r"] == f"{want_r:.2f}", title
        # the whole point: the coefficient of determination is the smaller number
        assert float(m["r2"]) < want_r ** 2


def test_scatter_grid_colours_the_title_on_the_true_r2(tmp_path, monkeypatch):
    """A panel whose r^2 is high but whose R^2 is not must not read as green."""
    from cancer_sbi.evaluation import figures

    rng = np.random.default_rng(13)
    n = 200
    theta = rng.normal(0, 0.2, (n, 44))
    mean = 0.3 * theta + rng.normal(0, 0.01, (n, 44))      # r^2 ~ 0.99, true R^2 < 0
    colours = []

    real_set_title = figures.plt.Axes.set_title

    def spy(self, label, *a, **kw):
        colours.append(kw.get("color"))
        return real_set_title(self, label, *a, **kw)

    monkeypatch.setattr(figures.plt.Axes, "set_title", spy)
    monkeypatch.setattr(figures, "_save_png_pdf",
                        lambda fig, out_dir, stem, **kw: {"png": Path(f"{stem}.png"),
                                                          "pdf": Path(f"{stem}.pdf")})
    figures.plot_true_vs_postmean_scatter(theta, mean, tmp_path)
    assert colours and all(c == figures.R2_POOR_ORANGE for c in colours)
