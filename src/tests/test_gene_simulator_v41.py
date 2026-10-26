"""Prior v4.1 (PRIOR_VERSION 5) of the gene-level simulator (simulate_gene_selection.py).

Design: research_notes/v4_1_design_2026-10-26/DECISION.md, scope revised after the design review: v4.1 =
PASS_TRUNK_GENS 1000, PASS_RATE_RANGE logU(1e-4, 1.5e-2) and one copula loading per rate (RHO_CIN_LOADINGS,
whole-chromosome rate decoupled). pi, the block draws and the passenger overlay are v4's.
  1. Reproduction: sample_theta with sgs.V4_SETTINGS equals a frozen copy of the v4 sample_theta bit for bit and the
     v4 calibration parameters.pkl (data_sim/gene_calib_v4 sim1-5, read-only) to ~1 ulp; with sgs.V3_SETTINGS the
     same against data_sim/gene_calib_v3 sim1-21.
  2. Pairing: under the v4.1 defaults delta, driver set, pi, q, m, length_mean, p_term, cin_level, u_pass, the
     *_indep rates (and the realised arm / focal rates, same loading) are v4's for the same prior seed.
  3. Loadings: chromosomal_rate == chromosomal_rate_indep exactly at loading 0; each loading changes only its own
     rate; correlations follow sqrt(rho_i rho_j); invalid loadings and the removed scalar RHO_CIN raise.
  4. End to end at min_detectable 2e4 under the v4.1 defaults with GENE_SIM_STRICT=1, and the checker on its output.
Tiny tumours only; nothing here touches the cluster, SISTEM's files or the calibration data (read-only).
"""
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

HERE = Path(__file__).resolve()
DATA_GEN = HERE.parents[1] / "data_generation"
if str(DATA_GEN) not in sys.path:
    sys.path.insert(0, str(DATA_GEN))

import simulate_gene_selection as sgs                       # noqa: E402  (runs the import-time self-tests)
import passenger_overlay as po                              # noqa: E402
import check_gene_selection_outputs as chk                 # noqa: E402

PROJ = Path("/Users/maryamkafi/Documents/PhD_Project/cancer")
V3_ROOT = PROJ / "data_sim" / "gene_calib_v3"
V4_ROOT = PROJ / "data_sim" / "gene_calib_v4"
TINY_MIN_DET = 2e4
SIM_ID = 1
V41_LOADINGS = {"arm_rate": 0.5, "chromosomal_rate": 0.0, "focal_driver_rate": 0.5, "pass_rate": 0.5}


def _prior_seed(sim_id, base=sgs.DEFAULT_SEED):
    return base + sim_id * sgs.PRIOR_SEED_STRIDE


def _flat(delta_dict, chrom_order=sgs.CHROM_ORDER):
    return np.asarray([x for c in chrom_order for x in delta_dict[c]], dtype=np.float64)


def _theta_under(settings, seed, monkeypatch, nregions=None, chrom_order=sgs.CHROM_ORDER):
    for k, v in settings.items():
        monkeypatch.setattr(sgs, k, v)
    if nregions is None:
        _, _, nregions = sgs.genome_grid()
    return sgs.sample_theta(np.random.default_rng(seed), nregions, chrom_order)


# ============================================================================ 1. V4 / V3 settings reproduce v4 / v3
def _sample_theta_v4_reference(rng, nregions, chrom_order):
    """Frozen copy of sample_theta as of prior v4 (commit 4ec27d750), constants inlined (RHO_CIN 0.5)."""
    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    arm_rate = logu(3e-6, 3e-4)
    chromosomal_rate = logu(3e-7, 1e-4)
    focal_driver_rate = logu(3e-5, 1e-3)
    length_mean = logu(1.0, 8.0)
    pi = float(rng.uniform(0.02, 0.12))
    q = float(rng.uniform(0.30, 0.55))
    m = float(rng.uniform(0.05, 0.15))
    delta_dict = {}
    for c in chrom_order:
        deltas = []
        for r in range(nregions[c]):
            z = int(rng.uniform() < pi)
            if z == 0:
                deltas.append(0.0)
            else:
                eta = min(rng.exponential(scale=m), 0.6)
                sign = 1.0 if rng.uniform() < q else -1.0
                deltas.append(float(np.exp(sign * eta) - 1))
        delta_dict[c] = deltas
    p_term = float(rng.uniform(0.25, 0.75))
    cin_level = float(rng.uniform())
    u_pass = float(rng.uniform())
    rho = 0.5
    indep = {"arm_rate": (arm_rate, (3e-6, 3e-4)), "chromosomal_rate": (chromosomal_rate, (3e-7, 1e-4)),
             "focal_driver_rate": (focal_driver_rate, (3e-5, 1e-3))}
    realised = {k: float(sgs.loguniform_from_quantile(sgs.cin_copula_quantile(sgs.loguniform_quantile(r, *rg),
                                                                              cin_level, rho), *rg))
                for k, (r, rg) in indep.items()}
    return {**realised, "length_mean": length_mean, "pi": pi, "q": q, "m": m, "delta_dict": delta_dict,
            "p_term": p_term, "cin_level": cin_level,
            "pass_rate": float(sgs.loguniform_from_quantile(sgs.cin_copula_quantile(u_pass, cin_level, rho), 5e-4, 1.5e-2)),
            "arm_rate_indep": arm_rate, "chromosomal_rate_indep": chromosomal_rate,
            "focal_driver_rate_indep": focal_driver_rate,
            "pass_rate_indep": float(sgs.loguniform_from_quantile(u_pass, 5e-4, 1.5e-2))}


V3_SCALARS = ("arm_rate", "chromosomal_rate", "focal_driver_rate", "length_mean", "pi", "q", "m", "p_term")
V4_SCALARS = V3_SCALARS + ("cin_level", "pass_rate", "pass_rate_indep", "arm_rate_indep", "chromosomal_rate_indep",
                           "focal_driver_rate_indep")
# the fields v4.1 must keep bit-identical to v4 (everything drawn, plus the realised rates whose loading is unchanged)
PAIRED_WITH_V4 = ("length_mean", "pi", "q", "m", "p_term", "cin_level", "arm_rate_indep", "chromosomal_rate_indep",
                  "focal_driver_rate_indep", "arm_rate", "focal_driver_rate")


def _compare_with_calibration(P, th, scalars, seed):
    """parameters.pkl written on the cluster vs a local draw: np.exp / uniform differ there in the last ulp (the
    unmodified v4 code run locally shows the same 1-2e-16 differences, checked 2026-10-26; test_gene_simulator_v3
    found the same for v2), so floats to 1e-12 relative and delta to 1e-15 absolute; the driver set and every sign
    are exact."""
    assert P["prior_seed"] == seed
    for k in scalars:
        assert np.isclose(P[k], th[k], rtol=1e-12, atol=0), k
    dp, dt = _flat(P["delta_dict"]), _flat(th["delta_dict"])
    assert np.array_equal(dp != 0, dt != 0) and np.array_equal(np.sign(dp), np.sign(dt))
    assert np.allclose(dp, dt, rtol=0, atol=1e-15)
    assert np.array_equal(P["driver_mask"], dt != 0) and P["n_drivers"] == int((dt != 0).sum())


def _v4_params(sim_id):
    pfile = V4_ROOT / "simulation_outputs" / f"sim{sim_id}" / "parameters.pkl"
    if not pfile.is_file():
        pytest.skip(f"{pfile} not on this machine")
    with open(pfile, "rb") as f:
        P = pickle.load(f)
    assert P["prior_version"] == 4 and P["rho_cin"] == 0.5
    return P


@pytest.mark.parametrize("sim_id", [1, 2, 3, 4, 5])
def test_v4_settings_reproduce_v4_theta(sim_id, monkeypatch):
    seed = _prior_seed(sim_id)
    _, _, nregions = sgs.genome_grid()
    ref = _sample_theta_v4_reference(np.random.default_rng(seed), nregions, sgs.CHROM_ORDER)
    th = _theta_under(sgs.V4_SETTINGS, seed, monkeypatch)
    assert sgs.rho_cin_loadings() == {k: 0.5 for k in sgs.CIN_RATE_KEYS}
    assert set(th) == set(ref)
    for k in ref:                                                   # bit for bit, same machine: every field
        assert th[k] == ref[k], k
    _compare_with_calibration(_v4_params(sim_id), th, V4_SCALARS, seed)


@pytest.mark.parametrize("sim_id", list(range(1, 22)))
def test_v3_settings_reproduce_v3_theta(sim_id, monkeypatch):
    pfile = V3_ROOT / "simulation_outputs" / f"sim{sim_id}" / "parameters.pkl"
    if not pfile.is_file():
        pytest.skip(f"{pfile} not on this machine")
    seed = _prior_seed(sim_id)
    th = _theta_under(sgs.V3_SETTINGS, seed, monkeypatch)
    assert sgs.rho_cin_loadings() == {k: 0.0 for k in sgs.CIN_RATE_KEYS} and sgs.PASSENGERS_ENABLED is False
    with open(pfile, "rb") as f:
        P = pickle.load(f)
    assert P["prior_version"] == 3
    _compare_with_calibration(P, th, V3_SCALARS, seed)
    for k in ("arm_rate", "chromosomal_rate", "focal_driver_rate", "pass_rate"):
        assert th[k] == th[f"{k}_indep"], k                       # no copula at all under v3


# ============================================================================ 2. delta pairing v4.1 vs v4
@pytest.mark.parametrize("sim_id", [1, 2, 3, 4, 5])
def test_v41_defaults_keep_delta_and_draws_paired_with_v4(sim_id):
    seed = _prior_seed(sim_id)
    _, _, nregions = sgs.genome_grid()
    th = sgs.sample_theta(np.random.default_rng(seed), nregions, sgs.CHROM_ORDER)     # v4.1 module defaults
    ref = _sample_theta_v4_reference(np.random.default_rng(seed), nregions, sgs.CHROM_ORDER)
    assert th["delta_dict"] == ref["delta_dict"]                    # bit for bit: delta, driver set, signs
    for k in PAIRED_WITH_V4:
        assert th[k] == ref[k], k
    # only the realised chromosomal rate (loading 0) and the passenger rate (new floor) differ
    assert th["chromosomal_rate"] == th["chromosomal_rate_indep"] != ref["chromosomal_rate"]
    assert th["pass_rate"] != ref["pass_rate"] and th["pass_rate_indep"] != ref["pass_rate_indep"]
    u = sgs.loguniform_quantile(th["pass_rate_indep"], *sgs.PASS_RATE_RANGE)           # same u_pass, new range
    assert abs(u - sgs.loguniform_quantile(ref["pass_rate_indep"], 5e-4, 1.5e-2)) < 1e-12
    # and against the v4 calibration on disk (cluster: ~1 ulp)
    _compare_with_calibration(_v4_params(sim_id), th, PAIRED_WITH_V4, seed)


def test_v41_delta_paired_with_v4_for_200_ids():
    _, _, nregions = sgs.genome_grid()
    for sim_id in range(1, 201):
        seed = _prior_seed(sim_id)
        th = sgs.sample_theta(np.random.default_rng(seed), nregions, sgs.CHROM_ORDER)
        ref = _sample_theta_v4_reference(np.random.default_rng(seed), nregions, sgs.CHROM_ORDER)
        assert th["delta_dict"] == ref["delta_dict"], sim_id
        assert all(th[k] == ref[k] for k in PAIRED_WITH_V4), sim_id


def test_v41_defaults_and_settings_dicts():
    for d in (sgs.V4_SETTINGS, sgs.V3_SETTINGS):
        for k in d:
            assert hasattr(sgs, k), k
    assert sgs.PRIOR_VERSION == 5
    assert (sgs.PASS_TRUNK_GENS, sgs.PASS_RATE_RANGE) == (1000.0, (1e-4, 1.5e-2))
    assert sgs.rho_cin_loadings() == V41_LOADINGS and sgs.CIN_RATE_KEYS == tuple(V41_LOADINGS)
    assert set(sgs.V4_SETTINGS) == {"PASS_TRUNK_GENS", "PASS_RATE_RANGE", "RHO_CIN_LOADINGS"}
    assert sgs.V4_SETTINGS["PASS_TRUNK_GENS"] == 2000.0 and sgs.V4_SETTINGS["PASS_RATE_RANGE"] == (5e-4, 1.5e-2)
    # unchanged by v4.1 (the pi floor and driver guard were dropped in review)
    assert (sgs.ARM_RATE_RANGE, sgs.CHROM_RATE_RANGE, sgs.FOCAL_RATE_RANGE, sgs.PTERM_RANGE, sgs.Q_RANGE, sgs.M_RANGE,
            sgs.PI_RANGE) == ((3e-6, 3e-4), (3e-7, 1e-4), (3e-5, 1e-3), (0.25, 0.75), (0.30, 0.55), (0.05, 0.15),
                              (0.02, 0.12))
    assert not hasattr(sgs, "RHO_CIN") and not hasattr(sgs, "MIN_DRIVERS")
    # the overlay is v4's: no loss-rule option; production passes the v4.1 trunk constant
    assert [f.name for f in dataclasses.fields(po.PassengerParams)] == \
           ["rate", "loss_share", "term_share", "len_mean", "term_len_mean", "max_cn", "trunk_gens", "min_ploidy",
            "max_ploidy"]
    assert sgs.passenger_params(0.1).trunk_gens == 1000.0


# ============================================================================ 3. per-rate loadings
RATE_RANGES = {"arm_rate": sgs.ARM_RATE_RANGE, "chromosomal_rate": sgs.CHROM_RATE_RANGE,
               "focal_driver_rate": sgs.FOCAL_RATE_RANGE, "pass_rate": sgs.PASS_RATE_RANGE}
CHANGES = ("arm_rate", "chromosomal_rate", "focal_driver_rate", "pass_rate")


def _draws(n, seed, nreg={"chrZ": 0}, order=("chrZ",)):
    """n real sample_theta draws on an empty genome (only the scalar draws run; same draw order otherwise)."""
    rng = np.random.default_rng(seed)
    return [sgs.sample_theta(rng, nreg, list(order)) for _ in range(n)]


def test_zero_loading_is_exact_for_chromosomal_rate():
    for t in _draws(2000, 1):
        assert t["chromosomal_rate"] == t["chromosomal_rate_indep"]          # exact, no transform


@pytest.mark.parametrize("key", CHANGES)
def test_each_loading_changes_only_its_own_rate(key, monkeypatch):
    base = _draws(300, 11)
    monkeypatch.setattr(sgs, "RHO_CIN_LOADINGS", {**V41_LOADINGS, key: 0.9})
    mod = _draws(300, 11)
    for a, b in zip(base, mod):
        for k in a:
            if k == key:
                continue
            assert a[k] == b[k], (key, k)
        assert a[key] != b[key]
    # and the changed rate is the documented copula at its new loading
    rg = RATE_RANGES[key]
    for b in mod[:50]:
        u = sgs.cin_copula_quantile(sgs.loguniform_quantile(b[f"{key}_indep"], *rg), b["cin_level"], 0.9)
        assert np.isclose(b[key], float(sgs.loguniform_from_quantile(u, *rg)), rtol=1e-12, atol=0)


def test_loadings_set_the_correlations_and_keep_marginals():
    th = _draws(20_000, 7)
    z = {}
    for k, rg in RATE_RANGES.items():
        q = sgs.loguniform_quantile(np.array([t[k] for t in th]), *rg)
        assert stats.kstest(q, "uniform").pvalue > 0.01, k
        z[k] = stats.norm.ppf(np.clip(q, 1e-12, 1 - 1e-12))
    zL = stats.norm.ppf(np.array([t["cin_level"] for t in th]))
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    assert abs(c(z["arm_rate"], z["focal_driver_rate"]) - 0.5) < 0.02            # sqrt(0.5 * 0.5)
    assert abs(c(z["arm_rate"], z["pass_rate"]) - 0.5) < 0.02
    assert abs(c(z["arm_rate"], zL) - np.sqrt(0.5)) < 0.02
    assert abs(c(z["chromosomal_rate"], z["arm_rate"])) < 0.03 and abs(c(z["chromosomal_rate"], zL)) < 0.03


def test_unequal_loadings_give_sqrt_product_correlation(monkeypatch):
    monkeypatch.setattr(sgs, "RHO_CIN_LOADINGS", {"arm_rate": 0.8, "chromosomal_rate": 0.2,
                                                  "focal_driver_rate": 0.5, "pass_rate": 0.0})
    th = _draws(20_000, 9)
    z = {k: stats.norm.ppf(np.clip(sgs.loguniform_quantile(np.array([t[k] for t in th]), *rg), 1e-12, 1 - 1e-12))
         for k, rg in RATE_RANGES.items()}
    c = lambda a, b: float(np.corrcoef(z[a], z[b])[0, 1])
    assert abs(c("arm_rate", "focal_driver_rate") - np.sqrt(0.4)) < 0.02
    assert abs(c("arm_rate", "chromosomal_rate") - np.sqrt(0.16)) < 0.02
    assert abs(c("arm_rate", "pass_rate")) < 0.03
    assert all(t["pass_rate"] == t["pass_rate_indep"] for t in th)


def test_invalid_loadings_and_removed_rho_cin_raise(monkeypatch):
    with pytest.raises(TypeError):
        sgs.RHO_CIN_LOADINGS["arm_rate"] = 0.9                      # read-only: replace the mapping, never edit it
    monkeypatch.setattr(sgs, "RHO_CIN_LOADINGS", {"arm_rate": 0.5, "focal_driver_rate": 0.5, "pass_rate": 0.5})
    with pytest.raises(ValueError, match="exactly the keys"):
        _draws(1, 0)
    monkeypatch.setattr(sgs, "RHO_CIN_LOADINGS", {**V41_LOADINGS, "arm_rate": 1.5})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _draws(1, 0)
    monkeypatch.setattr(sgs, "RHO_CIN_LOADINGS", dict(V41_LOADINGS))
    monkeypatch.setattr(sgs, "RHO_CIN", 0.8, raising=False)        # a v4 harness: must not be silently ignored
    with pytest.raises(RuntimeError, match="RHO_CIN_LOADINGS"):
        _draws(1, 0)


# ============================================================================ 4. end to end under the v4.1 defaults
@pytest.fixture(scope="module")
def e2e_v41(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("v41e2e") / "on"
    try:
        mp.setenv(sgs.STRICT_ENV_VAR, "1")
        mp.setattr(sgs, "MIN_DETECTABLE", TINY_MIN_DET)
        mp.setattr(sgs, "SIM_LOG", False)
        _, _, nregions = sgs.genome_grid()
        th = sgs.sample_theta(np.random.default_rng(_prior_seed(SIM_ID)), nregions, sgs.CHROM_ORDER)
        th = {**th, "pass_rate": 0.015}                               # top of PASS_RATE_RANGE: tiny tumours, few gens
        sim_dir = root / "simulation_outputs" / f"sim{SIM_ID}"
        sim_dir.mkdir(parents=True)
        sgs.write_manifest(str(root), sgs.DEFAULT_SEED, 1, None)
        sgs.simulate_gene_model(th, str(sim_dir), 1, _prior_seed(SIM_ID), SIM_ID)
    finally:
        mp.undo()
    with open(sim_dir / "1" / "results.pkl", "rb") as f:
        res = pickle.load(f)
    with open(sim_dir / "parameters.pkl", "rb") as f:
        P = pickle.load(f)
    z = dict(np.load(sim_dir / "1" / "clones.npz")) if res["complete"] else None
    return root, th, res, z, P


def test_e2e_v41_outputs_and_bookkeeping(e2e_v41):
    root, th, res, z, P = e2e_v41
    assert res["complete"], res
    assert P["prior_version"] == 5 and P["rho_cin_loadings"] == V41_LOADINGS and "rho_cin" not in P
    assert P["chromosomal_rate"] == P["chromosomal_rate_indep"]
    assert P["passenger_params"]["trunk_gens"] == 1000.0
    assert res["n_passengers_proposed"] == res["n_passengers_applied"] + res["n_passengers_rejected"]
    assert res["n_trunk_proposed"] == res["n_trunk_applied"] + res["n_trunk_rejected"] and res["n_trunk_applied"] > 0
    assert (z["block_cn"] >= 1).all()
    col = {c: i for i, c in enumerate(z["event_columns"].tolist())}
    cats = z["event_categories"].tolist()
    pas = np.isin(z["events"][:, col["category"]], (cats.index("passenger_gain"), cats.index("passenger_loss")))
    assert pas.sum() == res["n_passengers_applied"] + res["n_trunk_applied"]
    M = json.loads((root / "manifest.json").read_text())
    pr = M["prior"]
    assert M["prior_version"] == 5 and pr["rho_cin_loadings"] == V41_LOADINGS and "rho_cin" not in pr
    assert pr["pi"] == ["uniform", 0.02, 0.12] and pr["pass_rate"] == ["loguniform", 1e-4, 1.5e-2]
    assert pr["draw_order"].endswith("cin_level, u_pass (last)")
    ch = pr["v4_to_v41_changes"]
    assert len(ch["changes"]) == 3 and all("DECISION.md" in x for x in ch["changes"])
    assert pr["v4_settings"] == {"PASS_TRUNK_GENS": 2000.0, "PASS_RATE_RANGE": [5e-4, 1.5e-2],
                                 "RHO_CIN_LOADINGS": {k: 0.5 for k in sgs.CIN_RATE_KEYS}}
    ov = M["passenger_overlay"]
    assert ov["trunk_gens"] == 1000.0 and "Q4.1" in ov["known_artefact"]


def test_checker_passes_on_v41_output(e2e_v41, tmp_path):
    root = e2e_v41[0]
    out = tmp_path / "chk_v41"
    rc = chk.main(["--root", str(root), "--out-dir", str(out)])
    text = (out / "gene_pilot_checks.txt").read_text()
    assert rc == 0, text
    assert "prior_version 5" in text and "WARNING: unknown prior_version" not in text
    assert "by schema {3: 1}" in text and "[INFO] C9" in text and "0-copy blocks: trials 0, clones 0" in text
    assert 5 in chk.KNOWN_PRIOR_VERSIONS
