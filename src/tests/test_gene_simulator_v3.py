"""Prior v3 of the gene-level simulator (simulate_gene_selection.py): patches D-I and D-H, p_term, events schema 2.

Covers decision.test_requirements of research_notes/v3_design_2026-10-21/workflow_result.json:
  1. D-H installed with p_term = 0 (and D-I swapped back to the v2 D-E body) reproduces a run with the
     wrapper uninstalled, bit for bit (block_cn, popsize, events).
  2. sample_theta v3 is paired with v2: same block positions, driver set and |log(1+delta)| for a prior
     seed (checked against the real v2 calibration parameters.pkl when present); p_term is the last draw.
  3. _P_TERM is reset to 0.0 after simulate_gene_model returns or raises; the end-of-trial assert fires
     if theta['p_term'] is mutated; parameters.pkl holds p_term.
  4. events carry the 'terminal' column (schema 2); the checker passes on a v3 output and still reads v2.
  5. install_DH is idempotent and refuses unknown wrappers.
  6. The D-I and D-H self-tests run at import.
Tiny tumours (min_detectable 2e4, ~2 s per run). Nothing here touches the cluster or SISTEM's files.

Prior v4 (2026-10-23) adapted a few assertions here that v4 made obsolete (PRIOR_VERSION 4, events schema 3, two
extra prior draws after p_term, the caller's new keyword arguments); each such place says "v4:". The v4
behaviour itself is tested in test_gene_simulator_v4.py. Prior v4.1 (2026-10-26) changed only the version tag
asserted here (PRIOR_VERSION 5; "v4.1:"); V3_SETTINGS still restores v3 (now on top of V4_SETTINGS).
"""
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve()
DATA_GEN = HERE.parents[1] / "data_generation"
if str(DATA_GEN) not in sys.path:
    sys.path.insert(0, str(DATA_GEN))

import simulate_gene_selection as sgs                       # noqa: E402  (runs the import-time self-tests)
import check_gene_selection_outputs as chk                 # noqa: E402
import sistem.lineage.mutation as mutation                 # noqa: E402
from sistem import Parameters                              # noqa: E402
from sistem.selection.region_library import BaseRegionLibrary   # noqa: E402

V2_ROOT = Path("/Users/maryamkafi/Documents/PhD_Project/cancer/data_sim/gene_calib_v2")
V2_CONSTANTS = {"Q_RANGE": (0.40, 0.65), "FOCAL_RATE_RANGE": (1e-4, 2e-3)}
SIM_ID = 1                                                 # completes at 2e4 with seed prior_seed + 1
TINY_MIN_DET = 2e4


def _prior_seed(sim_id, base=sgs.DEFAULT_SEED):
    return base + sim_id * sgs.PRIOR_SEED_STRIDE


def _theta(sim_id=SIM_ID):
    _, _, nregions = sgs.genome_grid()
    return sgs.sample_theta(np.random.default_rng(_prior_seed(sim_id)), nregions, sgs.CHROM_ORDER)


def _tiny_run(theta, out_dir, seed):
    """One simulate_gene_model_caller attempt at min_detectable 2e4, seeded like a production trial."""
    chrom_lens, arm_ratios, _ = sgs.genome_grid()
    torch.manual_seed(seed)
    np.random.seed(seed % sgs.SEED_MODULUS)
    params = Parameters(region_len=sgs.REGION_LEN, focal_driver_rate=theta["focal_driver_rate"],
                        length_mean=theta["length_mean"], max_distinct_driv_ratio=sgs.MAX_DISTINCT_DRIV_RATIO,
                        min_detectable=TINY_MIN_DET, arm_rate=theta["arm_rate"],
                        chromosomal_rate=theta["chromosomal_rate"], out_dir=str(out_dir), log_path=os.devnull)
    return sgs.simulate_gene_model_caller(params, theta["delta_dict"], chrom_lens, arm_ratios)


# ----------------------------------------------------------------------------- 6. import-time self-tests
def test_patches_installed_and_selftests_pass_at_import():
    assert BaseRegionLibrary.compute_fitness is sgs._saturated_compute_fitness
    assert mutation.gen_focal_event is sgs._gen_focal_event_v3
    assert sgs._is_upstream_gen_focal(sgs._ORIG_GEN_FOCAL)
    assert sgs._P_TERM == 0.0
    # v4: PRIOR_VERSION is 4 since 2026-10-23 (D-I / D-H constants unchanged); v4.1: 5 since 2026-10-26
    assert sgs.PRIOR_VERSION == 5 and sgs.GAIN_SAT_EXTRA == 1 and sgs.TERMINAL_MAG == 1 and sgs.TERM_EXT_MEAN == 2.0
    for name in ("region_gain_dosage_saturation", "focal_terminal_events", "region_fitness_rounded_ploidy"):
        assert name in sgs.SISTEM_PATCHES
    # the self-tests are re-runnable and leave the global RNG untouched
    np.random.seed(11)
    before = np.random.get_state()[1].copy()
    sgs._selftest_DI()
    sgs._selftest_DH()
    sgs.apply_sistem_patches()                             # second install is a no-op, self-tests run again
    assert np.array_equal(before, np.random.get_state()[1])
    assert mutation.gen_focal_event is sgs._gen_focal_event_v3


def test_DI_saturates_gain_drivers_only():
    import types

    class Lib:
        regions = {"chrT": 4}
        drivers = {"chrT": [0, 1]}
        def __getitem__(self, c): return {0: 0.4, 1: -0.3}

    def fit(p, n_gain, n_loss, fn=sgs._saturated_compute_fitness):
        cl = types.SimpleNamespace(get_ploidy=lambda: p,
                                   attributes=types.SimpleNamespace(driver_counts={"chrT": {0: n_gain, 1: n_loss}}))
        return fn(Lib(), cl)

    assert fit(2.1, 10, 2) == fit(2.1, 3, 2)                 # ref 2 -> saturates at 3 copies
    assert fit(3.2, 10, 2) == fit(3.2, 4, 2)                 # ref 3 -> saturates at 4 copies
    assert fit(2.1, 2, 2) < fit(2.1, 3, 2)
    for n_loss in range(0, 9):                               # loss driver identical to D-E
        assert fit(2.1, 2, n_loss) == fit(2.1, 2, n_loss, sgs._rounded_compute_fitness)


# ----------------------------------------------------------------------------- 5. install_DH idempotent
def test_install_DH_idempotent_and_refuses_unknown_wrapper():
    hook = mutation.gen_focal_event
    sgs.install_DH()
    sgs.install_DH()
    assert mutation.gen_focal_event is hook is sgs._gen_focal_event_v3

    def foreign(*a, **k):
        return None
    try:
        mutation.gen_focal_event = foreign
        with pytest.raises(AssertionError, match="unknown wrapper"):
            sgs.install_DH()
    finally:
        mutation.gen_focal_event = hook
    assert mutation.gen_focal_event is sgs._gen_focal_event_v3


# ----------------------------------------------------------------------------- 1. p_term = 0 equivalence
def test_DH_at_pterm_zero_with_DE_reproduces_unwrapped_v2_run(tmp_path):
    theta = _theta()
    seed = _prior_seed(SIM_ID) + 1
    saved_fit, saved_hook = BaseRegionLibrary.compute_fitness, mutation.gen_focal_event
    try:
        BaseRegionLibrary.compute_fitness = sgs._rounded_compute_fitness      # v2 fitness (D-E)
        assert sgs._P_TERM == 0.0
        wrapped = _tiny_run(theta, tmp_path, seed)                          # D-H installed, p_term 0
        mutation.gen_focal_event = sgs._ORIG_GEN_FOCAL                      # D-H uninstalled
        unwrapped = _tiny_run(theta, tmp_path, seed)
    finally:
        BaseRegionLibrary.compute_fitness, mutation.gen_focal_event = saved_fit, saved_hook
    assert isinstance(wrapped, dict) and isinstance(unwrapped, dict), "tiny run did not complete"
    for k in ("block_cn", "popsize", "events", "event_counts", "fitness", "ploidy"):
        assert np.array_equal(wrapped[k], unwrapped[k]), k
    term = list(sgs.EVENT_COLUMNS).index("terminal")
    assert wrapped["events"][:, term].sum() == 0


def test_DH_with_pterm_produces_terminal_events_and_consistent_counts(tmp_path, monkeypatch):
    monkeypatch.setenv(sgs.STRICT_ENV_VAR, "1")
    theta = _theta()
    try:
        sgs._P_TERM = theta["p_term"]
        res = _tiny_run(theta, tmp_path, _prior_seed(SIM_ID) + 1)
    finally:
        sgs._P_TERM = 0.0
    assert isinstance(res, dict)
    assert res["driver_count_mismatches"] == 0
    col = {c: i for i, c in enumerate(sgs.EVENT_COLUMNS)}
    E = res["events"]
    term = E[:, col["terminal"]] == 1
    assert term.sum() > 0
    focal = np.isin(E[:, col["category"]], (sgs.EVENT_INDEX["focal_amplification"], sgs.EVENT_INDEX["focal_deletion"]))
    assert np.all(focal[term]) and np.all(E[term, col["driver"]] == 1)
    gains = term & (E[:, col["category"]] == sgs.EVENT_INDEX["focal_amplification"])
    assert np.all(E[gains, col["copies"]] == sgs.TERMINAL_MAG)


# ----------------------------------------------------------------------------- 2. prior pairing with v2
class _RecordingRng:
    """Wraps a numpy Generator and records (method, args) of every draw."""
    def __init__(self, seed):
        self._g, self.calls = np.random.default_rng(seed), []

    def __getattr__(self, name):
        f = getattr(self._g, name)

        def rec(*a, **k):
            self.calls.append((name, a, tuple(sorted(k.items()))))
            return f(*a, **k)
        return rec


def _theta_with(constants, seed):
    """v4: the v3 prior is the v4 code with sgs.V3_SETTINGS (v3 ranges, RHO_CIN = 0; v4.1: all RHO_CIN_LOADINGS 0);
    `constants` go on top."""
    _, _, nregions = sgs.genome_grid()
    constants = {**sgs.V3_SETTINGS, **constants}
    saved = {k: getattr(sgs, k) for k in constants}
    try:
        for k, v in constants.items():
            setattr(sgs, k, v)
        rng = _RecordingRng(seed)
        th = sgs.sample_theta(rng, nregions, sgs.CHROM_ORDER)
    finally:
        for k, v in saved.items():
            setattr(sgs, k, v)
    return th, rng.calls


@pytest.mark.parametrize("sim_id", [1, 2, 3, 7])
def test_sample_theta_v3_paired_with_v2(sim_id):
    seed = _prior_seed(sim_id)
    th3, calls3 = _theta_with({}, seed)
    th2, calls2 = _theta_with(V2_CONSTANTS, seed)
    # p_term is the last v3 draw and the only extra one over v2; everything before it consumes the rng identically.
    # v4: two more draws (cin_level, u_pass) follow p_term, so p_term is calls[-3] (v4 pairing: test_gene_simulator_v4.py)
    assert calls3[-3] == ("uniform", sgs.V3_RANGES["PTERM_RANGE"], ()) and calls3[-2:] == [("uniform", (), ())] * 2
    assert [c[0] for c in calls3] == [c[0] for c in calls2]
    assert 0.35 <= th3["p_term"] <= 0.75 and sgs.V3_RANGES["PTERM_RANGE"] == (0.35, 0.75)
    assert sgs.Q_RANGE[0] <= th3["q"] <= sgs.Q_RANGE[1]
    for k in ("arm_rate", "chromosomal_rate", "length_mean", "pi", "m"):
        assert th3[k] == th2[k], k
    d3 = np.asarray([x for c in sgs.CHROM_ORDER for x in th3["delta_dict"][c]])
    d2 = np.asarray([x for c in sgs.CHROM_ORDER for x in th2["delta_dict"][c]])
    assert np.array_equal(d3 != 0, d2 != 0)                                   # driver set / block positions
    assert np.allclose(np.abs(np.log1p(d3)), np.abs(np.log1p(d2)), rtol=0, atol=1e-12)   # eta = |log(1+delta)|
    same_sign = np.sign(d3) == np.sign(d2)
    assert np.array_equal(d3[same_sign], d2[same_sign])
    # against the real v2 calibration (read-only), when it is on this machine
    pfile = V2_ROOT / "simulation_outputs" / f"sim{sim_id}" / "parameters.pkl"
    if pfile.is_file():
        with open(pfile, "rb") as f:
            P = pickle.load(f)
        assert P["prior_version"] == 2 and P["prior_seed"] == seed
        # The v2 pickles were written on the cluster: np.exp there differs from macOS in the last ulp
        # (checked 2026-10-22: the unmodified v2 code run locally shows the same 1-2e-16 differences),
        # (and uniform(lo, hi) in the last ulp too, e.g. m of sim7), so values are compared to ~1 ulp;
        # the driver set is exact.
        dp = np.asarray([x for c in sgs.CHROM_ORDER for x in P["delta_dict"][c]])
        assert np.array_equal(dp != 0, d2 != 0)
        assert np.allclose(dp, d2, rtol=0, atol=1e-15)
        for k in ("q", "pi", "m", "focal_driver_rate", "arm_rate", "chromosomal_rate", "length_mean"):
            assert np.isclose(P[k], th2[k], rtol=1e-14, atol=0), k
        assert np.array_equal(P["driver_mask"], d3 != 0)


# ----------------------------------------------------------------------------- 3. _P_TERM lifecycle
def _fake_summary():
    D = 589
    # v4: 9 event categories, genealogy arrays and overlay counters are part of the summary (schema 3)
    return {"block_cn": np.full((1, D), 2, np.int16), "popsize": np.array([10], np.int64),
            "ploidy": np.array([2.0], np.float32), "fitness": np.array([1.0]), "nWGD": np.array([0], np.int16),
            "n_altered_drivers": np.array([0], np.int16), "n_altered_drivers_post": np.array([0], np.int16),
            "bulk": np.full(D, 2.0, np.float32),
            "event_counts": np.zeros((1, len(sgs.EVENT_CATEGORIES)), np.int32),
            "events": np.zeros((0, len(sgs.EVENT_COLUMNS)), np.int32), "driver_count_mismatches": 0,
            "node_parent": np.array([-1], np.int32), "node_birth_gen": np.array([0], np.int32),
            "node_clone": np.array([0], np.int32), "clone_node": np.array([0], np.int32),
            **{k: 0 for k in sgs.OVERLAY_RESULT_KEYS}}


def test_pterm_set_per_trial_and_reset_after_return(tmp_path, monkeypatch):
    theta = _theta()
    seen = []

    def fake_caller(params, delta_dict, chrom_lens, arm_ratios, **kw):     # v4: pass_rate / overlay_seed kwargs
        seen.append(sgs._P_TERM)
        return -2 if len(seen) == 1 else _fake_summary()       # one retry, then success
    monkeypatch.setattr(sgs, "simulate_gene_model_caller", fake_caller)
    sgs.simulate_gene_model(theta, str(tmp_path / "sim1"), 2, _prior_seed(SIM_ID), SIM_ID)
    assert seen == [theta["p_term"]] * 3                       # trial 1: retry + success; trial 2: success
    assert sgs._P_TERM == 0.0
    with open(tmp_path / "sim1" / "parameters.pkl", "rb") as f:
        P = pickle.load(f)
    assert P["prior_version"] == sgs.PRIOR_VERSION and P["p_term"] == theta["p_term"]     # v4: version 4
    with np.load(tmp_path / "sim1" / "1" / "clones.npz") as z:
        assert z["event_columns"].tolist() == list(sgs.EVENT_COLUMNS)
        assert int(z["events_schema_version"]) == sgs.EVENTS_SCHEMA_VERSION == 3              # v4: schema 3


def test_pterm_reset_after_raise(tmp_path, monkeypatch):
    def boom(*a, **k):
        assert sgs._P_TERM > 0
        raise RuntimeError("boom")
    monkeypatch.setattr(sgs, "simulate_gene_model_caller", boom)
    with pytest.raises(RuntimeError, match="boom"):
        sgs.simulate_gene_model(_theta(), str(tmp_path / "sim1"), 1, _prior_seed(SIM_ID), SIM_ID)
    assert sgs._P_TERM == 0.0


def test_end_of_trial_assert_fires_if_theta_pterm_mutated(tmp_path, monkeypatch):
    theta = _theta()

    def mutating(*a, **k):
        theta["p_term"] = 0.9
        return _fake_summary()
    monkeypatch.setattr(sgs, "simulate_gene_model_caller", mutating)
    with pytest.raises(AssertionError, match="p_term changed"):
        sgs.simulate_gene_model(theta, str(tmp_path / "sim1"), 1, _prior_seed(SIM_ID), SIM_ID)
    assert sgs._P_TERM == 0.0
    assert not (tmp_path / "sim1" / "1" / "results.pkl").exists()


# ----------------------------------------------------------------------------- 4. events + checker
def test_v3_output_has_terminal_column_and_passes_checker(tmp_path, monkeypatch):
    monkeypatch.setenv(sgs.STRICT_ENV_VAR, "1")
    monkeypatch.setattr(sgs, "MIN_DETECTABLE", TINY_MIN_DET)
    monkeypatch.setattr(sgs, "SIM_LOG", False)
    root = tmp_path / "v3"
    sim_dir = root / "simulation_outputs" / f"sim{SIM_ID}"
    sim_dir.mkdir(parents=True)
    sgs.write_manifest(str(root), sgs.DEFAULT_SEED, 1, None)
    theta = _theta()
    sgs.simulate_gene_model(theta, str(sim_dir), 1, _prior_seed(SIM_ID), SIM_ID)
    assert sgs._P_TERM == 0.0
    M = json.loads((root / "manifest.json").read_text())
    # v4: prior_version 4, p_term floor 0.25, focal floor 3e-5, schema 3 (11 columns, 'node' after 'terminal');
    # the v3 values stay recorded under prior.v3_ranges
    assert M["prior_version"] == 5                                                     # v4.1: 5
    assert M["prior"]["p_term"] == ["uniform", 0.25, 0.75] and M["prior"]["v3_ranges"]["PTERM_RANGE"] == [0.35, 0.75]
    assert M["prior"]["q"] == ["uniform", 0.30, 0.55]
    assert M["prior"]["focal_driver_rate"] == ["loguniform", 3e-5, 1e-3]
    assert M["prior"]["v3_ranges"]["FOCAL_RATE_RANGE"] == [1e-4, 1e-3]
    assert M["events"]["schema_version"] == 3 and M["events"]["event_columns"][-2:] == ["terminal", "node"]
    assert len(M["events"]["event_columns"]) == 11
    sp = M["sistem_patches"]
    assert sp["region_gain_dosage_saturation"]["id"] == "D-I" and sp["focal_terminal_events"]["id"] == "D-H"
    assert sp["superseded"] == {"region_fitness_rounded_ploidy": "region_gain_dosage_saturation"}
    assert M["fixed_parameters"]["n_sat_extra"] == 1 and M["fixed_parameters"]["terminal_mag"] == 1
    assert len(M["prior"]["v2_to_v3_changes"]["caveats"]) == 4
    with np.load(sim_dir / "1" / "clones.npz") as z:
        col = {c: i for i, c in enumerate(z["event_columns"].tolist())}
        assert z["events"].shape[1] == 11 and "terminal" in col                       # v4: 11 columns
        assert z["events"][:, col["terminal"]].sum() > 0
    out = tmp_path / "chk_v3"
    rc = chk.main(["--root", str(root), "--out-dir", str(out)])
    text = (out / "gene_pilot_checks.txt").read_text()
    assert rc == 0, text
    assert "[PASS] C4" in text and "[INFO] C8" in text and "by schema {3: 1}" in text       # v4: schema 3
    assert "prior_version 5" in text and "pooled driver_count_mismatches 0" in text      # v4.1: 5


@pytest.mark.skipif(not (V2_ROOT / "manifest.json").is_file(), reason="v2 calibration data not on this machine")
def test_checker_still_reads_v2_output(tmp_path):
    out = tmp_path / "chk_v2"
    chk.main(["--root", str(V2_ROOT), "--out-dir", str(out), "--max-sims", "1"])   # read-only
    text = (out / "gene_pilot_checks.txt").read_text()
    assert "[INFO] C8" in text and "by schema {1: 25}" in text and "no 'terminal' column" in text
    assert "prior_version 2" in text and "check crashed" not in text


# ----------------------------------------------------------------------------- reload safety (runs last)
def test_reload_rewraps_the_upstream_original():
    importlib.reload(sgs)
    assert mutation.gen_focal_event is sgs._gen_focal_event_v3
    assert sgs._is_upstream_gen_focal(sgs._ORIG_GEN_FOCAL)
    assert BaseRegionLibrary.compute_fitness is sgs._saturated_compute_fitness
