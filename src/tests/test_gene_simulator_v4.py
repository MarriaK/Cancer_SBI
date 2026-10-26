"""Prior v4 of the gene-level simulator (simulate_gene_selection.py) and the passenger overlay (passenger_overlay.py).

  1. Pairing: with V3_RANGES, PTERM_RANGE (0.35, 0.75) and RHO_CIN = 0, sample_theta v4 reproduces v3 exactly
     (frozen v3 reference below, and data_sim/gene_calib_v3 sim1/sim2 parameters.pkl read-only); with the v4
     defaults every uniform quantile, the driver set and |log(1+delta)| stay paired.
  2. Copula: marginals stay log-uniform (KS), corr(log arm, log focal) ~ RHO_CIN, RHO_CIN = 0 is the identity.
  3. Overlay unit tests on a hand-made 3-clone genealogy (rejection rules, windows, rows, counts, bulk, determinism).
  4. End to end at min_detectable 2e4: overlay off == overlay at rate 0; passenger rows; genealogy arrays; schema 3;
     GENE_SIM_STRICT=1 (driver_counts are compared with the PRE-overlay block_cn inside grow()).
  5. Checker passes on the v4 output and still reads v2 and v3 calibration outputs (read-only).

Prior v4.1 (2026-10-26, PRIOR_VERSION 5) changed the module defaults; prior v4 is now the current code with
sgs.V4_SETTINGS, so every test here runs under V4_SETTINGS (autouse fixture _v4_settings; the module-scoped e2e
fixture applies them itself). RHO_CIN became RHO_CIN_LOADINGS (all four loadings = the old RHO_CIN). Assertions
that changed say "v4.1:". v4.1 itself: test_gene_simulator_v41.py.
"""
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats

HERE = Path(__file__).resolve()
DATA_GEN = HERE.parents[1] / "data_generation"
if str(DATA_GEN) not in sys.path:
    sys.path.insert(0, str(DATA_GEN))

import simulate_gene_selection as sgs                       # noqa: E402  (runs the import-time self-tests)
import passenger_overlay as po                              # noqa: E402
import check_gene_selection_outputs as chk                 # noqa: E402
from sistem import Parameters                              # noqa: E402

PROJ = Path("/Users/maryamkafi/Documents/PhD_Project/cancer")
V2_ROOT = PROJ / "data_sim" / "gene_calib_v2"
V3_ROOT = PROJ / "data_sim" / "gene_calib_v3"
TINY_MIN_DET = 2e4
SIM_ID = 1


@pytest.fixture(autouse=True)
def _v4_settings(monkeypatch):
    """v4.1: prior v4 = the current code with sgs.V4_SETTINGS (restored after each test)."""
    for k, v in sgs.V4_SETTINGS.items():
        monkeypatch.setattr(sgs, k, v)


def _loadings(rho):
    """v4.1: the v4 scalar RHO_CIN = rho as per-rate loadings."""
    return {k: rho for k in sgs.CIN_RATE_KEYS}


def _prior_seed(sim_id, base=sgs.DEFAULT_SEED):
    return base + sim_id * sgs.PRIOR_SEED_STRIDE


def _flat(delta_dict, chrom_order=sgs.CHROM_ORDER):
    return np.asarray([x for c in chrom_order for x in delta_dict[c]], dtype=np.float64)


# ============================================================================ 1. pairing with v3
def _sample_theta_v3_reference(rng, nregions, chrom_order):
    """Frozen copy of sample_theta as of prior v3 (commit state 2026-10-22), constants inlined."""
    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    arm_rate = logu(1e-5, 3e-4)
    chromosomal_rate = logu(1e-6, 1e-4)
    focal_driver_rate = logu(1e-4, 1e-3)
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
    p_term = float(rng.uniform(0.35, 0.75))
    return {"arm_rate": arm_rate, "chromosomal_rate": chromosomal_rate, "focal_driver_rate": focal_driver_rate,
            "length_mean": length_mean, "pi": pi, "q": q, "m": m, "delta_dict": delta_dict, "p_term": p_term}


V3_SCALARS = ("arm_rate", "chromosomal_rate", "focal_driver_rate", "length_mean", "pi", "q", "m", "p_term")


def _v4_theta(seed, monkeypatch=None, **const):
    _, _, nregions = sgs.genome_grid()
    for k, v in const.items():
        monkeypatch.setattr(sgs, k, v)
    return sgs.sample_theta(np.random.default_rng(seed), nregions, sgs.CHROM_ORDER)


@pytest.mark.parametrize("prior_seed", [13345, 14345])
def test_pairing_v4_with_v3_settings_reproduces_v3_exactly(prior_seed, monkeypatch):
    _, _, nregions = sgs.genome_grid()
    rng3 = np.random.default_rng(prior_seed)
    th3 = _sample_theta_v3_reference(rng3, nregions, sgs.CHROM_ORDER)
    assert sgs.V3_RANGES["PTERM_RANGE"] == (0.35, 0.75)
    th4 = _v4_theta(prior_seed, monkeypatch, **{**sgs.V3_RANGES, "PTERM_RANGE": (0.35, 0.75),
                                                "RHO_CIN_LOADINGS": _loadings(0.0)})       # v4.1: RHO_CIN = 0
    for k in V3_SCALARS:
        assert th4[k] == th3[k], k                                  # exact, same machine
    assert th4["delta_dict"] == th3["delta_dict"]                   # exact floats, exact driver set and signs
    for k in ("arm_rate", "chromosomal_rate", "focal_driver_rate"):
        assert th4[k] == th4[f"{k}_indep"]                          # RHO_CIN = 0 branch: no transform
    # the two new draws follow p_term on the same stream
    assert th4["cin_level"] == float(rng3.uniform())
    u_pass = float(rng3.uniform())
    assert th4["pass_rate"] == th4["pass_rate_indep"] == float(sgs.loguniform_from_quantile(u_pass, *sgs.PASS_RATE_RANGE))
    # against the v3 calibration on disk (written on the cluster: ~1 ulp differences in np.exp / uniform)
    sim_id = (prior_seed - sgs.DEFAULT_SEED) // sgs.PRIOR_SEED_STRIDE
    pfile = V3_ROOT / "simulation_outputs" / f"sim{sim_id}" / "parameters.pkl"
    if pfile.is_file():
        with open(pfile, "rb") as f:
            P = pickle.load(f)
        assert P["prior_version"] == 3 and P["prior_seed"] == prior_seed
        for k in V3_SCALARS:
            assert np.isclose(P[k], th4[k], rtol=1e-12, atol=0), k
        dp, d4 = _flat(P["delta_dict"]), _flat(th4["delta_dict"])
        assert np.array_equal(dp != 0, d4 != 0) and np.array_equal(np.sign(dp), np.sign(d4))
        assert np.allclose(dp, d4, rtol=0, atol=1e-12)
        assert np.array_equal(P["driver_mask"], d4 != 0)


@pytest.mark.parametrize("prior_seed", [13345, 14345])
def test_pairing_v4_defaults_keep_quantiles_and_driver_set(prior_seed, monkeypatch):
    _, _, nregions = sgs.genome_grid()
    th3 = _sample_theta_v3_reference(np.random.default_rng(prior_seed), nregions, sgs.CHROM_ORDER)
    th4 = _v4_theta(prior_seed)                                     # module defaults (v4 floors, RHO_CIN 0.5)
    assert sgs.RHO_CIN_LOADINGS == _loadings(0.5) and sgs.PTERM_RANGE == (0.25, 0.75)     # v4.1: RHO_CIN 0.5
    for k in ("length_mean", "pi", "q", "m"):
        assert th4[k] == th3[k], k
    d3, d4 = _flat(th3["delta_dict"]), _flat(th4["delta_dict"])
    assert np.array_equal(d3, d4)
    pairs = (("arm_rate", "ARM_RATE_RANGE"), ("chromosomal_rate", "CHROM_RATE_RANGE"),
             ("focal_driver_rate", "FOCAL_RATE_RANGE"))
    for k, rg in pairs:                                             # same uniform, different range
        u3 = sgs.loguniform_quantile(th3[k], *sgs.V3_RANGES[rg])
        u4 = sgs.loguniform_quantile(th4[f"{k}_indep"], *getattr(sgs, rg))
        assert abs(u3 - u4) < 1e-12, k
        lo, hi = getattr(sgs, rg)
        assert lo <= th4[k] <= hi
    assert abs((th3["p_term"] - 0.35) / 0.40 - (th4["p_term"] - 0.25) / 0.50) < 1e-12


# ============================================================================ 2. copula
def test_copula_marginals_uniform_and_correlation():
    n, rho = 20_000, 0.5
    rng = np.random.default_rng(20261023)
    L = rng.uniform(size=n)
    U = rng.uniform(size=(4, n))                                   # arm, chrom, focal, pass quantiles
    ranges = (sgs.ARM_RATE_RANGE, sgs.CHROM_RATE_RANGE, sgs.FOCAL_RATE_RANGE, sgs.PASS_RATE_RANGE)
    rates = [sgs.loguniform_from_quantile(sgs.cin_copula_quantile(sgs.loguniform_quantile(
        sgs.loguniform_from_quantile(u, *rg), *rg), L, rho), *rg) for u, rg in zip(U, ranges)]
    for r, rg in zip(rates, ranges):
        q = sgs.loguniform_quantile(r, *rg)
        assert np.all((r >= rg[0] * (1 - 1e-12)) & (r <= rg[1] * (1 + 1e-12)))
        assert stats.kstest(q, "uniform").pvalue > 0.01
    c = np.corrcoef(np.log(rates[0]), np.log(rates[2]))[0, 1]
    # Pearson on the log (= uniform-quantile) scale is 6/pi*asin(rho/2) = 0.483 at rho 0.5; normal scale is rho
    assert abs(c - rho) < 0.05 and abs(c - 6 / np.pi * np.arcsin(rho / 2)) < 0.02
    zc = np.corrcoef(stats.norm.ppf(sgs.loguniform_quantile(rates[0], *ranges[0])),
                     stats.norm.ppf(sgs.loguniform_quantile(rates[2], *ranges[2])))[0, 1]
    assert abs(zc - rho) < 0.02
    # identity at rho 0
    u0 = U[0]
    assert sgs.cin_copula_quantile(u0, L, 0.0) is u0


def test_sample_theta_copula_20000_draws(monkeypatch):
    """20,000 real sample_theta draws (empty genome, so only the scalar draws run)."""
    nreg, order = {"chrZ": 0}, ["chrZ"]
    rng = np.random.default_rng(7)
    th = [sgs.sample_theta(rng, nreg, order) for _ in range(20_000)]
    keys = (("arm_rate", sgs.ARM_RATE_RANGE), ("chromosomal_rate", sgs.CHROM_RATE_RANGE),
            ("focal_driver_rate", sgs.FOCAL_RATE_RANGE), ("pass_rate", sgs.PASS_RATE_RANGE))
    for k, rg in keys:
        q = sgs.loguniform_quantile(np.array([t[k] for t in th]), *rg)
        assert stats.kstest(q, "uniform").pvalue > 0.01, k
    la, lf = (np.log([t[k] for t in th]) for k in ("arm_rate", "focal_driver_rate"))
    assert abs(np.corrcoef(la, lf)[0, 1] - sgs.RHO_CIN_LOADINGS["arm_rate"]) < 0.05      # v4.1: both loadings 0.5
    lai = np.log([t["arm_rate_indep"] for t in th])
    assert abs(np.corrcoef(lai, np.log([t["focal_driver_rate_indep"] for t in th]))[0, 1]) < 0.03
    # RHO_CIN = 0: realised == indep exactly
    monkeypatch.setattr(sgs, "RHO_CIN_LOADINGS", _loadings(0.0))                          # v4.1: RHO_CIN = 0
    rng = np.random.default_rng(8)
    for _ in range(500):
        t = sgs.sample_theta(rng, nreg, order)
        for k in ("arm_rate", "chromosomal_rate", "focal_driver_rate", "pass_rate"):
            assert t[k] == t[f"{k}_indep"], k


# ============================================================================ 3. overlay unit tests
NREG = {"chrA": 10, "chrB": 6}
ORDER = ["chrA", "chrB"]
CENT = {"chrA": 4, "chrB": 3}
OFFS = {"chrA": 0, "chrB": 10}
D = 16


def PP(**kw):
    """Edge-only overlay (trunk_gens 0) unless a test sets trunk_gens: the original unit tests predate the trunk."""
    kw.setdefault("trunk_gens", 0.0)
    return po.PassengerParams(**kw)


def _genealogy():
    # root (gen 0, extinct) -> node 1 (gen 10, clone 0) -> node 2 (gen 20, clone 1), node 3 (gen 25, clone 2)
    return po.Genealogy(node_parent=np.array([-1, 0, 1, 1], np.int32), node_birth_gen=np.array([0, 10, 20, 25], np.int32),
                        node_clone=np.array([-1, 0, 1, 2], np.int32), clone_node=np.array([1, 2, 3], np.int32))


POP = np.array([50, 30, 20])


def _desc(g, v):
    """Surviving clones whose ancestor-or-self chain contains node v (brute force)."""
    out = []
    for i, x in enumerate(g.clone_node):
        x = int(x)
        while x >= 0 and x != v:
            x = int(g.node_parent[x])
        if x == v:
            out.append(i)
    return out


def _run(B, pp, seed=1, dmask=None, cap=100, nWGD=None):
    g = _genealogy()
    return po.apply_passenger_overlay(B, POP, g, np.zeros(D, bool) if dmask is None else dmask, cap, NREG, ORDER, CENT,
                                      pp, np.random.default_rng(seed), nWGD=nWGD)


def test_overlay_rate_zero_is_identity():
    B = np.full((3, D), 2, np.int16)
    r = _run(B, PP(rate=0.0))
    assert r.n_proposed == 0 and r.events.shape == (0, len(po.EVENT_COLUMNS)) and np.array_equal(r.block_cn, B)
    assert r.block_cn is not B and r.event_counts_add.sum() == 0


def test_overlay_loss_onto_one_copy_rejected():
    B = np.full((3, D), 1, np.int16)
    r = _run(B, PP(rate=0.5, loss_share=1.0))
    assert r.n_proposed > 0 and r.n_rejected_zero == r.n_proposed and r.n_applied == 0
    assert np.array_equal(r.block_cn, B)


def test_overlay_gain_on_driver_at_max_cn_rejected():
    B = np.full((3, D), 3, np.int16)
    r = _run(B, PP(rate=0.5, loss_share=0.0, max_cn=3), dmask=np.ones(D, bool))
    assert r.n_proposed > 0 and r.n_rejected_cap == r.n_proposed and r.n_applied == 0
    # passenger (non-driver) blocks have no cap
    r2 = _run(B, PP(rate=0.5, loss_share=0.0, max_cn=3))
    assert r2.n_rejected_cap == 0 and r2.n_applied == r2.n_proposed and r2.block_cn.max() > 3


def test_overlay_altered_driver_cap_rejected():
    B = np.full((3, D), 2, np.int16)
    r = _run(B, PP(rate=0.5, loss_share=0.0), dmask=np.ones(D, bool), cap=0)
    assert r.n_proposed > 0 and r.n_rejected_altered == r.n_proposed and r.n_applied == 0
    # events that only RESTORE altered drivers to 2 copies are allowed at the cap
    B3 = np.full((3, D), 3, np.int16)
    r3 = _run(B3, PP(rate=0.5, loss_share=1.0), dmask=np.ones(D, bool), cap=D)
    assert r3.n_applied > 0 and r3.n_rejected_altered == 0
    # the reference copy number follows nWGD (2**(nWGD+1)), as SISTEM's check_viability
    B4 = np.full((3, D), 4, np.int16)
    pp = PP(rate=0.5, loss_share=0.0)
    r_wgd = _run(B4, pp, dmask=np.ones(D, bool), cap=3, nWGD=np.ones(3, np.int16))
    r_nowgd = _run(B4, pp, dmask=np.ones(D, bool), cap=3)
    assert r_wgd.n_applied > 0 and r_nowgd.n_applied == 0 and r_nowgd.n_rejected_altered == r_nowgd.n_proposed


def _check_rows_and_reconstruct(r, B0, g):
    col = {c: i for i, c in enumerate(po.EVENT_COLUMNS)}
    E = r.events
    assert np.all(E[:, col["driver"]] == 0) and np.all(E[:, col["copies"]] == 1)
    assert set(E[:, col["category"]].tolist()) <= {po.EVENT_INDEX["passenger_gain"], po.EVENT_INDEX["passenger_loss"]}
    assert np.all(E[:, col["gen"]] == g.node_birth_gen[E[:, col["node"]]]) and np.all(E[:, col["node"]] >= 0)
    # replay the rows: block_cn and event_counts_add follow exactly
    B = B0.astype(np.int64).copy()
    counts = np.zeros((B.shape[0], 2), np.int64)
    for row in E:
        c = ORDER[row[col["chrom"]]]
        s = OFFS[c] + row[col["first_block"]]
        e = OFFS[c] + row[col["last_block"]] + 1
        assert e - s == row[col["n_blocks"]]
        aff = _desc(g, int(row[col["node"]]))
        loss = row[col["category"]] == po.EVENT_INDEX["passenger_loss"]
        B[np.ix_(aff, np.arange(s, e))] += -1 if loss else 1
        counts[aff, 1 if loss else 0] += 1
    assert np.array_equal(B, r.block_cn) and np.array_equal(counts, r.event_counts_add)
    assert int(r.event_counts_add.sum()) == sum(len(_desc(g, int(v))) for v in E[:, col["node"]])
    assert r.n_applied + r.n_rejected == r.n_proposed
    return col


def test_overlay_terminal_windows_anchor_and_never_cross_centromere():
    B0 = np.full((3, D), 5, np.int16)
    g = _genealogy()
    r = _run(B0, PP(rate=2.0, term_share=1.0, term_len_mean=6.0), seed=3)
    assert r.n_applied > 20
    col = _check_rows_and_reconstruct(r, B0, g)
    for row in r.events:
        c = ORDER[row[col["chrom"]]]
        n, cen = NREG[c], CENT[c]
        assert row[col["terminal"]] == 1
        if row[col["arm"]] == 0:
            assert row[col["first_block"]] == 0 and row[col["last_block"]] <= cen - 2     # < whole p arm
        else:
            assert row[col["arm"]] == 1
            assert row[col["last_block"]] == n - 1 and row[col["first_block"]] >= cen + 1  # < whole q arm
    assert {0, 1} <= set(r.events[:, col["arm"]].tolist())


def test_overlay_interstitial_windows_stay_in_one_arm():
    B0 = np.full((3, D), 5, np.int16)
    g = _genealogy()
    r = _run(B0, PP(rate=2.0, term_share=0.0, len_mean=4.0), seed=4)
    col = _check_rows_and_reconstruct(r, B0, g)
    for row in r.events:
        c = ORDER[row[col["chrom"]]]
        cen, n = CENT[c], NREG[c]
        lo, hi = (0, cen) if row[col["arm"]] == 0 else (cen, n)
        assert lo <= row[col["first_block"]] <= row[col["last_block"]] < hi and row[col["terminal"]] == 0


def test_overlay_mixed_rows_counts_bulk_ploidy_and_determinism():
    B0 = np.random.default_rng(0).integers(1, 5, size=(3, D)).astype(np.int16)
    B_in = B0.copy()
    g = _genealogy()
    dm = np.zeros(D, bool)
    dm[[1, 5, 12]] = True
    pp = PP(rate=0.8)
    r = _run(B0, pp, seed=11, dmask=dm, cap=3)
    assert np.array_equal(B0, B_in)                                         # input untouched
    _check_rows_and_reconstruct(r, B0, g)
    assert r.block_cn.dtype == np.int16 and r.bulk.dtype == np.float32 and r.ploidy.dtype == np.float32
    assert np.allclose(r.bulk, (r.block_cn * POP[:, None]).sum(0) / POP.sum(), rtol=0, atol=1e-6)
    assert np.allclose(r.ploidy, r.block_cn.mean(1), rtol=0, atol=1e-6)
    assert (r.block_cn >= 1).all()
    r2 = _run(B0, pp, seed=11, dmask=dm, cap=3)
    for k in ("block_cn", "events", "event_counts_add", "bulk"):
        assert np.array_equal(getattr(r, k), getattr(r2, k)), k
    assert (r.n_proposed, r.n_rejected_zero, r.n_rejected_cap, r.n_rejected_altered) == \
           (r2.n_proposed, r2.n_rejected_zero, r2.n_rejected_cap, r2.n_rejected_altered)
    r3 = _run(B0, pp, seed=12, dmask=dm, cap=3)
    assert not np.array_equal(r.events, r3.events)


def test_overlay_never_draws_root_events_and_window_sampler_matches():
    """The trunk moved to the founder (grow()); the overlay gives the root no events whatever trunk_gens is."""
    B0 = np.full((3, D), 3, np.int16)
    col = {c: i for i, c in enumerate(po.EVENT_COLUMNS)}
    r0 = _run(B0, PP(rate=0.3, trunk_gens=0.0), seed=21)
    r = _run(B0, PP(rate=0.3, trunk_gens=2000.0), seed=21)
    assert r0.n_applied > 0 and not np.any(r.events[:, col["node"]] == 0)
    for k in ("block_cn", "events", "event_counts_add"):
        assert np.array_equal(getattr(r0, k), getattr(r, k)), k
    # sample_passenger_window: four draws, same rules (reference coordinates within the chromosome)
    pp = PP(rate=1.0, term_share=0.5, len_mean=3.0, term_len_mean=5.0)
    rng, ref = np.random.default_rng(3), np.random.default_rng(3)
    table = po.window_table(NREG, ORDER, CENT)
    for _ in range(300):
        c, s0, e0, term, loss = po.sample_passenger_window(rng, pp, NREG, ORDER, CENT, table=table)
        [ref.random(), ref.integers(D), ref.exponential(1.0), ref.random()]
        assert rng.bit_generator.state == ref.bit_generator.state
        n, cen = NREG[c], CENT[c]
        assert 0 <= s0 < e0 <= n and (e0 <= cen or s0 >= cen)          # one arm, never across the centromere
        if term:
            assert (s0 == 0 and e0 <= cen - 1) or (e0 == n and s0 >= cen + 1)
    # default PassengerParams carries the Stage 1 trunk constant
    assert po.PassengerParams(rate=0.1).trunk_gens == 2000.0 and sgs.passenger_params(0.1).trunk_gens == sgs.PASS_TRUNK_GENS


def test_overlay_ploidy_bounds_rejected():
    B0 = np.full((3, D), 2, np.int16)
    g = _genealogy()
    r = _run(B0, PP(rate=1.0, loss_share=1.0, min_ploidy=1.9), seed=5)
    assert r.n_rejected_ploidy > 0 and r.n_applied + r.n_rejected == r.n_proposed
    assert r.ploidy.min() >= 1.9 - 1e-6
    _check_rows_and_reconstruct(r, B0, g)
    rg = _run(B0, PP(rate=1.0, loss_share=0.0, max_ploidy=2.1), seed=5)
    assert rg.n_rejected_ploidy > 0 and rg.ploidy.max() <= 2.1 + 1e-6
    # bounds that cannot bind (the zero rule already keeps every block >= 1 copy) never fire
    assert _run(B0, PP(rate=1.0, loss_share=0.5, min_ploidy=1.0, max_ploidy=8.0), seed=5).n_rejected_ploidy == 0


def test_build_genealogy_order_and_dead_ancestors():
    def clone(name, gen, parent):
        return types.SimpleNamespace(name=name, birth_gen=gen, parent=parent, site=0)
    root = clone("P0", 0, None)
    a = clone("P3", 5, root)            # dead, ancestor of survivors
    b = clone("P1", 5, root)            # dead, no survivors -> not a node
    c = clone("P7", 9, a)               # survivor
    d = clone("P5", 9, a)               # survivor (same gen, created earlier -> lower node id)
    e = clone("P9", 12, c)              # survivor, child of a survivor
    g = po.build_genealogy([e, c, d])
    # nodes: root(0,0), a(5,3), d(9,5), c(9,7), e(12,9)
    assert g.node_birth_gen.tolist() == [0, 5, 9, 9, 12]
    assert g.node_parent.tolist() == [-1, 0, 1, 1, 3]
    assert g.clone_node.tolist() == [4, 3, 2] and g.node_clone.tolist() == [-1, -1, 2, 1, 0]
    assert all(x is not b for x in po.lineage_nodes([e, c, d]))
    with pytest.raises(ValueError):
        po.build_genealogy([clone(None, 3, root)])


# ============================================================================ 4. end to end at 2e4
def _theta():
    _, _, nregions = sgs.genome_grid()
    return sgs.sample_theta(np.random.default_rng(_prior_seed(SIM_ID)), nregions, sgs.CHROM_ORDER)


def _e2e(tmp_path, name, theta, monkeypatch, enabled):
    monkeypatch.setenv(sgs.STRICT_ENV_VAR, "1")
    monkeypatch.setattr(sgs, "MIN_DETECTABLE", TINY_MIN_DET)
    monkeypatch.setattr(sgs, "SIM_LOG", False)
    monkeypatch.setattr(sgs, "PASSENGERS_ENABLED", enabled)
    root = tmp_path / name
    sim_dir = root / "simulation_outputs" / f"sim{SIM_ID}"
    sim_dir.mkdir(parents=True)
    sgs.write_manifest(str(root), sgs.DEFAULT_SEED, 1, None)
    sgs.simulate_gene_model(theta, str(sim_dir), 1, _prior_seed(SIM_ID), SIM_ID)
    with open(sim_dir / "1" / "results.pkl", "rb") as f:
        res = pickle.load(f)
    assert res["complete"], res
    z = dict(np.load(sim_dir / "1" / "clones.npz"))
    with open(sim_dir / "parameters.pkl", "rb") as f:
        P = pickle.load(f)
    return root, res, z, P


@pytest.fixture(scope="module")
def e2e_runs(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("v4e2e")
    try:
        for k, v in sgs.V4_SETTINGS.items():                          # v4.1: module scope, before _v4_settings runs
            mp.setattr(sgs, k, v)
        th = _theta()
        off = _e2e(tmp, "off", dict(th), mp, False)
        zero = _e2e(tmp, "rate0", {**th, "pass_rate": 0.0}, mp, True)
        on = _e2e(tmp, "on", {**th, "pass_rate": 0.015}, mp, True)     # top of PASS_RATE_RANGE: tiny tumours, few gens
        mp.setattr(sgs, "PASS_TRUNK_GENS", 0.0)                        # edge overlay only: growth must equal "off"
        on_nt = _e2e(tmp, "on_notrunk", {**th, "pass_rate": 0.015}, mp, True)
    finally:
        mp.undo()
    return {"off": off, "zero": zero, "on": on, "on_notrunk": on_nt}


def test_e2e_overlay_off_equals_rate_zero(e2e_runs):
    _, res_off, z_off, P_off = e2e_runs["off"]
    _, res_0, z_0, _ = e2e_runs["zero"]
    for k in ("block_cn", "popsize", "events", "event_counts", "fitness", "ploidy", "bulk", "node_parent", "clone_node"):
        assert np.array_equal(z_off[k], z_0[k]), k
    assert res_off["passengers_enabled"] is False and res_0["passengers_enabled"] is True
    assert res_off["n_passengers_proposed"] == res_0["n_passengers_proposed"] == 0
    # v4.1: prior_version tags the code (5); the v4 coupling is recorded as per-rate loadings
    assert P_off["passengers_enabled"] is False and P_off["prior_version"] == sgs.PRIOR_VERSION == 5
    assert P_off["rho_cin_loadings"] == _loadings(0.5)


def test_e2e_passengers_rows_genealogy_schema(e2e_runs):
    root, res_off, z_off, _ = e2e_runs["off"]
    root_on, res, z, P = e2e_runs["on"]
    col = {c: i for i, c in enumerate(z["event_columns"].tolist())}
    cats = z["event_categories"].tolist()
    assert int(z["events_schema_version"]) == 3 and z["events"].shape[1] == 11 and len(cats) == 9
    assert z["event_counts"].shape == (z["block_cn"].shape[0], 9)
    E = z["events"]
    pas = np.isin(E[:, col["category"]], (cats.index("passenger_gain"), cats.index("passenger_loss")))
    assert pas.sum() > 0 and pas.sum() == res["n_passengers_applied"] + res["n_trunk_applied"]
    assert res["n_trunk_applied"] == int((E[pas, col["node"]] == 0).sum()) > 0         # founder (trunk) events
    assert res["n_trunk_proposed"] == res["n_trunk_applied"] + res["n_trunk_rejected"]
    assert res["n_passengers_proposed"] == res["n_passengers_applied"] + res["n_passengers_rejected"]
    assert np.all(E[pas, col["driver"]] == 0)
    n_nodes = len(z["node_parent"])
    assert np.all((E[:, col["node"]] >= 0) & (E[:, col["node"]] < n_nodes))
    assert np.all(E[pas, col["gen"]] == z["node_birth_gen"][E[pas, col["node"]]])
    # every clone inherits the founder's trunk events: passenger columns >= the trunk gain / loss counts
    T = E[pas & (E[:, col["node"]] == 0)]
    tg = int((T[:, col["category"]] == cats.index("passenger_gain")).sum())
    tl = int((T[:, col["category"]] == cats.index("passenger_loss")).sum())
    assert np.all(z["event_counts"][:, 7] >= tg) and np.all(z["event_counts"][:, 8] >= tl)
    assert np.all(T[:, col["gen"]] == 0) and (z["block_cn"] >= 1).all()
    # without the trunk, growth is untouched by the overlay (own RNG): growth rows, popsize, fitness equal "off"
    _, _, zn, _ = e2e_runs["on_notrunk"]
    En = zn["events"]
    pn = np.isin(En[:, col["category"]], (cats.index("passenger_gain"), cats.index("passenger_loss")))
    assert np.array_equal(En[~pn], z_off["events"]) and np.array_equal(zn["popsize"], z_off["popsize"])
    assert np.array_equal(zn["fitness"], z_off["fitness"]) and np.array_equal(zn["n_altered_drivers"], z_off["n_altered_drivers"])
    assert np.array_equal(zn["event_counts"][:, :7], z_off["event_counts"][:, :7])
    assert not np.array_equal(zn["block_cn"], z_off["block_cn"]) and not np.any(En[pn, col["node"]] == 0)
    # genealogy arrays
    par, cn_, nc = z["node_parent"], z["clone_node"], z["node_clone"]
    assert par[0] == -1 and np.all(par[1:] < np.arange(1, n_nodes))
    assert np.array_equal(nc[cn_], np.arange(len(cn_)))
    for v in cn_:
        steps, x = 0, int(v)
        while x != -1:
            x, steps = int(par[x]), steps + 1
            assert steps <= n_nodes
    # results / parameters bookkeeping
    for k in ("cin_level", "pass_rate", "arm_rate_indep", "chromosomal_rate_indep", "focal_driver_rate_indep"):
        assert k in P
    ts = (_prior_seed(SIM_ID) + 1) % sgs.SEED_MODULUS
    assert P["pass_rate"] == 0.015 and res["overlay_seed"] == [ts, 7_000_003, 1] and res["trunk_seed"] == [ts, 7_000_003, 0]
    # n_altered_drivers_post: SISTEM's rule on the post-overlay block_cn
    dmask = np.asarray(P["driver_mask"], bool)
    ref = 2 ** (z["nWGD"].astype(np.int64) + 1)
    assert np.array_equal(z["n_altered_drivers_post"], (z["block_cn"][:, dmask] != ref[:, None]).sum(1).astype(np.int16))
    M = json.loads((root_on / "manifest.json").read_text())
    assert M["prior"]["draw_order"].endswith("cin_level, u_pass (last)") and M["passenger_overlay"]["enabled"] is True
    assert M["events"]["schema_version"] == 3 and M["prior"]["rho_cin_loadings"] == _loadings(0.5)   # v4.1: loadings


def test_e2e_grow_then_overlay_does_not_mutate_growth(tmp_path, monkeypatch):
    """Harness use: grow() once, several overlays. STRICT on: grow() checks driver_counts against the PRE-overlay
    block_cn; get_ploidy == mean block_cn (else 'ploidy_growth' would be needed)."""
    monkeypatch.setenv(sgs.STRICT_ENV_VAR, "1")
    th = _theta()
    chrom_lens, arm_ratios, _ = sgs.genome_grid()
    seed = (_prior_seed(SIM_ID) + 1) % sgs.SEED_MODULUS
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = Parameters(region_len=sgs.REGION_LEN, focal_driver_rate=th["focal_driver_rate"], length_mean=th["length_mean"],
                        max_distinct_driv_ratio=sgs.MAX_DISTINCT_DRIV_RATIO, min_detectable=TINY_MIN_DET,
                        arm_rate=th["arm_rate"], chromosomal_rate=th["chromosomal_rate"], out_dir=str(tmp_path),
                        log_path=os.devnull)
    sgs._P_TERM = th["p_term"]
    try:
        gr = sgs.grow(params, th["delta_dict"], chrom_lens, arm_ratios)
    finally:
        sgs._P_TERM = 0.0
    assert isinstance(gr, sgs.GrowthResult) and gr.driver_count_mismatches == 0
    assert gr.event_counts.shape[1] == 7 and np.array_equal(gr.ploidy, gr.block_cn.astype(np.float64).mean(1).astype(np.float32))
    assert np.array_equal(gr.ploidy, np.asarray([c.get_ploidy() for c in gr.clones], np.float32))
    before = gr.block_cn.copy()
    pp = sgs.passenger_params(0.015)
    r1 = sgs.run_overlay(gr, pp, np.random.default_rng(5))
    r2 = sgs.run_overlay(gr, pp, np.random.default_rng(5))
    assert np.array_equal(gr.block_cn, before) and np.array_equal(r1.block_cn, r2.block_cn) and r1.n_applied > 0
    inp = sgs.overlay_inputs(gr)
    assert np.array_equal(inp["driver_mask"], _flat(th["delta_dict"]) != 0)
    assert inp["cap_altered"] == gr.library.max_distinct_driv
    summary = sgs.pack_summary(gr, r1)                              # strict post-overlay asserts run here
    assert summary["event_counts"].shape[1] == 9 and summary["n_passengers_applied"] == r1.n_applied


def _grow(tmp_path, th, **kw):
    chrom_lens, arm_ratios, _ = sgs.genome_grid()
    seed = (_prior_seed(SIM_ID) + 1) % sgs.SEED_MODULUS
    torch.manual_seed(seed)
    np.random.seed(seed)
    params = Parameters(region_len=sgs.REGION_LEN, focal_driver_rate=th["focal_driver_rate"], length_mean=th["length_mean"],
                        max_distinct_driv_ratio=sgs.MAX_DISTINCT_DRIV_RATIO, min_detectable=TINY_MIN_DET,
                        arm_rate=th["arm_rate"], chromosomal_rate=th["chromosomal_rate"], out_dir=str(tmp_path),
                        log_path=os.devnull)
    sgs._P_TERM = th["p_term"]
    try:
        return sgs.grow(params, th["delta_dict"], chrom_lens, arm_ratios, **kw)
    finally:
        sgs._P_TERM = 0.0


def test_trunk_gens_zero_leaves_founder_and_growth_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv(sgs.STRICT_ENV_VAR, "1")
    th = _theta()
    base = _grow(tmp_path, th)                                      # v3-style growth, no pass_rate
    monkeypatch.setattr(sgs, "PASS_TRUNK_GENS", 0.0)
    g0 = _grow(tmp_path, th, pass_rate=0.015, trunk_seed=99)
    assert isinstance(base, sgs.GrowthResult) and isinstance(g0, sgs.GrowthResult)
    assert g0.n_trunk_proposed == 0 and g0.trunk_events.shape[0] == 0
    for k in ("block_cn", "popsize", "events", "event_counts", "fitness"):
        assert np.array_equal(getattr(base, k), getattr(g0, k)), k
    root = po.lineage_nodes(g0.clones)[0]
    assert root.birth_gen == 0 and not any(getattr(e, "passenger", False) for e in root.events)


def test_trunk_on_founder_is_inherited_viable_and_consistent(tmp_path, monkeypatch):
    monkeypatch.setenv(sgs.STRICT_ENV_VAR, "1")                    # founder driver_counts asserted before growth
    th = _theta()
    gr = _grow(tmp_path, th, pass_rate=0.015, trunk_seed=12345)    # Poisson mean 30 trunk events
    assert isinstance(gr, sgs.GrowthResult) and gr.driver_count_mismatches == 0
    assert gr.n_trunk_applied > 0 and gr.n_trunk_proposed == gr.n_trunk_applied + gr.n_trunk_rejected
    assert sum(gr.trunk_rejected_by.values()) == gr.n_trunk_rejected
    col = {c: i for i, c in enumerate(sgs.EVENT_COLUMNS)}
    T = gr.trunk_events
    assert np.all(T[:, col["node"]] == 0) and np.all(T[:, col["gen"]] == 0) and np.all(T[:, col["driver"]] == 0)
    assert set(T[:, col["category"]].tolist()) <= {sgs.EVENT_INDEX["passenger_gain"], sgs.EVENT_INDEX["passenger_loss"]}
    # the root node of the genealogy IS the founder that carries them; growth event rows exclude them
    root = po.lineage_nodes(gr.clones)[0]
    assert sum(getattr(e, "passenger", False) for e in root.events) == gr.n_trunk_applied
    assert not np.any(np.isin(gr.events[:, col["category"]], (7, 8)))
    assert (gr.block_cn >= 1).all()
    # every clone inherits: packed passenger columns >= trunk counts; trunk rows are in the packed events
    summ = sgs.pack_summary(gr, None)
    tg = int((T[:, col["category"]] == 7).sum())
    assert np.all(summ["event_counts"][:, 7] >= tg) and np.all(summ["event_counts"][:, 8] >= T.shape[0] - tg)
    assert summ["n_trunk_applied"] == gr.n_trunk_applied and summ["events"].shape[0] == gr.events.shape[0] + T.shape[0]
    # deterministic in trunk_seed
    gr2 = _grow(tmp_path, th, pass_rate=0.015, trunk_seed=12345)
    assert np.array_equal(gr.trunk_events, gr2.trunk_events) and np.array_equal(gr.block_cn, gr2.block_cn)


def test_passenger_seed_streams_never_collide_across_trials():
    """Trunk(t) must not replay overlay(t-1) (the additive offsets did: trial_seed + 7_000_002 of trial t was
    trial_seed + 7_000_003 of trial t-1)."""
    rank_seed = _prior_seed(SIM_ID)
    draws = {}
    for t in range(0, 6):
        trunk, overlay = sgs.passenger_seeds((rank_seed + t) % sgs.SEED_MODULUS)
        assert trunk == [rank_seed + t, sgs.OVERLAY_SEED_OFFSET, 0] and overlay[-1] == 1
        draws[t] = (np.random.default_rng(trunk).random(5), np.random.default_rng(overlay).random(5))
    for t in range(1, 6):
        assert not np.allclose(draws[t][0], draws[t - 1][1])          # trunk(t) vs overlay(t-1)
        assert not np.allclose(draws[t][0], draws[t][1])              # trunk(t) vs overlay(t)
    all_first = [x for t in draws for x in (draws[t][0][0], draws[t][1][0])]
    assert len(set(all_first)) == len(all_first)


def test_trunk_terminal_windows_anchor_at_homolog_end(tmp_path):
    """Terminal trunk events are anchored at the homolog's own end and never reach/cross its centromere; rows record
    the realised windows; the rejection rules use the overlay's order."""
    assert sgs.TRUNK_REJECT_RULES == ("zero", "cap", "altered", "ploidy", "absent")
    founder = sgs._dh_real_clone_factory(str(tmp_path))()
    lib = founder.library
    calls = {}
    amp, dele = founder.focal_amplification, founder.focal_deletion

    def rec(fn):
        def w(h, p0, p1, *a, **k):
            before = (len(h.seq), h.cent, p0, p1)
            fn(h, p0, p1, *a, **k)
            calls[id(founder.events[-1])] = before
        return w
    founder.focal_amplification, founder.focal_deletion = rec(amp), rec(dele)
    pp = sgs.passenger_params(0.04)                                  # Poisson mean 80: rearranged homologs occur
    pp.term_share = 0.6
    rows, n, rej = sgs.apply_trunk_passengers(founder, lib, pp, np.random.default_rng(4))
    kept = [e for e in founder.events if getattr(e, "passenger", False)]
    assert len(kept) == rows.shape[0] > 10 and n == rows.shape[0] + sum(rej.values())
    col = {c: i for i, c in enumerate(sgs.EVENT_COLUMNS)}
    n_term = 0
    for e, row in zip(kept, rows):
        L, cent, p0, p1 = calls[id(e)]
        assert row[col["terminal"]] == int(e.terminal) and row[col["n_blocks"]] == p1 - p0
        if e.terminal:
            n_term += 1
            room_p = (cent - 1) if cent is not None else L - 1
            room_q = (L - cent - 1) if cent is not None else L - 1
            assert (p0 == 0 and p1 <= room_p) or (p1 == L and p1 - p0 <= room_q), (L, cent, p0, p1)
    assert n_term > 0
    cn = sgs.count_region_CNs(founder, total=True)
    assert all(cn[c][r] == founder.attributes.driver_counts[c][r] for c in sgs.CHROM_ORDER for r in lib.drivers[c])


# ============================================================================ 5. checker
def test_checker_passes_on_v4_output(e2e_runs, tmp_path):
    root = e2e_runs["on"][0]
    out = tmp_path / "chk_v4"
    rc = chk.main(["--root", str(root), "--out-dir", str(out)])
    text = (out / "gene_pilot_checks.txt").read_text()
    assert rc == 0, text
    assert "prior_version 5" in text and "by schema {3: 1}" in text and "[INFO] C9" in text     # v4.1: version 5
    assert "with passenger categories (schema 3) 1" in text and "0-copy blocks: trials 0, clones 0" in text
    assert "WARNING: unknown prior_version" not in text


@pytest.mark.parametrize("root,schema", [(V2_ROOT, 1), (V3_ROOT, 2)])
def test_checker_still_reads_v2_v3(root, schema, tmp_path):
    if not (root / "manifest.json").is_file():
        pytest.skip(f"{root} not on this machine")
    out = tmp_path / "chk"
    chk.main(["--root", str(root), "--out-dir", str(out), "--max-sims", "1"])     # read-only
    text = (out / "gene_pilot_checks.txt").read_text()
    assert "check crashed" not in text and f"by schema {{{schema}: 25}}" in text
    assert "[INFO] C9" in text and "not applicable" in text and "0-copy blocks: trials 0, clones 0" in text


# ============================================================================ reload safety of D-F (runs last)
def test_reload_keeps_DF_pointing_at_sistem_check_viability():
    """A reload once made _ORIG_CHECK_VIABILITY capture our own D-F wrapper (infinite recursion in every growth
    after test_gene_simulator_v3's reload test); fixed 2026-10-23."""
    import importlib
    from sistem.selection.region_library import BaseRegionLibrary
    importlib.reload(sgs)
    assert sgs._ORIG_CHECK_VIABILITY.__module__ == "sistem.selection.region_library"
    assert BaseRegionLibrary.check_viability is sgs._nullisomy_check_viability
    assert sgs._nullisomy_check_viability._df_original is sgs._ORIG_CHECK_VIABILITY
