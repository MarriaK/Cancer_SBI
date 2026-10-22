"""Tests for ppc.stats: PPC summary statistics on the real grid with the step-1 structural mask.

Run from ``src/``::

    ~/miniconda3/envs/cancer/bin/python -m pytest tests/test_ppc_stats.py -q
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from ppc.decompose import decompose_cohort
from ppc.grid import load_grid
from ppc.stats import (CW_COLUMNS, SIM_DIAG_COLUMNS, STAT_COLUMNS, STAT_KNOBS, apply_structural_mask,
                       arm_frequencies, block_frequencies, cohort_stats, per_tumour_stats,
                       sim_observed_profile)

PROJECT = Path("/Users/maryamkafi/Documents/PhD_Project/cancer")
MANIFEST = PROJECT / "data_sim/gene_calib_f10/manifest.json"
MASK = PROJECT / "reports/step1/structural_mask.npy"
REAL_ROOT = PROJECT / "real_data/bulk_cnv"
F10_ROOT = PROJECT / "data_sim/gene_calib_f10"
pytestmark = pytest.mark.skipif(not (MANIFEST.exists() and MASK.exists()),
                                reason="grid manifest or structural mask not available")
NON_CW = [c for c in STAT_COLUMNS if c not in CW_COLUMNS]


@pytest.fixture(scope="module")
def grid():
    return load_grid(str(MANIFEST))


@pytest.fixture(scope="module")
def mask(grid):
    m = np.load(MASK)
    assert m.shape == (grid.D,) and m.dtype == bool
    return m


def arm(grid, name):
    return next(a for a in grid.arms if a.name == name)


def masked_diploid(grid, mask):
    return apply_structural_mask(np.full(grid.D, 2, dtype=np.int64), mask)


def test_columns_and_knobs():
    assert len(STAT_COLUMNS) == len(set(STAT_COLUMNS))
    assert set(STAT_KNOBS) == set(STAT_COLUMNS)
    assert STAT_KNOBS["ploidy"] == ("q", "A", "C") and STAT_KNOBS["n_focal_strict"] == ("F",)
    assert "frac_arm_end_bounded" in STAT_COLUMNS and "frac_telomere_bounded" in STAT_COLUMNS
    assert STAT_KNOBS["frac_telomere_bounded"] == STAT_KNOBS["frac_arm_end_bounded"]
    assert "subclonal_frac_nobp" in STAT_COLUMNS and "subclonal_frac_nobp" in CW_COLUMNS
    assert not set(SIM_DIAG_COLUMNS) & set(STAT_COLUMNS)


def test_apply_structural_mask(grid, mask):
    xi = apply_structural_mask(np.full(grid.D, 3, dtype=np.int16), mask)
    xf = apply_structural_mask(np.full(grid.D, 3.0, dtype=np.float32), mask)
    assert (xi[mask] == -1).all() and (xi[~mask] == 3).all()
    assert np.isnan(xf[mask]).all() and (xf[~mask] == 3.0).all()
    X = apply_structural_mask(np.full((2, grid.D), 2, dtype=np.int16), mask)
    assert (X[:, mask] == -1).all()


def test_diploid(grid, mask):
    s = per_tumour_stats(masked_diploid(grid, mask), grid)
    assert set(s) == set(STAT_COLUMNS) | set(SIM_DIAG_COLUMNS)
    assert s["fga"] == 0 and s["aneuploidy_score"] == 0 and s["wgii"] == 0
    assert s["cn_frac_2"] == 1.0 and s["ploidy"] == 2.0 and s["cn_max"] == 2
    assert s["n_breakpoints"] == 0 and s["n_focal_strict"] == 0
    # no altered run: segment statistics are 0 ("0 = no altered segment"), not NaN
    for k in ("seg_len_median_mb", "seg_len_p90_mb", "frac_arm_end_bounded", "frac_telomere_bounded"):
        assert s[k] == 0.0
    for k in CW_COLUMNS + SIM_DIAG_COLUMNS:
        assert np.isnan(s[k])


def test_quiet_profile_kept_in_joint_tests(grid, mask):
    """F1: quiet (unaltered) real tumours must not be dropped as NaN rows by standardise."""
    from ppc.ppc_tests import standardise
    quiet = masked_diploid(grid, mask)
    noisy = masked_diploid(grid, mask)
    a = arm(grid, "8q")
    noisy[a.start:a.end] = 3
    noisy[a.start + 2] = 5
    real = cohort_stats(np.stack([quiet, quiet, noisy]), grid)
    assert not real[NON_CW].isna().any().any()
    rng = np.random.default_rng(0)
    sim = real.iloc[[2] * 20].reset_index(drop=True) + rng.normal(0, 0.1, (20, len(STAT_COLUMNS)))
    cols = ["fga", "seg_len_median_mb", "seg_len_p90_mb", "frac_arm_end_bounded"]
    _, zb, info = standardise(sim, real, cols)
    assert info["n_real_rows_dropped"] == 0 and zb.shape == (3, 4)


def test_fully_masked_profile_is_nan(grid):
    s = per_tumour_stats(np.full(grid.D, -1, dtype=np.int64), grid)
    for k in ("seg_len_median_mb", "seg_len_p90_mb", "frac_arm_end_bounded", "frac_telomere_bounded",
              "fga", "ploidy"):
        assert np.isnan(s[k])
    assert s["n_breakpoints"] == 0


def test_8q_gain_plus_amplified_block(grid, mask):
    x = masked_diploid(grid, mask)
    a = arm(grid, "8q")
    assert not mask[a.start:a.end].any()
    x[a.start:a.end] = 3
    x[a.start + 5] = 7
    n_unm = int((~mask).sum())
    s = per_tumour_stats(x, grid)
    assert s["aneuploidy_score"] == 1 and s["n_arm_gain_calls"] == 1 and s["n_arm_loss_calls"] == 0
    assert s["n_arm"] == 1 and s["n_chrom"] == 0
    assert s["n_focal_strict"] == 1
    assert s["cn_frac_3"] == pytest.approx((a.n_blocks - 1) / n_unm)
    assert s["cn_frac_5plus"] == pytest.approx(1 / n_unm)
    assert s["cn_max"] == 7
    assert s["fga"] == pytest.approx(a.n_blocks / n_unm)
    assert s["n_breakpoints"] == 3     # 8p|8q at the centromere, 3|7, 7|3
    assert s["breakpoints_per_100mb"] == pytest.approx(3 / (n_unm * 5) * 100)


def test_arm_end_bounded_run(grid, mask):
    x = masked_diploid(grid, mask)
    a = arm(grid, "1q")
    x[a.end - 3:a.end] = 3
    s = per_tumour_stats(x, grid)
    assert s["frac_arm_end_bounded"] == 1.0
    assert s["seg_len_median_mb"] == 15.0 and s["seg_len_p90_mb"] == 15.0
    # the same run in the middle of the arm is interstitial
    y = masked_diploid(grid, mask)
    y[a.start + 10:a.start + 13] = 3
    assert per_tumour_stats(y, grid)["frac_arm_end_bounded"] == 0.0


def _unm_in_arm(grid, x, name):
    a = arm(grid, name)
    return a.start + np.flatnonzero(x[a.start:a.end] != -1)


def test_telomere_bounded_p_arm_start_counts(grid, mask):
    x = masked_diploid(grid, mask)
    u = _unm_in_arm(grid, x, "3p")
    x[u[:3]] = 3                                   # touches the p-arm telomere
    s = per_tumour_stats(x, grid)
    assert s["frac_telomere_bounded"] == 1.0 and s["frac_arm_end_bounded"] == 1.0


def test_telomere_bounded_centromere_side_does_not_count(grid, mask):
    for name in ("3p", "1q"):
        x = masked_diploid(grid, mask)
        u = _unm_in_arm(grid, x, name)
        x[u[-3:] if name.endswith("p") else u[:3]] = 1     # centromere end of the arm only
        s = per_tumour_stats(x, grid)
        assert s["frac_arm_end_bounded"] == 1.0 and s["frac_telomere_bounded"] == 0.0, name


def test_telomere_bounded_q_arm_chrom_end_and_mixture(grid, mask):
    x = masked_diploid(grid, mask)
    u = _unm_in_arm(grid, x, "1q")
    x[u[-3:]] = 3                                  # q-arm telomere = chromosome end
    assert per_tumour_stats(x, grid)["frac_telomere_bounded"] == 1.0
    v = _unm_in_arm(grid, x, "3p")
    x[v[-2:]] = 4                                  # plus a centromere-bounded run
    x[v[5:7]] = 1                                  # plus an interstitial run
    s = per_tumour_stats(x, grid)
    assert s["frac_telomere_bounded"] == pytest.approx(1 / 3)
    assert s["frac_arm_end_bounded"] == pytest.approx(2 / 3)


def test_telomere_bounded_masked_end_blocks(grid, mask):
    """Masked end blocks are skipped: the telomere is the first / last UNMASKED block of the arm."""
    x = masked_diploid(grid, mask)
    u = _unm_in_arm(grid, x, "3p")
    x[u[:2]] = -1                                  # mask the two telomeric blocks of 3p
    x[u[2:5]] = 3                                  # the run starts at the first unmasked block
    w = _unm_in_arm(grid, x, "5q")
    x[w[-1]] = -1                                  # same at the q end
    x[w[-4:-1]] = 1
    s = per_tumour_stats(x, grid)
    assert s["frac_telomere_bounded"] == 1.0 and s["frac_arm_end_bounded"] == 1.0
    x[u[2]] = 2                                    # now one unmasked block from the telomere
    assert per_tumour_stats(x, grid)["frac_telomere_bounded"] == pytest.approx(0.5)


def test_value_runs_split_nested_amplitudes(grid, mask):
    x = masked_diploid(grid, mask)
    a = arm(grid, "3p")
    x[a.start + 2:a.start + 6] = 3       # 4 blocks at 3, then 2 at 4: two value-runs
    x[a.start + 6:a.start + 8] = 4
    s = per_tumour_stats(x, grid)
    assert s["seg_len_median_mb"] == pytest.approx(15.0)    # median of (20, 10)
    assert s["seg_len_p90_mb"] == pytest.approx(19.0)       # linear interpolation
    assert s["frac_arm_end_bounded"] == 0.0


def test_wgii_one_whole_chromosome(grid, mask):
    x = masked_diploid(grid, mask)
    sl = grid.chrom_slice("chr1")
    x[sl] = 3
    s = per_tumour_stats(x, grid)
    assert s["wgii"] == pytest.approx(1 / 22)
    assert s["n_chrom"] == 1 and s["aneuploidy_score"] == 2


def test_subclonal_frac(grid, mask):
    x = masked_diploid(grid, mask)
    cw = apply_structural_mask(np.full(grid.D, 2.0), mask)
    idx = np.flatnonzero(~mask)[:10]
    cw[idx] = 2.5
    cw[np.flatnonzero(~mask)[10]] = 2.05     # within tolerance: clonal
    s = per_tumour_stats(x, grid, x_cw=cw)
    n_unm = int((~mask).sum())
    assert s["subclonal_frac"] == pytest.approx(10 / n_unm)
    assert s["ploidy_cw"] == pytest.approx((2.0 * (n_unm - 11) + 25 + 2.05) / n_unm)
    assert np.isnan(per_tumour_stats(x, grid)["subclonal_frac"])


def test_subclonal_frac_nobp(grid, mask):
    x = masked_diploid(grid, mask)
    u = _unm_in_arm(grid, x, "8q")
    x[u[5:10]] = 3                                 # breakpoint blocks u[4], u[5], u[9], u[10]
    cw = np.where(x == -1, np.nan, x.astype(float))
    cw[u[5]] = 2.5                                 # straddles the 2|3 boundary: not subclonality
    cw[u[7]] = 3.4                                 # interior non-integer block: subclonal
    s = per_tumour_stats(x, grid, x_cw=cw)
    n_ok = int((x != -1).sum())
    assert s["subclonal_frac"] == pytest.approx(2 / n_ok)
    assert s["subclonal_frac_nobp"] == pytest.approx(1 / (n_ok - 4))
    # a masked block between two equal-CN blocks is not a breakpoint
    y = masked_diploid(grid, mask)
    y[u[15]] = -1
    cy = np.where(y == -1, np.nan, 2.0)
    cy[u[16]] = 2.5
    n_y = int((y != -1).sum())
    assert per_tumour_stats(y, grid, x_cw=cy)["subclonal_frac_nobp"] == pytest.approx(1 / n_y)
    assert np.isnan(per_tumour_stats(x, grid)["subclonal_frac_nobp"])


def test_cn_max_driver_passenger(grid, mask):
    x = masked_diploid(grid, mask)
    u = np.flatnonzero(x != -1)
    x[u[3]] = 7
    x[u[100]] = 4
    dm = np.zeros(grid.D, dtype=bool)
    dm[[u[3], u[50]]] = True
    s = per_tumour_stats(x, grid, driver_mask=dm)
    assert s["cn_max_driver"] == 7 and s["cn_max_passenger"] == 4
    assert np.isnan(per_tumour_stats(x, grid)["cn_max_driver"])
    assert np.isnan(per_tumour_stats(x, grid)["cn_max_passenger"])
    # gating-table statistics do not depend on the driver mask
    s0 = per_tumour_stats(x, grid)
    assert {k: s[k] for k in STAT_COLUMNS} == pytest.approx({k: s0[k] for k in STAT_COLUMNS}, nan_ok=True)
    with pytest.raises(ValueError):
        per_tumour_stats(x, grid, driver_mask=dm[:-1])


def test_subclonal_frac_float32_boundary(grid, mask):
    """F6: |cw - round(cw)| = 0.1 exactly (as float32 1.9 / 2.1) is clonal in float32 and float64."""
    x = masked_diploid(grid, mask)
    idx = np.flatnonzero(~mask)
    for dt in (np.float64, np.float32):
        cw = np.full(grid.D, 2.0, dtype=dt)
        cw[idx[:5]] = dt(1.9)
        cw[idx[5:10]] = dt(2.1)
        cw[idx[10:12]] = dt(2.15)        # clearly subclonal
        cw = apply_structural_mask(cw, mask)
        s = per_tumour_stats(x, grid, x_cw=cw)
        assert s["subclonal_frac"] == pytest.approx(2 / idx.size), dt


def test_sim_observed_profile():
    cn = np.array([[2, 2], [3, 2], [3, 4]])
    assert sim_observed_profile(cn, np.array([60, 30, 10]))[0] == 2
    assert sim_observed_profile(cn, np.array([40, 35, 25]))[0] == 3
    # exact tie 50/50 -> lower CN; second block: 4 held by 10 cells only
    out = sim_observed_profile(cn, np.array([50, 0, 50]))
    assert out.tolist() == [2, 2]
    out = sim_observed_profile(np.array([[2, 5], [3, 4]]), np.array([1.0, 1.0]))
    assert out.tolist() == [2, 4]


def test_arm_frequencies(grid, mask):
    x1 = masked_diploid(grid, mask)
    a = arm(grid, "8q")
    x1[a.start:a.end] = 3
    x2 = masked_diploid(grid, mask)
    b = arm(grid, "17p")
    x2[b.start:b.end] = 1
    x2[a.start:a.end] = 4
    f = arm_frequencies(np.stack([x1, x2]), grid)
    assert len(f) == 39 and list(f.columns) == ["gain_freq", "loss_freq"]
    assert f.loc["8q", "gain_freq"] == 1.0 and f.loc["8q", "loss_freq"] == 0.0
    assert f.loc["17p", "loss_freq"] == 0.5 and f.loc["17p", "gain_freq"] == 0.0
    assert f.drop(["8q", "17p"]).to_numpy().sum() == 0
    assert "13p" not in f.index


def test_block_frequencies(grid, mask):
    x1 = masked_diploid(grid, mask)
    x1[100] = 3
    x2 = masked_diploid(grid, mask)
    x2[100] = 1
    x2[101] = 4
    f = block_frequencies(np.stack([x1, x2]), grid, mask=mask)
    assert len(f) == grid.D and list(f.columns) == ["chrom", "r", "gain_freq", "loss_freq"]
    assert f["gain_freq"][mask].isna().all() and f["loss_freq"][mask].isna().all()
    assert f.loc[100, "gain_freq"] == 0.5 and f.loc[100, "loss_freq"] == 0.5
    assert f.loc[101, "gain_freq"] == 0.5 and f.loc[101, "loss_freq"] == 0.0
    assert f.loc[0, "gain_freq"] == 0.0
    i = grid.offsets["chr8"] + 3
    assert f.loc[i, "chrom"] == "chr8" and f.loc[i, "r"] == 3
    # the -1 entries alone already mask a block when no mask is passed
    assert f["gain_freq"][mask].equals(block_frequencies(np.stack([x1, x2]), grid)["gain_freq"][mask])


def test_cohort_stats_matches_per_tumour_and_baselines(grid, mask):
    x = masked_diploid(grid, mask)
    a = arm(grid, "8q")
    x[a.start:a.end] = 5
    X = np.stack([x, x * 0 + np.where(x == -1, -1, 4)])
    df = cohort_stats(X, grid, baselines=[2, 4])
    assert list(df.columns) == list(STAT_COLUMNS) and len(df) == 2
    assert df.loc[0, "aneuploidy_score"] == 1
    assert df.loc[1, "fga"] == 0 and df.loc[1, "ploidy"] == 4.0
    ref = {k: v for k, v in per_tumour_stats(x, grid).items() if k in STAT_COLUMNS}
    assert df.loc[0].to_dict() == pytest.approx(ref, nan_ok=True)


@pytest.mark.skipif(not (REAL_ROOT / "ascat").exists(), reason="TCGA ASCAT data not available")
def test_smoke_tcga_ov(grid, mask):
    from ppc.real_loaders import load_ascat_tcga
    coh = load_ascat_tcga(str(REAL_ROOT / "ascat"), grid, cancer_types=["OV"])
    X = apply_structural_mask(coh.block_cn_int[:50], mask)
    df = cohort_stats(X, grid)
    assert len(df) == 50 and not df[NON_CW].isna().any().any()
    assert df["subclonal_frac"].isna().all() and df["subclonal_frac_nobp"].isna().all()
    assert (df["frac_telomere_bounded"] <= df["frac_arm_end_bounded"]).all()
    assert df["frac_telomere_bounded"].gt(0).any()
    ref = decompose_cohort(X, grid)
    np.testing.assert_array_equal(df["n_breakpoints"].to_numpy(), ref["n_breakpoints"].to_numpy())
    np.testing.assert_allclose(df["fga"].to_numpy(), ref["fga"].to_numpy())
    np.testing.assert_allclose(df["ploidy"].to_numpy(), ref["ploidy"].to_numpy())


@pytest.mark.skipif(not (F10_ROOT / "simulation_outputs").exists(), reason="F10 sims not available")
def test_smoke_f10_largest_clone(grid, mask):
    from ppc.sim_loaders import load_sim_cohort
    sim = load_sim_cohort(str(F10_ROOT), grid, sim_ids=None)
    X = apply_structural_mask(sim.block_cn_int[:50], mask)
    C = apply_structural_mask(sim.block_cn_mean[:50], mask)
    df = cohort_stats(X, grid, X_cw=C)
    assert len(df) == 50 and not df[NON_CW].isna().any().any()
    assert not df["subclonal_frac"].isna().any()
    assert (df["frac_telomere_bounded"] <= df["frac_arm_end_bounded"]).all()
    assert (df["subclonal_frac_nobp"].dropna() >= 0).all()


def test_timing_2000_profiles(grid, mask):
    rng = np.random.default_rng(0)
    X = np.full((2000, grid.D), 2, dtype=np.int64)
    for r in range(2000):
        for _ in range(6):
            a = grid.arms[rng.integers(len(grid.arms))]
            lo = rng.integers(a.start, a.end)
            hi = min(a.end, lo + rng.integers(1, a.n_blocks + 1))
            X[r, lo:hi] = rng.integers(0, 7)
    X = apply_structural_mask(X, mask)
    t0 = time.perf_counter()
    df = cohort_stats(X, grid)
    assert time.perf_counter() - t0 < 30
    assert len(df) == 2000
