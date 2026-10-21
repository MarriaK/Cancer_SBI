"""Tests for ppc.decompose: whole-chromosome / arm / focal / broad decomposition on the real grid.

Run from ``src/``::

    ~/miniconda3/envs/cancer/bin/python -m pytest tests/test_ppc_decompose.py -q
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from ppc.decompose import COUNT_COLUMNS, decompose, decompose_cohort, segments, taylor_arm_calls
from ppc.grid import load_grid

MANIFEST = Path("/Users/maryamkafi/Documents/PhD_Project/cancer/data_sim/gene_calib_f10/manifest.json")
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="grid manifest not available")


@pytest.fixture(scope="module")
def grid():
    return load_grid(str(MANIFEST))


def arm(grid, name):
    return next(a for a in grid.arms if a.name == name)


def diploid(grid, base=2):
    return np.full(grid.D, base, dtype=np.int64)


def nonzero_counts(dec):
    return {k: v for k, v in dec.counts().items() if v}


def test_all_diploid_is_empty(grid):
    dec = decompose(diploid(grid), grid)
    assert nonzero_counts(dec) == {}
    assert dec.events == []
    assert set(dec.arm_calls) == {a.name for a in grid.called_arms} and len(dec.arm_calls) == 39
    assert all(v == 0 for v in dec.arm_calls.values())


def test_8q_gain_with_nested_focal_amplification(grid):
    x = diploid(grid)
    a = arm(grid, "8q")
    x[a.start:a.end] = 3
    x[a.start + 5] = 7
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"arm_gain": 1, "focal_gain": 1}
    assert [(e.level, e.arm, e.magnitude, e.n_blocks) for e in dec.events] == [
        ("arm", "8q", 1, 21), ("focal", "8q", 4, 1)]
    assert dec.focal_lengths == [1]
    assert dec.focal_gain_orig_baseline == 0 and not dec.events[1].orig_at_baseline
    assert dec.arm_calls["8q"] == 1 and dec.arm_levels["8q"] == 1


def test_chr8_whole_chrom_then_residual_arm(grid):
    x = diploid(grid)
    p, q = arm(grid, "8p"), arm(grid, "8q")
    x[p.start:p.end] = 3
    x[q.start:q.end] = 4
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"whole_chrom_gain": 1, "arm_gain": 1}
    assert [(e.level, e.arm, e.magnitude) for e in dec.events] == [("chrom", None, 1), ("arm", "8q", 1)]
    assert not dec.residual.any()


def test_whole_chrom_loss(grid):
    x = diploid(grid)
    x[grid.chrom_slice("chr17")] = 1
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"whole_chrom_loss": 1}
    assert dec.events[0].magnitude == -1


def test_opposite_signs_give_two_arm_events(grid):
    x = diploid(grid)
    p, q = arm(grid, "8p"), arm(grid, "8q")
    x[p.start:p.end] = 1
    x[q.start:q.end] = 3
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"arm_gain": 1, "arm_loss": 1}


def test_small_loss_in_1p_is_focal(grid):
    x = diploid(grid)
    a = arm(grid, "1p")
    x[a.start + 10:a.start + 13] = 1
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"focal_loss": 1}
    e = dec.events[0]
    assert (e.level, e.magnitude, e.n_blocks, e.start, e.end) == ("focal", -1, 3, a.start + 10, a.start + 13)


def test_13_of_25_blocks_on_1p_is_broad(grid):
    x = diploid(grid)
    a = arm(grid, "1p")
    assert a.n_blocks == 25
    x[a.start:a.start + 13] = 3
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"broad_gain": 1}
    assert dec.arm_calls["1p"] == 0


def test_12_of_25_blocks_on_1p_is_focal(grid):
    x = diploid(grid)
    a = arm(grid, "1p")
    x[a.start:a.start + 12] = 3
    assert nonzero_counts(decompose(x, grid)) == {"focal_gain": 1}


def test_17p_four_of_five_is_arm(grid):
    x = diploid(grid)
    a = arm(grid, "17p")
    assert a.n_blocks == 5
    x[a.start:a.start + 4] = 3
    dec = decompose(x, grid)
    # the 5th block becomes -1 after subtracting the arm level: 1 of 5 < 50% -> focal loss
    assert nonzero_counts(dec) == {"arm_gain": 1, "focal_loss": 1, "focal_loss_orig_baseline": 1}
    assert dec.focal_loss == 1 and dec.focal_loss_orig_baseline == 1
    assert dec.events[-1].orig_at_baseline
    assert taylor_arm_calls(x, grid)[0]["17p"] == 1


def test_17p_three_of_five_is_broad(grid):
    x = diploid(grid)
    a = arm(grid, "17p")
    x[a.start:a.start + 3] = 3
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"broad_gain": 1}
    assert dec.arm_calls["17p"] == 0


def test_13q_loss_is_acrocentric_arm_call(grid):
    x = diploid(grid)
    q, p = arm(grid, "13q"), arm(grid, "13p")
    x[q.start:q.end] = 1
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"arm_loss": 1, "acro_q_calls": 1}
    assert dec.whole_chrom_loss == 0 and dec.whole_chrom_gain == 0
    assert "13p" not in dec.arm_calls
    assert not dec.residual[p.start:p.end].any()
    assert not dec.residual.any()


def test_acrocentric_p_arm_is_scanned_for_runs(grid):
    x = diploid(grid)
    p = arm(grid, "13p")       # 4 blocks
    x[p.start] = 5
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"focal_gain": 1}
    assert dec.events[0].arm == "13p"


def test_arm_gain_with_one_block_at_baseline(grid):
    x = diploid(grid)
    a = arm(grid, "8q")
    x[a.start:a.end] = 3
    x[a.start + 7] = 2
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"arm_gain": 1, "focal_loss": 1, "focal_loss_orig_baseline": 1}
    assert dec.focal_loss == 1 and dec.focal_loss_orig_baseline == 1
    f = [e for e in dec.events if e.level == "focal"][0]
    assert (f.magnitude, f.n_blocks, f.start, f.orig_at_baseline) == (-1, 1, a.start + 7, True)


def test_masked_blocks_are_transparent(grid):
    x = diploid(grid)
    a = arm(grid, "1p")
    x[a.start + 5:a.start + 9] = 1     # 4-block loss ...
    x[a.start + 7] = -1                # ... with a masked block inside it
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"focal_loss": 1}
    e = dec.events[0]
    assert (e.n_blocks, e.start, e.end) == (3, a.start + 5, a.start + 9)
    assert dec.masked[a.start + 7] and dec.residual[a.start + 7] == 0
    runs = segments(x, grid)
    assert len(runs) == 1 and runs[0].n_blocks == 3 and runs[0].arm == "1p"


def test_masked_blocks_leave_denominators(grid):
    # 17p: 3 blocks gained + 2 masked -> 3/3 unmasked = 100% -> arm call
    x = diploid(grid)
    a = arm(grid, "17p")
    x[a.start:a.start + 3] = 3
    x[a.start + 3:a.end] = -1
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {"arm_gain": 1}
    assert dec.events[0].n_blocks == 3


def test_fully_masked_arm_is_not_called(grid):
    x = diploid(grid)
    a = arm(grid, "17p")
    x[a.start:a.end] = -1
    dec = decompose(x, grid)
    assert nonzero_counts(dec) == {} and dec.arm_calls["17p"] == 0


def test_baseline_4_flat_profile_is_empty(grid):
    dec = decompose(diploid(grid, 4), grid, baseline=4)
    assert nonzero_counts(dec) == {} and dec.events == []


def test_lower_median_level(grid):
    # 8q: 11 blocks at +1 and 10 at +2 -> median 1; 10 at +1, 11 at +2 -> median 2
    x = diploid(grid)
    a = arm(grid, "8q")
    x[a.start:a.end] = 3
    x[a.start:a.start + 10] = 4
    assert taylor_arm_calls(x, grid)[1]["8q"] == 1
    x[a.start:a.start + 11] = 4
    assert taylor_arm_calls(x, grid)[1]["8q"] == 2


def test_segments_by_value_split_nested_levels(grid):
    x = diploid(grid)
    a = arm(grid, "1q")
    x[a.start:a.start + 6] = 3
    x[a.start + 2:a.start + 4] = 5
    by_sign = segments(x, grid, by="sign")
    by_value = segments(x, grid, by="value")
    assert [(r.n_blocks, r.level) for r in by_sign] == [(6, 3)]
    assert [(r.n_blocks, r.level) for r in by_value] == [(2, 1), (2, 3), (2, 1)]
    dec = decompose(x, grid)
    assert dec.focal_lengths == [6] and dec.value_runs == [2, 2, 2]


def test_runs_do_not_cross_centromere(grid):
    x = diploid(grid)
    p, q = arm(grid, "1p"), arm(grid, "1q")
    x[p.end - 2:q.start + 2] = 3
    assert [r.arm for r in segments(x, grid)] == ["1p", "1q"]
    assert [r.arm for r in segments(x, grid, within="chrom")] == [None]
    assert nonzero_counts(decompose(x, grid)) == {"focal_gain": 2}


def _random_profiles(grid, n, seed):
    rng = np.random.default_rng(seed)
    X = np.full((n, grid.D), 2, dtype=np.int64)
    for i in range(n):
        for c in grid.chrom_order:              # whole chromosome / arm changes
            sl = grid.chrom_slice(c)
            u = rng.random()
            if u < 0.1:
                X[i, sl] += rng.choice([-1, 1, 2])
            elif u < 0.25:
                a = grid.arms[[k for k, aa in enumerate(grid.arms) if aa.chrom == c][rng.integers(2)]]
                X[i, a.start:a.end] += rng.choice([-1, 1])
        for _ in range(rng.integers(0, 15)):     # segments of random length
            s = rng.integers(grid.D)
            X[i, s:s + rng.integers(1, 30)] += rng.choice([-2, -1, 1, 3])
        X[i] = np.clip(X[i], 0, 12)
        X[i, rng.random(grid.D) < 0.03] = -1     # masked blocks
    return X


def test_random_profiles_reconstruct(grid):
    X = _random_profiles(grid, 300, seed=0)
    for x in X:
        dec = decompose(x, grid)
        c = dec.counts()
        assert all(v >= 0 for v in c.values())
        assert c["acro_q_calls"] <= c["arm_gain"] + c["arm_loss"]
        unm = x != -1
        d = np.where(unm, x - 2, 0)
        # chromosome + arm events, then the residual, rebuild the deviation exactly
        rebuilt = dec.residual.copy()
        for e in dec.events:
            if e.level == "chrom":
                rebuilt[grid.chrom_slice(e.chrom)] += e.magnitude
            elif e.level == "arm":
                a = arm(grid, e.arm)
                rebuilt[a.start:a.end] += e.magnitude
        assert np.array_equal(np.where(unm, rebuilt, 0), d)
        # every nonzero residual block lies in exactly one focal/broad run of its sign,
        # so blanking those runs leaves an all-zero residual on unmasked blocks
        left = dec.residual.copy()
        cover = np.zeros(grid.D, dtype=np.int64)
        for e in dec.events:
            if e.level in ("focal", "broad"):
                blk = slice(e.start, e.end)
                assert np.all(np.sign(left[blk][unm[blk]]) == np.sign(e.magnitude))
                assert np.abs(left[blk]).max() == abs(e.magnitude)
                assert unm[blk].sum() == e.n_blocks
                cover[blk] += 1
                left[blk] = 0
        assert not left[unm].any()
        assert cover.max(initial=0) <= 1
        assert c["focal_gain"] + c["focal_loss"] == len(dec.focal_lengths)
        for kind in ("focal_gain", "focal_loss", "broad_gain", "broad_loss"):
            assert 0 <= c[kind + "_orig_baseline"] <= c[kind]
        for e in dec.events:
            if e.level in ("focal", "broad"):
                blk = slice(e.start, e.end)
                assert e.orig_at_baseline == (not d[blk][unm[blk]].any())
        assert sum(dec.value_runs) == int(np.count_nonzero(dec.residual[unm]))


def test_cohort_frame(grid):
    X = _random_profiles(grid, 50, seed=1)
    X[0] = 4
    df = decompose_cohort(X, grid, baselines=[4] + [2] * 49)
    assert list(df.columns) == list(COUNT_COLUMNS) + ["ploidy", "fga", "n_breakpoints"]
    assert len(df) == 50
    assert df.iloc[0][list(COUNT_COLUMNS)].sum() == 0
    assert df.loc[0, "ploidy"] == 4 and df.loc[0, "fga"] == 0 and df.loc[0, "n_breakpoints"] == 0
    for i in (1, 2, 3):
        dec = decompose(X[i], grid)
        assert df.loc[i, "arm_gain"] == dec.arm_gain and df.loc[i, "focal_loss"] == dec.focal_loss


def test_breakpoints_skip_masked_and_chrom_boundaries(grid):
    x = diploid(grid)
    a = arm(grid, "1p")
    x[a.start + 3:a.start + 6] = 3      # 2 breakpoints
    x[a.start + 6] = -1                 # masked right after the gain: 3 -> (masked) -> 2 is one change
    x[grid.chrom_slice("chr2")] = 3     # chr1|chr2 boundary is not a breakpoint; chr2|chr3 neither
    df = decompose_cohort(x[None, :], grid)
    assert df.loc[0, "n_breakpoints"] == 2


def test_performance_2000_profiles(grid):
    X = _random_profiles(grid, 2000, seed=2)
    t0 = time.perf_counter()
    decompose_cohort(X, grid)
    assert time.perf_counter() - t0 < 10.0
