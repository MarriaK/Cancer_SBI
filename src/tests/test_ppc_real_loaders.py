"""Tests for ppc.real_loaders: segment binning on the 589-block grid and the real-data loaders."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ppc.grid import load_grid
from ppc.real_loaders import (
    ARM_NAMES, Cohort, ascat_segments_to_internal, bin_segments_to_blocks, load_ascat_tcga,
    load_pcawg, load_taylor_arm_calls, pcawg_segments_to_internal,
)

MANIFEST = "/Users/maryamkafi/Documents/PhD_Project/cancer/data_sim/gene_calib_f10/manifest.json"
BULK = "/Users/maryamkafi/Documents/PhD_Project/cancer/real_data/bulk_cnv"
L = 5_000_000


@pytest.fixture(scope="module")
def grid():
    if not os.path.exists(MANIFEST):
        pytest.skip("grid manifest not available")
    return load_grid(MANIFEST)


def segs(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "cn"]).astype(
        {"start": np.int64, "end": np.int64, "cn": float})


def test_exact_two_blocks(grid):
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(segs([("chr1", 0, 2 * L, 3)]), grid)
    o = grid.offsets["chr1"]
    assert cn_int.shape == (grid.D,) and cn_int.dtype == np.int16
    assert list(cn_int[o:o + 2]) == [3, 3]
    assert np.allclose(cov[o:o + 2], 1.0) and np.allclose(cn_mean[o:o + 2], 3.0)
    assert not bp.any()          # both segment ends sit on block edges
    # later blocks: masked
    assert cn_int[o + 2] == -1 and np.isnan(cn_mean[o + 2]) and cov[o + 2] == 0


def test_inner_boundary_and_tie_rule(grid):
    # block 1 = [5e6, 1e7): 2.5 Mb of CN 3 then 2.5 Mb of CN 4 -> tie; the earlier segment wins.
    s = segs([("chr1", 0, 7_500_000, 3), ("chr1", 7_500_000, 3 * L, 4)])
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(s, grid)
    o = grid.offsets["chr1"]
    assert list(bp[o:o + 3]) == [False, True, False]
    assert cn_int[o + 1] == 3
    assert cn_mean[o + 1] == pytest.approx(3.5)
    assert cov[o + 1] == pytest.approx(1.0)
    assert list(cn_int[o:o + 3]) == [3, 3, 4]
    # majority (not tie): 3 Mb of CN 4 beats 2 Mb of CN 3
    s = segs([("chr1", 0, 7_000_000, 3), ("chr1", 7_000_000, 3 * L, 4)])
    assert bin_segments_to_blocks(s, grid)[0][o + 1] == 4


def test_segment_ending_early_masks_later_blocks(grid):
    s = segs([("chr2", 0, 12_000_000, 2)])
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(s, grid)
    o = grid.offsets["chr2"]
    assert list(cn_int[o:o + 2]) == [2, 2]
    assert cov[o + 2] == pytest.approx(0.4) and cn_int[o + 2] == -1 and np.isnan(cn_mean[o + 2])
    assert bp[o + 2]             # the end at 12 Mb is inside block 2
    n = grid.nregions["chr2"]
    assert (cn_int[o + 3:o + n] == -1).all() and np.isnan(cn_mean[o + 3:o + n]).all()
    assert (cov[o + 3:o + n] == 0).all()


def test_truncated_last_block_full_coverage(grid):
    clen = grid.chrom_lens_bp["chr1"]
    s = segs([("1", 0, clen, 2)])          # '1' normalised to 'chr1'
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(s, grid)
    sl = grid.chrom_slice("chr1")
    last = sl.stop - 1
    lo, hi = grid.block_bounds_bp("chr1", grid.nregions["chr1"] - 1)
    assert hi - lo < L
    assert cov[last] == pytest.approx(1.0) and cn_int[last] == 2
    assert not bp[sl].any()     # the chromosome end is a block edge
    # a segment overrunning the hg19 length is clipped
    s2 = segs([("chr1", 0, clen + 10_000, 2)])
    assert bin_segments_to_blocks(s2, grid)[3][last] == pytest.approx(1.0)


def test_nan_segments_mask_and_x_dropped(grid):
    s = segs([("chr3", 0, L, np.nan), ("chr3", L, 2 * L, 5), ("chrX", 0, 1e8, 9), ("chrY", 0, 1e7, 9)])
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(s, grid)
    o = grid.offsets["chr3"]
    assert cn_int[o] == -1 and np.isnan(cn_mean[o]) and cov[o] == 0
    assert cn_int[o + 1] == 5
    assert (cn_int != -1).sum() == 1      # X and Y contributed nothing
    # majority segment NaN while coverage >= 0.5 -> cn_int masked, mean from the rest
    s = segs([("chr3", 0, 3_000_000, np.nan), ("chr3", 3_000_000, L, 2),
              ("chr3", L, 7_000_000, 2), ("chr3", 7_000_000, 2 * L, np.nan)])
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(s, grid)
    assert cov[o] == pytest.approx(0.4) and cn_int[o] == -1
    s = segs([("chr3", 0, 2_000_000, np.nan), ("chr3", 2_000_000, L, 2)])
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(s, grid)
    assert cov[o] == pytest.approx(0.6) and cn_int[o] == 2 and cn_mean[o] == pytest.approx(2.0)


def test_ascat_one_based_inclusive_conversion(grid):
    # ASCAT/PCAWG: [1, 10_000_000] 1-based inclusive == [0, 10_000_000) 0-based half-open
    raw = pd.DataFrame({"sample": ["s"] * 2, "chr": ["1", "1"], "startpos": [1, 10_000_001],
                        "endpos": [10_000_000, 15_000_000], "nMajor": [1, 2], "nMinor": [1, 1]})
    t = ascat_segments_to_internal(raw)
    assert list(t["start"]) == [0, 10_000_000] and list(t["end"]) == [10_000_000, 15_000_000]
    assert list(t["cn"]) == [2.0, 3.0]
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(t, grid)
    o = grid.offsets["chr1"]
    assert list(cn_int[o:o + 4]) == [2, 2, 3, -1] and not bp.any()
    assert np.allclose(cov[o:o + 3], 1.0)


def test_pcawg_cell_weighted_and_na(grid):
    raw = pd.DataFrame({
        "chromosome": ["1", "1", "X"], "start": [1, 5_000_001, 1], "end": [5_000_000, 10_000_000, 9],
        "total_cn": [2, "NA", 1], "major_cn": [1, 1, 1], "minor_cn": [1, 1, 0], "star": [3, 3, 3],
        "battenberg_nMaj1_A": [1, 2, 1], "battenberg_nMin1_A": [1, 1, 0],
        "battenberg_frac1_A": [1.0, 0.6, 1.0], "battenberg_nMaj2_A": [np.nan, 1, np.nan],
        "battenberg_nMin2_A": [np.nan, 1, np.nan], "battenberg_frac2_A": [np.nan, 0.4, np.nan],
    })
    t = pcawg_segments_to_internal(raw)
    assert t["start"].tolist()[:2] == [0, 5_000_000]
    assert t["cn_cw"].tolist()[:2] == pytest.approx([2.0, 0.6 * 3 + 0.4 * 2])
    assert np.isnan(t["cn"].iloc[1])
    cn_int, cn_mean, bp, cov = bin_segments_to_blocks(t, grid)
    o = grid.offsets["chr1"]
    assert cn_int[o] == 2 and cn_int[o + 1] == -1 and cov[o + 1] == 0


def test_arm_calls_cr_line_endings(tmp_path):
    header = ["Sample", "Type", "Aneuploidy Score"] + [
        {"13q": "13 (13q)", "14q": "14 (14q)", "15q": "15 (15q)", "21q": "21 (21q)",
         "22q": "22 (22q)"}.get(a, a) for a in ARM_NAMES]
    r1 = ["TCGA-18-3406-01", "LUSC", "8"] + ["-1", ""] + ["0"] * 37
    r2 = ["TCGA-18-3407-01", "LUSC", "25"] + ["1"] * 38 + [""]
    p = tmp_path / "arm.txt"
    p.write_bytes("\r".join("\t".join(r) for r in (header, r1, r2)).encode() + b"\r")
    df = load_taylor_arm_calls(str(p))
    assert list(df.index) == ["TCGA-18-3406-01", "TCGA-18-3407-01"]
    assert df.loc["TCGA-18-3406-01", "case_id"] == "TCGA-18-3406"
    assert df.loc["TCGA-18-3406-01", "1p"] == -1 and np.isnan(df.loc["TCGA-18-3406-01", "1q"])
    assert np.isnan(df.loc["TCGA-18-3407-01", "22q"]) and df.loc["TCGA-18-3407-01", "13q"] == 1
    assert df["aneuploidy_score"].tolist() == [8, 25]
    assert all(a in df.columns for a in ARM_NAMES)


def test_cohort_subset_and_ploidy(grid):
    D = grid.D
    mean = np.full((2, D), 2.0, np.float32)
    mean[1, :10] = np.nan
    mean[1, 10:] = 3.0
    coh = Cohort(np.full((2, D), 2, np.int16), mean, np.zeros((2, D), bool), np.ones((2, D), np.float32),
                 pd.DataFrame({"sample": ["a", "b"]}), grid)
    assert coh.ploidy_from_blocks() == pytest.approx([2.0, 3.0])
    sub = coh.subset(np.array([False, True]))
    assert len(sub) == 1 and sub.meta["sample"].tolist() == ["b"]


# --------------------------------------------------------------------------- real-file smokes
def _check_real(coh: Cohort, grid, label: str):
    n = len(coh)
    assert n > 0
    assert coh.block_cn_int.shape == (n, grid.D) and coh.block_cn_mean.shape == (n, grid.D)
    assert coh.breakpoint.shape == (n, grid.D) and coh.coverage.shape == (n, grid.D)
    assert coh.block_cn_int.min() >= -1 and coh.block_cn_int.max() <= 60
    unmasked = (coh.block_cn_int >= 0).mean(axis=1)
    agree = np.abs(coh.ploidy_from_blocks() - coh.meta["ploidy"].to_numpy()) <= 0.35
    print(f"\n{label}: n={n}, median unmasked={np.median(unmasked):.3f}, "
          f"min unmasked={unmasked.min():.3f}, ploidy within 0.35: {agree.mean():.3f}")
    assert (unmasked >= 0.95).mean() >= 0.9, f"unmasked fractions {np.round(unmasked, 3)}"
    assert agree.mean() >= 0.9


def test_real_ascat_smoke(grid):
    root = os.path.join(BULK, "ascat")
    if not os.path.exists(os.path.join(root, "segments.tar.gz")):
        pytest.skip("ASCAT files absent")
    coh = load_ascat_tcga(root, grid, wgd=None, qc=("Pass",), min_purity=None, limit=30)
    _check_real(coh, grid, "ASCAT (30 cases, QC pass)")


def test_real_pcawg_smoke(grid):
    root = os.path.join(BULK, "pcawg")
    if not os.path.exists(os.path.join(root, "consensus.20170119.somatic.cna.annotated.tar.gz")):
        pytest.skip("PCAWG files absent")
    coh = load_pcawg(root, grid, require_no_wgd=False, exclude_uncertain=False, min_purity=None,
                     one_per_donor=False, limit=30)
    assert coh.block_cn_cw is not None and coh.block_cn_cw.shape == (len(coh), grid.D)
    _check_real(coh, grid, "PCAWG (30 samples)")
