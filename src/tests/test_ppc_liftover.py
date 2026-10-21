"""Tests for ppc.builds / ppc.liftover and the genome-build plumbing in ppc.grid / ppc.real_loaders."""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from ppc.builds import GenomeBuildMismatch, HG19_CHR1_BP, HG38_CHR1_BP, normalise_build
from ppc.grid import Grid, load_grid
from ppc import liftover
from ppc.liftover import (CHAIN_DIR, CHAIN_FILES, assert_build, clear_caches, lift_segments,
                          segments_to_grid_build)
from ppc import real_loaders as rl

MANIFEST = "/Users/maryamkafi/Documents/PhD_Project/cancer/data_sim/gene_calib_f10/manifest.json"
L = 5_000_000


@pytest.fixture(scope="module")
def grid():
    if not os.path.exists(MANIFEST):
        pytest.skip("grid manifest not available")
    return load_grid(MANIFEST)


@pytest.fixture(scope="module")
def lifter():
    pytest.importorskip("cmmodule")
    for f in CHAIN_FILES.values():
        if not os.path.exists(os.path.join(CHAIN_DIR, f)):
            pytest.skip(f"chain file {f} missing")
    return lift_segments


def _tiny_grid(chr1_len: int, build: str) -> Grid:
    n = -(-chr1_len // L)
    return Grid(chrom_order=["chr1"], nregions={"chr1": n}, centromere_block={"chr1": n // 2},
                chrom_lens_bp={"chr1": chr1_len}, region_len=L, D=n, genome_build=build)


# --------------------------------------------------------------------------- builds / grid
def test_normalise_build():
    assert normalise_build("GRCh37/hg19 (SISTEM's table is misnamed hg38)") == "hg19"
    assert normalise_build("GRCh38") == "hg38"
    assert normalise_build("HG19") == "hg19" and normalise_build("custom") == "custom"
    with pytest.raises(ValueError, match="hg18"):
        normalise_build("hg18")
    with pytest.raises(ValueError, match="ambiguous"):           # names both builds, no 'misnamed'
        normalise_build("GRCh38 (lifted over from hg19)")
    with pytest.raises(ValueError, match="ambiguous"):           # 'misnamed' but not SISTEM's phrase
        normalise_build("GRCh38 (folder misnamed hg19)")


def test_grid_build(grid):
    assert grid.genome_build == "hg19"
    assert grid.chrom_lens_bp["chr1"] == HG19_CHR1_BP
    with pytest.raises(GenomeBuildMismatch):
        _tiny_grid(HG38_CHR1_BP, "hg19")
    with pytest.raises(GenomeBuildMismatch):
        _tiny_grid(20_000_000, "hg19")
    assert _tiny_grid(20_000_000, "custom").genome_build == "custom"
    assert _tiny_grid(HG38_CHR1_BP, "GRCh38").genome_build == "hg38"


def test_assert_build(grid):
    assert assert_build("GRCh37", grid) == "hg19"
    with pytest.raises(GenomeBuildMismatch) as e:
        assert_build("hg38", grid)
    assert "hg38" in str(e.value) and "hg19" in str(e.value)
    assert "segments_to_grid_build" in str(e.value)


def test_segments_to_grid_build_identity(grid):
    s = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [L], "cn": [2.0]})
    out, rep = segments_to_grid_build(s, "hg19", grid)
    assert out is s and rep is None
    with pytest.raises(GenomeBuildMismatch):
        segments_to_grid_build(s, "hg38", grid, allow_liftover=False)


# --------------------------------------------------------------------------- lift_segments
SIX = pd.DataFrame({
    "chrom": ["chr1", "chr1", "chr17", "chr13", "chr9", "chr18"],
    "start": [100_000_000, 240_000_000, 75_000_000, 100_000_000, 40_000_000, 70_000_000],
    "end": [105_000_000, 248_956_422, 83_257_441, 114_364_328, 48_000_000, 80_373_285],
    "cn": [2.0, 3.0, 4.0, 1.0, 2.0, 3.0],
    "sample": list("abcdef"),
}, index=[10, 11, 12, 13, 14, 15])


def test_lift_six_segments(lifter):
    out, rep = lifter(SIX, "hg38", "hg19")
    assert rep.n_in == 6 == rep.n_mapped + rep.n_unmapped
    assert out.loc[10, "start"] == 100_465_556 and abs(out.loc[10, "end"] - 105_542_622) <= 5_000
    assert abs(out.loc[11, "end"] - HG19_CHR1_BP) <= 20_000
    assert (out["chrom"] == SIX.loc[out.index, "chrom"]).all()
    assert (out["start"] < out["end"]).all() and (out["start"] >= 0).all()
    # the chr13 tail's first hit 10 kb inward is chrX:155,260,179; it must be rejected, so a kept
    # chr13 segment has to end inside hg19 chr13 (115,169,878 bp)
    if 13 in out.index:
        assert out.loc[13, "end"] <= 115_169_878
    for i in (13, 14, 15):        # chr13 tail, chr9 / chr18 new hg38 sequence
        assert i in out.index or (i in rep.unmapped.index and rep.unmapped.loc[i, "reason"]
                                  in {"start_unmapped", "end_unmapped", "inverted"})
    pd.testing.assert_frame_equal(out[["cn", "sample"]], SIX.loc[out.index, ["cn", "sample"]])
    assert list(out.index) == sorted(out.index)                          # input order kept
    assert rep.unmapped_reasons == rep.unmapped["reason"].value_counts().to_dict()
    assert "hg38->hg19" in str(rep)


def test_paralog_jump_rejected(lifter):
    """F1: pericentromeric chr9 paralog chains used to send both ends 17-24 Mb away (walk 0)."""
    s = pd.DataFrame({"chrom": ["chr9", "chr9"], "start": [61_986_410, 31_177_958],
                      "end": [64_794_547, 41_811_635], "cn": [2.0, 2.0]})
    out, rep = lifter(s, "hg38", "hg19")
    if 0 in out.index:      # used to come back as [45,122,562, 69,806,965), 24.7 Mb
        assert out.loc[0, "end"] - out.loc[0, "start"] < 6_000_000
    else:
        assert rep.unmapped.loc[0, "reason"] in {"start_unmapped", "end_unmapped", "length_changed"}
    # TCGA-02-0014's chr9 cn=2 segment used to stretch to hg19 65.7 Mb across the 45-60 Mb gap
    assert 1 not in out.index or out.loc[1, "end"] <= 50_000_000


def test_inverted_block_start_keeps_content(lifter, grid):
    """F2: a start inside the 1q21 '-' chain used to lift to the far edge (hg19 149,025,498)."""
    s = pd.DataFrame({"chrom": ["chr1"], "start": [143_184_588], "end": [248_946_422], "cn": [3.0]})
    out, rep = lifter(s, "hg38", "hg19")
    assert rep.n_mapped == 1 and out["start"].iloc[0] <= 145_000_000
    cn_int, _, _, cov = rl.bin_segments_to_blocks(out, grid)
    sl = grid.chrom_slice("chr1")
    assert (cov[sl][29:] > 0.9).all() and (cn_int[sl][29:] == 3).all()


ASCAT = "/Users/maryamkafi/Documents/PhD_Project/cancer/real_data/bulk_cnv/ascat"


def test_real_ascat_round_trip(lifter, grid):
    """30 real ASCAT samples hg19 -> hg38 -> hg19: no paralog stretching, few overlaps or drops."""
    if not os.path.exists(os.path.join(ASCAT, "segments.tar.gz")):
        pytest.skip("ASCAT files absent")
    names = list(rl._read_ascat_summary(ASCAT)["name"].iloc[:30])
    tabs = dict(rl._iter_tar_tables(os.path.join(ASCAT, "segments.tar.gz"), ".segments.txt", set(names)))
    names = [s for s in names if s in tabs]
    segs = pd.concat([rl.ascat_segments_to_internal(tabs[s]).assign(sample=s) for s in names],
                     ignore_index=True)
    segs["chrom"] = rl.normalise_chrom(segs["chrom"])
    segs = segs.dropna(subset=["chrom"])
    up, _ = lifter(segs, "hg19", "hg38")
    back, _ = lifter(up, "hg38", "hg19")
    assert len(back) > 0.98 * len(segs)                                         # (iii) < 2 % dropped
    ratio = (back["end"] - back["start"]) / (segs.loc[back.index, "end"] - segs.loc[back.index, "start"])
    assert ((ratio > 2) | (ratio < 0.5)).mean() < 0.01                          # (ii) < 1 % > 2x
    s = back.sort_values(["sample", "chrom", "start"], kind="stable")
    same = (s["sample"].to_numpy()[1:] == s["sample"].to_numpy()[:-1]) & \
           (s["chrom"].to_numpy()[1:] == s["chrom"].to_numpy()[:-1])
    over = s["end"].to_numpy()[:-1] - s["start"].to_numpy()[1:]
    # (i) <= 1.5 % of adjacent same-sample same-chromosome pairs overlap by > 1 Mb. With the
    # default 1 Mb local gate the first 30 samples sit at 0.27 %; over 20 windows of 30 ASCAT
    # samples the rate is 0.13-0.93 % (0.12-0.37 % with the gate off), so the limit sits above the
    # whole observed range rather than inside it. The overlaps come from CN breakpoints inside
    # inversions; see the liftover module docstring.
    assert (over[same] > 1_000_000).mean() <= 0.015
    meta = pd.DataFrame({"sample": names})
    a = rl.load_segment_table(segs, meta, grid, genome_build="hg19", source="native").block_cn_int
    b = rl.load_segment_table(back, meta, grid, genome_build="hg19", source="rt").block_cn_int
    k9 = [grid.offsets["chr9"] + r for r in (9, 10, 11, 12)]                  # chr9 45/50/55/60 Mb
    assert not ((a[:, k9] < 0) & (b[:, k9] >= 0)).any()                         # (iv) F1: never filled


def test_crossmap_does_not_touch_root_logger():
    """F4: CrossMap's import-time basicConfig and read_chain_file's logging.info must not leak.
    Run in a fresh interpreter: in this process cmmodule and the chains may already be loaded."""
    import subprocess
    import sys
    pytest.importorskip("cmmodule")
    if not os.path.exists(os.path.join(CHAIN_DIR, CHAIN_FILES[("hg38", "hg19")])):
        pytest.skip("chain file missing")
    code = (
        "import logging, pandas as pd\n"
        "r = logging.getLogger(); before = (r.level, list(r.handlers))\n"
        "from ppc.liftover import lift_segments\n"
        "lift_segments(pd.DataFrame({'chrom': ['chr1'], 'start': [100_000_000], 'end': [105_000_000]}),"
        " 'hg38', 'hg19')\n"
        "print((r.level, list(r.handlers)) == before)\n")
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "True"
    assert "Read the chain file" not in res.stderr


def test_lift_identity_same_build():
    out, rep = lift_segments(SIX, "hg19", "GRCh37")
    pd.testing.assert_frame_equal(out, SIX)
    assert rep.n_unmapped == 0 and rep.n_mapped == 6 and len(rep.unmapped) == 0


def test_lift_errors(tmp_path):
    with pytest.raises(ValueError):
        lift_segments(SIX, "hg19", "custom")
    with pytest.raises(FileNotFoundError, match="hg38ToHg19"):
        lift_segments(SIX, "hg38", "hg19", chain_dir=str(tmp_path))


def test_round_trip(lifter):
    rng = np.random.default_rng(0)
    chrom = [f"chr{c}" for c in rng.integers(1, 9, 20)]
    lens = {"chr1": 249_250_621, "chr2": 243_199_373, "chr3": 198_022_430, "chr4": 191_154_276,
            "chr5": 180_915_260, "chr6": 171_115_067, "chr7": 159_138_663, "chr8": 146_364_022}
    size = rng.integers(5_000_000, 40_000_001, 20)
    start = np.array([rng.integers(0, lens[c] - s) for c, s in zip(chrom, size)])
    s = pd.DataFrame({"chrom": chrom, "start": start, "end": start + size, "cn": 2.0})
    up, _ = lifter(s, "hg19", "hg38")
    back, _ = lifter(up, "hg38", "hg19")
    assert len(back) >= 15
    assert (np.abs(back["start"] - s.loc[back.index, "start"]) <= L).all()
    assert (np.abs(back["end"] - s.loc[back.index, "end"]) <= L).all()


def test_lifted_whole_chromosome_bins_cleanly(lifter, grid):
    s = pd.DataFrame({"chrom": ["chr2"], "start": [0], "end": [242_193_529], "cn": [3.0]})
    out, rep = lifter(s, "hg38", "hg19")
    assert rep.n_mapped == 1
    cn_int, _, _, cov = rl.bin_segments_to_blocks(out, grid)
    sl = grid.chrom_slice("chr2")
    c2, cv = cn_int[sl], cov[sl]
    assert (c2[cv >= 0.5] == 3).all()
    assert (cv[1:-1] > 0.9).all()


# --------------------------------------------------------------------------- loaders
def _two_sample_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    segs = pd.DataFrame({
        "sample": ["s1", "s1", "s2"], "chrom": ["chr1", "chr1", "chr3"],
        "start": [100_000_000, 110_000_000, 20_000_000], "end": [110_000_000, 120_000_000, 60_000_000],
        "cn": [2.0, 3.0, 4.0],
    })
    meta = pd.DataFrame({"purity": [0.5, 0.7], "cancer_type": ["X", "Y"]},
                        index=pd.Index(["s1", "s2"], name="sample"))
    return segs, meta


def test_load_segment_table_same_build(grid):
    segs, meta = _two_sample_table()
    coh = rl.load_segment_table(segs, meta, grid, genome_build="hg19", source="toy")
    assert coh.block_cn_int.shape == (2, grid.D) and coh.liftover is None
    assert coh.source_build == "hg19" and (coh.meta["source"] == "toy").all()
    assert list(coh.meta["sample"]) == ["s1", "s2"] and set(rl.META_COLUMNS) <= set(coh.meta.columns)
    o1, o3 = grid.offsets["chr1"], grid.offsets["chr3"]
    assert list(coh.block_cn_int[0, o1 + 20:o1 + 24]) == [2, 2, 3, 3]
    assert list(coh.block_cn_int[1, o3 + 4:o3 + 12]) == [4] * 8
    assert coh.subset([1]).source_build == "hg19"


def test_load_segment_table_hg38(grid):
    segs, meta = _two_sample_table()
    with pytest.raises(GenomeBuildMismatch):
        rl.load_segment_table(segs, meta, grid, genome_build="hg38", source="gdc", allow_liftover=False)
    pytest.importorskip("cmmodule")
    if not os.path.exists(os.path.join(CHAIN_DIR, CHAIN_FILES[("hg38", "hg19")])):
        pytest.skip("chain file missing")
    coh = rl.load_segment_table(segs, meta, grid, genome_build="GRCh38", source="gdc")
    assert coh.source_build == "hg38" and coh.liftover is not None
    assert coh.liftover.n_in == len(segs) and coh.liftover.to_build == "hg19"
    assert coh.block_cn_int.shape == (2, grid.D)


def test_cache_version_and_build_checks(grid, tmp_path):
    D = grid.D
    coh = rl.Cohort(np.zeros((1, D), np.int16), np.zeros((1, D), np.float32), np.zeros((1, D), bool),
                    np.ones((1, D), np.float32), pd.DataFrame({"sample": ["a"]}), grid)
    rl._save_cache(str(tmp_path), "toy", coh)
    assert os.path.basename(rl._cache_paths(str(tmp_path), "toy", grid)[0]) == f"toy_D{D}.npz"
    assert rl._load_cache(str(tmp_path), "toy", grid) is not None
    assert rl._load_cache(str(tmp_path), "toy", grid, "hg38") is None      # other file name, absent
    assert rl._cache_paths(str(tmp_path), "toy", grid, "hg38")[0].endswith(f"toy_D{D}_from_hg38.npz")
    # a version-1 cache (no build fields, bare meta pickle) is rejected
    npz, pkl = rl._cache_paths(str(tmp_path), "old", grid)
    np.savez(npz, block_cn_int=coh.block_cn_int, block_cn_mean=coh.block_cn_mean,
             breakpoint=coh.breakpoint, coverage=coh.coverage,
             grid_fp=rl._grid_fingerprint(grid), version=np.array(1))
    with open(pkl, "wb") as f:
        pickle.dump(coh.meta, f)
    assert rl._load_cache(str(tmp_path), "old", grid) is None
    # a v2 cache whose stored source build disagrees is rejected
    npz2, _ = rl._cache_paths(str(tmp_path), "toy", grid)
    z = dict(np.load(npz2))
    z["source_build"] = np.array("hg38")
    np.savez(npz2, **z)
    assert rl._load_cache(str(tmp_path), "toy", grid) is None


# --------------------------------------------------------------------------- round-3 regressions
HG38_LEN = {"chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189, "chr16": 90_338_345,
            "chr18": 80_373_285, "chr21": 46_709_983, "chr22": 50_818_468}


def test_walk_crosses_acrocentric_and_centromere_gaps(lifter, grid):
    """Whole acrocentric chromosomes and q-arms from the hg38 centromere used to be dropped as
    start_unmapped by a 2 Mb walk cap."""
    rows = [(c, 0, HG38_LEN[c]) for c in ("chr13", "chr14", "chr15", "chr21", "chr22")] + \
           [("chr16", 36_800_000, HG38_LEN["chr16"]), ("chr18", 18_500_000, HG38_LEN["chr18"])]
    s = pd.DataFrame(rows, columns=["chrom", "start", "end"]).assign(cn=3.0)
    out, rep = lifter(s, "hg38", "hg19")
    assert rep.n_mapped == len(s), rep.unmapped
    # hg19 chr16 35.28-46.39 Mb (16q11.2 heterochromatin) has no chain coverage at all: no lift
    # can cover block 8 (40-45 Mb), so it is excluded
    no_sequence = {("chr16", 8)}
    for i, row in out.iterrows():
        cn_int, _, _, cov = rl.bin_segments_to_blocks(out.loc[[i]], grid)
        sl = grid.chrom_slice(row["chrom"])
        q = [r for r in range(grid.centromere_block[row["chrom"]], grid.nregions[row["chrom"]])
             if (row["chrom"], r) not in no_sequence]
        low = [r for r in q if cov[sl][r] <= 0.9]
        assert all(r in (q[0], q[-1]) for r in low), (row["chrom"], low)      # one per end at most
        assert (cn_int[sl][[r for r in q if r not in low]] == 3).all()


def test_segment_inside_inverted_block(lifter):
    """Both endpoints inside the chr10 '-' block (hg38 46,840,017 + 660,916 bp): the start lifts
    above the end, so the envelope must be reordered rather than dropped or collapsed."""
    s = pd.DataFrame({"chrom": ["chr10", "chr10"], "start": [46_853_350, 46_853_350],
                      "end": [46_903_350, 46_858_350]})
    out, rep = lifter(s, "hg38", "hg19")
    assert rep.n_mapped == 2
    truth = [(48_836_011, 48_886_011), (48_881_011, 48_886_011)]
    for i, (ts, te) in enumerate(truth):
        assert abs(out.loc[i, "start"] - ts) <= 1_000 and abs(out.loc[i, "end"] - te) <= 1_000


def test_short_collapse_dropped(lifter):
    """hg38 chr21 [5.0, 6.85) Mb used to be kept as a 60 kb interval inside hg19 21q."""
    s = pd.DataFrame({"chrom": ["chr21"], "start": [5_000_000], "end": [6_850_000]})
    out, rep = lifter(s, "hg38", "hg19")
    assert len(out) == 0 and rep.unmapped_reasons == {"length_changed": 1}


def test_invalid_and_empty_coordinates(lifter):
    with pytest.raises(ValueError, match="1 segments"):
        lifter(pd.DataFrame({"chrom": ["chr1"], "start": [np.nan], "end": [5_000_000.0]}), "hg38", "hg19")
    with pytest.raises(ValueError, match="start < 0"):
        lifter(pd.DataFrame({"chrom": ["chr1"], "start": [-5], "end": [5_000_000]}), "hg38", "hg19")
    s = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1"], "start": [100_000_000, 100_000_000, 100_000_000],
                      "end": [100_000_000, 99_000_000, 105_000_000]})
    out, rep = lifter(s, "hg38", "hg19")
    assert list(out.index) == [2]
    assert rep.unmapped_reasons == {"empty": 2} and list(rep.unmapped["reason"]) == ["empty", "empty"]


def test_private_builder_normalises_build(grid, tmp_path):
    if not os.path.exists(os.path.join(ASCAT, "segments.tar.gz")):
        pytest.skip("ASCAT files absent")
    coh = rl._build_ascat(ASCAT, grid, limit=1, genome_build="GRCh37")
    assert coh.source_build == "hg19" and coh.liftover is None
    rl._save_cache(str(tmp_path), "ascat_tcga", coh)
    assert sorted(os.listdir(tmp_path)) == [f"ascat_tcga_D{grid.D}.npz", f"ascat_tcga_D{grid.D}_meta.pkl"]


def test_clear_caches(lifter):
    lifter(SIX, "hg38", "hg19")
    assert liftover._CHAINS and liftover._LATTICE
    clear_caches()
    assert not liftover._CHAINS and not liftover._LATTICE


# --------------------------------------------------------------------------- round-4 regressions
def test_walk_is_phase_independent(lifter):
    """Starts inside the hg38 chr21 p-arm gap probe the same genome lattice, so they land together
    (a 3 kb shift used to move the lifted start by 5.5 Mb)."""
    s = pd.DataFrame({"chrom": "chr21", "start": [100_000, 103_000, 105_000], "end": 46_000_000})
    out, rep = lifter(s, "hg38", "hg19")
    assert rep.n_mapped == 3 and out["start"].nunique() == 1


def test_bound_keeps_9q_and_16p_on_their_arm(lifter):
    """With a 20 Mb bound these crossed onto hg19 9p / across the chr16 centromere."""
    s = pd.DataFrame({"chrom": ["chr9", "chr16"], "start": [60_800_000, 5_000_000],
                      "end": [138_000_000, 34_700_000]})
    out, rep = lifter(s, "hg38", "hg19")
    assert rep.n_mapped == 2
    assert out.loc[0, "start"] > 50_000_000            # hg19 9q, not 9p (41.4-47.05 Mb)
    assert out.loc[1, "end"] < 35_300_000              # hg19 16p, before the 35.28 Mb gap


def test_local_gate_recovers_16p_by_default(lifter, grid):
    """hg38 16p ending at 36.5 Mb: endpoint-only (gate_window_bp=0) ends at hg19 32.1 Mb and leaves
    block 6 (30-35 Mb) under-covered; the default 1 Mb local gate recovers 16p11.2."""
    s = pd.DataFrame({"chrom": ["chr16"], "start": [20_000_000], "end": [36_500_000], "cn": [3.0]})
    out, rep = lifter(s, "hg38", "hg19")                                    # default behaviour
    cn_int, _, _, cov = rl.bin_segments_to_blocks(out, grid)
    assert rep.n_envelope_extended == 1
    assert cov[grid.offsets["chr16"] + 6] > 0.99 and cn_int[grid.offsets["chr16"] + 6] == 3
    out0, rep0 = lifter(s, "hg38", "hg19", gate_window_bp=0)                # contrast: gate off
    assert rep0.n_envelope_extended == 0
    assert rl.bin_segments_to_blocks(out0, grid)[3][grid.offsets["chr16"] + 6] < 0.5


def test_coordinate_validation_both_paths(lifter):
    bad = {"fractional": [1_000.5], "huge": [1e30], "nan": [np.nan]}
    for builds in (("hg19", "hg19"), ("hg38", "hg19")):
        for name, start in bad.items():
            with pytest.raises(ValueError):
                lifter(pd.DataFrame({"chrom": ["chr1"], "start": start, "end": [5_000_000.0]}), *builds)
    with pytest.raises(ValueError, match="int64"):
        lifter(pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [2**64]}), "hg19", "hg19")
    # integral floats are fine; start >= end is dropped as 'empty' on the same-build path too
    s = pd.DataFrame({"chrom": ["chr1"] * 2, "start": [100.0, 500.0], "end": [200.0, 400.0]}, index=[7, 8])
    out, rep = lifter(s, "hg19", "hg19")
    assert list(out.index) == [7] and out.loc[7, "start"] == 100.0
    assert rep.n_mapped == 1 and rep.unmapped_reasons == {"empty": 1}
    assert list(rep.unmapped["reason"]) == ["empty"]
