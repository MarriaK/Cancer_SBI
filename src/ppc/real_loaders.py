"""Loaders that turn real bulk copy-number datasets into integer profiles on the 589-block hg19 grid.

All sources are GRCh37/hg19, matching ``Grid.chrom_lens_bp`` (SISTEM's chromosome table).
Only autosomes chr1..chr22 are kept; X, Y and anything else are dropped.

Internal segment convention
---------------------------
After normalisation every segment table has columns ``chrom`` ('chr1'..'chr22'), ``start``,
``end`` as **0-based half-open** bp ``[start, end)`` plus one or more float value columns
(NaN = unknown). Both ASCAT and PCAWG publish **1-based inclusive** coordinates
(``[startpos, endpos]``; in PCAWG consecutive segments satisfy ``start_next == end_prev + 1``),
so the conversion is ``start0 = start1 - 1``, ``end0 = end1``. Segment ends beyond the hg19
chromosome length are clipped to it.

Sources
-------
* **TCGA ASCAT v3** (Steele et al. 2022, VanLoo-lab/ascat ReleasedData), ``<root>/summary.tsv`` and
  ``<root>/segments.tar.gz`` (``segments/<name>.segments.txt``).
  Segment columns: ``sample`` (= summary ``name``), ``chr`` ('1'..'22', 'X'), ``startpos``,
  ``endpos`` (1-based inclusive, SNP positions, so small gaps between segments and large gaps at
  telomeres / centromeres / acrocentric short arms are uncovered), ``nMajor``, ``nMinor``
  (allele-specific integer CN of the clonal solution). Value used: ``cn = nMajor + nMinor``.
  Summary columns used: ``name`` (sample id, sometimes the case id plus a suffix like '-c'),
  ``patient`` (12-char case id, unique per row), ``cancer_type`` (TCGA code), ``purity``,
  ``ploidy``, ``WGD`` ('0', '1', '1+'), ``QC`` ('Pass', 'Likely normal').
* **PCAWG consensus** (Dentro et al. 2021), ``<root>/consensus.20170119.somatic.cna.annotated.tar.gz``
  with one ``<aliquot>.consensus.20170119.somatic.cna.annotated.txt`` per sample.
  Columns used: ``chromosome`` ('1'..'22', 'X', 'Y'), ``start``, ``end`` (1-based inclusive),
  ``total_cn`` (consensus clonal total CN; NA for ~2% of segments -> NaN, never zero),
  ``battenberg_nMaj1_A``, ``battenberg_nMin1_A``, ``battenberg_frac1_A`` (clonal/major subclone
  state and its cell fraction), ``battenberg_nMaj2_A``, ``battenberg_nMin2_A``,
  ``battenberg_frac2_A`` (second subclonal state, NA when the segment is clonal).
  Values: ``cn = total_cn`` and the cell-weighted
  ``cn_cw = frac1*(nMaj1+nMin1) + frac2*(nMaj2+nMin2)`` with a missing second state counted as 0;
  ``cn_cw`` is NaN where the Battenberg first state is NA.
  ``<root>/pp.txt``: ``samplename`` (= aliquot id), ``purity``, ``ploidy``, ``purity_conf_mad``,
  ``wgd_status`` ('wgd' / 'no_wgd'), ``wgd_uncertain`` (TRUE/FALSE).
  ``<root>/ann.txt``: ``histology_abbreviation``, ``tumour_aliquot_id`` (= samplename),
  ``icgc_donor_id``, ``num_subclones``, ``is_preferred``, ...
  One sample per donor: the first ``tumour_aliquot_id`` in sorted order among the samples that
  pass the other filters.
* **Taylor et al. 2018 arm calls** (``arm.txt``, CR line endings, tab separated): ``Sample``
  (e.g. TCGA-18-3406-01), ``Type``, ``Aneuploidy Score``, 39 arm columns with -1/0/1 or blank.
* **TCGA ABSOLUTE** (``abs_tables.txt``): ``array`` (sample id, e.g. TCGA-OR-A5J1-01), ``sample``
  (full aliquot barcode), ``call status``, ``purity``, ``ploidy``, ``Genome doublings``,
  ``Coverage for 80% power``, ``Cancer DNA fraction``, ``Subclonal genome fraction``, ``solution``.

Block summaries (see :func:`bin_segments_to_blocks`)
----------------------------------------------------
coverage, length-weighted mean CN, majority-state integer CN and a breakpoint flag per block.

Caching
-------
The full binned dataset (before any filter) is stored once per source and grid as
``<cache_dir>/<source>_D<D>.npz`` (arrays + a grid fingerprint) and
``<cache_dir>/<source>_D<D>_meta.pkl`` (pickled meta DataFrame). Filters are applied on load.
"""
from __future__ import annotations

import io
import os
import pickle
import re
import tarfile
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ppc.grid import Grid

DEFAULT_CACHE_DIR = "/Users/maryamkafi/Documents/PhD_Project/cancer/real_data/bulk_cnv/cache"
META_COLUMNS = ["sample", "donor", "cancer_type", "purity", "ploidy", "wgd", "qc", "source"]
ARM_NAMES = [
    "1p", "1q", "2p", "2q", "3p", "3q", "4p", "4q", "5p", "5q", "6p", "6q", "7p", "7q",
    "8p", "8q", "9p", "9q", "10p", "10q", "11p", "11q", "12p", "12q", "13q", "14q", "15q",
    "16p", "16q", "17p", "17q", "18p", "18q", "19p", "19q", "20p", "20q", "21q", "22q",
]
_CACHE_VERSION = 1


# --------------------------------------------------------------------------- normalisation
def normalise_chrom(series: pd.Series) -> pd.Series:
    """'1' / 'chr1' / 1 -> 'chr1'; anything that is not an autosome 1..22 -> NaN."""
    s = series.astype(str).str.strip().str.replace(r"^chr", "", regex=True)
    ok = s.str.fullmatch(r"([1-9]|1[0-9]|2[0-2])")
    return ("chr" + s).where(ok)


def one_based_to_half_open(segs: pd.DataFrame, start_col: str, end_col: str) -> pd.DataFrame:
    """Convert 1-based inclusive [start1, end1] to 0-based half-open [start1 - 1, end1)."""
    out = segs.copy()
    out["start"] = out[start_col].astype(np.int64) - 1
    out["end"] = out[end_col].astype(np.int64)
    return out


# --------------------------------------------------------------------------- binning
def _bin_many(chrom: np.ndarray, start: np.ndarray, end: np.ndarray, values: np.ndarray,
              sid: np.ndarray, n: int, grid: Grid) -> dict[str, np.ndarray]:
    """Bin segments of ``n`` samples at once.

    ``values`` is [n_seg, k] (k value columns); the first column drives ``cn_int``,
    ``coverage`` and ``breakpoint``; every column gets its own length-weighted mean (with its own
    non-NaN coverage). Segments must be ordered by (sid, chrom, start) for the tie rule.
    """
    D, L = grid.D, grid.region_len
    cidx = {c: i for i, c in enumerate(grid.chrom_order)}
    offs = np.array([grid.offsets[c] for c in grid.chrom_order], dtype=np.int64)
    nreg = np.array([grid.nregions[c] for c in grid.chrom_order], dtype=np.int64)
    clen = np.array([grid.chrom_lens_bp[c] for c in grid.chrom_order], dtype=np.int64)
    # flat block bounds
    bstart = np.empty(D, dtype=np.int64)
    bend = np.empty(D, dtype=np.int64)
    for i, c in enumerate(grid.chrom_order):
        r = np.arange(nreg[i])
        bstart[offs[i] + r] = r * L
        bend[offs[i] + r] = np.minimum((r + 1) * L, clen[i])
    blen = (bend - bstart).astype(np.float64)

    ci = np.array([cidx.get(c, -1) for c in chrom], dtype=np.int64) if len(chrom) else np.zeros(0, np.int64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    k = values.shape[1]
    keep = ci >= 0
    ci, start, end, values, sid = ci[keep], np.asarray(start, np.int64)[keep], \
        np.asarray(end, np.int64)[keep], values[keep], np.asarray(sid, np.int64)[keep]
    end = np.minimum(end, clen[ci])
    start = np.maximum(start, 0)
    keep = end > start
    ci, start, end, values, sid = ci[keep], start[keep], end[keep], values[keep], sid[keep]
    order = np.arange(len(ci))

    # expand to (segment, block) pairs
    r0 = start // L
    r1 = np.minimum((end - 1) // L, nreg[ci] - 1)
    cnt = r1 - r0 + 1
    seg = np.repeat(order, cnt)
    within = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    blk = offs[ci[seg]] + r0[seg] + within
    ov = (np.minimum(end[seg], bend[blk]) - np.maximum(start[seg], bstart[blk])).astype(np.float64)
    key = sid[seg] * D + blk
    v = values[seg]

    out: dict[str, np.ndarray] = {}
    cov_by_col = []
    for j in range(k):
        good = ~np.isnan(v[:, j])
        cov = np.bincount(key[good], weights=ov[good], minlength=n * D).reshape(n, D)
        wsum = np.bincount(key[good], weights=ov[good] * v[good, j], minlength=n * D).reshape(n, D)
        cov_frac = np.minimum(cov / _blk_len_tile(blen, n), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cov_frac >= 0.5, wsum / np.where(cov > 0, cov, 1.0), np.nan)
        out[f"mean{j}"] = mean.astype(np.float32)
        cov_by_col.append(cov_frac.astype(np.float32))
    out["coverage"] = cov_by_col[0]

    # majority state: largest overlap per block; ties -> earliest segment (lowest start)
    cn_int = np.full(n * D, -1, dtype=np.int16)
    if len(key):
        o = np.lexsort((seg, -ov, key))
        ks = key[o]
        first = np.ones(len(ks), dtype=bool)
        first[1:] = ks[1:] != ks[:-1]
        winners = o[first]
        wv = v[winners, 0]
        wk = key[winners]
        okv = ~np.isnan(wv)
        cn_int[wk[okv]] = np.rint(wv[okv]).astype(np.int16)
    cn_int = cn_int.reshape(n, D)
    cn_int[out["coverage"] < 0.5] = -1
    out["cn_int"] = cn_int

    # breakpoints: start or end of a non-NaN segment strictly inside a block
    good = ~np.isnan(values[:, 0])
    pos = np.concatenate([start[good], end[good]])
    pc = np.concatenate([ci[good], ci[good]])
    ps = np.concatenate([sid[good], sid[good]])
    r = pos // L
    inside = (pos % L != 0) & (pos < clen[pc]) & (r < nreg[pc])
    bp = np.zeros(n * D, dtype=bool)
    bp[ps[inside] * D + offs[pc[inside]] + r[inside]] = True
    out["breakpoint"] = bp.reshape(n, D)
    return out


def _blk_len_tile(blen: np.ndarray, n: int) -> np.ndarray:
    return np.broadcast_to(blen, (n, blen.shape[0]))


def bin_segments_to_blocks(segs: pd.DataFrame, grid: Grid, value_col: str = "cn"
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin one sample's segments onto the grid.

    ``segs``: columns ``chrom`` (any of '1'/'chr1'; non-autosomes are dropped), ``start``, ``end``
    (0-based half-open bp) and ``value_col`` (float, NaN = unknown).

    Returns ``(cn_int int16[D], cn_mean float32[D], breakpoint bool[D], coverage float32[D])``:

    * coverage: fraction of the block's bp covered by segments with a non-NaN value;
    * cn_mean: length-weighted mean value over those segments, NaN if coverage < 0.5;
    * cn_int: value of the segment overlapping the largest share of the block (NaN segments
      included as candidates), rounded to the nearest int (``np.rint``); -1 if coverage < 0.5 or
      that segment's value is NaN. **Tie rule:** equal overlaps -> the earlier segment (lower
      start) wins;
    * breakpoint: a start or an end of a non-NaN segment lies strictly inside the block (not at
      its own start or end edge; the chromosome end counts as an edge).
    """
    s = segs.copy()
    s["chrom"] = normalise_chrom(s["chrom"])
    s = s.dropna(subset=["chrom"])
    s = s.assign(_c=s["chrom"].map({c: i for i, c in enumerate(grid.chrom_order)}))
    s = s.sort_values(["_c", "start"], kind="stable")
    r = _bin_many(s["chrom"].to_numpy(), s["start"].to_numpy(), s["end"].to_numpy(),
                  s[value_col].to_numpy(dtype=np.float64), np.zeros(len(s), np.int64), 1, grid)
    return r["cn_int"][0], r["mean0"][0], r["breakpoint"][0], r["coverage"][0]


# --------------------------------------------------------------------------- cohort
@dataclass
class Cohort:
    """Binned real copy-number profiles.

    ``block_cn_int`` int16 [n, D] (-1 masked), ``block_cn_mean`` float32 [n, D] (NaN masked),
    ``breakpoint`` bool [n, D], ``coverage`` float32 [n, D], optional ``block_cn_cw`` float32
    [n, D] (PCAWG cell-weighted mean CN), ``meta`` with at least ``META_COLUMNS``.
    """
    block_cn_int: np.ndarray
    block_cn_mean: np.ndarray
    breakpoint: np.ndarray
    coverage: np.ndarray
    meta: pd.DataFrame
    grid: Grid
    block_cn_cw: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, mask: np.ndarray | Sequence[bool] | Sequence[int]) -> "Cohort":
        m = np.asarray(mask)
        if m.dtype != bool:
            idx = m.astype(np.int64)
        else:
            idx = np.flatnonzero(m)
        return Cohort(
            block_cn_int=self.block_cn_int[idx], block_cn_mean=self.block_cn_mean[idx],
            breakpoint=self.breakpoint[idx], coverage=self.coverage[idx],
            meta=self.meta.iloc[idx].reset_index(drop=True), grid=self.grid,
            block_cn_cw=None if self.block_cn_cw is None else self.block_cn_cw[idx],
        )

    def ploidy_from_blocks(self) -> np.ndarray:
        """Mean of the unmasked ``block_cn_mean`` per sample (unweighted over blocks)."""
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.block_cn_mean.astype(np.float64), axis=1)


def _grid_fingerprint(grid: Grid) -> np.ndarray:
    vals = [grid.D, grid.region_len] + [grid.chrom_lens_bp[c] for c in grid.chrom_order] \
        + [grid.nregions[c] for c in grid.chrom_order]
    return np.array(vals, dtype=np.int64)


def _cache_paths(cache_dir: str, source: str, grid: Grid) -> tuple[str, str]:
    base = os.path.join(cache_dir, f"{source}_D{grid.D}")
    return base + ".npz", base + "_meta.pkl"


def _load_cache(cache_dir: str, source: str, grid: Grid) -> Optional[Cohort]:
    npz, pkl = _cache_paths(cache_dir, source, grid)
    if not (os.path.exists(npz) and os.path.exists(pkl)):
        return None
    z = np.load(npz)
    if int(z["version"]) != _CACHE_VERSION or not np.array_equal(z["grid_fp"], _grid_fingerprint(grid)):
        return None
    with open(pkl, "rb") as f:
        meta = pickle.load(f)
    return Cohort(block_cn_int=z["block_cn_int"], block_cn_mean=z["block_cn_mean"],
                  breakpoint=z["breakpoint"], coverage=z["coverage"], meta=meta, grid=grid,
                  block_cn_cw=z["block_cn_cw"] if "block_cn_cw" in z.files else None)


def _save_cache(cache_dir: str, source: str, cohort: Cohort) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    npz, pkl = _cache_paths(cache_dir, source, cohort.grid)
    arrs = dict(block_cn_int=cohort.block_cn_int, block_cn_mean=cohort.block_cn_mean,
                breakpoint=cohort.breakpoint, coverage=cohort.coverage,
                grid_fp=_grid_fingerprint(cohort.grid), version=np.array(_CACHE_VERSION))
    if cohort.block_cn_cw is not None:
        arrs["block_cn_cw"] = cohort.block_cn_cw
    tmp = npz + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, npz)
    with open(pkl + ".tmp", "wb") as f:
        pickle.dump(cohort.meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(pkl + ".tmp", pkl)


def _iter_tar_tables(tar_path: str, suffix: str, names: Optional[set[str]] = None
                     ) -> Iterable[tuple[str, pd.DataFrame]]:
    """Yield (file id, DataFrame) for members ending in ``suffix``, read in place from the tarball."""
    with tarfile.open(tar_path, "r:*") as tf:
        for mem in tf:
            if not mem.isfile() or not mem.name.endswith(suffix):
                continue
            fid = os.path.basename(mem.name)[: -len(suffix)]
            if names is not None and fid not in names:
                continue
            fh = tf.extractfile(mem)
            if fh is None:
                continue
            yield fid, pd.read_csv(io.BytesIO(fh.read()), sep="\t", low_memory=False)


def _bin_tables(tables: list[pd.DataFrame], grid: Grid, value_cols: list[str]) -> dict[str, np.ndarray]:
    """``tables`` already normalised (chrom, start, end, value cols); one per sample, in order."""
    n = len(tables)
    if n == 0:
        D = grid.D
        z = {"cn_int": np.zeros((0, D), np.int16), "coverage": np.zeros((0, D), np.float32),
             "breakpoint": np.zeros((0, D), bool)}
        for j in range(len(value_cols)):
            z[f"mean{j}"] = np.zeros((0, D), np.float32)
        return z
    cat = pd.concat([t.assign(_sid=i) for i, t in enumerate(tables)], ignore_index=True)
    cat["chrom"] = normalise_chrom(cat["chrom"])
    cat = cat.dropna(subset=["chrom"])
    cat["_c"] = cat["chrom"].map({c: i for i, c in enumerate(grid.chrom_order)})
    cat = cat.sort_values(["_sid", "_c", "start"], kind="stable")
    return _bin_many(cat["chrom"].to_numpy(), cat["start"].to_numpy(), cat["end"].to_numpy(),
                     cat[value_cols].to_numpy(dtype=np.float64), cat["_sid"].to_numpy(), n, grid)


def _finish_meta(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.reset_index(drop=True)
    for c in ("wgd", "qc"):
        meta[c] = meta[c].astype(str)
    return meta


# --------------------------------------------------------------------------- ASCAT
def _read_ascat_summary(root: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(root, "summary.tsv"), sep="\t", dtype={"WGD": str, "QC": str})


def ascat_segments_to_internal(raw: pd.DataFrame) -> pd.DataFrame:
    """ASCAT ``chr, startpos, endpos`` (1-based inclusive), ``nMajor + nMinor`` -> internal table."""
    t = one_based_to_half_open(raw, "startpos", "endpos")
    t["chrom"] = raw["chr"].astype(str)
    t["cn"] = raw["nMajor"].astype(float) + raw["nMinor"].astype(float)
    return t[["chrom", "start", "end", "cn"]]


def _build_ascat(root: str, grid: Grid, limit: Optional[int] = None) -> Cohort:
    summ = _read_ascat_summary(root)
    if limit is not None:
        summ = summ.iloc[:limit]
    want = set(summ["name"])
    tabs: dict[str, pd.DataFrame] = {}
    for fid, raw in _iter_tar_tables(os.path.join(root, "segments.tar.gz"), ".segments.txt", want):
        tabs[fid] = ascat_segments_to_internal(raw)
        if len(tabs) == len(want):
            break
    summ = summ[summ["name"].isin(tabs)].reset_index(drop=True)
    r = _bin_tables([tabs[nm] for nm in summ["name"]], grid, ["cn"])
    meta = pd.DataFrame({
        "sample": summ["name"], "donor": summ["patient"], "cancer_type": summ["cancer_type"],
        "purity": summ["purity"].astype(float), "ploidy": summ["ploidy"].astype(float),
        "wgd": summ["WGD"], "qc": summ["QC"], "source": "ascat_tcga",
        "sex": summ["sex"], "goodness_of_fit": summ["goodness_of_fit"],
    })
    return Cohort(r["cn_int"], r["mean0"], r["breakpoint"], r["coverage"], _finish_meta(meta), grid)


def load_ascat_tcga(root: str, grid: Grid, cancer_types: Optional[Sequence[str]] = None,
                    wgd: Optional[Sequence[str]] = ("0",), qc: Optional[Sequence[str]] = ("Pass",),
                    min_purity: Optional[float] = 0.3, use_cache: bool = True,
                    cache_dir: str = DEFAULT_CACHE_DIR, limit: Optional[int] = None) -> Cohort:
    """TCGA ASCAT v3 cohort, ``cn = nMajor + nMinor``; one row per case (``patient`` is unique).

    Filters (``None`` disables one): ``cancer_types`` (TCGA codes), ``wgd`` (allowed values of
    the summary ``WGD`` column: '0', '1', '1+'), ``qc`` (allowed ``QC`` values), ``purity >= min_purity``.
    ``limit`` bins only the first ``limit`` summary rows and bypasses the cache (for smoke tests).
    """
    coh = None
    if limit is None and use_cache:
        coh = _load_cache(cache_dir, "ascat_tcga", grid)
    if coh is None:
        coh = _build_ascat(root, grid, limit)
        if limit is None and use_cache:
            _save_cache(cache_dir, "ascat_tcga", coh)
    m = coh.meta
    keep = np.ones(len(m), dtype=bool)
    if cancer_types is not None:
        keep &= m["cancer_type"].isin(list(cancer_types)).to_numpy()
    if wgd is not None:
        keep &= m["wgd"].isin([str(w) for w in wgd]).to_numpy()
    if qc is not None:
        keep &= m["qc"].isin(list(qc)).to_numpy()
    if min_purity is not None:
        keep &= (m["purity"] >= min_purity).to_numpy()
    return coh.subset(keep)


# --------------------------------------------------------------------------- PCAWG
_PCAWG_SUFFIX = ".consensus.20170119.somatic.cna.annotated.txt"
_PCAWG_TAR = "consensus.20170119.somatic.cna.annotated.tar.gz"


def pcawg_segments_to_internal(raw: pd.DataFrame) -> pd.DataFrame:
    """PCAWG consensus segment table (1-based inclusive) -> internal table with ``cn``, ``cn_cw``."""
    t = one_based_to_half_open(raw, "start", "end")
    t["chrom"] = raw["chromosome"].astype(str)
    t["cn"] = pd.to_numeric(raw["total_cn"], errors="coerce").astype(float)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce").astype(float)  # noqa: E731
    s1 = num("battenberg_frac1_A") * (num("battenberg_nMaj1_A") + num("battenberg_nMin1_A"))
    s2 = (num("battenberg_frac2_A") * (num("battenberg_nMaj2_A") + num("battenberg_nMin2_A"))).fillna(0.0)
    t["cn_cw"] = s1 + s2
    return t[["chrom", "start", "end", "cn", "cn_cw"]]


def _read_pcawg_meta(root: str) -> pd.DataFrame:
    pp = pd.read_csv(os.path.join(root, "pp.txt"), sep="\t")
    ann = pd.read_csv(os.path.join(root, "ann.txt"), sep="\t")
    ann = ann[["tumour_aliquot_id", "icgc_donor_id", "histology_abbreviation", "num_subclones",
               "is_preferred"]]
    m = pp.merge(ann, left_on="samplename", right_on="tumour_aliquot_id", how="left")
    unc = m["wgd_uncertain"].astype(str).str.upper().eq("TRUE")
    return pd.DataFrame({
        "sample": m["samplename"], "donor": m["icgc_donor_id"],
        "cancer_type": m["histology_abbreviation"], "purity": m["purity"].astype(float),
        "ploidy": m["ploidy"].astype(float), "wgd": m["wgd_status"], "qc": "NA",
        "source": "pcawg", "wgd_uncertain": unc.to_numpy(),
        "purity_conf_mad": m["purity_conf_mad"], "num_subclones": m["num_subclones"],
        "is_preferred": m["is_preferred"],
    })


def _build_pcawg(root: str, grid: Grid, limit: Optional[int] = None) -> Cohort:
    meta = _read_pcawg_meta(root).sort_values("sample", kind="stable").reset_index(drop=True)
    if limit is not None:
        meta = meta.iloc[:limit]
    want = set(meta["sample"])
    tabs: dict[str, pd.DataFrame] = {}
    for fid, raw in _iter_tar_tables(os.path.join(root, _PCAWG_TAR), _PCAWG_SUFFIX, want):
        tabs[fid] = pcawg_segments_to_internal(raw)
        if len(tabs) == len(want):
            break
    meta = meta[meta["sample"].isin(tabs)].reset_index(drop=True)
    r = _bin_tables([tabs[s] for s in meta["sample"]], grid, ["cn", "cn_cw"])
    return Cohort(r["cn_int"], r["mean0"], r["breakpoint"], r["coverage"], _finish_meta(meta),
                  grid, block_cn_cw=r["mean1"])


def load_pcawg(root: str, grid: Grid, histologies: Optional[Sequence[str]] = None,
               require_no_wgd: bool = True, exclude_uncertain: bool = True,
               min_purity: Optional[float] = 0.3, use_cache: bool = True,
               cache_dir: str = DEFAULT_CACHE_DIR, one_per_donor: bool = True,
               limit: Optional[int] = None) -> Cohort:
    """PCAWG consensus cohort. ``block_cn_int`` from clonal ``total_cn``; ``block_cn_cw`` holds the
    length-weighted Battenberg cell-weighted CN. ``qc`` is 'NA' (PCAWG has no QC column here).

    Filters: ``histologies`` (``histology_abbreviation`` values), ``wgd_status == 'no_wgd'``,
    ``not wgd_uncertain``, ``purity >= min_purity``; then (``one_per_donor``) the first sample per
    ``icgc_donor_id`` by sorted ``tumour_aliquot_id`` among the survivors.
    ``limit`` bins only the first ``limit`` samples (sorted by aliquot) and bypasses the cache.
    """
    coh = None
    if limit is None and use_cache:
        coh = _load_cache(cache_dir, "pcawg", grid)
    if coh is None:
        coh = _build_pcawg(root, grid, limit)
        if limit is None and use_cache:
            _save_cache(cache_dir, "pcawg", coh)
    m = coh.meta
    keep = np.ones(len(m), dtype=bool)
    if histologies is not None:
        keep &= m["cancer_type"].isin(list(histologies)).to_numpy()
    if require_no_wgd:
        keep &= (m["wgd"] == "no_wgd").to_numpy()
    if exclude_uncertain:
        keep &= ~m["wgd_uncertain"].astype(bool).to_numpy()
    if min_purity is not None:
        keep &= (m["purity"] >= min_purity).to_numpy()
    idx = np.flatnonzero(keep)
    if one_per_donor and len(idx):
        # PCAWG's own representative sample first (is_preferred), then the lowest aliquot id
        sub = m.iloc[idx].assign(_i=idx, _pref=m.iloc[idx]["is_preferred"].astype(bool).to_numpy())
        sub = sub.sort_values(["_pref", "sample"], ascending=[False, True], kind="stable")
        idx = np.sort(sub.drop_duplicates("donor", keep="first")["_i"].to_numpy())
    return coh.subset(idx)


# --------------------------------------------------------------------------- tables
def _arm_col_name(col: str) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*\((\d+[pq])\)\s*", col)
    return m.group(2) if m else col.strip()


def load_taylor_arm_calls(path: str) -> pd.DataFrame:
    """Taylor et al. 2018 arm calls (CR line endings). Index ``Sample`` (e.g. TCGA-18-3406-01);
    columns ``case_id`` (first 12 chars), ``type``, ``aneuploidy_score`` and the 39 arms
    ('1p'..'22q', '13 (13q)' renamed '13q' etc.) as float -1/0/1 with NaN = not called."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    df = df.rename(columns=_arm_col_name)
    missing = [a for a in ARM_NAMES if a not in df.columns]
    if missing:
        raise ValueError(f"arm columns missing from {path}: {missing}")
    out = pd.DataFrame(index=pd.Index(df["Sample"].str.strip(), name="sample"))
    out["case_id"] = out.index.str[:12]
    out["type"] = df["Type"].to_numpy()
    out["aneuploidy_score"] = pd.to_numeric(df["Aneuploidy Score"], errors="coerce").to_numpy()
    for a in ARM_NAMES:
        out[a] = pd.to_numeric(df[a], errors="coerce").astype(float).to_numpy()
    return out


def load_absolute_table(path: str) -> pd.DataFrame:
    """TCGA ABSOLUTE purity/ploidy table indexed by case id (first 12 chars of ``array``).

    Columns: ``sample_id`` (``array``, e.g. TCGA-OR-A5J1-01), ``aliquot`` (``sample``),
    ``call_status`` ('called', 'legacy_call', 'maf_call', ..., NaN), ``purity``, ``ploidy``,
    ``genome_doublings``, ``coverage_80pct_power``, ``cancer_dna_fraction``,
    ``subclonal_genome_fraction``, ``solution``. Cases with several samples keep the first by
    sorted ``array`` id (so primary '-01' before metastatic '-06').
    """
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={
        "array": "sample_id", "sample": "aliquot", "call status": "call_status",
        "Genome doublings": "genome_doublings", "Coverage for 80% power": "coverage_80pct_power",
        "Cancer DNA fraction": "cancer_dna_fraction",
        "Subclonal genome fraction": "subclonal_genome_fraction",
    })
    df = df.sort_values("sample_id", kind="stable")
    df["case_id"] = df["sample_id"].str[:12]
    df = df.drop_duplicates("case_id", keep="first").set_index("case_id")
    return df
