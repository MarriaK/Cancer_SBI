"""Summary statistics for the prior predictive check (PPC, step 4): per-tumour scalars and cohort
frequency vectors, computed with ONE code path for real and simulated copy-number profiles.

Inputs follow the conventions of :mod:`ppc.decompose`: an integer profile ``x_int`` of length
``grid.D`` (total copy number per 5 Mb block, ``-1`` = masked) and, optionally, a float
cell-weighted profile ``x_cw`` (``NaN`` = masked; PCAWG ``block_cn_cw`` or the simulator's ``bulk``).
Apply the structural mask (``reports/step1/structural_mask.npy``) to every profile first, with
:func:`apply_structural_mask`, so real and simulated tumours are scored on the same blocks.

"Unmasked" below always means ``x_int != -1``. The cell-weighted statistics (``ploidy_cw``,
``subclonal_frac``) use the blocks that are unmasked in ``x_int`` AND finite in ``x_cw`` (PCAWG has
extra NaNs where the Battenberg first state is NA). Block length in Mb is ``grid.region_len / 1e6``
(5 Mb on the project grid); lengths count UNMASKED blocks, as ``Run.n_blocks`` does.

Per-tumour statistics (``STAT_COLUMNS``; prior knobs they respond to in ``STAT_KNOBS``):

=======================  ==========================================================================
ploidy / ploidy_cw       mean CN over unmasked blocks of ``x_int`` / ``x_cw``
fga                      fraction of unmasked blocks with ``x_int != baseline`` (fraction genome
                         altered; Hieronymus et al. 2014, PNAS 111:E2734, adapted to integer CN)
aneuploidy_score         number of the 39 called arms with a nonzero arm call (Taylor et al. 2018,
                         Cancer Cell 33:676); split into n_arm_gain_calls / n_arm_loss_calls
n_chrom, n_arm,          whole-chromosome, arm, strict focal (focal minus ``*_orig_baseline``) and
n_focal_strict, n_broad  broad event counts of :func:`ppc.decompose.decompose` (Zack et al. 2013;
                         Beroukhim et al. 2010 / Mermel et al. 2011 GISTIC 2.0)
n_breakpoints            adjacent unmasked block pairs within a chromosome whose CN differs (the
                         :func:`ppc.decompose.decompose_cohort` definition); breakpoints_per_100mb
                         = n_breakpoints / (n_unmasked * Mb per block) * 100
seg_len_median_mb,       median / 90th percentile length (Mb) of the maximal runs of IDENTICAL
seg_len_p90_mb           deviation over altered unmasked blocks, within arms (never crossing a
                         centromere), on the ORIGINAL profile
frac_arm_end_bounded     fraction of those runs touching the first or last unmasked block of their
                         arm, i.e. EITHER arm end (the centromere end included). This is broader
                         than the telomere-only "telomere-bounded" SCNAs of Zack et al. 2013, Nat
                         Genet 45:1134 (see the function docstring for the exact convention).
frac_telomere_bounded    fraction of the same runs touching the TELOMERIC end of their arm: the
                         first unmasked block of a p arm or the last unmasked block of a q arm
                         (Zack et al. 2013 telomere-bounded SCNAs; centromere-bounded runs do not
                         count), so frac_telomere_bounded <= frac_arm_end_bounded. Pre-registered
                         2026-10-22 before the calibration-v3 check.
                         The four segment statistics are 0 when the profile has no altered run
                         ("0 = no altered segment"), NaN only when the profile is fully masked.
cn_frac_0 .. cn_frac_5plus  fraction of unmasked blocks at total CN 0, 1, 2, 3, 4, >= 5 (Steele et
                         al. 2022, Nature 606:984, total-CN bins collapsed)
cn_max                   maximum unmasked CN
wgii                     weighted genome instability index: per chromosome the fraction of unmasked
                         blocks with CN != baseline, averaged over chromosomes (Burrell et al. 2013,
                         Nature 494:492; TRACERx usage, Jamal-Hanjani et al. 2017)
subclonal_frac           fraction of blocks with ``|x_cw - round(x_cw)| > 0.1 + 1e-6`` (the 1e-6
                         makes float32 inputs behave like float64 at the boundary), a proxy for the
                         genome fraction with subclonal CN (Dentro et al. 2021, Cell 184:2239)
subclonal_frac_nobp      subclonal_frac after removing every block that carries a breakpoint of
                         ``x_int`` (its CN differs from the nearest unmasked block to its left or
                         right within the chromosome; masked blocks are skipped, as in
                         n_breakpoints). Reason: on real data a block straddling a clonal segment
                         boundary has a non-integer length-weighted mean CN that is not
                         subclonality. Both are reported (pre-registered 2026-10-22).
=======================  ==========================================================================

Simulation-only diagnostics (``SIM_DIAG_COLUMNS``; never gating, not in ``STAT_COLUMNS``):
``cn_max_driver`` / ``cn_max_passenger`` = maximum unmasked ``x_int`` over the driver / non-driver
blocks of ``driver_mask``; NaN when no mask is given (always, for real tumours) or the set is empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ppc.decompose import _cohort_baselines, decompose, segments, taylor_arm_calls
from ppc.grid import Grid
from ppc.step1_change_sizes import apply_mask

SUBCLONAL_TOL = 0.1   # |cw - round(cw)| above this counts as a subclonal block
_SUBCLONAL_EPS = 1e-6  # added to the threshold so float32 inputs (e.g. 2.1f) behave like float64

STAT_COLUMNS: tuple[str, ...] = (
    "ploidy", "ploidy_cw", "fga",
    "aneuploidy_score", "n_arm_gain_calls", "n_arm_loss_calls",
    "n_chrom", "n_arm", "n_focal_strict", "n_broad",
    "n_breakpoints", "breakpoints_per_100mb",
    "seg_len_median_mb", "seg_len_p90_mb", "frac_arm_end_bounded", "frac_telomere_bounded",
    "cn_frac_0", "cn_frac_1", "cn_frac_2", "cn_frac_3", "cn_frac_4", "cn_frac_5plus",
    "cn_max", "wgii", "subclonal_frac", "subclonal_frac_nobp",
)

CW_COLUMNS: tuple[str, ...] = ("ploidy_cw", "subclonal_frac", "subclonal_frac_nobp")

# Simulation-only diagnostics (need the driver block mask); never part of the PPC tests.
SIM_DIAG_COLUMNS: tuple[str, ...] = ("cn_max_driver", "cn_max_passenger")

# Prior knob(s) each statistic is most sensitive to (Notion 3.5 section 4). Knob names: q (WGD /
# ploidy-shift probability), A (arm event rate), C (whole-chromosome event rate), F (focal driver
# rate), pi (driver fraction), m (driver effect size), cap (viability CN cap), length_mean (focal
# length mean); "pi*m" = their product. frac_arm_end_bounded, frac_telomere_bounded, n_arm_*_calls,
# cn_max and subclonal_frac_nobp are not in the Notion table; their entries follow the closest listed
# statistic (aneuploidy_score, cn_frac, subclonal_frac).
STAT_KNOBS: dict[str, tuple[str, ...]] = {
    "ploidy": ("q", "A", "C"),
    "ploidy_cw": ("q", "A", "C"),
    "fga": ("pi*m", "A", "C"),
    "aneuploidy_score": ("A", "C"),
    "n_arm_gain_calls": ("A", "C"),
    "n_arm_loss_calls": ("A", "C"),
    "n_chrom": ("C",),
    "n_arm": ("A",),
    "n_focal_strict": ("F",),
    "n_broad": ("length_mean",),
    "n_breakpoints": ("F", "A"),
    "breakpoints_per_100mb": ("F", "A"),
    "seg_len_median_mb": ("length_mean", "F"),
    "seg_len_p90_mb": ("length_mean", "F"),
    "frac_arm_end_bounded": ("A", "C", "F"),
    "frac_telomere_bounded": ("A", "C", "F"),
    "cn_frac_0": ("m", "pi", "cap"),
    "cn_frac_1": ("m", "pi", "cap"),
    "cn_frac_2": ("m", "pi", "cap"),
    "cn_frac_3": ("m", "pi", "cap"),
    "cn_frac_4": ("m", "pi", "cap"),
    "cn_frac_5plus": ("m", "pi", "cap"),
    "cn_max": ("m", "pi", "cap"),
    "wgii": ("pi", "A", "C"),
    "subclonal_frac": ("A", "C", "F", "m"),   # A, C, F raise it; strong selection (m) lowers it
    "subclonal_frac_nobp": ("A", "C", "F", "m"),
}


# ---------------------------------------------------------------------------------------------
def apply_structural_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of ``X`` (1-D profile or [n, D]) with the blocks in ``mask`` masked.

    Integer arrays get ``-1`` (via :func:`ppc.step1_change_sizes.apply_mask`, result int64);
    float arrays get ``NaN`` (result float64).
    """
    X = np.asarray(X)
    mask = np.asarray(mask, dtype=bool)
    if np.issubdtype(X.dtype, np.integer):
        return apply_mask(X, mask)
    Y = np.array(X, dtype=np.float64, copy=True)
    Y[..., mask] = np.nan
    return Y


class _GridInfo:
    """Grid-derived arrays reused across profiles."""

    def __init__(self, grid: Grid):
        self.chrom_id = grid.chrom_of_block().astype(np.int64)
        self.n_chrom = len(grid.chrom_order)
        self.arm_id = np.empty(grid.D, dtype=np.int64)
        for i, a in enumerate(grid.arms):
            self.arm_id[a.start:a.end] = i
        self.arm_index = {a.name: i for i, a in enumerate(grid.arms)}
        self.mb_per_block = grid.region_len / 1e6


_INFO_CACHE: dict[int, tuple[Grid, _GridInfo]] = {}


def _info(grid: Grid) -> _GridInfo:
    hit = _INFO_CACHE.get(id(grid))
    if hit is None or hit[0] is not grid:
        hit = (grid, _GridInfo(grid))
        _INFO_CACHE[id(grid)] = hit
    return hit[1]


def _segment_stats(x: np.ndarray, unm: np.ndarray, grid: Grid, baseline: int,
                   info: _GridInfo) -> tuple[float, float, float, float]:
    """seg_len_median_mb, seg_len_p90_mb, frac_arm_end_bounded, frac_telomere_bounded of one
    (int64) profile.

    Runs are :func:`ppc.decompose.segments` with ``by='value', within='arm'``: maximal runs of
    identical ``x - baseline != 0`` over unmasked blocks (masked blocks transparent), never crossing
    the centromere. Length = unmasked blocks * Mb per block. A run is "arm-end-bounded" if it
    contains the first or the last unmasked block of its arm, i.e. it touches EITHER end of the arm:
    an arm-end run at the centromere side counts, as does a whole-arm run. This is broader than
    Zack et al. 2013's telomere-bounded class, which is frac_telomere_bounded: the run contains the
    first unmasked block of a p arm or the last unmasked block of a q arm (telomere end only).

    A profile with no altered run returns (0.0, 0.0, 0.0, 0.0): 0 = no altered segment. This keeps
    quiet tumours (e.g. many THCA) in the joint tests instead of dropping them as NaN rows.
    """
    runs = segments(x, grid, baseline, by="value", within="arm")
    if not runs:
        return 0.0, 0.0, 0.0, 0.0
    lens = np.array([r.n_blocks for r in runs], dtype=np.float64) * info.mb_per_block
    bounded = telomeric = 0
    for r in runs:
        a = grid.arms[info.arm_index[r.arm]]
        u = np.flatnonzero(unm[a.start:a.end])
        first, last = a.start + int(u[0]), a.start + int(u[-1])
        at_first, at_last = r.start == first, r.end - 1 == last
        bounded += at_first or at_last
        telomeric += at_first if a.arm == "p" else at_last
    n = len(runs)
    return float(np.median(lens)), float(np.percentile(lens, 90)), bounded / n, telomeric / n


def _breakpoint_blocks(x: np.ndarray, unm: np.ndarray, info: _GridInfo) -> np.ndarray:
    """Bool [D]: unmasked blocks whose CN differs from the nearest unmasked block on either side
    within the same chromosome (both blocks of every pair counted in n_breakpoints)."""
    idx = np.flatnonzero(unm)
    xu, cu = x[idx], info.chrom_id[idx]
    step = (xu[1:] != xu[:-1]) & (cu[1:] == cu[:-1])
    out = np.zeros(x.shape, dtype=bool)
    out[idx[:-1][step]] = True
    out[idx[1:][step]] = True
    return out


def per_tumour_stats(x_int: np.ndarray, grid: Grid, baseline: int = 2,
                     x_cw: np.ndarray | None = None,
                     driver_mask: np.ndarray | None = None) -> dict[str, float]:
    """All ``STAT_COLUMNS`` plus ``SIM_DIAG_COLUMNS`` for ONE profile (definitions and references in
    the module docstring).

    ``x_int``: int block CN, ``-1`` masked (apply the structural mask first). ``baseline``: CN of
    an unaltered block (2, or ``2 * 2**n_wgd``). ``x_cw``: optional cell-weighted float profile
    (``NaN`` masked); without it the ``CW_COLUMNS`` are NaN. ASCAT (TCGA) profiles carry no
    cell-weighted information, so their ``subclonal_frac`` is always NaN; for simulations pass the
    cell-weighted ``bulk`` (``SimCohort.block_cn_mean``). ``driver_mask``: optional bool [D] of the
    simulator's driver blocks (simulations only); without it ``cn_max_driver`` and
    ``cn_max_passenger`` are NaN.

    Event counts come from :func:`ppc.decompose.decompose`, with ``n_focal_strict`` as in
    :func:`ppc.step1_change_sizes.derive_per_tumour`. A profile with no unmasked block returns NaN
    for every ratio/length statistic and 0 for the counts. A profile with unmasked blocks but no
    altered run gets 0 for the four segment statistics.
    """
    x = np.asarray(x_int).astype(np.int64)
    if x.shape != (grid.D,):
        raise ValueError(f"profile must have shape ({grid.D},), got {x.shape}")
    info = _info(grid)
    baseline = int(baseline)
    unm = x != -1
    n_unm = int(unm.sum())
    dec = decompose(x, grid, baseline)
    calls = np.array(list(dec.arm_calls.values()), dtype=np.int64)

    out: dict[str, float] = dict.fromkeys(STAT_COLUMNS + SIM_DIAG_COLUMNS, np.nan)
    out["aneuploidy_score"] = float(np.count_nonzero(calls))
    out["n_arm_gain_calls"] = float(np.count_nonzero(calls > 0))
    out["n_arm_loss_calls"] = float(np.count_nonzero(calls < 0))
    out["n_chrom"] = float(dec.whole_chrom_gain + dec.whole_chrom_loss)
    out["n_arm"] = float(dec.arm_gain + dec.arm_loss)
    out["n_focal_strict"] = float(dec.focal_gain + dec.focal_loss
                                  - dec.focal_gain_orig_baseline - dec.focal_loss_orig_baseline)
    out["n_broad"] = float(dec.broad_gain + dec.broad_loss)

    xu = x[unm]
    cu = info.chrom_id[unm]
    nbp = int(np.count_nonzero((xu[1:] != xu[:-1]) & (cu[1:] == cu[:-1]))) if n_unm else 0
    out["n_breakpoints"] = float(nbp)

    if n_unm:
        alt = xu != baseline
        out["ploidy"] = float(xu.mean())
        out["fga"] = float(alt.mean())
        out["breakpoints_per_100mb"] = nbp / (n_unm * info.mb_per_block) * 100.0
        for k in range(5):
            out[f"cn_frac_{k}"] = float(np.mean(xu == k))
        out["cn_frac_5plus"] = float(np.mean(xu >= 5))
        out["cn_max"] = float(xu.max())
        n_c = np.bincount(cu, minlength=info.n_chrom)
        a_c = np.bincount(cu, weights=alt, minlength=info.n_chrom)
        has = n_c > 0
        out["wgii"] = float(np.mean(a_c[has] / n_c[has]))
        (out["seg_len_median_mb"], out["seg_len_p90_mb"], out["frac_arm_end_bounded"],
         out["frac_telomere_bounded"]) = _segment_stats(x, unm, grid, baseline, info)

    if x_cw is not None:
        cw = np.asarray(x_cw, dtype=np.float64)
        if cw.shape != (grid.D,):
            raise ValueError(f"x_cw must have shape ({grid.D},), got {cw.shape}")
        ok = np.isfinite(cw) & unm
        if ok.any():
            v = cw[ok]
            out["ploidy_cw"] = float(v.mean())
            sub = np.abs(v - np.round(v)) > SUBCLONAL_TOL + _SUBCLONAL_EPS
            out["subclonal_frac"] = float(np.mean(sub))
            keep = ~_breakpoint_blocks(x, unm, info)[ok]
            if keep.any():
                out["subclonal_frac_nobp"] = float(np.mean(sub[keep]))

    if driver_mask is not None:
        dm = np.asarray(driver_mask, dtype=bool)
        if dm.shape != (grid.D,):
            raise ValueError(f"driver_mask must have shape ({grid.D},), got {dm.shape}")
        for col, sel in (("cn_max_driver", dm & unm), ("cn_max_passenger", ~dm & unm)):
            if sel.any():
                out[col] = float(x[sel].max())
    return out


def cohort_stats(X_int: np.ndarray, grid: Grid, baselines=None,
                 X_cw: np.ndarray | None = None) -> pd.DataFrame:
    """:func:`per_tumour_stats` over the rows of ``X_int`` ([N, D]); one row per profile,
    columns ``STAT_COLUMNS``. ``baselines``: None (all 2), a scalar, or one int per profile.
    ``X_cw``: optional [N, D] cell-weighted profiles aligned with ``X_int``."""
    X = np.asarray(X_int)
    if X.ndim != 2 or X.shape[1] != grid.D:
        raise ValueError(f"X_int must have shape (N, {grid.D}), got {X.shape}")
    N = X.shape[0]
    b = _cohort_baselines(N, baselines)
    if X_cw is not None:
        X_cw = np.asarray(X_cw)
        if X_cw.shape != X.shape:
            raise ValueError(f"X_cw must have shape {X.shape}, got {X_cw.shape}")
    rows = [per_tumour_stats(X[r], grid, int(b[r]), None if X_cw is None else X_cw[r])
            for r in range(N)]
    return pd.DataFrame(rows, columns=list(STAT_COLUMNS), dtype=np.float64)


def arm_frequencies(X_int: np.ndarray, grid: Grid, baselines=None) -> pd.DataFrame:
    """Per-arm gain/loss frequencies over a cohort ([N, D]).

    Indexed by the 39 called arm names (``grid.called_arms`` order); ``gain_freq`` / ``loss_freq`` =
    fraction of profiles whose Taylor et al. 2018 arm call (:func:`ppc.decompose.taylor_arm_calls`,
    80% of unmasked blocks) is +1 / -1. An empty cohort gives NaN.
    """
    X = np.asarray(X_int)
    if X.ndim != 2 or X.shape[1] != grid.D:
        raise ValueError(f"X_int must have shape (N, {grid.D}), got {X.shape}")
    N = X.shape[0]
    b = _cohort_baselines(N, baselines)
    names = [a.name for a in grid.called_arms]
    C = np.zeros((N, len(names)), dtype=np.int64)
    for r in range(N):
        calls, _ = taylor_arm_calls(X[r], grid, int(b[r]))
        C[r] = [calls[n] for n in names]
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = (C > 0).sum(0) / N if N else np.full(len(names), np.nan)
        loss = (C < 0).sum(0) / N if N else np.full(len(names), np.nan)
    return pd.DataFrame({"gain_freq": gain, "loss_freq": loss}, index=pd.Index(names, name="arm"))


def block_frequencies(X_int: np.ndarray, grid: Grid, baselines=None,
                      mask: np.ndarray | None = None) -> pd.DataFrame:
    """Per-block gain/loss frequencies over a cohort ([N, D]) - the cohort CN frequency plot.

    Index 0..D-1 with columns ``chrom``, ``r`` (block index within the chromosome), ``gain_freq``
    (fraction of profiles with CN > baseline) and ``loss_freq`` (CN < baseline). The denominator of
    each block is the number of profiles in which it is unmasked; blocks in ``mask`` (e.g. the
    structural mask) or masked in every profile are NaN.
    """
    X = np.asarray(X_int).astype(np.int64)
    if X.ndim != 2 or X.shape[1] != grid.D:
        raise ValueError(f"X_int must have shape (N, {grid.D}), got {X.shape}")
    b = _cohort_baselines(X.shape[0], baselines)[:, None]
    unm = X != -1
    if mask is not None:
        unm &= ~np.asarray(mask, dtype=bool)[None, :]
    n = unm.sum(0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(n > 0, (unm & (X > b)).sum(0) / n, np.nan)
        loss = np.where(n > 0, (unm & (X < b)).sum(0) / n, np.nan)
    cid = grid.chrom_of_block()
    chrom = np.array(grid.chrom_order, dtype=object)[cid]
    r = np.arange(grid.D) - np.array([grid.offsets[c] for c in grid.chrom_order])[cid]
    return pd.DataFrame({"chrom": chrom, "r": r, "gain_freq": gain, "loss_freq": loss},
                        index=pd.RangeIndex(grid.D, name="block"))


def sim_observed_profile(clones_block_cn: np.ndarray, popsize: np.ndarray) -> np.ndarray:
    """The "observed" integer profile B of one simulated tumour: per block, the cell-weighted mode.

    For every block, each CN state gets the summed ``popsize`` of the clones carrying it, and the
    state with the largest share wins; exact ties go to the LOWER CN. This mimics a bulk caller that
    reports the dominant clonal state (ASCAT / ABSOLUTE integer profiles): a subclonal state held by
    fewer cells than the dominant one - in particular any state below 50% of cells when the rest
    share one state - vanishes from the profile.

    ``clones_block_cn``: int [n_clones, D] (CN >= 0); ``popsize``: [n_clones] cell counts (>= 0, not
    all 0). Returns int64 [D].
    """
    cn = np.asarray(clones_block_cn)
    w = np.asarray(popsize, dtype=np.float64)
    if cn.ndim != 2 or w.shape != (cn.shape[0],):
        raise ValueError("clones_block_cn must be [n_clones, D] and popsize [n_clones]")
    if cn.size and cn.min() < 0:
        raise ValueError("clone copy numbers must be >= 0 (simulations have no masked blocks)")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("popsize must be non-negative with a positive total")
    cn = cn.astype(np.int64)
    D = cn.shape[1]
    W = np.zeros((D, int(cn.max()) + 1), dtype=np.float64)
    cols = np.broadcast_to(np.arange(D), cn.shape)
    np.add.at(W, (cols.ravel(), cn.ravel()), np.repeat(w, D))
    return np.argmax(W, axis=1).astype(np.int64)   # argmax returns the first (= lowest CN) on ties
