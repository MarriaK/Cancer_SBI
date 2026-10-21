"""Decompose an integer copy-number profile into whole-chromosome, arm-level and focal events.

One set of published rules, applied identically to real tumours and to simulated tumours on the
project's 589-block grid (see ``ppc.grid``). A profile ``x`` is a 1-D int array of length ``grid.D``
holding the total copy number per block; ``-1`` marks a masked/unknown block. The baseline is the
expected copy number of an unaltered block (2 for a non-WGD tumour, ``2 * 2**n_wgd`` in general).

Rules, applied big to small and subtracting as we go (the broad/focal deconstruction of GISTIC 2.0,
Mermel et al. 2011, Genome Biol 12:R41):

R1  Deviation ``d = x - baseline`` on unmasked blocks. Masked blocks are excluded from every count
    and every denominator, and are transparent for runs (a run continues across them).
R2  Arm call (Taylor et al. 2018, Cancer Cell 33:676): on each of the 39 called arms, +1 if at least
    80% of the unmasked blocks have ``d > 0``, -1 if at least 80% have ``d < 0``, else 0. The arm
    level is ``sign * median(|d|)`` over the altered blocks in the called direction.
R3  Whole chromosome (Zack et al. 2013, Nat Genet 45:1134): both arms called with the same sign give
    one whole-chromosome event of level ``sign * min(|L_p|, |L_q|)``, subtracted from the whole
    chromosome; each arm is then re-called (R2) on the residual and, if still called, counted and
    subtracted as an arm event. Otherwise every called arm is one arm event. On the acrocentric
    chromosomes (13, 14, 15, 21, 22) only q is called; such a call is an arm event, additionally
    tallied in ``acro_q_calls`` so Taylor's alternative "13 (13q) = whole chromosome" convention can
    be reconstructed.
R4  Focal (Beroukhim et al. 2010, Nature 463:899; GISTIC ``broad_len_cutoff = 0.5``): on the
    residual, maximal runs of same-sign deviation inside one arm (never across a centromere; the
    uncalled acrocentric p arms are scanned too). A run shorter than half the arm is focal, with
    magnitude ``sign * max|d|``; a run covering at least half the arm is "broad" and reported apart.

Conventions taken where the published rules leave a choice (all documented in the docstrings):
* "Half the arm" and "80% of the arm" use UNMASKED block counts (R1), which equal ``arm.n_blocks``
  whenever the arm has no masked block.
* The median in R2 is the LOWER median when the number of altered blocks is even, so the arm level
  is always an integer and never exceeds what at least half of the altered blocks carry.
* A ``Run``/``Event`` spans ``[start, end)`` from its first to its last unmasked block; ``n_blocks``
  counts its unmasked blocks only (so ``end - start`` can exceed ``n_blocks`` across masked gaps).
* ``Decomposition.residual`` is the DEVIATION left after the whole-chromosome and arm subtractions,
  i.e. the input of R4 (focal and broad runs are reported, not subtracted); masked blocks hold 0 and
  are flagged in ``Decomposition.masked``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ppc.grid import Grid

ARM_CALL_FRACTION = (4, 5)    # 80% as an exact integer ratio: 5 * n_altered >= 4 * n_unmasked
BROAD_LEN_CUTOFF = (1, 2)     # 50% as an exact integer ratio: 2 * n_run >= 1 * n_unmasked_arm

COUNT_COLUMNS = (
    "whole_chrom_gain", "whole_chrom_loss", "arm_gain", "arm_loss", "acro_q_calls",
    "focal_gain", "focal_loss", "broad_gain", "broad_loss",
    "focal_gain_orig_baseline", "focal_loss_orig_baseline",
    "broad_gain_orig_baseline", "broad_loss_orig_baseline",
)


@dataclass(frozen=True)
class Run:
    """A maximal run of unmasked blocks; ``level`` = sign*max|d| (by='sign') or the exact d (by='value')."""
    chrom: str
    arm: str | None     # arm name such as '8q', or None for chromosome-wide runs
    start: int          # flat index of the first unmasked block (inclusive)
    end: int            # flat index after the last unmasked block (exclusive)
    n_blocks: int       # number of UNMASKED blocks in the run
    level: int


@dataclass(frozen=True)
class Event:
    """One detected copy-number event. ``magnitude`` is signed; ``order`` is the detection order."""
    level: str          # 'chrom' | 'arm' | 'focal' | 'broad'
    chrom: str
    arm: str | None     # None for whole-chromosome events
    start: int
    end: int
    n_blocks: int       # unmasked blocks covered
    magnitude: int
    order: int
    # focal/broad only: True iff every unmasked block of the run was exactly at baseline in the
    # ORIGINAL profile, i.e. the run exists only because a chromosome/arm level was subtracted
    orig_at_baseline: bool = False


@dataclass
class Decomposition:
    """Per-profile result of :func:`decompose` (rule R5)."""
    whole_chrom_gain: int = 0
    whole_chrom_loss: int = 0
    arm_gain: int = 0
    arm_loss: int = 0
    acro_q_calls: int = 0       # subset of arm_gain + arm_loss (acrocentric q-arm calls)
    focal_gain: int = 0
    focal_loss: int = 0
    broad_gain: int = 0
    broad_loss: int = 0
    # subsets of the four counts above: runs whose blocks were all at baseline in the original
    # profile (artefacts of the chrom/arm subtraction rather than copy changes of their own)
    focal_gain_orig_baseline: int = 0
    focal_loss_orig_baseline: int = 0
    broad_gain_orig_baseline: int = 0
    broad_loss_orig_baseline: int = 0
    events: list[Event] = field(default_factory=list)
    arm_calls: dict[str, int] = field(default_factory=dict)     # R2 on the ORIGINAL profile
    arm_levels: dict[str, int] = field(default_factory=dict)    # R2 levels on the ORIGINAL profile
    residual: np.ndarray | None = None   # deviation after chrom+arm subtraction; 0 on masked blocks
    masked: np.ndarray | None = None     # bool, True where the input was -1
    focal_lengths: list[int] = field(default_factory=list)
    value_runs: list[int] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in COUNT_COLUMNS}


# ---------------------------------------------------------------------------------------------
# grid-derived lookup tables (computed once per Grid object)
# ---------------------------------------------------------------------------------------------
class _Tables:
    def __init__(self, grid: Grid):
        self.n_arms = len(grid.arms)
        self.arm_id = np.empty(grid.D, dtype=np.int64)
        for i, a in enumerate(grid.arms):
            self.arm_id[a.start:a.end] = i
        self.chrom_id = grid.chrom_of_block().astype(np.int64)
        self.arm_names = [a.name for a in grid.arms]
        called = {(a.chrom, a.arm) for a in grid.called_arms}
        self.is_called = np.array([(a.chrom, a.arm) in called for a in grid.arms])
        self.arms_of_chrom: dict[str, tuple[int, int]] = {}
        for i, a in enumerate(grid.arms):
            p, q = self.arms_of_chrom.get(a.chrom, (-1, -1))
            self.arms_of_chrom[a.chrom] = (i, q) if a.arm == "p" else (p, i)


_TABLE_CACHE: dict[int, tuple[Grid, _Tables]] = {}


def _tables(grid: Grid) -> _Tables:
    hit = _TABLE_CACHE.get(id(grid))
    if hit is None or hit[0] is not grid:
        hit = (grid, _Tables(grid))
        _TABLE_CACHE[id(grid)] = hit
    return hit[1]


def _deviation(x: np.ndarray, grid: Grid, baseline: int) -> tuple[np.ndarray, np.ndarray]:
    """R1: integer deviation (0 on masked blocks) and the unmasked mask."""
    x = np.asarray(x)
    if x.shape != (grid.D,):
        raise ValueError(f"profile must have shape ({grid.D},), got {x.shape}")
    x = x.astype(np.int64)
    unm = x != -1
    d = np.where(unm, x - int(baseline), 0)
    return d, unm


# ---------------------------------------------------------------------------------------------
# runs (R1 transparency, R4 run finding)
# ---------------------------------------------------------------------------------------------
def _runs_raw(d: np.ndarray, unm: np.ndarray, group: np.ndarray, by: str):
    """Maximal runs of nonzero d over unmasked blocks, split where ``group`` or the key changes.

    Returns arrays (first_flat, last_flat_exclusive, n_blocks, level, group_id).
    """
    u = np.flatnonzero(unm)
    if u.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e, e, e
    du = d[u]
    key = np.sign(du) if by == "sign" else du
    gu = group[u]
    brk = np.flatnonzero((key[1:] != key[:-1]) | (gu[1:] != gu[:-1])) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [u.size]))
    keep = key[starts] != 0
    starts, ends = starts[keep], ends[keep]
    if starts.size == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e, e, e
    if by == "sign":
        # reduceat spans [starts[k], starts[k+1]); the only extra blocks are dropped d == 0 runs,
        # which cannot raise the max
        mags = np.maximum.reduceat(np.abs(du), starts)
        level = np.sign(du[starts]) * mags
    else:
        level = du[starts]
    return u[starts], u[ends - 1] + 1, ends - starts, level, gu[starts]


def segments(x: np.ndarray, grid: Grid, baseline: int = 2, by: str = "sign",
             within: str = "arm") -> list[Run]:
    """Maximal runs of altered blocks (R1 + R4 run definition).

    A run is a maximal stretch of consecutive UNMASKED blocks with the same sign of
    ``d = x - baseline`` (``by='sign'``; level = sign * max|d|) or the same exact ``d``
    (``by='value'``; level = d), skipping masked blocks (-1) transparently (R1). Blocks with
    ``d == 0`` are not runs. Runs never cross an arm boundary (``within='arm'``, the GISTIC 2.0 /
    Beroukhim 2010 convention that focal events do not cross the centromere) or, with
    ``within='chrom'``, a chromosome boundary (``arm`` is then None). ``n_blocks`` counts unmasked
    blocks; ``[start, end)`` spans the first to the last of them in flat coordinates.
    """
    if by not in ("sign", "value"):
        raise ValueError("by must be 'sign' or 'value'")
    if within not in ("arm", "chrom"):
        raise ValueError("within must be 'arm' or 'chrom'")
    t = _tables(grid)
    d, unm = _deviation(x, grid, baseline)
    group = t.arm_id if within == "arm" else t.chrom_id
    s, e, n, lev, gid = _runs_raw(d, unm, group, by)
    out = []
    for k in range(s.size):
        g = int(gid[k])
        if within == "arm":
            a = grid.arms[g]
            out.append(Run(a.chrom, a.name, int(s[k]), int(e[k]), int(n[k]), int(lev[k])))
        else:
            out.append(Run(grid.chrom_order[g], None, int(s[k]), int(e[k]), int(n[k]), int(lev[k])))
    return out


# ---------------------------------------------------------------------------------------------
# arm calls (R2)
# ---------------------------------------------------------------------------------------------
def _lower_median_abs(v: np.ndarray) -> int:
    v = np.sort(np.abs(v))
    return int(v[(v.size - 1) // 2])


def _arm_calls_dev(d: np.ndarray, unm: np.ndarray, t: _Tables) -> tuple[np.ndarray, np.ndarray]:
    """R2 on a deviation vector, for all 44 arms (uncalled arms forced to 0). Returns (sign, level)."""
    n_unm = np.bincount(t.arm_id, weights=unm, minlength=t.n_arms)
    n_gain = np.bincount(t.arm_id, weights=unm & (d > 0), minlength=t.n_arms)
    n_loss = np.bincount(t.arm_id, weights=unm & (d < 0), minlength=t.n_arms)
    num, den = ARM_CALL_FRACTION
    ok = (n_unm > 0) & t.is_called
    sign = np.zeros(t.n_arms, dtype=np.int64)
    sign[ok & (den * n_gain >= num * n_unm)] = 1
    sign[ok & (den * n_loss >= num * n_unm)] = -1
    level = np.zeros(t.n_arms, dtype=np.int64)
    for i in np.flatnonzero(sign):
        sel = (t.arm_id == i) & unm & (np.sign(d) == sign[i])
        level[i] = sign[i] * _lower_median_abs(d[sel])
    return sign, level


def taylor_arm_calls(x: np.ndarray, grid: Grid, baseline: int = 2) -> tuple[dict[str, int], dict[str, int]]:
    """Arm-level calls of Taylor et al. 2018 (Cancer Cell 33:676), rule R2.

    For each of the 39 called arms (``grid.called_arms``; acrocentric p arms excluded): +1 if at least
    80% of the arm's UNMASKED blocks have ``x - baseline > 0``, -1 if at least 80% have it ``< 0``,
    else 0. The 80% is an exact block-count ratio (4 of 5 = 0.8 is called). An arm with no unmasked
    block is 0. The level is ``sign * median(|d|)`` over the altered blocks in the called direction,
    using the lower median for an even count (an integer >= 1); 0 for uncalled arms.

    Returns ``(calls, levels)``, two dicts keyed by arm name ('1p' ... '22q').
    """
    t = _tables(grid)
    d, unm = _deviation(x, grid, baseline)
    sign, level = _arm_calls_dev(d, unm, t)
    calls, levels = {}, {}
    for i, a in enumerate(grid.arms):
        if t.is_called[i]:
            calls[a.name] = int(sign[i])
            levels[a.name] = int(level[i])
    return calls, levels


# ---------------------------------------------------------------------------------------------
# full decomposition (R1-R5)
# ---------------------------------------------------------------------------------------------
def _span(arm_idx: list[int], grid: Grid, unm: np.ndarray) -> tuple[int, int, int]:
    lo = grid.arms[arm_idx[0]].start
    hi = grid.arms[arm_idx[-1]].end
    idx = np.flatnonzero(unm[lo:hi])
    if idx.size == 0:
        return lo, hi, 0
    return lo + int(idx[0]), lo + int(idx[-1]) + 1, int(idx.size)


def decompose(x: np.ndarray, grid: Grid, baseline: int = 2) -> Decomposition:
    """Whole-chromosome / arm / focal / broad decomposition of one profile (rules R1-R5).

    R1 deviation on unmasked blocks; R2 arm calls (Taylor 2018, 80% of unmasked blocks, lower-median
    level); R3 whole-chromosome events (Zack 2013: both arms called with the same sign, level
    ``sign * min(|L_p|, |L_q|)`` subtracted from the chromosome, then each arm re-called on the
    residual and subtracted if still called), otherwise one arm event per called arm (level L,
    subtracted from the arm; acrocentric q-arm calls also tallied in ``acro_q_calls``); R4 focal and
    broad runs on the residual (Beroukhim 2010 / GISTIC 2.0 ``broad_len_cutoff = 0.5``): a same-sign
    run within one arm shorter than half the arm's unmasked blocks is focal (magnitude
    ``sign * max|d|``), otherwise broad. Subtractions apply to unmasked blocks only. Each focal/broad
    event carries ``orig_at_baseline`` (all its blocks were at baseline in the original profile, so it
    exists only because of a chrom/arm subtraction), tallied in the ``*_orig_baseline`` subset counts
    so that event counting can be done either with or without these subtraction artefacts.

    Note: a broad run can also appear on an arm R2 did call, e.g. after a whole-chromosome
    subtraction leaves an arm in the 50-80% grey zone, or when exactly half of a called arm sits one
    level above the (lower) median.

    ``arm_calls``/``arm_levels`` are R2 on the ORIGINAL profile (for validation against Taylor's
    table); ``residual`` is the deviation after the chromosome and arm subtractions (0 on masked
    blocks); ``focal_lengths`` are the unmasked lengths of the focal events; ``value_runs`` are the
    unmasked lengths of the runs of identical nonzero residual value within arms (``by='value'``),
    which split nested amplitudes that one same-sign focal run merges.
    """
    t = _tables(grid)
    d, unm = _deviation(x, grid, baseline)
    res = d.copy()
    out = Decomposition(masked=~unm)

    sign0, level0 = _arm_calls_dev(d, unm, t)
    for i, a in enumerate(grid.arms):
        if t.is_called[i]:
            out.arm_calls[a.name] = int(sign0[i])
            out.arm_levels[a.name] = int(level0[i])

    events: list[Event] = []

    def add_arm_event(i: int, lev: int) -> None:
        a = grid.arms[i]
        s, e, n = _span([i], grid, unm)
        res[a.start:a.end] -= np.where(unm[a.start:a.end], lev, 0)
        events.append(Event("arm", a.chrom, a.name, s, e, n, int(lev), len(events)))
        if lev > 0:
            out.arm_gain += 1
        else:
            out.arm_loss += 1
        if a.arm == "q" and not t.is_called[t.arms_of_chrom[a.chrom][0]]:
            out.acro_q_calls += 1

    # R3, chromosome by chromosome in genome order
    for c in grid.chrom_order:
        ip, iq = t.arms_of_chrom[c]
        sp, sq = int(sign0[ip]), int(sign0[iq])
        if sp != 0 and sp == sq:
            S = sp * min(abs(int(level0[ip])), abs(int(level0[iq])))
            sl = grid.chrom_slice(c)
            res[sl] -= np.where(unm[sl], S, 0)
            s, e, n = _span([ip, iq], grid, unm)
            events.append(Event("chrom", c, None, s, e, n, int(S), len(events)))
            if S > 0:
                out.whole_chrom_gain += 1
            else:
                out.whole_chrom_loss += 1
            sign1, level1 = _arm_calls_dev(res, unm, t)
            for i in (ip, iq):
                if sign1[i] != 0:
                    add_arm_event(i, int(level1[i]))
        else:
            for i in (ip, iq):
                if sign0[i] != 0:
                    add_arm_event(i, int(level0[i]))

    # R4 focal / broad on the residual, within arms
    n_unm_arm = np.bincount(t.arm_id, weights=unm, minlength=t.n_arms)
    s_, e_, n_, lev_, gid_ = _runs_raw(res, unm, t.arm_id, "sign")
    num, den = BROAD_LEN_CUTOFF
    for k in range(s_.size):
        a = grid.arms[int(gid_[k])]
        nk, lv = int(n_[k]), int(lev_[k])
        blk = slice(int(s_[k]), int(e_[k]))
        at_base = not d[blk][unm[blk]].any()
        kind = "broad" if den * nk >= num * n_unm_arm[gid_[k]] else "focal"
        if kind == "focal":
            out.focal_lengths.append(nk)
        name = f"{kind}_{'gain' if lv > 0 else 'loss'}"
        setattr(out, name, getattr(out, name) + 1)
        if at_base:
            setattr(out, name + "_orig_baseline", getattr(out, name + "_orig_baseline") + 1)
        events.append(Event(kind, a.chrom, a.name, int(s_[k]), int(e_[k]), nk, lv, len(events),
                            at_base))

    out.value_runs = [int(v) for v in _runs_raw(res, unm, t.arm_id, "value")[2]]
    out.events = events
    out.residual = res
    return out


def _cohort_baselines(n: int, baselines) -> np.ndarray:
    if baselines is None:
        return np.full(n, 2, dtype=np.int64)
    b = np.asarray(baselines, dtype=np.int64)
    if b.ndim == 0:
        return np.full(n, int(b), dtype=np.int64)
    if b.shape != (n,):
        raise ValueError(f"baselines must be a scalar or have shape ({n},), got {b.shape}")
    return b


def decompose_cohort(X: np.ndarray, grid: Grid, baselines=None) -> pd.DataFrame:
    """Stack :func:`decompose` over the rows of ``X`` (shape ``(N, grid.D)``) into a DataFrame (R5).

    One row per profile with the thirteen event counts (nine counts plus the four
    ``*_orig_baseline`` subsets) plus three genome-wide summaries on unmasked
    blocks: ``ploidy`` = mean copy number; ``fga`` (fraction genome altered) = fraction of blocks with
    ``x != baseline``; ``n_breakpoints`` = number of adjacent unmasked block pairs within the same
    chromosome whose copy numbers differ (masked blocks skipped). ``baselines`` is None (all 2), a
    scalar, or one integer per profile. Rows with no unmasked block get NaN summaries.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != grid.D:
        raise ValueError(f"X must have shape (N, {grid.D}), got {X.shape}")
    N = X.shape[0]
    b = _cohort_baselines(N, baselines)
    t = _tables(grid)
    counts = np.zeros((N, len(COUNT_COLUMNS)), dtype=np.int64)
    ploidy = np.full(N, np.nan)
    fga = np.full(N, np.nan)
    nbp = np.zeros(N, dtype=np.int64)
    for r in range(N):
        x = X[r].astype(np.int64)
        dec = decompose(x, grid, int(b[r]))
        counts[r] = [getattr(dec, k) for k in COUNT_COLUMNS]
        unm = x != -1
        if unm.any():
            xu = x[unm]
            ploidy[r] = xu.mean()
            fga[r] = np.mean(xu != b[r])
            cu = t.chrom_id[unm]
            nbp[r] = int(np.count_nonzero((xu[1:] != xu[:-1]) & (cu[1:] == cu[:-1])))
    df = pd.DataFrame(counts, columns=list(COUNT_COLUMNS))
    df["ploidy"] = ploidy
    df["fga"] = fga
    df["n_breakpoints"] = nbp
    return df
