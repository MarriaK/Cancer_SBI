"""Endpoint liftover of copy-number segments between hg19 and hg38, and the grid build check.

Why not CrossMap's interval modes: a CNA segment is a claim "CN = v from here to there", not a
sequence. CrossMap's ``bed`` mode lifts the *interval* through every chain block and shatters one
5 Mb segment into thousands of fragments (many on other chromosomes); ``region`` mode then rejects
them all as cross-chromosome. Instead points are lifted with CrossMap's Python API
(``cmmodule.utils.map_coordinates``) and each segment is rebuilt from them.

Segment convention (as in :mod:`ppc.real_loaders`): columns ``chrom`` ('chr1'..'chr22'),
``start``, ``end`` 0-based half-open ``[start, end)``; any other columns are carried through.
``start``/``end`` must be finite and ``start >= 0`` (ValueError otherwise).

Rules
-----
* **Plausible hit.** A lifted point counts only if it lands on the **same chromosome name** and
  within ``max_offset_bp`` (default 10 Mb) of its source position. This rejects
  cross-chromosome hits (the hg38 chr13 tail maps to chrX) and far same-chromosome jumps. Most
  hg19/hg38 offsets are a few Mb. The exception is the main chr9 chain itself, which maps ~1 Mb
  of hg38 60.7-62.7 Mb onto hg19 9p at offsets of 15-19 Mb (e.g. hg38 chr9:61,986,410 -> hg19
  45,122,562). The bound deliberately rejects those hits so that 9q events stay on 9q, treating
  those ~2 Mb as unliftable. With a 20 Mb bound, 9q segments such as hg38 chr9
  [60.8 Mb, 138 Mb) were kept (1.29x, inside the length net) and spilled into hg19 9p blocks 8-9,
  and hg38 chr16 [5 Mb, 34.7 Mb) crossed the chr16 centromere. The length net is the second line
  of defence, not the first.
* **Endpoints.** ``start`` is lifted walking right (inward) and ``end - 1`` walking left until a
  plausible hit. The walk probes the endpoint itself, then the multiples of ``step_bp`` (10 kb)
  beyond it: a fixed genome lattice, so every start inside one gap sees the same probes and lands
  in the same place. The walk never goes past the other end of the segment, and goes at most
  ``max_walk_bp`` if that is given (default unbounded). Unbounded walks are needed to cross the
  5-20 Mb acrocentric p-arm gaps, hg38 centromere models and 1q12/9q12/16q11 heterochromatin.
  Each unique endpoint is walked once per call. Both endpoints are required: otherwise the
  segment is dropped as 'start_unmapped' / 'end_unmapped'. Acrocentric p-arm starts may land
  ~5 Mb off within the uncalled p arm. hg38 chr21 5-10 Mb is the GRCh38 21p false-duplication
  region, and a whole-chr21 segment starts at hg19 15.03 Mb rather than the primary ~9.4 Mb
  image. The called q-arm blocks are unaffected: hg19 chr21 block 3 coverage stays > 0.99.
* **Inverted blocks (strand-gated widening).** An endpoint inside an inverted ('-' strand) chain
  block lifts to the far edge of the block's image: whole-1q hg38 [143,184,588, ...) would start
  at hg19 149,025,498 and lose 4.7 Mb of the 1q21 inversion. So when either endpoint's accepted
  hit is on a '-' chain, the segment becomes ``[min, max + 1)`` over both endpoint hits and its
  interior points. Interior points are the multiples of ``sample_step_bp`` (100 kb; 0 disables
  them) strictly inside ``(start, end)``, each lifted with one plausible-hit call and no walk, on
  a genome-wide lattice cached per process. This also handles a segment lying wholly inside an
  inverted block, whose start lifts above its end. When both hits are on '+' chains the segment
  is endpoint-only and is dropped as 'inverted' unless the start hit is below the end hit.
  Why gated: a CN breakpoint inside an inversion is genuinely ambiguous, because the target
  profile is interleaved A, B, A, B and no pair of intervals represents it. Endpoint-only tiles
  the neighbours without overlap but leaves a gap when a segment has no neighbour there (the 1q
  case). Widening every segment made neighbours overlap across every inversion and
  segmental-duplication region. On a 300-sample ASCAT round trip (16,377 adjacent same-sample
  same-chromosome pairs), ungated widening gave 74 overlaps > 1 Mb, against 14 for one-sided
  gating; the current both-sides envelope gives 26 (0.16 % of pairs) before the local gate below.
* **Local non-collinearity gate** (``gate_window_bp``, default 1 Mb; 0 disables it). A '+'
  endpoint is also widened when lattice points within ``gate_window_bp`` of that end map beyond
  its image by more than ``step_bp``: ``lo = min(lattice in (s0, s0 + window))`` /
  ``hi = max(lattice in (e0 - window, e0))``. It only looks within 1 Mb of each end, unlike the
  full ungated envelope. It recovers content that endpoint-only reconstruction drops where a '+'
  endpoint's neighbourhood maps beyond its image:
  - 1q21 mosaic: whole-1q hg38 segments starting between 143.31 and 145.60 Mb lost a median of
    4.02 Mb (max 4.56 Mb) over 104 '+' starts; with the gate, 0.15 Mb (max 0.95 Mb), and hg19
    chr1 block 29 coverage is >= 0.95.
  - 16p11.2: an hg38 16p segment ending in ~36.1-36.8 Mb ended at hg19 32.1 Mb and lost 3.1 Mb,
    leaving hg19 chr16 block 6 (30-35 Mb) at 0.42 coverage; with the gate, 1.0.
  Why it is on: the PPC consumes blocks. On the 300-sample hg19 -> hg38 -> hg19 ASCAT round
  trip, the gate cut the block-level damage from 0.070 % to 0.025 % (cn_int changed among
  both-valid blocks + newly masked among natively valid blocks; 119 -> 42 sample-blocks). It
  also removed a systematic loss: hg19 chr9 block 13 was newly masked in 27 % of samples.
  The cost is more scattered overlaps between neighbours: 53 instead of 26 overlaps > 1 Mb of
  16,378 adjacent pairs (0.32 %). **Remaining limitation:** a gap > 1 Mb from the end, or a '+'
  endpoint whose stray content lies further in, is still not recovered.
* **Length net.** A segment is dropped as 'length_changed' if its lifted length is above
  ``src_len * (1 + max_len_change_frac) + max_len_change_bp`` or below
  ``src_len * (1 - max_len_change_frac)``. The lower bound is purely multiplicative, so collapses
  of short segments are caught too. Accepted side effect: it drops real hg19 pericentromeric chr9
  sub-arm segments (38-47 Mb) on the hg19->hg38 leg, because that region is rearranged in hg38.
  That is the non-production direction.
* Coordinates must be integers (integral floats are accepted) or a ValueError is raised: no NaN,
  non-finite, fractional, out-of-int64 or negative start. Segments with ``end <= start`` are
  dropped as 'empty', on both the lifting and the same-build path.

Dropped segments are listed in ``LiftoverReport.unmapped`` with a ``reason`` column. No clipping
to the target chromosome length (binning clips), but starts are floored at 0. CrossMap is
imported lazily (and its import-time ``logging.basicConfig`` is undone), so the rest of ``ppc``
works without it. Chains and the interior lattice are cached per process
(:func:`clear_caches` empties them); endpoint lifts are cached only within one call.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ppc.builds import GenomeBuildMismatch, normalise_build
from ppc.grid import Grid

log = logging.getLogger(__name__)

CHAIN_DIR = "/Users/maryamkafi/Documents/PhD_Project/cancer/real_data/liftover"
CHAIN_FILES = {("hg38", "hg19"): "hg38ToHg19.over.chain.gz",
               ("hg19", "hg38"): "hg19ToHg38.over.chain.gz"}

_CHAINS: dict[str, dict] = {}                              # chain path -> CrossMap mapping tree
_LATTICE: dict[tuple, np.ndarray] = {}                     # (path, chrom, step, max_off) -> float target of k*step (NaN = none)


def clear_caches() -> None:
    """Forget the loaded chains and interior-point lattices (module-level, per process)."""
    _CHAINS.clear()
    _LATTICE.clear()


@dataclass
class LiftoverReport:
    from_build: str
    to_build: str
    n_in: int
    n_mapped: int
    n_unmapped: int
    unmapped_reasons: dict[str, int]
    bp_in: int
    bp_out: int
    max_walk_used_bp: int
    n_envelope_extended: int          # kept '-' strand segments widened by interior points > step_bp
    unmapped: pd.DataFrame = field(repr=False)

    @property
    def dropped_bp_frac(self) -> float:
        """Fraction of input bp in dropped segments."""
        if self.bp_in == 0 or len(self.unmapped) == 0:
            return 0.0
        return float((self.unmapped["end"] - self.unmapped["start"]).clip(lower=0).sum()) / self.bp_in

    def __str__(self) -> str:
        return (f"liftover {self.from_build}->{self.to_build}: {self.n_mapped}/{self.n_in} segments "
                f"mapped, {self.n_unmapped} dropped {self.unmapped_reasons}, "
                f"{self.dropped_bp_frac:.2%} of bp dropped, bp {self.bp_in}->{self.bp_out}, "
                f"max endpoint walk {self.max_walk_used_bp} bp, "
                f"{self.n_envelope_extended} envelope-extended (inversions)")


_CM: list = []                                             # cached cmmodule.utils


@contextmanager
def _root_logging_preserved():
    """Restore the root logger's level and handlers afterwards. CrossMap calls
    ``logging.basicConfig(level=INFO)`` on import, and its module-level ``logging.info`` in
    ``read_chain_file`` installs a handler implicitly when the root logger has none."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


def _cm_utils():
    """``cmmodule.utils`` (lazy import, root logger left untouched)."""
    if _CM:
        return _CM[0]
    try:
        with _root_logging_preserved():
            import cmmodule.utils as u
    except ImportError as e:
        raise ImportError("CrossMap is not installed in this environment (pip install CrossMap)") from e
    _CM.append(u)
    return u


def _chain_path(from_build: str, to_build: str, chain_dir: str) -> str:
    if (from_build, to_build) not in CHAIN_FILES:
        raise ValueError(f"unsupported liftover {from_build} -> {to_build}; supported: {sorted(CHAIN_FILES)}")
    path = os.path.join(chain_dir, CHAIN_FILES[(from_build, to_build)])
    if not os.path.exists(path):
        raise FileNotFoundError(f"chain file for {from_build} -> {to_build} not found: {path}")
    return path


def _load_chain(path: str) -> dict:
    if path not in _CHAINS:
        u = _cm_utils()
        with _root_logging_preserved():
            _CHAINS[path] = u.read_chain_file(path, print_table=False)[0]
    return _CHAINS[path]


def _lift_point(tree: dict, chrom: str, pos: int, max_offset: int) -> Optional[tuple[int, str]]:
    """First plausible hit (same chromosome, within ``max_offset`` of ``pos``) of base ``pos``,
    as (target_pos, chain strand '+' / '-'), or None."""
    r = _cm_utils().map_coordinates(tree, chrom, pos, pos + 1, "+")
    if r:
        for t in r[1::2]:
            if t[0] == chrom and abs(int(t[1]) - pos) <= max_offset:
                return int(t[1]), t[3]
    return None


def _walk(tree: dict, chrom: str, pos: int, direction: int, step: int, limit: int,
          max_offset: int) -> Optional[tuple[int, int, str]]:
    """First plausible hit walking from ``pos`` in ``direction``: ``pos`` itself, then the
    multiples of ``step`` beyond it (a fixed genome lattice, so every start inside one gap sees the
    same probes), while the distance walked is <= limit. Returns (target_pos, walked, strand) or None."""
    t = _lift_point(tree, chrom, pos, max_offset)
    if t is not None:
        return t[0], 0, t[1]
    p = (pos // step + 1) * step if direction > 0 else ((pos - 1) // step) * step
    while p >= 0 and abs(p - pos) <= limit:
        t = _lift_point(tree, chrom, p, max_offset)
        if t is not None:
            return t[0], abs(p - pos), t[1]
        p += direction * step
    return None


def _lattice(tree: dict, path: str, chrom: str, step: int, max_offset: int, upto: int) -> np.ndarray:
    """Targets of the lattice points k * step, k = 0 .. upto // step (NaN = no plausible hit)."""
    key = (path, chrom, step, max_offset)
    have = _LATTICE.get(key, np.zeros(0))
    need = upto // step + 1
    if len(have) < need:
        extra = [_lift_point(tree, chrom, k * step, max_offset) for k in range(len(have), need)]
        have = np.concatenate([have, np.array([np.nan if t is None else t[0] for t in extra], float)])
        _LATTICE[key] = have
    return have


_I64_MAX = np.iinfo(np.int64).max


def _coord(col: pd.Series, name: str) -> np.ndarray:
    """One coordinate column as int64: integer dtypes, or floats that are finite and exactly integral."""
    try:
        v = pd.to_numeric(col, errors="coerce")
        if pd.api.types.is_integer_dtype(v):
            if len(v) and int(v.max()) > _I64_MAX:
                raise OverflowError
            return v.to_numpy(np.int64)
        f = v.to_numpy(np.float64)
    except OverflowError as e:
        raise ValueError(f"{name} coordinates outside the int64 range") from e
    bad = ~np.isfinite(f)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} segments have a NaN / non-finite / non-numeric {name}")
    big = np.abs(f) >= 2.0 ** 63
    if big.any():
        raise ValueError(f"{int(big.sum())} segments have a {name} outside the int64 range")
    frac = f != np.floor(f)
    if frac.any():
        raise ValueError(f"{int(frac.sum())} segments have a non-integer {name}")
    return f.astype(np.int64)


def _coords(segs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validated int64 ``start``/``end`` (ValueError on NaN, non-finite, non-integral, out of range
    or negative start). ``end <= start`` is not an error here: such rows are dropped as 'empty'."""
    st, en = _coord(segs["start"], "start"), _coord(segs["end"], "end")
    if (st < 0).any():
        raise ValueError(f"{int((st < 0).sum())} segments have start < 0")
    return st, en


def lift_segments(segs: pd.DataFrame, from_build: str, to_build: str, chain_dir: str = CHAIN_DIR,
                  step_bp: int = 10_000, max_walk_bp: Optional[int] = None, *,
                  max_offset_bp: int = 10_000_000, sample_step_bp: int = 100_000,
                  gate_window_bp: int = 1_000_000,
                  max_len_change_frac: float = 0.5, max_len_change_bp: int = 1_000_000
                  ) -> tuple[pd.DataFrame, LiftoverReport]:
    """Lift segments ``from_build`` -> ``to_build`` (rules in the module docstring).

    Returns the surviving rows (input order and index kept, ``start``/``end`` replaced, other
    columns unchanged) and a :class:`LiftoverReport`. When the builds are equal, valid rows are
    returned unchanged (rows with ``end <= start`` are still dropped as 'empty').
    ``max_walk_bp=None`` lets an endpoint walk up to the other end of its segment.
    """
    fb, tb = normalise_build(from_build), normalise_build(to_build)
    start, end = _coords(segs)
    bp_in = int(np.maximum(end - start, 0).sum())
    if fb == tb:
        ok = end > start
        unmapped = segs.loc[~ok].assign(reason="empty")
        reasons = {"empty": int((~ok).sum())} if (~ok).any() else {}
        return segs.loc[ok].copy(), LiftoverReport(fb, tb, len(segs), int(ok.sum()), int((~ok).sum()),
                                                   reasons, bp_in, bp_in, 0, 0, unmapped)
    path = _chain_path(fb, tb, chain_dir)
    tree = _load_chain(path)

    chrom = segs["chrom"].astype(str).to_numpy()
    n = len(segs)
    seg_len = end - start
    nonempty = seg_len > 0
    cap = np.iinfo(np.int64).max if max_walk_bp is None else int(max_walk_bp)
    # one walk per unique endpoint, as far as the longest segment sharing it needs
    lift: dict[tuple, Optional[tuple[int, int, str]]] = {}
    for pos, d in ((start, 1), (end - 1, -1)):
        ep = pd.DataFrame({"c": chrom[nonempty], "p": pos[nonempty], "L": seg_len[nonempty]})
        for (c, p), L in ep.groupby(["c", "p"], sort=False)["L"].max().items():
            lift[(c, int(p), d)] = _walk(tree, c, int(p), d, step_bp, int(min(cap, L - 1)), max_offset_bp)
    lat = {}
    if sample_step_bp > 0 and nonempty.any():
        for c in np.unique(chrom[nonempty]):
            t = _lattice(tree, path, c, sample_step_bp, max_offset_bp, int(end[nonempty & (chrom == c)].max()))
            lat[c] = (np.where(np.isnan(t), np.inf, t), np.where(np.isnan(t), -np.inf, t))

    new_s = np.full(n, -1, np.int64)
    new_e = np.full(n, -1, np.int64)
    reason = np.full(n, "", dtype=object)
    max_walk_used, n_ext = 0, 0
    for i in range(n):
        s0, e0, L = int(start[i]), int(end[i]), int(seg_len[i])
        if L <= 0:
            reason[i] = "empty"
            continue
        a, b = lift[(chrom[i], s0, 1)], lift[(chrom[i], e0 - 1, -1)]
        if a is None or a[1] >= L:
            reason[i] = "start_unmapped"
            continue
        if b is None or b[1] >= L:
            reason[i] = "end_unmapped"
            continue
        widened = False
        if a[2] == "+" and b[2] == "+":
            lo, hi = a[0], b[0]
            if lat and gate_window_bp > 0:   # local non-collinearity gate, within gate_window_bp of each end
                la, ha = lat[chrom[i]]
                k0, k1 = s0 // sample_step_bp + 1, (min(e0, s0 + gate_window_bp) - 1) // sample_step_bp
                if k1 >= k0 and la[k0:k1 + 1].min() < a[0] - step_bp:
                    lo, widened = int(la[k0:k1 + 1].min()), True
                k0, k1 = max(s0, e0 - gate_window_bp) // sample_step_bp + 1, (e0 - 1) // sample_step_bp
                if k1 >= k0 and ha[k0:k1 + 1].max() > b[0] + step_bp:
                    hi, widened = int(ha[k0:k1 + 1].max()), True
        else:                         # an endpoint sits in an inverted block: envelope of everything
            lo, hi = min(a[0], b[0]), max(a[0], b[0])
            if lat:
                k0, k1 = s0 // sample_step_bp + 1, (e0 - 1) // sample_step_bp   # points in (s0, e0)
                if k1 >= k0:
                    lo = int(min(lo, lat[chrom[i]][0][k0:k1 + 1].min()))
                    hi = int(max(hi, lat[chrom[i]][1][k0:k1 + 1].max()))
            widened = (min(a[0], b[0]) - lo > step_bp) or (hi - max(a[0], b[0]) > step_bp)
        s1, e1 = max(lo, 0), hi + 1
        if s1 >= e1:
            reason[i] = "inverted"
        elif e1 - s1 < L * (1 - max_len_change_frac) or \
                e1 - s1 > L * (1 + max_len_change_frac) + max_len_change_bp:
            reason[i] = "length_changed"
        else:
            new_s[i], new_e[i] = s1, e1
            max_walk_used = max(max_walk_used, a[1], b[1])
            n_ext += widened

    ok = reason == ""
    out = segs.loc[ok].copy()
    out["start"] = new_s[ok]
    out["end"] = new_e[ok]
    unmapped = segs.loc[~ok].copy()
    unmapped["reason"] = reason[~ok].astype(str)
    reasons = {str(k): int(v) for k, v in unmapped["reason"].value_counts().items()}
    rep = LiftoverReport(fb, tb, n, int(ok.sum()), int((~ok).sum()), reasons, bp_in,
                         int((new_e[ok] - new_s[ok]).sum()), int(max_walk_used), int(n_ext), unmapped)
    return out, rep


def assert_build(declared: str, grid: Grid) -> str:
    """Normalised ``declared`` if it equals ``grid.genome_build``, else GenomeBuildMismatch."""
    b = normalise_build(declared)
    if b != grid.genome_build:
        raise GenomeBuildMismatch(
            f"segments are {b} but the grid is {grid.genome_build}; binning them as-is would shift "
            f"blocks. Pass the segments through segments_to_grid_build (ppc.liftover) first.")
    return b


def segments_to_grid_build(segs: pd.DataFrame, declared_build: str, grid: Grid, *,
                           allow_liftover: bool = True, **lift_kwargs
                           ) -> tuple[pd.DataFrame, Optional[LiftoverReport]]:
    """Return ``segs`` in ``grid.genome_build`` coordinates.

    Same build -> ``(segs, None)`` unchanged. Different build -> lifted with :func:`lift_segments`
    (INFO log line; WARNING if more than 1 % of the bp is dropped), or GenomeBuildMismatch when
    ``allow_liftover`` is False.
    """
    b = normalise_build(declared_build)
    if b == grid.genome_build:
        return segs, None
    if not allow_liftover:
        assert_build(b, grid)
    out, rep = lift_segments(segs, b, grid.genome_build, **lift_kwargs)
    log.info("%s", rep)
    if rep.dropped_bp_frac > 0.01:
        log.warning("liftover %s->%s dropped %.2f%% of segment bp (%d segments: %s)", b,
                    grid.genome_build, 100 * rep.dropped_bp_frac, rep.n_unmapped, rep.unmapped_reasons)
    return out, rep
