"""Region-selection SISTEM simulator (FittedRegionLibrary, 5 Mb blocks).

Simulates tumours with SISTEM 1.0.5 under the region (5 Mb block) selection
model, with a sparse spike-and-slab prior on the per-block selection
coefficients, and writes per-clone block copy numbers plus clone sizes for
NPE training. Everything not listed below is kept identical to the arm-model
script simulate_arm_selection.py (MPI layout, directory structure, retry loop,
seeding scheme), so the two files can be diffed side by side.

Prior, version 4.1 = PRIOR_VERSION 5 (2026-10-26; research_notes/v4_1_design_2026-10-26/DECISION.md, scope revised
after the design review):
    Version 4 below with three changes, same draws in the same order: PASS_TRUNK_GENS 2000 -> 1000;
    PASS_RATE_RANGE logU(5e-4, 1.5e-2) -> logU(1e-4, 3e-2); the single copula weight RHO_CIN is replaced by one
    loading per rate, RHO_CIN_LOADINGS (final, from the factorial and the 5e6 pilot: arm 0.5, chromosomal 0.4,
    focal 0.8, pass 0.8). Pairing: for the same prior seed delta, the driver set, pi, q, m, length_mean, p_term,
    cin_level, u_pass, the *_indep rates and (same loading) arm_rate are bit-identical to v4; chromosomal_rate,
    focal_driver_rate and pass_rate differ. V4_SETTINGS restores v4
    exactly; V3_SETTINGS restores v3 exactly. The pi range, the driver set rules and the D-J overlay are v4's.

Prior, version 4 (2026-10-23, Stage 1 provisional; one draw per theta id, rng = default_rng(base_seed + sim_id*1000)):
    Same draw order and count as v3 through p_term, then two NEW draws: cin_level L ~ U(0,1), u_pass ~ U(0,1).
    arm_rate ~ logU(3e-6, 3e-4) [v3 1e-5]; chromosomal_rate ~ logU(3e-7, 1e-4) [v3 1e-6];
    focal_driver_rate ~ logU(3e-5, 1e-3) [v3 1e-4]; p_term ~ U(0.25, 0.75) [v3 0.35]; everything else as v3.
    Shared instability level (Gaussian copula, RHO_CIN = 0.5): for each of arm/chrom/focal rate,
    u = (log r - log lo)/(log hi - log lo), z' = sqrt(RHO)*Phi^-1(L) + sqrt(1-RHO)*Phi^-1(u),
    rate = exp(log lo + Phi(z')*(log hi - log lo)) (marginals stay log-uniform; rates correlated; length_mean
    untouched). pass_rate ~ the same copula on u_pass over PASS_RATE_RANGE = logU(5e-4, 1.5e-2) passenger
    events per lineage per generation (cin_copula_quantile). With RHO_CIN = 0 the realised rates ARE the drawn
    ones. theta keeps the pre-copula draws as *_indep. Passengers: deviation D-J below.
    Pairing: with V3_RANGES restored and RHO_CIN = 0 (since v4.1: V3_SETTINGS, all loadings 0), every v3 field
    equals v3's for the same prior seed;
    with the v4 floors, block positions, driver set, |log(1+delta)|, q, pi, m, length_mean, the uniform
    quantile of p_term and the uniform quantiles of the PRE-copula *_indep rates are paired. Under RHO_CIN > 0
    the realised rates are NOT paired with v3 (they also depend on cin_level).

Prior, version 3 (2026-10-22 to 2026-10-23; one draw per theta id, rng = default_rng(base_seed + sim_id*1000)):
    event rates and the focal length are LOG-uniform (equal mass per decade):
    arm_rate ~ logU(1e-5, 3e-4);  chromosomal_rate ~ logU(1e-6, 1e-4)
    focal_driver_rate ~ logU(1e-4, 1e-3);  length_mean ~ logU(1, 8) blocks (mean focal length)
    (prior version 1 used arm U(1e-5,1e-4), chrom U(1e-6,1e-5), fixed focal 5e-4 and
    length_mean 1.5; revised 2026-10-20 after the step-1 change-size comparison: simulated
    tumours had too few arm events, no partial-arm segments and too little altered genome)
    pi ~ U(0.02, 0.12)  (driver fraction);  q ~ U(0.30, 0.55)  (P(positive))
    m ~ U(0.05, 0.15)  (mean effect size on the log scale)
    per block: z ~ Bernoulli(pi)  (every block, including the last one of each chromosome)
    z == 0 -> delta = 0.0;  z == 1 -> eta = min(Exp(mean=m), 0.6), sign = +1 w.p. q
    delta = exp(sign*eta) - 1, so delta lies in [exp(-0.6)-1, exp(0.6)-1] ≈ [-0.4512, 0.8221]
    p_term ~ U(0.35, 0.75), drawn LAST in v3 (after the blocks): share of driver focal events made
    end-anchored by patch D-H. A nuisance parameter, not an inference target.
    (prior version 2, 2026-10-20 to 2026-10-22: q ~ U(0.40, 0.65), focal_driver_rate ~
    logU(1e-4, 2e-3), no p_term. Version 3 adopted 2026-10-22 together with patches D-H and D-I;
    design, sweep and verification in research_notes/v3_design_2026-10-21/. Because p_term is
    the last draw, block positions, the driver set and eta = |log(1+delta)| are identical to
    version 2 for the same prior seed; only signs whose uniform falls between the two q
    values differ.)

Output contract (under --output):
    manifest.json                              written once by rank 0
    simulation_outputs/sim{ID}/parameters.pkl  theta + library facts, per theta
    simulation_outputs/sim{ID}/{t}/results.pkl always written; bookkeeping only
                                               (failures, complete, n_clones, mismatches, wall time)
    simulation_outputs/sim{ID}/{t}/clones.npz  only when the trial completed; all observational data:
        block_cn int16 (n_clones, D), popsize int64, ploidy float32,
        fitness float64, nWGD int16, n_altered_drivers int16,
        bulk float32 (D,) = popsize-weighted mean of block_cn.
        Since prior v4, block_cn, bulk and ploidy are AFTER the passenger overlay (D-J); fitness and
        n_altered_drivers are as at the end of growth (pre-overlay; edge passengers are neutral);
        n_altered_drivers_post int16 applies the same rule to the post-overlay block_cn.
        event_counts int32 (n_clones, 9): CNA events along each clone's lineage back to the
            founder, per category in EVENT_CATEGORIES order (7 growth + passenger_gain/loss); events
            int32 (n_events, 11): the unique events of all surviving lineages, growth rows first then
            passenger rows (columns in EVENT_COLUMNS, also stored as event_columns; events_schema_version
            = EVENTS_SCHEMA_VERSION = 3; schema 1 (prior v2) had 9 columns, schema 2 (prior v3) 10 without
            'node', so readers index columns by name).
            Growth rows are recovered from SISTEM's own per-clone event records (clone.events,
            parent links survive clone death), so these are true counts, not inferred ones.
        node_parent, node_birth_gen, node_clone int32 (n_nodes,), clone_node int32 (n_clones,): the
            lineage genealogy (surviving clones + all ancestors; node order (birth_gen, creation order);
            see passenger_overlay.py). events[:, 'node'] refers to these node ids.
        The largest clone is argmax(popsize); it is not stored separately.
    simulation_outputs/sim{ID}/{t}/sim.log     written by SISTEM itself (absent with --no-sim-log)
Block vectors are flattened in CHROM_ORDER (chr1..chr22, numeric order).

Approved deviations from the user's spec:
    D-A  per-clone arrays live in clones.npz (compressed), not in results.pkl.
    D-B  the arm-average check (arm_CNratios_single cannot run under the region
         library) is replaced by a driver_counts cross-check; arm-average
         recovery from block_cn is done by check_gene_selection_outputs.py.
    D-C  mpi4py is optional: without it the script runs as a single process.
    D-D  SISTEM's BaseRegionLibrary.get_driver_start_regions is patched at import (see
         apply_sistem_patches): upstream scans sequence[:-size], so no focal event could ever
         cover the last position of a chromosome and the last block was forced neutral in the
         prior as a workaround. With the fix the last block is a normal driver candidate.
         Recorded in manifest.json under "sistem_patches".
    D-E  BaseRegionLibrary.compute_fitness is patched to use the ploidy ROUNDED to the nearest
         integer in the exponent, (1+delta)^(n/round(p)), as in CINner (Dinh et al. 2025, S1 Eq 1-2),
         the model SISTEM's region model was adapted from. With the continuous p, deleting neutral
         DNA raised fitness and tumours collapsed to the 1.5 ploidy floor (calibration 2026-10-19:
         median ploidy 1.51, passengers at 1.4 copies).
    D-F  BaseRegionLibrary.check_viability is patched so a clone with any block at 0 copies is
         non-viable (CINner nullisomy_max = 0; SISTEM has no minimum copy number). Together D-E and
         D-F hold ploidy at ~2.2 with no empty blocks (research report: reports/Fixing the ploidy
         collapse in SISTEM.md).
    D-G  Cell.focal_deletion records the deleted block ids (captured before deleting). Bookkeeping only.
    D-I  (prior v3; supersedes the D-E body, keeps its round(p)) BaseRegionLibrary.compute_fitness
         saturates the dosage of GAIN drivers: fitness = prod_r (1+delta_r)^(n_eff/ref), ref =
         max(1, round(p)), n_eff = min(n_r, ref + GAIN_SAT_EXTRA) if delta_r > 0 else n_r. Copies
         above ref+1 are neutral, not lethal. Without it, strong gain drivers ratcheted up to
         MAX_REGION_CN and dragged ploidy up (v3 sweep A2 vs A0: ploidy -0.23, cn_frac_5plus 0.039 -> 0).
    D-H  (prior v3) sistem.lineage.mutation.gen_focal_event is wrapped: with probability p_term a
         DRIVER focal event becomes end-anchored (homolog end -> target driver + floor(Exp(2)) blocks
         toward the centromere, strictly inside the arm), adding TERMINAL_MAG = 1 copy or deleting;
         the record gets terminal = True. Passenger calls (driver=False) and p_term = 0 delegate to
         the original with no extra RNG draw. Adopted only together with D-I (new event class;
         advisor sign-off pending).
    D-J  (prior v4) passenger CNAs, NOT a SISTEM patch, in two parts (driver-blind windows, same sampler):
         (1) TRUNK, before growth: Poisson(pass_rate * PASS_TRUNK_GENS) events put on the FOUNDER genome with
             SISTEM's own Clone methods (grow(); every clone inherits them; kept only if the patched
             check_viability passes). They change the founder genome, so with PASS_TRUNK_GENS > 0 growth is a
             DIFFERENT trajectory from the passenger-free run (identical only with PASS_TRUNK_GENS = 0). They are
             NOT neutral: a trunk window covering a driver block changes the founder's driver_counts, so it
             enters fitness, the altered-driver cap and the gain-saturation headroom (realistic for truncal
             events). Lead decision 2026-10-23: keep it; trunk size is a growth-arm dimension of the sweep.
         (2) EDGE overlay, after growth (passenger_overlay.py): Poisson(pass_rate * edge length) +-1 copy events
             on every genealogy edge, applied to all surviving descendants in block_cn space; neutral by
             construction (fitness and driver bookkeeping are not touched; rejected if a block would hit 0 copies,
             a driver block would exceed MAX_REGION_CN, the altered-driver cap would be exceeded, or ploidy would
             leave [MIN_PLOIDY, MAX_PLOIDY]). It never alters growth. Known artefact (kept in v4.1): one
             1-copy descendant vetoes a loss, so deep edges take mostly gains (v4.1 diagnostics/REPORT.md Q4.1).
         RNGs: the global NumPy stream drives growth only; trunk and overlay use their own Generators,
         default_rng([trial_seed, OVERLAY_SEED_OFFSET, 0]) and default_rng([trial_seed, OVERLAY_SEED_OFFSET, 1])
         (seed sequences: no collision across trials, unlike additive offsets). grow() = growth (+ trunk);
         simulate_gene_model_caller = grow + overlay + packing. PASSENGERS_ENABLED = False skips both.
         Growth-time insertion was rejected on cost (REVIEW.md P1).

See docs/GENE_LEVEL_SIMULATION_PLAN.md and Notion 3.2.1 / 3.3.
"""
import os
import sys
import copy
import json
import time
import pickle as pkl
import argparse
import datetime
import subprocess

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Union

import numpy as np
import torch
from scipy.special import ndtr, ndtri      # standard normal CDF / inverse CDF (what scipy.stats.norm.cdf/ppf call)

# D-J passenger overlay (pure numpy). Imported relative to the package when this file is loaded as
# data_generation.simulate_gene_selection, else top-level (script run, tests, harness), so the module
# object is the same one the caller imports alongside us.
if __package__:
    from . import passenger_overlay as po
else:
    import passenger_overlay as po

# --- MPI detection / helpers (deviation D-C: mpi4py is optional) ---

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    USING_MPI = True
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1
    USING_MPI = False


def is_mpi_active() -> bool:
    """True iff mpi4py imported AND world size > 1."""
    return USING_MPI and SIZE > 1


def is_slurm_multi_alloc() -> bool:
    """True iff SLURM allocated >1 tasks."""
    try:
        return int(os.environ.get("SLURM_NTASKS", "1")) > 1
    except ValueError:
        return False


def require_mpi_or_exit():
    """
    If SLURM allocated multiple tasks but MPI world==1, print a clear error and exit.
    Prevents silent 'rank 0 everywhere' behavior (every task would redo all theta ids).
    """
    if is_slurm_multi_alloc() and not is_mpi_active():
        if RANK == 0:  # single process anyway
            print(
                "[ERROR] SLURM_NTASKS indicates a multi-task job, "
                "but MPI world size == 1.\n"
                "This means mpi4py is missing or not wired to the launcher.\n"
                "Use either:\n"
                "  srun python simulate_gene_selection.py  (optionally with --mpi=pmix or pmix_v3),\n"
                "or\n"
                "  mpirun -np $SLURM_NTASKS python simulate_gene_selection.py\n"
                "and ensure mpi4py was built against the loaded OpenMPI.",
                file=sys.stderr, flush=True,
            )
        sys.exit(2)


def mpi_print(msg: str):
    """Rank-tagged printing; flush to avoid buffering in MPI runs."""
    print(f"[rank {RANK}/{SIZE}] {msg}", flush=True)


import sistem
from sistem import Parameters
from sistem.genome.genome import init_diploid_genome
from sistem.genome.utils import get_num_regions, hg38_chrom_lengths_from_cytoband
from sistem.lineage.cell import Cell, Clone
from sistem.lineage.mutation import CNA
import sistem.lineage.mutation as mutation      # D-H hooks mutation.gen_focal_event (module global, read at call time)
import sistem.utilities.utilities as sistem_utilities   # MBIT is saved/restored around the D-H self-test
from sistem.ancestry import GrowthSimulator
from sistem.anatomy import SimpleAnatomy
from sistem.data.utils import count_region_CNs
from sistem.selection.region_library import FittedRegionLibrary, BaseRegionLibrary
from sistem.utilities.utilities import get_reg_id, init_mbit


# ----------------- SISTEM patches (deviation D-D) -----------------
SISTEM_PATCHES = [                                          # recorded in manifest.json
    "region_driver_start_regions_tail_fix",                 # D-D
    "region_fitness_rounded_ploidy",                        # D-E (superseded since prior v3: its round(p) lives on inside D-I)
    "region_nullisomy_lethal",                              # D-F
    "focal_deletion_gene_ids_record_fix",                   # D-G (bookkeeping only)
    "region_gain_dosage_saturation",                        # D-I (prior v3; replaces the D-E body)
    "focal_terminal_events",                                # D-H (prior v3)
]

# Patch constants (prior v3). Defined here because the patches read them at call time and the
# import-time self-tests run before the rest of the constants below.
GAIN_SAT_EXTRA = 1          # D-I: gain drivers saturate at round(p) + GAIN_SAT_EXTRA copies (K = 1)
TERMINAL_MAG = 1            # D-H: a terminal gain adds exactly one tandem copy
TERM_EXT_MEAN = 2.0         # D-H: blocks beyond the target driver toward the centromere, floor(Exp(mean))
_P_TERM = 0.0               # D-H: per-theta p_term, set by simulate_gene_model per trial (try/finally), else 0
TERMINAL_WINDOW_RULE = (
    "with prob p_term a driver focal event becomes end-anchored: homolog length-weighted as upstream; "
    "target driver uniform among feasible drivers; side = target position vs current cent; "
    "window = homolog end -> target + floor(Exp(TERM_EXT_MEAN)) blocks toward the centromere, clipped to "
    "stay strictly inside the arm (never whole arm, never crosses cent); acentric homologs: either end "
    "with prob 1/2, may be centromere-bounded; fallback to the interstitial event if no feasible window")


def _fixed_get_driver_start_regions(self, cell, chromosome, size):
    """Corrected BaseRegionLibrary.get_driver_start_regions (SISTEM 1.0.3/1.0.5, region_library.py:144).

    Upstream iterates ``sequence[:-size]``: the last ``size`` positions of a homolog are never
    examined, so no focal driver event can cover the LAST position, and a driver at position p
    only attracts events of size <= len-1-p. (The passenger version right below it already uses
    the correct bound ``len - size + 1``.) This version examines every position and bounds each
    start so the event still fits: start in [max(0, i-size+1), min(i, len-size)]. Only the loop
    bounds change; the raw ``r in targets`` test is upstream's.
    """
    chrname = chromosome.name
    seq = chromosome.seq
    n = len(seq)
    targets = set(self.drivers[chrname]) | set(self.essential[chrname])
    possible_start = set()
    for i, r in enumerate(seq):
        if r in targets:
            lo = max(0, i - size + 1)
            hi = min(i, n - size)
            for j in range(lo, hi + 1):
                possible_start.add(j)
    return list(possible_start)


_ORIG_CHECK_VIABILITY = BaseRegionLibrary.check_viability
# Reload safety (fixed 2026-10-23): after importlib.reload the class already holds our D-F wrapper, whose
# _ORIG_CHECK_VIABILITY would resolve to itself (same module dict) and recurse forever. Unwrap it.
_ORIG_CHECK_VIABILITY = getattr(_ORIG_CHECK_VIABILITY, "_df_original", _ORIG_CHECK_VIABILITY)
assert getattr(_ORIG_CHECK_VIABILITY, "__module__", None) == "sistem.selection.region_library", \
    f"D-F: BaseRegionLibrary.check_viability is not SISTEM's ({_ORIG_CHECK_VIABILITY!r})"
DELETION_EVENTS = ("focal_deletion", "arm_deletion", "chromosomal_deletion")


def _rounded_ploidy(p):
    """CINner's reference copy number: the mean CN rounded to the nearest integer (Python round is
    half-to-even, like R's). Never below 1."""
    return max(1, round(p))


def _rounded_compute_fitness(self, clone):
    """Deviation D-E: BaseRegionLibrary.compute_fitness (region_library.py:28) with the exponent
    n / round(p) instead of n / p.

    SISTEM's region model is adapted from CINner, whose fitness is prod lambda_r^(c_r / c) with c the
    mean copy number ROUNDED to the nearest integer (Dinh et al. 2025, S1 Eq 1-2); SISTEM uses the
    continuous ploidy. With the continuous value every deleted passenger block lowers p and raises all
    driver exponents, so shedding neutral DNA is selected once S = sum n_r log(1+delta_r) > 0; tumours
    then sink to the min_ploidy floor. With rounding, small losses do not change the exponent, and a
    doubled genome (p ~ 4) stays fitness-neutral exactly as before. init_base_fit needs no change
    (diploid: round(2) = 2); init_max_fit keeps the continuous p (it only feeds metastatic growth
    curves, unused with one site)."""
    ref = _rounded_ploidy(clone.get_ploidy())
    fitness = 1
    for chrname in self.regions:
        for r in self.drivers[chrname]:
            s = self[chrname][r]
            n = clone.attributes.driver_counts[chrname][r]
            fitness *= (1 + s) ** (n / ref)
    return fitness


def _saturated_compute_fitness(self, clone):
    """Deviation D-I (prior v3): D-E with sign-specific dosage saturation for GAIN drivers.

    fitness = prod over driver blocks r of (1+delta_r)^(n_eff/ref), ref = max(1, round(p)) as in D-E,
    n_eff = min(n_r, ref + GAIN_SAT_EXTRA) if delta_r > 0 else n_r. Loss drivers are exactly D-E.
    Copies above ref + GAIN_SAT_EXTRA are neutral, not lethal (viability rules D-F and
    MAX_REGION_CN are unchanged). Without saturation the per-copy benefit had no bound, so strong
    gain drivers ratcheted up to the cap and dragged hitchhikers (and ploidy) with them.
    Ported from research_notes/v3_design_2026-10-21/v3_common.py (_sat_compute_fitness, K = 1)."""
    ref = _rounded_ploidy(clone.get_ploidy())
    sat = ref + GAIN_SAT_EXTRA
    fitness = 1
    for chrname in self.regions:
        for r in self.drivers[chrname]:
            s = self[chrname][r]
            n = clone.attributes.driver_counts[chrname][r]
            if s > 0 and n > sat:
                n = sat
            fitness *= (1 + s) ** (n / ref)
    return fitness


def _selftest_DI():
    """D-I import-time self-test on a one-driver fake library (no RNG)."""
    import types

    class _FakeLib1:
        regions = {"chrT": 6}
        drivers = {"chrT": [1]}
        def __init__(self, s): self.s = s
        def __getitem__(self, chrname): return {1: self.s, 3: 0.0}

    def fake_clone(p, n, n_nondriver=2):
        return types.SimpleNamespace(get_ploidy=lambda p=p: p,
                                     attributes=types.SimpleNamespace(driver_counts={"chrT": {1: n, 3: n_nondriver}}))

    f = lambda s, p, n: _saturated_compute_fitness(_FakeLib1(s), fake_clone(p, n))
    f_de = lambda s, p, n: _rounded_compute_fitness(_FakeLib1(s), fake_clone(p, n))
    for s in (0.05, 0.1, 0.5, 0.8):
        for p, ref in ((2.0, 2), (1.93, 2), (3.0, 3), (2.6, 3)):
            sat = ref + GAIN_SAT_EXTRA
            assert abs(f(s, p, 10) - f(s, p, sat)) < 1e-12, f"D-I: gain n=10 != n={sat} at ref {ref}"
            for n in range(sat + 1, 13):
                assert f(s, p, n) == f(s, p, sat), f"D-I: gain driver not saturated at n={n}, ref {ref}"
            for n in range(0, sat):
                assert f(s, p, n) < f(s, p, n + 1), f"D-I: not strictly increasing at n={n}, ref {ref}"
            for n in range(0, sat + 1):
                assert f(s, p, n) == f_de(s, p, n), f"D-I: gain driver below saturation differs from D-E at n={n}"
    if GAIN_SAT_EXTRA == 1:                         # the adopted K: ref 2 saturates at 3, ref 3 at 4
        assert f(0.5, 2.0, 10) == f(0.5, 2.0, 3) and f(0.5, 3.0, 10) == f(0.5, 3.0, 4)
    for s in (-0.05, -0.1, -0.4):
        for p in (1.5, 2.0, 2.6, 3.0, 4.2):
            for n in range(0, 13):
                assert f(s, p, n) == f_de(s, p, n), f"D-I: loss driver changed at p={p}, n={n}"
    # non-drivers have no effect: block 3 is in driver_counts but not in drivers
    for nn in (0, 2, 7, 15):
        assert _saturated_compute_fitness(_FakeLib1(0.5), fake_clone(2.0, 3, nn)) == f(0.5, 2.0, 3), \
            "D-I: a non-driver block changed fitness"
    no_drv = types.SimpleNamespace(regions={"chrT": 6}, drivers={"chrT": []})
    assert _saturated_compute_fitness(no_drv, fake_clone(2.0, 9)) == 1, "D-I: fitness without drivers != 1"


# ----------------- D-H: terminal (end-anchored) driver focal events -----------------
_ORIG_GEN_FOCAL = None      # upstream mutation.gen_focal_event, captured once by install_DH


def _feasible(t, n, cent):
    """Feasible sides for target position t on a homolog of length n with current centromere cent.
    p-side window [0, e) needs t+1 <= e <= cent-1; q-side [b, n) needs cent+1 <= b <= t; acentric
    homologs (cent None) allow either end, strictly inside the homolog."""
    if cent is not None:
        if t < cent:
            return ['p'] if t + 1 <= cent - 1 else []
        return ['q'] if t >= cent + 1 else []
    sides = []
    if t + 1 <= n - 1: sides.append('p')
    if t >= 1: sides.append('q')
    return sides


def _terminal_window(chromosome, targets):
    """(start, end, t) of an end-anchored window on this homolog, or None if no target is feasible.
    RNG: randint (target), [random (side, acentric with 2 feasible sides)], exponential (extension).
    Side is by POSITION relative to the CURRENT cent (tandem gains shift cent), not by block id."""
    seq, n, cent = chromosome.seq, len(chromosome.seq), chromosome.cent
    P = [i for i, r in enumerate(seq) if r in targets]
    feas = [(t, _feasible(t, n, cent)) for t in P]
    feas = [(t, s) for t, s in feas if s]
    if not feas:
        return None
    t, sides = feas[np.random.randint(len(feas))]
    side = sides[0] if len(sides) == 1 else ('p' if np.random.random() < 0.5 else 'q')
    k = int(np.floor(np.random.exponential(TERM_EXT_MEAN)))
    if side == 'p':
        room = (cent - 1 - (t + 1)) if cent is not None else (n - 1 - (t + 1))
        e = t + 1 + min(k, max(room, 0))
        return 0, e, t
    room = (t - (cent + 1)) if cent is not None else (t - 1)
    b = t - min(k, max(room, 0))
    return b, n, t


def _gen_focal_event_v3(clone, focal_gain_rate, length_mean, mag_mean, driver=True):
    """Deviation D-H: wrapper on sistem.lineage.mutation.gen_focal_event (same signature).

    driver=False (reevolve passengers) or _P_TERM <= 0: the original, with NO extra RNG draw.
    Otherwise u = np.random.random(); u >= _P_TERM: the original (interstitial event). Else the
    terminal branch: homolog drawn length-weighted with get_random_chrom (exclude list, as upstream),
    target uniform among feasible drivers, window from the homolog end to the target plus
    floor(Exp(TERM_EXT_MEAN)) blocks toward the centromere, strictly inside the arm. Gains (prob
    focal_gain_rate) add TERMINAL_MAG copies; losses go through focal_deletion. Applied through the
    Clone methods only, so nregions, driver_counts, cent, D-F and D-G all work unchanged. No feasible
    window anywhere: fall back to the original, so the number of focal events is unchanged.
    Ported from research_notes/v3_design_2026-10-21/v3_common.py (audit counters dropped)."""
    if (not driver) or _P_TERM <= 0:
        return _ORIG_GEN_FOCAL(clone, focal_gain_rate, length_mean, mag_mean, driver=driver)
    u = np.random.random()
    if u >= _P_TERM:
        return _ORIG_GEN_FOCAL(clone, focal_gain_rate, length_mean, mag_mean, driver=driver)
    lib = clone.library
    exclude = []
    win = None
    while True:
        chromosome = clone.genome.get_random_chrom(exclude=exclude)
        if chromosome is None:
            break
        name = chromosome.name
        targets = set(lib.drivers[name]) | set(lib.essential[name])
        win = _terminal_window(chromosome, targets)
        if win is not None:
            break
        exclude.append(chromosome)
    if win is None:
        return _ORIG_GEN_FOCAL(clone, focal_gain_rate, length_mean, mag_mean, driver=driver)
    start, end, _t = win
    if np.random.binomial(1, focal_gain_rate):
        clone.focal_amplification(chromosome, start, end, TERMINAL_MAG)
    else:
        clone.focal_deletion(chromosome, start, end)
    clone.events[-1].terminal = True        # set after construction: CNA.__init__ rejects unknown kwargs


def _is_upstream_gen_focal(fn):
    return getattr(fn, "__module__", None) == "sistem.lineage.mutation" and getattr(fn, "__name__", None) == "gen_focal_event"


def install_DH():
    """Install the D-H wrapper on mutation.gen_focal_event. Idempotent: a second call is a no-op.
    A wrapper left by an earlier execution of THIS module (importlib.reload; same module globals)
    is unwrapped first. Any other non-upstream function on the hook is an unknown wrapper: assert."""
    global _ORIG_GEN_FOCAL
    cur = mutation.gen_focal_event
    if cur is _gen_focal_event_v3:
        return
    if getattr(cur, "__globals__", None) is globals() and hasattr(cur, "_dh_original"):
        cur = cur._dh_original                  # reload: unwrap our own earlier wrapper
    assert _is_upstream_gen_focal(cur), \
        f"D-H: mutation.gen_focal_event is an unknown wrapper ({cur!r}); refusing to wrap it"
    assert _ORIG_GEN_FOCAL is None or _ORIG_GEN_FOCAL is cur, "D-H: upstream gen_focal_event changed identity"
    _ORIG_GEN_FOCAL = cur
    _gen_focal_event_v3._dh_original = cur      # lets a reload of this module find the original again
    mutation.gen_focal_event = _gen_focal_event_v3


class _StubChr:
    def __init__(self, n, cent, name='chrT'):
        self.name, self.allele, self.homolog_id = name, 0, 0
        self.seq, self.cent = list(range(n)), cent
    def __len__(self): return len(self.seq)


class _StubGenome:
    def __init__(self, ch): self.ch = ch
    def get_random_chrom(self, exclude=[]):
        return None if self.ch in exclude else self.ch


class _StubClone:
    def __init__(self, ch, drivers):
        import types
        self.genome = _StubGenome(ch)
        self.library = types.SimpleNamespace(drivers={ch.name: drivers}, essential={ch.name: []})
        self.events, self.calls = [], []
    def focal_amplification(self, chromosome, s, e, m):
        import types
        self.calls.append(('amp', s, e, m)); self.events.append(types.SimpleNamespace(category='focal_amplification'))
    def focal_deletion(self, chromosome, s, e):
        import types
        self.calls.append(('del', s, e)); self.events.append(types.SimpleNamespace(category='focal_deletion'))


def _dh_real_clone_factory(tmpdir):
    """A factory of fresh founder Clones on the real 5 Mb grid with a fixed driver layout (both signs,
    blocks next to the centromere and at the chromosome ends included). For self-tests (e) and (f)."""
    chrom_lens, arm_ratios, nregions = genome_grid()
    delta = {c: [(0.3 if (r // 3) % 2 == 0 else -0.2) if r % 3 == 0 or r == nregions[c] - 1 else 0.0
                 for r in range(nregions[c])] for c in CHROM_ORDER}
    lib = build_library(delta, chrom_lens, arm_ratios, out_dir=tmpdir)
    ig = init_diploid_genome(lib)
    return lambda: Clone(genome=copy.deepcopy(ig), library=lib, popsize=1)


def _selftest_DH():
    """D-H import-time self-tests (a)-(h) of the terminal_event_patch_spec, ported from v3_common.selftest_DH.
    (a)-(d), (g), (h) on stub homologs; (e), (f) on real Clones. The global NumPy RNG state and SISTEM's
    MBIT are restored afterwards, so importing this module has no RNG side effect."""
    import tempfile
    global _P_TERM, _ORIG_GEN_FOCAL
    assert mutation.gen_focal_event is _gen_focal_event_v3 and _is_upstream_gen_focal(_ORIG_GEN_FOCAL)
    saved_p, saved_orig = _P_TERM, _ORIG_GEN_FOCAL
    saved_rng, saved_mbit = np.random.get_state(), sistem_utilities.MBIT
    try:
        np.random.seed(20261022)
        _P_TERM = 1.0
        # (a) n=12 cent=6 driver 2 -> [0,e), 3<=e<=5; (h) terminal flag; gains add TERMINAL_MAG
        for _ in range(300):
            c = _StubClone(_StubChr(12, 6), [2]); _gen_focal_event_v3(c, 0.5, 2, 1.2)
            kind, s, e = c.calls[0][:3]
            assert s == 0 and 3 <= e <= 5 and s <= 2 < e, f"D-H (a) bad window {c.calls[0]}"
            assert c.events[-1].terminal is True, "D-H (h) terminal flag missing"
            if kind == 'amp':
                assert c.calls[0][3] == TERMINAL_MAG, "D-H: terminal gain magnitude != TERMINAL_MAG"
        # (b) driver 9 -> [b,12), 7<=b<=9
        for _ in range(300):
            c = _StubClone(_StubChr(12, 6), [9]); _gen_focal_event_v3(c, 0.5, 2, 1.2)
            kind, s, e = c.calls[0][:3]
            assert e == 12 and 7 <= s <= 9, f"D-H (b) bad window {c.calls[0]}"
        # (c) driver at cent-1 or cent -> infeasible -> fallback to the original
        for d in (5, 6):
            got = []
            _ORIG_GEN_FOCAL = lambda *a, **k: got.append(1)
            c = _StubClone(_StubChr(12, 6), [d]); _gen_focal_event_v3(c, 0.5, 2, 1.2)
            assert got == [1] and c.calls == [], f"D-H (c) driver {d} should fall back"
        _ORIG_GEN_FOCAL = saved_orig
        # (d) acentric: anchored at 0 or n, strictly inside
        for _ in range(300):
            c = _StubClone(_StubChr(12, None), [4]); _gen_focal_event_v3(c, 0.5, 2, 1.2)
            kind, s, e = c.calls[0][:3]
            assert (s == 0 and 5 <= e <= 11) or (e == 12 and 1 <= s <= 4), f"D-H (d) bad window {c.calls[0]}"
        # (g) driver=False delegates
        got = []
        _ORIG_GEN_FOCAL = lambda *a, **k: got.append(k.get('driver'))
        c = _StubClone(_StubChr(12, 6), [2]); _gen_focal_event_v3(c, 0.5, 2, 1.2, driver=False)
        assert got == [False] and c.calls == [], "D-H (g) driver=False did not delegate"
        _ORIG_GEN_FOCAL = saved_orig
        with tempfile.TemporaryDirectory(prefix="gene_sim_selftest_") as tmpdir:
            factory = _dh_real_clone_factory(tmpdir)
            # (e) 200 terminal events on a real clone: driver_counts / nregions / cent consistent
            cl = factory()
            for _ in range(200):
                _gen_focal_event_v3(cl, 0.5, 2, 1.2)
            assert any(getattr(ev, "terminal", False) for ev in cl.events), "D-H (e) no terminal event recorded"
            dc = copy.deepcopy(cl.attributes.driver_counts)
            cl.library.init_attributes(cl)
            assert dc == cl.attributes.driver_counts, "D-H (e) driver_counts drifted"
            assert cl.genome.nregions == sum(len(ch) for chn in cl.genome for ch in cl.genome[chn]), "D-H (e) nregions drifted"
            for chn in cl.genome:
                for ch in cl.genome[chn]:
                    assert ch.cent is None or 0 < ch.cent < len(ch), "D-H (e) cent out of range"
            # (f) with _P_TERM = 0 the RNG stream equals the original's after 1000 calls
            _P_TERM = 0.0
            c1, c2 = factory(), factory()
            np.random.seed(7); [_gen_focal_event_v3(c1, 0.5, 2, 1.2) for _ in range(1000)]; s1 = np.random.get_state()[1].copy()
            np.random.seed(7); [saved_orig(c2, 0.5, 2, 1.2) for _ in range(1000)]; s2 = np.random.get_state()[1].copy()
            assert np.array_equal(s1, s2), "D-H (f) RNG stream differs at p_term=0"
            assert c1.attributes.driver_counts == c2.attributes.driver_counts, "D-H (f) genomes differ at p_term=0"
            assert not any(getattr(ev, "terminal", False) for ev in c1.events), "D-H (f) terminal event at p_term=0"
    finally:
        _ORIG_GEN_FOCAL = saved_orig
        _P_TERM = saved_p
        np.random.set_state(saved_rng)
        sistem_utilities.MBIT = saved_mbit


def _has_cn0_block(clone):
    """True iff some block of this clone has total copy number 0 over all homologs.

    Only chromosomes named in the clone's OWN deletion events are recounted: a new mutant starts with
    an empty event list (Cell.__init__), its parent passed the same check, and only deletions can
    create a zero. This keeps the check cheap (it runs on every new mutant clone)."""
    chroms = {e.chrom for e in clone.events if getattr(e, "category", None) in DELETION_EVENTS}
    for chrname in chroms:
        nreg = clone.library.regions[chrname]
        cn = np.zeros(nreg, dtype=np.int64)
        for chromosome in clone.genome[chrname]:
            if len(chromosome.seq):
                cn += np.bincount([get_reg_id(x) for x in chromosome.seq], minlength=nreg)
        if (cn == 0).any():
            return True
    return False


def _nullisomy_check_viability(self, clone):
    """Deviation D-F: SISTEM's check_viability (driver CN <= max_region_CN, ploidy in [min, max],
    altered drivers <= cap) plus 'no block at 0 copies' (CINner nullisomy_max = 0). Every 5 Mb block
    carries essential genes, so homozygous loss of a whole block is lethal."""
    return bool(_ORIG_CHECK_VIABILITY(self, clone)) and not _has_cn0_block(clone)


_nullisomy_check_viability._df_original = _ORIG_CHECK_VIABILITY   # lets a reload of this module find SISTEM's again


def _fixed_focal_deletion(self, chromosome, start, end, record_event=True, driver=False):
    """Deviation D-G: Cell.focal_deletion (lineage/cell.py:47) records gene_ids = seq[start:end] AFTER
    the deletion, i.e. the blocks that moved into those positions (or nothing, at a chromosome end),
    not the deleted blocks. Capture the slice first. Bookkeeping only: the genome change is identical."""
    ids = list(chromosome.seq[start:end])
    chromosome.delete(start, end)
    if record_event:
        self.events.append(CNA(category="focal_deletion", chrom=chromosome.name, allele=chromosome.allele,
                               homolog_id=chromosome.homolog_id, start=start, end=end, gene_ids=ids,
                               driver=driver, gen=self.gen))


def apply_sistem_patches():
    """Install the corrected methods on BaseRegionLibrary (Random/Fitted/Hybrid inherit them) and on
    sistem.lineage.mutation, and self-test each one. Safe to call again (every install is idempotent)."""
    import types
    BaseRegionLibrary.get_driver_start_regions = _fixed_get_driver_start_regions
    # D-I replaces the D-E body (prior v3); _rounded_compute_fitness stays defined as its reference.
    BaseRegionLibrary.compute_fitness = _saturated_compute_fitness
    BaseRegionLibrary.check_viability = _nullisomy_check_viability
    # Clone.focal_deletion (cell.py:192) updates nregions/driver counts and then calls
    # super().focal_deletion, which resolves to Cell.focal_deletion at call time, so patching Cell is enough.
    Cell.focal_deletion = _fixed_focal_deletion
    # D-H: select_CNA_events looks gen_focal_event up as a module global at call time, so replacing
    # the module attribute reroutes every growth-phase focal event (ancestry.py -> mut.select_CNA_events).
    install_DH()
    # Self-test on a fake 6-block homolog whose only driver sits at the LAST position.
    fake_lib = types.SimpleNamespace(drivers={"chrT": [5]}, essential={"chrT": []})
    fake_chr = types.SimpleNamespace(name="chrT", seq=list(range(6)))
    for size in (1, 2, 3, 6):
        starts = _fixed_get_driver_start_regions(fake_lib, None, fake_chr, size)
        assert starts and all(0 <= j and j + size <= 6 for j in starts), "tail fix: start out of range"
        assert any(j <= 5 < j + size for j in starts), "tail fix: last position still unreachable"

    # D-E self-test: exponent uses round(p) (1.93 -> 2, 2.6 -> 3), driver delta 0.5 at CN 3.
    class _FakeLib:
        regions = {"chrT": 6}
        drivers = {"chrT": [1]}
        def __getitem__(self, chrname):
            return {1: 0.5}
    for p, ref in ((1.93, 2), (2.6, 3), (1.5, 2), (0.9, 1)):
        fake_clone = types.SimpleNamespace(get_ploidy=lambda p=p: p,
                                           attributes=types.SimpleNamespace(driver_counts={"chrT": {1: 3}}))
        got = _rounded_compute_fitness(_FakeLib(), fake_clone)
        assert abs(got - 1.5 ** (3 / ref)) < 1e-12, f"rounded-ploidy fitness wrong at p={p}"

    # D-F self-test (needs the block-id decoder; the real library build re-initialises MBIT anyway).
    init_mbit((50).bit_length())
    def fake_clone_with(seqs, deleted=True):
        ev = [types.SimpleNamespace(category="focal_deletion", chrom="chrT")] if deleted else []
        return types.SimpleNamespace(events=ev, library=types.SimpleNamespace(regions={"chrT": 4}),
                                     genome={"chrT": [types.SimpleNamespace(seq=list(q)) for q in seqs]})
    assert _has_cn0_block(fake_clone_with([[0, 1, 2], [1, 2]])) is True, "nullisomy: block 3 at CN 0 missed"
    assert _has_cn0_block(fake_clone_with([[0, 1, 2], [1, 2, 3]])) is False, "nullisomy: false positive"
    assert _has_cn0_block(fake_clone_with([[0, 1, 2], [1, 2]], deleted=False)) is False, "nullisomy: scanned without a deletion"

    # D-G self-test: a deletion of positions 3..4 on [0..5] must record ids [3, 4], not [5] (upstream) or [].
    class _FakeChr:
        name, allele, homolog_id, cent = "chrT", 0, 0, None
        def __init__(self): self.seq = list(range(6))
        def delete(self, a, b): self.seq = self.seq[:a] + self.seq[b:]
    fake_cell = types.SimpleNamespace(events=[], gen=7)
    fc = _FakeChr(); _fixed_focal_deletion(fake_cell, fc, 3, 5)
    assert fc.seq == [0, 1, 2, 5] and fake_cell.events[0].gene_ids == [3, 4], "focal_deletion record fix: wrong ids"

    # D-I and D-H self-tests (prior v3; test_requirements of research_notes/v3_design_2026-10-21)
    _selftest_DI()
    _selftest_DH()


# ----------------- Constants -----------------
DATASET_KIND = "gene_clone_sets"           # spec: parameters.pkl / manifest "dataset_kind"

# Genome grid
REGION_LEN = int(5e6)                       # spec: 5 Mb blocks, fixed constant (not a CLI knob)
CHROM_ORDER = [f"chr{i}" for i in range(1, 23)]  # numeric order; SISTEM's dicts are string-sorted
CHROM_INDEX = {c: i for i, c in enumerate(CHROM_ORDER)}

# Events schema 3 (prior v4), defined once in passenger_overlay.py. SISTEM's 7 CNA categories
# (lineage/cell.py) come first, in the column order of clones.npz["event_counts"], then the two
# passenger categories of the D-J overlay.
EVENT_CATEGORIES = po.EVENT_CATEGORIES      # 7 growth + passenger_gain, passenger_loss
GROWTH_EVENT_CATEGORIES = po.GROWTH_EVENT_CATEGORIES
EVENT_INDEX = po.EVENT_INDEX
ARM_CODE = po.ARM_CODE                      # 'w' = whole chromosome used as one arm (acentric homolog)
# Columns of clones.npz["events"]; -1 = not applicable. 'terminal' (0/1, prior v3) marks D-H
# end-anchored events (and terminal passenger events); 'node' (prior v4) is the genealogy node the
# event occurred on. Readers must index columns by name (event_columns), not by position.
EVENT_COLUMNS = po.EVENT_COLUMNS
EVENTS_SCHEMA_VERSION = po.EVENTS_SCHEMA_VERSION   # 1 = prior v2 (9 cols); 2 = prior v3 (10, +terminal); 3 = prior v4 (11, +node; 9 categories)
assert EVENTS_SCHEMA_VERSION == 3 and EVENT_COLUMNS[-2:] == ("terminal", "node") and len(EVENT_CATEGORIES) == 9
assert GROWTH_EVENT_CATEGORIES == ("focal_amplification", "focal_deletion", "arm_duplication", "arm_deletion",
                                   "chromosomal_duplication", "chromosomal_deletion", "WGD")   # v3 order kept

# Prior ranges (spec "PRIOR"); each is (low, high) of a uniform
PRIOR_VERSION = 5                           # bump whenever sample_theta's distribution or draw order changes (5 = prior v4.1)
ARM_RATE_RANGE = (3e-6, 3e-4)               # log-uniform; v4 floor 3e-6; v2 = v3 = (1e-5, 3e-4); v1 U(1e-5, 1e-4) (arm script)
CHROM_RATE_RANGE = (3e-7, 1e-4)             # log-uniform; v4 floor 3e-7; v2 = v3 = (1e-6, 1e-4); v1 U(1e-6, 1e-5) (arm script)
FOCAL_RATE_RANGE = (3e-5, 1e-3)             # log-uniform; v4 floor 3e-5; v3 (1e-4, 1e-3); v2 (1e-4, 2e-3); v1 fixed 5e-4
LENGTH_MEAN_RANGE = (1.0, 8.0)              # log-uniform, blocks; v1 fixed SISTEM default 1.5; v2 = v3 = v4
PI_RANGE = (0.02, 0.12)                     # spec: driver fraction pi
Q_RANGE = (0.30, 0.55)                      # spec: P(sign = +1) q; v3 = v4; v1 and v2 U(0.40, 0.65)
M_RANGE = (0.05, 0.15)                      # spec: mean of Exponential effect size m
ETA_CAP = 0.6                               # spec: eta = min(Exp(mean=m), 0.6)
PTERM_RANGE = (0.25, 0.75)                  # v4 floor 0.25; v3 (0.35, 0.75); drawn after the blocks (D-H nuisance); v1/v2 none (= 0)
# Shared instability level L (prior v4; per event type since v4.1). Gaussian-copula loading of L per coupled rate
# (0 = that rate is independent of L: its realised value IS its *_indep draw, exactly). Prior v4 used ONE weight,
# RHO_CIN = 0.5, for all four; v4.1 (DECISION.md, factorial + 5e6 pilot): focal and passenger rates share one
# segmental mechanism (0.8), arm 0.5, whole-chromosome 0.4 (full decoupling, 0, made the realised chromosome-focal
# COUNT correlation -0.37, beyond the real range, because whole-chromosome events replace selected focal events). Override with
# setattr(sgs, "RHO_CIN_LOADINGS", {...}) giving all four keys (sample_theta rejects missing or extra keys).
CIN_RATE_KEYS = ("arm_rate", "chromosomal_rate", "focal_driver_rate", "pass_rate")
RHO_CIN_LOADINGS = MappingProxyType(     # read-only: replace the whole mapping, never edit it in place
    {"arm_rate": 0.5, "chromosomal_rate": 0.4, "focal_driver_rate": 0.8, "pass_rate": 0.8})
PASS_RATE_RANGE = (1e-4, 3e-2)              # log-uniform (through the copula): passenger events per lineage per generation;
                                            # v4.1 floor 1e-4 (quiet tumours), ceiling 3e-2 (busy end with trunk 1000); v4 (5e-4, 1.5e-2)
# The v3 values of every range v4 changed. setattr(sgs, k, v) for each item restores the v3 prior ranges;
# V3_SETTINGS (below the passenger constants) restores the full v3 model.
V3_RANGES = {
    "ARM_RATE_RANGE": (1e-5, 3e-4),
    "CHROM_RATE_RANGE": (1e-6, 1e-4),
    "FOCAL_RATE_RANGE": (1e-4, 1e-3),
    "PTERM_RANGE": (0.35, 0.75),
}

# D-J passenger overlay (prior v4; passenger_overlay.PassengerParams fields)
PASSENGERS_ENABLED = True                   # False: no overlay (block_cn = growth block_cn), schema 3 still written
PASS_LOSS_SHARE = 0.75                      # P(loss) per passenger event
PASS_TERM_SHARE = 0.30                      # P(terminal, telomere-anchored) per passenger event
PASS_LEN_MEAN = 2.0                         # interstitial length, blocks: max(1, round(Exp(mean)))
PASS_TERM_LEN_MEAN = 4.0                    # terminal length, blocks: max(1, round(Exp(mean))), at most arm - 1
PASS_TRUNK_GENS = 1000.0                    # generations of pre-growth trunk (before the founding cell, never simulated by
                                            # growth): grow() puts Poisson(pass_rate * PASS_TRUNK_GENS) passengers on the founder
                                            # genome before growth, clonal in every cell. v4.1 1000 (arm-end share); v4 2000
OVERLAY_SEED_OFFSET = 7_000_003             # namespace constant of the passenger seed sequences (passenger_seeds)

# The v4 values of every setting v4.1 changed: setattr(sgs, k, v) for each item restores prior v4 exactly (every theta
# field, the trunk and the overlay). PRIOR_VERSION stays 5: it tags the code and output contract, not the prior; the
# manifest and parameters.pkl record the settings in effect.
V4_SETTINGS = {
    "PASS_TRUNK_GENS": 2000.0,
    "PASS_RATE_RANGE": (5e-4, 1.5e-2),
    "RHO_CIN_LOADINGS": MappingProxyType({k: 0.5 for k in CIN_RATE_KEYS}),   # v4: one weight RHO_CIN = 0.5 for every rate
}
# The full v3 model: the v4 settings + the v3 ranges, no shared level, no passengers.
V3_SETTINGS = {**V4_SETTINGS, **V3_RANGES, "RHO_CIN_LOADINGS": MappingProxyType({k: 0.0 for k in CIN_RATE_KEYS}),
               "PASSENGERS_ENABLED": False}


def passenger_seeds(trial_seed):
    """(trunk_seed, overlay_seed) seed-sequence lists of one trial: [trial_seed, OVERLAY_SEED_OFFSET, 0] for the
    pre-growth trunk and [..., 1] for the post-growth edge overlay (np.random.default_rng(list)). Seed sequences
    keep every (trial, stream) pair independent; additive offsets made trial t's trunk equal trial t-1's overlay."""
    return [int(trial_seed), OVERLAY_SEED_OFFSET, 0], [int(trial_seed), OVERLAY_SEED_OFFSET, 1]

# Fixed simulation / library parameters (spec "LIBRARY" and "SIMULATION PARAMETERS")
MAX_DISTINCT_DRIV_RATIO = 0.9               # identical to the arm script; cap = int(0.9 * n_drivers)
MIN_DETECTABLE = 5e6                        # identical to the arm script
MAX_REGION_CN = 10                          # spec LIBRARY (= SISTEM default)
MAX_PLOIDY = 8                              # spec LIBRARY (= SISTEM default)
MIN_PLOIDY = 1.5                            # spec LIBRARY (= SISTEM default)

# Retry / seeding / CLI defaults (identical to the arm script)
MAX_ATTEMPTS = 5                            # arm script: while not complete and count < 5
PRIOR_SEED_STRIDE = 1000                    # arm script: seed = base_seed + sim_id * 1000
SEED_MODULUS = 2**32 - 1                    # arm script: np.random.seed((seed + t) % (2**32 - 1))
DEFAULT_START, DEFAULT_END = 1, 1000        # arm script --start / --end defaults
DEFAULT_NTRIALS = 25                        # arm script --ntrials default
DEFAULT_SEED = 12345                        # arm script --seed default
DEFAULT_OUTPUT = "gene_selection_data"      # new default; the arm script used 'all_data_gaus_pos'

PICKLE_PROTOCOL = 5                         # house style (arm script uses protocol 5 for CNratios_all)
SIM_LOG = True                              # False with --no-sim-log: SISTEM logs to os.devnull, no sim.log
STRICT_ENV_VAR = "GENE_SIM_STRICT"          # spec: assert driver_count_mismatches == 0 when set to 1


def is_strict() -> bool:
    return os.environ.get(STRICT_ENV_VAR, "0") == "1"


def genome_grid():
    """chrom_lens, arm_ratios (chr1..chr22) and SISTEM's own block count per chromosome."""
    chrom_lens, arm_ratios = hg38_chrom_lengths_from_cytoband(include_allosomes=False)
    nregions = get_num_regions(chrom_lens, REGION_LEN)
    assert set(nregions) == set(CHROM_ORDER), "unexpected chromosome set"
    return chrom_lens, arm_ratios, nregions


# ----------------- Prior -----------------
def _log_uniform(rng, lo, hi):
    """One draw with equal probability per decade: exp(U(log lo, log hi)). A plain uniform on
    (1e-6, 1e-4) would put 99% of its draws above 2e-6; this gives each power of ten a third."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


COPULA_EPS = 1e-12      # quantiles are clipped to [eps, 1-eps] before Phi^-1 (u = 0 or a rounding u = 1 would give +-inf)
COPULA_RULE = ("u_i = (log r_i - log lo_i)/(log hi_i - log lo_i) for arm_rate, chromosomal_rate, focal_driver_rate "
               "(u_pass for pass_rate); z'_i = sqrt(rho_i)*Phi^-1(L) + sqrt(1-rho_i)*Phi^-1(u_i); u'_i = Phi(z'_i); "
               "rate_i = exp(log lo_i + u'_i*(log hi_i - log lo_i)); quantiles clipped to [1e-12, 1-1e-12]; "
               "Phi = scipy.special.ndtr, Phi^-1 = scipy.special.ndtri; rho_i = RHO_CIN_LOADINGS[i] (since prior v4.1; "
               "v4 used one RHO_CIN for all four); rho_i = 0: realised rate i is the drawn one exactly (no transform); "
               "two rates i, j have normal-scale correlation sqrt(rho_i * rho_j); length_mean is not coupled")


def loguniform_quantile(r, lo, hi):
    """u in [0, 1] of a value on a log-uniform range (inverse of loguniform_from_quantile; vectorised)."""
    return (np.log(r) - np.log(lo)) / (np.log(hi) - np.log(lo))


def loguniform_from_quantile(u, lo, hi):
    """exp(log lo + u (log hi - log lo)) (vectorised)."""
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def cin_copula_quantile(u, L, rho):
    """Gaussian copula with the shared instability level L: Phi(sqrt(rho) Phi^-1(L) + sqrt(1-rho) Phi^-1(u)).
    Each output is U(0,1) marginally; two outputs sharing L with loadings rho_i, rho_j have normal-scale
    correlation sqrt(rho_i rho_j) (rho for equal loadings). rho == 0 returns u unchanged (exactly). Vectorised
    over u and L."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"copula loading must be in [0, 1], got {rho}")
    if rho == 0:
        return u
    u = np.clip(u, COPULA_EPS, 1 - COPULA_EPS)
    L = np.clip(L, COPULA_EPS, 1 - COPULA_EPS)
    return ndtr(np.sqrt(rho) * ndtri(L) + np.sqrt(1.0 - rho) * ndtri(u))


def rho_cin_loadings():
    """RHO_CIN_LOADINGS as floats, validated: exactly the CIN_RATE_KEYS, each in [0, 1]. Raises if the removed
    scalar RHO_CIN was set on the module (prior v4 harnesses), which would otherwise be silently ignored."""
    if "RHO_CIN" in globals():
        raise RuntimeError("RHO_CIN was replaced by RHO_CIN_LOADINGS in prior v4.1 (one loading per rate); "
                           "set e.g. {k: 0.5 for k in CIN_RATE_KEYS} for the v4 coupling")
    lo = dict(RHO_CIN_LOADINGS)
    if set(lo) != set(CIN_RATE_KEYS):
        raise ValueError(f"RHO_CIN_LOADINGS needs exactly the keys {CIN_RATE_KEYS}, got {sorted(lo)}")
    out = {k: float(lo[k]) for k in CIN_RATE_KEYS}
    for k, v in out.items():
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"RHO_CIN_LOADINGS[{k!r}] must be in [0, 1], got {v}")
    return out


def _couple(r, rg, cin_level, rho):
    """Realised rate of a pre-copula draw r on log-uniform range rg: r itself (exactly) at rho 0, else the copula."""
    if rho == 0:
        return r
    return float(loguniform_from_quantile(cin_copula_quantile(loguniform_quantile(r, *rg), cin_level, rho), *rg))


def sample_theta(rng, nregions, chrom_order):
    """One prior draw (PRIOR_VERSION 5 = prior v4.1). The rng is consumed in this fixed order:
    arm_rate, chromosomal_rate, focal_driver_rate, length_mean (each log-uniform: exp of a
    uniform on the log bounds), pi, q, m, then blocks in chrom_order / r order
    (per block: one uniform for z; if z == 1, one exponential for eta and one
    uniform for the sign), then p_term (one uniform on PTERM_RANGE), then (v4, new) cin_level
    L = rng.uniform() and u_pass = rng.uniform() LAST.
    Draw order and count through p_term are version 3's (and up to the blocks version 2's), so for the
    same prior seed the block positions, driver set and eta = |log(1+delta)| equal versions 2 and 3, and
    with V3_SETTINGS (v3 ranges, all loadings 0, no passengers) every v3 field is reproduced exactly. The three event rates are then
    coupled through the shared level L (cin_copula_quantile; COPULA_RULE); the pre-copula draws are kept
    as *_indep. pass_rate = the same copula on u_pass over PASS_RATE_RANGE.
    Prior v4.1: each rate has its own loading RHO_CIN_LOADINGS[k] (v4: one RHO_CIN = 0.5 for all; v4.1
    arm 0.5, chromosomal 0.4, focal 0.8, pass 0.8), so a loading changes only its own rate. The draws and their
    order are v4's: for the same prior seed every block draw (delta, driver set), pi, q, m, length_mean, p_term,
    cin_level, u_pass and the *_indep rates are bit-identical to v4; only realised rates whose loading changed
    (v4.1: chromosomal_rate, focal_driver_rate) and pass_rate / pass_rate_indep (PASS_RATE_RANGE) differ. With
    V4_SETTINGS every v4 field is reproduced exactly, with V3_SETTINGS every v3 field.
    Every block is a driver candidate, including the last one of each chromosome: the SISTEM off-by-one
    that made it unreachable by focal events is patched at import (deviation D-D, apply_sistem_patches).
    p_term (D-H), cin_level and pass_rate are nuisance parameters, not inference targets.
    """
    arm_rate = _log_uniform(rng, *ARM_RATE_RANGE)
    chromosomal_rate = _log_uniform(rng, *CHROM_RATE_RANGE)
    focal_driver_rate = _log_uniform(rng, *FOCAL_RATE_RANGE)
    length_mean = _log_uniform(rng, *LENGTH_MEAN_RANGE)
    pi = float(rng.uniform(*PI_RANGE))
    q = float(rng.uniform(*Q_RANGE))
    m = float(rng.uniform(*M_RANGE))

    delta_dict = {}
    for c in chrom_order:
        deltas = []
        for r in range(nregions[c]):
            z = int(rng.uniform() < pi)                 # z ~ Bernoulli(pi), every block
            if z == 0:
                deltas.append(0.0)                      # exact Python float zero
            else:
                eta = min(rng.exponential(scale=m), ETA_CAP)
                sign = 1.0 if rng.uniform() < q else -1.0
                deltas.append(float(np.exp(sign * eta) - 1))
        delta_dict[c] = deltas

    p_term = float(rng.uniform(*PTERM_RANGE))   # after the blocks: deltas stay paired with v2/v3
    cin_level = float(rng.uniform())            # v4: shared instability level L (after p_term: v3 fields stay paired)
    u_pass = float(rng.uniform())               # v4: passenger-rate quantile, LAST draw

    indep = {"arm_rate": (arm_rate, ARM_RATE_RANGE), "chromosomal_rate": (chromosomal_rate, CHROM_RATE_RANGE),
             "focal_driver_rate": (focal_driver_rate, FOCAL_RATE_RANGE)}
    rho = rho_cin_loadings()                    # v4.1: one loading per rate; loading 0 = exact (realised IS the draw)
    realised = {k: _couple(r, rg, cin_level, rho[k]) for k, (r, rg) in indep.items()}
    pass_rate_indep = float(loguniform_from_quantile(u_pass, *PASS_RATE_RANGE))
    pass_rate = float(loguniform_from_quantile(cin_copula_quantile(u_pass, cin_level, rho["pass_rate"]), *PASS_RATE_RANGE))

    return {
        "arm_rate": realised["arm_rate"],                       # realised (post-copula) rates drive SISTEM
        "chromosomal_rate": realised["chromosomal_rate"],
        "focal_driver_rate": realised["focal_driver_rate"],
        "length_mean": length_mean,
        "pi": pi,
        "q": q,
        "m": m,
        "delta_dict": delta_dict,
        "p_term": p_term,
        "cin_level": cin_level,
        "pass_rate": pass_rate,
        "arm_rate_indep": arm_rate,                             # pre-copula draws (diagnostics)
        "chromosomal_rate_indep": chromosomal_rate,
        "focal_driver_rate_indep": focal_driver_rate,
        "pass_rate_indep": pass_rate_indep,
    }


def driver_layout(delta_dict, chrom_order):
    """driver_mask (bool, length D, chrom_order flattening) and driver_index [(chr, r), ...].
    Uses SISTEM's own driver rule: block r is a driver iff delta != 0."""
    mask = np.array([x != 0 for c in chrom_order for x in delta_dict[c]], dtype=np.bool_)
    index = [(c, r) for c in chrom_order for r, x in enumerate(delta_dict[c]) if x != 0]
    return mask, index


# ----------------- Library -----------------
def build_library(delta_dict, chrom_lens, arm_ratios, out_dir):
    """Fresh FittedRegionLibrary for one simulation attempt (never reused).

    out_dir must be an existing directory. It is passed via a base Parameters
    object because, with params=None, SISTEM's BaseLibrary builds a default
    Parameters(out_dir='out') and so creates a stray ./out in the cwd.
    fill_params copies `params` and overrides only the non-None kwargs, so the
    explicit kwargs below still win."""
    base = Parameters(region_len=REGION_LEN, out_dir=out_dir)   # out_dir exists; avoids a stray ./out
    library = FittedRegionLibrary(
        params=base,
        chrom_lens=chrom_lens,
        arm_ratios=arm_ratios,
        region_len=REGION_LEN,
        max_distinct_driv_ratio=MAX_DISTINCT_DRIV_RATIO,
        max_region_CN=MAX_REGION_CN,
        max_ploidy=MAX_PLOIDY,
        min_ploidy=MIN_PLOIDY,
    )
    library.initialize(delta=delta_dict)

    n_nonzero_deltas = sum(1 for c in delta_dict for x in delta_dict[c] if x != 0)
    assert sum(len(v) for v in library.drivers.values()) == n_nonzero_deltas, \
        "library driver set does not match the nonzero deltas"
    assert all(len(v) == 0 for v in library.essential.values()), \
        "FittedRegionLibrary should have no essential regions"
    return library


# Installed here, not right after the patch definitions, because the D-H self-test builds a real
# library on the real grid (genome_grid / build_library above). Nothing before this line runs SISTEM.
apply_sistem_patches()


# ----------------- Simulation -----------------
def trial_is_complete(trial_dir):
    """True iff this trial finished in an earlier run: results.pkl says complete and clones.npz exists.
    results.pkl is written AFTER clones.npz (both atomically), so complete=True implies the arrays are whole."""
    res_path = os.path.join(trial_dir, "results.pkl")
    npz_path = os.path.join(trial_dir, "clones.npz")
    if not (os.path.isfile(res_path) and os.path.isfile(npz_path)):
        return False
    try:
        with open(res_path, "rb") as f:
            return bool(pkl.load(f).get("complete"))
    except Exception:
        return False


def simulate_gene_model(theta, out_dir, num_trials, prior_seed, sim_id, skip_complete=True):
    """Runs num_trials trials of one theta and writes parameters + results under out_dir.

    Resumable: with skip_complete (the default) trials that already finished are skipped, so a
    job killed by SLURM (time limit, node failure, preemption) is resumed by resubmitting the
    same id range. Each trial reseeds from rank_seed + t at its start, so the resumed trials are
    byte-identical to what an uninterrupted run would have produced.

    D-H reads the module global _P_TERM at call time. It is set to theta['p_term'] for the attempts of
    each trial and reset to 0.0 in a finally block, so retries see the same value and nothing leaks
    to the next theta of the same process (MPI ranks and array tasks run many thetas).

    Prior v4 (D-J): when PASSENGERS_ENABLED, each successful attempt gets the passenger overlay with
    theta['pass_rate'] and rng default_rng(trial_seed + OVERLAY_SEED_OFFSET), trial_seed = (rank_seed + t) %
    SEED_MODULUS; parameters.pkl records cin_level, pass_rate, the *_indep rates, rho_cin_loadings (v4: the scalar
    rho_cin), passengers_enabled and passenger_params; results.pkl the overlay and trunk counters and both seed lists
    (passenger_seeds)."""
    global _P_TERM
    os.makedirs(out_dir, exist_ok=True)
    chrom_lens, arm_ratios, nregions = genome_grid()
    delta_dict = theta["delta_dict"]
    rank_seed = prior_seed + RANK               # arm script: rank_seed = seed + RANK

    # A probe library, used only to record n_drivers / cap; never simulated with.
    probe = build_library(delta_dict, chrom_lens, arm_ratios, out_dir=out_dir)
    n_drivers = sum(len(v) for v in probe.drivers.values())
    assert n_drivers >= 1, f"sim{sim_id}: prior drew zero driver blocks (pi={theta['pi']:.4f}); cap would be 0 and every focal driver event would fail"
    cap = probe.max_distinct_driv
    assert cap == int(MAX_DISTINCT_DRIV_RATIO * n_drivers)
    del probe

    driver_mask, driver_index = driver_layout(delta_dict, CHROM_ORDER)
    # Flat per-block delta in the same order as the columns of block_cn (the NPE target);
    # zeros mark non-driver blocks, so (delta_vector != 0) == driver_mask.
    delta_vector = np.asarray([x for c in CHROM_ORDER for x in delta_dict[c]], dtype=np.float32)

    # Persist theta used (written up front, as in the arm script)
    parameters = {
        "prior_version": PRIOR_VERSION,
        "arm_rate": theta["arm_rate"],
        "chromosomal_rate": theta["chromosomal_rate"],
        "focal_driver_rate": theta["focal_driver_rate"],
        "length_mean": theta["length_mean"],
        "pi": theta["pi"],
        "q": theta["q"],
        "m": theta["m"],
        "p_term": theta["p_term"],              # D-H nuisance parameter (prior v3), not an inference target
        # prior v4: shared instability level, passenger rate, pre-copula rates (the three rates above are realised)
        "cin_level": theta["cin_level"],
        "pass_rate": theta["pass_rate"],
        "pass_rate_indep": theta["pass_rate_indep"],
        "arm_rate_indep": theta["arm_rate_indep"],
        "chromosomal_rate_indep": theta["chromosomal_rate_indep"],
        "focal_driver_rate_indep": theta["focal_driver_rate_indep"],
        "rho_cin_loadings": rho_cin_loadings(), # module settings at run time (harnesses override them); v4.1: per rate (v4 "rho_cin")
        "passengers_enabled": bool(PASSENGERS_ENABLED),
        "passenger_params": dataclasses.asdict(passenger_params(theta["pass_rate"])),
        "delta_dict": delta_dict,
        "delta_vector": delta_vector,           # float32[D], block_cn column order; 0.0 off the driver set
        "driver_mask": driver_mask,
        "driver_index": driver_index,
        "n_drivers": n_drivers,
        "cap": cap,
        "ntrials_used": num_trials,
        "seed": prior_seed,                     # spec key; equals the arm script's seed = base_seed + sim_id*1000
        "prior_seed": prior_seed,
        "rank_seed": rank_seed,
        "sim_id": sim_id,
        "dataset_kind": DATASET_KIND,
        "chrom_order": list(CHROM_ORDER),
    }
    with open(os.path.join(out_dir, "parameters.pkl"), "wb") as f:
        pkl.dump(parameters, f, protocol=PICKLE_PROTOCOL)

    for t in range(1, num_trials + 1):
        trial_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_dir, exist_ok=True)

        if skip_complete and trial_is_complete(trial_dir):
            mpi_print(f"sim{sim_id} trial {t}: already complete, skipped")
            continue

        # Deterministic seeding per (rank, trial), exactly as the arm script
        torch.manual_seed(rank_seed + t)
        trial_seed = (rank_seed + t) % SEED_MODULUS
        np.random.seed(trial_seed)
        # D-J: trunk and overlay have their own Generators (seed sequences), so the GLOBAL growth stream is not
        # consumed by them. Growth is still a different trajectory when the trunk changes the founder genome
        # (PASS_TRUNK_GENS > 0); it equals the passenger-free growth only with PASS_TRUNK_GENS = 0.
        trunk_seed, overlay_seed = passenger_seeds(trial_seed)
        pass_rate = float(theta["pass_rate"]) if PASSENGERS_ENABLED else None

        params = Parameters(
            region_len=REGION_LEN,
            focal_driver_rate=theta["focal_driver_rate"],
            length_mean=theta["length_mean"],
            max_distinct_driv_ratio=MAX_DISTINCT_DRIV_RATIO,
            min_detectable=MIN_DETECTABLE,
            arm_rate=theta["arm_rate"],
            chromosomal_rate=theta["chromosomal_rate"],
            out_dir=trial_dir,
            log_path=None if SIM_LOG else os.devnull,   # None -> SISTEM writes <trial>/sim.log
        )

        complete = False
        count = 0
        failed_sim, failed_call = 0, 0
        summary = None
        wall_seconds = None

        try:
            _P_TERM = float(theta["p_term"])      # D-H, read at call time; same value for every retry
            while not complete and count < MAX_ATTEMPTS:
                t0 = time.perf_counter()
                result = simulate_gene_model_caller(params, delta_dict, chrom_lens, arm_ratios,
                                                    pass_rate=pass_rate, overlay_seed=overlay_seed,
                                                    trunk_seed=trunk_seed)
                elapsed = time.perf_counter() - t0
                if isinstance(result, dict):
                    complete = True
                    summary = result
                    wall_seconds = float(elapsed)     # successful attempt only
                elif result == -1:
                    failed_call += 1
                elif result == -2:
                    failed_sim += 1
                count += 1
            # the trial must have run under the p_term recorded in parameters.pkl
            assert _P_TERM == theta["p_term"] == parameters["p_term"], \
                f"sim{sim_id} trial {t}: p_term changed during the trial ({_P_TERM} vs {theta['p_term']})"
        finally:
            _P_TERM = 0.0

        results = {
            "failures": [failed_sim, failed_call],
            "complete": complete,
            "n_clones": None,
            "n_events": None,
            "driver_count_mismatches": None,
            "wall_seconds": None,
            "sistem_version": sistem.__version__,
            "trunk_seed": list(trunk_seed),            # np.random.default_rng(seed sequence)
            "overlay_seed": list(overlay_seed),
            "passengers_enabled": pass_rate is not None,
        }
        if complete:
            results.update(
                n_clones=int(summary["block_cn"].shape[0]),
                n_events=int(summary["events"].shape[0]),
                driver_count_mismatches=summary["driver_count_mismatches"],
                wall_seconds=wall_seconds,      # growth + overlay of the successful attempt
                **{k: summary[k] for k in OVERLAY_RESULT_KEYS},
            )

        # Write order matters for resumability: clones.npz first, then results.pkl, each via
        # temp-file + os.replace, so results.pkl with complete=True can only exist once the
        # array file is whole. A kill between the two just leaves the trial "not complete".
        if complete:
            npz_final = os.path.join(trial_dir, "clones.npz")
            npz_tmp = f"{npz_final}.{os.getpid()}.tmp"
            with open(npz_tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    block_cn=summary["block_cn"],
                    popsize=summary["popsize"],
                    ploidy=summary["ploidy"],
                    fitness=summary["fitness"],
                    nWGD=summary["nWGD"],
                    n_altered_drivers=summary["n_altered_drivers"],
                    n_altered_drivers_post=summary["n_altered_drivers_post"],
                    bulk=summary["bulk"],
                    event_counts=summary["event_counts"],
                    events=summary["events"],
                    event_columns=np.asarray(EVENT_COLUMNS),
                    event_categories=np.asarray(EVENT_CATEGORIES),
                    events_schema_version=np.int16(EVENTS_SCHEMA_VERSION),
                    node_parent=summary["node_parent"],
                    node_birth_gen=summary["node_birth_gen"],
                    node_clone=summary["node_clone"],
                    clone_node=summary["clone_node"],
                )
            os.replace(npz_tmp, npz_final)

        # results.pkl is always written (bookkeeping), last, atomically
        res_final = os.path.join(trial_dir, "results.pkl")
        res_tmp = f"{res_final}.{os.getpid()}.tmp"
        with open(res_tmp, "wb") as f:
            pkl.dump(results, f, protocol=PICKLE_PROTOCOL)
        os.replace(res_tmp, res_final)


def _event_row(e, cat, node=-1):
    """One int32 row of EVENT_COLUMNS for a SISTEM CNA event record (node = genealogy node id, -1 if unknown)."""
    chrom = CHROM_INDEX.get(getattr(e, "chrom", None), -1)
    arm = ARM_CODE.get(getattr(e, "arm", None), -1)
    ids = getattr(e, "gene_ids", None)
    start, end = getattr(e, "start", None), getattr(e, "end", None)
    if ids:                                     # focal events: block ids covered (plain Chromosome never re-keys ids)
        first, last = int(min(ids)), int(max(ids))
    else:
        first = last = -1
    n = int(end - start) if (start is not None and end is not None) else (len(ids) if ids else -1)
    copies = getattr(e, "copies", None)
    driver = getattr(e, "driver", None)
    return [EVENT_INDEX[cat], chrom, arm, first, last, n,
            -1 if copies is None else int(copies),
            -1 if driver is None else int(bool(driver)),
            int(getattr(e, "gen", -1)),
            int(bool(getattr(e, "terminal", False))),   # D-H end-anchored event (set only by the wrapper)
            int(node)]


def collect_events(clones, node_of=None):
    """True CNA event history of the surviving clones, from SISTEM's own records.

    Each clone's `events` holds only the mutations that created it from its parent; the parent
    link survives clone death (Clone.die keeps events and parent), so walking the chain gives the
    whole lineage. Returns (event_counts int32 [n_clones, 7] over GROWTH_EVENT_CATEGORIES, events int32
    [n_events, len(EVENT_COLUMNS)]): per-clone counts per growth category along the lineage, and the unique
    events over all lineages (an ancestor's event is listed once however many descendants share it), in
    v3's row order. node_of maps id(clone) -> genealogy node id for the 'node' column (else -1).
    Lineages that went extinct are not reachable and are not counted, as in a real tumour."""
    rows, seen = [], {}
    counts_cache = {}                           # id(clone) -> counts vector of that clone's whole lineage
    n_growth = len(GROWTH_EVENT_CATEGORIES)

    def lineage_counts(c):
        k = id(c)
        if k in counts_cache:
            return counts_cache[k]
        own = np.zeros(n_growth, dtype=np.int32)
        node = -1 if node_of is None else node_of[id(c)]
        for e in c.events:
            cat = getattr(e, "category", None)
            if cat not in EVENT_INDEX or EVENT_INDEX[cat] >= n_growth or getattr(e, "passenger", False):
                continue                        # trunk passengers (founder) are reported by grow() as trunk rows
            own[EVENT_INDEX[cat]] += 1
            if id(e) not in seen:
                seen[id(e)] = len(rows)
                rows.append(_event_row(e, cat, node))
        total = own if c.parent is None else own + lineage_counts(c.parent)
        counts_cache[k] = total
        return total

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 100_000))   # lineages can be thousands of clones deep
    try:
        event_counts = np.stack([lineage_counts(c) for c in clones]).astype(np.int32)
    finally:
        sys.setrecursionlimit(old_limit)
    events = np.asarray(rows, dtype=np.int32).reshape(-1, len(EVENT_COLUMNS))
    return event_counts, events


@dataclass
class GrowthResult:
    """End-of-growth state of one successful attempt (no passenger overlay). grow() never mutates anything the
    overlay reads afterwards: block_cn is a snapshot array (the overlay copies it again), clones are SISTEM
    objects kept for inspection only."""
    clones: list                    # surviving SISTEM Clone objects (gs.clones[0] order = block_cn row order)
    library: Any                    # the FittedRegionLibrary used (library.max_distinct_driv = altered-driver cap)
    genealogy: Any                  # passenger_overlay.Genealogy
    block_cn: np.ndarray            # int16 [n_clones, D], total CN per block from count_region_CNs
    popsize: np.ndarray             # int64 [n_clones]
    ploidy: np.ndarray              # float32 [n_clones], clone.get_ploidy() = genome.nregions / D = mean block_cn
    fitness: np.ndarray             # float64 [n_clones], recomputed at run end (update_fitness)
    nWGD: np.ndarray                # int16 [n_clones]
    n_altered_drivers: np.ndarray   # int16 [n_clones], SISTEM's rule: driver blocks with cn != 2**(nWGD+1)
    event_counts: np.ndarray        # int32 [n_clones, 7], growth categories (trunk passengers NOT included)
    events: np.ndarray              # int32 [n_growth_events, len(EVENT_COLUMNS)], node column filled (growth rows only)
    driver_count_mismatches: int
    # pre-growth trunk passengers on the founder (clonal; node 0, gen 0); empty without pass_rate
    trunk_events: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, len(EVENT_COLUMNS)), np.int32))
    n_trunk_proposed: int = 0
    n_trunk_applied: int = 0
    n_trunk_rejected: int = 0
    trunk_rejected_by: dict = dataclasses.field(default_factory=dict)   # zero / cap / altered / ploidy / absent


TRUNK_REJECT_RULES = ("zero", "cap", "altered", "ploidy", "absent")


def _trunk_failed_rule(clone, library):
    """Which viability rule a founder fails, in the overlay's order: zero copies (D-F), driver CN cap, altered-driver
    cap, ploidy bounds (the same order as passenger_overlay's rejection counters)."""
    if _has_cn0_block(clone):
        return "zero"
    p = 2 ** (clone.genome.nWGD + 1)
    counts = [c for chrname in library.regions for r, c in clone.attributes.driver_counts[chrname].items() if c != p]
    if counts and max(counts) > library.max_region_CN:
        return "cap"
    if len(counts) > library.max_distinct_driv:
        return "altered"
    return "ploidy"


def apply_trunk_passengers(founder, library, pp, rng):
    """Pre-growth trunk (prior v4, D-J): n ~ Poisson(pp.rate * pp.trunk_gens) passenger events applied to the FOUNDER
    genome with SISTEM's own Clone methods, so growth starts from (and its viability checks respect) the trunk.

    Per event: window = po.sample_passenger_window (same rules and four draws as the overlay), then one draw
    rng.integers(n_homologs) picks a homolog of that chromosome. The reference window [start, end) of length L is
    mapped to the homolog's CURRENT positions (the founder is diploid with seq = block ids, so the mapping is the
    identity unless an earlier trunk event rearranged that homolog):
      terminal: anchored at the homolog's own end on the drawn side, positions [0, L') on p or [len - L', len) on q,
        L' = min(L, room) with room = cent - 1 (p) or len - cent - 1 (q) on the homolog's current centromere
        (len - 1 if acentric): never the whole arm, never across the centromere; room < 1 -> rejected 'absent';
      interstitial: the first contiguous run of positions whose block id lies in [start, end); none -> 'absent'.
    The row records what was REALISED: first_block/last_block = min/max block id covered, n_blocks = positions
    covered, terminal = 1 for anchored terminal events (interstitial events keep 0 even if a clipped window touches
    an end, as in the overlay), arm = the drawn side. Residual approximation: on a homolog rearranged by earlier
    trunk events, a terminal window covers the homolog's current end blocks, which can differ from the reference
    window, and an interstitial window covers only its first run (rare: a few % of trunk events at most). Gains:
    Clone.focal_amplification(m=1); losses: Clone.focal_deletion. The record is then marked driver=False,
    passenger=True, terminal as realised, gen=0. Viability: genome + attributes are deep-copied before each event;
    after it library.check_viability (patched: D-F zero copies, driver CN cap, altered-driver cap, ploidy bounds)
    must pass, else the snapshot is restored and the event counted by rule (_trunk_failed_rule, same order as the
    overlay). NOT neutral: a window covering a driver block changes the founder's driver_counts. Only `rng` is consumed (never the
    global NumPy RNG). Returns (rows int32 [n_applied, len(EVENT_COLUMNS)] with node 0 / gen 0, n_proposed,
    rejected_by dict)."""
    rejected = {k: 0 for k in TRUNK_REJECT_RULES}
    rows = []
    lam = pp.rate * pp.trunk_gens
    n = int(rng.poisson(lam)) if lam > 0 else 0
    if n == 0:
        return np.zeros((0, len(EVENT_COLUMNS)), np.int32), 0, rejected
    nregions = {c: int(library.regions[c]) for c in CHROM_ORDER}
    cent = {c: round(nregions[c] * library.arm_ratios[c]) for c in CHROM_ORDER}
    table = po.window_table(nregions, CHROM_ORDER, cent)
    for _ in range(n):
        chrom, start, end, terminal, loss = po.sample_passenger_window(rng, pp, nregions, CHROM_ORDER, cent, table=table)
        homologs = founder.genome[chrom]
        h = homologs[int(rng.integers(len(homologs)))]
        ids = [get_reg_id(x) for x in h.seq]
        n_pos = len(ids)
        if terminal:
            p_side = start < cent[chrom]
            if h.cent is not None:
                room = (h.cent - 1) if p_side else (n_pos - h.cent - 1)
            else:
                room = n_pos - 1
            L = min(end - start, room)
            if L < 1:
                rejected["absent"] += 1
                continue
            p0, p1 = (0, L) if p_side else (n_pos - L, n_pos)
        else:
            pos = [i for i, r in enumerate(ids) if start <= r < end]
            if not pos:
                rejected["absent"] += 1
                continue
            p0 = p1 = pos[0]
            while p1 < n_pos and start <= ids[p1] < end:
                p1 += 1
        snap_genome, snap_attr, n_ev = copy.deepcopy(founder.genome), copy.deepcopy(founder.attributes), len(founder.events)
        if loss:
            founder.focal_deletion(h, p0, p1)
        else:
            founder.focal_amplification(h, p0, p1, 1)
        ev = founder.events[-1]
        ev.driver, ev.passenger, ev.terminal, ev.gen = False, True, bool(terminal), 0
        if not library.check_viability(founder):
            rule = _trunk_failed_rule(founder, library)
            founder.genome, founder.attributes = snap_genome, snap_attr
            del founder.events[n_ev:]
            rejected[rule] += 1
            continue
        covered = ids[p0:p1]
        rows.append([EVENT_INDEX["passenger_loss" if loss else "passenger_gain"], CHROM_INDEX[chrom],
                     0 if start < cent[chrom] else 1, int(min(covered)), int(max(covered)), p1 - p0, 1, 0, 0,
                     int(bool(terminal)), 0])
    return np.asarray(rows, dtype=np.int32).reshape(-1, len(EVENT_COLUMNS)), n, rejected


def grow(params, delta_dict, chrom_lens, arm_ratios, pass_rate=None, trunk_seed=None) -> Union[GrowthResult, int]:
    """One growth attempt (SISTEM, no post-growth overlay). Returns a GrowthResult, -1 if SISTEM raised,
    -2 if extinct. Growth consumes only the global NumPy RNG (as v3); no overlay side effect.
    Pre-growth trunk (prior v4): when pass_rate is given and PASS_TRUNK_GENS > 0, apply_trunk_passengers puts
    Poisson(pass_rate * PASS_TRUNK_GENS) passenger events on the founder (GrowthSimulator's own initial clone) with
    rng = np.random.default_rng(trunk_seed) before simulate_agents; every clone inherits them. The trunk changes the
    founder genome (and its driver_counts where a window covers a driver), so growth is a different trajectory.
    pass_rate None or PASS_TRUNK_GENS 0: founder untouched, identical to v3 growth.
    Under GENE_SIM_STRICT=1 asserts the founder's driver_counts after the trunk and, at the end, block_cn against
    SISTEM's driver_counts (the PRE-overlay check)."""
    library = build_library(delta_dict, chrom_lens, arm_ratios, out_dir=params.out_dir)

    # params= is passed so fill_params does not build a default Parameters(out_dir='out')
    # and os.makedirs('out') in the cwd (a stray dir, and a FileExistsError race across ranks)
    anatomy = SimpleAnatomy(library, params=params, nsites=params.nsites, growth_rate=params.growth_rate, capacities=params.capacities)
    anatomy.init_pop_dynamic(0, 0, params.t_max, N0=params.N0)

    igenome = init_diploid_genome(library)
    _ = Clone(genome=copy.deepcopy(igenome), library=library, popsize=params.N0, site=0, birth_gen=0)

    gs = GrowthSimulator(anatomy)
    pp = passenger_params(pass_rate) if pass_rate is not None else None
    trunk_rows, n_trunk, trunk_rej = np.zeros((0, len(EVENT_COLUMNS)), np.int32), 0, {k: 0 for k in TRUNK_REJECT_RULES}
    if pp is not None and pp.trunk_gens > 0:
        assert trunk_seed is not None, "the pre-growth trunk needs trunk_seed"
        founder = gs._init_clone            # the clone growth starts from (the `_` clone above is never simulated)
        trunk_rows, n_trunk, trunk_rej = apply_trunk_passengers(founder, library, pp, np.random.default_rng(trunk_seed))
        if is_strict():
            cn = count_region_CNs(founder, total=True)
            bad = sum(cn[c][r] != founder.attributes.driver_counts[c][r] for c in CHROM_ORDER for r in library.drivers[c])
            assert bad == 0, f"founder driver_counts disagree with its genome after the trunk ({bad} blocks)"
            assert founder.genome.nregions == sum(int(cn[c].sum()) for c in CHROM_ORDER), "founder nregions drifted"
    try:
        gs.simulate_agents(params=params)
    except Exception:
        return -1

    clones = list(gs.clones[0])
    if len(clones) == 0:
        return -2

    int16_max = np.iinfo(np.int16).max
    block_rows, popsize, ploidy, fitness, nWGD, n_altered = [], [], [], [], [], []
    driver_count_mismatches = 0

    for clone in clones:
        clone.update_fitness()                  # clone.fitness is stale at run end

        cn = count_region_CNs(clone, total=True)    # {chr: copies per block, both alleles}
        row = np.concatenate([cn[c] for c in CHROM_ORDER])
        assert row.max() <= int16_max, "block copy number overflows int16"
        block_rows.append(row.astype(np.int16))

        driver_counts = clone.attributes.driver_counts
        # Deviation D-B: block_cn must agree with SISTEM's own driver bookkeeping
        for c in CHROM_ORDER:
            for r in library.drivers[c]:
                if cn[c][r] != driver_counts[c][r]:
                    driver_count_mismatches += 1

        # SISTEM's "altered driver" rule, mirrored from region_library.check_viability
        p = 2 ** (clone.genome.nWGD + 1)
        n_altered.append(sum(1 for c in library.regions for r, cnt in driver_counts[c].items() if cnt != p))

        popsize.append(clone.popsize)
        ploidy.append(clone.get_ploidy())
        fitness.append(clone.fitness)
        nWGD.append(clone.genome.nWGD)

    if is_strict():
        assert driver_count_mismatches == 0, \
            f"{driver_count_mismatches} (clone, driver block) pairs disagree with driver_counts"

    nodes = po.lineage_nodes(clones)
    genealogy = po.build_genealogy(clones, nodes=nodes)
    node_of = {id(x): i for i, x in enumerate(nodes)}
    event_counts, events = collect_events(clones, node_of=node_of)
    return GrowthResult(
        clones=clones, library=library, genealogy=genealogy,
        block_cn=np.stack(block_rows).astype(np.int16),
        popsize=np.asarray(popsize, dtype=np.int64),
        ploidy=np.asarray(ploidy, dtype=np.float32),
        fitness=np.asarray(fitness, dtype=np.float64),
        nWGD=np.asarray(nWGD, dtype=np.int16),
        n_altered_drivers=np.asarray(n_altered, dtype=np.int16),
        event_counts=event_counts, events=events,
        driver_count_mismatches=int(driver_count_mismatches),
        trunk_events=trunk_rows, n_trunk_proposed=int(n_trunk), n_trunk_applied=int(trunk_rows.shape[0]),
        n_trunk_rejected=int(sum(trunk_rej.values())), trunk_rejected_by=dict(trunk_rej),
    )


def passenger_params(pass_rate):
    """PassengerParams from the module constants (read at call time, so overrides apply)."""
    return po.PassengerParams(rate=float(pass_rate), loss_share=PASS_LOSS_SHARE, term_share=PASS_TERM_SHARE,
                              len_mean=PASS_LEN_MEAN, term_len_mean=PASS_TERM_LEN_MEAN, max_cn=MAX_REGION_CN,
                              trunk_gens=PASS_TRUNK_GENS, min_ploidy=MIN_PLOIDY, max_ploidy=MAX_PLOIDY)


def overlay_inputs(growth):
    """kwargs of po.apply_passenger_overlay other than pp and rng, from a GrowthResult (driver set, cap, grid
    and centromere blocks read off its library, nWGD from the growth)."""
    lib = growth.library
    nregions = {c: int(lib.regions[c]) for c in CHROM_ORDER}
    driver_mask = np.zeros(sum(nregions.values()), dtype=bool)
    off = 0
    for c in CHROM_ORDER:
        driver_mask[[off + r for r in lib.drivers[c]]] = True
        off += nregions[c]
    centromere_block = {c: round(nregions[c] * lib.arm_ratios[c]) for c in CHROM_ORDER}   # SISTEM's own rule
    return dict(block_cn=growth.block_cn, popsize=growth.popsize, genealogy=growth.genealogy,
                driver_mask=driver_mask, cap_altered=int(lib.max_distinct_driv), nregions=nregions,
                chrom_order=list(CHROM_ORDER), centromere_block=centromere_block, nWGD=growth.nWGD)


def run_overlay(growth, pp, rng):
    """po.apply_passenger_overlay on a GrowthResult. The GrowthResult is not modified, so several overlay
    settings can be run on one growth (Stage 1 harness)."""
    return po.apply_passenger_overlay(pp=pp, rng=rng, **overlay_inputs(growth))


OVERLAY_RESULT_KEYS = ("n_passengers_proposed", "n_passengers_applied", "n_passengers_rejected",
                       "n_passengers_rejected_zero", "n_passengers_rejected_cap", "n_passengers_rejected_altered",
                       "n_passengers_rejected_ploidy", "n_trunk_proposed", "n_trunk_applied", "n_trunk_rejected",
                       "n_trunk_rejected_by", "overlay_seconds")


def pack_summary(growth, overlay=None, overlay_seconds=0.0):
    """The clones.npz / results.pkl summary dict of one trial: block_cn, bulk and ploidy after the overlay (if any);
    fitness and n_altered_drivers at the end of growth (pre-overlay; passengers are neutral); event_counts widened to
    9 columns (passenger columns = the clone's edge passengers + every applied trunk passenger, which all clones
    inherit); events = growth rows + trunk rows (node 0) + edge passenger rows; genealogy arrays; overlay counters
    n_passengers_* (post-growth edge events; 0 without overlay) and trunk counters n_trunk_* (pre-growth founder)."""
    n = growth.block_cn.shape[0]
    if overlay is None:
        block_cn = growth.block_cn.copy()
        add = np.zeros((n, len(po.PASSENGER_EVENT_CATEGORIES)), dtype=np.int32)
        p_events = np.zeros((0, len(EVENT_COLUMNS)), dtype=np.int32)
        counters = dict(n_passengers_proposed=0, n_passengers_applied=0, n_passengers_rejected=0,
                        n_passengers_rejected_zero=0, n_passengers_rejected_cap=0, n_passengers_rejected_altered=0,
                        n_passengers_rejected_ploidy=0)
    else:
        block_cn, add, p_events = overlay.block_cn, overlay.event_counts_add, overlay.events
        counters = dict(n_passengers_proposed=int(overlay.n_proposed), n_passengers_applied=overlay.n_applied,
                        n_passengers_rejected=overlay.n_rejected,
                        n_passengers_rejected_zero=int(overlay.n_rejected_zero),
                        n_passengers_rejected_cap=int(overlay.n_rejected_cap),
                        n_passengers_rejected_altered=int(overlay.n_rejected_altered),
                        n_passengers_rejected_ploidy=int(overlay.n_rejected_ploidy))
    T = growth.trunk_events
    cat = T[:, EVENT_COLUMNS.index("category")]
    trunk_add = np.array([(cat == EVENT_INDEX["passenger_gain"]).sum(), (cat == EVENT_INDEX["passenger_loss"]).sum()],
                         dtype=np.int32)
    add = add + trunk_add[None, :]                  # every surviving clone descends from the founder
    counters.update(n_trunk_proposed=int(growth.n_trunk_proposed), n_trunk_applied=int(growth.n_trunk_applied),
                    n_trunk_rejected=int(growth.n_trunk_rejected), n_trunk_rejected_by=dict(growth.trunk_rejected_by))
    # popsize-weighted mean over clones (same weighting as arm_CNratios_bulk); ploidy = mean block CN, which
    # equals SISTEM's get_ploidy (genome.nregions / D) on the growth state (asserted under GENE_SIM_STRICT)
    bulk = np.average(block_cn.astype(np.float64), axis=0, weights=growth.popsize).astype(np.float32)
    ploidy = block_cn.astype(np.float64).mean(axis=1).astype(np.float32)
    # SISTEM's altered-driver rule (cn != 2**(nWGD+1)) on the POST-overlay block_cn; n_altered_drivers stays end-of-growth
    inp = overlay_inputs(growth)
    dm, ref = inp["driver_mask"], 2 ** (growth.nWGD.astype(np.int64) + 1)
    n_alt_post = (block_cn[:, dm].astype(np.int64) != ref[:, None]).sum(axis=1).astype(np.int16)
    g = growth.genealogy
    out = {
        "block_cn": block_cn.astype(np.int16),
        "popsize": growth.popsize,
        "ploidy": ploidy,
        "fitness": growth.fitness,
        "nWGD": growth.nWGD,
        "n_altered_drivers": growth.n_altered_drivers,
        "n_altered_drivers_post": n_alt_post,
        "bulk": bulk,
        "event_counts": np.hstack([growth.event_counts, add]).astype(np.int32),
        "events": np.vstack([growth.events, growth.trunk_events, p_events]).astype(np.int32),
        "driver_count_mismatches": growth.driver_count_mismatches,
        "node_parent": g.node_parent, "node_birth_gen": g.node_birth_gen,
        "node_clone": g.node_clone, "clone_node": g.clone_node,
        "overlay_seconds": float(overlay_seconds),
        **counters,
    }
    if is_strict():
        mean_growth = growth.block_cn.astype(np.float64).mean(axis=1).astype(np.float32)
        assert np.array_equal(mean_growth, growth.ploidy), "get_ploidy differs from mean block_cn"
        assert (block_cn >= 1).all(), "a block at 0 copies after the overlay"
        assert not dm.any() or int(block_cn[:, dm].max()) <= MAX_REGION_CN, "driver block above MAX_REGION_CN"
        assert ((block_cn[:, dm] != ref[:, None]).sum(axis=1) <= inp["cap_altered"]).all(), "altered-driver cap exceeded"
        assert ((ploidy >= MIN_PLOIDY) & (ploidy <= MAX_PLOIDY)).all(), "ploidy outside [MIN_PLOIDY, MAX_PLOIDY]"
    return out


def simulate_gene_model_caller(params, delta_dict, chrom_lens, arm_ratios, pass_rate=None, overlay_seed=None,
                               trunk_seed=None):
    """One simulation attempt = grow() (with, if PASSENGERS_ENABLED and pass_rate is given, the pre-growth trunk on
    the founder, rng default_rng(trunk_seed)) + the D-J post-growth overlay with passenger_params(pass_rate)
    and rng default_rng(overlay_seed) + pack_summary (seeds from passenger_seeds(trial_seed)). Returns the summary dict,
    -1 if SISTEM raised, -2 if extinct. Called without pass_rate (v3 call signature) it runs no overlay."""
    enabled = PASSENGERS_ENABLED and pass_rate is not None
    if enabled:
        assert overlay_seed is not None and trunk_seed is not None, "passengers need overlay_seed and trunk_seed"
    # pre-growth trunk on the founder (rng default_rng(trunk_seed)), then growth
    growth = grow(params, delta_dict, chrom_lens, arm_ratios, pass_rate=pass_rate if enabled else None,
                  trunk_seed=trunk_seed if enabled else None)
    if not isinstance(growth, GrowthResult):
        return growth
    overlay, dt = None, 0.0
    if enabled:
        assert overlay_seed is not None, "the passenger overlay needs overlay_seed"
        t0 = time.perf_counter()
        overlay = run_overlay(growth, passenger_params(pass_rate), np.random.default_rng(overlay_seed))
        dt = time.perf_counter() - t0
    return pack_summary(growth, overlay, overlay_seconds=dt)

# ----------------- Manifest -----------------
def git_info():
    """Best-effort git state of the repo holding this script.

    Returns {"git_commit": `git rev-parse HEAD` or None,
             "git_dirty": bool(`git status --porcelain` output) or None};
    each value is None when its git call fails."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        commit = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        commit = None
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        dirty = bool(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def write_manifest(output_dir, base_seed, ntrials, ntrials_range):
    """Dataset-level description, written once by rank 0 before any simulation."""
    chrom_lens, arm_ratios, nregions = genome_grid()
    D = sum(nregions.values())

    blocks = []
    index = 0
    for c in CHROM_ORDER:
        for r in range(nregions[c]):
            blocks.append({
                "index": index, "chr": c, "r": r,
                "start_bp": r * REGION_LEN,
                "end_bp": min((r + 1) * REGION_LEN, chrom_lens[c]),
            })
            index += 1

    # Read the remaining values off the same Parameters() call the trials use
    # (out_dir=output_dir already exists, so no new directory is created).
    p = Parameters(
        region_len=REGION_LEN,
        max_distinct_driv_ratio=MAX_DISTINCT_DRIV_RATIO,
        min_detectable=MIN_DETECTABLE,
        out_dir=output_dir,
    )
    fixed_parameters = {                       # focal_driver_rate and length_mean are DRAWN since prior v2
        "max_distinct_driv_ratio": p.max_distinct_driv_ratio,
        "min_detectable": p.min_detectable,
        "max_region_CN": MAX_REGION_CN,
        "max_ploidy": MAX_PLOIDY,
        "min_ploidy": MIN_PLOIDY,
        "N0": p.N0,
        "growth_rate": p.growth_rate,
        "capacities": p.capacities,
        "t_max": p.t_max,
        "WGD_rate": p.WGD_rate,
        "focal_gain_rate": p.focal_gain_rate,
        "chrom_dup_rate": p.chrom_dup_rate,
        "mag_mean": p.mag_mean,
        "lifespan_mean": p.lifespan_mean,
        # prior v3: D-I gain-dosage saturation and D-H terminal events
        "n_sat_extra": GAIN_SAT_EXTRA,
        "n_sat_reference": "round(p)",
        "n_sat_applies_to": "gain drivers (delta>0) only",
        "terminal_mag": TERMINAL_MAG,
        "terminal_ext_mean_blocks": TERM_EXT_MEAN,
        "terminal_gain_rate": "focal_gain_rate",
        "terminal_window_rule": TERMINAL_WINDOW_RULE,
    }

    manifest = {
        "dataset_kind": DATASET_KIND,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sistem_version": sistem.__version__,
        "script": {"path": os.path.abspath(__file__), **git_info()},
        "region_len": REGION_LEN,
        # SISTEM's hg38_chrom_lengths_from_cytoband actually holds GRCh37/hg19 lengths
        # (chr1 = 249,250,621 bp); real data must be on hg19 or lifted over before block comparison
        "genome_build": "GRCh37/hg19 (SISTEM's table is misnamed hg38)",
        "chrom_lens_bp": {c: int(chrom_lens[c]) for c in CHROM_ORDER},
        "nregions": {c: nregions[c] for c in CHROM_ORDER},
        "D": D,
        "chrom_order": list(CHROM_ORDER),
        "blocks": blocks,
        "arm_ratios": {c: arm_ratios[c] for c in CHROM_ORDER},
        # same rule as sistem.genome.genome / arm_library: round(nregions * arm_ratio)
        "centromere_block": {c: round(nregions[c] * arm_ratios[c]) for c in CHROM_ORDER},
        "prior_version": PRIOR_VERSION,
        "prior": {
            "arm_rate": ["loguniform", *ARM_RATE_RANGE],
            "chromosomal_rate": ["loguniform", *CHROM_RATE_RANGE],
            "focal_driver_rate": ["loguniform", *FOCAL_RATE_RANGE],
            "rates_note": "marginals log-uniform on these ranges; since prior v4 the rates are coupled through cin_level (copula_rule), since v4.1 with one loading per rate (rho_cin_loadings); parameters.pkl holds the realised rates and the pre-copula draws as *_indep",
            "length_mean_blocks": ["loguniform", *LENGTH_MEAN_RANGE],
            "cin_level": ["uniform", 0.0, 1.0],
            "rho_cin_loadings": rho_cin_loadings(),
            "rho_cin_loadings_note": "Gaussian-copula loading of cin_level per rate (prior v4.1; final values from the factorial and the 5e6 pilot, research_notes/v4_1_design_2026-10-26/DECISION.md); 0 = the realised rate is its *_indep draw exactly; v4 used one rho_cin = 0.5 for all four (v4_settings)",
            "pass_rate": ["loguniform", *PASS_RATE_RANGE],
            "pass_rate_note": "expected passenger events per lineage per generation (D-J overlay), through the same copula as the rates from u_pass (loading rho_cin_loadings['pass_rate']); ~2,600 generations -> ~0.26 to ~39 per lineage",
            "copula_rule": COPULA_RULE,
            "draw_order": "arm, chrom, focal, length_mean, pi, q, m, blocks, p_term, cin_level, u_pass (last)",
            "v3_ranges": {k: list(v) for k, v in V3_RANGES.items()},
            "v4_settings": {k: (list(v) if isinstance(v, tuple) else dict(v) if isinstance(v, MappingProxyType) else v)
                            for k, v in V4_SETTINGS.items()},
            "v4_to_v41_changes": {
                "adopted": "2026-10-26, prior v4.1 = prior_version 5 (research_notes/v4_1_design_2026-10-26/DECISION.md, scope revised after the design review: no pi / driver-guard change, overlay unchanged); setattr(sgs, k, v) for each item of v4_settings restores v4 exactly",
                "changes": [
                    "PASS_TRUNK_GENS 2000 -> 1000: restores the arm-end share (sweep B1 - B0 +0.044 [0.016, 0.072]; diagnostics projection 0.632-0.642) without moving driver Spearman or sign recovery (DECISION.md table)",
                    "PASS_RATE_RANGE logU(5e-4, 1.5e-2) -> logU(1e-4, 3e-2): floor = pre-registered fallback (tumours without segmental-instability mechanisms); ceiling 3e-2 restores the busy end lost with trunk 1000 (factorial G3/G4; pilot G4 P(bp >= 45) 0.167 vs G1 0.133; DECISION.md)",
                    "one copula loading per rate (RHO_CIN_LOADINGS) instead of one RHO_CIN; arm 0.5, chromosomal 0.4, focal 0.8, pass 0.8: focal and passenger events share a segmental mechanism; whole-chromosome partly decoupled (distinct mechanism, Davoli 2017) at 0.4 because full decoupling (0) drove the realised chromosome-focal count correlation to -0.37 and Spearman(bp, fga) to 0.21 in the 5e6 pilot (DECISION.md)",
                ],
                "pairing": "draws and draw order unchanged: for the same prior seed every block draw (delta, driver set), pi, q, m, length_mean, p_term, cin_level, u_pass and the *_indep rates are bit-identical to v4; arm_rate too (same loading 0.5); chromosomal_rate, focal_driver_rate (new loadings) and pass_rate / pass_rate_indep (new range) differ. V4_SETTINGS reproduces every v4 field exactly",
                "unchanged": "every range except PASS_RATE_RANGE, pi U(0.02, 0.12), q, m, eta cap 0.6, p_term U(0.25, 0.75), MAX_REGION_CN 10, MAX_PLOIDY 8, MIN_PLOIDY 1.5, cap ratio 0.9, min_detectable 5e6, patches D-D..D-I, the D-J edge overlay (all-or-none loss rejection, known gain bias: diagnostics/REPORT.md Q4.1), passenger loss / terminal shares and lengths, events schema 3",
            },
            "v3_to_v4_changes": {
                "adopted": "2026-10-23, Stage 1 provisional (research_notes/v4_design_2026-10-23/review/REVIEW.md sections 4-5); the Stage 1 harness overrides these per arm",
                "changes": [
                    "rate floors lowered (quiet tail): arm_rate logU(1e-5, 3e-4) -> logU(3e-6, 3e-4); chromosomal_rate logU(1e-6, 1e-4) -> logU(3e-7, 1e-4); focal_driver_rate logU(1e-4, 1e-3) -> logU(3e-5, 1e-3)",
                    "p_term floor: U(0.35, 0.75) -> U(0.25, 0.75) (P6)",
                    "shared instability level (P7): cin_level ~ U(0,1) drawn after p_term; Gaussian copula with RHO_CIN couples arm/chrom/focal rates (marginals unchanged in shape); length_mean not coupled",
                    "neutral passenger overlay (P1, deviation D-J): pass_rate drawn last (u_pass) through the same copula; post-growth overlay on the clone genealogy; events schema 3 (passenger categories, 'node' column) and genealogy arrays in clones.npz",
                ],
                "pairing": "draw order and count through p_term unchanged: with V3_RANGES and RHO_CIN = 0 every v3 field is reproduced exactly; with the v4 ranges block positions, driver set, |log(1+delta)|, q, pi, m, length_mean, the p_term quantile and the quantiles of the pre-copula *_indep rates are paired; under RHO_CIN > 0 the realised rates are not (they depend on cin_level). With --ntrials-range the ntrials draw now follows cin_level and u_pass, so ntrials_used is not paired with v3",
                "unchanged": "length_mean, pi, q, m, eta cap 0.6, MAX_REGION_CN 10, MAX_PLOIDY 8, MIN_PLOIDY 1.5, cap ratio 0.9, min_detectable 5e6, patches D-D..D-I",
            },
            "v3_draw_order": "arm_rate, chromosomal_rate, focal_driver_rate, length_mean, pi, q, m, then blocks, then p_term (last)",
            "v1_to_v2_change": "2026-10-20 after step 1 (change-size comparison with TCGA/PCAWG non-WGD cohorts): rates log-uniform; chromosome range widened to 1e-4; focal rate drawn (was fixed 5e-4); focal length_mean drawn (was fixed 1.5 blocks) so partial-arm segments can occur",
            "v2_to_v3_changes": {
                "adopted": "2026-10-22, multi-agent design/sweep/verification in research_notes/v3_design_2026-10-21/ (workflow_result.json decision.final_v3)",
                "changes": [
                    "q: U(0.40, 0.65) -> U(0.30, 0.55) (judgement call: TSG-like loci outnumber OG-like ones, losses slightly more common; flagged for the advisor)",
                    "focal_driver_rate: logU(1e-4, 2e-3) -> logU(1e-4, 1e-3) (cost: rates above 1e-3 ran up to 3,731 s per trial and added almost no visible focal events)",
                    "p_term: new per-theta nuisance prior U(0.35, 0.75), drawn LAST after the block loop (block positions, driver set and |log(1+delta)| stay paired with v2 for the same prior seed)",
                    "patch D-I region_gain_dosage_saturation: gain drivers use exponent min(n, round(p)+1)/round(p); loss drivers unchanged; supersedes the D-E body and keeps its round(p)",
                    "patch D-H focal_terminal_events: with prob p_term a driver focal event becomes end-anchored (TERMINAL_MAG = 1 copy for gains, TERM_EXT_MEAN = 2 blocks); adopted only together with D-I; new event class, advisor sign-off pending",
                    "events table: 10th column 'terminal' (0/1), events schema_version 2; readers index columns by name",
                ],
                "unchanged": "arm_rate, chromosomal_rate, length_mean, pi, m, eta cap 0.6, MAX_REGION_CN 10, MAX_PLOIDY 8, MIN_PLOIDY 1.5, cap ratio 0.9, min_detectable 5e6, patches D-D, D-F, D-G",
                "caveats": [
                    "(i) terminal windows never target drivers at blocks cent-1 / cent (a window there would be a whole arm); those drivers only get interstitial focal events",
                    "(ii) on acentric homologs (cent None) terminal windows may be centromere-bounded rather than telomere-to-centromere-side",
                    "(iii) MAX_REGION_CN applies to driver blocks only, so passenger blocks can exceed it",
                    "(iv) length_mean governs interstitial focal events only; terminal window lengths are set by the target position and TERM_EXT_MEAN",
                ],
            },
            "pi": ["uniform", *PI_RANGE],
            "q": ["uniform", *Q_RANGE],
            "m": ["uniform", *M_RANGE],
            "p_term": ["uniform", *PTERM_RANGE],
            "p_term_note": "nuisance parameter, not an inference target (D-H share of end-anchored driver focal events; drawn right after the blocks, before the v4 draws cin_level and u_pass)",
            "eta": f"min(Exponential(mean=m), {ETA_CAP})",
            "delta": "z~Bernoulli(pi) for every block; if z: eta=min(Exp(mean=m),0.6), sign=+1 if U<q else -1, delta=exp(sign*eta)-1 in [exp(-0.6)-1, exp(0.6)-1] ≈ [-0.4512, 0.8221]; else delta=0.0",
            "last_block_rule": "none: the last block is a normal candidate because SISTEM's tail off-by-one is patched (see sistem_patches)",
        },
        "sistem_patches": {
            "applied": SISTEM_PATCHES,
            "region_driver_start_regions_tail_fix": "BaseRegionLibrary.get_driver_start_regions scanned sequence[:-size] (region_library.py:144), so focal driver events could never cover the last position of a homolog; replaced by a version that examines every position with start in [max(0,i-size+1), min(i,len-size)], matching the bound the passenger function already uses",
            "superseded": {"region_fitness_rounded_ploidy": "region_gain_dosage_saturation"},
            "region_fitness_rounded_ploidy": "SUPERSEDED since prior v3 by region_gain_dosage_saturation (D-I), whose body keeps this rounding. BaseRegionLibrary.compute_fitness exponent is n/round(p) (CINner, Dinh et al. 2025 S1 Eq 1-2) instead of n/p with continuous p (region_library.py:28); the continuous form rewarded deleting neutral DNA and tumours collapsed to the min_ploidy floor (ploidy median 1.51 in the 2026-10-19 calibration). init_max_fit unpatched (metastatic growth only)",
            "focal_deletion_gene_ids_record_fix": "Cell.focal_deletion recorded gene_ids after applying the deletion (lineage/cell.py:47), so the record held the blocks that slid into those positions, or nothing at a chromosome end; the slice is now captured before deleting. Bookkeeping only, no change to the simulated genomes",
            "region_nullisomy_lethal": "BaseRegionLibrary.check_viability additionally rejects a clone with any block at 0 total copies (CINner nullisomy_max = 0); SISTEM had no minimum copy number (19.9% of arms at CN 0 in the arm dataset)",
            "region_gain_dosage_saturation": {
                "id": "D-I",
                "name": "region_gain_dosage_saturation",
                "hook": "BaseRegionLibrary.compute_fitness (supersedes the D-E body; keeps round(p))",
                "rule": "gain drivers: exponent min(n, round(p)+1)/round(p); loss drivers unchanged; extra copies neutral, not lethal",
                "formula": "fitness = prod_r (1+delta_r)^(n_eff/ref), ref = max(1, round(p)), n_eff = min(n_r, ref + n_sat_extra) if delta_r > 0 else n_r",
                "parameters": {"n_sat_extra": GAIN_SAT_EXTRA, "n_sat_reference": "round(p)", "applies_to": "gain drivers (delta>0) only"},
            },
            "focal_terminal_events": {
                "id": "D-H",
                "name": "focal_terminal_events",
                "hook": "sistem.lineage.mutation.gen_focal_event (wrapper; driver=False and p_term=0 delegate to the original with no extra RNG draw)",
                "rule": TERMINAL_WINDOW_RULE,
                "gain": "focal_gain_rate (0.5), +1 copy",
                "loss": "focal_deletion",
                "parameters": {"p_term": ["uniform", *PTERM_RANGE], "TERMINAL_MAG": TERMINAL_MAG, "TERM_EXT_MEAN": TERM_EXT_MEAN},
                "record": "clone.events[-1].terminal = True; events table column 'terminal'",
            },
        },
        "cap_formula": f"int({MAX_DISTINCT_DRIV_RATIO} * n_drivers)",
        "fixed_parameters": fixed_parameters,
        "ntrials_policy": {
            "ntrials": None if ntrials_range is not None else ntrials,
            "ntrials_range": list(ntrials_range) if ntrials_range is not None else None,
        },
        "seeding": {
            "base_seed": base_seed,
            "prior_seed_formula": f"base_seed + sim_id*{PRIOR_SEED_STRIDE}",
            "rank_seed_formula": "prior_seed + RANK",
            "trial_seed_formula": "(rank_seed + t) % (2**32-1)",
            "trunk_seed_formula": f"np.random.default_rng([trial_seed, {OVERLAY_SEED_OFFSET}, 0]) (prior v4, pre-growth trunk)",
            "overlay_seed_formula": f"np.random.default_rng([trial_seed, {OVERLAY_SEED_OFFSET}, 1]) (prior v4, post-growth edge overlay)",
        },
        "sim_log": SIM_LOG,
        "events": {"schema_version": EVENTS_SCHEMA_VERSION,
                   "event_categories": list(EVENT_CATEGORIES), "event_columns": list(EVENT_COLUMNS),
                   "note": "growth rows: true per-lineage CNA events from clone.events (surviving lineages only); -1 = not applicable; "
                           "first_block/last_block are block ids (within the chromosome) for focal and passenger events, n_blocks their count; "
                           "terminal = 1 for D-H end-anchored events and terminal passenger events (schema 2+); "
                           "node = genealogy node id the event occurred on (schema 3; growth: the clone that recorded it; "
                           "passenger: the node whose incoming edge carried it). Passenger rows (categories passenger_gain/"
                           "passenger_loss) follow the growth rows: driver 0, copies 1, gen = birth_gen of the node. "
                           "event_counts has one column per category (9). Schema 1 = prior v2 (9 columns), 2 = prior v3 "
                           "(10, +terminal), 3 = prior v4 (11, +node): index columns by name via event_columns",
                   "genealogy": "clones.npz node_parent (-1 root), node_birth_gen, node_clone (-1 = not a surviving clone), "
                                "clone_node; nodes = surviving clones + all ancestors, ordered by (birth_gen, site, creation counter "
                                "= integer suffix of SISTEM's clone name), so node_parent[i] < i"},
        "passenger_overlay": {
            "id": "D-J",
            "enabled": bool(PASSENGERS_ENABLED),
            "module": "passenger_overlay.py (apply_passenger_overlay)",
            "rate": "theta['pass_rate'] (prior block)",
            "loss_share": PASS_LOSS_SHARE, "term_share": PASS_TERM_SHARE,
            "len_mean_blocks": PASS_LEN_MEAN, "term_len_mean_blocks": PASS_TERM_LEN_MEAN, "max_cn": MAX_REGION_CN,
            "trunk_gens": PASS_TRUNK_GENS,
            "trunk_note": "applied to the founder before growth; edge passengers after growth. Poisson(pass_rate * trunk_gens) "
                          "passengers (same window sampler) are put on the founder genome with SISTEM's Clone methods "
                          "before simulate_agents, each kept only if check_viability passes (else restored); every clone "
                          "inherits them; rows node 0, gen 0. NOT neutral: a window covering a driver block changes the "
                          "founder's driver_counts (fitness, altered-driver cap, saturation headroom), and with trunk_gens > 0 "
                          "growth is a different trajectory from the passenger-free run (identical only at trunk_gens = 0). "
                          "Edge passengers are neutral by construction",
            "min_ploidy": MIN_PLOIDY, "max_ploidy": MAX_PLOIDY,
            "rng": f"trunk: np.random.default_rng([trial_seed, {OVERLAY_SEED_OFFSET}, 0]); overlay: np.random.default_rng([trial_seed, {OVERLAY_SEED_OFFSET}, 1]); trial_seed = (rank_seed + t) % (2**32-1); neither consumes the global growth stream (results.pkl trunk_seed, overlay_seed)",
            "rules": dict(po.OVERLAY_RULES),
            "known_artefact": "all-or-none loss rejection: a loss is rejected if any descendant would reach 0 copies, so edges with many descendants take mostly gains (loss share 0.20 on edges with > 1000 descendants vs nominal 0.75); kept unchanged in prior v4.1 (research_notes/v4_1_design_2026-10-26/diagnostics/REPORT.md Q4.1)",
            "outputs": "block_cn, bulk, ploidy after the overlay; fitness and n_altered_drivers at the end of growth (pre-overlay); "
                       "results.pkl n_passengers_proposed/applied/rejected(+_zero/_cap/_altered/_ploidy) for edge events, "
                       "n_trunk_proposed/applied/rejected/rejected_by for the founder trunk, overlay_seconds",
        },
        "files": {"sim": ["parameters.pkl"], "trial": ["results.pkl", "clones.npz"] + (["sim.log"] if SIM_LOG else [])},
    }
    # Atomic write: in a SLURM job array every task is "rank 0" and writes this same
    # file concurrently (identical settings), so write to a temp name and rename.
    final = os.path.join(output_dir, "manifest.json")
    tmp = f"{final}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, final)


# ----------------- MPI work distribution -----------------
def chunk_jobs_round_robin(jobs, size):
    chunks = [[] for _ in range(size)]
    for i, job in enumerate(jobs):
        chunks[i % size].append(job)
    return chunks


def gather_simulation_pairs_mpi(start_id, end_id, output_dir, num_trials, base_seed, ntrials_range=None,
                                skip_complete=True):
    """
    Rank 0:
      - writes manifest.json
      - samples theta for [start_id, end_id], one rng per theta id
      - builds jobs = [(theta_i (dict), sim_dir, ntrials_used, prior_seed, sim_id), ...]
      - scatters job chunks to all ranks
    Other ranks:
      - receive their chunk and execute
    Without MPI (deviation D-C) the single process is rank 0 and runs every job.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "simulation_outputs")
    if RANK == 0:
        os.makedirs(data_dir, exist_ok=True)

    # Everyone waits until output dirs exist
    if USING_MPI:
        COMM.Barrier()

    if RANK == 0:
        write_manifest(output_dir, base_seed, num_trials, ntrials_range)
        _, _, nregions = genome_grid()

        jobs = []
        for sim_id in range(start_id, end_id + 1):
            sim_dir = os.path.join(data_dir, f"sim{sim_id}")
            # Create the directory tree ahead of time to reduce rank contention
            os.makedirs(sim_dir, exist_ok=True)
            # Rank-independent, so theta for a given sim_id is reproducible
            prior_seed = base_seed + sim_id * PRIOR_SEED_STRIDE
            rng = np.random.default_rng(prior_seed)
            theta = sample_theta(rng, nregions, CHROM_ORDER)
            if ntrials_range is not None:
                a, b = ntrials_range
                ntrials_used = int(rng.integers(a, b + 1))   # same rng, after the deltas
            else:
                ntrials_used = num_trials
            jobs.append((theta, sim_dir, ntrials_used, prior_seed, sim_id))

        # Round-robin split for better balancing if some sims run longer
        chunks = chunk_jobs_round_robin(jobs, SIZE)
    else:
        chunks = None

    # Scatter chunks
    if USING_MPI:
        jobs_local = COMM.scatter(chunks, root=0)
    else:
        jobs_local = chunks[0]

    # Execute local jobs
    for theta_i, sim_dir, ntrials, prior_seed, sim_id in jobs_local:
        mpi_print(f"sim{sim_id}: {ntrials} trials")
        simulate_gene_model(theta_i, sim_dir, ntrials, prior_seed, sim_id, skip_complete=skip_complete)

    # Ensure all ranks finish before job exits
    if USING_MPI:
        COMM.Barrier()


# ----------------- CLI / main -----------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', type=int, default=DEFAULT_START, help='Start id of simulations.')
    parser.add_argument('--end', type=int, default=DEFAULT_END, help='End id of simulations (inclusive).')
    parser.add_argument('--ntrials', type=int, default=DEFAULT_NTRIALS, help='Number of i.i.d. simulations per parameter set.')
    parser.add_argument('--ntrials-range', type=int, nargs=2, metavar=('A', 'B'), default=None,
                        help='Draw ntrials ~ U{A..B} per theta (overrides --ntrials; recorded in parameters.pkl).')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help='Output directory.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Base RNG seed.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Redo trials that already completed (default: skip them, which makes a '
                             'killed job resumable by resubmitting the same id range).')
    parser.add_argument('--no-sim-log', action='store_true',
                        help="Do not write SISTEM's per-trial sim.log (~133 KB each, more than clones.npz).")
    args = parser.parse_args()
    if args.ntrials_range is not None:
        a, b = args.ntrials_range
        if not (1 <= a <= b):
            parser.error('--ntrials-range A B needs 1 <= A <= B')
    elif args.ntrials < 1:
        parser.error('--ntrials must be >= 1')
    if args.end < args.start:
        parser.error('--end must be >= --start')
    return args


def main():
    # Quick visibility
    mpi_print(f"USING_MPI={USING_MPI} SIZE={SIZE} SLURM_NTASKS={os.environ.get('SLURM_NTASKS', '-')}")

    # If SLURM says multi-task but mpi size==1, fail fast with instructions
    require_mpi_or_exit()

    args = parse_args()
    global SIM_LOG
    SIM_LOG = not args.no_sim_log
    if is_mpi_active():
        mpi_print("MPI is active; running distributed path.")
    else:
        mpi_print("MPI not active; running single-process fallback.")

    # Same function in both cases: without MPI, rank 0 receives every job.
    gather_simulation_pairs_mpi(
        start_id=args.start,
        end_id=args.end,
        output_dir=args.output,
        num_trials=args.ntrials,
        base_seed=args.seed,
        ntrials_range=args.ntrials_range,
        skip_complete=not args.overwrite,
    )


if __name__ == '__main__':
    main()
