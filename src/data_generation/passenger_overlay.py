"""Neutral passenger CNA overlay on the clone genealogy (prior v4, Stage 1). Pure numpy, no SISTEM import.

After SISTEM's growth phase the surviving clones and all their ancestors (clone.parent links survive clone
death) form a rooted genealogy. This module (1) turns that genealogy into four integer arrays and (2) adds
neutral passenger copy-number events along its edges in total-CN block space, applied to every surviving
clone that descends from the edge. Used by simulate_gene_selection.py (one overlay per trial) and by the
Stage 1 sweep harness (many overlay settings on one growth). Design: research_notes/v4_design_2026-10-23/
review/REVIEW.md section 4 P1 (overlay instead of growth-time insertion: growth-time passengers cost 4-12x
wall time and SISTEM's own post-growth passenger path crashes with our patches).

GENEALOGY (build_genealogy)
    nodes = every surviving clone plus all its ancestors via clone.parent. Node order (the node id) is
    sorted by (birth_gen, site, creation counter), where the creation counter is the integer suffix of
    SISTEM's clone name (GrowthSimulator._assign_clone_name gives 'P0', 'P1', ... in creation order,
    non-viable clones included). A parent is created before its children, so node_parent[i] < i for every
    non-root node (asserted). Arrays (int32): node_parent (-1 for the root), node_birth_gen,
    node_clone (index of the surviving clone this node is, else -1), clone_node (node id per surviving
    clone, in the order of the clones list passed in).

OVERLAY RULES (apply_passenger_overlay)
    Trunk: NOT drawn here. The passengers of the pre-growth trunk (Poisson(rate * trunk_gens), clonal) are
    applied to the FOUNDER genome before growth by simulate_gene_selection.grow(), with the same window
    sampler (sample_passenger_window), so growth respects them; trunk_gens is carried in PassengerParams for
    that. The root node gets no events here.
    Edges: nodes are visited in node order; for a node with a parent, dt = birth_gen - parent birth_gen and
    n ~ Poisson(rate * dt) events land on that edge. An edge event applies to every surviving clone
    descending from the node (the node's own clone included if it survives). Time after a leaf's birth
    carries no events (the genealogy only has branch points). Events are processed in node
    order and, within a node, in draw order, each seeing the CN left by earlier events (ancestors first).
    Per event, exactly four RNG draws in this order: (1) rng.random() < term_share -> terminal, else
    interstitial; (2) k = rng.integers(D): a uniform block of the genome, which picks the chromosome with
    probability proportional to its block count, the arm (p or q) proportional to its block count within
    that chromosome, and, for interstitial events, the start block (uniform within the arm); (3) x =
    rng.exponential(len_mean or term_len_mean), L = max(1, round(x)) (Python round, half to even);
    (4) rng.random() < loss_share -> loss (-1 copy), else gain (+1 copy).
    Interstitial window: [k, min(k + L, arm end)) (clipped to the arm end; may touch it).
    Terminal window: L clipped to arm_blocks - 1 (never the whole arm, never across the centromere),
    anchored at the telomere end of the arm: [arm start, arm start + L) on p, [arm end - L, arm end) on q.
    Arms are [0, cent) = p and [cent, n) = q per chromosome (cent = centromere_block, SISTEM's
    round(nregions * arm_ratio) rule). Driver-blind: windows may include driver blocks.
    REJECTION (the event is counted, not applied) if for ANY affected surviving clone: a loss would take a
    block of the window to 0 copies (n_rejected_zero; D-F nullisomy rule), else a gain would take a DRIVER
    block above max_cn (n_rejected_cap; MAX_REGION_CN applies to driver blocks only, as in growth), else
    the clone's altered-driver count (driver blocks with cn != 2**(nWGD+1)) would exceed cap_altered
    (n_rejected_altered), else the clone's ploidy (mean block_cn) would leave [min_ploidy, max_ploidy]
    (n_rejected_ploidy; SISTEM's check_viability bounds, inclusive). The first failing rule in that order is
    the one counted; an event rejected for one affected clone is rejected for all.
    Known artefact, kept unchanged in prior v4.1: because one 1-copy descendant vetoes a loss, edges with many
    descendants take mostly gains (loss share 0.20 on edges with > 1000 descendants vs nominal 0.75), see
    research_notes/v4_1_design_2026-10-26/diagnostics/REPORT.md Q4.1 and the v4.1 design review.
    Passengers never change fitness: they are neutral by construction (fitness is not recomputed) and
    viability is enforced by rejection only.

EVENT ROWS
    Columns EVENT_COLUMNS (events schema 3). Passenger rows: category passenger_gain / passenger_loss,
    chrom = index in chrom_order, arm 0 = p / 1 = q, first_block/last_block = block indices within the
    chromosome (inclusive), n_blocks, copies = 1, driver = 0, gen = birth_gen of the node the edge leads
    to (the event happened at some time in (parent birth_gen, node birth_gen]; the node's birth_gen is
    recorded; trunk rows written by grow(): node 0, gen 0), terminal 0/1, node = that node id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------- events schema 3 (single source of truth)
GROWTH_EVENT_CATEGORIES = ("focal_amplification", "focal_deletion", "arm_duplication", "arm_deletion",
                           "chromosomal_duplication", "chromosomal_deletion", "WGD")   # SISTEM's, schema 1-3
PASSENGER_EVENT_CATEGORIES = ("passenger_gain", "passenger_loss")                     # schema 3 (prior v4)
EVENT_CATEGORIES = GROWTH_EVENT_CATEGORIES + PASSENGER_EVENT_CATEGORIES
EVENT_INDEX = {c: i for i, c in enumerate(EVENT_CATEGORIES)}
EVENT_COLUMNS = ("category", "chrom", "arm", "first_block", "last_block", "n_blocks", "copies", "driver", "gen",
                 "terminal", "node")
EVENTS_SCHEMA_VERSION = 3
ARM_CODE = {"p": 0, "q": 1, "w": 2}

OVERLAY_RULES = {                                          # recorded in manifest.json["passenger_overlay"]
    "trunk": "not part of the post-growth overlay: Poisson(rate * trunk_gens) passengers are applied to the FOUNDER genome "
             "before growth (simulate_gene_selection.grow, same window sampler); the root node gets no overlay events",
    "edges": "nodes in node order; non-root node: n ~ Poisson(rate * (birth_gen - parent birth_gen)); "
             "an edge event applies to every surviving clone descending from the node (incl. the node's own clone); "
             "time after a leaf's birth carries no events",
    "draws_per_event": "rng.random() < term_share -> terminal; k = rng.integers(D) (chromosome prop. to blocks, arm prop. to "
                       "arm blocks, interstitial start uniform within the arm); L = max(1, round(rng.exponential(len_mean "
                       "or term_len_mean))); rng.random() < loss_share -> loss (-1) else gain (+1)",
    "interstitial": "window [k, min(k + L, arm end)), clipped to the arm end",
    "terminal": "L clipped to arm_blocks - 1; anchored at the telomere end: [p start, p start + L) or [q end - L, q end); "
                "never the whole arm, never across the centromere",
    "driver_blind": "windows may include driver blocks",
    "rejection": "per event, over every affected surviving clone, first failing rule counted: loss to 0 copies "
                 "(n_rejected_zero, D-F); gain of a DRIVER block above max_cn (n_rejected_cap); altered drivers "
                 "(cn != 2**(nWGD+1)) above cap_altered (n_rejected_altered); ploidy (mean block_cn) outside "
                 "[min_ploidy, max_ploidy] (n_rejected_ploidy); rejected for one affected clone = rejected for all",
    "neutral": "fitness is not recomputed (passengers are neutral by construction); viability enforced by rejection only",
    "order": "events processed in node order then draw order; each sees the CN left by earlier events (ancestors first)",
    "event_rows": "category passenger_gain/passenger_loss, copies 1, driver 0, gen = birth_gen of the node, node = node id",
}


# ---------------------------------------------------------------- data classes
@dataclass
class PassengerParams:
    rate: float                      # expected passenger events per lineage per generation
    loss_share: float = 0.75         # P(loss)
    term_share: float = 0.30         # P(terminal, telomere-anchored)
    len_mean: float = 2.0            # interstitial length, blocks: max(1, round(Exp(len_mean)))
    term_len_mean: float = 4.0       # terminal length, blocks: max(1, round(Exp(term_len_mean))), <= arm - 1
    max_cn: int = 10                 # driver-block copy-number cap (MAX_REGION_CN)
    trunk_gens: float = 2000.0       # pre-growth trunk generations: Poisson(rate * trunk_gens) passengers applied to the
                                     # founder before growth by grow() (not by apply_passenger_overlay)
    min_ploidy: float = 1.5          # SISTEM's viability bounds on mean block_cn (MIN_PLOIDY, MAX_PLOIDY)
    max_ploidy: float = 8.0


@dataclass
class Genealogy:
    node_parent: np.ndarray          # int32 [n_nodes], -1 for the root
    node_birth_gen: np.ndarray       # int32 [n_nodes]
    node_clone: np.ndarray           # int32 [n_nodes], index of the surviving clone, else -1
    clone_node: np.ndarray           # int32 [n_clones]


@dataclass
class OverlayResult:
    block_cn: np.ndarray             # int16 [n_clones, D], a copy with the passenger events applied
    bulk: np.ndarray                 # float32 [D], popsize-weighted mean of block_cn
    ploidy: np.ndarray               # float32 [n_clones], mean of block_cn per clone
    event_counts_add: np.ndarray     # int32 [n_clones, 2], passenger gain / loss events along each clone's lineage
    events: np.ndarray               # int32 [n_applied, len(EVENT_COLUMNS)], one row per applied event
    n_proposed: int
    n_rejected_zero: int
    n_rejected_cap: int
    n_rejected_altered: int
    n_rejected_ploidy: int = 0

    @property
    def n_applied(self) -> int:
        """Applied events. Every node has >= 1 surviving descendant, so each applied event changes the
        block_cn of at least one surviving clone (the realised number of visible changes)."""
        return int(self.events.shape[0])

    @property
    def n_rejected(self) -> int:
        return int(self.n_rejected_zero + self.n_rejected_cap + self.n_rejected_altered + self.n_rejected_ploidy)


# ---------------------------------------------------------------- genealogy
_NAME_COUNTER = re.compile(r"(\d+)$")


def _node_key(c):
    """(birth_gen, site, creation counter) of a SISTEM Clone; the counter is the integer suffix of its name."""
    m = _NAME_COUNTER.search(str(getattr(c, "name", "") or ""))
    if m is None or c.birth_gen is None:
        raise ValueError(f"cannot order clone {c!r}: needs birth_gen and a name ending in its creation counter")
    return int(c.birth_gen), int(getattr(c, "site", 0) or 0), int(m.group(1))


def lineage_nodes(clones):
    """The ordered node objects of the genealogy (surviving clones + all ancestors), in node-id order."""
    seen = {}
    for c in clones:
        x = c
        while x is not None and id(x) not in seen:
            seen[id(x)] = x
            x = x.parent
    nodes = sorted(seen.values(), key=_node_key)
    keys = [_node_key(x) for x in nodes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (birth_gen, site, counter) keys: node order would not be deterministic")
    return nodes


def build_genealogy(clones, nodes=None) -> Genealogy:
    """Genealogy arrays from SISTEM Clone objects (uses only .parent, .birth_gen, .name, .site; nothing is mutated).
    nodes: optional precomputed lineage_nodes(clones) (saves a second walk)."""
    nodes = lineage_nodes(clones) if nodes is None else nodes
    index = {id(x): i for i, x in enumerate(nodes)}
    node_parent = np.asarray([-1 if x.parent is None else index[id(x.parent)] for x in nodes], dtype=np.int32)
    node_birth_gen = np.asarray([int(x.birth_gen) for x in nodes], dtype=np.int32)
    clone_node = np.asarray([index[id(c)] for c in clones], dtype=np.int32)
    node_clone = np.full(len(nodes), -1, dtype=np.int32)
    node_clone[clone_node] = np.arange(len(clones), dtype=np.int32)
    g = Genealogy(node_parent=node_parent, node_birth_gen=node_birth_gen, node_clone=node_clone, clone_node=clone_node)
    validate_genealogy(g)
    return g


def validate_genealogy(g: Genealogy) -> None:
    """Structural invariants: one root (node 0), parents precede children, birth_gen non-decreasing along
    edges, node_clone / clone_node are inverse maps, every node has >= 1 surviving descendant (itself included;
    nodes are by definition ancestors of surviving clones, so every applied event changes some clone)."""
    n = len(g.node_parent)
    if n == 0:
        raise ValueError("empty genealogy")
    if len(g.node_birth_gen) != n or len(g.node_clone) != n:
        raise ValueError("genealogy arrays differ in length")
    roots = np.flatnonzero(g.node_parent < 0)
    if roots.tolist() != [0]:
        raise ValueError(f"genealogy must have exactly one root at node 0, got {roots.tolist()}")
    nz = np.arange(1, n)
    if not np.all(g.node_parent[1:] < nz):
        raise ValueError("node order violates parent-before-child")
    if np.any(g.node_birth_gen[1:] < g.node_birth_gen[g.node_parent[1:]]):
        raise ValueError("a node is born before its parent")
    if not np.array_equal(g.node_clone[g.clone_node], np.arange(len(g.clone_node))):
        raise ValueError("node_clone[clone_node] is not the identity")
    if int((g.node_clone >= 0).sum()) != len(g.clone_node):
        raise ValueError("node_clone marks a node that clone_node does not reference")
    cnt = (np.asarray(g.node_clone) >= 0).astype(np.int64)
    for v in range(n - 1, 0, -1):
        cnt[g.node_parent[v]] += cnt[v]
    if np.any(cnt == 0):
        raise ValueError("a node has no surviving descendant")


def _preorder_ranges(node_parent):
    """(tin, tout): DFS preorder entry index and exclusive end of each node's subtree (children in node order)."""
    n = len(node_parent)
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[int(node_parent[v])].append(v)
    tin = np.empty(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    t, stack = 0, [0]
    while stack:
        v = stack.pop()
        tin[v] = t
        t += 1
        stack.extend(reversed(children[v]))
    for v in range(n - 1, 0, -1):                     # parents precede children -> reverse order accumulates sizes
        size[node_parent[v]] += size[v]
    return tin, tin + size


# ---------------------------------------------------------------- overlay
def _arm_table(nregions, chrom_order, centromere_block):
    """Per flat block: (chrom index, arm code, arm start, arm end, chromosome offset)."""
    D = sum(int(nregions[c]) for c in chrom_order)
    chrom_idx = np.empty(D, dtype=np.int64)
    arm_code = np.empty(D, dtype=np.int64)
    arm_lo = np.empty(D, dtype=np.int64)
    arm_hi = np.empty(D, dtype=np.int64)
    offset = np.empty(D, dtype=np.int64)
    off = 0
    for ci, c in enumerate(chrom_order):
        n, cen = int(nregions[c]), int(centromere_block[c])
        if not 0 <= cen <= n:
            raise ValueError(f"{c}: centromere block {cen} outside [0, {n}]")
        for a, (lo, hi) in enumerate(((off, off + cen), (off + cen, off + n))):
            chrom_idx[lo:hi], arm_code[lo:hi], arm_lo[lo:hi], arm_hi[lo:hi] = ci, a, lo, hi
        offset[off:off + n] = off
        off += n
    return D, chrom_idx, arm_code, arm_lo, arm_hi, offset


def _sample_flat_window(rng, pp, table):
    """One passenger window in flat block coordinates. Exactly four draws, in this order: rng.random() (terminal?),
    rng.integers(D) (block k), rng.exponential(len mean), rng.random() (loss?). Returns
    (chrom index, arm code, flat start, flat end, terminal, is_loss)."""
    D, chrom_idx, arm_code, arm_lo, arm_hi, _offset = table
    terminal = bool(rng.random() < pp.term_share)
    k = int(rng.integers(D))
    x = float(rng.exponential(pp.term_len_mean if terminal else pp.len_mean))
    loss = bool(rng.random() < pp.loss_share)
    L = max(1, round(x))
    a0, a1 = int(arm_lo[k]), int(arm_hi[k])
    if terminal:
        L = min(L, a1 - a0 - 1)
        s, e = (a0, a0 + L) if arm_code[k] == 0 else (a1 - L, a1)
    else:
        s, e = k, min(k + L, a1)
    return int(chrom_idx[k]), int(arm_code[k]), s, e, terminal, loss


def window_table(nregions, chrom_order, centromere_block):
    """Precomputed grid table for sample_passenger_window (optional; saves rebuilding it per call)."""
    return _arm_table(nregions, chrom_order, centromere_block)


def sample_passenger_window(rng, pp, nregions, chrom_order, centromere_block, table=None):
    """One passenger window with the overlay rules (module docstring; same four draws in the same order).
    Returns (chrom name, start, end, terminal, is_loss): start/end are block indices within the chromosome
    (reference coordinates, end exclusive). Used by the post-growth overlay and by the pre-growth trunk in
    simulate_gene_selection.grow()."""
    t = window_table(nregions, chrom_order, centromere_block) if table is None else table
    ci, _arm, s, e, terminal, loss = _sample_flat_window(rng, pp, t)
    off = int(t[5][s])
    return chrom_order[ci], s - off, e - off, terminal, loss


def apply_passenger_overlay(block_cn: np.ndarray, popsize: np.ndarray, genealogy: Genealogy, driver_mask: np.ndarray,
                            cap_altered: int, nregions: dict, chrom_order: list, centromere_block: dict,
                            pp: PassengerParams, rng: np.random.Generator,
                            nWGD: Optional[np.ndarray] = None) -> OverlayResult:
    """Apply neutral passenger events along the genealogy (rules in the module docstring). Inputs are not
    mutated. nWGD (optional, int per clone, default all 0) sets the unaltered reference 2**(nWGD+1) of the
    altered-driver cap, as SISTEM's check_viability does; pass the growth nWGD when WGD clones can exist."""
    B = np.array(block_cn, dtype=np.int16, copy=True)
    popsize = np.asarray(popsize)
    driver_mask = np.asarray(driver_mask, dtype=bool)
    n_clones, D_b = B.shape
    D, chrom_idx, arm_code, arm_lo, arm_hi, offset = _arm_table(nregions, chrom_order, centromere_block)
    if D_b != D or driver_mask.shape != (D,):
        raise ValueError(f"block_cn has {D_b} columns, driver_mask {driver_mask.shape}, grid D = {D}")
    if popsize.shape != (n_clones,) or len(genealogy.clone_node) != n_clones:
        raise ValueError("popsize / clone_node length differs from the number of clones")
    if not (0 <= pp.loss_share <= 1 and 0 <= pp.term_share <= 1 and pp.rate >= 0
            and pp.len_mean > 0 and pp.term_len_mean > 0 and pp.trunk_gens >= 0 and pp.min_ploidy <= pp.max_ploidy):
        raise ValueError(f"invalid PassengerParams {pp}")
    arm_len = arm_hi - arm_lo
    if pp.term_share > 0 and np.any(arm_len < 2):
        raise ValueError("terminal events need every non-empty arm to have >= 2 blocks")
    validate_genealogy(genealogy)
    nwgd = np.zeros(n_clones, dtype=np.int64) if nWGD is None else np.asarray(nWGD, dtype=np.int64)
    ref = (2 ** (nwgd + 1)).astype(np.int32)                         # unaltered CN per clone

    parent = genealogy.node_parent
    bgen = genealogy.node_birth_gen
    tin, tout = _preorder_ranges(parent)
    clone_tin = tin[genealogy.clone_node]
    order = np.argsort(clone_tin, kind="stable")
    sorted_tin = clone_tin[order]

    altered = (B[:, driver_mask].astype(np.int32) != ref[:, None]).sum(axis=1)   # current altered-driver counts
    cn_sum = B.astype(np.int64).sum(axis=1)                         # current total copies per clone (ploidy * D)
    counts_add = np.zeros((n_clones, 2), dtype=np.int32)
    rows = []
    n_prop = n_zero = n_cap = n_alt = n_ploidy = 0
    gain_code, loss_code = EVENT_INDEX["passenger_gain"], EVENT_INDEX["passenger_loss"]

    table = (D, chrom_idx, arm_code, arm_lo, arm_hi, offset)
    for v in range(1, len(parent)):                                 # root: no events (the trunk is applied before growth)
        dt = int(bgen[v]) - int(bgen[parent[v]])
        n_ev = int(rng.poisson(pp.rate * dt))
        if n_ev == 0:
            continue
        lo_i = np.searchsorted(sorted_tin, tin[v], side="left")
        hi_i = np.searchsorted(sorted_tin, tout[v], side="left")
        aff = order[lo_i:hi_i]                                      # surviving clones descending from v
        for _ in range(n_ev):
            ci, arm, s, e, terminal, loss = _sample_flat_window(rng, pp, table)
            n_prop += 1
            cols = np.arange(s, e)
            sub = B[np.ix_(aff, cols)].astype(np.int32)
            d = -1 if loss else 1
            if loss and (sub <= 1).any():
                n_zero += 1
                continue
            dm = driver_mask[cols]
            new_alt = None
            if dm.any():
                sd = sub[:, dm]
                if not loss and (sd + 1 > pp.max_cn).any():
                    n_cap += 1
                    continue
                r = ref[aff][:, None]
                new_alt = altered[aff] - (sd != r).sum(axis=1) + (sd + d != r).sum(axis=1)
                if (new_alt > cap_altered).any():
                    n_alt += 1
                    continue
            new_sum = cn_sum[aff] + d * (e - s)
            new_ploidy = new_sum / D
            if (new_ploidy < pp.min_ploidy).any() or (new_ploidy > pp.max_ploidy).any():
                n_ploidy += 1
                continue
            if new_alt is not None:
                altered[aff] = new_alt
            cn_sum[aff] = new_sum
            B[np.ix_(aff, cols)] = (sub + d).astype(np.int16)
            counts_add[aff, 1 if loss else 0] += 1
            off = int(offset[s])
            rows.append([loss_code if loss else gain_code, ci, arm, s - off, e - 1 - off,
                         e - s, 1, 0, int(bgen[v]), int(terminal), v])

    events = np.asarray(rows, dtype=np.int32).reshape(-1, len(EVENT_COLUMNS))
    ploidy = B.astype(np.float64).mean(axis=1).astype(np.float32)
    bulk = np.average(B.astype(np.float64), axis=0, weights=popsize).astype(np.float32)
    return OverlayResult(block_cn=B, bulk=bulk, ploidy=ploidy, event_counts_add=counts_add, events=events,
                         n_proposed=n_prop, n_rejected_zero=n_zero, n_rejected_cap=n_cap, n_rejected_altered=n_alt,
                         n_rejected_ploidy=n_ploidy)


def _selftest():
    """Tiny import-free sanity run (python passenger_overlay.py)."""
    g = Genealogy(node_parent=np.array([-1, 0, 1, 1], np.int32), node_birth_gen=np.array([0, 10, 20, 30], np.int32),
                  node_clone=np.array([-1, 0, 1, 2], np.int32), clone_node=np.array([1, 2, 3], np.int32))
    B = np.full((3, 10), 2, np.int16)
    res = apply_passenger_overlay(B, np.array([5, 3, 2]), g, np.zeros(10, bool), 0, {"chrA": 10}, ["chrA"], {"chrA": 4},
                                  PassengerParams(rate=0.2), np.random.default_rng(1))
    assert res.n_applied + res.n_rejected == res.n_proposed and (res.block_cn >= 1).all()

    def n_desc(v):                                  # brute force: surviving clones whose ancestor chain hits v
        n = 0
        for x in g.clone_node:
            while x >= 0 and x != v:
                x = g.node_parent[x]
            n += int(x == v)
        return n
    assert int(res.event_counts_add.sum()) == sum(n_desc(int(r[-1])) for r in res.events)
    return res


if __name__ == "__main__":
    r = _selftest()
    print(f"passenger_overlay self-test ok: proposed {r.n_proposed}, applied {r.n_applied}, rejected {r.n_rejected}")
