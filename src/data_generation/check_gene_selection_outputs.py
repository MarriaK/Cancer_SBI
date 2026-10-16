#!/usr/bin/env python
"""Pilot acceptance checks (C1-C7) for the SISTEM region-model simulator.

Read-only. It runs over the output tree written by
``simulate_gene_selection.py`` and checks it against the file contract:

    <root>/manifest.json
    <root>/simulation_outputs/sim<ID>/parameters.pkl
    <root>/simulation_outputs/sim<ID>/<t>/results.pkl
    <root>/simulation_outputs/sim<ID>/<t>/clones.npz   (only when complete)
    <root>/simulation_outputs/sim<ID>/<t>/sim.log

    python check_gene_selection_outputs.py --root ~/cancer/data/gene_pilot
    python check_gene_selection_outputs.py --root <dir> --out-dir reports --max-sims 20

Writes ``<out-dir>/gene_pilot_checks.txt`` (same text as stdout). Exit status
is 0 when every check is PASS or INFO and 1 when any check FAILs, so it can
gate the next step of a job script. 2 means the root itself is unusable.

Block flattening: block index = sum(nregions[c'] for c' before c in
chrom_order) + r, the same order as ``driver_mask`` and ``block_cn`` columns.

The checks
----------
C1  No crash, retry rate <= 10%. Over every trial found, the retry rate is the
    fraction with failures[0] + failures[1] > 0. Incomplete trials
    (complete == False, or results.pkl unreadable) are counted too, and so
    are trials that were never written: per sim, max(0, ntrials_used - trial
    dirs found). A sim directory without a readable parameters.pkl counts as
    one failed sim (rank 0 pre-creates every sim dir, so this is what a dead
    rank leaves). FAIL if retry rate > 0.10, any trial is incomplete or
    missing, any sim lacks parameters.pkl, or any driver_mask length != D.

C2  Sign-recovery sanity, on the LARGEST clone of each complete trial
    (argmax(popsize) in clones.npz; the largest clone is not stored separately).
    With p = 2**(nWGD + 1) for that clone, among driver blocks with delta > 0
    report the fraction with block_cn > p, among delta < 0 the fraction with
    block_cn < p, and the mean (block_cn - p) of each group, pooled over
    trials. PASS if mean(block_cn - p | delta > 0) > mean(block_cn - p | delta < 0).

C3  Arm averages from blocks. For every clone, the 2 * len(chrom_order) arm
    averages (44 for chr1-22) are computed from block_cn with the manifest's
    ``centromere_block``: p = mean of blocks [0, cent), q = mean of [cent, n),
    which is how SISTEM's arm library splits a chromosome
    (cent_idx = round(nregions * arm_ratio)). Reports the shape, the fraction
    of arm averages exactly 0 (complete arm loss; the arm dataset had about a
    fifth), and min/max. Also pools ``driver_count_mismatches`` from
    results.pkl. PASS if mismatches == 0 and no arm average is NaN. There is
    no external reference inside this script, so the numbers are INFO.

C4  n_altered_drivers < cap (strictly) for > 95% of surviving clones
    (popsize > 0), cap from parameters.pkl. Reports the fraction and the
    distribution of n_altered_drivers / cap. FAIL if the fraction < 0.95.

C5  Every growth-phase focal event overlaps >= 1 driver block. NOT checkable
    from saved outputs: SISTEM does not record individual events, only the
    resulting copy numbers. Printed as INFO and skipped; it needs an
    instrumented run: requires per-event instrumentation of SISTEM's mutation functions; not checkable from saved outputs.

C6  CN and ploidy bounds. No DRIVER block may exceed fixed_parameters
    max_region_CN (SISTEM applies that cap to driver regions only; non-driver
    blocks can legitimately go higher through arm/chromosome events, so their
    max is reported separately as INFO). Every clone ploidy must lie in
    [min_ploidy, max_ploidy] (inclusive). FAIL on any violation.

C7  Cost. wall_seconds mean/median/max, mean bytes per trial of clones.npz +
    results.pkl + sim.log, extrapolated to 6,000 x 25 and 10,000 x 25 trials.
    INFO.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RETRY_RATE_MAX = 0.10
CAP_FRACTION_MIN = 0.95
EXTRAPOLATE_THETAS = (6000, 10000)
EXTRAPOLATE_TRIALS = 25
CLONE_ARRAYS = ["block_cn", "popsize", "ploidy", "fitness", "nWGD", "n_altered_drivers", "bulk"]
# bulk is (D,) not (n_clones,): the popsize-weighted mean profile of the trial.


def sim_sort_key(name: str) -> tuple:
    """Sort sim10 after sim9, not after sim1 (same as update_checker.py)."""
    digits = "".join(c for c in name if c.isdigit())
    return (0, int(digits)) if digits else (1, 0), name


def is_trial_folder(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def fmt(x, nd=4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not np.isfinite(x):
        return str(x)
    return f"{float(x):.{nd}g}"


def dist(a) -> str:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "n/a (empty)"
    return (f"min {fmt(a.min())}  median {fmt(np.median(a))}  "
            f"p95 {fmt(np.percentile(a, 95))}  max {fmt(a.max())}")


# ----------------------------------------------------------------- loading

def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_trial(trial_dir: Path) -> Dict:
    """Read one trial. Never raises; problems go into trial['errors']."""
    trial = {"dir": trial_dir, "name": trial_dir.name, "errors": [],
             "results": None, "clones": None}
    try:
        trial["results"] = load_pickle(trial_dir / "results.pkl")
    except Exception as exc:  # missing, empty or corrupt pickle
        trial["errors"].append(f"results.pkl unreadable: {exc!r}")
        return trial
    if trial["results"].get("complete"):
        try:
            with np.load(trial_dir / "clones.npz") as z:
                missing = [k for k in CLONE_ARRAYS if k not in z.files]
                if missing:
                    trial["errors"].append(f"clones.npz missing arrays {missing}")
                else:
                    trial["clones"] = {k: z[k] for k in CLONE_ARRAYS}
        except Exception as exc:
            trial["errors"].append(f"clones.npz unreadable: {exc!r}")
    return trial


def load_sim(sim_dir: Path) -> Dict:
    sim = {"dir": sim_dir, "name": sim_dir.name, "errors": [], "params": None, "trials": []}
    try:
        sim["params"] = load_pickle(sim_dir / "parameters.pkl")
    except Exception as exc:
        sim["errors"].append(f"parameters.pkl unreadable: {exc!r}")
    trial_dirs = sorted((p for p in sim_dir.iterdir() if is_trial_folder(p)),
                        key=lambda p: int(p.name))
    sim["trials"] = [load_trial(t) for t in trial_dirs]
    return sim


def is_complete(trial: Dict) -> bool:
    return (trial["results"] is not None and bool(trial["results"].get("complete"))
            and trial["clones"] is not None)


def driver_delta_vector(params: Dict, manifest: Dict) -> np.ndarray:
    """delta for every block in chrom_order flattening (length D)."""
    out = []
    for c in manifest["chrom_order"]:
        d = list(params["delta_dict"][c])
        if len(d) != manifest["nregions"][c]:
            raise ValueError(f"delta_dict[{c}] has {len(d)} entries, "
                             f"nregions says {manifest['nregions'][c]}")
        out.extend(d)
    return np.asarray(out, dtype=float)


# ------------------------------------------------------------------ checks

def check_c1(sims: List[Dict], manifest: Dict) -> Tuple[str, List[str]]:
    n_trials = n_retry = n_incomplete = 0
    sum_failed_sim = sum_failed_call = 0
    n_no_params = n_missing_trials = n_bad_mask = 0
    D = manifest["D"]
    bad = []
    for sim in sims:
        params = sim["params"]
        if not isinstance(params, dict) or "ntrials_used" not in params:
            n_no_params += 1
            why = "; ".join(sim["errors"]) or "parameters.pkl has no ntrials_used"
            bad.append(f"    {sim['name']}: no usable parameters.pkl ({why})")
        else:
            missing = max(0, int(params["ntrials_used"]) - len(sim["trials"]))
            if missing > 0:
                n_missing_trials += missing
                bad.append(f"    sim{params.get('sim_id', sim['name'])}: ntrials_used "
                           f"{params['ntrials_used']}, trial dirs found {len(sim['trials'])}")
            mlen = len(params.get("driver_mask", []))
            if mlen != D:
                n_bad_mask += 1
                bad.append(f"    {sim['name']}: driver_mask length {mlen} != D {D}")
        for t in sim["trials"]:
            n_trials += 1
            res = t["results"]
            if res is not None:
                f = list(res.get("failures") or [0, 0]) + [0, 0]
                fs, fc = int(f[0]), int(f[1])
                sum_failed_sim += fs
                sum_failed_call += fc
                if fs + fc > 0:
                    n_retry += 1
            if not is_complete(t):
                n_incomplete += 1
                why = "; ".join(t["errors"]) or "complete == False"
                bad.append(f"    {sim['name']}/{t['name']}: {why}")
    rate = n_retry / n_trials if n_trials else float("nan")
    lines = [f"trials found {n_trials}; with a retry {n_retry}; retry rate {fmt(rate)} "
             f"(limit {RETRY_RATE_MAX})",
             f"total failed_sim {sum_failed_sim}; total failed_call {sum_failed_call}",
             f"incomplete trials {n_incomplete}",
             f"sims without parameters.pkl: {n_no_params}",
             f"trials missing vs ntrials_used: {n_missing_trials}",
             f"sims with driver_mask length != D: {n_bad_mask}"]
    lines += bad[:20] + ([f"    ... and {len(bad) - 20} more"] if len(bad) > 20 else [])
    ok = (n_trials > 0 and rate <= RETRY_RATE_MAX and n_incomplete == 0
          and n_no_params == 0 and n_missing_trials == 0 and n_bad_mask == 0)
    return ("PASS" if ok else "FAIL"), lines


def check_c2(sims: List[Dict], manifest: Dict) -> Tuple[str, List[str]]:
    pos, neg = [], []  # block_cn - p, pooled
    n_used = 0
    for sim in sims:
        if sim["params"] is None:
            continue
        delta = driver_delta_vector(sim["params"], manifest)
        mask = np.asarray(sim["params"]["driver_mask"], dtype=bool)
        for t in sim["trials"]:
            if not is_complete(t):
                continue
            cl = t["clones"]
            idx = int(np.argmax(cl["popsize"]))     # largest clone; first maximum on ties
            p = 2 ** (int(cl["nWGD"][idx]) + 1)
            cn = cl["block_cn"][idx].astype(float)
            pos.append(cn[mask & (delta > 0)] - p)
            neg.append(cn[mask & (delta < 0)] - p)
            n_used += 1
    pos = np.concatenate(pos) if pos else np.empty(0)
    neg = np.concatenate(neg) if neg else np.empty(0)
    m_pos = pos.mean() if pos.size else float("nan")
    m_neg = neg.mean() if neg.size else float("nan")
    lines = [f"largest clones used {n_used}",
             f"delta>0 driver blocks {pos.size}: frac cn > p {fmt((pos > 0).mean() if pos.size else None)}; "
             f"mean(cn - p) {fmt(m_pos)}",
             f"delta<0 driver blocks {neg.size}: frac cn < p {fmt((neg < 0).mean() if neg.size else None)}; "
             f"mean(cn - p) {fmt(m_neg)}"]
    ok = bool(pos.size and neg.size and m_pos > m_neg)
    if not (pos.size and neg.size):
        lines.append("no blocks in one sign group: cannot compare")
    return ("PASS" if ok else "FAIL"), lines


def arm_averages(block_cn: np.ndarray, manifest: Dict) -> np.ndarray:
    """(n_clones, 2 * n_chrom) arm means, ordered chr1p, chr1q, chr2p, ..."""
    cols, start = [], 0
    for c in manifest["chrom_order"]:
        n = manifest["nregions"][c]
        cent = manifest["centromere_block"][c]
        chrom = block_cn[:, start:start + n].astype(float)
        for arm in (chrom[:, :cent], chrom[:, cent:]):
            cols.append(arm.mean(axis=1) if arm.shape[1] else np.full(chrom.shape[0], np.nan))
        start += n
    return np.stack(cols, axis=1)


def check_c3(sims: List[Dict], manifest: Dict) -> Tuple[str, List[str]]:
    lines = []
    # Cross-check the manifest's centromere blocks against SISTEM's own rule.
    disagree, empty = [], []
    for c in manifest["chrom_order"]:
        n, cent = manifest["nregions"][c], manifest["centromere_block"][c]
        if "arm_ratios" in manifest and round(n * manifest["arm_ratios"][c]) != cent:
            disagree.append(f"{c}: manifest {cent} vs round({n}*{manifest['arm_ratios'][c]}) "
                            f"= {round(n * manifest['arm_ratios'][c])}")
        if cent <= 0 or cent >= n:
            empty.append(c)
    if disagree:
        lines.append("WARNING centromere_block disagrees with round(nregions*arm_ratio): "
                     + "; ".join(disagree))
    if empty:
        lines.append(f"chromosomes with an empty arm (gives NaN): {empty}")

    arms, mismatches, n_mm_missing = [], 0, 0
    for sim in sims:
        for t in sim["trials"]:
            if not is_complete(t):
                continue
            arms.append(arm_averages(t["clones"]["block_cn"], manifest))
            mm = t["results"].get("driver_count_mismatches")
            if mm is None:
                n_mm_missing += 1
            else:
                mismatches += int(mm)
    if not arms:
        return "FAIL", lines + ["no complete trials to compute arm averages from"]
    A = np.concatenate(arms, axis=0)
    n_nan = int(np.isnan(A).sum())
    lines += [f"arm-average matrix shape {A.shape} (expected (n_clones, {2 * len(manifest['chrom_order'])}))",
              f"fraction exactly 0 (complete arm loss) {fmt((A == 0).mean())}  [arm dataset ~0.2]",
              f"min {fmt(np.nanmin(A))}  max {fmt(np.nanmax(A))}  NaN count {n_nan}",
              f"pooled driver_count_mismatches {mismatches} (expected 0)"
              + (f"; {n_mm_missing} complete trials had None" if n_mm_missing else "")]
    ok = mismatches == 0 and n_nan == 0  # a centromere disagreement is a warning only
    return ("PASS" if ok else "FAIL"), lines


def check_c4(sims: List[Dict]) -> Tuple[str, List[str]]:
    ratios, below, total = [], 0, 0
    for sim in sims:
        if sim["params"] is None:
            continue
        cap = sim["params"]["cap"]
        for t in sim["trials"]:
            if not is_complete(t):
                continue
            cl = t["clones"]
            alive = cl["popsize"] > 0
            nad = cl["n_altered_drivers"][alive].astype(float)
            total += nad.size
            below += int((nad < cap).sum())
            ratios.append(nad / cap if cap > 0 else np.full(nad.shape, np.inf))
    frac = below / total if total else float("nan")
    r = np.concatenate(ratios) if ratios else np.empty(0)
    lines = [f"surviving clones {total}; with n_altered_drivers < cap {below}; "
             f"fraction {fmt(frac)} (need >= {CAP_FRACTION_MIN})",
             f"n_altered_drivers / cap: {dist(r)}"]
    return ("PASS" if total and frac >= CAP_FRACTION_MIN else "FAIL"), lines


def check_c5() -> Tuple[str, List[str]]:
    return "INFO", ["requires per-event instrumentation of SISTEM's mutation functions; not checkable from saved outputs",
                    "(focal events are not recorded in saved outputs; check skipped)"]


def check_c6(sims: List[Dict], manifest: Dict) -> Tuple[str, List[str]]:
    fp = manifest["fixed_parameters"]
    cap_cn, lo, hi = fp["max_region_CN"], fp["min_ploidy"], fp["max_ploidy"]
    drv_max = nondrv_max = None
    n_drv_over = n_nondrv_over = n_ploidy_bad = n_clones = 0
    pl_all = []
    for sim in sims:
        if sim["params"] is None:
            continue
        mask = np.asarray(sim["params"]["driver_mask"], dtype=bool)
        for t in sim["trials"]:
            if not is_complete(t):
                continue
            cl = t["clones"]
            cn = cl["block_cn"]
            n_clones += cn.shape[0]
            if mask.any():
                m = int(cn[:, mask].max())
                drv_max = m if drv_max is None else max(drv_max, m)
                n_drv_over += int((cn[:, mask] > cap_cn).sum())
            if (~mask).any():
                m = int(cn[:, ~mask].max())
                nondrv_max = m if nondrv_max is None else max(nondrv_max, m)
                n_nondrv_over += int((cn[:, ~mask] > cap_cn).sum())
            pl = cl["ploidy"].astype(float)
            pl_all.append(pl)
            n_ploidy_bad += int(((pl < lo) | (pl > hi)).sum())
    pl_all = np.concatenate(pl_all) if pl_all else np.empty(0)
    lines = [f"clones {n_clones}; max_region_CN {cap_cn}; ploidy bounds [{lo}, {hi}]",
             f"driver blocks: max CN {fmt(drv_max)}; cells > {cap_cn}: {n_drv_over}",
             f"non-driver blocks (INFO): max CN {fmt(nondrv_max)}; cells > {cap_cn}: {n_nondrv_over}",
             f"ploidy: {dist(pl_all)}; outside bounds {n_ploidy_bad}"]
    ok = n_clones > 0 and n_drv_over == 0 and n_ploidy_bad == 0
    return ("PASS" if ok else "FAIL"), lines


def check_c7(sims: List[Dict]) -> Tuple[str, List[str]]:
    walls, sizes = [], []
    for sim in sims:
        for t in sim["trials"]:
            res = t["results"]
            if res is not None and res.get("wall_seconds") is not None:
                walls.append(float(res["wall_seconds"]))
            sizes.append(sum((t["dir"] / f).stat().st_size
                             for f in ("clones.npz", "results.pkl", "sim.log")
                             if (t["dir"] / f).exists()))
    w = np.asarray(walls)
    lines = []
    if w.size:
        lines.append(f"wall_seconds over {w.size} trials: mean {fmt(w.mean())}  "
                     f"median {fmt(np.median(w))}  max {fmt(w.max())}")
    else:
        lines.append("wall_seconds: none recorded")
    if sizes:
        mb = float(np.mean(sizes))
        lines.append(f"bytes per trial (clones.npz + results.pkl + sim.log): mean {mb:,.0f}")
        for n_theta in EXTRAPOLATE_THETAS:
            n = n_theta * EXTRAPOLATE_TRIALS
            hours = f"; CPU-hours {w.mean() * n / 3600:,.0f}" if w.size else ""
            lines.append(f"extrapolated {n_theta:,} theta x {EXTRAPOLATE_TRIALS} = {n:,} trials: "
                         f"{mb * n / 1e9:,.2f} GB{hours}")
    return "INFO", lines


# -------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Output directory holding manifest.json and simulation_outputs/.")
    p.add_argument("--out-dir", type=Path, default=Path("reports"),
                   help="Where to write gene_pilot_checks.txt (default: reports).")
    p.add_argument("--max-sims", type=int, default=None,
                   help="Check only the first N simulations (in sim-number order).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser()
    sim_root = root / "simulation_outputs"
    if not (root / "manifest.json").is_file() or not sim_root.is_dir():
        print(f"error: need {root}/manifest.json and {sim_root}/", file=sys.stderr)
        return 2
    manifest = json.loads((root / "manifest.json").read_text())

    sim_dirs = sorted((p for p in sim_root.iterdir() if p.is_dir() and p.name.startswith("sim")),
                      key=lambda p: sim_sort_key(p.name))
    if args.max_sims is not None:
        sim_dirs = sim_dirs[:args.max_sims]
    if not sim_dirs:
        print(f"error: no sim*/ directories under {sim_root}", file=sys.stderr)
        return 2

    sims = []
    for n, d in enumerate(sim_dirs, 1):
        sims.append(load_sim(d))
        if n % 50 == 0 or n == len(sim_dirs):
            print(f"  loaded {n}/{len(sim_dirs)}", file=sys.stderr, flush=True)

    out: List[str] = []
    n_trials = sum(len(s["trials"]) for s in sims)
    n_complete = sum(is_complete(t) for s in sims for t in s["trials"])
    out += ["=" * 70,
            f"Root: {root}",
            f"dataset_kind {manifest.get('dataset_kind')}; sistem_version {manifest.get('sistem_version')}",
            f"D {manifest['D']}; sum(nregions) {sum(manifest['nregions'].values())}; "
            f"chromosomes {len(manifest['chrom_order'])}",
            f"ntrials_policy {json.dumps(manifest.get('ntrials_policy'))}",
            f"sims found {len(sims)}; trials found {n_trials}; complete trials {n_complete}",
            "=" * 70, "",
            "Per-theta summary:",
            f"  {'sim_id':>8} {'n_drivers':>9} {'cap':>5} {'ntrials_used':>12} {'trials':>6} {'n_complete':>10}"]
    if manifest["D"] != sum(manifest["nregions"].values()):
        out.append("  WARNING: manifest D != sum(nregions)")
    for s in sims:
        p = s["params"] or {}
        out.append(f"  {str(p.get('sim_id', s['name'])):>8} {str(p.get('n_drivers', '?')):>9} "
                   f"{str(p.get('cap', '?')):>5} {str(p.get('ntrials_used', '?')):>12} "
                   f"{len(s['trials']):>6} {sum(is_complete(t) for t in s['trials']):>10}")
        for e in s["errors"]:
            out.append(f"    ! {e}")
        if p and len(p.get("driver_mask", [])) != manifest["D"]:
            out.append(f"    ! driver_mask length {len(p.get('driver_mask', []))} != D {manifest['D']}")
    out.append("")

    checks = [("C1 no crash / retry rate", lambda: check_c1(sims, manifest)),
              ("C2 sign recovery (largest clone)", lambda: check_c2(sims, manifest)),
              ("C3 arm averages from blocks", lambda: check_c3(sims, manifest)),
              ("C4 n_altered_drivers < cap", lambda: check_c4(sims)),
              ("C5 focal events hit a driver", check_c5),
              ("C6 driver CN / ploidy bounds", lambda: check_c6(sims, manifest)),
              ("C7 wall time and bytes", lambda: check_c7(sims))]
    statuses = []
    for title, fn in checks:
        try:
            status, lines = fn()
        except Exception as exc:  # a malformed file must not hide the other checks
            status, lines = "FAIL", [f"check crashed: {exc!r}"]
        statuses.append((title, status))
        out.append(f"[{status}] {title}")
        out += [f"    {ln}" for ln in lines]
        out.append("")

    out.append("Summary: " + "  ".join(f"{t.split()[0]}={s}" for t, s in statuses))
    any_fail = any(s == "FAIL" for _, s in statuses)
    out.append("OVERALL: " + ("FAIL" if any_fail else "PASS"))

    text = "\n".join(out) + "\n"
    print(text, end="")
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "gene_pilot_checks.txt"
    report.write_text(text)
    print(f"Saved: {report}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
