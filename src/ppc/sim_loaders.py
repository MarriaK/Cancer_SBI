"""Load simulated tumours (simulate_gene_selection.py output) into the same per-profile format the
real-data loaders produce, so the decomposition and the summary statistics run identically on both.

A dataset directory looks like
    <root>/manifest.json
    <root>/simulation_outputs/sim{ID}/parameters.pkl
    <root>/simulation_outputs/sim{ID}/{t}/clones.npz   (only for completed trials)

Two profiles per trial are offered: the LARGEST clone's integer copy number (the analogue of a
clonal ASCAT/ABSOLUTE call) and the cell-weighted mean copy number ``bulk`` (the analogue of a
Battenberg cell-weighted profile). Masked blocks do not exist in simulations (coverage is 1).
"""
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ppc.grid import Grid, load_grid


@dataclass
class SimCohort:
    block_cn_int: np.ndarray      # int16 [n, D], largest clone of each trial
    block_cn_mean: np.ndarray     # float32 [n, D], cell-weighted mean over clones (bulk)
    meta: pd.DataFrame            # sim_id, trial, n_clones, largest_frac, ploidy_largest, n_drivers, pi, q, m, arm_rate, chromosomal_rate, nWGD_largest
    grid: Grid
    clones: list[dict] | None = None   # optional per-trial dicts with the raw arrays (block_cn, popsize, event_counts, events)

    def subset(self, mask: np.ndarray) -> "SimCohort":
        idx = np.flatnonzero(mask)
        return SimCohort(self.block_cn_int[idx], self.block_cn_mean[idx], self.meta.iloc[idx].reset_index(drop=True),
                         self.grid, None if self.clones is None else [self.clones[i] for i in idx])


def _sim_ids(root: str) -> list[int]:
    dirs = glob.glob(os.path.join(root, "simulation_outputs", "sim*"))
    return sorted(int(os.path.basename(d)[3:]) for d in dirs if os.path.basename(d)[3:].isdigit())


def load_sim_cohort(root: str, grid: Grid | None = None, one_trial_per_theta: int | None = None,
                    keep_clones: bool = False, sim_ids: list[int] | None = None) -> SimCohort:
    """Load every completed trial under ``root``.

    one_trial_per_theta: if an int, keep only that trial number (1-based) per theta, which gives
    independent prior draws for cohort-level tests; None keeps all trials (replicates share theta).
    keep_clones: also retain the raw per-trial arrays (memory: ~a few MB per trial).
    """
    grid = grid or load_grid(os.path.join(root, "manifest.json"))
    ints, means, rows, raws = [], [], [], []
    for sid in (sim_ids or _sim_ids(root)):
        sdir = os.path.join(root, "simulation_outputs", f"sim{sid}")
        try:
            with open(os.path.join(sdir, "parameters.pkl"), "rb") as f:
                P = pickle.load(f)
        except FileNotFoundError:
            continue
        for npz_path in sorted(glob.glob(os.path.join(sdir, "*", "clones.npz")), key=lambda p: int(os.path.basename(os.path.dirname(p)))):
            t = int(os.path.basename(os.path.dirname(npz_path)))
            if one_trial_per_theta is not None and t != one_trial_per_theta:
                continue
            z = np.load(npz_path)
            block_cn, pop = z["block_cn"], z["popsize"]
            assert block_cn.shape[1] == grid.D, f"{npz_path}: {block_cn.shape[1]} blocks, grid has {grid.D}"
            i = int(np.argmax(pop))
            ints.append(block_cn[i].astype(np.int16))
            means.append(z["bulk"].astype(np.float32))
            rows.append(dict(sim_id=sid, trial=t, n_clones=int(len(pop)), largest_frac=float(pop[i] / pop.sum()),
                             ploidy_largest=float(z["ploidy"][i]), nWGD_largest=int(z["nWGD"][i]),
                             n_drivers=int(P["n_drivers"]), pi=float(P["pi"]), q=float(P["q"]), m=float(P["m"]),
                             arm_rate=float(P["arm_rate"]), chromosomal_rate=float(P["chromosomal_rate"]),
                             focal_driver_rate=float(P.get("focal_driver_rate", 5e-4)), length_mean=float(P.get("length_mean", 1.5)),
                             prior_version=int(P.get("prior_version", 1)),
                             has_events="events" in z.files))
            if keep_clones:
                raws.append({k: z[k] for k in z.files})
    if not rows:
        raise FileNotFoundError(f"no completed trials under {root}")
    return SimCohort(np.stack(ints), np.stack(means), pd.DataFrame(rows), grid, raws if keep_clones else None)


def driver_mask_for(root: str, sim_id: int) -> np.ndarray:
    with open(os.path.join(root, "simulation_outputs", f"sim{sim_id}", "parameters.pkl"), "rb") as f:
        return np.asarray(pickle.load(f)["driver_mask"], dtype=bool)
