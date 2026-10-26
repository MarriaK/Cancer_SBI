# data_generation: SISTEM simulators

Two simulator families, one per SISTEM selection model. Both share the same MPI
layout, retry loop (up to 5 attempts per trial), per-trial seeding and CLI
(`--start --end --ntrials --output --seed`), so the two scripts can be diffed.

| | Arm model | Gene (region) model |
|---|---|---|
| Simulator | `simulate_arm_selection.py` (was `dataSimulator_MPI.py`) | `simulate_gene_selection.py` |
| SISTEM library | `FittedArmLibrary`, 44 arm coefficients | `FittedRegionLibrary`, one coefficient per 5 Mb block (D = 589) |
| Prior | Normal(0, 0.2) on arm deltas; rates uniform; focal rate and length fixed | sparse spike-and-slab on block deltas; event rates, focal rate and focal length log-uniform and drawn per theta; prior v3 (2026-10-22): q ~ U(0.30, 0.55), focal rate logU(1e-4, 1e-3), plus the nuisance parameter p_term ~ U(0.35, 0.75) drawn last; prior v4 (2026-10-23, below): lower rate floors, p_term floor 0.25, rates coupled by a shared instability level, neutral passenger overlay; prior v4.1 (2026-10-26, `prior_version` 5, below): trunk 1000, passenger floor 1e-4, one copula loading per rate with the whole-chromosome rate decoupled (see script docstring; `prior_version` in manifest and parameters.pkl) |
| Per-trial output | `results.pkl` (largest-clone + bulk arm CN ratios), `CNratios_all.pkl.gz` | `results.pkl` (bookkeeping only), `clones.npz` (per-clone block CNs + popsize, ploidy, fitness, nWGD, n_altered_drivers, the bulk profile, and the TRUE event history: `event_counts` per clone per category and the unique `events` table of all surviving lineages, 10 columns since prior v3 incl. `terminal`, 11 since prior v4 incl. `node`, plus the genealogy arrays; read columns by name via `event_columns`; largest clone = argmax popsize) |
| Checker | `update_checker.py` | `check_gene_selection_outputs.py` |
| Repair | `fixed_dataSimulation.py` | none yet |

`simulate_arm_selection.py` is the provenance record of the existing arm dataset
and is kept behaviour-identical (its known defects included); do not edit it.

## Gene-model layout

    <output>/manifest.json                       grid, prior, fixed params, seeding (rank 0, once)
    <output>/simulation_outputs/sim{ID}/parameters.pkl
    <output>/simulation_outputs/sim{ID}/{t}/results.pkl   always written (bookkeeping: failures, complete, n_clones, wall time)
    <output>/simulation_outputs/sim{ID}/{t}/clones.npz    only for completed trials (all data: block_cn, popsize, ..., bulk)
    <output>/simulation_outputs/sim{ID}/{t}/sim.log       written by SISTEM (omitted with --no-sim-log; manifest["sim_log"])

Block vectors are flattened chr1..chr22 in numeric order (`manifest["chrom_order"]`).

**SISTEM runtime patches.** The gene script does not edit SISTEM; `apply_sistem_patches()`
replaces three `BaseRegionLibrary` methods, `Cell.focal_deletion` and `mutation.gen_focal_event`
in memory at import time (same effect with the pip-installed package on the cluster),
self-tests each one, and records them in `manifest["sistem_patches"]`:

1. `get_driver_start_regions`: off-by-one (`sequence[:-size]`) meant the last block of every
   chromosome could never receive a focal event.
2. `compute_fitness`: exponent `n / round(p)` as in CINner (which SISTEM's region model is
   adapted from) instead of `n / p` with continuous ploidy; the continuous form rewarded
   deleting neutral DNA and tumours collapsed to the 1.5 ploidy floor (calibration 2026-10-19).
   (D-E. Superseded since prior v3 by patch 5, whose body keeps this rounding.)
3. `check_viability`: a clone with any block at 0 copies is non-viable (CINner `nullisomy_max = 0`).
4. `Cell.focal_deletion`: SISTEM recorded a deletion's `gene_ids` *after* deleting (so the record held the
   wrong blocks); the slice is captured first. Bookkeeping only; needed for the event tables.
5. `compute_fitness` again (D-I, `region_gain_dosage_saturation`, prior v3): gain drivers
   (delta > 0) saturate, exponent `min(n, round(p) + 1) / round(p)`; loss drivers unchanged;
   extra copies are neutral, not lethal. Removes the ploidy / high-copy excess of v2.
6. `mutation.gen_focal_event` wrapper (D-H, `focal_terminal_events`, prior v3): with probability
   `p_term` a driver focal event is end-anchored (homolog end to the target driver plus
   floor(Exp(2)) blocks toward the centromere, strictly inside the arm); gains add 1 copy.
   Passenger calls and `p_term = 0` delegate with no extra RNG draw. Recorded as `terminal = 1`
   in the events table. Adopted only together with patch 5; advisor sign-off pending.

**Prior v4 (2026-10-23, Stage 1 provisional; `PRIOR_VERSION = 4`; restored by `V4_SETTINGS` since v4.1).** Design and review:
`cancer/research_notes/v4_design_2026-10-23/review/REVIEW.md` (P1, P6, P7). The Stage 1 harness overrides the
constants per arm; `V3_RANGES` (and `V3_SETTINGS`, which adds `RHO_CIN = 0`, `PASSENGERS_ENABLED = False`)
restore v3.

- *Prior.* Same draws as v3 through `p_term`, then `cin_level` L ~ U(0,1) and `u_pass` ~ U(0,1). Lower rate floors
  (arm 3e-6, chromosome 3e-7, focal 3e-5; v3: 1e-5, 1e-6, 1e-4) and `p_term` floor 0.25 (v3 0.35). A Gaussian copula
  (`RHO_CIN = 0.5`, `scipy.special.ndtr/ndtri`) couples the three event rates through L; marginals stay
  log-uniform. `pass_rate` ~ the same copula on `u_pass` over logU(5e-4, 1.5e-2) passenger events per lineage per
  generation. parameters.pkl holds the realised rates plus `*_indep` (pre-copula), `cin_level`, `pass_rate`.
  With `V3_RANGES` and `RHO_CIN = 0` every v3 field is reproduced exactly for the same prior seed. Under
  `RHO_CIN > 0` only the quantiles of the pre-copula `*_indep` rates are paired with v3, not the realised rates.
  With `--ntrials-range`, the ntrials draw now comes after `cin_level` and `u_pass`, so `ntrials_used` is not
  paired with v3.
- *Passenger overlay (deviation D-J, `passenger_overlay.py`).* After growth, Poisson(`pass_rate` x edge length)
  +-1 copy events (neutral by construction) land on each edge of the clone genealogy and apply to every surviving descendant
  (driver-blind; 75% losses; 30% telomere-anchored, never a whole arm; interstitial length Exp(2), terminal Exp(4)
  blocks). Rejected, not applied, if a block would reach 0 copies, a driver block would exceed `MAX_REGION_CN`,
  the altered-driver cap would be exceeded, or a clone's ploidy would leave [1.5, 8].
  Trunk: before growth, `grow()` puts Poisson(`pass_rate` x `PASS_TRUNK_GENS` = 2000) passengers (same window rules)
  on the founder genome with SISTEM's own Clone methods, each kept only if SISTEM's (patched) viability check passes.
  These are the clonal passengers of the lineage before the founding cell; every clone inherits them and growth
  respects them (rows with node 0, gen 0). They are not neutral: a trunk window covering a driver block changes
  the founder's driver counts (fitness, altered-driver cap, saturation headroom). Because the founder genome
  changes, growth with `PASS_TRUNK_GENS > 0` is a different trajectory from the passenger-free run; it is identical
  only with `PASS_TRUNK_GENS = 0`. Kept by lead decision; trunk size is a growth-arm dimension of the sweep.
  RNGs: trunk `np.random.default_rng([trial_seed, 7_000_003, 0])`, overlay `default_rng([trial_seed, 7_000_003, 1])`
  (seed sequences; both lists saved in results.pkl as `trunk_seed`, `overlay_seed`); neither touches the global
  growth stream. `PASSENGERS_ENABLED = False` turns both off.
- *Outputs.* `block_cn`, `bulk`, `ploidy` are post-overlay; `fitness`, `n_altered_drivers` are end-of-growth.
  Events schema 3: 9 categories (`passenger_gain`, `passenger_loss` appended), 11 columns (`node` appended).
  clones.npz adds the genealogy (`node_parent`, `node_birth_gen`, `node_clone`, `clone_node`); results.pkl adds
  `n_passengers_proposed/applied/rejected` (+ `_zero/_cap/_altered/_ploidy`) for the post-growth edge events,
  `n_trunk_proposed/applied/rejected/rejected_by` for the founder trunk (rule order zero, cap, altered, ploidy, plus
  `absent` for the trunk only), and `overlay_seconds`. clones.npz also has `n_altered_drivers_post` (same rule on the
  post-overlay block_cn) next to the end-of-growth `n_altered_drivers`.
- *API for the harness.* `grow(params, delta_dict, chrom_lens, arm_ratios, pass_rate=None, trunk_seed=None)` returns a
  `GrowthResult` (founder trunk when `pass_rate` is given; no overlay);
  `run_overlay(growth, passenger_params(rate), rng)` or `passenger_overlay.apply_passenger_overlay(...)` with
  `overlay_inputs(growth)`; `pack_summary(growth, overlay)` builds the saved dict.
- *Checker.* Reads schemas 1-3; new C9 reports passenger counts, loss, terminal and root (trunk) shares, rejections,
  and fails on any 0-copy block.

**Prior v4.1 (2026-10-26; `PRIOR_VERSION = 5`).** Decision: `cancer/research_notes/v4_1_design_2026-10-26/DECISION.md`
(scope revised after the design review: the pi floor / driver guard and the per-clone overlay rule were dropped).
Same draws in the same order as v4; three settings changed:

- `PASS_TRUNK_GENS` 2000 -> 1000 (arm-end share) and `PASS_RATE_RANGE` logU(5e-4, 1.5e-2) -> logU(1e-4, 3e-2)
  (floor: quiet tumours; ceiling: keeps the busy end with the shorter trunk).
- `RHO_CIN` (one weight for every rate) is replaced by `RHO_CIN_LOADINGS`, one copula loading per rate
  (`arm_rate`, `chromosomal_rate`, `focal_driver_rate`, `pass_rate`). Final v4.1 values (factorial + 5e6 pilot, DECISION.md) 0.5 / 0.4 / 0.8 / 0.8:
  focal and passenger rates share one segmental mechanism; the whole-chromosome rate is only partly coupled
  (full decoupling made whole-chromosome and focal event counts anti-correlated beyond the real range). Harnesses override it with
  `setattr(sgs, "RHO_CIN_LOADINGS", {...})` (all four keys; the module value is read-only); setting the removed
  `RHO_CIN` makes `sample_theta` raise instead of being silently ignored. parameters.pkl and manifest.json record
  `rho_cin_loadings` (v4 outputs: scalar `rho_cin`).
- Pairing: delta, the driver set, pi, q, m, length_mean, p_term, cin_level, u_pass, the `*_indep` rates and the
  realised arm and focal rates are bit-identical to v4 for the same prior seed; the chromosomal rate and
  `pass_rate` differ. `V4_SETTINGS` (setattr each item) restores v4 exactly, `V3_SETTINGS` v3; `PRIOR_VERSION`
  stays 5 under either (it tags the code; the settings in effect are recorded).
- Unchanged: the edge overlay keeps the all-or-none loss rejection; its gain bias on deep edges is a known artefact
  (v4.1 `diagnostics/REPORT.md` Q4.1). The checker knows prior_version 5 and checks it like 4.

Research and measurements: `cancer/reports/Fixing the ploidy collapse in SISTEM.md`;
prior v3 and patches 5-6: `cancer/research_notes/v3_design_2026-10-21/` (`workflow_result.json`
holds the decision; `v3_common.py` the verified reference implementation).
The arm script is left as is, so the arm dataset keeps SISTEM's original behaviour
(including the ploidy collapse and 20% of arms at 0 copies).
`check_gene_selection_outputs.py` validates this layout and recovers arm averages
from `block_cn`.
`manifest.json` records `script.git_commit` and `script.git_dirty`; commit before a production run so the dataset points at a real commit.

## Running

Interpreter: the script needs numpy, torch and an importable `sistem` (the 1.0.5
source tree additionally needs `pyfaidx`). On this machine the Framework
python3.12 lacks torch and the `rag` env's torch is broken; use the `cancer` env,
`~/miniconda3/envs/cancer/bin/python` (torch present; install pyfaidx into it or
point PYTHONPATH at an environment where sistem imports):

    cd code/src/data_generation && PYTHONPATH=/Users/maryamkafi/Documents/PhD_Project/cancer/sistem-main ~/miniconda3/envs/cancer/bin/python simulate_gene_selection.py --start 1 --end 1 --ntrials 1 --output smoke_gene

Without mpi4py the script runs as a single process (deviation D-C).

On the cluster, two ways to run in parallel; both use the script unchanged:

1. **Job array (recommended):** `src/jobs/simulate_gene_array.sh`. Each array task is one
   process simulating a contiguous block of theta ids (`CHUNK` ids per task). No mpi4py
   needed. Theta depends only on (`--seed`, `sim_id`), so the split does not change the data,
   and a failed task is resubmitted alone. Example: `CHUNK=50 NTRIALS=25 sbatch --array=0-119%40
   src/jobs/simulate_gene_array.sh` covers ids 1..6000, 40 tasks at a time.
2. **One MPI job:** `src/jobs/simulate_gene_mpi.sh` (`srun`, rank 0 scatters ids round-robin;
   this is how the arm dataset was produced). Needs mpi4py built against the cluster OpenMPI.

    # optionally in either: --ntrials-range A B   (ntrials ~ U{A..B} per theta)

**Interruptions.** A lost SSH connection never affects a batch job. A task killed by SLURM is
resumed by resubmitting the same id range: completed trials (`results.pkl` with `complete` and a
whole `clones.npz`) are skipped and the rest are reproduced byte-identically thanks to per-trial
reseeding. Files are written atomically, `clones.npz` before `results.pkl`, so a kill mid-write
cannot leave a trial that claims completion without its arrays. `--overwrite` forces a redo.

Size the array only after the calibration run (20 theta x 25 trials) has measured wall time
and MaxRSS (`sacct -j <id> --format=JobID,Elapsed,MaxRSS`); the `-t` and `--mem` values in
the job scripts are placeholders.

## Staged plan

1. Smoke: 1 theta x 1 trial (`--end 1 --ntrials 1`), then run the checker.
2. Pilot: 10 theta x 5 trials (`--end 10 --ntrials 5`). `min_detectable` is fixed
   at 5e6 by the spec, so the pilot is kept small through `--end`/`--ntrials`,
   never through a smaller tumour. Use `GENE_SIM_STRICT=1` here so any
   block_cn / driver_counts disagreement stops the run.
3. Full run on the cluster only after the checker passes on the pilot.
