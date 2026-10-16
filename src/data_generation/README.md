# data_generation: SISTEM simulators

Two simulator families, one per SISTEM selection model. Both share the same MPI
layout, retry loop (up to 5 attempts per trial), per-trial seeding and CLI
(`--start --end --ntrials --output --seed`), so the two scripts can be diffed.

| | Arm model | Gene (region) model |
|---|---|---|
| Simulator | `simulate_arm_selection.py` (was `dataSimulator_MPI.py`) | `simulate_gene_selection.py` |
| SISTEM library | `FittedArmLibrary`, 44 arm coefficients | `FittedRegionLibrary`, one coefficient per 5 Mb block (D = 589) |
| Prior | Normal(0, 0.2) on arm deltas | sparse spike-and-slab on block deltas (see script docstring) |
| Per-trial output | `results.pkl` (largest-clone + bulk arm CN ratios), `CNratios_all.pkl.gz` | `results.pkl` (bookkeeping only), `clones.npz` (per-clone block CNs + popsize, ploidy, fitness, nWGD, n_altered_drivers, and the bulk profile; largest clone = argmax popsize) |
| Checker | `update_checker.py` | `check_gene_selection_outputs.py` |
| Repair | `fixed_dataSimulation.py` | none yet |

`simulate_arm_selection.py` is the provenance record of the existing arm dataset
and is kept behaviour-identical (its known defects included); do not edit it.

## Gene-model layout

    <output>/manifest.json                       grid, prior, fixed params, seeding (rank 0, once)
    <output>/simulation_outputs/sim{ID}/parameters.pkl
    <output>/simulation_outputs/sim{ID}/{t}/results.pkl   always written (bookkeeping: failures, complete, n_clones, wall time)
    <output>/simulation_outputs/sim{ID}/{t}/clones.npz    only for completed trials (all data: block_cn, popsize, ..., bulk)
    <output>/simulation_outputs/sim{ID}/{t}/sim.log       written by SISTEM

Block vectors are flattened chr1..chr22 in numeric order (`manifest["chrom_order"]`).

**SISTEM runtime patch.** SISTEM's `get_driver_start_regions` has an off-by-one
(`sequence[:-size]`) so the last block of every chromosome can never receive a focal
event. The gene script does not edit SISTEM; `apply_sistem_patches()` replaces that one
method in memory at import time (same effect with the pip-installed package on the
cluster), self-tests the replacement, and records it in `manifest["sistem_patches"]`.
The arm script is left as is, so the arm dataset keeps SISTEM's original behaviour.
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
