#!/bin/bash
# Gene-selection (FittedRegionLibrary) simulations as a SLURM job array.
# RECOMMENDED way to parallelise: each array task is ONE ordinary process that
# simulates a contiguous range of theta ids (no mpi4py needed; the script's
# single-process fallback is used). Theta for a given sim_id depends only on
# (--seed, sim_id), so splitting the id range across tasks does not change the data.
#
# Interruptions: a dropped SSH session does not affect a batch job. --requeue covers node
# failure and preemption only; a task that hits the TIME LIMIT ends in state TIMEOUT and is
# NOT requeued (seen on calibration v2, 2026-10-21: 6 of 21 tasks). Resume by resubmitting
# the SAME array indices (e.g. --array=3,4,7,12,14): the script skips trials that already
# completed and reproduces the rest byte-identically (per-trial reseeding). Pass --overwrite
# to redo. For the production run, re-run `sacct ... | grep TIMEOUT` after each wave and
# resubmit those indices until every theta has 25 completed trials.
#
# Sizing (F10 calibration on EPYC 9534, 2026-10-19/20: one 5e6-cell trial = 780 s median, 1,239 s max;
# one theta x 25 trials = 4.5-6.3 h; MaxRSS 2.2 GB): CHUNK=1 theta per task is the only size that is
# safely inside general's 12 h limit (CHUNK=2 could reach 12.6 h). SLURM caps an array at 10,000
# tasks, so 20,000 theta = two submissions: START_ID=1 and START_ID=10001, each --array=0-9999.
# Every partition we can use shares QOS 'general' = 2,304 CPUs per user, so %2304 is the ceiling.
# Expected: ~110k core-hours -> ~2 days of wall clock at 2,304 concurrent tasks; ~75 GB on scratch.
# Usage:
#   sbatch src/jobs/simulate_gene_array.sh                                # ids 1..10000
#   START_ID=10001 sbatch src/jobs/simulate_gene_array.sh                 # ids 10001..20000
#   OUT=/gpfs/scratchfs1/pas23008/$USER/gene_calib_v2 CHUNK=1 sbatch --array=0-20 src/jobs/simulate_gene_array.sh   # 21-theta calibration (CHUNK=3 would exceed 12 h)
#   SIM_LOG=1 ... keeps SISTEM's sim.log (133 KB/trial; off by default)
#
#SBATCH -J gene_sim
#SBATCH -p general
#SBATCH -n 1 -N 1
#SBATCH --mem-per-cpu=3G
#SBATCH -t 12:00:00
#SBATCH --array=0-9999%2304
#SBATCH --requeue
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err

set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
CHUNK="${CHUNK:-1}"                   # theta ids per array task (one theta x 25 trials = 4.5-6.3 h)
START_ID="${START_ID:-1}"             # first theta id of this submission (10001 for the second half)
NTRIALS="${NTRIALS:-25}"              # trials per theta (or set NTRIALS_RANGE="10 50")
SEED="${SEED:-12345}"
OUT="${OUT:-/gpfs/scratchfs1/pas23008/$USER/gene_selection_20k}"   # scratch (60-day purge!); home has ~65 GB free
LOGFLAG="--no-sim-log"; [[ "${SIM_LOG:-0}" == "1" ]] && LOGFLAG=""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi

# Fail fast if SISTEM is missing or too old (selection code is identical from 1.0.3 on)
python - <<'EOF'
import sistem, sys
v = tuple(int(x) for x in sistem.__version__.split(".")[:3])
print("sistem", sistem.__version__, "at", sistem.__file__)
sys.exit(0 if v >= (1, 0, 3) else 1)
EOF

START=$(( START_ID + SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START_ID + (SLURM_ARRAY_TASK_ID + 1) * CHUNK - 1 ))
echo "task $SLURM_ARRAY_TASK_ID: sim ids $START..$END -> $OUT"

mkdir -p "$OUT"
cd "$CANCER/src/data_generation"
if [[ -n "${NTRIALS_RANGE:-}" ]]; then
    # shellcheck disable=SC2086
    python simulate_gene_selection.py --start "$START" --end "$END" \
        --ntrials-range $NTRIALS_RANGE --output "$OUT" --seed "$SEED" $LOGFLAG
else
    python simulate_gene_selection.py --start "$START" --end "$END" \
        --ntrials "$NTRIALS" --output "$OUT" --seed "$SEED" $LOGFLAG
fi
