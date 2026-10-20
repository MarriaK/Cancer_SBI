#!/bin/bash
# Gene-selection (FittedRegionLibrary) simulations as a SLURM job array.
# RECOMMENDED way to parallelise: each array task is ONE ordinary process that
# simulates a contiguous range of theta ids (no mpi4py needed; the script's
# single-process fallback is used). Theta for a given sim_id depends only on
# (--seed, sim_id), so splitting the id range across tasks does not change the data.
#
# Interruptions: a dropped SSH session does not affect a batch job. A task killed by the
# time limit, a node failure or preemption is resumed by resubmitting the SAME array
# index (or automatically via --requeue): the script skips trials that already completed
# and reproduces the rest byte-identically (per-trial reseeding). Pass --overwrite to redo.
#
# Sizing (measured 2026-10-19: one 5e6-cell trial = 228-357 s on an M1 core, so one theta x 25
# trials ~ 2.5-3 h): CHUNK=3 theta per task keeps a task at ~8-9 h, inside general's 12 h limit.
# Every partition we can use shares QOS 'general' = 2,304 CPUs per user, so %2304 is the ceiling.
# Usage:
#   sbatch src/jobs/simulate_gene_array.sh                                # ids 1..20001 (6,667 tasks x 3)
#   OUT=/gpfs/scratchfs1/pas23008/$USER/gene_calib CHUNK=3 sbatch --array=0-6 src/jobs/simulate_gene_array.sh   # 21-theta calibration
#   SIM_LOG=1 ... keeps SISTEM's sim.log (133 KB/trial; off by default)
#
#SBATCH -J gene_sim
#SBATCH -p general
#SBATCH -n 1 -N 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 12:00:00
#SBATCH --array=0-6666%2304
#SBATCH --requeue
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err

set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
CHUNK="${CHUNK:-3}"                   # theta ids per array task (~3 h each with 25 trials)
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

START=$(( 1 + SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * CHUNK ))
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
