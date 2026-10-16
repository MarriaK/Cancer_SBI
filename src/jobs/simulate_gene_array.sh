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
# Usage (sizes below are placeholders until the calibration run measures them):
#   sbatch src/jobs/simulate_gene_array.sh              # ids 1..6000 in chunks of CHUNK
#   CHUNK=50 NTRIALS=25 sbatch --array=0-119%40 src/jobs/simulate_gene_array.sh
#   OUT=$CANCER/data/gene_pilot CHUNK=1 NTRIALS=5 sbatch --array=0-9 src/jobs/simulate_gene_array.sh
#
#SBATCH -J gene_sim
#SBATCH -p general
#SBATCH -n 1 -N 1
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --array=0-59%30
#SBATCH --requeue
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err

set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
CHUNK="${CHUNK:-100}"                 # theta ids per array task
NTRIALS="${NTRIALS:-25}"              # trials per theta (or set NTRIALS_RANGE="10 50")
SEED="${SEED:-12345}"
OUT="${OUT:-$CANCER/data/gene_selection}"

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

cd "$CANCER/src/data_generation"
if [[ -n "${NTRIALS_RANGE:-}" ]]; then
    # shellcheck disable=SC2086
    python simulate_gene_selection.py --start "$START" --end "$END" \
        --ntrials-range $NTRIALS_RANGE --output "$OUT" --seed "$SEED"
else
    python simulate_gene_selection.py --start "$START" --end "$END" \
        --ntrials "$NTRIALS" --output "$OUT" --seed "$SEED"
fi
