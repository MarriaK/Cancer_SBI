#!/bin/bash
#SBATCH -J eval-analyze
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 01:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Stage 2: every metric and figure, from what stage 1 wrote. No GPU, minutes at most.
#   sbatch jobs/analyze.sh
#   POST=$HOME/cancer/results/2026-09-24/R1/posteriors \
#   OUT=$HOME/cancer/results/2026-09-24/R1 sbatch jobs/analyze.sh
#
# Env overrides: POST (input dir of posteriors_<model>.npz), OUT (results dir),
#   DRY_RUN=1 (print the command, run nothing).
set -euo pipefail
CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
POST="${POST:-$CANCER/results/posteriors}"
OUT="${OUT:-$CANCER/results}"
CMD=(python -m cancer_sbi.evaluation.poster_metrics --in-dir "$POST" --out-dir "$OUT")
# A matrix run's posteriors are tagged (posteriors_<model>_<RUN_TAG>.npz, from
# sample.sh RUN_TAG=); without the same tag here the script finds nothing.
if [ -n "${RUN_TAG:-}" ]; then
  CMD+=(--run-tag "$RUN_TAG")
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "${CMD[@]}"
  exit 0
fi

# Everything below actually runs.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
# conda ships a newer libstdc++ than /lib64; without this scipy dies with
#   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$CANCER/src"
echo "host=$(hostname)  in=$POST  out=$OUT"
mkdir -p "$OUT"
echo "${CMD[@]}"
"${CMD[@]}"
