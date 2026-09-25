#!/bin/bash
#SBATCH -J eval-analyze
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 01:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Stage 2: every metric and figure, from what stage 1 wrote. No GPU, minutes at most.
# Two commands: poster_metrics (tables, figures A/B/C/S1, summary_arrays.npz) and then
# tarp (the joint calibration test, figure T).
#   sbatch jobs/analyze.sh
#   POST=$HOME/cancer/results/2026-09-24/R1/posteriors \
#   OUT=$HOME/cancer/results/2026-09-24/R1 sbatch jobs/analyze.sh
#
# Env overrides: POST (input dir of posteriors_<model>.npz), OUT (results dir),
#   DRY_RUN=1 (print both commands, run nothing).
set -euo pipefail
CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
# 3-key split: --partition val needs val_ids (the 2-key file made every val sampling job fail, 2026-09-24).
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
POST="${POST:-$CANCER/results/posteriors}"
OUT="${OUT:-$CANCER/results}"
CMD=(python -m cancer_sbi.evaluation.poster_metrics --in-dir "$POST" --out-dir "$OUT")
# The joint calibration test. Same inputs, same outputs folder, own module: it reads the
# 5,000 stored draws (which poster_metrics never touches) and is the only part of stage 2
# whose cost grows with the number of draws, so it is a separate command and can be
# skipped or re-run on its own.
TARP=(python -m cancer_sbi.evaluation.tarp --in-dir "$POST" --out-dir "$OUT")
# A matrix run's posteriors are tagged (posteriors_<model>_<RUN_TAG>.npz, from
# sample.sh RUN_TAG=); without the same tag here the script finds nothing.
if [ -n "${RUN_TAG:-}" ]; then
  CMD+=(--run-tag "$RUN_TAG")
  TARP+=(--run-tag "$RUN_TAG")
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "${CMD[@]}"
  echo "${TARP[@]}"
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
echo "${TARP[@]}"
"${TARP[@]}"
