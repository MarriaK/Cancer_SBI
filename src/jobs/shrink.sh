#!/bin/bash
#SBATCH -J eval-shrink
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 00:30:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Figure D (posterior mean vs truth), screen and poster builds.
#   sbatch jobs/shrink.sh
#   POST=$HOME/cancer/results/2026-09-24/R1/posteriors \
#   OUT=$HOME/cancer/results/2026-09-24/R1 sbatch jobs/shrink.sh
#
# Env overrides: POST (input dir), OUT (results dir), DRY_RUN=1.
set -euo pipefail
CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
# 3-key split: --partition val needs val_ids (the 2-key file made every val sampling job fail, 2026-09-24).
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
POST="${POST:-$CANCER/results/posteriors}"
OUT="${OUT:-$CANCER/results}"
CMD=(python -m cancer_sbi.evaluation.fig_shrinkage --in-dir "$POST" --out-dir "$OUT")
# A matrix run's posteriors are tagged (posteriors_<model>_<RUN_TAG>.npz, from
# sample.sh RUN_TAG=); without the same tag here the script finds nothing.
if [ -n "${RUN_TAG:-}" ]; then
  CMD+=(--run-tag "$RUN_TAG")
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "${CMD[@]}"
  echo "${CMD[@]} --poster"
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
"${CMD[@]}"
"${CMD[@]}" --poster
