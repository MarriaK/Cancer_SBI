#!/bin/bash
#SBATCH -J eval-shrink
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 00:30:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Figure D (posterior mean vs truth), screen and poster builds.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
# conda ships a newer libstdc++ than /lib64; without this scipy dies with
#   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
POST="$CANCER/results/posteriors"
cd "$CANCER/src"
python -m cancer_sbi.evaluation.fig_shrinkage --in-dir "$POST" --out-dir "$CANCER/results"
python -m cancer_sbi.evaluation.fig_shrinkage --in-dir "$POST" --out-dir "$CANCER/results" --poster
