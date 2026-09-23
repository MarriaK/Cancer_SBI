#!/bin/bash
#SBATCH -J eval-sample
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# Stage 1: draw and save the posterior for every held-out tumour, one task per model.
#   sbatch jobs/sample.sh                    # all three
#   sbatch --array=0 jobs/sample.sh          # CloneMLP only
#   EXTRA="--limit 4" sbatch jobs/sample.sh  # smoke test
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
MODELS=(clonemlp cloneatt dominantclone)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname)  model=$MODEL"
mkdir -p "$POST"
python -m cancer_sbi.evaluation.sample_posteriors --model "$MODEL" --out-dir "$POST" ${EXTRA:-}
