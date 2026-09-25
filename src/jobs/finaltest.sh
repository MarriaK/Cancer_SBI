#!/bin/bash
#SBATCH -J final-test
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4 -N 1
#SBATCH -t 00:40:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="$CANCER/data/Guassian_Normal/simulation_outputs"
export CANCER_SBI_SPLIT="$CANCER/data/train_test_split.pkl"
export CANCER_SBI_RUNS="$CANCER/runs"

FINAL="$CANCER/_final"
cd "$CANCER/src"

echo "--- sampling: all three models, env-var paths only, nothing outside src/ data/ runs/ ---"
for M in clonemlp cloneatt dominantclone; do
  python -m cancer_sbi.evaluation.sample_posteriors --model "$M" \
      --out-dir "$FINAL" --limit 4 --seed 0 | grep "saved ->"
  # --limit writes posteriors_<M>_limit4.npz so a smoke test can never overwrite a
  # real run. poster_metrics.py only discovers the plain name, so inside this
  # throwaway directory rename it back.
  mv "$FINAL/posteriors_${M}_limit4.npz" "$FINAL/posteriors_${M}.npz"
done

echo
echo "--- analysis + figures from those samples ---"
python -m cancer_sbi.evaluation.poster_metrics --in-dir "$FINAL" \
       --out-dir "$FINAL/results" 2>&1 | grep -E "saved ->|true R2" | head -8

echo
echo "REORGANISED TREE OK"
