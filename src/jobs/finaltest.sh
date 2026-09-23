#!/bin/bash
#SBATCH -J final-test
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4 -N 1
#SBATCH -t 00:40:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CANCER_SBI_DATA_ROOT="$HOME/cancer/data/Guassian_Normal/simulation_outputs"
export CANCER_SBI_SPLIT="$HOME/cancer/data/train_test_split.pkl"
export CANCER_SBI_RUNS="$HOME/cancer/runs"

cd "$HOME/cancer"

echo "--- sampling: all three models, env-var paths only, nothing outside src/ data/ runs/ ---"
for M in clonemlp cloneatt dominantclone; do
  python src/evaluation/sample_posteriors.py --model "$M" \
      --out-dir "$HOME/cancer/_final" --limit 4 --seed 0 | grep "saved ->"
done

echo
echo "--- analysis + figures from those samples ---"
python src/evaluation/analyze.py --in-dir "$HOME/cancer/_final" \
       --out-dir "$HOME/cancer/_final/results" 2>&1 | grep -E "saved ->|true R2" | head -8

echo
echo "REORGANISED TREE OK"
