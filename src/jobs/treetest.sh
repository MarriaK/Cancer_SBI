#!/bin/bash
#SBATCH -J tree-test
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

# Proves the reorganised tree works end to end: scripts in src/, split in data/,
# checkpoints in runs/<model>/checkpoints/.
cd "$HOME/cancer"
for M in clonemlp cloneatt dominantclone; do
  echo "=== $M ==="
  python src/evaluation/sample_posteriors.py \
      --model "$M" \
      --data-root "$HOME/cancer/Guassian_Normal/simulation_outputs" \
      --split     "$HOME/cancer/data/train_test_split.pkl" \
      --ckpt      "$HOME/cancer/runs/$M/checkpoints/best.pt" \
      --out-dir   "$HOME/cancer/_treetest" \
      --limit 4 --seed 0
done
echo "ALL THREE OK"
