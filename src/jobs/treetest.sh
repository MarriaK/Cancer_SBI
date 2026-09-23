#!/bin/bash
#SBATCH -J tree-test
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

# Proves the reorganised tree works end to end: the package under src/cancer_sbi,
# the data under data/Guassian_Normal/, the split in data/, and the checkpoints in
# runs/<model>/checkpoints/.
CANCER="$HOME/cancer"
cd "$CANCER/src"
for M in clonemlp cloneatt dominantclone; do
  echo "=== $M ==="
  python -m cancer_sbi.evaluation.sample_posteriors \
      --model "$M" \
      --data-root "$CANCER/data/Guassian_Normal/simulation_outputs" \
      --split     "$CANCER/data/train_test_split.pkl" \
      --ckpt      "$CANCER/runs/$M/checkpoints/best.pt" \
      --out-dir   "$CANCER/_treetest" \
      --limit 4 --seed 0
done
echo "ALL THREE OK"
