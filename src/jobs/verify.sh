#!/bin/bash
#SBATCH -J verify-refactor
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi

# The conda env ships a newer libstdc++ than /lib64 provides. Without putting it
# first on the library path, scipy.optimize._highspy dies with
#   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$HOME/cancer"
python verify_refactor.py \
    --data-root Guassian_Normal/simulation_outputs \
    --with-models
