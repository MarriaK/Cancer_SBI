#!/bin/bash
#SBATCH -J eval-ensemble
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 01:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Stage 1b: pool several seeds of one run into one ensemble posterior file. No GPU;
# it only reads and rewrites .npz files, so it is minutes and a lot of disk.
#
# The two-line recipe (ensemble, then the ordinary stage 2 on the result):
#   INPUTS="$HOME/cancer/results/2026-09-24/R12s0/posteriors/posteriors_cloneatt_R12s0.npz \
#           $HOME/cancer/results/2026-09-24/R12s1/posteriors/posteriors_cloneatt_R12s1.npz" \
#   OUT=$HOME/cancer/results/2026-09-24/R12ens/posteriors \
#   MODEL=cloneatt RUN_TAG=R12ens sbatch jobs/ensemble.sh
#   POST=<OUT> OUT=<results dir> RUN_TAG=<tag> sbatch jobs/analyze.sh
#
# Env overrides: INPUTS (space-separated .npz paths, required), OUT (output dir),
#   MODEL (clonemlp|cloneatt|dominantclone), RUN_TAG (label for the ensemble file),
#   NUM_SAMPLES (subsample the pooled draws, equal share per member),
#   EXTRA (extra flags, last-wins), DRY_RUN=1 (print the command, run nothing).
set -euo pipefail
CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
OUT="${OUT:-$CANCER/results/posteriors}"
MODEL="${MODEL:-clonemlp}"
# Unquoted on purpose: INPUTS is a space-separated list of paths, one per seed.
INPUTS="${INPUTS:-}"
if [ -z "$INPUTS" ]; then
  echo "REFUSING: INPUTS is empty." >&2
  echo "  INPUTS=\"a.npz b.npz\" OUT=<dir> MODEL=<model> RUN_TAG=<tag> sbatch jobs/ensemble.sh" >&2
  exit 1
fi

CMD=(python -m cancer_sbi.evaluation.ensemble_posteriors --inputs ${INPUTS} \
     --out-dir "$OUT" --model "$MODEL")
# Without a RUN_TAG the ensemble is written as posteriors_<model>.npz, which is the
# name a plain single-seed run already owns in a shared directory.
if [ -n "${RUN_TAG:-}" ]; then
  CMD+=(--run-tag "$RUN_TAG")
fi
if [ -n "${NUM_SAMPLES:-}" ]; then
  CMD+=(--num-samples "$NUM_SAMPLES")
fi
# EXTRA stays last: argparse is last-wins, so it can still override anything above.
CMD+=(${EXTRA:-})

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
echo "host=$(hostname)  model=$MODEL  out=$OUT  tag=${RUN_TAG:-<none>}"
mkdir -p "$OUT"
echo "${CMD[@]}"
"${CMD[@]}"
