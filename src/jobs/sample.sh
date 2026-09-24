#!/bin/bash
#SBATCH -J eval-sample
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# Stage 1: draw and save the posterior for every held-out tumour, one task per model.
#   sbatch jobs/sample.sh                    # all three, published checkpoints
#   sbatch --array=0 jobs/sample.sh          # CloneMLP only
#   EXTRA="--limit 4" sbatch jobs/sample.sh  # smoke test (writes posteriors_<m>_limit4.npz)
#
# Per-run evaluation (the 2026-09-24 matrix): point it at one run's checkpoint and
# its own output directory, so nothing silently re-samples the old published model.
#   RUN_TAG=R1 \
#   CKPT=$HOME/cancer/runs/2026-09-24/R1/checkpoints/best.pt \
#   POST=$HOME/cancer/results/2026-09-24/R1/posteriors \
#   sbatch --array=0 jobs/sample.sh
# or, without remembering the index:  MODEL=clonemlp ... sbatch --array=0 jobs/sample.sh
# (CKPT with the default 0-2 array is refused: a checkpoint belongs to one model.)
#
# Env overrides: MODEL (clonemlp|cloneatt|dominantclone), CKPT (explicit checkpoint), POST (output dir), RUN_TAG (output
#   file label; the script reads RUN_TAG, not RUN), EXTRA (extra flags),
#   DRY_RUN=1 (print the command, run nothing).
#
# -t is 12:00:00, above the ~8 h estimate for 651 cases x 5000 draws: an array task
# killed at the wall clock loses the whole sample, and the queue cost of the extra
# four hours is nil.
set -euo pipefail
CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
POST="${POST:-$CANCER/results/posteriors}"
MODELS=(clonemlp cloneatt dominantclone)
# MODEL= names the model directly; otherwise the array index picks it. A CKPT
# belongs to exactly one model, so a CKPT run submitted with the default 0-2
# array would point two of its three tasks at a checkpoint of the wrong model.
# Refuse that up front rather than let two GPU tasks fail on the mismatch check.
MODEL="${MODEL:-${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}}"
if [ -n "${CKPT:-}" ] && [ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]; then
  echo "REFUSING: CKPT is set but this is a ${SLURM_ARRAY_TASK_COUNT}-task array." >&2
  echo "  A checkpoint belongs to one model: submit with --array=<0|1|2> or MODEL=<name>." >&2
  exit 1
fi

CMD=(python -m cancer_sbi.evaluation.sample_posteriors --model "$MODEL" --out-dir "$POST")
# An explicit --ckpt is the only way to be sure which model was sampled: without it
# sample_posteriors.py defaults to $CANCER_SBI_RUNS/<model>/checkpoints/best.pt, the
# OLD published checkpoint, and the job succeeds with the numbers you already had.
if [ -n "${CKPT:-}" ]; then
  CMD+=(--ckpt "$CKPT")
fi
if [ -n "${RUN_TAG:-}" ]; then
  CMD+=(--run-tag "$RUN_TAG")
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
echo "host=$(hostname)  model=$MODEL  post=$POST  ckpt=${CKPT:-<default: published run>}"
mkdir -p "$POST"
echo "${CMD[@]}"
"${CMD[@]}"
