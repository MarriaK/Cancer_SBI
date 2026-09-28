#!/bin/bash
#SBATCH -J eval-recalibrate
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 01:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Stage 1c: fit a per-arm affine correction on the VALIDATION posteriors and apply it to the
# TEST posteriors. No GPU; it only reads and rewrites .npz files, so it is minutes.
#
# The three-line recipe (a validation pass, the correction, then the ordinary stage 2):
#   (a) the validation sample, from the SAME checkpoint and into the same directory:
#   PARTITION=val RUN_TAG=AT0 \
#   CKPT=$HOME/cancer/runs/2026-09-24/AT0/checkpoints/best.pt \
#   POST=$HOME/cancer/results/2026-09-24/AT0/posteriors \
#   MODEL=armtoken sbatch --array=0 jobs/sample.sh
#
#   (b) the correction:
#   VAL=$HOME/cancer/results/2026-09-24/AT0/posteriors/posteriors_armtoken_AT0_val.npz \
#   TEST=$HOME/cancer/results/2026-09-24/AT0/posteriors/posteriors_armtoken_AT0.npz \
#   OUT=$HOME/cancer/results/2026-09-24/AT0rc/posteriors \
#   MODEL=armtoken RUN_TAG=AT0rc sbatch jobs/recalibrate.sh
#
#   (c) stage 2 on the result, unchanged:
#   POST=$HOME/cancer/results/2026-09-24/AT0rc/posteriors \
#   OUT=$HOME/cancer/results/2026-09-24/AT0rc RUN_TAG=AT0rc sbatch jobs/analyze.sh
#
# Step (a) needs a split with val_ids: CANCER_SBI_SPLIT below defaults to the three-key
# train_val_test_split.pkl for that reason, and sample_posteriors.py refuses a two-key file.
#
# Env overrides: VAL (validation .npz, required), TEST (test .npz, required), OUT (output dir),
#   MODEL (clonemlp|cloneatt|dominantclone|armtoken), RUN_TAG (label for the corrected file),
#   METHOD (affine|shift), EXTRA (extra flags, last-wins), DRY_RUN=1 (print the command, run nothing).
set -euo pipefail
CANCER="$HOME/cancer"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
export CANCER_SBI_RUNS="${CANCER_SBI_RUNS:-$CANCER/runs}"
OUT="${OUT:-$CANCER/results/posteriors}"
MODEL="${MODEL:-clonemlp}"
VAL="${VAL:-}"
TEST="${TEST:-}"
if [ -z "$VAL" ] || [ -z "$TEST" ]; then
  echo "REFUSING: VAL and TEST must both be set." >&2
  echo "  VAL=<..._val.npz> TEST=<....npz> OUT=<dir> MODEL=<model> RUN_TAG=<tag> sbatch jobs/recalibrate.sh" >&2
  exit 1
fi

CMD=(python -m cancer_sbi.evaluation.recalibrate_posteriors --val "$VAL" --test "$TEST" \
     --out-dir "$OUT" --model "$MODEL")
# Without a RUN_TAG the corrected file is written as posteriors_<model>.npz, which is the
# name the uncorrected run already owns in a shared directory.
if [ -n "${RUN_TAG:-}" ]; then
  CMD+=(--run-tag "$RUN_TAG")
fi
if [ -n "${METHOD:-}" ]; then
  CMD+=(--method "$METHOD")
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
echo "host=$(hostname)  model=$MODEL  out=$OUT  tag=${RUN_TAG:-<none>}  method=${METHOD:-affine}"
mkdir -p "$OUT"
echo "${CMD[@]}"
"${CMD[@]}"
