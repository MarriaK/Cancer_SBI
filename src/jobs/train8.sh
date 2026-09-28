#!/bin/bash
#SBATCH -J train-matrix8
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-7
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The EIGHTH matrix, one array task per run. jobs/train.sh (matrix 1) and
# jobs/train2.sh .. jobs/train7.sh are untouched.
#   sbatch jobs/train8.sh                 # all eight
#   sbatch --array=2 jobs/train8.sh       # AT12 only
#   DRY_RUN=1 bash jobs/train8.sh         # print all eight commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN. Same split gate, same
# refusal of a non-empty checkpoint directory, same 12 h wall clock as train5.sh.
#
# Matrix 8 asks the one question the ArmToken design wrote down and then never
# tested: what scale does the encoder actually hand its own arithmetic? The
# eight moments per (trial, arm) live on four different ranges -- a weighted
# mean and two extrema in [-1, 3], an sd in [0, ~2], three fractions in
# [0, 1] -- and matrix 5 fed them, plus their across-trial mean and sd, into
# `arm_mlp` raw; the 416-wide context then reaches the flow with
# `z_score_y="none"`. Neither decision was an oversight, but neither was
# measured, and sbi's own `z_score_y` is not the tool for the second one: it
# standardises the RAW input tensor before the embedding net, so it never sees
# the context at all.
#
# Every run in this matrix is armtoken, and every row is AT0 plus exactly one
# thing, so the whole matrix is read against AT0 (R^2 0.898 +- 0.002 over three
# seeds) -- the tightest seed band any model in this project has produced, which
# is why it is worth asking small questions of.
#
# What each run tests:
#   AT11  is copy space actually better than the stored log2 values, isolated
#         in the model with the tightest seed band?
#   AT12  does normalising the moment features help? LayerNorm over each arm
#         token's own 16 numbers, one module shared by all 44 arms.
#   AT13  the same question with BatchNorm1d over the (B * 44, 16) view: a
#         learned per-feature z-scoring with running statistics, so the four
#         scales are put on one footing across the dataset rather than within
#         each token.
#   AT14  does normalising the CONTEXT help? LayerNorm on the 416 numbers the
#         flow is given, per sample.
# Each is paired with a seed-1 replicate, because +-0.05 on R^2 is the spread
# matrices 2-4 measured and no single-seed difference below it is readable.
#
# Both switches default to the published behaviour and build NO module when
# left alone, so an unflagged armtoken run's state_dict is byte for byte AT0's.
#
# Every run is armtoken and every run uses the clone cache when CACHE_DIR is
# set -- there is no DominantClone exception in this matrix.
set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
RUNS="${RUNS_ROOT:-$CANCER/runs/2026-09-24}"
SEED="${SEED:-20260924}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Pre-built clone cache (src/utilities/build_clone_cache.py). Empty == read the
# gzipped trial files at ~28 min/epoch; set it to a built cache for ~13x faster
# epochs. The flag is only added when it is non-empty. The COMPLETE-sim cache,
# not matrix 7's partial one: every row here is read against AT0.
CACHE_DIR="${CACHE_DIR:-}"

# Gate: the split must carry a third, validation set -- identical to train5.sh.
# Without one, cli/train.py falls back to early stopping on the TEST set and
# every run's best.pt is selected on the sims it is scored on, so no two runs in
# any matrix can be compared. Stdlib-only and run before `conda activate`, so
# it takes whichever python is on PATH.
has_val_ids() {
  local py
  py="$(command -v python3 || command -v python)" || return 1
  "$py" - "$CANCER_SBI_SPLIT" <<'PY'
import pickle, sys
try:
    with open(sys.argv[1], "rb") as handle:
        split = pickle.load(handle)
except OSError:
    sys.exit(1)
val = split.get("val_ids") if hasattr(split, "get") else None
sys.exit(0 if val is not None and len(val) else 1)
PY
}

refuse_without_val_ids() {
  echo "REFUSING: $CANCER_SBI_SPLIT has no 'val_ids' (or does not exist)." >&2
  echo "Early stopping would then run on the TEST set. Carve a validation set once:" >&2
  echo "  python -m cancer_sbi.cli.make_split --add-val --frac 0.1 --seed 20260924 \\" >&2
  echo "      --in \$CANCER/data/train_test_split.pkl \\" >&2
  echo "      --out \$CANCER/data/train_val_test_split.pkl" >&2
  echo "Set ALLOW_TEST_AS_VAL=1 to train on the test set as validation anyway." >&2
}

# array index -> run.   Index:  0     1       2     3       4     5       6     7
RUNNAME=(AT11 AT11s1 AT12 AT12s1 AT13 AT13s1 AT14 AT14s1)
MODEL=(armtoken armtoken armtoken armtoken armtoken armtoken armtoken armtoken)
# No BASE_* line: the armtoken preset already carries matrices 2-4's findings
# (--z-score-x structured, --input-space copy, attn_ln, a 3-transform flow), so
# naming them would make a preset default look like a per-run choice.
#
# AT11 is the one row that UNDOES a preset default rather than adding to it:
# `--input-space log2` puts the stored values back, which is the comparison
# repair T2 never ran on this encoder.
INPUT_SPACE=(log2 log2 "" "" "" "" "" "")
ARM_FEATURE_NORM=("" "" layernorm layernorm batchnorm batchnorm "" "")
# store_true on the CLI side, so "1" here means "pass the bare flag".
ARM_CONTEXT_NORM=("" "" "" "" "" "" 1 1)
# Per-run seed. Every row is paired with its own seed-1 replicate; everything
# else keeps the matrix default so it is comparable with matrices 3-7.
RUNSEED=("" 1 "" 1 "" 1 "" 1)
USES_CACHE=(1 1 1 1 1 1 1 1)   # every run is a clone-set model

# Build the full command for one run index into the global array CMD.
build_cmd() {
  local i="$1"
  CMD=(python -m cancer_sbi.cli.train
       --model "${MODEL[$i]}"
       --data-root "$CANCER_SBI_DATA_ROOT"
       --split "$CANCER_SBI_SPLIT"
       --ckpt-dir "$RUNS/${RUNNAME[$i]}/checkpoints"
       --out "$RUNS/${RUNNAME[$i]}"
       --seed "${RUNSEED[$i]:-$SEED}"
       --min-epochs 1
       --tail-bound "${TAIL_BOUND:-5}"
       --stop-after-epochs 15
       --max-epochs 60
       --num-workers "$NUM_WORKERS")
  if [ -n "$CACHE_DIR" ] && [ "${USES_CACHE[$i]}" = "1" ]; then
    CMD+=(--cache-dir "$CACHE_DIR")
  fi
  if [ -n "${INPUT_SPACE[$i]}" ]; then
    CMD+=(--input-space "${INPUT_SPACE[$i]}")
  fi
  if [ -n "${ARM_FEATURE_NORM[$i]}" ]; then
    CMD+=(--arm-feature-norm "${ARM_FEATURE_NORM[$i]}")
  fi
  # A store_true flag takes no value, so this one is appended bare.
  if [ -n "${ARM_CONTEXT_NORM[$i]}" ]; then
    CMD+=(--arm-context-norm)
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4 5 6 7; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the eight command lines, and it is usually done on a
  # machine that does not have the cluster's split file at all.
  if [ "${ALLOW_TEST_AS_VAL:-0}" = "1" ]; then
    echo "# split gate: WAIVED by ALLOW_TEST_AS_VAL=1 ($CANCER_SBI_SPLIT)"
  elif has_val_ids; then
    echo "# split gate: OK, $CANCER_SBI_SPLIT carries val_ids"
  else
    echo "# split gate: would REFUSE -- $CANCER_SBI_SPLIT has no val_ids"
  fi
  exit 0
fi

if [ "${ALLOW_TEST_AS_VAL:-0}" != "1" ] && ! has_val_ids; then
  refuse_without_val_ids
  exit 1
fi

i="${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID, or use DRY_RUN=1}"
CKPT_DIR="$RUNS/${RUNNAME[$i]}/checkpoints"

# Safety barrier: a fresh checkpoint directory per run. A non-empty one means either
# a resume (the fragile path) or that this run would overwrite another run's best.pt.
if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
  echo "REFUSING: $CKPT_DIR is not empty." >&2
  echo "Move it aside (or pick another RUNS_ROOT) before re-running ${RUNNAME[$i]}." >&2
  exit 1
fi
mkdir -p "$CKPT_DIR"

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  input_space=${INPUT_SPACE[$i]:-<preset>}  arm_feature_norm=${ARM_FEATURE_NORM[$i]:-<preset>}  arm_context_norm=${ARM_CONTEXT_NORM[$i]:-<preset>}  seed=${RUNSEED[$i]:-$SEED}"
# Everything below actually runs.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
# conda ships a newer libstdc++ than /lib64; without this scipy dies with
#   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$CANCER/src"
build_cmd "$i"
echo "${CMD[@]}"
"${CMD[@]}"
