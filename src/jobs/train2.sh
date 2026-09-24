#!/bin/bash
#SBATCH -J train-matrix2
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-6
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The SECOND repair matrix (R3 / R2s1 / R2s2 / R5 / R6 / R7 / R8), one array
# task per run. jobs/train.sh is the first matrix and is untouched.
#   sbatch jobs/train2.sh                 # all seven
#   sbatch --array=3 jobs/train2.sh       # R5 only
#   DRY_RUN=1 bash jobs/train2.sh         # print all seven commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN. Same split gate, same
# refusal of a non-empty checkpoint directory, same 12 h wall clock as train.sh.
#
# What each run tests:
#   R3   does weight decay (flow 1e-3, embed 1e-4) stop R2 overfitting from epoch ~15?
#   R2s1 R2 again on seed 1 -- how much of R2's R^2=0.415 is seed noise?
#   R2s2 R2 again on seed 2 -- the second point of that same seed spread.
#   R5   CloneAtt with the frequency as a log10 FEATURE plus LayerNorm, i.e. the
#        two things R4 lacked: R4's renormalised multiply still left every token
#        at ~1/100 scale with no LayerNorm to rescale it (R^2 = 0.025).
#   R6   R5 plus the copy-space input, so CloneAtt gets T2 as well (R2's repair).
#   R7   R6 plus embed_lr 5e-4: the published 1e-4 leaves the encoder near-frozen.
#   R8   R7 plus encoder dropout 0.1, which trap 4 means CloneAtt has never had.
#
# Every run uses the clone cache when CACHE_DIR is set: all seven are clone-set
# models, so none of them is the DominantClone exception train.sh has.
set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
RUNS="${RUNS_ROOT:-$CANCER/runs/2026-09-24}"
SEED="${SEED:-20260924}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Pre-built clone cache (src/utilities/build_clone_cache.py). Empty == read the
# gzipped trial files at ~28 min/epoch; set it to a built cache for ~13x faster
# epochs. The flag is only added when it is non-empty.
CACHE_DIR="${CACHE_DIR:-}"

# Gate: the split must carry a third, validation set -- identical to train.sh.
# Without one, cli/train.py falls back to early stopping on the TEST set and
# every run's best.pt is selected on the sims it is scored on, so no two runs in
# either matrix can be compared. Stdlib-only and run before `conda activate`, so
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

# array index -> run.   Index:  0     1     2     3    4    5    6
RUNNAME=(R3 R2s1 R2s2 R5 R6 R7 R8)
MODEL=(clonemlp clonemlp clonemlp cloneatt cloneatt cloneatt cloneatt)
# Every run in this matrix whitens theta: R1 settled that question in matrix 1.
ZSCORE=(structured structured structured structured structured structured structured)
INPUT_SPACE=(copy copy copy "" copy copy copy)   # R5 is the one run without T2
FREQ_MODE=("" "" "" feature feature feature feature)   # cloneatt only
ATTN_LN=(0 0 0 1 1 1 1)                                # cloneatt only
ENCODER_DROPOUT=("" "" "" "" "" "" 0.1)                # R8 only (trap 4)
EMBED_LR=("" "" "" "" "" 5e-4 5e-4)                    # R7 onwards
EMBED_WD=(1e-4 "" "" "" "" "" "")                      # R3 only
FLOW_WD=(1e-3 "" "" "" "" "" "")                       # R3 only
# Per-run seed. R2s1/R2s2 are R2 repeated to measure the seed spread; everything
# else keeps the matrix default so it is comparable with matrix 1.
RUNSEED=("" 1 2 "" "" "" "")
USES_CACHE=(1 1 1 1 1 1 1)   # all seven are clone-set models

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
       --stop-after-epochs 15
       --max-epochs 60
       --num-workers "$NUM_WORKERS")
  if [ -n "${ZSCORE[$i]}" ]; then
    CMD+=(--z-score-x "${ZSCORE[$i]}")
  fi
  if [ -n "$CACHE_DIR" ] && [ "${USES_CACHE[$i]}" = "1" ]; then
    CMD+=(--cache-dir "$CACHE_DIR")
  fi
  if [ -n "${INPUT_SPACE[$i]}" ]; then
    CMD+=(--input-space "${INPUT_SPACE[$i]}")
  fi
  if [ -n "${FREQ_MODE[$i]}" ]; then
    CMD+=(--freq-mode "${FREQ_MODE[$i]}")
  fi
  if [ "${ATTN_LN[$i]}" = "1" ]; then
    CMD+=(--attn-ln)
  fi
  if [ -n "${ENCODER_DROPOUT[$i]}" ]; then
    CMD+=(--encoder-dropout "${ENCODER_DROPOUT[$i]}")
  fi
  if [ -n "${EMBED_LR[$i]}" ]; then
    CMD+=(--embed-lr "${EMBED_LR[$i]}")
  fi
  if [ -n "${EMBED_WD[$i]}" ]; then
    CMD+=(--embed-weight-decay "${EMBED_WD[$i]}")
  fi
  if [ -n "${FLOW_WD[$i]}" ]; then
    CMD+=(--flow-weight-decay "${FLOW_WD[$i]}")
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4 5 6; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the seven command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  z_score_x=${ZSCORE[$i]:-<preset>}  input_space=${INPUT_SPACE[$i]}  freq_mode=${FREQ_MODE[$i]}  attn_ln=${ATTN_LN[$i]}  encoder_dropout=${ENCODER_DROPOUT[$i]}  embed_lr=${EMBED_LR[$i]}  seed=${RUNSEED[$i]:-$SEED}"
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
