#!/bin/bash
#SBATCH -J train-matrix6
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-8
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The SIXTH matrix, one array task per run. jobs/train.sh (matrix 1),
# jobs/train2.sh, train3.sh, train4.sh, train4b.sh and train5.sh are untouched.
#   sbatch jobs/train6.sh                 # all nine
#   sbatch --array=5 jobs/train6.sh       # AT6 only
#   DRY_RUN=1 bash jobs/train6.sh         # print all nine commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN. Same split gate, same
# refusal of a non-empty checkpoint directory, same 12 h wall clock as train5.sh.
#
# Matrix 5 answered the architecture question: ArmToken (AT0) reaches R^2 0.898
# against CloneAtt's best-ever 0.568 (R26). This matrix asks the two follow-ups
# that answer leaves open.
#
# Rows 0-2, the HYBRID. A fifth model: ArmToken's encoder and R26's CloneAtt
# encoder run side by side on the same input, their contexts concatenated
# (416 + 256 = 672) and handed to one flow. The per-arm moments are lossy in a
# specific way -- every statistic is taken per arm, so the joint pattern of two
# arms WITHIN one clone is thrown away -- and clone-level attention is the one
# encoder here that keeps it. If the hybrid matches AT0 and no more, the clone
# branch adds nothing and matrix 5's answer is final; three seeds because the
# difference being tested is of the size of the seed spread (+-0.05 on R^2).
#
# Rows 3-8, ArmToken's FLOW. Its encoder is 62k parameters and its flow, at
# 3 transforms x 50 hidden features, is now the small half of the model; every
# matrix so far shrank the flow to fight overfitting, and none has asked
# whether ArmToken's context can feed a bigger one. AT5/AT6 move one dimension
# each, AT8 both; AT7 and AT9 add a learning-rate schedule, which is the other
# thing a run that early-stops at 60 epochs on a patience of 15 might be short
# of. AT0s2 is the third seed of the matrix-5 yardstick, so the six flow rows
# are read against a three-seed baseline rather than a single number.
#
# What each run tests:
#   H0    the hybrid itself, on the matrix seed. Reads against AT0.
#   H0s1  H0 on seed 1  -- a new architecture needs its own seed spread before
#   H0s2  H0 on seed 2     any conclusion about +-0.05 can be drawn.
#   AT0s2 the matrix-5 yardstick's third seed, for the same reason.
#   AT5   5 spline transforms instead of the preset's 3: is R12's shrink still
#         right now that the encoder is doing the work?
#   AT6   100 hidden features instead of 50. 50 is the published width in
#         every run of every matrix so far; nothing has ever tested it.
#   AT7   ReduceLROnPlateau(factor 0.5, patience 5) on the validation loss.
#   AT8   AT5 and AT6 together -- the bigger flow, both dimensions.
#   AT9   AT8 plus the schedule: the biggest flow with the best chance of
#         settling into it.
#
# Every run is a clone-set model and uses the clone cache when CACHE_DIR is set.
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

# array index -> run.   Index:  0   1     2     3      4    5    6    7    8
RUNNAME=(H0 H0s1 H0s2 AT0s2 AT5 AT6 AT7 AT8 AT9)
MODEL=(hybrid hybrid hybrid armtoken armtoken armtoken armtoken armtoken armtoken)
# No BASE_* line: both presets already carry matrices 2-5's findings
# (--z-score-x structured, --input-space copy, attn_ln, a 3-transform flow, and
# for hybrid R26's --freq-mode feature and --d-model 256), so naming them would
# make a preset default look like a per-run choice. Each row adds only its own.
FLOW_TRANSFORMS=("" "" "" "" 5 "" "" 5 5)
FLOW_HIDDEN=("" "" "" "" "" 100 "" 100 100)
# "1" adds --lr-plateau (a store_true flag), "" leaves it off.
LR_PLATEAU=("" "" "" "" "" "" 1 "" 1)
# Per-run seed. H0s1/H0s2 and AT0s2 are the replicates; everything else keeps
# the matrix default so it is comparable with matrices 3-5.
RUNSEED=("" 1 2 2 "" "" "" "" "")
USES_CACHE=(1 1 1 1 1 1 1 1 1)   # every run is a clone-set model

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
  if [ -n "${FLOW_TRANSFORMS[$i]}" ]; then
    CMD+=(--flow-num-transforms "${FLOW_TRANSFORMS[$i]}")
  fi
  if [ -n "${FLOW_HIDDEN[$i]}" ]; then
    CMD+=(--flow-hidden-features "${FLOW_HIDDEN[$i]}")
  fi
  if [ -n "${LR_PLATEAU[$i]}" ]; then
    CMD+=(--lr-plateau)
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4 5 6 7 8; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the nine command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  flow_num_transforms=${FLOW_TRANSFORMS[$i]:-<preset>}  flow_hidden_features=${FLOW_HIDDEN[$i]:-<preset>}  lr_plateau=${LR_PLATEAU[$i]:-off}  seed=${RUNSEED[$i]:-$SEED}"
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
