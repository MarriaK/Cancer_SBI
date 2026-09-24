#!/bin/bash
#SBATCH -J train-matrix3
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-13
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The THIRD repair matrix, one array task per run. jobs/train.sh (matrix 1) and
# jobs/train2.sh (matrix 2) are untouched.
#   sbatch jobs/train3.sh                 # all fourteen
#   sbatch --array=8 jobs/train3.sh       # R11 only
#   DRY_RUN=1 bash jobs/train3.sh         # print all fourteen commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN. Same split gate, same
# refusal of a non-empty checkpoint directory, same 12 h wall clock as train2.sh.
#
# Matrix 2 said every run overfits within 10-25 epochs (train ~= -40 nat vs val
# ~= -20), that weight decay helps (R3), that CloneAtt is repaired by
# `--freq-mode feature --attn-ln --input-space copy` (R6, R^2 0.42), and that
# seed noise on R^2 is +-0.05. So this matrix is regularisation and augmentation
# on top of R3 and R6, plus the seed replicates that say which differences are
# real.
#
# What each run tests:
#   R3s1 R3 on seed 1 -- how much of R3's R^2 is seed noise (spread is +-0.05)?
#   R3s2 R3 on seed 2 -- the second point of that spread.
#   R6s1 R6 on seed 1 -- the same question for the repaired CloneAtt.
#   R6s2 R6 on seed 2 -- the second point of that spread.
#   D0s1 the same-sims DominantClone baseline on seed 1, so the third model has
#        a seed spread too. No cache: it reads results.pkl, not clone sets.
#   D0s2 D0 on seed 2.
#   R9   does R3's weight decay help CloneAtt as well, or only CloneMLP?
#   R10  CloneMLP with the frequency as a log10 FEATURE (45th MLP input) on top
#        of R3 -- the one repair that fixed CloneAtt, asked of CloneMLP. The
#        weighted-mean pooling is kept: it is normalised, so it shrinks nothing.
#   R11  more dropout INSIDE the flow (0.3 vs the published 0.2; every run
#        overfits by epoch 10-25): is the flow's capacity the overfit?
#   R12  a 3-transform flow instead of 5: the same question by shrinking it.
#   R13  train on a random 16 of the 25 trials per sim, redrawn every epoch --
#        augmentation rather than regularisation. Val/test keep all 25.
#   R14  R11 for CloneMLP.
#   R15  R13 for CloneMLP.
#   R16  R12 for CloneMLP.
#
# Every clone-set run uses the clone cache when CACHE_DIR is set; D0s1/D0s2 are
# the DominantClone exception train.sh also has -- the cache holds
# (25, top_k, 45) clone sets, which that dataset does not read.
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

# Gate: the split must carry a third, validation set -- identical to train2.sh.
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

# array index -> run.   Index:  0    1    2    3    4    5    6  7   8   9  10  11  12  13
RUNNAME=(R3s1 R3s2 R6s1 R6s2 D0s1 D0s2 R9 R10 R11 R12 R13 R14 R15 R16)
MODEL=(clonemlp clonemlp cloneatt cloneatt dominantclone dominantclone cloneatt clonemlp cloneatt cloneatt cloneatt clonemlp clonemlp clonemlp)
# BASE_R3 (clonemlp) = --z-score-x structured --input-space copy
#                      --flow-weight-decay 1e-3 --embed-weight-decay 1e-4
# BASE_R6 (cloneatt) = --z-score-x structured --input-space copy
#                      --freq-mode feature --attn-ln
# D0s1/D0s2 name no --z-score-x: the dominantclone preset is already
# "structured", and naming it would make a preset default look like a choice.
ZSCORE=(structured structured structured structured "" "" structured structured structured structured structured structured structured structured)
INPUT_SPACE=(copy copy copy copy "" "" copy copy copy copy copy copy copy copy)
FREQ_MODE=("" "" feature feature "" "" feature feature feature feature feature "" "" "")
ATTN_LN=(0 0 1 1 0 0 1 0 1 1 1 0 0 0)
EMBED_WD=(1e-4 1e-4 "" "" "" "" 1e-4 1e-4 "" "" "" 1e-4 1e-4 1e-4)
FLOW_WD=(1e-3 1e-3 "" "" "" "" 1e-3 1e-3 "" "" "" 1e-3 1e-3 1e-3)
# The three switches this matrix adds. Each defaults to the published value and
# appears on exactly the runs that test it.
FLOW_DROPOUT=("" "" "" "" "" "" "" "" 0.3 "" "" 0.3 "" "")
FLOW_TRANSFORMS=("" "" "" "" "" "" "" "" "" 3 "" "" "" 3)
TRIAL_SUBSAMPLE=("" "" "" "" "" "" "" "" "" "" 16 "" 16 "")
# DominantClone only: train/val/test on the clone-set models' sim set, so the
# three models' numbers are comparable.
REQUIRE_ALL=(0 0 0 0 1 1 0 0 0 0 0 0 0 0)
# Per-run seed. The six replicates are seeds 1 and 2; everything else keeps the
# matrix default so it is comparable with matrix 2.
RUNSEED=(1 2 1 2 1 2 "" "" "" "" "" "" "" "")
USES_CACHE=(1 1 1 1 0 0 1 1 1 1 1 1 1 1)   # 0 for the two DominantClone runs

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
  if [ -n "${EMBED_WD[$i]}" ]; then
    CMD+=(--embed-weight-decay "${EMBED_WD[$i]}")
  fi
  if [ -n "${FLOW_WD[$i]}" ]; then
    CMD+=(--flow-weight-decay "${FLOW_WD[$i]}")
  fi
  if [ -n "${FLOW_DROPOUT[$i]}" ]; then
    CMD+=(--flow-dropout "${FLOW_DROPOUT[$i]}")
  fi
  if [ -n "${FLOW_TRANSFORMS[$i]}" ]; then
    CMD+=(--flow-num-transforms "${FLOW_TRANSFORMS[$i]}")
  fi
  if [ -n "${TRIAL_SUBSAMPLE[$i]}" ]; then
    CMD+=(--trial-subsample "${TRIAL_SUBSAMPLE[$i]}")
  fi
  if [ "${REQUIRE_ALL[$i]}" = "1" ]; then
    CMD+=(--require-all-trials)
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the fourteen command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  z_score_x=${ZSCORE[$i]:-<preset>}  input_space=${INPUT_SPACE[$i]}  freq_mode=${FREQ_MODE[$i]}  attn_ln=${ATTN_LN[$i]}  flow_dropout=${FLOW_DROPOUT[$i]}  flow_num_transforms=${FLOW_TRANSFORMS[$i]}  trial_subsample=${TRIAL_SUBSAMPLE[$i]}  seed=${RUNSEED[$i]:-$SEED}"
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
