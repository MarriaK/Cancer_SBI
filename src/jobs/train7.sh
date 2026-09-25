#!/bin/bash
#SBATCH -J train-matrix7
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The SEVENTH matrix, one array task per run. jobs/train.sh (matrix 1) through
# jobs/train6.sh are untouched.
#   sbatch jobs/train7.sh                 # all four
#   sbatch --array=2 jobs/train7.sh       # R27 only
#   DRY_RUN=1 bash jobs/train7.sh         # print all four commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, MIN_TRIALS, ALLOW_TEST_AS_VAL, DRY_RUN. Same split
# gate, same refusal of a non-empty checkpoint directory, same 12 h wall clock
# as train5.sh and train6.sh.
#
# This matrix changes the DATA, not the model. Trap 10 drops a simulation whole
# if even one of its 25 replicate tumours is missing a CNratios_all.pkl.gz, so
# 441 of the cluster's 3,600 sims -- every one with 1-24 complete replicates --
# have never been seen by a clone-set model. --min-trials 5 keeps them, with
# NaN in the slots they have no file for; ArmToken masks those slots natively
# and CloneAtt/CloneMLP through sbi's NaN-aware trial pooling, so no encoder
# needs a line changed. Roughly +12% training sims.
#
# The TEST set is deliberately NOT relaxed: data/loaders.py applies --min-trials
# to the training and validation datasets only, and the test dataset keeps the
# published complete-sim rule. Every number this matrix reports is therefore on
# exactly the same cases matrices 1-6 reported on, and a gain here is a gain
# from more training data rather than from an easier test set.
#
# What each run tests:
#   AT10   the matrix-5 yardstick (R^2 0.898) on the bigger training set. The
#          whole matrix is read against this row.
#   AT10s1 AT10 on seed 1 -- the +-0.05 seed spread has to be known before the
#          difference from AT0 can be called real.
#   R27    R26, the best CloneAtt ever run (R^2 0.568), with the partial sims.
#          CloneAtt has always looked data-starved rather than mis-specified;
#          this is the cheapest test of that reading.
#   H1     matrix 6's hybrid, same question.
#
# CACHE_DIR defaults to the PARTIAL cache (jobs/prepare_partial.sh), not
# prepare.sh's complete-only one: a dataset asking for min_trials=5 against the
# complete-only cache refuses at construction, naming the sims that cache lacks.
set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
RUNS="${RUNS_ROOT:-$CANCER/runs/2026-09-24}"
SEED="${SEED:-20260924}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Pre-built PARTIAL clone cache (jobs/prepare_partial.sh): it holds the sims
# with 1-24 replicates as well, which is what --min-trials 5 below asks for.
CACHE_DIR="${CACHE_DIR:-$CANCER/data/cache/clone_top100_partial_v1}"
# The trial bar, shared by every run so the four are comparable. Must be <= the
# cache's own --min-trials, or the dataset refuses.
MIN_TRIALS="${MIN_TRIALS:-5}"

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

# array index -> run.   Index:  0     1      2    3
RUNNAME=(AT10 AT10s1 R27 H1)
MODEL=(armtoken armtoken cloneatt hybrid)
# R27 is R26 spelled out: the cloneatt preset does NOT carry matrices 2-4's
# findings the way the armtoken one does, so its four repair flags have to be
# named on the command line, exactly as train4.sh named them.
R26_FLAGS=(--z-score-x structured --input-space copy --freq-mode feature
           --attn-ln --flow-num-transforms 3 --d-model 256)
IS_R26=(0 0 1 0)
# Per-run seed. AT10s1 is the replicate; everything else keeps the matrix
# default so it is comparable with matrices 5 and 6.
RUNSEED=("" 1 "" "")
USES_CACHE=(1 1 1 1)   # every run is a clone-set model

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
       --num-workers "$NUM_WORKERS"
       --min-trials "$MIN_TRIALS")
  if [ -n "$CACHE_DIR" ] && [ "${USES_CACHE[$i]}" = "1" ]; then
    CMD+=(--cache-dir "$CACHE_DIR")
  fi
  if [ "${IS_R26[$i]}" = "1" ]; then
    CMD+=("${R26_FLAGS[@]}")
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the four command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  min_trials=$MIN_TRIALS  cache=${CACHE_DIR:-<none>}  seed=${RUNSEED[$i]:-$SEED}"
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
