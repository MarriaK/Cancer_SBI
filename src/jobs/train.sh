#!/bin/bash
#SBATCH -J train-matrix
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-4
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The five-run matrix (R0 / R1 / R2 / R4 / D0). One array task per run.
#   sbatch jobs/train.sh                 # all five
#   sbatch --array=1 jobs/train.sh       # R1 only (highest priority)
#   DRY_RUN=1 bash jobs/train.sh         # print all five commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN.
#
# The split default is the THREE-key train_val_test_split.pkl, and the script
# refuses a split with no val_ids: without one, early stopping runs on the test
# set and best.pt is chosen on the sims the score is reported on. Carve it once
# with `make_split --add-val` (see jobs/README.md). ALLOW_TEST_AS_VAL=1 waives
# the check deliberately.
#
# -t is 12:00:00, not the 08:00:00 of the first draft: at ~28 min/epoch uncached
# (see CACHE_DIR), 60 epochs does not fit in eight hours, and an array task
# killed at the wall clock loses the run.
#
# R0 is the control (z_score_x unchanged). R1 adds theta whitening. R2 is R1 plus
# the copy-space input. R4 is CloneAtt with the frequency renormalisation, ln off.
# D0 retrains DominantClone with --require-all-trials, i.e. on the same sims the
# clone-set models see (trap 10), so the three models can be compared honestly.
# D0 passes no --z-score-x: its preset is already "structured", and naming the
# value here would make the preset's default look like a per-run choice.
set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
export CANCER_SBI_DATA_ROOT="${CANCER_SBI_DATA_ROOT:-$CANCER/data/Guassian_Normal/simulation_outputs}"
export CANCER_SBI_SPLIT="${CANCER_SBI_SPLIT:-$CANCER/data/train_val_test_split.pkl}"
RUNS="${RUNS_ROOT:-$CANCER/runs/2026-09-24}"
SEED="${SEED:-20260924}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Pre-built clone cache (src/utilities/build_clone_cache.py). Empty == read the
# gzipped trial files, which is what every published run did. Set CACHE_DIR to a
# built cache to make every run read from it instead; the flag is only added when
# it is non-empty.
CACHE_DIR="${CACHE_DIR:-}"

# Gate: the split must carry a third, validation set. MODEL_IMPROVEMENT_PLAN.md
# §5 step 1 -- with only train/test, cli/train.py falls back to early stopping on
# the TEST set and every run's best.pt is selected on the sims it is scored on,
# so no two runs in the matrix can be compared honestly. The check is stdlib-only
# and runs before `conda activate`, so it takes whichever python is on PATH.
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

# array index -> run.  Index:      0         1          2            3          4
RUNNAME=(R0 R1 R2 R4 D0)
MODEL=(clonemlp clonemlp clonemlp cloneatt dominantclone)
ZSCORE=(none structured structured structured "")   # D0 keeps the preset's "structured"
INPUT_SPACE=(log2 log2 copy "" "")    # copy-space input is R2 only; --input-space is
                                      # clonemlp-only, so R4 and D0 leave it unset
FREQ_RENORM=(0 0 0 1 0)               # CloneAtt frequency repair is R4 only
REQUIRE_ALL_TRIALS=(0 0 0 0 1)        # the same-sims retrain is D0 only
USES_CACHE=(1 1 1 1 0)                # the clone cache holds (25, top_k, 45) clone
                                      # sets, which DominantClone does not read

# Build the full command for one run index into the global array CMD.
build_cmd() {
  local i="$1"
  CMD=(python -m cancer_sbi.cli.train
       --model "${MODEL[$i]}"
       --data-root "$CANCER_SBI_DATA_ROOT"
       --split "$CANCER_SBI_SPLIT"
       --ckpt-dir "$RUNS/${RUNNAME[$i]}/checkpoints"
       --out "$RUNS/${RUNNAME[$i]}"
       --seed "$SEED"
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
  if [ "${FREQ_RENORM[$i]}" = "1" ]; then
    CMD+=(--freq-renorm)
  fi
  if [ "${REQUIRE_ALL_TRIALS[$i]}" = "1" ]; then
    CMD+=(--require-all-trials)
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the five command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  z_score_x=${ZSCORE[$i]:-<preset>}  input_space=${INPUT_SPACE[$i]}  freq_renorm=${FREQ_RENORM[$i]}  require_all_trials=${REQUIRE_ALL_TRIALS[$i]}  seed=$SEED"
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
