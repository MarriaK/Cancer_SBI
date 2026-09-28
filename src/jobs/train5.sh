#!/bin/bash
#SBATCH -J train-matrix5
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-5
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The FIFTH matrix, one array task per run. jobs/train.sh (matrix 1),
# jobs/train2.sh, jobs/train3.sh and jobs/train4.sh are untouched.
#   sbatch jobs/train5.sh                 # all six
#   sbatch --array=2 jobs/train5.sh       # AT1 only
#   DRY_RUN=1 bash jobs/train5.sh         # print all six commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN. Same split gate, same
# refusal of a non-empty checkpoint directory, same 12 h wall clock as train3.sh.
#
# This matrix is not another switch on CloneAtt: it is a fourth model. Matrices
# 1-4 moved CloneAtt's R^2 from ~0 to 0.41 and then stalled, and the per-arm
# analysis kept saying the same thing -- the best-learned arms carry a
# consistent posterior-mean bias, and the worst-learned ones are not learned at
# all. Every encoder so far tokenises a CLONE and pools the clones away, so
# nothing ties output arm 17 to input arm 17. ArmToken inverts the set: the 44
# ARMS are the tokens, summarised over clones by eight frequency-weighted
# moments, every weight shared across arms, and arm identity reaches the flow
# only through the order the 44 per-arm blocks are concatenated in.
#
# The preset already carries matrices 2-4's findings (--z-score-x structured,
# --input-space copy, LayerNorm on, a 3-transform flow), so the base run names
# no repair flag at all; the rows below each add exactly one thing.
#
# What each run tests:
#   AT0   the encoder itself, on the matrix seed. The yardstick.
#   AT0s1 AT0 on seed 1 -- a brand-new architecture needs its own seed spread
#         before any row below can be read (matrices 2-4: +-0.05 on R^2).
#   AT1   pool the 25 trials with a PMA per arm instead of the masked
#         [mean, sd] over trials: is the between-replicate spread better used
#         by attention than by being handed to the MLP as two more numbers?
#   AT2   trap 6's attention temperature, the matrix-4 question asked of the
#         arm attention: sqrt(d_token/n_heads) instead of sqrt(d_token).
#   AT3   NO mixing across arms at all (0 ISABs). The strictest form of the
#         equivariance, and the ablation that says whether cross-arm context
#         is worth its parameters -- if AT3 matches AT0, the ISAB is dead
#         weight and the model is 44 independent regressions.
#   AT4   the embedding group's learning rate at 1e-3 instead of the published
#         1e-4. Matrix 2's R7 asked this of CloneAtt; ArmToken's encoder is
#         61k parameters against the flow's, so near-freezing it is a bigger
#         handicap here than it was there.
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
# epochs. The flag is only added when it is non-empty.
CACHE_DIR="${CACHE_DIR:-}"

# Gate: the split must carry a third, validation set -- identical to train3.sh.
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

# array index -> run.   Index:  0     1      2    3    4    5
RUNNAME=(AT0 AT0s1 AT1 AT2 AT3 AT4)
MODEL=(armtoken armtoken armtoken armtoken armtoken armtoken)
# No BASE_* line: the armtoken preset already carries --z-score-x structured,
# --input-space copy, attn_ln and a 3-transform flow, so naming them would make
# a preset default look like a per-run choice.
TRIAL_POOL=("" "" attention "" "" "")
ATTN_SCALE=("" "" "" standard "" "")
# "" means "leave the preset's 1 alone"; "0" is a real value and must reach the
# command line, so this array is tested with -n (set) and not with -n "$x" the
# way the optional-value arrays above are.
ARM_LAYERS=("" "" "" "" 0 "")
EMBED_LR=("" "" "" "" "" 1e-3)
# Per-run seed. AT0s1 is the replicate; everything else keeps the matrix
# default so it is comparable with matrices 3 and 4.
RUNSEED=("" 1 "" "" "" "")
USES_CACHE=(1 1 1 1 1 1)   # every run is a clone-set model

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
  if [ -n "${TRIAL_POOL[$i]}" ]; then
    CMD+=(--trial-pool "${TRIAL_POOL[$i]}")
  fi
  if [ -n "${ATTN_SCALE[$i]}" ]; then
    CMD+=(--attn-scale "${ATTN_SCALE[$i]}")
  fi
  # `0` is the value AT3 tests, so the emptiness test is on the string itself.
  if [ -n "${ARM_LAYERS[$i]}" ]; then
    CMD+=(--arm-layers "${ARM_LAYERS[$i]}")
  fi
  if [ -n "${EMBED_LR[$i]}" ]; then
    CMD+=(--embed-lr "${EMBED_LR[$i]}")
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4 5; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the six command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  trial_pool=${TRIAL_POOL[$i]:-<preset>}  attn_scale=${ATTN_SCALE[$i]:-<preset>}  arm_layers=${ARM_LAYERS[$i]:-<preset>}  embed_lr=${EMBED_LR[$i]:-<preset>}  seed=${RUNSEED[$i]:-$SEED}"
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
