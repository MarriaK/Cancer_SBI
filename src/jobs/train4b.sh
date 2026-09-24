#!/bin/bash
#SBATCH -J train-matrix4b
#SBATCH -p general-gpu
#SBATCH --gres=gpu:1
#SBATCH -C a100
#SBATCH -n 10 -N 1
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err
#
# The FOURTH repair matrix, one array task per run. jobs/train.sh (matrix 1),
# jobs/train2.sh (matrix 2) and jobs/train3.sh (matrix 3) are untouched.
#   sbatch jobs/train4.sh                 # all twelve
#   sbatch --array=3 jobs/train4.sh       # R20 only
#   DRY_RUN=1 bash jobs/train4.sh         # print all twelve commands, run nothing
#
# Env overrides: CANCER_SBI_DATA_ROOT, CANCER_SBI_SPLIT, RUNS_ROOT, SEED,
# NUM_WORKERS, CACHE_DIR, ALLOW_TEST_AS_VAL, DRY_RUN. Same split gate, same
# refusal of a non-empty checkpoint directory, same 12 h wall clock as train3.sh.
#
# Matrix 3's best model is CloneAtt R12:
#   BASE_R12 = --z-score-x structured --input-space copy --freq-mode feature
#              --attn-ln --flow-num-transforms 3      (R^2 0.413, 8/44 SBC fails)
# Every run below is that command plus the one thing it tests, so a difference
# in R^2 is attributable. The per-arm analysis says the best-learned arms carry
# a small consistent posterior-mean bias, and names two suspects: the flow's
# tail_bound=3 clipping (R18) and the scalar `structured` theta standardisation
# (R19). The rest ask whether CloneAtt's attention is under-powered rather than
# mis-regularised.
#
# What each run tests:
#   R17   trap 6 put right: logits scaled by sqrt(d_model/n_heads) instead of
#         sqrt(d_model), so the attention is 2.83x sharper. Is the flat
#         attention why the encoder underuses its clones?
#   R18   a wider spline support (5 instead of the published 3): is the
#         posterior-mean bias the flow clipping theta at the tail bound?
#   R19   per-dimension theta whitening instead of the single scalar
#         `structured` one: is the bias the other suspect, the shared scale?
#   R20   pool the 25 trial embeddings with a PMA instead of sbi's masked mean
#         -- do the trials of a sim deserve different weights?
#   R20s1 R20 on seed 1, because seed noise on R^2 is +-0.05 and R20 is the one
#         run here that adds a module rather than changing a constant.
#   R21   a 256-wide per-trial embedding (published 128): capacity, width.
#   R22   4 attention heads instead of 8: capacity, head count.
#   R23   64 inducing points instead of 32: capacity, bottleneck.
#   R24   R21 and R23 together, the one capacity combination worth a task.
#   R12s1 the BASE itself on seed 1 -- the yardstick every row above is read
#         against.
#   R12s2 the base on seed 2, the second point of that spread.
#   R25   R17 + R20 + R18 together: if the three independent switches each help
#         a little, do they compose?
#
# Every run is cloneatt and every one reads clone sets, so USES_CACHE is 1
# throughout -- unlike train3.sh, which has two DominantClone exceptions.
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

# array index -> run.   Index:   0   1   2   3     4     5   6   7   8     9      10   11
RUNNAME=(R18s1 R18s2 R26 R26s1)   # follow-up: seeds of the tail_bound win, and tail_bound 5 + d_model 256
# Every run in this matrix is CloneAtt.
MODEL=(cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt cloneatt)
# BASE_R12 (cloneatt) = --z-score-x structured --input-space copy
#                       --freq-mode feature --attn-ln --flow-num-transforms 3
# The four base flags below are the same on every run; only R19 moves one of
# them, replacing `structured` with `independent`, which is what it tests.
ZSCORE=(structured structured structured structured)
INPUT_SPACE=(copy copy copy copy copy copy copy copy copy copy copy copy)
FREQ_MODE=(feature feature feature feature feature feature feature feature feature feature feature feature)
ATTN_LN=(1 1 1 1 1 1 1 1 1 1 1 1)
FLOW_TRANSFORMS=(3 3 3 3 3 3 3 3 3 3 3 3)
# The six switches this matrix adds. Each defaults to the published value and
# appears on exactly the runs that test it.
ATTN_SCALE=("" "" "" "")
TAIL_BOUND=(5 5 5 5)
TRIAL_POOL=("" "" "" "")
D_MODEL=("" "" 256 256)
N_HEADS=("" "" "" "")
NUM_INDUCING=("" "" "" "")
# Per-run seed. The three replicates are seeds 1 and 2; everything else keeps
# the matrix default so it is comparable with matrix 3.
RUNSEED=(1 2 "" 1)
USES_CACHE=(1 1 1 1)

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
  if [ -n "${FLOW_TRANSFORMS[$i]}" ]; then
    CMD+=(--flow-num-transforms "${FLOW_TRANSFORMS[$i]}")
  fi
  if [ -n "${ATTN_SCALE[$i]}" ]; then
    CMD+=(--attn-scale "${ATTN_SCALE[$i]}")
  fi
  if [ -n "${TAIL_BOUND[$i]}" ]; then
    CMD+=(--tail-bound "${TAIL_BOUND[$i]}")
  fi
  if [ -n "${TRIAL_POOL[$i]}" ]; then
    CMD+=(--trial-pool "${TRIAL_POOL[$i]}")
  fi
  if [ -n "${D_MODEL[$i]}" ]; then
    CMD+=(--d-model "${D_MODEL[$i]}")
  fi
  if [ -n "${N_HEADS[$i]}" ]; then
    CMD+=(--n-heads "${N_HEADS[$i]}")
  fi
  if [ -n "${NUM_INDUCING[$i]}" ]; then
    CMD+=(--num-inducing "${NUM_INDUCING[$i]}")
  fi
}

if [ "${DRY_RUN:-0}" = "1" ]; then
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    build_cmd "$i"
    echo "# ${RUNNAME[$i]}  model=${MODEL[$i]}  ckpt=$RUNS/${RUNNAME[$i]}/checkpoints"
    echo "${CMD[@]}"
  done
  # A dry run reports the split gate rather than failing on it: the point of the
  # dry run is to read the twelve command lines, and it is usually done on a
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

echo "host=$(hostname)  run=${RUNNAME[$i]}  model=${MODEL[$i]}  z_score_x=${ZSCORE[$i]:-<preset>}  attn_scale=${ATTN_SCALE[$i]}  tail_bound=${TAIL_BOUND[$i]}  trial_pool=${TRIAL_POOL[$i]}  d_model=${D_MODEL[$i]}  n_heads=${N_HEADS[$i]}  num_inducing=${NUM_INDUCING[$i]}  seed=${RUNSEED[$i]:-$SEED}"
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
