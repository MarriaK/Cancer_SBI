#!/bin/bash
#SBATCH -J prepare-matrix
#SBATCH -p general
#SBATCH -n 16 -N 1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# One-time preparation before the repair matrix (MODEL_IMPROVEMENT_PLAN.md §5
# steps 1 and 3): carve the validation split, build the clone cache, verify it.
#   sbatch jobs/prepare.sh
# Idempotent: refuses to re-carve an already carved split; re-uses the cache
# unless FORCE_CACHE=1.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CANCER="${CANCER:-$HOME/cancer}"
DATA="$CANCER/data/Guassian_Normal/simulation_outputs"
SPLIT_IN="$CANCER/data/train_test_split.pkl"
SPLIT_OUT="$CANCER/data/train_val_test_split.pkl"
CACHE="$CANCER/data/cache/clone_top100_v1"
cd "$CANCER/src"

echo "=== 1. validation split ==="
if [[ -f "$SPLIT_OUT" ]]; then
    echo "already exists: $SPLIT_OUT"
else
    python -m cancer_sbi.cli.make_split --add-val --in "$SPLIT_IN" --out "$SPLIT_OUT"
fi
python - "$SPLIT_OUT" <<'PY'
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb"))
print({k: len(v) for k, v in d.items()})
assert "val_ids" in d and not set(d["val_ids"]) & set(d["test_ids"])
PY

echo "=== 2. clone cache (expect 3600 simulations discovered) ==="
FORCE=""; [[ "${FORCE_CACHE:-0}" == "1" ]] && FORCE="--force"
python utilities/build_clone_cache.py --root "$DATA" --split "$SPLIT_OUT" \
    --out "$CACHE" --workers 16 $FORCE

echo "=== 3. verify cache (bit-exact spot check) ==="
python utilities/verify_clone_cache.py --root "$DATA" --split "$SPLIT_OUT" \
    --cache "$CACHE" --n-sims 40 --n-trials 3
du -sh "$CACHE"
echo "PREPARE OK"
