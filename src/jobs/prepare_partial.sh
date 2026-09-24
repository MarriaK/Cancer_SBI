#!/bin/bash
#SBATCH -J prepare-partial
#SBATCH -p general
#SBATCH -n 16 -N 1
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Preparation for matrix 7 (jobs/train7.sh): a SECOND clone cache that also
# holds the simulations with fewer than 25 replicate tumours.
#   sbatch jobs/prepare_partial.sh
#   DRY_RUN=1 bash jobs/prepare_partial.sh    # print the two commands, run nothing
#
# jobs/prepare.sh is untouched and still owns the validation split and the
# complete-only cache (data/cache/clone_top100_v1). This script does NOT carve
# a split: it reads the three-key split that one already wrote, so both caches
# describe the same partition. Re-uses an existing cache unless FORCE_CACHE=1.
#
# The only difference from prepare.sh's cache is --min-trials 5: 441 of the
# cluster's 3,600 sims have 1-24 complete replicates and are dropped whole by
# trap 10. At 5 they come back with NaN in the slots they have no file for, and
# the encoders mask those slots (ArmToken natively, CloneAtt/CloneMLP through
# sbi's NaN-aware trial pooling). 5 rather than 1 because a sim summarised from
# one or two replicates carries almost no between-replicate signal, which is
# half of what the per-arm moments are.
#
# This cache is for TRAINING and VALIDATION. The test set keeps the published
# complete-sim rule wherever it is built (data/loaders.py), so matrix 7's
# numbers stay comparable with matrices 1-6; a cache is a superset of what a
# split needs, so the same directory serves the unchanged test loader too.
set -euo pipefail

CANCER="${CANCER:-$HOME/cancer}"
DATA="$CANCER/data/Guassian_Normal/simulation_outputs"
SPLIT="$CANCER/data/train_val_test_split.pkl"
CACHE="$CANCER/data/cache/clone_top100_partial_v1"
MIN_TRIALS="${MIN_TRIALS:-5}"
FORCE=""; [[ "${FORCE_CACHE:-0}" == "1" ]] && FORCE="--force"

BUILD=(python utilities/build_clone_cache.py --root "$DATA" --split "$SPLIT"
       --out "$CACHE" --workers 16 --min-trials "$MIN_TRIALS" $FORCE)
VERIFY=(python utilities/verify_clone_cache.py --root "$DATA" --split "$SPLIT"
        --cache "$CACHE" --n-sims 40 --n-trials 3)

# Printed before `conda activate`, so the dry run works on a laptop that has no
# cancer-sbi environment and no cluster paths at all.
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "# 1. partial clone cache (expect 3600 simulations discovered)"
    echo "${BUILD[@]}"
    echo "# 2. verify cache (bit-exact spot check, plus the NaN padding)"
    echo "${VERIFY[@]}"
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$CANCER/src"

# The split is prepare.sh's output, not this script's: matrix 7 changes which
# sims are KEPT, never which side of the partition a sim falls on.
if [[ ! -f "$SPLIT" ]]; then
    echo "REFUSING: $SPLIT does not exist. Run jobs/prepare.sh first." >&2
    exit 1
fi

echo "=== 1. partial clone cache (expect 3600 simulations discovered) ==="
"${BUILD[@]}"

echo "=== 2. verify cache (bit-exact spot check, plus the NaN padding) ==="
"${VERIFY[@]}"
du -sh "$CACHE"
echo "PREPARE PARTIAL OK"
