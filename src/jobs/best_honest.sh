#!/bin/bash
#SBATCH -J eval-besthonest
#SBATCH -p general
#SBATCH -n 4 -N 1
#SBATCH -t 02:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err
#
# Stage 3: rebuild the whole best-honest figure set from the stored posteriors.
# CPU only -- it reads .npz files and draws; the GPU work happened in jobs/sample.sh.
#
#   sbatch jobs/best_honest.sh
#   RESULTS=$HOME/cancer/results/2026-09-24 OUT=$HOME/cancer/results/best_honest \
#     sbatch jobs/best_honest.sh
#   DRY_RUN=1 bash jobs/best_honest.sh          # print every command, run nothing
#
# For each run named by the manifest it re-runs the per-run stage (metrics + figures
# A-C, TARP, figure D), then collects the winners into $OUT and draws the cross-run
# figures (E, S3, the calibration panel, the pooled D panel) from all of them.
#
# Env overrides: MANIFEST (seed-family manifest), RESULTS (campaign dir holding
#   <run>/posteriors/), OUT (collection dir, must be a sibling of RESULTS),
#   DRY_RUN=1.
set -euo pipefail
CANCER="$HOME/cancer"
MANIFEST="${MANIFEST:-$CANCER/src/cancer_sbi/evaluation/manifests/best_honest_2026-09-24.json}"
RESULTS="${RESULTS:-$CANCER/results/2026-09-24}"
OUT="${OUT:-$CANCER/results/best_honest}"

if [ ! -f "$MANIFEST" ]; then
  echo "REFUSING: no manifest at $MANIFEST" >&2
  exit 1
fi
# collect_best_honest.py writes <results root>/<out name>, so the two must share a parent.
ROOT="$(dirname "$RESULTS")"
CAMPAIGN="$(basename "$RESULTS")"
OUTNAME="$(basename "$OUT")"
if [ "$(dirname "$OUT")" != "$ROOT" ]; then
  echo "REFUSING: OUT ($OUT) must sit beside RESULTS ($RESULTS)." >&2
  echo "  RESULTS=$ROOT/<campaign> OUT=$ROOT/<name> sbatch jobs/best_honest.sh" >&2
  exit 1
fi

# The run list comes from the manifest, never from a copy pasted here. Parsed with the
# stdlib before conda is activated, so DRY_RUN works on a login node with no env.
PY_JSON="python3"
command -v "$PY_JSON" >/dev/null 2>&1 || PY_JSON="python"
command -v "$PY_JSON" >/dev/null 2>&1 || { echo "REFUSING: no python3 to read the manifest" >&2; exit 1; }
RUNS="$("$PY_JSON" -c 'import json,sys
m = json.load(open(sys.argv[1]))
runs = [r for e in m["encoders"] for r in [e["headline"], *e["members"]]]
print(" ".join(dict.fromkeys(runs)))' "$MANIFEST")"
if [ -z "$RUNS" ]; then
  echo "REFUSING: the manifest names no runs" >&2
  exit 1
fi

# Everything below the next line actually runs -- unless DRY_RUN=1, in which case
# `step` only prints and nothing (conda included) is touched.
if [ "${DRY_RUN:-0}" != "1" ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate cancer-sbi
  # conda ships a newer libstdc++ than /lib64; without this scipy dies with
  #   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  cd "$CANCER/src"
  mkdir -p "$OUT"
  echo "host=$(hostname)  results=$RESULTS  out=$OUT"
fi

step () {
  echo "$*"
  if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi
}

# ---- per run: metrics + figures A-C, the joint TARP test, figure D
for RUN in $RUNS; do
  echo "=== $RUN"
  P="$RESULTS/$RUN/posteriors"
  O="$RESULTS/$RUN"
  step python -m cancer_sbi.evaluation.poster_metrics --in-dir "$P" --out-dir "$O" --run-tag "$RUN"
  step python -m cancer_sbi.evaluation.tarp --in-dir "$P" --out-dir "$O" --run-tag "$RUN"
  step python -m cancer_sbi.evaluation.fig_shrinkage --in-dir "$P" --out-dir "$O" --run-tag "$RUN"
done

# ---- cross run: collect the winners, then draw the comparisons
echo "=== collect"
step python utilities/collect_best_honest.py --results "$ROOT" --campaign "$CAMPAIGN" \
  --out "$OUTNAME" --manifest "$MANIFEST"
step python -m cancer_sbi.evaluation.best_honest_figures --results "$RESULTS" \
  --manifest "$MANIFEST" --out-dir "$OUT"
