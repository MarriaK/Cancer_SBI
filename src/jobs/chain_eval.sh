#!/bin/bash
# Submit the evaluation chain for one trained run: test sampling -> val sampling
# -> metrics + figure D -> recalibration -> recalibrated metrics.
#
#   jobs/chain_eval.sh RUN MODEL [TRAIN_DEP] [PREV_SAMPLER]
#
#   RUN           run tag, e.g. AT10  (checkpoint runs/2026-09-24/RUN/checkpoints/best.pt)
#   MODEL         clonemlp | cloneatt | dominantclone | armtoken | hybrid
#   TRAIN_DEP     optional SLURM job id (or array task, 123_4) the sampling must wait for
#   PREV_SAMPLER  optional job id of the previous chain's val sampler: samplers run one at
#                 a time (afterany), because the per-user GPU allowance is small
#                 (QOSMaxGRESPerUser killed five training tasks on 2026-09-24)
#
# Prints the val sampler's job id on stdout (feed it as PREV_SAMPLER to the next call);
# everything else goes to stderr. Run from ~/cancer/src on the cluster.
set -euo pipefail
RUN="$1"; MODEL="$2"; TRAIN_DEP="${3:-}"; PREV="${4:-}"
CANCER="${CANCER:-$HOME/cancer}"
RUNS_ROOT="${RUNS_ROOT:-$CANCER/runs/2026-09-24}"
RESULTS_ROOT="${RESULTS_ROOT:-$CANCER/results/2026-09-24}"
P="$RESULTS_ROOT/$RUN/posteriors"; O="$RESULTS_ROOT/$RUN"
CKPT="$RUNS_ROOT/$RUN/checkpoints/best.pt"

deps=()
[ -n "$TRAIN_DEP" ] && deps+=("afterok:$TRAIN_DEP")
[ -n "$PREV" ] && deps+=("afterany:$PREV")
depflag=""
if [ ${#deps[@]} -gt 0 ]; then depflag="--dependency=$(IFS=,; echo "${deps[*]}")"; fi

sid=$(MODEL=$MODEL CKPT=$CKPT POST=$P RUN_TAG=$RUN sbatch --parsable $depflag --array=0 -J sample-$RUN jobs/sample.sh); sid=${sid%%;*}
vid=$(MODEL=$MODEL CKPT=$CKPT POST=$P RUN_TAG=$RUN PARTITION=val sbatch --parsable --dependency=afterany:$sid --array=0 -J sampleval-$RUN jobs/sample.sh); vid=${vid%%;*}
POST=$P OUT=$O RUN_TAG=$RUN sbatch --parsable --dependency=afterok:$sid -J analyze-$RUN jobs/analyze.sh >/dev/null
POST=$P OUT=$O RUN_TAG=$RUN sbatch --parsable --dependency=afterok:$sid -J shrink-$RUN jobs/shrink.sh >/dev/null
rid=$(VAL=$P/posteriors_${MODEL}_${RUN}_val.npz TEST=$P/posteriors_${MODEL}_${RUN}.npz OUT=$RESULTS_ROOT/${RUN}rc/posteriors MODEL=$MODEL RUN_TAG=${RUN}rc sbatch --parsable --dependency=afterok:$sid:$vid -J recal-$RUN jobs/recalibrate.sh)
POST=$RESULTS_ROOT/${RUN}rc/posteriors OUT=$RESULTS_ROOT/${RUN}rc RUN_TAG=${RUN}rc sbatch --parsable --dependency=afterok:$rid -J analyze-${RUN}rc jobs/analyze.sh >/dev/null
echo "$RUN: sample=$sid val=$vid recal=$rid deps=[${depflag#--dependency=}]" >&2
echo "$vid"
