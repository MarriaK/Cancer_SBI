#!/bin/bash
# Prior v4.1 pilot (research_notes/v4_1_design_2026-10-26/DECISION.md): one theta per array task, one trial,
# with the settings of factorial arm G1 or G4 applied as runtime overrides of the module constants before
# simulate_gene_selection.main() runs (the manifest and parameters.pkl record the settings actually used).
# Same prior/trial seeds as calibration v4, so delta and trial 1 are paired by theta with gene_calib_v4.
# Usage:
#   ARM=G1 OUT=/gpfs/scratchfs1/pas23008/$USER/gene_pilot_v41_G1 sbatch -J pilot_v41_G1 --array=0-59 src/jobs/simulate_gene_pilot_v41.sh
#   ARM=G4 OUT=/gpfs/scratchfs1/pas23008/$USER/gene_pilot_v41_G4 sbatch -J pilot_v41_G4 --array=0-59 src/jobs/simulate_gene_pilot_v41.sh
#
#SBATCH -p general
#SBATCH -n 1 -N 1
#SBATCH --mem-per-cpu=3G
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o /home/mak23055/cancer/logs/%x_%A_%a.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%A_%a.err

set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
ARM="${ARM:?set ARM=G1 or ARM=G4}"
START_ID="${START_ID:-1}"
NTRIALS="${NTRIALS:-1}"
SEED="${SEED:-12345}"
OUT="${OUT:?set OUT to a scratch directory}"
SIM_ID=$(( START_ID + SLURM_ARRAY_TASK_ID ))

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
mkdir -p "$OUT"
cd "$CANCER/src/data_generation"
echo "arm $ARM: sim id $SIM_ID -> $OUT"

ARM="$ARM" python - --start "$SIM_ID" --end "$SIM_ID" --ntrials "$NTRIALS" --output "$OUT" --seed "$SEED" --no-sim-log <<'EOF'
import os, sys
from types import MappingProxyType
import simulate_gene_selection as sgs

ARMS = {   # factorial arms (DECISION.md); trunk 1000 is the module default in both
    "G1": dict(PASS_RATE_RANGE=(1e-4, 1.5e-2),
               RHO_CIN_LOADINGS={"arm_rate": 0.5, "chromosomal_rate": 0.5, "focal_driver_rate": 0.5, "pass_rate": 0.5}),
    "G4": dict(PASS_RATE_RANGE=(1e-4, 3e-2),
               RHO_CIN_LOADINGS={"arm_rate": 0.5, "chromosomal_rate": 0.0, "focal_driver_rate": 0.8, "pass_rate": 0.8}),
}
arm = os.environ["ARM"]
assert sgs.PRIOR_VERSION == 5 and sgs.PASS_TRUNK_GENS == 1000.0, "expected the prior v4.1 code"
for k, v in ARMS[arm].items():
    setattr(sgs, k, MappingProxyType(dict(v)) if k == "RHO_CIN_LOADINGS" else v)
sgs.rho_cin_loadings()                 # validates the override
print("arm", arm, {k: getattr(sgs, k) for k in ("PASS_TRUNK_GENS", "PASS_RATE_RANGE")}, dict(sgs.RHO_CIN_LOADINGS))
sys.argv = ["simulate_gene_selection.py"] + sys.argv[1:]
sgs.main()
EOF
