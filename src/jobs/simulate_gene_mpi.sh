#!/bin/bash
# Gene-selection simulations as ONE MPI job (the way the arm dataset was made).
# Rank 0 samples every theta and scatters sim ids round-robin to all ranks.
# Needs mpi4py in the env, built against the loaded OpenMPI; if the world size
# comes up as 1 under a multi-task allocation the script exits with instructions.
# Prefer simulate_gene_array.sh unless you specifically want the single-job form.
#
#SBATCH -J gene_sim_mpi
#SBATCH -p general
#SBATCH -n 40 -N 1
#SBATCH --mem-per-cpu=8G
#SBATCH -t 24:00:00
#SBATCH -o /home/mak23055/cancer/logs/%x_%j.out
#SBATCH -e /home/mak23055/cancer/logs/%x_%j.err

set -euo pipefail
CANCER="${CANCER:-$HOME/cancer}"
START="${START:-1}"
END="${END:-6000}"
NTRIALS="${NTRIALS:-25}"
SEED="${SEED:-12345}"
OUT="${OUT:-$CANCER/data/gene_selection}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-sbi
python -c "import mpi4py, sistem; print('mpi4py', mpi4py.__version__, 'sistem', sistem.__version__)"

cd "$CANCER/src/data_generation"
# If srun does not wire MPI, try: srun --mpi=pmix  or  mpirun -np $SLURM_NTASKS python ...
srun python simulate_gene_selection.py --start "$START" --end "$END" \
    --ntrials "$NTRIALS" --output "$OUT" --seed "$SEED"
