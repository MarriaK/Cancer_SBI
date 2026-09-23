import sys
import os
import copy
import pickle as pkl
import argparse
import numpy as np
import torch
from torch.distributions import Independent, Uniform, Normal, Beta, Laplace
import numpy as np, gzip, pickle as pkl
import os
import sys

# --- MPI detection / helpers ---

from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
USING_MPI = True

'''
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1
    USING_MPI = False
'''

def is_mpi_active() -> bool:
    """True iff mpi4py imported AND world size > 1."""
    return USING_MPI and SIZE > 1

def is_slurm_multi_alloc() -> bool:
    """True iff SLURM allocated >1 tasks."""
    try:
        return int(os.environ.get("SLURM_NTASKS", "1")) > 1
    except ValueError:
        return False

def require_mpi_or_exit():
    """
    If SLURM allocated multiple tasks but MPI world==1, print a clear error and exit.
    Prevents silent 'rank 0 everywhere' behavior.
    """
    if is_slurm_multi_alloc() and not is_mpi_active():
        if RANK == 0:  # single process anyway
            print(
                "[ERROR] SLURM_NTASKS indicates a multi-task job, "
                "but MPI world size == 1.\n"
                "This means mpi4py is not wired to the launcher.\n"
                "Use either:\n"
                "  srun python your_script.py  (optionally with --mpi=pmix or pmix_v3),\n"
                "or\n"
                "  mpirun -np $SLURM_NTASKS python your_script.py\n"
                "and ensure mpi4py was built against the loaded OpenMPI.",
                file=sys.stderr, flush=True,
            )
        sys.exit(2)

def mpi_print(msg: str):
    """Rank-tagged printing; flush to avoid buffering in MPI runs."""
    print(f"[rank {RANK}/{SIZE}] {msg}", flush=True)



import sistem
from sistem.genome.genome import init_diploid_genome
from sistem.genome.utils import get_num_regions, hg38_chrom_lengths_from_cytoband
from sistem.utilities.utilities import compute_growth_rate
from sistem.lineage.cell import Clone
from sistem.ancestry import GrowthSimulator
from sistem.anatomy import BaseAnatomy, SimpleAnatomy
from sistem import Parameters
from sistem.data.summarize import extract_largest_clones, arm_CNratios_single, arm_CNratios_bulk

import sistem.selection.arm_library as lib
from sbi.utils.user_input_checks import process_prior

# ----------------- Device -----------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ----------------- Priors -----------------
arm_low, arm_high = 1e-5, 1e-4
chrom_low, chrom_high = 1e-6, 1e-5

####### gaussian normal ##########
coeff_mu, coeff_var = 0, 0.2 

####### gaussian negative ##########
#coeff_mu, coeff_var = -0.25, 0.10  # mean shifted negative, narrow spread

####### laplace positive ##########
#coeff_mu, coeff_var = 0.35, 0.15   # mean shifted further positive, heavier tails

####### laplace negative ##########
#coeff_mu, coeff_var = -0.35, 0.15  # this range give selection coffeficent smaller than - 1 and sistem model gives error for calculating the cancer fitness
#coeff_mu , coeff_var= -0.2, 0.1

def create_prior(arm_low, arm_high, chrom_low, chrom_high, coeff_mu, coeff_var, device):
    low = torch.tensor([arm_low, chrom_low], dtype=torch.float32).to(device)
    high = torch.tensor([arm_high, chrom_high], dtype=torch.float32).to(device)
    mutation_rate_prior = Independent(Uniform(low, high, validate_args=False), reinterpreted_batch_ndims=1)

    #selection_coeffient_prior = Independent(Laplace(torch.tensor([coeff_mu] * 44, dtype=torch.float32).to(device), 
    #                                               torch.tensor([coeff_var] * 44, dtype=torch.float32).to(device), 
    #                                                validate_args=False), reinterpreted_batch_ndims=1)
    
    selection_coeffient_prior = Independent(Normal(torch.tensor([coeff_mu] * 44, dtype=torch.float32).to(device), 
                                                   torch.tensor([coeff_var] * 44, dtype=torch.float32).to(device)
                                                   , validate_args=False), reinterpreted_batch_ndims=1)
    
    combined_prior = [mutation_rate_prior, selection_coeffient_prior]
    prior, _, _ = process_prior(combined_prior)
    return prior

# ----------------- Simulation -----------------
def simulate_arm_model(theta, out_dir, num_trials, base_seed=None):
    """Runs up to num_trials and writes parameters+results under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    # Persist theta used
    with open(os.path.join(out_dir, 'parameters.pkl'), 'wb') as f:
        pkl.dump(theta, f)

    arm_rate = float(theta[0])
    chromosomal_rate = float(theta[1])
    arm_deltas = {}
    for i in range(0, 44, 2):
        arm_deltas[f'chr{int(i/2) + 1}'] = [float(theta[i+2]), float(theta[i+3])]

    for t in range(1, num_trials + 1):
        trial_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_dir, exist_ok=True)

        # Optional deterministic seeding per (rank, trial) for reproducibility
        if base_seed is not None:
            torch.manual_seed(base_seed + t)
            np.random.seed((base_seed + t) % (2**32 - 1))

        params = Parameters(
            region_len=5e6,
            focal_driver_rate=5e-4,
            max_distinct_driv_ratio=0.9,
            min_detectable=5e6,
            arm_rate=arm_rate,
            chromosomal_rate=chromosomal_rate,
            out_dir=trial_dir
        )

        complete = False
        count = 0
        failed_sim, failed_call = 0, 0
        CNratios_largest = None
        CNratios_bulk = None

        while not complete and count < 5:
            result = simulate_arm_model_caller(params, arm_deltas)
            if isinstance(result, tuple):
                complete = True
                CNratios_largest, CNratios_bulk, CNratios_all = result
            elif result == -1:
                failed_call += 1
            elif result == -2:
                failed_sim += 1
            count += 1

        with open(os.path.join(trial_dir, 'results.pkl'), 'wb') as f:
            if complete:
                pkl.dump(([failed_sim, failed_call], CNratios_largest, CNratios_bulk), f)
            else:
                pkl.dump([failed_sim, failed_call], f)


        CNratios_all = np.asarray(CNratios_all, dtype=np.float32)  # compact binary
        
        # write CNratios_all separately (compressed)
        with gzip.open(os.path.join(trial_dir, "CNratios_all.pkl.gz"), "wb") as g:
            pkl.dump(CNratios_all, g, protocol=5)


def simulate_arm_model_caller(params, arm_deltas):
    chrom_lens, arm_ratios = hg38_chrom_lengths_from_cytoband(include_allosomes=True)
    _ = get_num_regions(chrom_lens, params.region_len)

    library = lib.FittedArmLibrary(arm_ratios=arm_ratios, max_distinct_driv_ratio=params.max_distinct_driv_ratio)
    library.initialize(delta=arm_deltas)
    if not hasattr(library, 'arm_sizes'):
        library.arm_sizes = library._arm_sizes

    anatomy = SimpleAnatomy(library, nsites=params.nsites, growth_rate=params.growth_rate, capacities=params.capacities)
    anatomy.init_pop_dynamic(0, 0, params.t_max, N0=params.N0)

    igenome = init_diploid_genome(library)
    _ = Clone(genome=copy.deepcopy(igenome), library=library, popsize=params.N0, site=0, birth_gen=0)

    gs = GrowthSimulator(anatomy)
    try:
        gs.simulate_agents(params=params)
        # print(f"[RANK {rank}] Simulation ran.")
    except Exception:
        return -1

    if len(gs.clones[0]) == 0:
        return -2

    normalize = lambda x: np.log2(max(x, 0.001)) - 1
    chrom_names = [f'chr{i}' for i in range(1, 23)]

    largest_clone = extract_largest_clones(gs)[0]
    CNratios_largest = arm_CNratios_single(largest_clone)
    CNratios_largest_flat = [normalize(val) for chrname in chrom_names for val in CNratios_largest[chrname]]

    CNratios_bulk = arm_CNratios_bulk(gs.clones[0])
    CNratios_bulk_flat = [normalize(val) for chrname in chrom_names for val in CNratios_bulk[chrname]]

    CNratios_all = []
    for i, clone in enumerate(gs.clones[0]):
        CNratios = arm_CNratios_single(clone)
        CNratios_flat = [normalize(val) for chrname in chrom_names for val in CNratios[chrname]]
        CNratios_all.append(CNratios_flat)

    return CNratios_largest_flat, CNratios_bulk_flat, CNratios_all

# ----------------- MPI work distribution -----------------
def chunk_jobs_round_robin(jobs, size):
    chunks = [[] for _ in range(size)]
    for i, job in enumerate(jobs):
        chunks[i % size].append(job)  # <-- was SIZE
    return chunks

def gather_simulation_pairs_mpi(start_id, end_id, output_dir, num_trials, base_seed=None):
    """
    Rank 0:
      - samples theta for [start_id, end_id]
      - builds jobs = [(theta_i (np.array), sim_dir, num_trials, seed), ...]
      - scatters job chunks to all ranks
    Other ranks:
      - receive their chunk and execute
    """
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'simulation_outputs')
    if RANK == 0:
        os.makedirs(data_dir, exist_ok=True)

    # Everyone waits until output dirs exist
    if USING_MPI:
        COMM.Barrier()

    jobs_local = None

    if RANK == 0:
        prior = create_prior(arm_low, arm_high, chrom_low, chrom_high, coeff_mu, coeff_var, device)
        sim_ids = list(range(start_id, end_id + 1))
        num_simulations = len(sim_ids)
        
        # Sample from torch prior on the correct device then move to CPU/NumPy
        with torch.no_grad():
            theta_tensor = prior.sample((num_simulations,))  # shape [N, 46]
        theta_np = theta_tensor.detach().cpu().numpy()

        jobs = []
        for idx, sim_id in enumerate(sim_ids):
            sim_dir = os.path.join(data_dir, f'sim{sim_id}')
            # Create the directory tree ahead of time to reduce rank contention
            os.makedirs(sim_dir, exist_ok=True)
            # Optional: unique base seed per simulation (offset by rank later if desired)
            seed = (base_seed if base_seed is not None else 12345) + sim_id * 1000
            jobs.append((theta_np[idx, :], sim_dir, num_trials, seed))

        # Round-robin split for better balancing if some sims run longer
        chunks = chunk_jobs_round_robin(jobs, SIZE)
    else:
        chunks = None

    # Scatter chunks
    if USING_MPI:
        jobs_local = COMM.scatter(chunks, root=0)
    else:
        # Single-process fallback
        jobs_local = chunks[0] if chunks is not None else []

    # Execute local jobs
    for theta_i, sim_dir, ntrials, seed in jobs_local:
        # seed offset per-rank to avoid identical sequences across ranks
        rank_seed = seed + RANK
        simulate_arm_model(theta_i, sim_dir, ntrials, base_seed=rank_seed)

    # Optional barrier to ensure all ranks finish before job exits
    if USING_MPI:
        COMM.Barrier()

# ----------------- CLI / main -----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1, help='Start id of simulations.')
    parser.add_argument('--end', type=int, default=1000, help='End id of simulations.')
    parser.add_argument('--ntrials', type=int, default=25, help='Number of i.i.d. simulations per parameter set.')
    parser.add_argument('--output', type=str, default='all_data_gaus_pos', help='Output directory.')
    parser.add_argument('--seed', type=int, default=12345, help='Base RNG seed.')
    return parser.parse_args()

'''
def main():
    args = parse_args()
    output_dir = args.output

    # MPI-distributed execution
    gather_simulation_pairs_mpi(
        start_id=args.start,
        end_id=args.end,
        output_dir=output_dir,
        num_trials=args.ntrials,
        base_seed=args.seed
    )

    # NOTE: Removed the undefined 'results' write. All outputs live under:
    #   {output_dir}/simulation_outputs/sim{ID}/{trial}/results.pkl
'''

def main():
    # Quick visibility
    mpi_print(f"USING_MPI={USING_MPI} SIZE={SIZE} SLURM_NTASKS={os.environ.get('SLURM_NTASKS','-')}")

    
    # If SLURM says multi-task but mpi size==1, fail fast with instructions
    require_mpi_or_exit()

    args = parse_args()
    output_dir = args.output
    if is_mpi_active():
        mpi_print("MPI is active; running distributed path.")
        # call your MPI path, e.g. gather_simulation_pairs_mpi(...)
        gather_simulation_pairs_mpi(
            start_id=args.start,
            end_id=args.end,
            output_dir=output_dir,
            num_trials=args.ntrials,
            base_seed=args.seed,
        )
    else:
        mpi_print("MPI not active; running single-process fallback.")
        # Single-process path (rank=0 behavior)
        # e.g., sample locally and loop:
        prior = create_prior(...)
        # sample just for this process
        # run simulate_arm_model(...) in a simple for-loop

if __name__ == '__main__':
    main()