import sys
#sys.path.insert(0,'../MetastasisSimulator/source')
import sistem
import numpy as np
import os
import copy
import pickle as pkl
from itertools import repeat
import argparse
import multiprocessing as mp
from torch.distributions import Laplace
import gzip
import contextlib
import io
from sistem.genome.genome import init_diploid_genome
from sistem.genome.utils import get_num_regions
from sistem.utilities.utilities import compute_growth_rate 
from sistem.genome.utils import hg38_chrom_lengths_from_cytoband
from sistem.lineage.cell import Clone
from sistem.ancestry import GrowthSimulator
from sistem.anatomy import BaseAnatomy, SimpleAnatomy
from sistem import Parameters
from sistem.data.summarize import extract_largest_clones, arm_CNratios_single, arm_CNratios_bulk
import torch
from torch.distributions import Independent, Uniform, Normal, Beta

import sistem.selection.arm_library as lib
from sbi.inference import NPE
import multiprocessing
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


#from main import setup_logger
# lower and upper bounds for chromosome-arm mutation rate
arm_low, arm_high = 1e-5, 1e-4

# lower and upper bounds for whole-chromosome mutation rate
chrom_low, chrom_high = 1e-6, 1e-5

coeff_mu, coeff_var = 0, 0.2


def create_prior(arm_low, arm_high, chrom_low, chrom_high, coeff_mu, coeff_var, device):
    priors = []

    # 1) Two mutation rate priors: arm_rate, chrom_rate
    priors.append(
        Uniform(
            torch.tensor([arm_low], dtype=torch.float32, device=device),
            torch.tensor([arm_high], dtype=torch.float32, device=device),
            validate_args=False,
        )
    )
    priors.append(
        Uniform(
            torch.tensor([chrom_low], dtype=torch.float32, device=device),
            torch.tensor([chrom_high], dtype=torch.float32, device=device),
            validate_args=False,
        )
    )

    # 2) 44 selection coefficient priors: independent Normals
    for _ in range(44):
        priors.append(
            Normal(
                torch.tensor([coeff_mu], dtype=torch.float32, device=device),
                torch.tensor([coeff_var], dtype=torch.float32, device=device),
                validate_args=False,
            )
        )

    # Let sbi process the list -> MultipleIndependent with 46 parameters
    prior, _, _ = process_prior(priors)
    return prior

# ----------------- Simulation -----------------
def simulate_arm_model(theta, sim_id, t, arm_deltas, base_output_dir, base_seed=None):
    out_dir = os.path.join(base_output_dir, f"sim{sim_id}")
    trial_dir = os.path.join(out_dir, str(t))
    os.makedirs(trial_dir, exist_ok=True)

    if base_seed is not None:
        seed = base_seed + sim_id * 10000 + t
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))

    arm_rate = float(theta[0])
    chromosomal_rate = float(theta[1])

    params = Parameters(
        region_len=5e6,
        focal_driver_rate=5e-4,
        max_distinct_driv_ratio=0.9,
        min_detectable=5e6,
        arm_rate=arm_rate,
        chromosomal_rate=chromosomal_rate,
        out_dir=trial_dir,
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

    if complete:
        CNratios_all = np.asarray(CNratios_all, dtype=np.float32)
        with gzip.open(os.path.join(trial_dir, "CNratios_all.pkl.gz"), "wb") as g:
            pkl.dump(CNratios_all, g, protocol=5)
    # --- END OF PROCESSING FOR ONE TRIAL ---
    print(f"[DONE] Simulation {sim_id}, trial {t}", flush=True)


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
        with contextlib.redirect_stdout(io.StringIO()):
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



def gather_simulation_pairs(start_id, end_id, prior, output_dir, num_trials, num_processors=1, base_seed=None):
    data_dir = os.path.join(output_dir, 'simulation_outputs')
    os.makedirs(data_dir, exist_ok=True)

    sim_ids = list(range(start_id, end_id + 1))
    num_sims = len(sim_ids)

    theta = prior.sample((num_sims,))

    # Precompute arm_deltas per sim (as you do in simulate_arm_model)
    arm_deltas_list = []
    for row in theta:
        arm_deltas = {}
        for i in range(0, 44, 2):
            arm_deltas[f'chr{int(i/2) + 1}'] = [float(row[i+2]), float(row[i+3])]
        arm_deltas_list.append(arm_deltas)

    # Build all tasks: (theta, sim_id, t, arm_deltas, base_output_dir, base_seed)
    tasks = []
    for idx, sim_id in enumerate(sim_ids):
        for t in range(1, num_trials + 1):
            tasks.append(
                (theta[idx], sim_id, t, arm_deltas_list[idx], data_dir, base_seed)
            )

    if num_processors > 1:
        with multiprocessing.Pool(processes=num_processors) as pool:
            pool.starmap(simulate_arm_model, tasks)
    else:
        for args in tasks:
            simulate_arm_model(*args)

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1, help='Start id of simulations.')
    parser.add_argument('--end', type=int, default=10,  help='End id of simulations.')
    parser.add_argument('--ntrials', type=int, default=1, help='Number of i.i.d. simulations to generate per parameter set.')
    parser.add_argument('--output', type=str, default='data',  help='Output directory.')
    parser.add_argument('-n', type=int, default=1, help='Number of CPUs.')
    arguments = parser.parse_args()
    return arguments

def main():
    #args = parse_args()
    output_dir = 'Guassian_Normal'
    
    #output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    #with open('prior.pkl', 'rb') as f:
    #    prior = pkl.load(f)
    prior = create_prior(arm_low, arm_high, chrom_low, chrom_high, coeff_mu, coeff_var, device)
    gather_simulation_pairs(1501, 1620, prior, output_dir, 25, num_processors=os.cpu_count(), base_seed = 12345)
    #with open(os.path.join(output_dir, f'{1}-{10}_results.pkl'), 'wb') as f:
    #    pkl.dump(results, f)


if __name__ == '__main__':
    #mp.set_start_method('spawn', force=True)
    main()
