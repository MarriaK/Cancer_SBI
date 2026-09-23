import os
import copy
import io
import gzip
import shutil
import pickle as pkl
import contextlib
import argparse
import multiprocessing

import numpy as np
import torch
from torch.distributions import Uniform, Normal

import sistem
from sistem import Parameters
from sistem.genome.genome import init_diploid_genome
from sistem.genome.utils import get_num_regions, hg38_chrom_lengths_from_cytoband
from sistem.lineage.cell import Clone
from sistem.ancestry import GrowthSimulator
from sistem.anatomy import SimpleAnatomy
from sistem.data.summarize import (
    extract_largest_clones,
    arm_CNratios_single,
    arm_CNratios_bulk,
)
import sistem.selection.arm_library as lib

from sbi.utils.user_input_checks import process_prior


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Prior settings
arm_low, arm_high = 1e-5, 1e-4
chrom_low, chrom_high = 1e-6, 1e-5
coeff_mu, coeff_var = 0, 0.2


def create_prior(arm_low, arm_high, chrom_low, chrom_high, coeff_mu, coeff_var, device):
    priors = []

    # arm mutation rate
    priors.append(
        Uniform(
            torch.tensor([arm_low], dtype=torch.float32, device=device),
            torch.tensor([arm_high], dtype=torch.float32, device=device),
            validate_args=False,
        )
    )

    # whole chromosome mutation rate
    priors.append(
        Uniform(
            torch.tensor([chrom_low], dtype=torch.float32, device=device),
            torch.tensor([chrom_high], dtype=torch.float32, device=device),
            validate_args=False,
        )
    )

    # 44 selection coefficients
    for _ in range(44):
        priors.append(
            Normal(
                torch.tensor([coeff_mu], dtype=torch.float32, device=device),
                torch.tensor([coeff_var], dtype=torch.float32, device=device),
                validate_args=False,
            )
        )

    prior, _, _ = process_prior(priors)
    return prior


def theta_to_arm_deltas(theta):
    arm_deltas = {}
    for i in range(0, 44, 2):
        arm_deltas[f"chr{int(i/2) + 1}"] = [float(theta[i + 2]), float(theta[i + 3])]
    return arm_deltas


def save_sim_parameters(sim_dir, theta):
    os.makedirs(sim_dir, exist_ok=True)
    theta_np = np.asarray(theta, dtype=np.float32)
    with open(os.path.join(sim_dir, "parameters.pkl"), "wb") as f:
        pkl.dump(theta_np, f)


def load_or_sample_theta(sim_dir, prior):
    """
    If parameters.pkl exists, load it.
    Otherwise sample a new theta from prior, save it, and return it.
    """
    param_path = os.path.join(sim_dir, "parameters.pkl")

    if os.path.exists(param_path):
        with open(param_path, "rb") as f:
            theta = pkl.load(f)
        theta = np.asarray(theta, dtype=np.float32)
        return theta, "loaded"

    theta = prior.sample((1,)).squeeze(0).detach().cpu().numpy().astype(np.float32)
    save_sim_parameters(sim_dir, theta)
    return theta, "sampled"


def trial_is_complete(trial_dir):
    """
    A trial is considered complete if these files exist.
    Change this list if needed.
    """
    required_files = ["results.pkl", "CNratios_all.pkl.gz", "sim.log"]

    if not os.path.isdir(trial_dir):
        return False

    for fname in required_files:
        if not os.path.exists(os.path.join(trial_dir, fname)):
            return False

    return True


def find_missing_trials(sim_dir, expected_num_trials=25):
    missing_trials = []
    for t in range(1, expected_num_trials + 1):
        trial_dir = os.path.join(sim_dir, str(t))
        if not trial_is_complete(trial_dir):
            missing_trials.append(t)
    return missing_trials


def read_failed_sim_list(txt_path):
    """
    Reads failed_simulations.txt containing lines like:
      sim1501
      sim1502
    or
      1501
      1502
    """
    sim_ids = []

    with open(txt_path, "r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue

            if name.startswith("sim"):
                sim_id = int(name.replace("sim", ""))
            else:
                sim_id = int(name)

            sim_ids.append(sim_id)

    return sim_ids


def simulate_arm_model_caller(params, arm_deltas):
    chrom_lens, arm_ratios = hg38_chrom_lengths_from_cytoband(include_allosomes=True)
    _ = get_num_regions(chrom_lens, params.region_len)

    library = lib.FittedArmLibrary(
        arm_ratios=arm_ratios,
        max_distinct_driv_ratio=params.max_distinct_driv_ratio,
    )
    library.initialize(delta=arm_deltas)

    if not hasattr(library, "arm_sizes"):
        library.arm_sizes = library._arm_sizes

    anatomy = SimpleAnatomy(
        library,
        nsites=params.nsites,
        growth_rate=params.growth_rate,
        capacities=params.capacities,
    )
    anatomy.init_pop_dynamic(0, 0, params.t_max, N0=params.N0)

    igenome = init_diploid_genome(library)
    _ = Clone(
        genome=copy.deepcopy(igenome),
        library=library,
        popsize=params.N0,
        site=0,
        birth_gen=0,
    )

    gs = GrowthSimulator(anatomy)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            gs.simulate_agents(params=params)
    except Exception as e:
        return -1

    if len(gs.clones[0]) == 0:
        return -2

    normalize = lambda x: np.log2(max(x, 0.001)) - 1
    chrom_names = [f"chr{i}" for i in range(1, 23)]

    largest_clone = extract_largest_clones(gs)[0]
    CNratios_largest = arm_CNratios_single(largest_clone)
    CNratios_largest_flat = [
        normalize(val)
        for chrname in chrom_names
        for val in CNratios_largest[chrname]
    ]

    CNratios_bulk = arm_CNratios_bulk(gs.clones[0])
    CNratios_bulk_flat = [
        normalize(val)
        for chrname in chrom_names
        for val in CNratios_bulk[chrname]
    ]

    CNratios_all = []
    for clone in gs.clones[0]:
        CNratios = arm_CNratios_single(clone)
        CNratios_flat = [
            normalize(val)
            for chrname in chrom_names
            for val in CNratios[chrname]
        ]
        CNratios_all.append(CNratios_flat)

    return CNratios_largest_flat, CNratios_bulk_flat, CNratios_all


def simulate_arm_model(theta, sim_id, t, arm_deltas, base_output_dir, base_seed=None, clean_incomplete=True):
    out_dir = os.path.join(base_output_dir, f"sim{sim_id}")
    trial_dir = os.path.join(out_dir, str(t))

    # Remove incomplete trial folder before rerun
    if clean_incomplete and os.path.isdir(trial_dir):
        shutil.rmtree(trial_dir)

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
    CNratios_all = None

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

    with open(os.path.join(trial_dir, "results.pkl"), "wb") as f:
        if complete:
            pkl.dump(([failed_sim, failed_call], CNratios_largest, CNratios_bulk), f)
        else:
            pkl.dump([failed_sim, failed_call], f)

    if complete:
        CNratios_all = np.asarray(CNratios_all, dtype=np.float32)
        with gzip.open(os.path.join(trial_dir, "CNratios_all.pkl.gz"), "wb") as g:
            pkl.dump(CNratios_all, g, protocol=5)

        # Write your own completion log so the checker is reliable
        with open(os.path.join(trial_dir, "sim.log"), "w") as f:
            f.write("Simulation completed successfully.\n")

    print(f"[DONE] Simulation sim{sim_id}, trial {t}", flush=True)


def build_tasks_from_failed_list(failed_txt, prior, output_dir, num_trials):
    """
    For each sim listed in failed_simulations.txt:
      - load parameters.pkl if it exists
      - otherwise sample new theta from the same prior and save it
      - rerun only incomplete trials
    """
    data_dir = os.path.join(output_dir, "simulation_outputs")
    os.makedirs(data_dir, exist_ok=True)

    sim_ids = read_failed_sim_list(failed_txt)
    tasks = []

    for sim_id in sim_ids:
        sim_dir = os.path.join(data_dir, f"sim{sim_id}")

        theta, source = load_or_sample_theta(sim_dir, prior)
        arm_deltas = theta_to_arm_deltas(theta)

        missing_trials = find_missing_trials(sim_dir, expected_num_trials=num_trials)

        if len(missing_trials) == 0:
            print(f"[SKIP] sim{sim_id} already complete ({source} parameters)")
            continue

        print(f"[QUEUE] sim{sim_id}: {source} parameters, trials to run = {missing_trials}")

        for t in missing_trials:
            tasks.append((theta, sim_id, t, arm_deltas, data_dir))

    return tasks


def run_tasks(tasks, num_processors=1, base_seed=None):
    if len(tasks) == 0:
        print("[INFO] No tasks to run.")
        return

    full_tasks = [
        (theta, sim_id, t, arm_deltas, data_dir, base_seed)
        for (theta, sim_id, t, arm_deltas, data_dir) in tasks
    ]

    print(f"[INFO] Running {len(full_tasks)} trial tasks with {num_processors} process(es).")

    if num_processors > 1:
        with multiprocessing.Pool(processes=num_processors) as pool:
            pool.starmap(simulate_arm_model, full_tasks)
    else:
        for task in full_tasks:
            simulate_arm_model(*task)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--failed_list",
        type=str,
        required=True,
        help="SetTransformer_NPE/failed_simulations.txt",
    )
    parser.add_argument(
        "--ntrials",
        type=int,
        default=25,
        help="Expected number of trials per sim",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Gaussian_Normal",
        help="Base output directory",
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=12345,
        help="Base random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    prior = create_prior(
        arm_low=arm_low,
        arm_high=arm_high,
        chrom_low=chrom_low,
        chrom_high=chrom_high,
        coeff_mu=coeff_mu,
        coeff_var=coeff_var,
        device=device,
    )

    tasks = build_tasks_from_failed_list(
        failed_txt=args.failed_list,
        prior=prior,
        output_dir=args.output,
        num_trials=args.ntrials,
    )

    print(f"\nTotal trial tasks to run: {len(tasks)}\n")

    run_tasks(
        tasks=tasks,
        num_processors=args.nproc,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    # On some systems, especially with PyTorch / macOS, spawn is safer.
    multiprocessing.set_start_method("spawn", force=True)
    main()