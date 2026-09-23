from pathlib import Path

# ==== change this path ====
ROOT = Path("/path/to/Gaussian_Normal/multi_selection_outputs")

EXPECTED_NUM_TRIALS = 25
REQUIRED_SIM_FILES = ["parameters.pkl"]
REQUIRED_TRIAL_FILES = ["results.pkl", "CNratios_all.pkl.gz", "sim.log"]


def is_trial_folder(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit()


def check_one_sim(sim_dir: Path):
    report = {
        "sim_name": sim_dir.name,
        "ok": True,
        "problems": []
    }

    # 1) check parameters.pkl in sim#
    for fname in REQUIRED_SIM_FILES:
        if not (sim_dir / fname).exists():
            report["ok"] = False
            report["problems"].append(f"Missing sim-level file: {fname}")

    # 2) check trial folders
    trial_folders = [p for p in sim_dir.iterdir() if is_trial_folder(p)]
    found_trials = sorted(int(p.name) for p in trial_folders)

    expected_trials = set(range(1, EXPECTED_NUM_TRIALS + 1))
    found_trials_set = set(found_trials)

    missing_trials = sorted(expected_trials - found_trials_set)
    extra_trials = sorted(found_trials_set - expected_trials)

    if missing_trials:
        report["ok"] = False
        report["problems"].append(f"Missing trial folders: {missing_trials}")

    if extra_trials:
        report["ok"] = False
        report["problems"].append(f"Unexpected extra trial folders: {extra_trials}")

    # 3) check files inside each expected existing trial folder
    for i in sorted(expected_trials & found_trials_set):
        trial_dir = sim_dir / str(i)
        missing_files = [f for f in REQUIRED_TRIAL_FILES if not (trial_dir / f).exists()]
        if missing_files:
            report["ok"] = False
            report["problems"].append(
                f"Trial {i} missing files: {missing_files}"
            )

    return report


def main():
    sim_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()], key=lambda x: x.name)

    all_reports = []
    failed_sims = []

    for sim_dir in sim_dirs:
        report = check_one_sim(sim_dir)
        all_reports.append(report)
        if not report["ok"]:
            failed_sims.append(report)

    print("=" * 70)
    print(f"Total sim folders checked: {len(sim_dirs)}")
    print(f"Successful simulations: {len(sim_dirs) - len(failed_sims)}")
    print(f"Failed simulations: {len(failed_sims)}")
    print("=" * 70)

    if failed_sims:
        print("\nFailed simulations:\n")
        for rep in failed_sims:
            print(f"{rep['sim_name']}:")
            for p in rep["problems"]:
                print(f"  - {p}")
            print()

    # save failed sim names only
    with open("failed_simulations.txt", "w") as f:
        for rep in failed_sims:
            f.write(rep["sim_name"] + "\n")

    # save full detailed report
    with open("failed_simulations_detailed.txt", "w") as f:
        for rep in failed_sims:
            f.write(f"{rep['sim_name']}\n")
            for p in rep["problems"]:
                f.write(f"  - {p}\n")
            f.write("\n")

    print("Saved:")
    print("  failed_simulations.txt")
    print("  failed_simulations_detailed.txt")


if __name__ == "__main__":
    main()