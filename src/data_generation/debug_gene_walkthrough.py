"""Step-by-step walkthrough of simulate_gene_selection.py on small inputs (learning / debugging only).

Each STEP calls one real function from the simulator on a tiny input and prints what came
back. Put a breakpoint on any line marked  # <-- BREAKPOINT  and step into (F11) the call to
watch the function work. Nothing here changes the simulator; it only imports it.

Run (from anywhere):
    PYTHONPATH=/Users/maryamkafi/Documents/PhD_Project/cancer/sistem-main \
        ~/miniconda3/envs/cancer/bin/python debug_gene_walkthrough.py
or use the "Gene sim walkthrough" configuration in .vscode/launch.json (F5).

STEPS 1-5 take well under a second. STEP 6 grows ONE small tumour (20,000 cells instead of
5 million) and STEP 7 runs the full per-theta writer with 2 trials of that size; both take
seconds. All files go to a throwaway folder printed at the start.
"""
import os
import sys
import types
import pickle
import tempfile

import numpy as np

# --- Keep SISTEM's ORIGINAL method before the simulator patches it, so STEP 1 can compare ---
from sistem.selection.region_library import BaseRegionLibrary
ORIGINAL_get_driver_start_regions = BaseRegionLibrary.get_driver_start_regions

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_gene_selection as s          # importing runs apply_sistem_patches()
from sistem import Parameters

OUT = tempfile.mkdtemp(prefix="gene_walkthrough_")
np.set_printoptions(linewidth=120, precision=3, suppress=True)


def banner(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


# ======================================================================================
banner("STEP 0  the import patched SISTEM in memory (SISTEM's files are untouched)")
print("patched method is ours:", BaseRegionLibrary.get_driver_start_regions is s._fixed_get_driver_start_regions)
print("recorded patches      :", s.SISTEM_PATCHES)
print("output folder         :", OUT)

# ======================================================================================
banner("STEP 1  _fixed_get_driver_start_regions on a toy chromosome")
# A fake library and a fake chromosome: only the attributes the function reads.
#   seq = block ids along ONE homolog; drivers = block ids with delta != 0
toy_lib = types.SimpleNamespace(drivers={"chrT": [1, 5]}, essential={"chrT": []})
toy_chr = types.SimpleNamespace(name="chrT", seq=[0, 1, 2, 3, 4, 5])      # 6 blocks, last id = 5
print("seq =", toy_chr.seq, "  drivers =", toy_lib.drivers["chrT"])
for size in (1, 2, 3, 6):
    fixed = sorted(s._fixed_get_driver_start_regions(toy_lib, None, toy_chr, size))     # <-- BREAKPOINT
    orig = sorted(ORIGINAL_get_driver_start_regions(toy_lib, None, toy_chr, size))
    windows = [toy_chr.seq[j:j + size] for j in fixed]
    print(f"size={size}: fixed starts {fixed}  (windows {windows})")
    print(f"        SISTEM original {orig}   <- never reaches driver 5 at the last position")

# Same idea after a deletion and an amplification: ids no longer equal positions.
toy_chr2 = types.SimpleNamespace(name="chrT", seq=[0, 1, 1, 1, 4, 5])     # block 1 x3, blocks 2-3 lost
print("\nafter events: seq =", toy_chr2.seq)
for size in (1, 2):
    print(f"size={size}: starts {sorted(s._fixed_get_driver_start_regions(toy_lib, None, toy_chr2, size))}")

# ======================================================================================
banner("STEP 2  genome_grid(): the 5 Mb block grid")
chrom_lens, arm_ratios, nregions = s.genome_grid()                         # <-- BREAKPOINT
D = sum(nregions.values())
print("blocks per chromosome:", {c: nregions[c] for c in s.CHROM_ORDER})
print("D (total blocks)     :", D)
print("chr1 length (bp)     :", chrom_lens["chr1"], " arm ratio:", arm_ratios["chr1"],
      " -> centromere at block", round(nregions["chr1"] * arm_ratios["chr1"]))

# ======================================================================================
banner("STEP 3  sample_theta(): one draw from the prior")
base_seed, sim_id = s.DEFAULT_SEED, 1
prior_seed = base_seed + sim_id * s.PRIOR_SEED_STRIDE                      # same formula as main()
rng = np.random.default_rng(prior_seed)
theta = s.sample_theta(rng, nregions, s.CHROM_ORDER)                       # <-- BREAKPOINT
print(f"prior_seed={prior_seed}")
for k in ("arm_rate", "chromosomal_rate", "focal_driver_rate", "length_mean", "pi", "q", "m"):
    print(f"  {k:16s} = {theta[k]:.6g}")
nz = [(c, r, d) for c in s.CHROM_ORDER for r, d in enumerate(theta["delta_dict"][c]) if d != 0]
print(f"  driver blocks    = {len(nz)} of {D}  (expected about pi*D = {theta['pi'] * D:.1f})")
print("  first 8 drivers (chrom, block, delta):", [(c, r, round(d, 3)) for c, r, d in nz[:8]])
print("  chr1 deltas:", np.array(theta["delta_dict"]["chr1"]))
print("  same seed -> same theta:",
      s.sample_theta(np.random.default_rng(prior_seed), nregions, s.CHROM_ORDER)["delta_dict"] == theta["delta_dict"])

# ======================================================================================
banner("STEP 4  driver_layout(): flatten to the length-D vectors the network sees")
driver_mask, driver_index = s.driver_layout(theta["delta_dict"], s.CHROM_ORDER)   # <-- BREAKPOINT
print("driver_mask: shape", driver_mask.shape, " number True =", int(driver_mask.sum()))
print("driver_index[:5]:", driver_index[:5])
offset = {c: sum(nregions[x] for x in s.CHROM_ORDER[:i]) for i, c in enumerate(s.CHROM_ORDER)}
c0, r0 = driver_index[0]
print(f"flat column of ({c0}, block {r0}) = offset[{c0}] + {r0} = {offset[c0] + r0}  -> mask there:",
      driver_mask[offset[c0] + r0])

# ======================================================================================
banner("STEP 5  build_library(): SISTEM's FittedRegionLibrary for this theta")
lib = s.build_library(theta["delta_dict"], chrom_lens, arm_ratios, out_dir=OUT)    # <-- BREAKPOINT
print("type:", type(lib).__name__)
print("drivers on chr1 (block ids):", lib.drivers["chr1"])
print("n_drivers:", sum(len(v) for v in lib.drivers.values()),
      " viability cap max_distinct_driv = int(0.9 * n_drivers) =", lib.max_distinct_driv)
print(f"base_fit (fitness of a normal diploid cell) = {lib.base_fit:.4f}")
print("limits: max_region_CN", lib.max_region_CN, " ploidy in [", lib.min_ploidy, ",", lib.max_ploidy, "]")

# ======================================================================================
banner("STEP 6  simulate_gene_model_caller(): grow ONE small tumour and summarise its clones")
trial_dir = os.path.join(OUT, "step6")
os.makedirs(trial_dir, exist_ok=True)
np.random.seed(prior_seed + 1)                       # SISTEM uses numpy's global RNG
params = Parameters(
    region_len=s.REGION_LEN,
    focal_driver_rate=theta["focal_driver_rate"],   # drawn per theta since prior v2
    length_mean=theta["length_mean"],
    max_distinct_driv_ratio=s.MAX_DISTINCT_DRIV_RATIO,
    min_detectable=20_000,                           # tiny tumour for learning (production: 5e6)
    arm_rate=theta["arm_rate"],
    chromosomal_rate=theta["chromosomal_rate"],
    out_dir=trial_dir,
)
summary = s.simulate_gene_model_caller(params, theta["delta_dict"], chrom_lens, arm_ratios)   # <-- BREAKPOINT
if not isinstance(summary, dict):
    print("attempt failed:", {-1: "SISTEM raised", -2: "tumour went extinct"}[summary], "- rerun with another seed")
else:
    bc, pop = summary["block_cn"], summary["popsize"]
    print("block_cn shape (clones x blocks):", bc.shape, bc.dtype)
    order = np.argsort(-pop)
    print("clone  cells   ploidy  nWGD  altered_drivers  fitness")
    for i in order[:5]:
        print(f"{i:5d} {pop[i]:7d}   {summary['ploidy'][i]:.2f}   {summary['nWGD'][i]:3d}  "
              f"{summary['n_altered_drivers'][i]:10d}       {summary['fitness'][i]:.4f}")
    big = order[0]
    changed = np.flatnonzero(bc[big] != 2)
    print(f"largest clone: {len(changed)} blocks differ from 2 copies;",
          f"{int(driver_mask[changed].sum())} of them are driver blocks")
    print("bulk (cell-weighted mean CN) on the first 10 driver columns:",
          summary["bulk"][np.flatnonzero(driver_mask)[:10]])
    print("driver_count_mismatches (must be 0):", summary["driver_count_mismatches"])

# ======================================================================================
banner("STEP 7  simulate_gene_model(): the per-theta writer (2 small trials) and its files")
s.MIN_DETECTABLE = 20_000                            # shrink the tumour for this demo only
sim_dir = os.path.join(OUT, "simulation_outputs", f"sim{sim_id}")
s.simulate_gene_model(theta, sim_dir, num_trials=2, prior_seed=prior_seed, sim_id=sim_id)   # <-- BREAKPOINT
for root, _, files in sorted(os.walk(sim_dir)):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, OUT):45s} {os.path.getsize(p):>10,d} bytes")
with open(os.path.join(sim_dir, "parameters.pkl"), "rb") as f:
    P = pickle.load(f)
print("parameters.pkl keys:", sorted(P))
with open(os.path.join(sim_dir, "1", "results.pkl"), "rb") as f:
    print("trial 1 results.pkl:", pickle.load(f))
if os.path.exists(os.path.join(sim_dir, "1", "clones.npz")):
    z = np.load(os.path.join(sim_dir, "1", "clones.npz"))
    print("trial 1 clones.npz:", {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
print("\nRun it again: completed trials are skipped (resumability):")
s.simulate_gene_model(theta, sim_dir, num_trials=2, prior_seed=prior_seed, sim_id=sim_id)
print(f"\nDone. Files left in {OUT} (safe to delete).")
