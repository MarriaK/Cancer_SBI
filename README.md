# Selection coefficients from clonal copy-number data

This directory holds the code for a simulation-based inference (SBI) study: given the copy-number
profiles of the clones inside a simulated tumour, infer the 44 chromosome-arm selection coefficients
(22 chromosomes x p/q arms) that produced it. Tumours are generated with a forward clonal-evolution
simulator, and a neural posterior estimator (NPE) -- a normalising flow conditioned on a learned
embedding of the tumour's clone set -- is trained on those simulations to approximate the posterior
over the 44 coefficients. Three embeddings are compared; the flow and the training loop are the same
in all three, so the comparison is about how the clone set is summarised before the flow sees it.

## The three models

| Name in the paper | Folder in this repo  | Encoder, in one line                                              |
| ----------------- | -------------------- | ----------------------------------------------------------------- |
| CloneMLP-NPE      | `Base_NPE/`          | per-clone MLP over the 100 most frequent arm profiles, then pooled |
| CloneAtt-NPE      | `SetTransformer_NPE/`| set transformer (ISAB attention) over the same clone set           |
| DominantClone-NPE | `Plain_NPE/`         | the largest clone only (44 values), small feedforward net          |

The folder names are the old working names and were deliberately **not** renamed. The scripts import
their neighbours by bare name and reach each other by relative path (for example `Plain_NPE/main.py`
opens `../Base_NPE/train_test_split.pkl`), and the same three folders exist under the same names on
the HPC cluster. Renaming them here would break both. If the folder names ever do change, they must
change on the cluster at the same time.

## Top level

```
README.md                 this file
docs/                     DATA_GENERATION.md (how the simulated data was made)
                          REORGANIZATION.md, reorg_manifest.tsv, undo_reorg.sh (the 2026-09-21 tidy-up)
data_generation/          the simulators that produced Guassian_Normal/ + a completeness checker
Guassian_Normal/          the simulated dataset: simulation_outputs/sim<N>/<trial>/...  (888 sims, ~3.6 GB)
Base_NPE/                 CloneMLP-NPE      (train, evaluate, figures)
SetTransformer_NPE/       CloneAtt-NPE      (train, evaluate, figures)
Plain_NPE/                DominantClone-NPE (train, evaluate, figures)
baselines/                ridge regression on four hand-made summary statistics per arm
overview_figure/          scripts that draw the paper's overview figure, panel by panel
utilities/                small helpers shared across models (loss curves, chromosome-arm lookup)
logs/                     SLURM .out/.err from the cluster runs, per model
archive/                  older experiments and superseded copies; nothing here is on the live path
```

Inside each model folder the layout is the same: the scripts sit at the top level of the folder
(they import each other by bare name, so they have to), and `figures/` holds what those scripts drew.

## Where things are

- **Simulated data**: `Guassian_Normal/simulation_outputs/`. One directory per simulation
  (`sim1`, `sim2`, ...), each with `parameters.pkl` (the true 44 coefficients, after two leading
  entries) and 25 numbered replicate directories holding `CNratios_all.pkl.gz`, `results.pkl` and
  `sim.log`. It is large and is listed in `.gitignore`; it is not in version control.
- **Train/test split**: `Base_NPE/train_test_split.pkl`. All three models read this same file, so the
  split is identical across models. `Plain_NPE` and the baseline reach it by relative path.
- **Pickled posteriors**: `SetTransformer_NPE/SetTransformer_NPE_Freq_mean.pkl` is written by
  `SetTransformer_NPE/inference_model.py` (line 246), and `Base_NPE/SetTransformer_NPE_Freq_mean.pkl`
  is a byte-identical copy of it. Both have to stay where they are: the scripts open the file **by
  bare name**, so each reads the copy in its own folder and cannot see the other. In `Base_NPE/` the
  readers are `ppc-plot.py` (line 82), `PriorPredictiveCheck.ipynb` (cell 2) and `test.ipynb`
  (cell 5). Identical content is not a reason to move one of them away; see
  `docs/REORGANIZATION.md`.
- **Trained checkpoints**: not here. Training runs on the HPC cluster and the checkpoints
  (`<model>/checkpoints/latest.pt`) live there. That is why `utilities/plot_losses.py` and
  `overview_figure/export_posterior.py` are written to run on the cluster, and why the overview
  figure falls back to "context only" panels when no posterior export is present locally.
- **Results**: `<model>/figures/` (z-score summaries, recovery scatter plots, correlation
  spreadsheets). `baselines/summary_stat_baseline.csv` holds the summary-statistic reference scores.
- **Logs**: `logs/<model>/encoder/`, the raw SLURM output of the cluster jobs.
- **Docs**: `docs/DATA_GENERATION.md` explains how the simulated dataset was produced;
  `docs/REORGANIZATION.md` records what this reorganisation moved and how to undo it.

## The relative-path contract

Scripts are written to be run **from inside their own model folder**, not from here:

```
cd Base_NPE
python main.py
```

From there, two relative paths must resolve:

- `../Guassian_Normal/simulation_outputs` -- the simulated data (this is why the dataset sits at the
  top level under exactly that name, and why the three model folders must stay siblings of it);
- `../Base_NPE/train_test_split.pkl` -- the shared split, which `Plain_NPE/main.py` opens by name.

`baselines/summary_stat_baseline.py` and the `overview_figure/` scripts are the exception: they work
out the repository root from their own location, so they can be run from their own folder without
arguments. `overview_figure/export_posterior.py` keeps HPC-relative paths and is meant to be run from
`Base_NPE/` on the cluster.

## Python environment

Everything runs in the conda environment `cancer`:

```
~/miniconda3/envs/cancer/bin/python <script>.py
```

NumPy 2 or newer is required: the pickles in `Guassian_Normal/` were written with it and the system
Python cannot read them. Training additionally needs `torch` and `sbi`; the simulators need `sistem`
(and `mpi4py` for the MPI variant).
