# `jobs/` — the SLURM scripts

`/home` on the cluster is `noexec`, so nothing runs outside SLURM. Every script here activates
`cancer-sbi`, puts the conda `lib` first on `LD_LIBRARY_PATH` (otherwise scipy dies with a
`GLIBCXX_3.4.30` ImportError), writes its logs to absolute paths under
`/home/mak23055/cancer/logs/`, and `cd`s to `$HOME/cancer/src`. Submit them from there:
`sbatch jobs/<script>.sh`.

Cluster tree: `~/cancer/{src,data,runs,results,cache,logs}` — there is no `code/` level.

| script | what it does | env overrides |
| --- | --- | --- |
| `train.sh` | The four-run repair matrix R0/R1/R2/R4 as a `--array=0-3` job. Refuses to start if the run's checkpoint directory is non-empty, or if the split carries no `val_ids`. 12 h, a100. | `RUNS_ROOT`, `SEED`, `NUM_WORKERS`, `CACHE_DIR`, `ALLOW_TEST_AS_VAL`, `CANCER_SBI_DATA_ROOT`, `CANCER_SBI_SPLIT`, `DRY_RUN` |
| `sample.sh` | Stage 1: 5000 posterior draws per held-out tumour, one array task per model. 12 h, a100. | `CKPT` (the checkpoint to sample — **always pass it for a matrix run**), `POST` (output dir), `RUN_TAG`, `EXTRA` (extra flags, last-wins), `DRY_RUN` |
| `analyze.sh` | Stage 2: `poster_metrics` — every metric, table and figure, from stage 1's `.npz`. CPU, minutes. | `POST` (input dir), `OUT` (results dir), `DRY_RUN` |
| `shrink.sh` | Figure D (`fig_shrinkage`), screen and poster builds. CPU. | `POST`, `OUT`, `DRY_RUN` |
| `treetest.sh` | Smoke test: all three models sample 4 cases from the reorganised tree. | — |
| `finaltest.sh` | Smoke test: sampling *and* metrics end to end into `~/cancer/_final`. | — |
| `verify.sh` | `src/verify_refactor.py --with-models` against the archived legacy tree. Expect 42/42. | — |

`DRY_RUN=1 bash jobs/<script>.sh` prints the command(s) it would run and exits. It needs no conda
env, so it works on the laptop.

`CACHE_DIR` is worth its own line: the default `sbatch` runs **uncached**, straight off the gzipped
trial files at ~28 min/epoch, and `CACHE_DIR=$CANCER/data/cache/clone_top100_v1` points every run at
the pre-built clone cache instead, which is roughly **13x faster per epoch**.

## Carve the validation split before the first sbatch

`train.sh` refuses a split with no `val_ids`, because without one `cli/train.py` falls back to early
stopping on the **test** set and each run's `best.pt` is chosen on the very sims its score is
reported on. The cluster's `train_test_split.pkl` is the 2,880/720 two-key file, so carve the third
set once, before anything is submitted:

```bash
python -m cancer_sbi.cli.make_split --add-val --frac 0.1 --seed 20260924 \
  --in  $HOME/cancer/data/train_test_split.pkl \
  --out $HOME/cancer/data/train_val_test_split.pkl
```

`test_ids` is copied through untouched, so every reported number stays on the same held-out cases as
the published ones; only `train_ids` shrinks. `train.sh` defaults `CANCER_SBI_SPLIT` to that
three-key file. `ALLOW_TEST_AS_VAL=1` waives the check, deliberately and visibly.

## The two hazards these overrides exist for

**The silent old checkpoint.** Without `--ckpt`, `sample_posteriors.py` defaults to
`$CANCER_SBI_RUNS/<model>/checkpoints/best.pt` — the *old published* run. The job succeeds and every
number is the one you already had. Pass `CKPT=` for anything from the 2026-09-24 matrix.

**The filename collision.** R0, R1 and R2 are all `clonemlp`, so they write the same
`posteriors_clonemlp.npz`. Give each run its own `POST` directory
(`~/cancer/results/2026-09-24/<run>/posteriors`); `poster_metrics.py` discovers only the plain
filename, so a per-run directory — not a suffix — is the convention. `--limit N` and `--run-tag TAG`
do add a suffix (`posteriors_clonemlp_limit4.npz`), so a smoke test can never overwrite a real run.

## One run, end to end

```bash
RUN=R1
sbatch --array=1 jobs/train.sh

CKPT=$HOME/cancer/runs/2026-09-24/$RUN/checkpoints/best.pt \
POST=$HOME/cancer/results/2026-09-24/$RUN/posteriors \
  sbatch --array=0 jobs/sample.sh

POST=$HOME/cancer/results/2026-09-24/$RUN/posteriors \
OUT=$HOME/cancer/results/2026-09-24/$RUN sbatch jobs/analyze.sh

POST=$HOME/cancer/results/2026-09-24/$RUN/posteriors \
OUT=$HOME/cancer/results/2026-09-24/$RUN sbatch jobs/shrink.sh
```
