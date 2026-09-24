# `jobs/` — the SLURM scripts

`/home` on the cluster is `noexec`, so nothing runs outside SLURM. Every script here activates
`cancer-sbi`, puts the conda `lib` first on `LD_LIBRARY_PATH` (otherwise scipy dies with a
`GLIBCXX_3.4.30` ImportError), writes its logs to absolute paths under
`/home/mak23055/cancer/logs/`, and `cd`s to `$HOME/cancer/src`. Submit them from there:
`sbatch jobs/<script>.sh`.

Cluster tree: `~/cancer/{src,data,runs,results,cache,logs}` — there is no `code/` level.

| script | what it does | env overrides |
| --- | --- | --- |
| `train.sh` | The five-run matrix R0/R1/R2/R4/D0 as a `--array=0-4` job. Refuses to start if the run's checkpoint directory is non-empty, or if the split carries no `val_ids`. 12 h, a100. | `RUNS_ROOT`, `SEED`, `NUM_WORKERS`, `CACHE_DIR`, `ALLOW_TEST_AS_VAL`, `CANCER_SBI_DATA_ROOT`, `CANCER_SBI_SPLIT`, `DRY_RUN` |
| `train2.sh` | The seven-run matrix 2 (R3/R2s1/R2s2/R5/R6/R7/R8) as a `--array=0-6` job. Same split gate, same non-empty-checkpoint refusal, same 12 h a100 as `train.sh`. | `RUNS_ROOT`, `SEED`, `NUM_WORKERS`, `CACHE_DIR`, `ALLOW_TEST_AS_VAL`, `CANCER_SBI_DATA_ROOT`, `CANCER_SBI_SPLIT`, `DRY_RUN` |
| `train3.sh` | The fourteen-run matrix 3 (R3s1/R3s2/R6s1/R6s2/D0s1/D0s2/R9–R16) as a `--array=0-13` job. Adds `--flow-dropout`, `--flow-num-transforms`, `--trial-subsample` and `--freq-mode feature` for clonemlp. Same split gate, same non-empty-checkpoint refusal, same 12 h a100. | `RUNS_ROOT`, `SEED`, `NUM_WORKERS`, `CACHE_DIR`, `ALLOW_TEST_AS_VAL`, `CANCER_SBI_DATA_ROOT`, `CANCER_SBI_SPLIT`, `DRY_RUN` |
| `sample.sh` | Stage 1: 5000 posterior draws per held-out tumour, one array task per model. 12 h, a100. For one run: `--array=<i>` or `MODEL=`; `CKPT` with the full 0-2 array is refused. | `MODEL`, `CKPT` (the checkpoint to sample — **always pass it for a matrix run**), `POST` (output dir), `RUN_TAG`, `EXTRA` (extra flags, last-wins), `DRY_RUN` |
| `ensemble.sh` | Stage 1b: pools several seeds' stage-1 `.npz` files into one equal-weight mixture posterior, `posteriors_<MODEL>_<RUN_TAG>.npz`, which `analyze.sh` then reads unchanged. CPU, minutes. | `INPUTS` (space-separated `.npz` paths, required), `OUT` (output dir), `MODEL`, `RUN_TAG`, `NUM_SAMPLES` (subsample, equal share per member), `EXTRA`, `DRY_RUN` |
| `analyze.sh` | Stage 2: `poster_metrics` — every metric, table and figure, from stage 1's `.npz`. CPU, minutes. | `POST` (input dir), `OUT` (results dir), `DRY_RUN` |
| `shrink.sh` | Figure D (`fig_shrinkage`), screen and poster builds. CPU. | `POST`, `OUT`, `DRY_RUN` |
| `treetest.sh` | Smoke test: all three models sample 4 cases from the reorganised tree. | — |
| `finaltest.sh` | Smoke test: sampling *and* metrics end to end into `~/cancer/_final`. | — |
| `verify.sh` | `src/verify_refactor.py --with-models` against the archived legacy tree. Expect 42/42. | — |

`DRY_RUN=1 bash jobs/<script>.sh` prints the command(s) it would run and exits. It needs no conda
env, so it works on the laptop.

`CACHE_DIR` is worth its own line: the default `sbatch` runs **uncached**, straight off the gzipped
trial files at ~28 min/epoch, and `CACHE_DIR=$CANCER/data/cache/clone_top100_v1` points every run at
the pre-built clone cache instead, which is roughly **13x faster per epoch**. It is added to the
four clone-set runs only: the cache holds `(25, top_k, 45)` clone sets, which DominantClone does not
read, so D0 never gets `--cache-dir`.

**D0, the same-sims retrain.** Index 4 is `dominantclone --require-all-trials`: the DominantClone
dataset normally NaN-pads a missing trial and keeps the sim (trap 10), so it trains and is scored on
~3,506 sims where the clone-set models get 3,159. With the flag it uses exactly the clone-set sim
set (2,261 train / 247 val / 651 test on the cluster), which is what makes the three models'
numbers comparable. It passes no `--z-score-x`: the `dominantclone` preset is already `structured`.
The flag is recorded in the checkpoint's `effective_config`, and `sample_posteriors.py` /
`cli/evaluate.py` rebuild the test loader with the same restriction automatically.

## Matrix 2 (`train2.sh`)

Matrix 1 left two findings to act on: R2 (CloneMLP, `--z-score-x structured --input-space copy`)
reached R² = 0.415 but overfits from epoch ~15, and R4 left CloneAtt at R² = 0.025 because
renormalising the frequencies still spreads one unit of mass over 100 clones, so every token stays
at ~1/100 of its scale with no LayerNorm to rescale it — on an encoder whose learning rate is 1e-4
and which has no dropout at all (trap 4). The seven runs below are those two threads. All of them
whiten theta (`--z-score-x structured`), all use the clone cache, and all share
`--min-epochs 1 --stop-after-epochs 15 --max-epochs 60 --num-workers 8`; the seed is 20260924
except where the table names another.

| idx | run | model | flags |
| --- | --- | --- | --- |
| 0 | R3 | clonemlp | `--z-score-x structured --input-space copy --flow-weight-decay 1e-3 --embed-weight-decay 1e-4` |
| 1 | R2s1 | clonemlp | `--z-score-x structured --input-space copy --seed 1` |
| 2 | R2s2 | clonemlp | `--z-score-x structured --input-space copy --seed 2` |
| 3 | R5 | cloneatt | `--z-score-x structured --freq-mode feature --attn-ln` |
| 4 | R6 | cloneatt | R5 + `--input-space copy` |
| 5 | R7 | cloneatt | R6 + `--embed-lr 5e-4` |
| 6 | R8 | cloneatt | R7 + `--encoder-dropout 0.1` |

R3 asks whether weight decay stops R2's overfit; R2s1/R2s2 are R2 on two more seeds, so the next
comparison knows how much of 0.415 is seed noise. R5–R8 are a ladder on CloneAtt, one switch at a
time: the frequency as a log10 *feature* rather than a multiplier plus LayerNorm (R5), then the
copy-space input R2 uses (R6), then an encoder learning rate that is not near-frozen (R7), then the
dropout trap 4 denied it (R8). Every switch defaults to the published behaviour and is recorded in
the checkpoint's `effective_config`, so `sample_posteriors.py` and `cli/evaluate.py` rebuild the
right network without being told.

## Matrix 3 (`train3.sh`)

Matrix 2 left four findings. Every run overfits inside 10–25 epochs (train ≈ −40 nat against
val ≈ −20, on a 421k-parameter flow fitted to 2,261 sims); weight decay helps (R3); CloneAtt is
repaired by `--freq-mode feature --attn-ln --input-space copy` (R6, R² = 0.42); and the seed spread
on R² is ±0.05, which is the size of most of the differences being read. So matrix 3 is
regularisation and augmentation stacked on the two survivors, plus the replicates that say which
differences are real. Two bases, named in the script:

- **BASE_R3** (clonemlp) `--z-score-x structured --input-space copy --flow-weight-decay 1e-3 --embed-weight-decay 1e-4`
- **BASE_R6** (cloneatt) `--z-score-x structured --input-space copy --freq-mode feature --attn-ln`

All fourteen share `--min-epochs 1 --stop-after-epochs 15 --max-epochs 60 --num-workers 8`, and the
seed is 20260924 except where the table names another. D0s1/D0s2 get no `--cache-dir`: the clone
cache holds `(25, top_k, 45)` clone sets, which DominantClone does not read.

| idx | run | model | flags |
| --- | --- | --- | --- |
| 0 | R3s1 | clonemlp | BASE_R3 `--seed 1` |
| 1 | R3s2 | clonemlp | BASE_R3 `--seed 2` |
| 2 | R6s1 | cloneatt | BASE_R6 `--seed 1` |
| 3 | R6s2 | cloneatt | BASE_R6 `--seed 2` |
| 4 | D0s1 | dominantclone | `--require-all-trials --seed 1` (no cache) |
| 5 | D0s2 | dominantclone | `--require-all-trials --seed 2` (no cache) |
| 6 | R9 | cloneatt | BASE_R6 `--flow-weight-decay 1e-3 --embed-weight-decay 1e-4` |
| 7 | R10 | clonemlp | BASE_R3 `--freq-mode feature` |
| 8 | R11 | cloneatt | BASE_R6 `--flow-dropout 0.3` |
| 9 | R12 | cloneatt | BASE_R6 `--flow-num-transforms 3` |
| 10 | R13 | cloneatt | BASE_R6 `--trial-subsample 16` |
| 11 | R14 | clonemlp | BASE_R3 `--flow-dropout 0.3` |
| 12 | R15 | clonemlp | BASE_R3 `--trial-subsample 16` |
| 13 | R16 | clonemlp | BASE_R3 `--flow-num-transforms 3` |

Indices 0–5 are the six replicates: two seeds each of R3, R6 and D0, so the next comparison of the
three models knows its own error bar. R9 asks whether R3's weight decay helps CloneAtt too. R10
asks the reverse question — the log10-frequency *feature* that repaired CloneAtt, given to
CloneMLP; note that CloneMLP's frequency-weighted mean pooling is **kept**, because it is a
normalised weighted mean and shrinks nothing, unlike CloneAtt's raw token multiply (trap 19). R11–R16
are the three new knobs on both models: dropout inside the flow (0.1, *below* the published 0.2,
since 0.2 already overfits), a 3-transform flow instead of 5 (`hidden_features` stays 50), and
training on a random 16 of the 25 trials per sim, redrawn every epoch.

`--trial-subsample` is training-only by construction: `build_clone_set_dataloaders` hands it to the
train dataset and to nothing else, so validation and test keep all 25 trials — the published
evaluation condition, and the one every reported number is measured under. The draw comes from the
ambient torch RNG, so it is fresh each epoch and reproducible from `--seed` alone, including with
`--num-workers 8`. `K >= 25` is inert and bit-identical to leaving the flag off. DominantClone
warns and ignores the flag: a random 16 of its 25 NaN-padded rows could be all-NaN for a sim the
published path keeps, which would silently change its sim set.

Every switch defaults to the published behaviour and every one is recorded in the checkpoint's
`effective_config`, so `sample_posteriors.py` and `cli/evaluate.py` rebuild the right network —
including the 3-transform flow and the 45-input MLP — without being told.

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

## Matrix 4 (`train4.sh`)

Matrix 3's best model is CloneAtt **R12** (R² = 0.413, 8/44 SBC failures), and the per-arm analysis
says the best-learned arms carry a small consistent posterior-mean bias. Two things are suspected of
it: the flow's `tail_bound = 3` clipping the spline's support, and the single scalar `structured`
θ standardisation. Matrix 4 tests those two, the one published oddity matrix 2 left alone
(trap 6's attention temperature), the mean over the 25 trials, and CloneAtt's capacity — each on top
of one base, one switch at a time, so a difference in R² is attributable.

- **BASE_R12** (cloneatt) `--z-score-x structured --input-space copy --freq-mode feature --attn-ln --flow-num-transforms 3`

All twelve are `cloneatt`, share `--min-epochs 1 --stop-after-epochs 15 --max-epochs 60
--num-workers 8`, and use the clone cache when `CACHE_DIR` is set — there is no DominantClone
exception in this matrix. The seed is 20260924 except where the table names another.

| idx | run | flags on top of BASE_R12 | what it tests |
| --- | --- | --- | --- |
| 0 | R17 | `--attn-scale standard` | trap 6: logits scaled by `sqrt(d_model/n_heads)`, so the attention is 2.83× sharper |
| 1 | R18 | `--tail-bound 5` | is the posterior-mean bias the flow clipping θ at the spline's tail? |
| 2 | R19 | `--z-score-x independent` (replaces `structured`) | the other suspect: per-dimension θ whitening instead of one scalar |
| 3 | R20 | `--trial-pool attention` | pool the 25 trial embeddings with a PMA instead of sbi's masked mean |
| 4 | R20s1 | `--trial-pool attention --seed 1` | R20 on a second seed — it is the one run that adds a module |
| 5 | R21 | `--d-model 256` | capacity: a wider per-trial embedding |
| 6 | R22 | `--n-heads 4` | capacity: fewer, wider heads (same parameter count) |
| 7 | R23 | `--num-inducing 64` | capacity: a wider ISAB bottleneck |
| 8 | R24 | `--d-model 256 --num-inducing 64` | the two capacity knobs together |
| 9 | R12s1 | `--seed 1` | the base itself, the yardstick every row above is read against |
| 10 | R12s2 | `--seed 2` | the second point of that spread (±0.05 on R²) |
| 11 | R25 | `--attn-scale standard --trial-pool attention --tail-bound 5` | do the three independent switches compose? |

`--attn-scale`, `--n-heads` and `--num-inducing` are CloneAtt-only; `--trial-pool` and `--d-model`
are read by both clone-set models; `--tail-bound` by all three. Every one of them defaults to the
published behaviour, and `--d-model` must stay divisible by `--n-heads` — `cli/train.py` refuses the
pair rather than letting the attention silently drop the remainder of every token. `--trial-pool
attention` keeps the post-pooling MLP and therefore the flow's context width, so R20 is comparable
with the base on everything else; it does change the `state_dict`, so its checkpoints must be
evaluated with the config they carry (which `sample_posteriors.py` does by default).

## Matrix 5 — ArmToken (`train5.sh`)

A fourth model, not another switch. Every encoder so far tokenises a **clone** — a 44-vector of log2
ratios — and pools the clones away, so the only thing the flow ever sees of arm 17 is whatever
survived a projection that mixed all 44 arms on its first layer, and nothing ties output arm 17 to
input arm 17. `ArmTokenEmbedding` inverts the set: the 44 **arms** are the tokens and the clones are
summarised away by eight frequency-weighted moments per (trial, arm) — weighted mean, weighted sd,
the weighted fractions lost / gained / deeply lost, the max, the min, and the value in the single
most frequent clone — computed in copy space with weights renormalised to sum to 1, which is trap 19
avoided at the source rather than repaired downstream.

Those moments are pooled over the 25 trials (masked mean **and** sd, so the between-replicate spread
survives), projected by one MLP **shared across all 44 arms**, mixed by a stack of ISABs over the
44-element arm set, and read out to `d_arm = 8` numbers per arm; the context is the 44 blocks
concatenated in arm order plus a 64-wide global block (a PMA over the arms, invariant, plus
log10 of the kept frequency mass and of the largest clone frequency). Every weight is shared across
arms, so **permuting the 44 input columns permutes the 44 output blocks exactly** — arm identity
reaches the flow only through the concatenation order. The module is the whole embedding net and is
*not* wrapped in `TrialsSBIEmbedding`: that wrapper's MLP would blend the per-arm blocks back
together and destroy the property. Context width is `44 * 8 + 64 = 416`; the encoder is ~62k
parameters, about an eighth of CloneAtt's.

The `armtoken` preset already carries matrices 2–4's findings (`z_score_x = structured`,
`input_space = copy`, LayerNorm on, a 3-transform flow, `hidden_features = 50`), so the base run
names no repair flag at all. All six are `armtoken`, share `--min-epochs 1 --stop-after-epochs 15
--max-epochs 60 --num-workers 8`, and use the clone cache when `CACHE_DIR` is set. The seed is
20260924 except where the table names another.

| idx | run | flags on top of the preset | what it tests |
| --- | --- | --- | --- |
| 0 | AT0 | *(none)* | the encoder itself — the yardstick every row below is read against |
| 1 | AT0s1 | `--seed 1` | a brand-new architecture's own seed spread (matrices 2–4: ±0.05 on R²) |
| 2 | AT1 | `--trial-pool attention` | pool the 25 trials with a per-arm PMA instead of the masked `[mean, sd]` |
| 3 | AT2 | `--attn-scale standard` | trap 6's temperature, asked of the arm attention: `sqrt(d_token/n_heads)` |
| 4 | AT3 | `--arm-layers 0` | no cross-arm mixing at all — is the ISAB worth its parameters, or is this 44 independent regressions? |
| 5 | AT4 | `--embed-lr 1e-3` | R7's question for a 62k-parameter encoder the published 1e-4 near-freezes |

`--arm-layers`, `--d-arm` and `--arm-num-inducing` are ArmToken-only; `--trial-pool`, `--attn-scale`,
`--attn-ln`, `--n-heads`, `--input-space` and `--encoder-dropout` are shared with the other models.
`--freq-mode`, `--freq-renorm`, `--d-model`, `--num-inducing` and `--require-all-trials` are warned
about and ignored for `armtoken` — the moments renormalise the weights themselves, the output width
is `44 * d_arm + d_global` rather than a `d_model`, and `--num-inducing` is deliberately a separate
flag from `--arm-num-inducing` so that one cannot silently reshape the other model. `44 * d_arm +
d_global` is capped at 512: beyond that the flow's first layer is again the biggest thing in the
network, which is what this encoder exists to avoid, so `cli/train.py` refuses the `--d-arm` rather
than letting it through.

## Matrix 4b — follow-up (`train4b.sh`)

| idx | run | flags on top of BASE_R12 |
| --- | --- | --- |
| 0 | R18s1 | `--tail-bound 5 --seed 1` |
| 1 | R18s2 | `--tail-bound 5 --seed 2` |
| 2 | R26 | `--tail-bound 5 --d-model 256` |
| 3 | R26s1 | `--tail-bound 5 --d-model 256 --seed 1` |
