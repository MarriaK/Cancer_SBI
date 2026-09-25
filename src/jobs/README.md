# `jobs/` — the SLURM scripts

## Defaults changed on 2026-09-25

`--model clonemlp`, `--model cloneatt` and `--model dominantclone` no longer mean "the model as
published". They now build the best **honest** configuration the 2026-09-24 campaign found — R2,
R26 and D0 respectively — chosen on calibration first and accuracy only outside the seed band. What
changed per model, and why, is written beside each preset in `src/cancer_sbi/config.py` under
"The repaired defaults"; the evidence is `docs/CAMPAIGN_REPORT_2026-09-24.md` §4 and findings 2, 4
and 8.

| you want | you pass |
| --- | --- |
| the repaired default (R2 / R26 / D0) | `--model clonemlp` (nothing else) |
| the model **as published** | `--model clonemlp --published`, or `--model clonemlp_published` |

`--published` exists on `cli/train.py`, `cli/evaluate.py` and `evaluation/sample_posteriors.py`, and
is an **error** for `armtoken` and `hybrid`, which were introduced by the campaign and have no
published version.

Two consequences for this directory:

* **`train.sh`, `train2.sh`, `train3.sh`, `train4.sh` and `train4b.sh` now pass `--published` on
  every run.** Each of those runs is a campaign run — "the published preset plus the flags this row
  names" — and each relied on the old defaults for the fields it does not name (R0 never names
  `--input-space`; R5–R26 never name `--tail-bound` until R18; nothing before matrix 4 names
  `--d-model`). The flag restores exactly those, so every run's effective config is unchanged. A
  per-run flag still wins over it, so D0's `--require-all-trials` lands on top of the published
  DeepSet as before.
* **`train5.sh`–`train8.sh` are untouched.** They are `armtoken` and `hybrid` runs and every one of
  them already passes `--tail-bound 5` explicitly, so the ArmToken preset now carrying `5.0` (it
  always should have — no ArmToken run was ever made at sbi's 3.0) changes nothing they do.

A checkpoint records the **resolved** preset name, so evaluation rebuilds what training built. A
checkpoint written *before* 2026-09-24 carries no config at all — the three published checkpoints on
the cluster — and `resolve_eval_config` falls back to the `_published` preset for it, with a warning
that says so.

### Two things to get right when evaluating

**`--published` selects the preset, not a path.** It does not move any checkpoint. The cluster's
published checkpoints are still where they always were, under the **bare** name
(`runs/clonemlp/checkpoints/best.pt`, and likewise for the other two), and they carry no
`effective_config`. So evaluating them is plain `--model clonemlp` **without** the flag: the
no-config fallback already picks `clonemlp_published` and says so in a `[warn]` line. Passing
`--published` as well is harmless for the architecture but changes the default run directory to
`runs/clonemlp_published/`, where there is nothing — pass `--ckpt` if you do.

**A checkpoint trained with `--published` stores `model=<name>_published`,** and every consumer
cross-checks that name against `--model`. For such a run, pass `MODEL=<name>_published` to
`sample.sh`, `recalibrate.sh`, `ensemble.sh`, `finaltest.sh` and `treetest.sh` — not the bare name.
Get it wrong and nothing is silently mis-scored; you get

```
<ckpt> was trained as model 'clonemlp_published', but --model says 'clonemlp'.
Evaluate it as 'clonemlp_published'.
```

Its stage-1 output is `posteriors_<name>_published[_<tag>].npz`, which `analyze.sh` and `shrink.sh`
already know about, so a published run and a repaired run of the same model never overwrite each
other.

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
| `sample.sh` | Stage 1: 5000 posterior draws per held-out tumour, one array task per model. 12 h, a100. For one run: `--array=<i>` or `MODEL=`; `CKPT` with the full 0-2 array is refused. | `MODEL`, `CKPT` (the checkpoint to sample — **always pass it for a matrix run**), `POST` (output dir), `RUN_TAG`, `PARTITION` (`test`\|`val`, default `test`; `val` samples the validation ids into `..._val.npz` for `recalibrate.sh`), `EXTRA` (extra flags, last-wins), `DRY_RUN` |
| `ensemble.sh` | Stage 1b: pools several seeds' stage-1 `.npz` files into one equal-weight mixture posterior, `posteriors_<MODEL>_<RUN_TAG>.npz`, which `analyze.sh` then reads unchanged. CPU, minutes. | `INPUTS` (space-separated `.npz` paths, required), `OUT` (output dir), `MODEL`, `RUN_TAG`, `NUM_SAMPLES` (subsample, equal share per member), `EXTRA`, `DRY_RUN` |
| `recalibrate.sh` | Stage 1c: fits a per-arm affine correction (bias `a_j`, width factor `b_j`, both in posterior-sd units) on a **validation** stage-1 file written by `PARTITION=val jobs/sample.sh`, applies it to the test file, and writes `posteriors_<MODEL>_<RUN_TAG>.npz`, which `analyze.sh` then reads unchanged. `log_prob_true` is NaN in the output (the corrected draws are no longer the flow's density). CPU, minutes. | `VAL` (validation `.npz`, required), `TEST` (test `.npz`, required), `OUT` (output dir), `MODEL`, `RUN_TAG`, `METHOD` (`affine`\|`shift`), `EXTRA`, `DRY_RUN` |
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

## Matrix 6 — the hybrid, and ArmToken's flow (`train6.sh`)

Matrix 5 settled the architecture question: ArmToken (AT0) reaches R² 0.898 where CloneAtt's
best-ever run (R26) reaches 0.568. This matrix asks the two follow-ups that answer leaves open, as a
`--array=0-8` job with the same split gate, non-empty-checkpoint refusal and 12 h a100 as
`train5.sh`.

**Rows 0–2, the hybrid.** A fifth model, `--model hybrid` (`Hybrid-NPE`, `models/hybrid.py`):
ArmToken's encoder and R26's CloneAtt encoder run **side by side** on the same input, their contexts
concatenated — 416 from the arm branch, 256 from the clone branch, **672** to the flow. The fusion is
*late* on purpose: the branches share no weights and never see each other's activations, so the arm
branch is bit-for-bit the module matrix 5 measured and a hybrid that only matches AT0 says the clone
branch added nothing. The symmetry is correspondingly partial and the tests assert all three halves:
the first 352 outputs are exactly arm-equivariant, the next 64 exactly arm-invariant, and the last
256 are **neither** — which is the point, since per-arm moments throw away the joint pattern of two
arms *within* one clone and clone-level attention is the one encoder here that keeps it. Three seeds,
because the difference being tested is the size of the seed spread (±0.05 on R²).

**Rows 3–8, ArmToken's flow.** Its encoder is 62k parameters and its flow, at 3 transforms × 50
hidden features, is now the small half of the model; every matrix so far shrank the flow to fight
overfitting and none has asked whether ArmToken's context can feed a bigger one. AT0s2 is the
yardstick's third seed so the six flow rows are read against a three-seed baseline.

All nine share `--min-epochs 1 --stop-after-epochs 15 --max-epochs 60 --num-workers 8 --tail-bound
5`, use the clone cache when `CACHE_DIR` is set, and take seed 20260924 except where the table says.

| idx | run | flags on top of the preset | what it tests |
| --- | --- | --- | --- |
| 0 | H0 | `--model hybrid` | the hybrid itself — does clone-level attention add anything on top of the per-arm moments? |
| 1 | H0s1 | `--model hybrid --seed 1` | the hybrid's own seed spread |
| 2 | H0s2 | `--model hybrid --seed 2` | likewise |
| 3 | AT0s2 | `--model armtoken --seed 2` | matrix 5's yardstick, third seed |
| 4 | AT5 | `--flow-num-transforms 5` | is R12's shrink to 3 transforms still right now the encoder does the work? |
| 5 | AT6 | `--flow-hidden-features 100` | 50 is the published width in *every* run of every matrix; nothing has tested it |
| 6 | AT7 | `--lr-plateau` | `ReduceLROnPlateau(factor 0.5, patience 5)` on the validation loss |
| 7 | AT8 | `--flow-num-transforms 5 --flow-hidden-features 100` | the bigger flow, both dimensions |
| 8 | AT9 | `--flow-num-transforms 5 --flow-hidden-features 100 --lr-plateau` | the biggest flow with the best chance of settling into it |

Both new flags default to the published behaviour. `--flow-hidden-features` overrides
`FlowConfig.hidden_features`, which matrices 1–5 deliberately left without a flag because 50 is the
published width — it is an experiment flag, not a repair. `--lr-plateau` builds **no scheduler at
all** when absent, so a default run's training is bitwise what it was; when present the scheduler
steps on the validation loss once per epoch, prints a line whenever it lowers a rate (both optimiser
groups move together), and its state rides in the checkpoint so a resume continues with the rate it
had reached. Two seeded two-epoch runs, one with the flag and one without, are pinned to produce
identical weights.

For `hybrid`, `--d-model`, `--n-heads`, `--num-inducing`, `--freq-mode` and `--attn-scale` size the
**clone** branch; `--arm-layers`, `--d-arm` and `--arm-num-inducing` the **arm** branch; `--attn-ln`,
`--input-space`, `--encoder-dropout` and `--trial-pool` apply to **both**. `--freq-renorm` and
`--require-all-trials` are warned about and ignored. One `--n-heads` serves both branches, so it must
divide the clone branch's `d_model` (256) *and* the arm branch's `d_token` (64); 8 does both and
`cli/train.py` refuses a value that does not.

## Matrix 7 — partial sims (`train7.sh`, `prepare_partial.sh`)

This matrix changes the **data**, not the model. Trap 10 drops a simulation whole when even one of
its 25 replicate tumours is missing a `CNratios_all.pkl.gz`, so the 441 of the cluster's 3,600 sims
that have 1–24 complete replicates (94 more have 0) have never been seen by a clone-set model.
`--min-trials 5` keeps them: the slots with no file stay NaN and the encoders mask them — ArmToken
natively (`models/arm_tokens.py`), CloneAtt/CloneMLP through sbi's NaN-aware trial pooling
(`models/trials.py`) — so not one encoder needs a line changed. Roughly +12% training sims. The bar
is 5 rather than 1 because a sim summarised from one or two replicates carries almost none of the
between-replicate spread, which is half of what the per-arm moments measure.

**The rule: train and validation gain the partial sims; the test set does not.**
`build_clone_set_dataloaders` gives `min_trials` to the training and validation datasets only and
takes a separate `test_min_trials`, which no caller sets, so the test dataset keeps the published
complete-sim rule and stays the same 651 cases every earlier matrix reported on. A gain here is
therefore a gain from more training data, not from an easier test set. Validation follows training
rather than test: early stopping should see the kind of item the model is being fitted on, and the
validation set is never reported.

Run `sbatch jobs/prepare_partial.sh` first — it builds `data/cache/clone_top100_partial_v1` with
`--min-trials 5` from the split `prepare.sh` already carved, and verifies it (the verifier also
checks the NaN padding and the recorded trial counts). `jobs/prepare.sh` and
`data/cache/clone_top100_v1` are untouched and still complete-only; a partial cache additionally
carries `trial_counts.npy` and `min_trials` in its manifest, and a dataset asking for `min_trials=5`
against the complete-only cache refuses at construction, naming the sims that cache lacks.

All four share `--min-epochs 1 --stop-after-epochs 15 --max-epochs 60 --num-workers 8 --tail-bound 5
--min-trials 5`, default `CACHE_DIR` to the partial cache, and take seed 20260924 except where the
table says. `--array=0-3`.

| idx | run | flags on top of the preset | what it tests |
| --- | --- | --- | --- |
| 0 | AT10 | `--model armtoken` | matrix 5's yardstick (R² 0.898) on the bigger training set — the row the matrix is read against |
| 1 | AT10s1 | `--model armtoken --seed 1` | the seed spread (±0.05 on R²), without which no difference from AT0 can be called real |
| 2 | R27 | `--model cloneatt --z-score-x structured --input-space copy --freq-mode feature --attn-ln --flow-num-transforms 3 --d-model 256` | R26, the best CloneAtt ever run (R² 0.568), plus the partial sims — is CloneAtt data-starved rather than mis-specified? |
| 3 | H1 | `--model hybrid` | matrix 6's hybrid, same question |

R27 names R26's six flags explicitly because the `cloneatt` preset, unlike `armtoken`'s, does not
carry matrices 2–4's findings. `--min-trials` defaults to unset everywhere else, so every earlier
matrix's behaviour is byte-identical; it is warned about and ignored for `dominantclone`, which
NaN-pads missing trials already and so has no bar to lower.

## Matrix 8 — ArmToken input space and feature normalisation (`train8.sh`)

Matrix 5 built ArmToken with two things written down as accepted risks and never measured. The first
is scale *inside* the encoder: the eight moments per (trial, arm) sit on four different ranges — a
weighted mean and the min/max/dominant values in `[-1, 3]`, an sd in `[0, ~2]`, three fractions in
`[0, 1]` — and those eight, plus their across-trial mean and sd, go straight into `arm_mlp` with no
normalisation at all. The second is scale *out* of it: the 416-wide context reaches the flow with
`z_score_y="none"`, so the flow sees whatever the arm and global heads happen to produce. sbi's
`z_score_y` is **not** the switch for that second one — it standardises the raw `(B, T, K, 45)` input
tensor *before* the embedding net, so it never touches the context — which is why matrix 8 adds a
LayerNorm inside `ArmTokenEmbedding` instead.

Two new flags, both defaulting to AT0's behaviour and both building **no module at all** when left
alone, so a default `armtoken` run's `state_dict` is byte for byte the one matrix 5 measured:

* `--arm-feature-norm {none,layernorm,batchnorm}` normalises the pooled moment vector `(B, 44, P)`
  (`P` = 16 on the mean trial-pool path, `d_token` on the attention path) immediately before
  `arm_mlp`. `layernorm` is `nn.LayerNorm(P)` — each arm token z-scored across its own P features.
  `batchnorm` is `nn.BatchNorm1d(P)` over the `(B·44, P)` view — a learned per-feature z-scoring
  whose running statistics are used at eval time. Either way it is **one module shared by all 44
  arms**, and the batchnorm's statistics are pooled over the whole `(B·44)` set, which a permutation
  of the arms reorders without changing — so the arm-equivariance ArmToken exists for survives both,
  and the test suite pins that with the same permutation helper matrix 5 used.
* `--arm-context-norm` puts `nn.LayerNorm(416)` on the encoder's output. It normalises all 416
  entries at once, per sample; permuting the arms permutes those entries, leaving the mean and
  variance it divides by unchanged, so the 44 blocks stay equivariant.

For `hybrid`, both flags reach the **arm branch only** — they are forwarded by
`build_arm_token_encoder`, which the clone branch does not go through — so a hybrid's
`CloneSetEmbedding` and its `TrialsSBIEmbedding` wrapper are untouched. They are warned about and
ignored for `clonemlp`, `cloneatt` and `dominantclone`. Both land in the checkpoint's effective
config and are honoured on rebuild (`[config] rebuilt` prints them for `armtoken` and `hybrid`), and
a checkpoint written before they existed rebuilds as AT0.

All eight runs are `--model armtoken` and share `--min-epochs 1 --stop-after-epochs 15 --max-epochs
60 --num-workers 8 --tail-bound 5`, seed 20260924 except where the table says. `--array=0-7`. Every
row is AT0 plus exactly one thing, and every row is paired with a seed-1 replicate, because ±0.05 on
R² is the spread matrices 2–4 measured and AT0's own band is ±0.002 over three seeds.

| idx | run | flags on top of the preset | what it tests |
| --- | --- | --- | --- |
| 0 | AT11 | `--input-space log2` | is copy space actually better than the stored log2 values, isolated in the model with the tightest seed band? |
| 1 | AT11s1 | `--input-space log2 --seed 1` | the same, on the replicate seed |
| 2 | AT12 | `--arm-feature-norm layernorm` | does normalising the moment features help — each arm token across its own 16 numbers? |
| 3 | AT12s1 | `--arm-feature-norm layernorm --seed 1` | the same, on the replicate seed |
| 4 | AT13 | `--arm-feature-norm batchnorm` | does normalising the moment features help — per feature, across the whole `(B·44)` set? |
| 5 | AT13s1 | `--arm-feature-norm batchnorm --seed 1` | the same, on the replicate seed |
| 6 | AT14 | `--arm-context-norm` | does normalising the context help — the 416 numbers the flow is actually given? |
| 7 | AT14s1 | `--arm-context-norm --seed 1` | the same, on the replicate seed |

AT11 is the one row that *undoes* a preset default rather than adding to it: the `armtoken` preset
carries `input_space="copy"` from repair T2, and no run has ever put the stored log2 values back on
this encoder. `CACHE_DIR` defaults to empty and, when set, must be the **complete-sim** cache
(`clone_top100_v1`) — not matrix 7's partial one — because every row here is read against AT0.

## Best-honest evaluation (`best_honest.sh`, 2026-09-25)

`jobs/best_honest.sh` rebuilds the whole poster figure set from the stored posteriors. CPU only,
`-p general`, 2 h — it reads `.npz` files and draws; the GPU work happened in `jobs/sample.sh`.

```
sbatch jobs/best_honest.sh
RESULTS=$HOME/cancer/results/2026-09-24 OUT=$HOME/cancer/results/best_honest sbatch jobs/best_honest.sh
DRY_RUN=1 bash jobs/best_honest.sh          # print every command, run nothing
```

It runs in two stages. **Per run**, for each of the twelve runs the manifest names, it re-runs
`poster_metrics` (metrics tables, figures A–C, `summary_arrays.npz`), `tarp` (the joint
calibration test, `tarp_curve.csv` + `tarp_summary.json`) and `fig_shrinkage` (figure D), all with
`--in-dir $RESULTS/<run>/posteriors --out-dir $RESULTS/<run> --run-tag <run>`. **Across runs**, it
calls `utilities/collect_best_honest.py` to copy the winners into `$OUT` with its README table, and
then `cancer_sbi.evaluation.best_honest_figures`, which writes into `$OUT/figures/`:

| file | what it is |
| --- | --- |
| `fig_E_per_arm_r2.{png,pdf}` | the headline: per-arm true R² (1 − SSE/SST), 44 arms on the x-axis, one line per encoder, error bars = sd over that encoder's training seeds |
| `fig_S3_per_arm_coverage.{png,pdf}` | the same strip for per-arm 95 % coverage, with the ±2σ binomial band |
| `fig_calibration_panel.{png,pdf}` | one column per encoder: the 44 stacked SBC rank-ECDF differences with a simulated simultaneous band, and the TARP curve below |
| `fig_D_panel.{png,pdf}` | the pooled truth-vs-posterior-mean hexbins side by side, true R² and r² annotated |
| `per_arm_summary.csv` / `.md` | the S2 table: mean, sd and n per arm per encoder for true R², r², RMSE, contraction, coverage₉₅ and the SBC KS p |
| `headline_table.md` | one row per encoder — headline run, true R², the seed band, log p(θ*), coverage, SBC failures and the TARP ATC |

A figure whose input is missing is skipped with a `[warn]`, not an error: the cross-run script
runs unchanged on the CSV-only copies under `results/best_honest/`, where there is no
`summary_arrays.npz` and no TARP curve.

`posterior_export.npz` is no longer a side effect of the metrics run: `poster_metrics` writes it
only with `--poster-export`, and the poster's own producer is `overview_figure/export_posterior.py`
— the cross-run stage reads `summary_arrays*.npz` instead, which carries every model rather than
CloneMLP alone.

An out-dir may hold several models at once (`results/published/` has three). Everything the
per-run stage writes says which model it belongs to — a `model` column in `metrics_*.csv` and
`tarp_curve.csv`, a list of objects in `tarp_summary.json`, and `summary_arrays_<model>.npz`
beside the plain `summary_arrays.npz` — and the cross-run stage filters each file down to the
`model` its manifest entry names, so a shared directory is read correctly without any flag.

`$OUT` must be a **sibling** of `$RESULTS` (`collect_best_honest.py` writes
`<results root>/<out name>`); the script refuses otherwise rather than writing somewhere surprising.

### The manifest, and how to add a run

Which runs are evaluated is not written in any script. It is
`src/cancer_sbi/evaluation/manifests/best_honest_2026-09-24.json`, one entry per encoder:

```json
{"key": "armtoken", "label": "ArmToken-NPE", "model": "armtoken",
 "headline": "AT0ens3", "members": ["AT0", "AT0s1", "AT0s2"],
 "config": "--model armtoken --tail-bound 5; ...", "why": "...", "colour": "#2a78d6"}
```

`headline` is the run whose numbers are quoted; `members` are the seed replicates of the *same*
configuration, and they alone give every error bar and every "mean ± sd" — an ensemble headline like
`AT0ens3` is deliberately not one of them. `collect_best_honest.py`, `best_honest_figures` and this
job script all read that one file, so:

* **another seed of an existing encoder**: train and evaluate it, then add its run tag to that
  encoder's `members`. Nothing else changes; the error bars widen by themselves.
* **a new encoder**: add an object with a new `key` (which is also its folder name under
  `results/best_honest/`), a poster `label`, the preset `model` name, a `headline`, at least one
  member, the `config` string the README prints and a `colour` that is not already in use.
* **a new campaign**: copy the file to `best_honest_<date>.json`, edit it, and pass
  `MANIFEST=` to the job (or `--manifest` to either Python entry point).

`RESULTS` may also point at an encoder tree (`<encoder>/<run>/`) instead of a campaign directory
(`<run>/`) — `best_honest_figures` accepts both, which is how the figures can be redrawn from
`results/best_honest/` on a laptop with no cluster access.
