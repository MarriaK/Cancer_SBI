# Best honest configuration per encoder

Assembled by `src/utilities/collect_best_honest.py` from `results/2026-09-24/`.
All runs are evaluated on the same 651 held-out simulations with 5,000 posterior draws each.
"Honest" means calibrated posteriors first, accuracy quoted with its seed spread; see
`docs/CAMPAIGN_REPORT_2026-09-24.md` (§4, §6) for the full campaign.

| Encoder | Headline run | true R² (headline) | true R² (seeds, mean ± sd) | log p(θ*) | 95 % coverage | SBC failures /44 | Configuration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| armtoken | AT0ens3 | 0.904 | 0.898 ± 0.002 (n=3) | 68.0 | 0.961 | 21 | `--model armtoken --tail-bound 5; AT0ens3 = equal-weight mixture of the three seeds' posteriors (jobs/ensemble.sh)` |
| cloneatt | R26 | 0.568 | 0.579 ± 0.015 (n=2) | 29.8 | 0.930 | 28 | `--z-score-x structured --input-space copy --freq-mode feature --attn-ln --flow-num-transforms 3 --tail-bound 5 --d-model 256` |
| clonemlp | R2 | 0.415 | 0.363 ± 0.053 (n=3) | 17.8 | 0.938 | 20 | `--z-score-x structured --input-space copy` |
| dominantclone | D0 | 0.175 | 0.177 ± 0.017 (n=3) | 11.7 | 0.954 | 9 | `--require-all-trials  (same 651 test cases as the other encoders)` |

Ranking on identical data: ArmToken ≫ CloneAtt > CloneMLP ≫ DominantClone. Differences in R² below ~0.05
between the two clone-set models are within their seed noise; ArmToken's seed band is ±0.002.
