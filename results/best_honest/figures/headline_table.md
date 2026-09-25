# Best honest configuration per encoder — headline numbers

Campaign `2026-09-24`, 651 held-out simulations, 44 chromosome arms. true $R^2$ is 1 − SSE/SST, averaged over the arms; the seed column is the mean ± sd over the member runs of the same configuration. TARP dist. is the mean |gap| to the diagonal, a distance; TARP direction is that gap with its sign — positive means the posterior is too wide, negative over-confident, and under 0.01 either way it is on the diagonal.

| Encoder | Headline run | true R² (headline) | true R² (seeds, mean ± sd) | log p(θ*) | 95 % coverage | SBC fails /44 (headline; seed range) | TARP dist. | TARP direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ArmToken-NPE | AT0ens3 | 0.904 | 0.898 ± 0.002 (n=3) | 68.0 | 0.961 | 21 (15–37) | 0.022 | +0.011 (too wide) |
| CloneAtt-NPE | R26 | 0.568 | 0.579 ± 0.015 (n=2) | 29.8 | 0.930 | 28 (26–28) | 0.016 | -0.017 (over-confident) |
| CloneMLP-NPE | R2 | 0.415 | 0.363 ± 0.053 (n=3) | 17.8 | 0.938 | 20 (8–21) | 0.080 | +0.069 (too wide) |
| DominantClone-NPE | D0 | 0.175 | 0.177 ± 0.017 (n=3) | 11.7 | 0.954 | 9 (8–10) | 0.011 | +0.016 (too wide) |

Configurations:

* **ArmToken-NPE** (`AT0ens3`): `--model armtoken --tail-bound 5; AT0ens3 = equal-weight mixture of the three seeds' posteriors (jobs/ensemble.sh)`
* **CloneAtt-NPE** (`R26`): `--z-score-x structured --input-space copy --freq-mode feature --attn-ln --flow-num-transforms 3 --tail-bound 5 --d-model 256`
* **CloneMLP-NPE** (`R2`): `--z-score-x structured --input-space copy`
* **DominantClone-NPE** (`D0`): `--require-all-trials  (same 651 test cases as the other encoders)`
