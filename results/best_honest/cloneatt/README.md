# cloneatt

Headline run: **R26**. Seed members: R26, R26s1.

Configuration: `--z-score-x structured --input-space copy --freq-mode feature --attn-ln --flow-num-transforms 3 --tail-bound 5 --d-model 256`

Why this one: Repaired attention model (published 0.049); slightly over-confident (cov95 0.93) -- R27rc is the recalibrated variant.

Each sub-folder holds that run's `metrics_summary.csv`, `metrics_per_arm.csv`, coverage curve, shrinkage table and figures A-D. Checkpoints and raw posterior dumps are on the cluster at `~/cancer/runs/2026-09-24/<run>/checkpoints/best.pt` and `~/cancer/results/2026-09-24/<run>/posteriors/`.
