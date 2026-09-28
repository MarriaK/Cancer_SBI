# clonemlp

Headline run: **R2**. Seed members: R2, R2s1, R2s2.

Configuration: `--z-score-x structured --input-space copy`

Why this one: Published model was sharp because over-confident (39/44 SBC failures); R2 trades ~0.05-0.1 R2 (inside seed noise) for honest posteriors.

Each sub-folder holds that run's `metrics_summary.csv`, `metrics_per_arm.csv`, coverage curve, shrinkage table, `summary_arrays.npz`, the TARP curve and figures A-D. Checkpoints and raw posterior dumps are on the cluster at `~/cancer/runs/2026-09-24/<run>/checkpoints/best.pt` and `~/cancer/results/2026-09-24/<run>/posteriors/`.
