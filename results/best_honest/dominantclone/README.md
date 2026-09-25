# dominantclone

Headline run: **D0**. Seed members: D0, D0s1, D0s2.

Configuration: `--require-all-trials  (same 651 test cases as the other encoders)`

Why this one: Baseline: the largest clone only. Already calibrated; no encoder repair was ever applied to DeepSet.

Each sub-folder holds that run's `metrics_summary.csv`, `metrics_per_arm.csv`, coverage curve, shrinkage table, `summary_arrays.npz`, the TARP curve and figures A-D. Checkpoints and raw posterior dumps are on the cluster at `~/cancer/runs/2026-09-24/<run>/checkpoints/best.pt` and `~/cancer/results/2026-09-24/<run>/posteriors/`.
