# armtoken

Headline run: **AT0ens3**. Seed members: AT0, AT0s1, AT0s2.

Configuration: `--model armtoken --tail-bound 5; AT0ens3 = equal-weight mixture of the three seeds' posteriors (jobs/ensemble.sh)`

Why this one: Highest accuracy of the campaign and calibrated; the 3-seed mixture sharpens the mean without over-widening.

Each sub-folder holds that run's `metrics_summary.csv`, `metrics_per_arm.csv`, coverage curve, shrinkage table and figures A-D. Checkpoints and raw posterior dumps are on the cluster at `~/cancer/runs/2026-09-24/<run>/checkpoints/best.pt` and `~/cancer/results/2026-09-24/<run>/posteriors/`.
