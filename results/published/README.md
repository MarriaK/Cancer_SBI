# The published models — the "before"

Evaluation of the three original checkpoints exactly as published, produced on
2026-09-23 before any repair: CloneMLP-NPE (true R² 0.472, 39/44 SBC failures),
CloneAtt-NPE (0.049), DominantClone-NPE (0.170 on 707 test cases — the campaign's
D0 re-evaluates it on the same 651 as the others). Every "published" row in
`docs/CAMPAIGN_REPORT_2026-09-24.md` and in `src/utilities/campaign_summary.py`
reads `metrics_summary.csv` here. Reproduce with `--published`
(`jobs/README.md`, "Defaults changed on 2026-09-25").
