#!/usr/bin/env python
"""Assemble results/best_honest/: the best calibrated configuration of each encoder.

    python utilities/collect_best_honest.py --results ../results

"Best honest" = the run family the 2026-09-24 campaign recommends for each
encoder once calibration is required (docs/CAMPAIGN_REPORT_2026-09-24.md, §4 and
§6): not the highest single R², but the configuration whose posteriors are
honest and whose accuracy is quoted with its seed spread. The raw posterior
dumps (530 MB each) stay on the cluster; everything else -- metrics, per-arm
tables, coverage curves, figures A-D -- is copied here, one folder per encoder,
one sub-folder per run, plus a README with the comparison table generated from
the copied files. Re-run after any of these runs is re-evaluated.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import statistics as st
from pathlib import Path

# encoder -> (headline run, [member runs], configuration, one-line rationale)
BEST = {
    "armtoken": (
        "AT0ens3", ["AT0", "AT0s1", "AT0s2"],
        "--model armtoken --tail-bound 5; AT0ens3 = equal-weight mixture of the three seeds' posteriors (jobs/ensemble.sh)",
        "Highest accuracy of the campaign and calibrated; the 3-seed mixture sharpens the mean without over-widening.",
    ),
    "cloneatt": (
        "R26", ["R26", "R26s1"],
        "--z-score-x structured --input-space copy --freq-mode feature --attn-ln --flow-num-transforms 3 --tail-bound 5 --d-model 256",
        "Repaired attention model (published 0.049); slightly over-confident (cov95 0.93) -- R27rc is the recalibrated variant.",
    ),
    "clonemlp": (
        "R2", ["R2", "R2s1", "R2s2"],
        "--z-score-x structured --input-space copy",
        "Published model was sharp because over-confident (39/44 SBC failures); R2 trades ~0.05-0.1 R2 (inside seed noise) for honest posteriors.",
    ),
    "dominantclone": (
        "D0", ["D0", "D0s1", "D0s2"],
        "--require-all-trials  (same 651 test cases as the other encoders)",
        "Baseline: the largest clone only. Already calibrated; no encoder repair was ever applied to DeepSet.",
    ),
}
COPY_GLOBS = ["metrics_summary.csv", "metrics_per_arm.csv", "coverage_curve.csv", "shrinkage.csv", "fig_*.png", "fig_*.pdf"]


def read_summary(run_dir: Path) -> dict:
    with open(run_dir / "metrics_summary.csv") as fh:
        return list(csv.DictReader(fh))[-1]


def fam(values):
    return (st.mean(values), st.stdev(values) if len(values) > 1 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="../results")
    ap.add_argument("--campaign", default="2026-09-24")
    ap.add_argument("--out", default="best_honest")
    args = ap.parse_args()
    root = Path(args.results); src = root / args.campaign; out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Best honest configuration per encoder", "",
             f"Assembled by `src/utilities/collect_best_honest.py` from `results/{args.campaign}/`.",
             "All runs are evaluated on the same 651 held-out simulations with 5,000 posterior draws each.",
             "\"Honest\" means calibrated posteriors first, accuracy quoted with its seed spread; see",
             "`docs/CAMPAIGN_REPORT_2026-09-24.md` (§4, §6) for the full campaign.", "",
             "| Encoder | Headline run | true R² (headline) | true R² (seeds, mean ± sd) | log p(θ*) | 95 % coverage | SBC failures /44 | Configuration |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for enc, (head, members, cfg, why) in BEST.items():
        enc_dir = out / enc; enc_dir.mkdir(exist_ok=True)
        rows = {}
        for run in dict.fromkeys([head] + members):
            d = src / run
            if not (d / "metrics_summary.csv").exists():
                raise SystemExit(f"missing {d}/metrics_summary.csv")
            dst = enc_dir / run; dst.mkdir(exist_ok=True)
            for pat in COPY_GLOBS:
                for f in d.glob(pat):
                    shutil.copy2(f, dst / f.name)
            rows[run] = read_summary(d)
        h = rows[head]
        r2 = fam([float(rows[m]["mean_true_r2"]) for m in members])
        lines.append("| %s | %s | %.3f | %.3f ± %.3f (n=%d) | %.1f | %.3f | %s | `%s` |" % (
            enc, head, float(h["mean_true_r2"]), r2[0], r2[1], len(members), float(h["mean_log_prob_true"]),
            float(h["coverage_95"]), h["n_arms_ks_reject_fdr05"], cfg))
        (enc_dir / "README.md").write_text(
            f"# {enc}\n\nHeadline run: **{head}**. Seed members: {', '.join(members)}.\n\n"
            f"Configuration: `{cfg}`\n\nWhy this one: {why}\n\n"
            "Each sub-folder holds that run's `metrics_summary.csv`, `metrics_per_arm.csv`, coverage curve, "
            "shrinkage table and figures A-D. Checkpoints and raw posterior dumps are on the cluster at "
            f"`~/cancer/runs/{args.campaign}/<run>/checkpoints/best.pt` and `~/cancer/results/{args.campaign}/<run>/posteriors/`.\n")
    lines += ["", "Ranking on identical data: ArmToken ≫ CloneAtt > CloneMLP ≫ DominantClone. Differences in R² below ~0.05",
              "between the two clone-set models are within their seed noise; ArmToken's seed band is ±0.002.", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(f"wrote {out}/README.md and {sum(1 for _ in out.rglob('*') if _.is_file())} files")


if __name__ == "__main__":
    main()
