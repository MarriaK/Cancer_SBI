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

Which runs those are is NOT decided here: it is read from
``cancer_sbi/evaluation/manifests/best_honest_2026-09-24.json`` (``--manifest``),
the same file ``cancer_sbi.evaluation.best_honest_figures`` and ``jobs/best_honest.sh``
read, so a new seed is added in one place.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics as st
from pathlib import Path

#: The seed families live in one manifest, read by this script, by
#: ``cancer_sbi.evaluation.best_honest_figures`` and by ``jobs/best_honest.sh``.
#: Editing the list here used to mean editing it in three places.
DEFAULT_MANIFEST = (Path(__file__).resolve().parents[1] / "cancer_sbi" / "evaluation" /
                    "manifests" / "best_honest_2026-09-24.json")


def load_best(manifest_path=DEFAULT_MANIFEST):
    """encoder -> (headline run, [member runs], configuration, one-line rationale)."""
    with open(manifest_path) as fh:
        man = json.load(fh)
    return {e["key"]: (e["headline"], list(e["members"]), e["config"], e["why"])
            for e in man["encoders"]}


COPY_GLOBS = ["metrics_summary.csv", "metrics_per_arm.csv", "coverage_curve.csv", "shrinkage.csv",
              # stage-2 arrays and the joint-calibration test (the 2026-09-25 redesign).
              # The arrays are summary_arrays.npz for a one-model run and
              # summary_arrays_<model>.npz where several models share an out-dir.
              "summary_arrays*.npz", "tarp_curve.csv", "tarp_summary.json",
              "fig_*.png", "fig_*.pdf", "fig_S1_*", "fig_T_*"]


def load_models(manifest_path=DEFAULT_MANIFEST):
    """encoder key -> preset model name, for the files that are written per model."""
    with open(manifest_path) as fh:
        return {e["key"]: e["model"] for e in json.load(fh)["encoders"]}


def tarp_numbers(run_dir: Path, model: str | None = None):
    """``(distance, gap, verdict)`` from this model's TARP summary, or three Nones.

    ``atc`` is the mean ``|ecp - alpha|``: a distance, never negative, so it is printed
    without a sign. The side of the diagonal lives in ``atc_signed``, a sum over the
    upper half of the alpha grid; scaled back to a mean gap it is the direction
    ``tarp.py``'s own CLI prints -- positive = too wide, negative = over-confident.
    ``tarp_summary.json`` is a list, one object per model in that out-dir, and an entry
    for another model is never used for this one.
    """
    path = run_dir / "tarp_summary.json"
    if not path.exists():
        return None, None, None
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        mine = [s for s in payload if model is None or s.get("model") == model]
        if not mine:
            found = ", ".join(sorted(str(s.get("model")) for s in payload)) or "nothing"
            print(f"[warn] {run_dir.name}/tarp_summary.json: no entry for model {model}; "
                  f"found {found}")
            return None, None, None
        payload = mine[0]
    atc = payload.get("atc")
    if payload.get("atc_signed") is None:
        return atc, None, None
    n_alpha = int(payload.get("n_alpha") or 50)
    gap = float(payload["atc_signed"]) / max(n_alpha - n_alpha // 2, 1)
    verdict = ("on the diagonal" if abs(gap) < 0.01 else
               "too wide" if gap > 0 else "over-confident")
    return atc, gap, verdict


def read_summary(run_dir: Path, model: str | None = None) -> dict:
    """This model's row of ``metrics_summary.csv``.

    An out-dir can hold several models (``results/published/``), so taking the last row
    would quietly put one encoder's numbers under another's name. A file with no
    ``model`` column comes from a run that held one model, so its last row is the row.
    """
    with open(run_dir / "metrics_summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{run_dir}/metrics_summary.csv is empty")
    if model is None or "model" not in rows[0]:
        return rows[-1]
    mine = [r for r in rows if r["model"] == model]
    if not mine:
        found = ", ".join(sorted({r["model"] for r in rows})) or "nothing"
        raise SystemExit(f"{run_dir}/metrics_summary.csv has no row for model {model}; "
                         f"found {found}")
    return mine[-1]


def fam(values):
    return (st.mean(values), st.stdev(values) if len(values) > 1 else float("nan"))


def describe_scale(rows) -> str:
    """The "same N simulations, M draws" sentence, read off the rows rather than typed."""
    def span(key):
        vals = sorted({int(float(r[key])) for r in rows if r.get(key)})
        return vals or None
    cases, draws = span("n_cases"), span("num_samples")
    if not cases:
        return ""
    if len(cases) == 1:
        text = f"All runs are evaluated on the same {cases[0]:,} held-out simulations"
    else:
        text = f"Runs cover {cases[0]:,}-{cases[-1]:,} held-out simulations"
    if draws:
        text += (f" with {draws[0]:,} posterior draws each." if len(draws) == 1
                 else f" with {draws[0]:,}-{draws[-1]:,} posterior draws each.")
    else:
        text += "."
    return text


def describe_ranking(headline_r2: dict, bands: dict) -> list:
    """The ranking and seed-band sentences, in the order the numbers put them."""
    if not headline_r2:
        return []
    order = sorted(headline_r2, key=headline_r2.get, reverse=True)
    chain = " > ".join(f"{k} {headline_r2[k]:.3f}" for k in order)
    out = [f"Ranking on identical data, by headline true R\u00b2: {chain}."]
    finite = {k: v for k, v in bands.items() if v == v}
    if finite:
        lo = min(finite, key=finite.get); hi = max(finite, key=finite.get)
        out.append(f"Seed bands (sd over the members) run from \u00b1{finite[lo]:.3f} ({lo}) to "
                   f"\u00b1{finite[hi]:.3f} ({hi}); a difference smaller than the wider band is noise.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="../results")
    ap.add_argument("--campaign", default="2026-09-24")
    ap.add_argument("--out", default="best_honest")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="seed-family manifest (default: the one under cancer_sbi/evaluation/)")
    args = ap.parse_args()
    best = load_best(args.manifest)
    models = load_models(args.manifest)
    root = Path(args.results); src = root / args.campaign; out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    table = ["| Encoder | Headline run | true R\u00b2 (headline) | true R\u00b2 (seeds, mean \u00b1 sd) | log p(\u03b8*) | 95 % coverage | SBC failures /44 | TARP dist. | TARP direction | Configuration |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    seen_rows, bands, headline_r2 = [], {}, {}
    for enc, (head, members, cfg, why) in best.items():
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
            rows[run] = read_summary(d, models.get(enc))
        seen_rows += list(rows.values())
        h = rows[head]
        r2 = fam([float(rows[m]["mean_true_r2"]) for m in members])
        bands[enc] = r2[1]
        headline_r2[enc] = float(h["mean_true_r2"])
        atc, gap, verdict = tarp_numbers(src / head, models.get(enc))
        table.append("| %s | %s | %.3f | %.3f \u00b1 %.3f (n=%d) | %.1f | %.3f | %s | %s | %s | `%s` |" % (
            enc, head, float(h["mean_true_r2"]), r2[0], r2[1], len(members), float(h["mean_log_prob_true"]),
            float(h["coverage_95"]), h["n_arms_ks_reject_fdr05"],
            "\u2014" if atc is None else f"{float(atc):.3f}",
            "\u2014" if gap is None else f"{gap:+.3f} ({verdict})", cfg))
        (enc_dir / "README.md").write_text(
            f"# {enc}\n\nHeadline run: **{head}**. Seed members: {', '.join(members)}.\n\n"
            f"Configuration: `{cfg}`\n\nWhy this one: {why}\n\n"
            "Each sub-folder holds that run's `metrics_summary.csv`, `metrics_per_arm.csv`, coverage curve, "
            "shrinkage table, `summary_arrays.npz`, the TARP curve and figures A-D. "
            "Checkpoints and raw posterior dumps are on the cluster at "
            f"`~/cancer/runs/{args.campaign}/<run>/checkpoints/best.pt` and `~/cancer/results/{args.campaign}/<run>/posteriors/`.\n")

    lines = ["# Best honest configuration per encoder", "",
             f"Assembled by `src/utilities/collect_best_honest.py` from `results/{args.campaign}/`."]
    scale = describe_scale(seen_rows)
    if scale:
        lines.append(scale)
    lines += ["\"Honest\" means calibrated posteriors first, accuracy quoted with its seed spread; see",
              "`docs/CAMPAIGN_REPORT_2026-09-24.md` (\u00a74, \u00a76) for the full campaign.", "",
              "TARP dist. is the mean |gap| to the diagonal (a distance); TARP direction is that gap with",
              "its sign: positive = the posterior is too wide, negative = over-confident.", ""]
    lines += table
    lines += ["", *describe_ranking(headline_r2, bands), ""]
    (out / "README.md").write_text("\n".join(lines))
    print(f"wrote {out}/README.md and {sum(1 for _ in out.rglob('*') if _.is_file())} files")


if __name__ == "__main__":
    main()
