#!/usr/bin/env python
"""
Stage 2 of the evaluation: every metric and figure, computed from the files stage 1 saved.

Pure numpy + matplotlib - no torch, no sbi, no GPU - so it runs on the cluster or on a laptop and
can be re-run as often as the wording of a figure changes.

What it produces in --out-dir:
  metrics_per_arm.csv     44 rows per model: true R^2, Pearson r and r^2, RMSE, MAE, z-score
                          summaries, posterior contraction, coverage at 50/90/95 %, SBC KS test
  metrics_summary.csv     one row per model: the pooled numbers, including mean |coverage gap|
  coverage_curve.csv      empirical coverage against nominal level, per model
  fig_A_coverage.{png,pdf}          are the uncertainties honest?
  fig_B_contraction_zscore.{png,pdf} which arms did the model actually learn?
  fig_C_sbc_ecdf.{png,pdf}          simulation-based calibration, one panel per model
  posterior_export.npz    drop-in input for the poster's panels e and g (CloneMLP)

Two identities keep this cheap and exact. Writing u = (rank + 0.5)/(S+1) for the quantile position
of the true value among its own posterior draws:
  - the truth lies inside the central credible interval of level a  <=>  |u - 0.5| <= a/2,
    so the whole coverage curve falls out of the ranks with no sorting;
  - SBC asks exactly that u be Uniform(0,1), so the same array answers both.

    python analyze.py --in-dir out --out-dir results
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# the poster's tokens, inlined so this file has no project dependencies
INK, INK2, MUTED, GRID, AXIS, PANEL = "#0b0b0b", "#52514e", "#5c5b55", "#c4c3ba", "#8a8982", "#f5f4f0"
C_MLP, C_DOM, C_ATT = "#4a3aa7", "#eb6834", "#1baf7a"
# The three "_published" names are the models as published (config.py, 2026-09-25).
# sample_posteriors labels a run with the RESOLVED preset name, so a file lands
# under one of them in two cases: a run made with --published, and a run over a
# checkpoint that carries no effective_config (the cluster's three legacy
# published checkpoints), where the resolver falls back to the published twin.
# The bare names are kept beside them, so `posteriors_clonemlp.npz` files that
# already exist on the cluster are still found. A file for a model missing from
# this list is silently skipped -- which is exactly what happened to ArmToken on
# 2026-09-24.
MODEL_ORDER = ["clonemlp", "cloneatt", "dominantclone", "armtoken", "hybrid",
               "clonemlp_published", "cloneatt_published", "dominantclone_published"]
LABEL = {"clonemlp": "CloneMLP-NPE", "cloneatt": "CloneAtt-NPE", "dominantclone": "DominantClone-NPE", "armtoken": "ArmToken-NPE", "hybrid": "Hybrid-NPE",
         "clonemlp_published": "CloneMLP-NPE (published)", "cloneatt_published": "CloneAtt-NPE (published)", "dominantclone_published": "DominantClone-NPE (published)"}
COLOUR = {"clonemlp": C_MLP, "cloneatt": C_ATT, "dominantclone": C_DOM, "armtoken": "#6a3d9a", "hybrid": "#e31a1c",
          "clonemlp_published": C_MLP, "cloneatt_published": C_ATT, "dominantclone_published": C_DOM}
ARMS = [f"{c}{a}" for c in range(1, 23) for a in "pq"]

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 300,
})


# ------------------------------------------------------------------ metrics
def true_r2(y, pred):
    """Coefficient of determination, 1 - SSE/SST. Penalises bias and shrinkage; can go negative."""
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst == 0 else 1.0 - sse / sst


def pearson_r(y, pred):
    if y.std() == 0 or pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def ks_uniform(u):
    """Two-sided KS statistic of u against Uniform(0,1), with the asymptotic p-value."""
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    if n == 0:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1)
    d = max((i / n - u).max(), (u - (i - 1) / n).max())
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d          # Stephens' small-sample correction
    k = np.arange(1, 101)
    p = 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2))
    return float(d), float(min(max(p, 0.0), 1.0))


def dkw_band(n, alpha=0.05):
    """Half-width of a simultaneous confidence band for an ECDF (Dvoretzky-Kiefer-Wolfowitz)."""
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg: how many of these p-values survive control of the false discovery rate."""
    p = np.sort(np.asarray([x for x in pvals if np.isfinite(x)], dtype=float))
    if p.size == 0:
        return 0
    thresh = alpha * np.arange(1, p.size + 1) / p.size
    hits = np.where(p <= thresh)[0]
    return int(hits[-1] + 1) if hits.size else 0


def load_run(path):
    d = dict(np.load(path, allow_pickle=True))
    d["meta"] = json.loads(str(d.pop("meta_json")))
    s = d["meta"]["num_samples"]
    # the rank lies in {0..S}: S+1 possible outcomes, so u is a genuine probability in (0,1)
    d["u"] = (d["sbc_ranks"].astype(np.float64) + 0.5) / (s + 1.0)
    return d


EPS = 1e-8            # her scripts add this to the posterior sd before dividing; keep it identical


def zscores(run):
    """(theta_true - posterior mean) / posterior sd - her convention, from z-score_violin.py."""
    return (run["theta_true"] - run["post_mean"]) / (run["post_std"] + EPS)


def per_arm_table(run):
    theta, mean, std, u = run["theta_true"], run["post_mean"], run["post_std"], run["u"]
    prior_sd = run["meta"]["prior_sd"]
    z = (theta - mean) / (std + EPS)                           # her convention, from z-score_violin.py
    rows = []
    for j, arm in enumerate(ARMS):
        d, p = ks_uniform(u[:, j])
        rows.append({
            "chromosome_arm": arm,
            "true_r2": true_r2(theta[:, j], mean[:, j]),
            "pearson_r": pearson_r(theta[:, j], mean[:, j]),
            "pearson_r2": pearson_r(theta[:, j], mean[:, j]) ** 2,
            "rmse": float(np.sqrt(((theta[:, j] - mean[:, j]) ** 2).mean())),
            "mae": float(np.abs(theta[:, j] - mean[:, j]).mean()),
            "mean_z": float(z[:, j].mean()),
            "std_z": float(z[:, j].std(ddof=0)),
            "mean_abs_z": float(np.abs(z[:, j]).mean()),
            "contraction": float(1.0 - (std[:, j] ** 2).mean() / prior_sd ** 2),
            "coverage_50": float((np.abs(u[:, j] - 0.5) <= 0.25).mean()),
            "coverage_90": float((np.abs(u[:, j] - 0.5) <= 0.45).mean()),
            "coverage_95": float((np.abs(u[:, j] - 0.5) <= 0.475).mean()),
            "sbc_ks_stat": d,
            "sbc_ks_p": p,
        })
    return rows


def coverage_curve(u, levels):
    """Fraction of true values inside the central credible interval, pooled over cases and arms."""
    flat = u.ravel()
    return np.array([float((np.abs(flat - 0.5) <= a / 2).mean()) for a in levels])


def write_csv(path, rows, fields):
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(
                "" if r[k] is None else (f"{float(r[k]):.6g}"
                                         if isinstance(r[k], (float, np.floating)) else str(r[k]))
                for k in fields) + "\n")


# ------------------------------------------------------------------ figures
def save(fig, base):
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved ->", base + ".png")


def fig_coverage(runs, levels, curves, out_base):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.text(0.63, 0.56, "perfectly calibrated", fontsize=6.6, color=INK2, rotation=39,
            ha="center", va="center")
    for name in MODEL_ORDER:
        if name not in runs:
            continue
        gap = float(np.mean(curves[name] - levels))
        ax.plot(levels, curves[name], color=COLOUR[name], lw=2.0, solid_capstyle="round", zorder=5,
                label=f"{LABEL[name]}  ({gap:+.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("credible level the model claims", fontsize=7.6)
    ax.set_ylabel("fraction of true θ actually inside", fontsize=7.6)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.03, 0.95, "above: intervals too wide", fontsize=6.4, color=MUTED, va="top")
    ax.text(0.97, 0.05, "below: over-confident", fontsize=6.4, color=MUTED, ha="right", va="bottom")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7, title="mean gap to the diagonal")
    plt.setp(leg.get_title(), fontsize=6.4, color=MUTED)
    ax.set_title("Are the uncertainties honest?", fontsize=10, fontweight="bold", color=INK, loc="left", pad=8)
    save(fig, out_base)


def fig_contraction(runs, tables, out_base):
    names = [n for n in MODEL_ORDER if n in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    # Shared x-range driven by the data. Contraction is 1 - postvar/priorvar, so it goes
    # NEGATIVE when a posterior is wider than the prior -- a hardcoded (0, 1) window
    # silently drops those arms off the left edge, which is the whole story for CloneAtt.
    _all = np.concatenate([[r["contraction"] for r in tables[n]] for n in names])
    _all = _all[np.isfinite(_all)]
    x_lo = min(float(_all.min()), 0.0) - 0.05
    x_hi = min(max(float(_all.max()), 0.45) + 0.06, 1.0)
    for ax, name in zip(axes, names):
        rows = tables[name]
        contr = np.array([r["contraction"] for r in rows])
        mz = np.array([r["mean_z"] for r in rows])
        ax.axhspan(-2, 2, color=PANEL, lw=0, zorder=1)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=3)
        if x_lo < 0:
            ax.axvspan(x_lo, 0.0, color="#f6e2e2", lw=0, zorder=2)
            ax.axvline(0.0, color="#b4544f", lw=0.9, ls=":", zorder=4)
        ax.scatter(contr, mz, s=26, fc=COLOUR[name], ec="white", lw=0.6, zorder=6)
        best = int(np.argmax(contr))
        worst = int(np.argmin(contr))
        for j in {best, worst}:
            ax.annotate(rows[j]["chromosome_arm"], (contr[j], mz[j]), textcoords="offset points",
                        xytext=(7, 6), fontsize=6.4, fontweight="bold", color=INK, zorder=9,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.set_title(LABEL[name], fontsize=8.4, fontweight="bold", color=COLOUR[name], loc="left")
        ax.set_xlabel("posterior contraction", fontsize=7.4)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"mean posterior z-score  $(\theta_{true}-\mu)/\sigma$", fontsize=7.4)
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_ylim(-3, 3)
    if x_lo < 0:
        axes[0].text(x_lo + 0.012, 2.72, "posterior wider\nthan the prior", fontsize=6.0,
                     color="#b4544f", ha="left", va="top", fontweight="bold", linespacing=1.25)
    axes[-1].text(x_hi - 0.02, -2.8, "ideal: right, and inside the band", fontsize=6.2,
                  color="#127c56", ha="right", va="bottom", fontweight="bold")
    fig.suptitle("Which arms did the model actually learn?  (one point per chromosome arm)",
                 fontsize=10, fontweight="bold", color=INK, x=0.02, ha="left", y=1.02)
    save(fig, out_base)


def fig_sbc(runs, tables, out_base):
    """One panel per model: the rank ECDF of each arm against uniform, with a band for n = cases.

    The band is drawn per arm, at n = number of held-out tumours. Pooling all 44 arms into one
    test would be wrong: the arms of a single tumour share an encoder and one conditioned flow, so
    they move together, and a band built on N*44 'independent' points rejects calibrated posteriors
    most of the time.
    """
    names = [n for n in MODEL_ORDER if n in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    grid = np.linspace(0, 1, 201)
    for ax, name in zip(axes, names):
        u = runs[name]["u"]
        n_cases = u.shape[0]
        eps = dkw_band(n_cases, alpha=0.05)
        # The band must read as a band, not as background: filling it in GRID (the same
        # colour as the gridlines) at alpha 0.55 made it invisible under 44 coloured lines.
        ax.fill_between(grid, -eps, eps, color="#e4e1d6", lw=0, zorder=1)
        for edge in (-eps, eps):
            ax.plot([0, 1], [edge, edge], ls=(0, (4, 2.5)), color="#6f6e68", lw=1.0, zorder=7)
        ax.annotate(f"95 % band  (n={n_cases})", xy=(0.985, eps), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.0,
                    color="#56554f", fontweight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
        for j in range(u.shape[1]):
            ecdf = np.searchsorted(np.sort(u[:, j]), grid, side="right") / n_cases
            ax.plot(grid, ecdf - grid, color=COLOUR[name], lw=0.7, alpha=0.45, zorder=4)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=3)
        n_rej = int((np.array([r["sbc_ks_p"] for r in tables[name]]) < 0.05).sum())
        n_bh = bh_reject([r["sbc_ks_p"] for r in tables[name]])
        ax.set_title(f"{LABEL[name]}\n{n_rej}/44 arms p<0.05, {n_bh} after FDR",
                     fontsize=7.6, fontweight="bold", color=COLOUR[name], loc="left")
        ax.set_xlabel("normalised rank of the true \u03b8", fontsize=7.4)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("ECDF \u2212 uniform", fontsize=7.4)
    axes[0].set_xlim(0, 1)
    axes[0].text(0.03, 0.92 * axes[0].get_ylim()[1],
                 "one line per arm\nshaded = 95 % simultaneous (DKW) band",
                 fontsize=6.2, color=MUTED, va="top", linespacing=1.4)
    fig.suptitle("Simulation-based calibration: is the true \u03b8 a typical posterior draw?",
                 fontsize=10, fontweight="bold", color=INK, x=0.02, ha="left", y=1.04)
    save(fig, out_base)


# ------------------------------------------------------------------ poster export
def write_poster_export(run, out_path, prefer="sim714"):
    """The array bundle overview_figure/make_overview_figure.py expects for panels e and g."""
    theta, mean, std = run["theta_true"], run["post_mean"], run["post_std"]
    sims = [str(s) for s in run["sim_ids"]]
    if prefer in sims:
        k = sims.index(prefer)
    else:                                     # the median-error tumour, as her export_posterior.py does
        err = np.linalg.norm(theta - mean, axis=1)
        k = int(np.argsort(err)[len(err) // 2])
    q = np.percentile(run["samples"][k], [2.5, 25, 75, 97.5], axis=0).T      # (44, 4)
    np.savez_compressed(out_path, theta_true=theta, post_mean=mean, post_std=std,
                        ex_sim=np.array(sims[k]), ex_theta=theta[k], ex_mean=mean[k], ex_q=q)
    print(f"saved -> {out_path}   example tumour = {sims[k]}")
    return sims[k]


def posterior_path(in_dir, name, run_tag=None):
    """``posteriors_<model>.npz``, or ``posteriors_<model>_<tag>.npz`` for a tagged run."""
    suffix = f"_{run_tag}" if run_tag else ""
    return os.path.join(in_dir, f"posteriors_{name}{suffix}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="out", help="folder holding posteriors_<model>.npz")
    ap.add_argument("--run-tag", default=None,
                    help="read posteriors_<model>_<tag>.npz instead, as written by "
                         "sample_posteriors --run-tag (the 2026-09-24 matrix runs)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--example-sim", default="sim714", help="tumour to feature in the poster export")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs, tables = {}, {}
    for name in MODEL_ORDER:
        path = posterior_path(args.in_dir, name, args.run_tag)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        runs[name] = load_run(path)
        tables[name] = per_arm_table(runs[name])
        m = runs[name]["meta"]
        print(f"loaded {name}: {m['n_cases']} cases x {m['num_samples']} draws "
              f"(checkpoint epoch {m['checkpoint_epoch']})")
    if not runs:
        raise SystemExit(f"no posteriors_*.npz files in {args.in_dir}")

    # ---- per-arm table
    arm_fields = ["model", "chromosome_arm", "true_r2", "pearson_r", "pearson_r2", "rmse", "mae",
                  "mean_z", "std_z", "mean_abs_z", "contraction",
                  "coverage_50", "coverage_90", "coverage_95", "sbc_ks_stat", "sbc_ks_p"]
    arm_rows = []
    for name in runs:
        for r in tables[name]:
            arm_rows.append({"model": name, **r})
    write_csv(os.path.join(args.out_dir, "metrics_per_arm.csv"), arm_rows, arm_fields)

    # ---- coverage curve
    levels = np.linspace(0.0, 1.0, 101)
    curves = {n: coverage_curve(runs[n]["u"], levels) for n in runs}
    cov_rows = [{"model": n, "nominal": float(a), "empirical": float(c)}
                for n in runs for a, c in zip(levels, curves[n])]
    write_csv(os.path.join(args.out_dir, "coverage_curve.csv"), cov_rows,
              ["model", "nominal", "empirical"])

    # ---- pooled summary
    summary_rows = []
    for name in runs:
        run, rows = runs[name], tables[name]
        z = zscores(run)
        col = lambda k: np.array([r[k] for r in rows], dtype=float)
        summary_rows.append({
            "model": name,
            "n_cases": run["meta"]["n_cases"],
            "num_samples": run["meta"]["num_samples"],
            "mean_true_r2": float(np.nanmean(col("true_r2"))),
            "mean_pearson_r2": float(np.nanmean(col("pearson_r2"))),
            "mean_rmse": float(np.nanmean(col("rmse"))),
            "mean_contraction": float(np.nanmean(col("contraction"))),
            "pooled_mean_z": float(z.mean()),
            "pooled_std_z": float(z.std(ddof=0)),
            "pct_within_2z": float((np.abs(z) <= 2).mean() * 100),
            "coverage_50": float((np.abs(run["u"] - 0.5) <= 0.25).mean()),
            "coverage_90": float((np.abs(run["u"] - 0.5) <= 0.45).mean()),
            "coverage_95": float((np.abs(run["u"] - 0.5) <= 0.475).mean()),
            "mean_coverage_gap": float(np.mean(curves[name] - levels)),
            "n_arms_ks_reject_05": int((col("sbc_ks_p") < 0.05).sum()),
            "n_arms_ks_reject_fdr05": bh_reject(col("sbc_ks_p")),
            "mean_log_prob_true": float(np.nanmean(run["log_prob_true"])),
        })
    write_csv(os.path.join(args.out_dir, "metrics_summary.csv"), summary_rows,
              list(summary_rows[0].keys()))

    # ---- figures
    fig_coverage(runs, levels, curves, os.path.join(args.out_dir, "fig_A_coverage"))
    fig_contraction(runs, tables, os.path.join(args.out_dir, "fig_B_contraction_zscore"))
    fig_sbc(runs, tables, os.path.join(args.out_dir, "fig_C_sbc_ecdf"))

    if "clonemlp" in runs:
        m = runs["clonemlp"]["meta"]
        sims = [str(x) for x in runs["clonemlp"]["sim_ids"]]
        if m["n_cases"] != m["n_cases_available"]:
            print(f"[skip] poster export: this run covers {m['n_cases']} of "
                  f"{m['n_cases_available']} cases (a --limit run)")
        elif any(x.startswith("idx") for x in sims):
            print("[skip] poster export: simulation names could not be recovered")
        else:
            write_poster_export(runs["clonemlp"], os.path.join(args.out_dir, "posterior_export.npz"),
                                prefer=args.example_sim)

    print("\n=== summary ===")
    for r in summary_rows:
        print(f"{LABEL[r['model']]:>20}  true R2 {r['mean_true_r2']:.3f}   "
              f"Pearson r2 {r['mean_pearson_r2']:.3f}   RMSE {r['mean_rmse']:.4f}   "
              f"cov90 {r['coverage_90']:.3f}   sd(z) {r['pooled_std_z']:.3f}   "
              f"SBC fail {r['n_arms_ks_reject_fdr05']}/44 arms (FDR)")


if __name__ == "__main__":
    main()
