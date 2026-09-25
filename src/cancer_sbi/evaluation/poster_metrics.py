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
  summary_arrays.npz      the per-run array bundle every cross-run script reads (see
                          write_summary_arrays); one file per model when several share a folder
  fig_A_coverage.{png,pdf}          are the uncertainties honest?
  fig_B_contraction_zscore.{png,pdf} which arms did the model actually learn?
  fig_C_sbc_ecdf.{png,pdf}          simulation-based calibration, one panel per model
  fig_S1_recovery_grid.{png,pdf}    supplement: the 44 per-arm recovery scatters
  posterior_export.npz    drop-in input for the poster's panels e and g (CloneMLP), --poster-export

The joint calibration test (TARP) lives next door in ``tarp.py``; ``jobs/analyze.sh`` runs both.

Two identities keep this cheap and exact. Writing u = (rank + 0.5)/(S+1) for the quantile position
of the true value among its own posterior draws:
  - the truth lies inside the central credible interval of level a  <=>  |u - 0.5| <= a/2,
    so the whole coverage curve falls out of the ranks with no sorting;
  - SBC asks exactly that u be Uniform(0,1), so the same array answers both.

    python analyze.py --in-dir out --out-dir results
"""
import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

from cancer_sbi.arms import ARM_LABELS, strip_chr_prefix
from cancer_sbi.evaluation.style import (
    ECDF_BAND_EDGE, ECDF_BAND_FILL, IDENTITY_GREY, OLS_FIT_RED,
)

# the poster's tokens, inlined: the only project imports here are the arm table and the
# colour literals above, neither of which pulls in torch, sbi or a GPU
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
#: The 44 arms in theta order, short spelling. ``cancer_sbi.arms.ARM_LABELS`` is the one
#: table ("chr1p" ... "chr22q"); the CSVs and figures here have always used the
#: un-prefixed form, so the prefix comes off once, explicitly, and never again.
ARMS = list(strip_chr_prefix(ARM_LABELS))

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
    """Half-width of a simultaneous confidence band for an ECDF (Dvoretzky-Kiefer-Wolfowitz).

    Closed form and conservative at every n. Kept as the cheap reference; figure C draws
    :func:`ecdf_band` instead, which is the same quantity computed exactly.
    """
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def ecdf_band(n, gamma=0.95, n_draws=1000, seed=20260925):
    """Half-width of a simultaneous (1-gamma) band for an ECDF of n draws, by simulation.

    The Saeilynoja, Buerkner & Vehtari (2022) recipe: draw ``n_draws`` uniform samples of
    size n, take each one's largest deviation of its ECDF from the uniform CDF, and read off
    the gamma quantile. That largest deviation *is* the two-sided KS statistic, so the band
    drawn here and the p-value :func:`ks_uniform` reports are the same test seen twice - a
    curve leaves the band exactly when its arm rejects at 1-gamma, and :func:`ecdf_difference_curve`
    is what makes that true of the pixels as well as of the arithmetic.

    That agreement is the reason to simulate rather than to keep :func:`dkw_band`. DKW's
    95 % half-width is 1.3581/sqrt(n), the *asymptotic* KS critical value, and it lies
    **above** the exact finite-n quantile at every n - 0.053228 against 0.052965 at
    n = 651, 0.13581 against 0.13403 at n = 100. It is a conservative bound, as its own
    docstring says, so a band drawn from it is slightly too wide and quietly keeps inside
    it arms that the count in the title has already rejected.

    Accuracy of what is returned: 1,000 draws pin the quantile to a few times 1e-4, which
    at n = 651 lands within 1e-5 of both the exact quantile and the D at which
    ``ks_uniform``'s p crosses 0.05. Seeded, so the published band is reproducible.
    """
    rng = np.random.default_rng(seed)
    u = np.sort(rng.random((int(n_draws), int(n))), axis=1)
    i = np.arange(1, int(n) + 1)
    d = np.maximum((i / n - u).max(axis=1), (u - (i - 1) / n).max(axis=1))
    return float(np.quantile(d, gamma))


def ecdf_difference_curve(u):
    """The exact ``ECDF(u) - u`` step function of one arm, as ``(xs, ys)`` ready to plot.

    An ECDF is a step function, and its largest deviation from the uniform CDF is reached
    *at* a jump - so sampling it on an evenly spaced grid systematically understates that
    deviation. On a 201-point grid at n = 651 roughly one arm in a hundred that rejects on
    the exact KS statistic still looks comfortably inside the band, which is the one thing
    figure C must never do: the picture and the count in its title have to be the same
    claim. This returns the polyline with both sides of every jump, so the drawn curve's
    extreme *is* ``ks_uniform(u)[0]``, to floating point.

    The polyline runs (0, 0), then for each sorted u_(i) the left limit (i-1)/n - u_(i) and
    the right limit i/n - u_(i) at the same x, then (1, 0).
    """
    us = np.sort(np.asarray(u, dtype=float))
    n = us.size
    xs = np.empty(2 * n + 2)
    ys = np.empty(2 * n + 2)
    xs[0], ys[0] = 0.0, 0.0
    xs[1:-1:2], ys[1:-1:2] = us, np.arange(n) / n - us              # before each jump
    xs[2:-1:2], ys[2:-1:2] = us, np.arange(1, n + 1) / n - us       # after each jump
    xs[-1], ys[-1] = 1.0, 0.0
    return xs, ys


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


#: Width of one model's panel in figure B, in inches. The canvas is this times the number
#: of models, so a one-model run gets a one-model canvas: before this was a constant, the
#: suptitle ran off a 3 in figure and ``bbox_inches="tight"`` grew the saved PNG to 6.22 in
#: around a 3 in panel, i.e. half the image was the title's shadow.
PANEL_W_B = 3.3

#: Candidate label offsets in typographic points, from the dot outwards. The first pair
#: that collides with nothing wins. ``+`` is right/up; the sign of dx also flips the
#: anchor, so a negative dx hangs the label to the left of its dot.
_LABEL_SLOTS = ((4, 2), (4, -8), (-4, 2), (-4, -8),
                (4, 9), (4, -15), (-4, 9), (-4, -15),
                (0, 14), (0, -20), (11, 2), (-11, 2),
                (4, 17), (4, -24), (-4, 17), (-4, -24),
                (16, 2), (-16, 2), (0, 22), (0, -28))

#: Half-width of the obstacle reserved around each marker, in points. A label that lands on
#: another arm's dot is worse than one an extra line away.
_MARKER_PAD = 3.0


def _label_points(ax, xs, ys, labels, emphasise=(), fontsize=5.0):
    """Annotate every point, each label nudged to the first free slot around its dot.

    44 arms in one panel overlap badly if every label sits up and to the right. This walks
    the points in x order, keeps the display-space rectangles of everything already on the
    axes - the 44 markers, then each label as it is placed - and gives the next label the
    first offset from :data:`_LABEL_SLOTS` that clears them all. The slot order is turned
    round for points near an edge, so a label at the right margin hangs to the left instead
    of off the figure. If every slot collides the furthest is used anyway: a crowded label
    is still information, a missing one is not. No ``adjustText`` dependency - that package
    is not installed on the cluster.

    Returns the list of ``Text`` objects, in the order of ``labels``.
    """
    ax.figure.canvas.draw()                       # fixes transData before we measure
    px_per_pt = ax.figure.dpi / 72.0
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    pad = _MARKER_PAD * px_per_pt
    obstacles = [(p[0] - pad, p[1] - pad, p[0] + pad, p[1] + pad) for p in pts]
    lo_x, hi_x = ax.get_xlim()
    x0_ax = float(ax.transData.transform((lo_x, 0.0))[0])
    x1_ax = float(ax.transData.transform((hi_x, 0.0))[0])
    texts = [None] * len(labels)
    for j in np.argsort(xs):
        text = str(labels[j])
        big = j in emphasise
        size = fontsize + 1.4 if big else fontsize
        w = 0.62 * size * len(text) * px_per_pt   # a rough but stable advance width
        h = 1.15 * size * px_per_pt
        cx, cy = pts[j]
        # near the right edge, prefer the slots that hang left (and vice versa)
        frac = (cx - x0_ax) / max(x1_ax - x0_ax, 1.0)
        slots = _LABEL_SLOTS
        if frac > 0.85:
            slots = sorted(_LABEL_SLOTS, key=lambda s: (s[0] > 0, abs(s[1])))
        elif frac < 0.15:
            slots = sorted(_LABEL_SLOTS, key=lambda s: (s[0] < 0, abs(s[1])))
        for dx, dy in slots:
            bx = cx + dx * px_per_pt - (w if dx < 0 else 0)
            by = cy + dy * px_per_pt
            box = (bx, by, bx + w, by + h)
            if not any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in obstacles):
                break
        obstacles.append(box)
        texts[j] = ax.annotate(
            text, (xs[j], ys[j]), textcoords="offset points",
            xytext=(dx, dy), ha="right" if dx < 0 else "left", va="bottom",
            fontsize=size, color=INK if big else INK2,
            fontweight="bold" if big else "normal", zorder=9,
            path_effects=[pe.withStroke(linewidth=1.9 if big else 1.5, foreground="white")])
    return texts


def fig_contraction(runs, tables, out_base):
    """One panel per model: posterior contraction against mean z, one dot per arm, all labelled.

    Every arm carries its name. The identifiability view is only useful if you can say
    *which* arm sits in the bad corner, and "best and worst only" answered that for two of
    forty-four. The x-range is read off the data with a small margin - the old window
    reserved room out to 0.45 whatever the model did, which squeezed CloneAtt's whole
    cloud into the left third of an otherwise empty panel.
    """
    names = [n for n in MODEL_ORDER if n in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(PANEL_W_B * len(names), 3.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    # Shared x-range driven by the data. Contraction is 1 - postvar/priorvar, so it goes
    # NEGATIVE when a posterior is wider than the prior -- a hardcoded (0, 1) window
    # silently drops those arms off the left edge, which is the whole story for CloneAtt.
    _all = np.concatenate([[r["contraction"] for r in tables[n]] for n in names])
    _all = _all[np.isfinite(_all)]
    lo, hi = float(_all.min()), float(_all.max())
    pad = max(0.09 * (hi - lo), 0.02)              # room for the labels, not for a fixed window
    x_lo = min(lo - pad, 0.0)                      # 0 is the "learned nothing" mark: always in
    x_hi = min(max(hi + pad, 0.0 + pad), 1.0)
    for ax, name in zip(axes, names):
        rows = tables[name]
        contr = np.array([r["contraction"] for r in rows])
        mz = np.array([r["mean_z"] for r in rows])
        ax.axhspan(-2, 2, color=PANEL, lw=0, zorder=1)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=3)
        if x_lo < 0:
            ax.axvspan(x_lo, 0.0, color="#f6e2e2", lw=0, zorder=2)
            ax.axvline(0.0, color="#b4544f", lw=0.9, ls=":", zorder=4)
        ax.scatter(contr, mz, s=18, fc=COLOUR[name], ec="white", lw=0.5, zorder=6)
        ax.set_title(LABEL[name], fontsize=8.4, fontweight="bold", color=COLOUR[name], loc="left")
        ax.set_xlabel("posterior contraction", fontsize=7.4)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"mean posterior z-score  $(\theta_{true}-\mu)/\sigma$", fontsize=7.4)
    axes[0].set_xlim(x_lo, x_hi)
    axes[0].set_ylim(-3, 3)
    # Labels last: the placement reads display coordinates, so the limits must be final.
    for ax, name in zip(axes, names):
        rows = tables[name]
        contr = np.array([r["contraction"] for r in rows])
        mz = np.array([r["mean_z"] for r in rows])
        _label_points(ax, contr, mz, [r["chromosome_arm"] for r in rows],
                      emphasise={int(np.argmax(contr)), int(np.argmin(contr))})
    if x_lo < 0:
        axes[0].text(x_lo + 0.008, 2.86, "posterior wider\nthan the prior", fontsize=6.0,
                     color="#b4544f", ha="left", va="top", fontweight="bold", linespacing=1.25)
    axes[-1].text(x_hi - 0.01, -2.9, "ideal: right, and inside the band", fontsize=6.2,
                  color="#127c56", ha="right", va="bottom", fontweight="bold")
    # Two short lines rather than one long one: the title has to fit inside the narrowest
    # canvas this figure ever gets (one model), or tight bbox pads the image out to it.
    fig.suptitle("Which arms did the model learn?", fontsize=9.5, fontweight="bold",
                 color=INK, x=0.01, ha="left", va="bottom", y=1.055)
    fig.text(0.01, 1.012, "one labelled point per chromosome arm", fontsize=6.6, color=MUTED,
             ha="left", va="bottom")
    save(fig, out_base)


def fig_sbc(runs, tables, out_base, band_draws=1000, band_seed=20260925):
    """One panel per model: each arm's rank ECDF *minus* uniform, with a simultaneous band.

    The ECDF-difference form (BayesFlow's stacked ``calibration_ecdf``, Saeilynoja et al.
    2022). Plotting ECDF(u) - u rather than ECDF(u) against u costs nothing and buys the
    whole vertical range: a 0.05 deviation is a fifth of the panel instead of a line width
    off the diagonal. The difference form is a superset of the old one - the diagonal is
    now the horizontal axis - so only this one is drawn.

    The band is drawn per arm, at n = number of held-out tumours, and comes from
    :func:`ecdf_band`. Pooling all 44 arms into one test would be wrong: the arms of a
    single tumour share an encoder and one conditioned flow, so they move together, and a
    band built on N*44 'independent' points rejects calibrated posteriors most of the time.

    The counts in the title are the same KS test the band is: ``ks_uniform`` per arm, then
    Benjamini-Hochberg over the 44 p-values. Each curve is drawn as the exact step function
    (:func:`ecdf_difference_curve`), not sampled on a grid, so an arm's line crosses the
    band if and only if that arm is one of the rejections the title counts.
    """
    names = [n for n in MODEL_ORDER if n in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        u = runs[name]["u"]
        n_cases = u.shape[0]
        eps = ecdf_band(n_cases, gamma=0.95, n_draws=band_draws, seed=band_seed)
        # The band must read as a band, not as background: filling it in GRID (the same
        # colour as the gridlines) at alpha 0.55 made it invisible under 44 coloured lines.
        ax.axhspan(-eps, eps, color=ECDF_BAND_FILL, lw=0, zorder=1)
        for edge in (-eps, eps):
            ax.plot([0, 1], [edge, edge], ls=(0, (4, 2.5)), color=ECDF_BAND_EDGE, lw=1.0, zorder=7)
        ax.annotate(f"95 % band  (n={n_cases})", xy=(0.985, eps), xytext=(0, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.0,
                    color="#56554f", fontweight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
        for j in range(u.shape[1]):
            xs, ys = ecdf_difference_curve(u[:, j])
            ax.plot(xs, ys, color=COLOUR[name], lw=0.7, alpha=0.45, zorder=4)
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
                 "one line per arm\nshaded = 95 % simultaneous band",
                 fontsize=6.2, color=MUTED, va="top", linespacing=1.4)
    fig.suptitle("Simulation-based calibration: is the true \u03b8 a typical posterior draw?",
                 fontsize=10, fontweight="bold", color=INK, x=0.02, ha="left", y=1.04)
    save(fig, out_base)


def fig_recovery_grid(run, out_base, nrows=4, ncols=11, model=None):
    """Supplement S1: one recovery scatter per arm, truth on x and posterior mean on y.

    44 panels, 4 x 11, so the bottom row of every column carries tick labels - the 6 x 8
    version left the last four cells empty and with them the tick labels of the two
    rightmost columns. Limits are per panel: a global window makes 43 panels a study of
    the one arm with the widest truth range.

    Each title carries both numbers, BayesFlow-2 style: the coefficient of determination
    ``R2 = 1 - SSE/SST``, which is what a reader means by "how much of this arm did we
    recover", and the *unsquared* Pearson r. They differ exactly by shrinkage and bias -
    r is blind to both, and r^2 labelled R^2 is the one real defect the audit found in the
    older grid. R2 <= r^2 always, because r^2 is the best R2 any affine rescaling of the
    posterior mean could reach.

    Returns one dict per arm (``chromosome_arm``, ``true_r2``, ``pearson_r``, ``slope``,
    ``intercept``, ``n``) so a test can check the numbers without reading the pixels.
    """
    theta, mean = np.asarray(run["theta_true"]), np.asarray(run["post_mean"])
    n_arms = theta.shape[1]
    colour = COLOUR.get(model, C_MLP)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.42 * ncols, 1.52 * nrows))
    axes = np.atleast_2d(axes)
    stats = []
    for j in range(n_arms):
        ax = axes[j // ncols, j % ncols]
        x, y = theta[:, j], mean[:, j]
        r2, r = true_r2(x, y), pearson_r(x, y)
        slope, intercept = (np.polyfit(x, y, 1) if x.std() > 0 else (np.nan, np.nan))
        stats.append({"chromosome_arm": ARMS[j] if j < len(ARMS) else str(j),
                      "true_r2": float(r2), "pearson_r": float(r),
                      "slope": float(slope), "intercept": float(intercept), "n": int(x.size)})
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = max(0.04 * (hi - lo), 1e-6)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color=IDENTITY_GREY, lw=0.8, ls=(0, (3, 2)), zorder=3)
        ax.scatter(x, y, s=3.0, fc=colour, ec="none", alpha=0.45, zorder=4)
        if np.isfinite(slope):
            xs = np.array([lo, hi])
            ax.plot(xs, intercept + slope * xs, color=OLS_FIT_RED, lw=0.9, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{stats[-1]['chromosome_arm']}   R\u00b2 = {r2:.2f}   r = {r:.2f}",
                     fontsize=5.6, color=INK, pad=2.0)
        # three ticks a panel: at 1.4 in wide, five labels of "-0.25" run into each other
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.tick_params(labelsize=4.8, length=2, pad=1.2)
        ax.grid(color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j // ncols != nrows - 1:
            ax.set_xticklabels([])
        if j % ncols != 0:
            ax.set_yticklabels([])
    for k in range(n_arms, nrows * ncols):            # only if a future grid is not exact
        axes[k // ncols, k % ncols].axis("off")
    fig.supxlabel("true selection coefficient", fontsize=7.4, color=INK2)
    fig.supylabel("posterior mean", fontsize=7.4, color=INK2)
    fig.tight_layout()
    title = "S1  Per-arm recovery" + (f" - {LABEL.get(model, model)}" if model else "")
    # above the canvas, as in figure B: tight bbox takes them in, and nothing is laid out
    # underneath them, so the two lines cannot collide with the first row of panels
    fig.suptitle(title, fontsize=9.5, fontweight="bold", color=INK, x=0.004, ha="left",
                 va="bottom", y=1.035)
    fig.text(0.004, 1.008, "dashed = identity, solid = least squares; "
                           "R\u00b2 = 1 \u2212 SSE/SST, r = Pearson (unsquared)",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")
    save(fig, out_base)
    return stats


# ------------------------------------------------------------------ per-run export
#: Every key of ``summary_arrays.npz``, with its dtype. This is the contract the cross-run
#: stage reads, so it is written down once and asserted against in the tests.
SUMMARY_ARRAY_KEYS = ("theta_true", "post_mean", "post_std", "u",
                      "sim_ids", "arms", "model", "run_tag", "meta_json")


def summary_arrays_name(model=None):
    """``summary_arrays.npz``, or ``summary_arrays_<model>.npz`` when a folder holds several.

    One glob - ``summary_arrays*.npz`` - finds exactly one file per model either way.
    """
    return "summary_arrays.npz" if model is None else f"summary_arrays_{model}.npz"


def write_summary_arrays(run, out_path, model):
    """The whole per-run evaluation as nine arrays, about a megabyte, no ``samples``.

    What every cross-run script needs and nothing else: the truth, the posterior's first
    two moments, and ``u`` - the normalised rank, byte-identical to what :func:`load_run`
    computes, because a re-derivation downstream is a second definition waiting to drift.
    The 5,000 draws stay in the stage-1 file; they are 573 MB per run and no cross-run
    figure has ever needed them.

    This replaces the CloneMLP-only ``posterior_export.npz``, which carried three of these
    arrays for one hard-coded model and one hand-picked tumour.
    """
    meta = run["meta"]
    np.savez_compressed(
        out_path,
        theta_true=np.asarray(run["theta_true"], dtype=np.float32),
        post_mean=np.asarray(run["post_mean"], dtype=np.float32),
        post_std=np.asarray(run["post_std"], dtype=np.float32),
        u=np.asarray(run["u"], dtype=np.float64),
        sim_ids=np.array([str(s) for s in run["sim_ids"]]),
        arms=np.array(ARMS[:run["theta_true"].shape[1]]),
        model=np.array(str(model)),
        run_tag=np.array(str(meta.get("run_tag") or "")),
        meta_json=np.array(json.dumps(meta)),
    )
    print(f"saved -> {out_path}   ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


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


def unknown_models(in_dir, run_tag=None, known=MODEL_ORDER):
    """Files that look like stage-1 output but name a model :data:`MODEL_ORDER` has never heard of.

    The fixed allow-list is what silently dropped ArmToken on 2026-09-24: the file was
    there, the loop never looked at it, and the run was sampled for hours and then never
    measured. A missing model is now loud.

    Returns the offending stems, e.g. ``["armtoken"]``, in filename order.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(in_dir, "posteriors_*.npz"))):
        stem = os.path.basename(path)[len("posteriors_"):-len(".npz")]
        if run_tag and stem.endswith(f"_{run_tag}"):
            stem = stem[: -len(run_tag) - 1]
        # a known name either is the stem or is followed by _<something>: a tag, a
        # --limit marker or a partition, none of which change which model it is
        if any(stem == k or stem.startswith(k + "_") for k in known):
            continue
        out.append(stem)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="out", help="folder holding posteriors_<model>.npz")
    ap.add_argument("--run-tag", default=None,
                    help="read posteriors_<model>_<tag>.npz instead, as written by "
                         "sample_posteriors --run-tag (the 2026-09-24 matrix runs)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--example-sim", default="sim714", help="tumour to feature in the poster export")
    ap.add_argument("--poster-export", action="store_true",
                    help="also write the CloneMLP-only posterior_export.npz the 2026 poster's "
                         "panels e and g read; summary_arrays.npz replaces it for everything else")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for stem in unknown_models(args.in_dir, args.run_tag):
        print(f"[warn] posteriors_{stem}.npz present but {stem} not in MODEL_ORDER "
              f"-- that run will not be measured")
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

    # ---- the per-run array bundle the cross-run stage reads
    multi = len(runs) > 1
    for name in runs:
        write_summary_arrays(runs[name],
                             os.path.join(args.out_dir, summary_arrays_name(name if multi else None)),
                             name)

    # ---- figures
    fig_coverage(runs, levels, curves, os.path.join(args.out_dir, "fig_A_coverage"))
    fig_contraction(runs, tables, os.path.join(args.out_dir, "fig_B_contraction_zscore"))
    fig_sbc(runs, tables, os.path.join(args.out_dir, "fig_C_sbc_ecdf"))
    for name in runs:
        suffix = f"_{name}" if multi else ""
        fig_recovery_grid(runs[name],
                          os.path.join(args.out_dir, f"fig_S1_recovery_grid{suffix}"), model=name)

    if args.poster_export and "clonemlp" in runs:
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
