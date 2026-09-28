#!/usr/bin/env python
"""Stage 3 of the evaluation: the cross-run figures of the best-honest campaign.

Stages 1 and 2 are per run -- sample the posterior, then turn one run's draws into
``metrics_per_arm.csv``, ``metrics_summary.csv``, ``summary_arrays.npz`` and
``tarp_curve.csv``. Nothing there can show a seed band or put two encoders side by
side, because nothing there sees more than one run. This module is the stage that
reads *all* the runs named by a manifest and draws the comparisons:

  fig_E_per_arm_r2        the headline: per-arm true R^2, 44 arms on the x-axis,
                          one line per encoder, error bars = sd over the seeds
  fig_S3_per_arm_coverage the same strip for 95 % coverage, with the binomial band
  fig_calibration_panel   stacked SBC rank-ECDF *difference* per encoder with a
                          simulated simultaneous band, plus the TARP curves
  fig_D_panel             the pooled truth-vs-posterior-mean hexbins, side by side
  per_arm_summary.csv/.md the S2 table (mean, sd, n per arm per encoder)
  headline_table.md       one row per encoder, the numbers the poster quotes

Why one panel and not 44: ``docs/EVALUATION_PRACTICE_REVIEW.md`` sections 4-5. Per-arm
*accuracy* is 3-15x larger than the seed noise and carries a real result, so it gets
an axis; per-arm *calibration* does not survive a re-seed, so it stays aggregate.

The metric is always the coefficient of determination, ``1 - SSE/SST``, never the
squared Pearson correlation.

Every input is optional except ``metrics_per_arm.csv``/``metrics_summary.csv``: a
missing ``summary_arrays.npz`` or ``tarp_curve.csv`` prints ``[warn]`` and skips that
panel, so this runs unchanged on the CSV-only copies under ``results/best_honest/``.

Every input may also hold several models at once -- the per-run stage writes
``summary_arrays_<model>.npz`` beside its plain name, a ``model`` column in
``metrics_*.csv`` and ``tarp_curve.csv``, and a list in ``tarp_summary.json`` -- so
each file is filtered down to the model the manifest names for that encoder.

    python -m cancer_sbi.evaluation.best_honest_figures \
        --results ../results/2026-09-24 \
        --manifest cancer_sbi/evaluation/manifests/best_honest_2026-09-24.json \
        --out-dir ../results/best_honest
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

#: Default manifest, shipped beside this module.
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "manifests" / "best_honest_2026-09-24.json"

#: The 44 arms in genomic order -- the order of every 44-vector in this project.
ARMS = [f"{c}{a}" for c in range(1, 23) for a in "pq"]

#: Acrocentric p-arms. They carry almost no unique sequence, so every encoder is
#: guessing there; shading them stops a reader reading that dip as a model defect.
ACROCENTRIC = ("13p", "14p", "15p", "21p", "22p")

#: The per-arm statistics that go into the S2 table, in the order they are written.
SUMMARY_STATS = ("true_r2", "pearson_r2", "rmse", "contraction", "coverage_95", "sbc_ks_p")

# Poster ink, as in poster_metrics.py -- inlined so this module has no project imports.
INK, INK2, MUTED, GRID, AXIS, PANEL = "#0b0b0b", "#52514e", "#5c5b55", "#d7d6cf", "#8a8982", "#f0eee6"

RC = {
    "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 300,
}


def warn(msg):
    print(f"[warn] {msg}")


def minus(value, fmt=".3f"):
    """Format a number with a typographic minus, not a hyphen."""
    return format(float(value), fmt).replace("-", "\u2212")


#: Keys already warned about: two figures ask for the same missing file.
_WARNED: set = set()


def warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warn(msg)


# --------------------------------------------------------------------- manifest
def load_manifest(path=DEFAULT_MANIFEST):
    """Read the seed-family manifest. Returns the parsed dict, validated."""
    with open(path) as fh:
        man = json.load(fh)
    if not man.get("encoders"):
        raise SystemExit(f"{path}: no encoders")
    seen = set()
    for enc in man["encoders"]:
        for key in ("key", "label", "model", "headline", "members", "config", "colour"):
            if key not in enc:
                raise SystemExit(f"{path}: encoder {enc.get('key', '?')} has no '{key}'")
        if enc["key"] in seen:
            raise SystemExit(f"{path}: duplicate encoder key {enc['key']}")
        seen.add(enc["key"])
    return man


def manifest_runs(man):
    """Every run the manifest names, headline first, de-duplicated, in order."""
    runs = [r for enc in man["encoders"] for r in [enc["headline"], *enc["members"]]]
    return list(dict.fromkeys(runs))


# ------------------------------------------------------------------ small stats
def rankdata(x):
    """Ranks with ties averaged (scipy.stats.rankdata, in five lines of numpy)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average the ranks inside each tie group
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rho over the pairs where both are finite."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg: how many of these p-values survive FDR control at alpha."""
    p = np.sort(np.asarray([x for x in pvals if np.isfinite(x)], dtype=float))
    if p.size == 0:
        return 0
    hits = np.where(p <= alpha * np.arange(1, p.size + 1) / p.size)[0]
    return int(hits[-1] + 1) if hits.size else 0


def true_r2(y, pred):
    """Coefficient of determination, 1 - SSE/SST. Penalises bias and shrinkage."""
    y, pred = np.asarray(y, dtype=float).ravel(), np.asarray(pred, dtype=float).ravel()
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst == 0 else float(1.0 - ((y - pred) ** 2).sum() / sst)


def pearson_r2(y, pred):
    y, pred = np.asarray(y, dtype=float).ravel(), np.asarray(pred, dtype=float).ravel()
    if y.std() == 0 or pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1] ** 2)


def ecdf_difference(u, grid):
    """ECDF of ``u`` minus the uniform CDF, evaluated on ``grid``."""
    u = np.sort(np.asarray(u, dtype=float))
    return np.searchsorted(u, grid, side="right") / max(u.size, 1) - grid


def ks_statistic(u):
    """sup |ECDF(u) - u|, evaluated exactly at the jumps rather than on a grid."""
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max((i / n - u).max(), (u - (i - 1) / n).max()))


def simultaneous_band(n, n_draws=1000, alpha=0.05, seed=20260925):
    """Half-width of a simultaneous 95 % band for an ECDF-difference of n uniforms.

    Simulated rather than taken from the DKW inequality: DKW is conservative, and
    Saeilynoja et al. 2022 recommend the sampled null distribution of the maximum
    deviation. Seeded, so the band is the same number every time the figure is redrawn.

    Deliberately a local re-implementation -- this module reads files, not other
    modules -- but it is the same statistic with the same defaults as
    ``poster_metrics.ecdf_band`` (gamma 0.95, 1,000 draws, seed 20260925), so the
    per-run figure C and this cross-run panel draw the identical band, and a test
    pins that. The band is per arm, at n = held-out tumours:
    pooling the 44 arms would reject calibrated posteriors, because the arms of one
    tumour share an encoder and one conditioned flow and move together.
    """
    rng = np.random.default_rng(seed)
    dev = np.array([ks_statistic(rng.random(n)) for _ in range(n_draws)])
    return float(np.quantile(dev, 1.0 - alpha))


# --------------------------------------------------------------------- file I/O
def run_dir(results_root, enc, run):
    """Locate ``<run>/`` under a campaign root or under a per-encoder tree.

    ``--results ../results/2026-09-24`` holds ``<run>/`` directly; the collected copy
    ``results/best_honest/`` nests them as ``<encoder>/<run>/``. Both are accepted so
    the figures can be rebuilt from either.
    """
    root = Path(results_root)
    for cand in (root / run, root / enc["key"] / run):
        if (cand / "metrics_per_arm.csv").exists() or (cand / "metrics_summary.csv").exists():
            return cand
    return None


def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _model_rows(rows, model, where=""):
    """Only this model's rows -- never another model's, and never silently.

    A file with no ``model`` column was written by a run that held one model, so all of
    it belongs to that run. A file that *has* the column and does not have this model is
    a file about something else: it warns and yields nothing, because the alternative
    (fall back to whatever rows are there) puts one encoder's numbers under another
    encoder's name on the poster.
    """
    if not rows or "model" not in rows[0]:
        return rows
    mine = [r for r in rows if r["model"] == model]
    if not mine:
        found = ", ".join(sorted({r["model"] for r in rows})) or "nothing"
        warn_once(("model", where, model),
                  f"{where}: no rows for model {model}; found {found}")
    return mine


def per_arm_values(results_root, enc, run, stat):
    """One run's per-arm ``stat`` as a 44-vector in genomic order (nan where absent)."""
    d = run_dir(results_root, enc, run)
    if d is None or not (d / "metrics_per_arm.csv").exists():
        return None
    rows = _model_rows(read_rows(d / "metrics_per_arm.csv"), enc["model"],
                       f"{run}/metrics_per_arm.csv")
    by_arm = {}
    for r in rows:
        if stat in r and r[stat] not in ("", None):
            try:
                by_arm[r["chromosome_arm"]] = float(r[stat])
            except ValueError:
                pass
    return np.array([by_arm.get(a, np.nan) for a in ARMS], dtype=float)


def summary_row(results_root, enc, run):
    """The last row of one run's ``metrics_summary.csv``, or None."""
    d = run_dir(results_root, enc, run)
    if d is None or not (d / "metrics_summary.csv").exists():
        return None
    rows = _model_rows(read_rows(d / "metrics_summary.csv"), enc["model"],
                       f"{run}/metrics_summary.csv")
    return rows[-1] if rows else None


def seed_family(results_root, enc, stat):
    """(mean, sd, n) over the member seeds, each a 44-vector.

    sd is the sample sd (ddof=1) and is nan where fewer than two seeds carry the arm,
    which is how a single-run encoder ends up with no error bars.
    """
    stack = []
    for run in enc["members"]:
        v = per_arm_values(results_root, enc, run, stat)
        if v is None:
            warn(f"{enc['key']}: no metrics_per_arm.csv for member {run}")
            continue
        stack.append(v)
    if not stack:
        return None
    a = np.vstack(stack)
    n = np.isfinite(a).sum(axis=0)
    # an arm can be all-NaN (a run whose model is not in the file): nanmean says so with
    # a RuntimeWarning, which is not news here -- the missing rows already warned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(a, axis=0) if a.size else np.full(44, np.nan)
        sd = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.full(44, np.nan)
    sd = np.where(n > 1, sd, np.nan)
    return mean, sd, n


#: Two figures read the same ``summary_arrays.npz``; cached so a missing file warns once.
_ARRAY_CACHE: dict = {}


def summary_array_path(run_directory, model):
    """The ``summary_arrays*.npz`` that belongs to ``model``, or None.

    A run directory that holds one model's posteriors gets ``summary_arrays.npz``; one
    that holds several (``results/published/``, three models side by side) gets
    ``summary_arrays_<model>.npz``. Prefer the name, fall back to the stored ``model``
    entry, and accept a lone file whatever it is called.
    """
    if run_directory is None:
        return None
    named = run_directory / f"summary_arrays_{model}.npz"
    if named.exists():
        return named
    candidates = sorted(run_directory.glob("summary_arrays*.npz"))
    if not candidates:
        return None
    found = []
    for cand in candidates:                       # ask the files, never assume
        with np.load(cand, allow_pickle=False) as z:
            stored = str(z["model"]) if "model" in z.files else None
        if stored == model:
            return cand
        if stored is None and len(candidates) == 1:
            return cand                           # written before the model key existed
        found.append(stored or cand.name)
    warn_once(("npz-model", str(run_directory), model),
              f"{run_directory.name}: no summary_arrays for model {model}; "
              f"found {', '.join(found)}")
    return None


def load_summary_arrays(results_root, enc, run):
    """``<run>/summary_arrays[_<model>].npz`` as a dict, or None (with a warn)."""
    d = run_dir(results_root, enc, run)
    path = summary_array_path(d, enc["model"])
    if path is None or not path.exists():
        # the calibration panel and figure D both ask for this file
        warn_once(("npz", enc["key"], run),
                  f"{enc['key']}: no summary_arrays.npz for {run} -- panel skipped")
        return None
    key = (enc["key"], run, str(path), path.stat().st_mtime_ns)
    if key not in _ARRAY_CACHE:
        z = np.load(path, allow_pickle=False)
        arrays = {k: z[k] for k in z.files}
        # the columns are read positionally everywhere below; say so if they ever move
        if "arms" in arrays and [str(a) for a in arrays["arms"]] != ARMS:
            warn_once(("arms", enc["key"], run),
                      f"{enc['key']}/{run}: summary_arrays.npz does not carry the 44 arms in "
                      "genomic order -- the per-arm lines may be mislabelled")
        _ARRAY_CACHE[key] = arrays
    return _ARRAY_CACHE[key]


def load_tarp(results_root, enc, run):
    """(curve, summary) from ``tarp_curve.csv`` / ``tarp_summary.json``, or (None, None).

    Both files carry every model of the run: the CSV has a leading ``model`` column and
    the JSON is a list of one object per model, so both are filtered down to this
    encoder's model. A one-model file written without that column still reads.
    """
    d = run_dir(results_root, enc, run)
    if d is None or not (d / "tarp_curve.csv").exists():
        warn_once(("tarp", enc["key"], run),
                  f"{enc['key']}: no tarp_curve.csv for {run} -- TARP row skipped")
        return None, None
    rows = _model_rows(read_rows(d / "tarp_curve.csv"), enc["model"], f"{run}/tarp_curve.csv")
    if not rows:
        return None, None
    curve = {k: np.array([float(r[k]) for r in rows]) for k in ("alpha", "ecp", "ecp_lo", "ecp_hi")
             if k in rows[0]}
    summary = None
    if (d / "tarp_summary.json").exists():
        with open(d / "tarp_summary.json") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            mine = [s for s in payload if s.get("model") == enc["model"]]
            if not mine:
                found = ", ".join(sorted(str(s.get("model")) for s in payload)) or "nothing"
                warn_once(("tarp-model", run, enc["model"]),
                          f"{run}/tarp_summary.json: no entry for model {enc['model']}; "
                          f"found {found}")
            summary = mine[0] if mine else None
        else:
            summary = payload
    return curve, summary


def tarp_direction(summary):
    """``(gap, verdict)`` for a TARP summary, in ``tarp.py``'s own convention.

    ``atc`` is a *distance* -- the mean ``|ecp - alpha|``, never negative -- so it says
    how far from the diagonal a model is, not which side of it. ``atc_signed`` is
    ``check_tarp``'s summed gap over the upper half of the alpha grid; dividing by the
    number of points in that half turns it back into a mean gap, which is what
    ``tarp.py``'s CLI prints as a direction: positive = the posterior is too wide
    (under-confident), negative = over-confident, and under 0.01 either way it is on the
    diagonal. Printing ``atc`` itself with a sign would invent a direction it does not
    carry.
    """
    if not summary or summary.get("atc_signed") is None:
        return None, None
    n_alpha = int(summary.get("n_alpha") or 50)
    gap = float(summary["atc_signed"]) / max(n_alpha - n_alpha // 2, 1)
    verdict = ("on the diagonal" if abs(gap) < 0.01 else
               "too wide" if gap > 0 else "over-confident")
    return gap, verdict


# --------------------------------------------------------------------- plotting
def save(fig, base):
    # dpi explicitly, not from rcParams: these figures are also drawn by callers that
    # never went through build(), and a 100-dpi PNG of a 44-arm strip is unreadable.
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved ->", f"{base}.png")
    return {"png": Path(f"{base}.png"), "pdf": Path(f"{base}.pdf")}


def _arm_axis(ax, ylabel):
    """The shared furniture of the two 44-arm strips."""
    for j, a in enumerate(ARMS):
        if a in ACROCENTRIC:
            ax.axvspan(j - 0.5, j + 0.5, color=PANEL, lw=0, zorder=0)
    ax.annotate("acrocentric p-arms", xy=(ARMS.index("14p"), 0.975),
                xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.6,
                color="#6b6a62", fontweight="bold", zorder=10,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(44))
    ax.set_xticklabels(ARMS, rotation=90, fontsize=6.2)
    ax.set_xlim(-0.8, 43.8)
    ax.set_xlabel("chromosome arm", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _spread(ys, gap, lo, hi):
    """Push overlapping direct labels apart, keeping their order and staying in range."""
    ys = np.asarray(ys, dtype=float)
    out = ys.copy()
    order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
    for i in range(1, order.size):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    over = out[order[-1]] - hi
    if over > 0:
        out -= over
    under = lo - out[order[0]]
    if under > 0:
        out += min(under, hi - out[order[-1]])
    return out


def _strip(ax, encoders, family, label_fmt="{label}   {mean:.2f}", mean_line=True):
    """One line + markers per encoder, error bars = seed sd, direct labels at the right.

    Call after the y-limits are set: the labels are de-collided against them.
    """
    x = np.arange(44)
    singletons = []
    for enc in encoders:
        mean, sd, _ = family[enc["key"]]
        c = enc["colour"]
        if mean_line:
            ax.axhline(np.nanmean(mean), color=c, lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=2)
        yerr = None if not np.isfinite(sd).any() else np.where(np.isfinite(sd), sd, 0.0)
        if yerr is None:
            singletons.append(enc["label"])
        ax.errorbar(x, mean, yerr=yerr, color=c, lw=2.0, marker="o", ms=3.4, mfc=c,
                    mec="white", mew=0.7, elinewidth=1.0, capsize=1.8, ecolor=c,
                    alpha=0.95, zorder=5, solid_capstyle="round")
    lo, hi = ax.get_ylim()
    ys = _spread([float(family[e["key"]][0][-1]) for e in encoders], 0.055 * (hi - lo), lo, hi)
    for enc, y in zip(encoders, ys):
        ax.annotate(label_fmt.format(label=enc["label"],
                                     mean=float(np.nanmean(family[enc["key"]][0]))),
                    xy=(43.35, y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.6, fontweight="bold", color=enc["colour"],
                    zorder=9, path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.legend([plt.Line2D([0], [0], color=e["colour"], lw=2.0, marker="o", ms=3.4, mec="white")
               for e in encoders], [e["label"] for e in encoders], loc="lower left",
              ncol=len(encoders), frameon=False, fontsize=7, handlelength=1.8,
              columnspacing=1.4, bbox_to_anchor=(0.0, -0.005))
    return singletons


def data_n_cases(results_root, man, default=None):
    """How many held-out simulations the result files actually report.

    The manifest's ``n_test_cases`` is a label somebody typed; ``metrics_summary.csv``
    is what the run evaluated. A figure caption states the second.
    """
    for enc in man["encoders"]:
        row = summary_row(results_root, enc, enc["headline"])
        if row and row.get("n_cases"):
            return int(float(row["n_cases"]))
    return default if default is not None else man.get("n_test_cases")


def _has_data(fam):
    """A seed family worth drawing: present, and not all-NaN (a model absent from a run)."""
    return fam is not None and bool(np.isfinite(fam[0]).any())


def fig_E_per_arm_r2(results_root, man, out_base, family=None, n_cases=None):
    """Figure E: which arms each encoder actually learns."""
    encoders = [e for e in man["encoders"]]
    family = family or {e["key"]: seed_family(results_root, e, "true_r2") for e in encoders}
    encoders = [e for e in encoders if _has_data(family.get(e["key"]))]
    if not encoders:
        warn("figure E: no per-arm CSVs found -- skipped")
        return None
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    _arm_axis(ax, "per-arm true $R^2$  (1 − SSE/SST)")
    ax.set_ylim(-0.02, 1.0)
    singletons = _strip(ax, encoders, family)
    cases = n_cases if n_cases is not None else data_n_cases(results_root, man)
    ax.set_title("Which arms each encoder actually learns — 44 arms, "
                 f"{cases if cases is not None else '?'} held-out sims, "
                 "error bars = sd over training seeds",
                 fontsize=9.5, fontweight="bold", color=INK, loc="left", pad=14)
    seeds = "  ·  ".join(f"{e['label']} {int(np.nanmax(family[e['key']][2]))}" for e in encoders)
    note = "dashed line = that encoder's mean over the 44 arms  ·  seeds: " + seeds
    if singletons:
        note += "  ·  no error bars for " + ", ".join(singletons) + " (one seed only)"
    ax.annotate(note, xy=(0, 1.005), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=6.4, color=MUTED)
    chrom = np.array([int(a[:-1]) for a in ARMS], dtype=float)
    # the hyphen-minus only inside the number: the encoder labels contain hyphens too
    rhos = "  ·  ".join(
        "{} {}".format(e["label"], f"{spearman(family[e['key']][0], chrom):+.2f}".replace("-", "−"))
        for e in encoders)
    ax.annotate("Spearman ρ of per-arm $R^2$ with chromosome number:  " + rhos,
                xy=(0.0, -0.30), xycoords="axes fraction", ha="left", va="top",
                fontsize=6.6, color=MUTED)
    fig.tight_layout()
    return save(fig, out_base)


def fig_S3_per_arm_coverage(results_root, man, out_base, n_cases=None, family=None):
    """Supplement S3: per-arm 95 % coverage with seed bars and the binomial band."""
    encoders = list(man["encoders"])
    family = family or {e["key"]: seed_family(results_root, e, "coverage_95") for e in encoders}
    encoders = [e for e in encoders if _has_data(family.get(e["key"]))]
    if not encoders:
        warn("figure S3: no per-arm CSVs found -- skipped")
        return None
    if n_cases is None:
        n_cases = data_n_cases(results_root, man) or 651
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    # +/-2 sd of a Binomial(n, 0.95) proportion: the spread a perfectly calibrated
    # posterior still shows at this many test cases. Outside it is a real miss.
    sigma = float(np.sqrt(0.95 * 0.05 / max(n_cases, 1)))
    ax.axhspan(0.95 - 2 * sigma, 0.95 + 2 * sigma, color="#e9e6da", lw=0, zorder=1)
    ax.axhline(0.95, color="#6f6e68", lw=1.0, ls=(0, (4, 2.5)), zorder=3)
    _arm_axis(ax, "per-arm 95 % credible-interval coverage")
    # Data-driven limits: coverage lives in a few per cent around 0.95, and a fixed
    # (0.8, 1.0) window spends four fifths of the panel on empty space.
    lows, highs = [], []
    for e in encoders:
        mean, sd, _ = family[e["key"]]
        band = np.nan_to_num(sd)
        finite = np.isfinite(mean)
        if finite.any():
            lows.append(float(np.min((mean - band)[finite])))
            highs.append(float(np.max((mean + band)[finite])))
    lo = min(lows + [0.95 - 3 * sigma]) - 0.004
    hi = max(highs + [0.95 + 3 * sigma]) + 0.004
    ax.set_ylim(lo, hi)
    singletons = _strip(ax, encoders, family, label_fmt="{label}   {mean:.3f}")
    ax.annotate(f"nominal 0.95  ± 2σ binomial band (n={n_cases}, σ={sigma:.4f})",
                xy=(0.008, 0.95 + 2 * sigma), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.0, color="#56554f", fontweight="bold",
                zorder=8, path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.set_title("Is every arm's uncertainty honest?  per-arm 95 % coverage, "
                 "error bars = sd over training seeds",
                 fontsize=9.5, fontweight="bold", color=INK, loc="left", pad=14)
    note = "dashed line = nominal 0.95; shaded = the spread a calibrated posterior still shows"
    if singletons:
        note += "  ·  no error bars for " + ", ".join(singletons) + " (one seed only)"
    ax.annotate(note, xy=(0, 1.005), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=6.4, color=MUTED)
    fig.tight_layout()
    return save(fig, out_base)


def fig_calibration_panel(results_root, man, out_base, seed=20260925, band_draws=1000):
    """One column per encoder: the stacked SBC ECDF-difference, and the TARP curve.

    The band is simulated at n = number of held-out tumours and drawn per arm. Pooling
    the 44 arms into one test would be wrong: the arms of one tumour share an encoder
    and one conditioned flow, so they move together.

    Returns a dict of per-encoder diagnostics (band half-width, per-arm maximum
    deviation, the FDR failure count and its range over the seeds) so a caller can
    assert on the numbers behind the picture; None if no encoder had the arrays.
    """
    grid = np.linspace(0.0, 1.0, 201)
    panels, stats = [], {}
    for enc in man["encoders"]:
        arrays = load_summary_arrays(results_root, enc, enc["headline"])
        if arrays is None:
            continue
        if "u" not in arrays:
            warn(f"{enc['key']}: summary_arrays.npz carries no SBC PIT 'u' -- panel skipped")
            continue
        u = np.asarray(arrays["u"], dtype=float)
        n_cases = u.shape[0]
        eps = simultaneous_band(n_cases, n_draws=band_draws, seed=seed)
        # the same statistic the band is a quantile of, so "outside the band" is exact
        dev = np.array([ks_statistic(u[:, j]) for j in range(u.shape[1])])
        head = per_arm_values(results_root, enc, enc["headline"], "sbc_ks_p")
        fails = bh_reject(head) if head is not None else None
        per_seed = [bh_reject(v) for v in
                    (per_arm_values(results_root, enc, r, "sbc_ks_p") for r in enc["members"])
                    if v is not None]
        stats[enc["key"]] = {
            "n_cases": n_cases, "band": eps, "max_dev": dev,
            "fdr_headline": fails, "fdr_range": (min(per_seed), max(per_seed)) if per_seed else None,
        }
        curve, tsum = load_tarp(results_root, enc, enc["headline"])
        panels.append((enc, u, grid, eps, curve, tsum))
    if not panels:
        warn("calibration panel: no summary_arrays.npz for any encoder -- figure skipped")
        return None, stats

    n_rows = 2 if any(p[4] for p in panels) else 1
    fig, axes = plt.subplots(n_rows, len(panels), squeeze=False,
                             figsize=(3.0 * len(panels), 3.2 * n_rows), sharex=True, sharey="row")
    for k, (enc, u, g, eps, curve, tsum) in enumerate(panels):
        ax = axes[0][k]
        ax.fill_between(g, -eps, eps, color="#e4e1d6", lw=0, zorder=1)
        for edge in (-eps, eps):
            ax.plot([0, 1], [edge, edge], ls=(0, (4, 2.5)), color="#6f6e68", lw=1.0, zorder=7)
        for j in range(u.shape[1]):
            ax.plot(g, ecdf_difference(u[:, j], g), color=enc["colour"], lw=0.7, alpha=0.45, zorder=4)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=3)
        st = stats[enc["key"]]
        title = enc["label"]
        if st["fdr_headline"] is not None:
            title += f"\n{st['fdr_headline']}/44 arms fail after FDR"
            if st["fdr_range"] is not None:
                # its own line: four of these side by side overrun each other on one
                title += f"\n(range over seeds {st['fdr_range'][0]}–{st['fdr_range'][1]})"
        ax.set_title(title, fontsize=7.0, fontweight="bold", color=enc["colour"], loc="left",
                     linespacing=1.45)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if n_rows == 1:
            ax.set_xlabel("normalised rank of the true θ", fontsize=7.4)
        ax.annotate(f"95 % simultaneous band  (n={st['n_cases']})", xy=(0.985, eps),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.0, color="#56554f", fontweight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
        if n_rows == 2:
            bx = axes[1][k]
            if curve:
                bx.plot([0, 1], [0, 1], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
                if "ecp_lo" in curve and "ecp_hi" in curve:
                    bx.fill_between(curve["alpha"], curve["ecp_lo"], curve["ecp_hi"],
                                    color=enc["colour"], alpha=0.20, lw=0, zorder=3)
                bx.plot(curve["alpha"], curve["ecp"], color=enc["colour"], lw=2.0, zorder=5)
                if tsum is not None and "atc" in tsum:
                    # atc is |gap| to the diagonal; the side it falls on is atc_signed
                    gap, verdict = tarp_direction(tsum)
                    text = f"TARP dist. {float(tsum['atc']):.3f}"
                    if "ks_p" in tsum:
                        text += f"   KS p {float(tsum['ks_p']):.3f}"
                    if gap is not None:
                        # second line: four of these run off the panel on one
                        text += f"\n{minus(gap, '+.3f')} {verdict}"
                    bx.annotate(text,
                                xy=(0.03, 0.95), xycoords="axes fraction", ha="left", va="top",
                                fontsize=6.6, fontweight="bold", color=enc["colour"],
                                linespacing=1.5)
                bx.set_xlim(0, 1)
                bx.set_ylim(0, 1)
            else:
                bx.text(0.5, 0.5, "no TARP curve", ha="center", va="center",
                        fontsize=7, color=MUTED, transform=bx.transAxes)
                bx.set_xticks([])
                bx.set_yticks([])
            bx.set_xlabel("credibility level", fontsize=7.4)
            bx.grid(color=GRID, lw=0.6)
            bx.set_axisbelow(True)
            for sp in ("top", "right"):
                bx.spines[sp].set_visible(False)
    axes[0][0].set_xlim(0, 1)
    axes[0][0].set_ylabel("ECDF − uniform", fontsize=7.4)
    axes[0][0].annotate("one line per arm", xy=(0.03, 0.05), xycoords="axes fraction",
                        fontsize=6.2, color=MUTED, va="bottom", zorder=9,
                        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    if n_rows == 2:
        axes[1][0].set_ylabel("expected coverage (TARP)", fontsize=7.4)
    fig.suptitle("Calibration: marginal (SBC, top) and joint (TARP, bottom)"
                 if n_rows == 2 else
                 "Calibration: is the true θ a typical posterior draw?",
                 fontsize=10, fontweight="bold", color=INK, x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    return save(fig, out_base), stats


def fig_D_panel(results_root, man, out_base):
    """The pooled truth-vs-posterior-mean hexbins, one per encoder, side by side.

    Returns ``{encoder key: {"true_r2", "pearson_r2", "slope", "intercept", "n"}}``.
    """
    panels = []
    for enc in man["encoders"]:
        arrays = load_summary_arrays(results_root, enc, enc["headline"])
        if arrays is None or "theta_true" not in arrays or "post_mean" not in arrays:
            continue
        t = np.asarray(arrays["theta_true"], dtype=float).ravel()
        m = np.asarray(arrays["post_mean"], dtype=float).ravel()
        slope, intercept = np.polyfit(t, m, 1)
        panels.append((enc, t, m, {"true_r2": true_r2(t, m), "pearson_r2": pearson_r2(t, m),
                                   "slope": float(slope), "intercept": float(intercept),
                                   "n": int(t.size)}))
    if not panels:
        warn("figure D panel: no summary_arrays.npz for any encoder -- figure skipped")
        return None, {}
    stats = {enc["key"]: s for enc, _, _, s in panels}
    lim = max(float(np.abs(t).max()) for _, t, _, _ in panels) * 1.05
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4),
                             squeeze=False, sharex=True, sharey=True)
    for ax, (enc, t, m, s) in zip(axes[0], panels):
        ax.hexbin(t, m, gridsize=55, extent=(-lim, lim, -lim, lim), mincnt=1,
                  cmap="Greys", bins="log", linewidths=0, zorder=2)
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=5)
        xs = np.array([-lim, lim])
        ax.plot(xs, s["slope"] * xs + s["intercept"], color=enc["colour"], lw=1.6, zorder=6)
        ax.set_title(enc["label"], fontsize=8.4, fontweight="bold", color=enc["colour"], loc="left")
        ax.annotate(f"true $R^2$ = {minus(s['true_r2'])}\n$r^2$ = {minus(s['pearson_r2'])}\n"
                    f"slope = {minus(s['slope'], '.2f')}",
                    xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.0, color=INK2, linespacing=1.5,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
        ax.set_xlabel(r"true $\theta$", fontsize=7.4)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("posterior mean", fontsize=7.4)
    axes[0][0].set_xlim(-lim, lim)
    axes[0][0].set_ylim(-lim, lim)
    fig.suptitle("Pooled recovery: every arm of every held-out tumour  "
                 "(dashed = identity, solid = OLS)",
                 fontsize=10, fontweight="bold", color=INK, x=0.02, ha="left", y=1.03)
    fig.tight_layout()
    return save(fig, out_base), stats


# ----------------------------------------------------------------------- tables
def per_arm_summary(results_root, man, stats=SUMMARY_STATS):
    """{(encoder key, stat): (mean, sd, n)} for the S2 table."""
    out = {}
    for enc in man["encoders"]:
        for stat in stats:
            fam = seed_family(results_root, enc, stat)
            if fam is not None:
                out[(enc["key"], stat)] = fam
    return out


def write_per_arm_summary(table, man, csv_path, md_path, stats=SUMMARY_STATS):
    """The S2 table: 44 rows, one block of (mean, sd, n) per encoder per statistic."""
    keys = [e["key"] for e in man["encoders"] if any((e["key"], s) in table for s in stats)]
    fields = ["chromosome_arm"] + [f"{k}_{s}_{w}" for k in keys for s in stats
                                   for w in ("mean", "sd", "n")]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i, arm in enumerate(ARMS):
            row = [arm]
            for k in keys:
                for s in stats:
                    fam = table.get((k, s))
                    if fam is None:
                        row += ["", "", ""]
                    else:
                        mean, sd, n = fam
                        row += ["" if not np.isfinite(mean[i]) else f"{mean[i]:.6g}",
                                "" if not np.isfinite(sd[i]) else f"{sd[i]:.6g}",
                                str(int(n[i]))]
            w.writerow(row)
    print("saved ->", csv_path)

    labels = {e["key"]: e["label"] for e in man["encoders"]}
    # an encoder whose model is absent from every run has nothing to put in a cell
    md_keys = [k for k in keys if _has_data(table.get((k, "true_r2")))]
    lines = ["# Per-arm true $R^2$, mean ± sd over training seeds", "",
             "Coefficient of determination, 1 − SSE/SST, per chromosome arm. The full table "
             "(all statistics) is `per_arm_summary.csv`.", "",
             "| arm | " + " | ".join(labels[k] for k in md_keys) + " |",
             "| --- | " + " | ".join("---" for _ in md_keys) + " |"]
    for i, arm in enumerate(ARMS):
        cells = []
        for k in md_keys:
            mean, sd, n = table[(k, "true_r2")]
            cells.append("—" if not np.isfinite(mean[i]) else
                         (f"{mean[i]:.3f} ± {sd[i]:.3f}" if np.isfinite(sd[i])
                          else f"{mean[i]:.3f}"))
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    Path(md_path).write_text("\n".join(lines) + "\n")
    print("saved ->", md_path)
    return fields


def headline_rows(results_root, man):
    """One dict per encoder with exactly the numbers ``headline_table.md`` prints."""
    rows = []
    for enc in man["encoders"]:
        head = summary_row(results_root, enc, enc["headline"])
        if head is None:
            warn(f"{enc['key']}: no metrics_summary.csv for headline {enc['headline']} -- row skipped")
            continue
        seeds = [summary_row(results_root, enc, r) for r in enc["members"]]
        r2 = [float(s["mean_true_r2"]) for s in seeds if s is not None]
        fails = [bh_reject(v) for v in
                 (per_arm_values(results_root, enc, r, "sbc_ks_p") for r in enc["members"])
                 if v is not None]
        _, tsum = load_tarp(results_root, enc, enc["headline"])
        gap, verdict = tarp_direction(tsum)
        rows.append({
            "key": enc["key"], "label": enc["label"], "headline": enc["headline"],
            "true_r2_headline": float(head["mean_true_r2"]),
            "true_r2_mean": float(np.mean(r2)) if r2 else float("nan"),
            "true_r2_sd": float(np.std(r2, ddof=1)) if len(r2) > 1 else float("nan"),
            "n_seeds": len(r2),
            "log_prob_true": float(head["mean_log_prob_true"]),
            "coverage_95": float(head["coverage_95"]),
            "sbc_fails_headline": int(float(head["n_arms_ks_reject_fdr05"])),
            "sbc_fails_range": (min(fails), max(fails)) if fails else None,
            # a non-negative distance, and separately the side of the diagonal it is on
            "tarp_atc": float(tsum["atc"]) if tsum and "atc" in tsum else None,
            "tarp_gap": gap,
            "tarp_verdict": verdict,
            "config": enc["config"],
        })
    return rows


def write_headline_table(rows, man, path, n_cases=None):
    cases = n_cases if n_cases is not None else man.get("n_test_cases", "?")
    lines = [f"# {man.get('title', 'Best honest configuration per encoder')} — headline numbers",
             "",
             f"Campaign `{man.get('campaign', '?')}`, {cases} held-out "
             "simulations, 44 chromosome arms. true $R^2$ is 1 − SSE/SST, averaged over the arms; "
             "the seed column is the mean ± sd over the member runs of the same configuration. "
             "TARP dist. is the mean |gap| to the diagonal, a distance; TARP direction is that "
             "gap with its sign — positive means the posterior is too wide, negative "
             "over-confident, and under 0.01 either way it is on the diagonal.",
             "",
             "| Encoder | Headline run | true R² (headline) | true R² (seeds, mean ± sd) | "
             "log p(θ*) | 95 % coverage | SBC fails /44 (headline; seed range) | TARP dist. | "
             "TARP direction |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        seeds = ("—" if not np.isfinite(r["true_r2_mean"]) else
                 (f"{r['true_r2_mean']:.3f} ± {r['true_r2_sd']:.3f} (n={r['n_seeds']})"
                  if np.isfinite(r["true_r2_sd"]) else f"{r['true_r2_mean']:.3f} (n=1)"))
        rng = ("—" if r["sbc_fails_range"] is None
               else f"{r['sbc_fails_range'][0]}–{r['sbc_fails_range'][1]}")
        atc = "—" if r["tarp_atc"] is None else f"{r['tarp_atc']:.3f}"
        direction = ("—" if r.get("tarp_gap") is None
                     else f"{r['tarp_gap']:+.3f} ({r['tarp_verdict']})")
        lines.append(f"| {r['label']} | {r['headline']} | {r['true_r2_headline']:.3f} | {seeds} | "
                     f"{r['log_prob_true']:.1f} | {r['coverage_95']:.3f} | "
                     f"{r['sbc_fails_headline']} ({rng}) | {atc} | {direction} |")
    lines += ["", "Configurations:", ""]
    for r in rows:
        lines.append(f"* **{r['label']}** (`{r['headline']}`): `{r['config']}`")
    Path(path).write_text("\n".join(lines) + "\n")
    print("saved ->", path)
    return lines


# ------------------------------------------------------------------------- main
def build(results_root, manifest_path=DEFAULT_MANIFEST, out_dir="best_honest",
          seed=20260925, band_draws=1000):
    """Everything this module makes. Returns the paths and the numbers behind them."""
    plt.rcParams.update(RC)
    man = load_manifest(manifest_path)
    figures = Path(out_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    r2_family = {e["key"]: seed_family(results_root, e, "true_r2") for e in man["encoders"]}
    cov_family = {e["key"]: seed_family(results_root, e, "coverage_95") for e in man["encoders"]}
    # the CSVs decide how many test cases this campaign has, not the manifest
    n_cases = data_n_cases(results_root, man, default=man.get("n_test_cases", 651))

    out = {"manifest": man, "out_dir": Path(out_dir), "figures_dir": figures}
    out["fig_E"] = fig_E_per_arm_r2(results_root, man, str(figures / "fig_E_per_arm_r2"),
                                    family=r2_family, n_cases=n_cases)
    out["fig_S3"] = fig_S3_per_arm_coverage(results_root, man,
                                            str(figures / "fig_S3_per_arm_coverage"),
                                            n_cases=n_cases, family=cov_family)
    out["fig_calibration"], out["calibration"] = fig_calibration_panel(
        results_root, man, str(figures / "fig_calibration_panel"), seed=seed, band_draws=band_draws)
    out["fig_D"], out["fig_D_stats"] = fig_D_panel(results_root, man, str(figures / "fig_D_panel"))

    table = per_arm_summary(results_root, man)
    out["per_arm"] = table
    out["per_arm_fields"] = write_per_arm_summary(table, man, figures / "per_arm_summary.csv",
                                                  figures / "per_arm_summary.md")
    out["headline"] = headline_rows(results_root, man)
    write_headline_table(out["headline"], man, figures / "headline_table.md", n_cases=n_cases)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="campaign root holding <run>/ directories (or an encoder tree "
                         "holding <encoder>/<run>/)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="seed-family manifest (default: the one shipped with this module)")
    ap.add_argument("--out-dir", default="best_honest",
                    help="figures and tables land in <out-dir>/figures/")
    ap.add_argument("--seed", type=int, default=20260925,
                    help="seed of the simulated SBC band (poster_metrics.ecdf_band's)")
    ap.add_argument("--band-draws", type=int, default=1000,
                    help="draws used to simulate the SBC band")
    args = ap.parse_args(argv)

    out = build(args.results, args.manifest, args.out_dir, seed=args.seed,
                band_draws=args.band_draws)
    print("\n=== headline ===")
    for r in out["headline"]:
        sd = "" if not np.isfinite(r["true_r2_sd"]) else f" ± {r['true_r2_sd']:.3f}"
        print(f"{r['label']:>20}  {r['headline']:>8}  true R2 {r['true_r2_headline']:.3f}  "
              f"(seeds {r['true_r2_mean']:.3f}{sd})  cov95 {r['coverage_95']:.3f}  "
              f"SBC fail {r['sbc_fails_headline']}/44")
    print(f"\nwrote {sum(1 for _ in out['figures_dir'].iterdir())} files to {out['figures_dir']}")
    return out


if __name__ == "__main__":
    main()
