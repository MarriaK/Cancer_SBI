#!/usr/bin/env python
"""
Stage 2b: TARP, the *joint* calibration test, computed from the samples stage 1 stored.

Figure C asks whether each of the 44 arms is calibrated on its own. That question is
provably blind to a posterior that gets every marginal right and every correlation wrong
(Modrak et al. 2025), and this model has 44 parameters whose correlations are the whole
point. TARP - Lemos, Coogan et al. 2023, ICML (arXiv:2302.03026) - is the test that is
necessary *and* sufficient for the full joint posterior, needs only samples (no density
evaluation), and was demonstrated at 256 dimensions. It is the one diagnostic the suite
was missing.

The algorithm, per simulation i:
  1. draw a reference point theta_r from a fixed reference distribution;
  2. measure the distance from theta_r to each of the S posterior draws, and to the truth;
  3. f_i = the fraction of draws that are *closer* to theta_r than the truth is.
If the samples really are the posterior, then theta_true and any draw are exchangeable, so
f_i is Uniform(0,1). The curve ECP(alpha) = mean_i[f_i < alpha] is then the diagonal: the
credible region of level alpha, built around theta_r, contains the truth alpha of the time.
Below the diagonal = over-confident (too narrow, or biased); above = too wide.

Why this is re-implemented rather than imported. ``sbi.diagnostics.run_tarp`` wants a live
``NeuralPosterior`` object and re-samples it; the 5,000 draws per tumour already exist in
the stage-1 ``.npz`` and re-drawing them would need the checkpoint, a GPU and an hour. The
inner function ``sbi.diagnostics.tarp._run_tarp`` is what is reproduced here, and the
conventions were read off that source (sbi 0.23, ``diagnostics/tarp.py``):

  * f is ``(sample_dists < theta_dists).sum() / S`` - a *strict* comparison, so a posterior
    whose every draw beats the truth scores exactly 1.0. Kept.
  * ECP is the empirical CDF of f, and it is compared against ``alpha`` itself, not
    ``1 - alpha``: sbi builds it as ``cumsum(histogram(f))`` and returns the histogram's
    bin edges as the alpha grid. So calibrated <=> ECP(alpha) = alpha, and that is the
    convention used here. The one deliberate difference: sbi's alpha grid is the histogram
    of the *observed* f, so it runs from min(f) to max(f) and moves with the data. Here
    alpha is a fixed ``linspace(0, 1, n_alpha)``, which is what makes two models
    comparable on one axis and what lets ``atc`` mean the same thing twice.
  * ``check_tarp`` reports ``atc`` as the signed *sum* of ``ecp - alpha`` over the upper
    half of the grid (negative = under-dispersed) and a *two-sample* KS between the ecp
    curve and the alpha grid. Both are reported here, but the headline ``atc`` is the
    scalar this project asked for, ``mean |ecp - alpha|`` - always non-negative, and a
    distance rather than a direction - with sbi's signed version beside it as
    ``atc_signed`` and the direction it carries spelled out in the JSON.
  * z-scoring: sbi rescales every dimension to the observed [min, max] of theta before
    measuring distance. Here the default is a per-dimension z-score by the sd of
    ``theta_true``. Note that all 44 dimensions share one N(0, 0.2) prior, so if you pass
    a single ``prior_sd`` the rescaling is a common factor on every distance and f is
    unchanged - the option exists to be explicit, not because it moves the number.

What it writes in --out-dir:
  tarp_curve.csv     model, alpha, ecp, ecp_lo, ecp_hi  (the bootstrap band over sims)
  tarp_summary.json  a list, one object per model: atc, ks_p and what produced them
                     (ks_p uses one reference point per simulation - see tarp_scalars)
  fig_T_tarp.{png,pdf}  the ECP curve of every model against the diagonal

    python -m cancer_sbi.evaluation.tarp --in-dir out --run-tag R1 --out-dir results
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cancer_sbi.evaluation.poster_metrics import (
    AXIS, COLOUR, GRID, INK, INK2, LABEL, MODEL_ORDER, MUTED,
    ks_uniform, posterior_path, save, unknown_models, write_csv,
)
from cancer_sbi.evaluation.style import ECDF_BAND_EDGE, TARP_BAND_FILL

#: How the per-dimension distance scale is chosen. ``"theta_sd"`` is the default and the
#: only one that both z-scores and varies by arm; ``"range"`` reproduces sbi's min-max;
#: ``"prior_sd"`` uses the single number from the run's meta; ``"none"`` measures raw
#: Euclidean distance in selection-coefficient units.
Z_SCORE_MODES = ("theta_sd", "range", "prior_sd", "none")


# ------------------------------------------------------------------ the statistic
def distance_scale(theta_true, z_score="theta_sd", prior_sd=None):
    """The per-dimension divisor applied before any distance is measured.

    Returns a ``(D,)`` array of strictly positive numbers. A dimension with no spread at
    all (a constant column) would divide by zero, so it falls back to 1.
    """
    theta = np.asarray(theta_true, dtype=np.float64)
    if z_score not in Z_SCORE_MODES:
        raise ValueError(f"z_score must be one of {Z_SCORE_MODES}, got {z_score!r}")
    if z_score == "none":
        scale = np.ones(theta.shape[1])
    elif z_score == "theta_sd":
        scale = theta.std(axis=0, ddof=0)
    elif z_score == "range":                      # sbi's own normalisation
        scale = theta.max(axis=0) - theta.min(axis=0)
    else:
        if prior_sd is None:
            raise ValueError("z_score='prior_sd' needs prior_sd (it is in the run's meta)")
        scale = np.full(theta.shape[1], float(prior_sd))
    scale = np.asarray(scale, dtype=np.float64)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return scale


def draw_references(theta_true, num_references=1, reference="random", seed=0):
    """Reference points, shape ``(N, R, D)``.

    ``"random"`` is sbi's ``get_tarp_references``: one uniform draw per simulation from the
    box ``[min, max]`` of ``theta_true`` in each dimension. ``"prior"`` draws instead from
    the empirical prior - *another* tumour's true theta, picked at random - which
    concentrates the references where parameter vectors actually live rather than
    spreading them over the corners of a 44-dimensional box, where nothing does.

    TARP holds for *any* reference distribution with support over the parameter space, so
    the two must agree on a calibrated posterior; where they disagree, the posterior is
    mis-calibrated somewhere and the references say roughly where.

    There is no option to centre the reference on the simulation's *own* truth, and this
    is not an oversight. TARP's proof needs the reference to be independent of which of the
    exchangeable points is labelled "truth"; draw theta_r around theta_true and the truth
    is by construction the nearest point, f collapses onto 0, and a perfectly calibrated
    posterior is reported as catastrophically mis-calibrated. Measured on the synthetic
    calibrated posterior in ``tests/test_eval_redesign_wp1.py``: max |ecp - alpha| = 0.94.
    """
    theta = np.asarray(theta_true, dtype=np.float64)
    n, d = theta.shape
    rng = np.random.default_rng(seed)
    if reference == "random":
        lo, hi = theta.min(axis=0), theta.max(axis=0)
        return lo + (hi - lo) * rng.random((n, int(num_references), d))
    if reference == "prior":
        pick = rng.integers(0, n, (n, int(num_references)))
        if n > 1:                       # never a tumour's own truth: that is the degenerate case
            own = pick == np.arange(n)[:, None]
            pick[own] = (pick[own] + 1) % n
        return theta[pick]
    raise ValueError(f"reference must be 'random' or 'prior', got {reference!r}")


def tarp_coverage_values(theta_true, samples, num_references=1, reference="random",
                         z_score="theta_sd", prior_sd=None, seed=0):
    """f, the fraction of posterior draws closer to the reference than the truth is.

    Shape ``(N, R)``: one value per simulation per reference point. Uniform(0,1) over the
    simulations exactly when the samples are the posterior - this is Lemos' algorithm 2.

    Computed one simulation at a time. ``samples`` on a real run is 651 x 5000 x 44
    float32, 573 MB; vectorising the distance over simulations as well would need a second
    copy of it, and the loop costs a second.
    """
    theta = np.asarray(theta_true, dtype=np.float64)
    n, d = theta.shape
    if samples.shape[0] != n or samples.shape[2] != d:
        raise ValueError(f"samples {samples.shape} do not match theta_true {theta.shape}")
    scale = distance_scale(theta, z_score, prior_sd)
    refs = draw_references(theta, num_references, reference, seed=seed)
    f = np.empty((n, int(num_references)), dtype=np.float64)
    for i in range(n):
        draws = np.asarray(samples[i], dtype=np.float64) / scale          # (S, D)
        truth = theta[i] / scale
        for r in range(int(num_references)):
            ref = refs[i, r] / scale
            d_theta = np.linalg.norm(truth - ref)
            d_draws = np.linalg.norm(draws - ref, axis=1)
            # strict <, as sbi's _run_tarp has it
            f[i, r] = float((d_draws < d_theta).mean())
    return f


def ecp_curve(f, alpha):
    """Expected coverage probability: the empirical CDF of f read at each alpha."""
    flat = np.asarray(f, dtype=np.float64).ravel()
    return np.array([float((flat < a).mean()) for a in np.asarray(alpha, dtype=np.float64)])


def tarp_from_samples(theta_true, samples, n_alpha=50, num_references=1,
                      reference="random", z_score="theta_sd", prior_sd=None, seed=0):
    """The TARP curve: ``(alpha, ecp)``, both ``(n_alpha,)``, calibrated when they are equal."""
    alpha = np.linspace(0.0, 1.0, int(n_alpha))
    f = tarp_coverage_values(theta_true, samples, num_references, reference,
                             z_score, prior_sd, seed)
    return alpha, ecp_curve(f, alpha)


def tarp_bootstrap(f, alpha, n_boot=200, level=0.95, seed=0):
    """Percentile band for the ECP curve, resampling *simulations* with replacement.

    The simulations are the independent unit here: every f in one row shares a truth and a
    posterior, so resampling f values directly would pretend R references are R tumours and
    make the band too tight.
    """
    f = np.atleast_2d(np.asarray(f, dtype=np.float64))
    rng = np.random.default_rng(seed)
    n = f.shape[0]
    curves = np.empty((int(n_boot), len(alpha)))
    for b in range(int(n_boot)):
        curves[b] = ecp_curve(f[rng.integers(0, n, n)], alpha)
    q = (1.0 - level) / 2.0
    return (np.quantile(curves, q, axis=0), np.quantile(curves, 1.0 - q, axis=0))


def tarp_scalars(alpha, ecp, f):
    """The numbers that go in the JSON: ``atc``, sbi's signed ``atc``, and the KS of f.

    ``atc`` is the mean absolute gap to the diagonal - 0 for a perfect posterior, and a
    distance, so it never cancels a too-wide half against a too-narrow one.
    ``atc_signed`` is ``check_tarp``'s: the summed gap over the upper half of the grid,
    negative for an under-dispersed (over-confident) posterior, positive for a too-wide
    one. The KS test is the one-sample test of f against Uniform(0,1), through the same
    ``ks_uniform`` figure C uses, rather than ``check_tarp``'s two-sample test between the
    two curves - one KS implementation in this package, and the one-sample form is the
    test the theory actually names.

    With more than one reference point per simulation the KS test uses the **first
    reference only**. A KS test needs independent draws; the R values of one simulation
    share a truth and a posterior and are strongly dependent, so ravelling an (N, R) array
    into N*R "observations" inflates the sample size R-fold without adding information and
    the p-value stops meaning anything - at R = 10 on a perfectly calibrated posterior it
    rejected 14 times in 40. Averaging over references would not fix it either: the mean of
    R uniforms is not uniform, so the null hypothesis would no longer be the right one.
    One reference per simulation gives N genuinely independent uniforms and an exact test.
    The ECP curve and its bootstrap band still use all N*R values - they are estimates, not
    tests, and the bootstrap resamples simulations, so the dependence is accounted for.
    """
    alpha, ecp = np.asarray(alpha, dtype=float), np.asarray(ecp, dtype=float)
    mid = alpha.shape[0] // 2
    f = np.asarray(f, dtype=float)
    stat, p = ks_uniform(f if f.ndim == 1 else f[:, 0])
    return {
        "atc": float(np.mean(np.abs(ecp - alpha))),
        "atc_signed": float(np.sum(ecp[mid:] - alpha[mid:])),
        "ks_stat": stat,
        "ks_p": p,
    }


def run_tarp_on_file(path, model, n_alpha=50, num_references=1, reference="random",
                     z_score="theta_sd", n_boot=200, seed=0, max_sims=None):
    """Everything this module knows about one ``posteriors_<model>.npz``, as a dict."""
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["meta_json"]))
        theta = np.asarray(d["theta_true"], dtype=np.float64)
        samples = d["samples"]
        if max_sims is not None and theta.shape[0] > int(max_sims):
            theta, samples = theta[: int(max_sims)], samples[: int(max_sims)]
        else:
            samples = np.asarray(samples)
        alpha = np.linspace(0.0, 1.0, int(n_alpha))
        f = tarp_coverage_values(theta, samples, num_references, reference,
                                 z_score, meta.get("prior_sd"), seed)
    ecp = ecp_curve(f, alpha)
    lo, hi = tarp_bootstrap(f, alpha, n_boot=n_boot, seed=seed)
    out = {
        "model": model,
        "run_tag": meta.get("run_tag") or "",
        **tarp_scalars(alpha, ecp, f),
        "num_sims": int(theta.shape[0]),
        "num_samples": int(samples.shape[1]),
        "num_references": int(num_references),
        "distance": "l2",
        "z_score": z_score,
        "reference": reference,
        "n_alpha": int(n_alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
    return {"summary": out, "alpha": alpha, "ecp": ecp, "ecp_lo": lo, "ecp_hi": hi, "f": f}


# ------------------------------------------------------------------ figure
def fig_tarp(results, out_base):
    """Figure T: expected coverage against credibility, one curve per model.

    The diagonal is a calibrated joint posterior. A curve that sags below it says the
    posterior is too narrow or in the wrong place *jointly* - which figure C cannot see,
    because every marginal can be flawless while the 44-dimensional shape is wrong.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.text(0.63, 0.56, "perfectly calibrated", fontsize=6.6, color=INK2, rotation=39,
            ha="center", va="center")
    for name, res in results.items():
        colour = COLOUR.get(name, ECDF_BAND_EDGE)
        ax.fill_between(res["alpha"], res["ecp_lo"], res["ecp_hi"],
                        color=COLOUR.get(name, TARP_BAND_FILL), alpha=0.16, lw=0, zorder=3)
        ax.plot(res["alpha"], res["ecp"], color=colour, lw=2.0, solid_capstyle="round",
                zorder=5, label=f"{LABEL.get(name, name)}  ({res['summary']['atc']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("credibility level α", fontsize=7.6)
    ax.set_ylabel("expected coverage probability", fontsize=7.6)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # All three notes in the upper-left corner, where no curve of a too-narrow posterior can
    # reach: the bottom-right belongs to the legend, and one of the two must move.
    ax.text(0.03, 0.96, "above the diagonal: posterior too wide\nbelow: over-confident\n"
                        "shaded = 95 % bootstrap over tumours",
            fontsize=6.4, color=MUTED, va="top", linespacing=1.6)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=7,
                    title="mean |gap| to the diagonal (atc)")
    plt.setp(leg.get_title(), fontsize=6.4, color=MUTED)
    ax.set_title("TARP: is the joint posterior honest?", fontsize=10, fontweight="bold",
                 color=INK, loc="left", pad=8)
    save(fig, out_base)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="out", help="folder holding posteriors_<model>.npz")
    ap.add_argument("--run-tag", default=None,
                    help="read posteriors_<model>_<tag>.npz instead, exactly as poster_metrics does")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-alpha", type=int, default=50, help="points on the credibility grid")
    ap.add_argument("--num-references", type=int, default=1,
                    help="reference points per simulation (sbi draws 1). More of them "
                         "smooth the ECP curve and its bootstrap band, but the KS test "
                         "still uses the first reference of each simulation only: the R "
                         "values of one simulation are dependent, so pooling them would "
                         "make the p-value anti-conservative")
    ap.add_argument("--reference", default="random", choices=("random", "prior"),
                    help="reference points: uniform in the theta box (sbi's own) or "
                         "drawn from the empirical prior, i.e. another tumour's truth")
    ap.add_argument("--z-score", default="theta_sd", choices=Z_SCORE_MODES,
                    help="per-dimension scaling before the distance is measured")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples over tumours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-sims", type=int, default=None,
                    help="use only the first N tumours (a smoke test, not a result)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for stem in unknown_models(args.in_dir, args.run_tag):
        print(f"[warn] posteriors_{stem}.npz present but {stem} not in MODEL_ORDER "
              f"-- that run will not be measured")

    results = {}
    for name in MODEL_ORDER:
        path = posterior_path(args.in_dir, name, args.run_tag)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        res = run_tarp_on_file(
            path, name, n_alpha=args.n_alpha, num_references=args.num_references,
            reference=args.reference, z_score=args.z_score, n_boot=args.n_boot,
            seed=args.seed, max_sims=args.max_sims)
        results[name] = res
        s = res["summary"]
        print(f"loaded {name}: {s['num_sims']} tumours x {s['num_samples']} draws  "
              f"atc {s['atc']:.4f} (signed {s['atc_signed']:+.3f})  KS p {s['ks_p']:.3g}")
    if not results:
        raise SystemExit(f"no posteriors_*.npz files in {args.in_dir}")

    rows = [{"model": n, "alpha": float(a), "ecp": float(e), "ecp_lo": float(lo),
             "ecp_hi": float(hi)}
            for n, res in results.items()
            for a, e, lo, hi in zip(res["alpha"], res["ecp"], res["ecp_lo"], res["ecp_hi"])]
    curve_path = os.path.join(args.out_dir, "tarp_curve.csv")
    write_csv(curve_path, rows, ["model", "alpha", "ecp", "ecp_lo", "ecp_hi"])
    print("saved ->", curve_path)

    json_path = os.path.join(args.out_dir, "tarp_summary.json")
    with open(json_path, "w") as fh:
        json.dump([res["summary"] for res in results.values()], fh, indent=2)
        fh.write("\n")
    print("saved ->", json_path)

    fig_tarp(results, os.path.join(args.out_dir, "fig_T_tarp"))

    print("\n=== TARP ===")
    for name, res in results.items():
        s = res["summary"]
        # atc_signed is a sum over half the grid, so scale it back to a mean gap before
        # calling a direction: a 0.001 average gap is the diagonal, not a verdict
        gap = s["atc_signed"] / max(args.n_alpha - args.n_alpha // 2, 1)
        verdict = ("on the diagonal" if abs(gap) < 0.01 else
                   "too wide" if gap > 0 else "over-confident")
        print(f"{LABEL.get(name, name):>26}  atc {s['atc']:.4f}   signed {s['atc_signed']:+.3f} "
              f"({verdict})   KS p {s['ks_p']:.3g}")


if __name__ == "__main__":
    main()
