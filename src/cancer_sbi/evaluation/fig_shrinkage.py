#!/usr/bin/env python
"""Figure D: where the systematic error in the posterior mean lives.

The r2-minus-R2 gap says the posterior means are affinely wrong, but not how.
Writing s = sd(mean)/sd(theta) and assuming negligible global bias,

    mse/var(theta) = 1 + s^2 - 2 r s = 1 - R^2   =>   s = r +/- sqrt(r^2 - R^2)

so sqrt(r2 - R2) is the discriminant and there are TWO admissible roots. This
script settles which one by measuring s directly from the saved posterior means,
and draws the regression that produced it.

Poster build:  python fig_shrinkage.py --poster
"""
import argparse, json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, to_rgb

MODEL_ORDER = ["clonemlp", "cloneatt", "dominantclone", "armtoken", "hybrid"]
LABEL = {"clonemlp": "CloneMLP-NPE", "cloneatt": "CloneAtt-NPE",
         "dominantclone": "DominantClone-NPE", "armtoken": "ArmToken-NPE",
         "hybrid": "Hybrid-NPE"}
C_MLP, C_ATT, C_DOM = "#4c3fa5", "#17a673", "#e8622a"
COLOUR = {"clonemlp": C_MLP, "cloneatt": C_ATT, "dominantclone": C_DOM, "armtoken": "#6a3d9a", "hybrid": "#e31a1c"}
INK, MUTED, GRID, RULE = "#141412", "#63625c", "#dcdad2", "#4a4a46"
WARN, GOOD = "#a8443f", "#12714f"


def tint(colour, f):
    """Blend `colour` toward white by fraction f (1 = white)."""
    r, g, b = to_rgb(colour)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def density_cmap(colour):
    """White -> pale model colour -> model colour. Keeps each panel on-brand."""
    return LinearSegmentedColormap.from_list(
        "dens", [(0.0, "#ffffff"), (0.12, tint(colour, 0.90)),
                 (0.45, tint(colour, 0.55)), (1.0, colour)])


def load(in_dir, name, run_tag=None):
    suffix = f"_{run_tag}" if run_tag else ""
    p = os.path.join(in_dir, f"posteriors_{name}{suffix}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    return dict(t=d["theta_true"].astype(np.float64),
                m=d["post_mean"].astype(np.float64),
                prior_sd=float(meta.get("prior_sd", 0.2)))


def per_arm_stats(t, m):
    """Per-arm OLS of truth on mean, and the spread ratio s = sd(m)/sd(t)."""
    out = []
    for j in range(t.shape[1]):
        tj, mj = t[:, j], m[:, j]
        out.append(dict(
            s=tj.std(ddof=1) and mj.std(ddof=1) / max(tj.std(ddof=1), 1e-12),
            b=np.cov(tj, mj, ddof=1)[0, 1] / max(mj.var(ddof=1), 1e-12),
            r=np.corrcoef(tj, mj)[0, 1]))
    return out


def verdict_for(s_med, b_pool):
    # `s` very small means the estimate is near-constant, which dominates any
    # slope reading. Keep that threshold tight (0.30) so it fires only for a
    # genuinely frozen estimate, and let the slope speak for everything else --
    # otherwise two different failure modes get labelled with the same phrase.
    if s_med < 0.30:
        return "the mean barely moves", WARN
    if b_pool < 0.95:
        return "means pushed too far out", WARN
    if b_pool > 1.05:
        return "means pulled toward zero", WARN
    return "no systematic scale error", GOOD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="out")
    ap.add_argument("--run-tag", default=None,
                    help="read posteriors_<model>_<tag>.npz (sample_posteriors --run-tag)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--poster", action="store_true",
                    help="drop the figure title and enlarge type for a poster block")
    args = ap.parse_args()

    P = args.poster
    fs_name = 15.5 if P else 11.0
    fs_big = 30.0 if P else 21.0
    fs_verd = 11.0 if P else 8.0
    fs_axis = 11.5 if P else 8.6
    fs_tick = 10.0 if P else 7.6
    fs_note = 9.5 if P else 7.0
    fs_inline = 10.0 if P else 7.2

    runs = {n: load(args.in_dir, n, args.run_tag) for n in MODEL_ORDER}
    runs = {k: v for k, v in runs.items() if v is not None}
    names = [n for n in MODEL_ORDER if n in runs]
    if not names:
        raise SystemExit(f"no posteriors_*.npz found in {args.in_dir}")

    pw = 4.5 if P else 3.6
    fig, axes = plt.subplots(1, len(names), figsize=(pw * len(names), pw * 1.16),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    lim = 0.72
    rows = []
    for ax, name in zip(axes, names):
        R = runs[name]
        t, m = R["t"], R["m"]
        col = COLOUR[name]
        st = per_arm_stats(t, m)
        s_med = float(np.median([x["s"] for x in st]))
        b_med = float(np.median([x["b"] for x in st]))

        tf, mf = t.ravel(), m.ravel()
        b_pool = np.cov(tf, mf, ddof=1)[0, 1] / mf.var(ddof=1)
        a_pool = tf.mean() - b_pool * mf.mean()
        r_pool = np.corrcoef(tf, mf)[0, 1]
        R2 = 1.0 - ((tf - mf) ** 2).sum() / ((tf - tf.mean()) ** 2).sum()
        gap = r_pool ** 2 - R2
        d = math.sqrt(max(gap, 0.0))
        rows.append(dict(model=name, s_measured=s_med, b_truth_on_mean=b_med,
                         b_pooled=b_pool, r=r_pool, r2=r_pool ** 2, true_r2=R2,
                         gap=gap, root_low=r_pool - d, root_high=r_pool + d))

        ax.set_facecolor("white")
        xs = np.array([-lim, lim])
        fit = a_pool + b_pool * xs

        # The wedge between the fit and the identity IS the systematic error:
        # it opens out with |theta|, which is the point the panel has to make.
        ax.fill_between(xs, fit, xs, color=col, alpha=0.15, lw=0, zorder=3)

        ax.hexbin(mf, tf, gridsize=44, extent=(-lim, lim, -lim, lim),
                  cmap=density_cmap(col), bins="log", mincnt=1, linewidths=0,
                  zorder=2)
        ax.plot(xs, xs, ls=(0, (5, 3)), color=RULE, lw=1.5, zorder=5)
        ax.plot(xs, fit, color=col, lw=3.0, zorder=6,
                solid_capstyle="round")

        # Inline labels instead of a legend box. The two lines cross near the
        # origin and diverge both ways, so label identity on the RIGHT and the fit
        # on the LEFT, each nudged to the side away from the other line. Without
        # this the labels collide whenever the slope is close to 1.
        halo = [pe.withStroke(linewidth=3.2, foreground="white")]
        x_r, x_l = 0.585, -0.555
        fit_above_right = (a_pool + b_pool * x_r) > x_r
        ax.text(x_r, x_r, "perfect", rotation=45, rotation_mode="anchor",
                ha="center", va="top" if fit_above_right else "bottom",
                fontsize=fs_inline, color=RULE, fontweight="bold", zorder=8,
                path_effects=halo)
        fit_above_left = (a_pool + b_pool * x_l) > x_l
        ax.text(x_l, a_pool + b_pool * x_l, "fitted",
                rotation=math.degrees(math.atan(b_pool)), rotation_mode="anchor",
                ha="center", va="bottom" if fit_above_left else "top",
                fontsize=fs_inline, color=col, fontweight="bold", zorder=8,
                path_effects=halo)

        # Upper-left corner is always empty (the cloud is central and diagonal).
        ax.text(0.045, 0.975, "slope", transform=ax.transAxes, ha="left", va="top",
                fontsize=fs_note, color=MUTED, fontweight="bold", zorder=9,
                path_effects=halo)
        ax.text(0.038, 0.935, f"{b_pool:.2f}", transform=ax.transAxes, ha="left",
                va="top", fontsize=fs_big, color=col, fontweight="bold", zorder=9,
                path_effects=[pe.withStroke(linewidth=5.0, foreground="white")])
        vtxt, vcol = verdict_for(s_med, b_pool)
        ax.text(0.045, 0.775, vtxt, transform=ax.transAxes, ha="left", va="top",
                fontsize=fs_verd, color=vcol, fontweight="bold", zorder=9,
                path_effects=halo)

        ax.text(0.965, 0.045, f"sd(mean) / sd($\\theta$) = {s_med:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=fs_note,
                color=MUTED, zorder=9,
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

        ax.set_title(LABEL[name], fontsize=fs_name, fontweight="bold", color=col,
                     loc="left", pad=10)
        ax.set_xlabel("posterior mean", fontsize=fs_axis, color=INK)
        ax.set_aspect("equal")
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.tick_params(labelsize=fs_tick, colors=MUTED, length=3, width=0.9)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
            ax.spines[sp].set_linewidth(0.9)

    axes[0].set_ylabel(r"true $\theta$", fontsize=fs_axis, color=INK)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    if not P:
        fig.suptitle("Posterior mean against truth: where the systematic error lives",
                     fontsize=12.5, fontweight="bold", color=INK, x=0.015, ha="left",
                     y=1.015)
    fig.subplots_adjust(wspace=0.14)

    base = os.path.join(args.out_dir, "fig_D_shrinkage" + ("_poster" if P else ""))
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", facecolor="white", bbox_inches="tight",
                    dpi=320 if ext == "png" else None)
    plt.close(fig)
    print("saved ->", base + ".png")

    if not P:
        hdr = ["model", "s_measured", "b_truth_on_mean", "b_pooled", "r", "r2",
               "true_r2", "gap", "root_low", "root_high"]
        csv_path = os.path.join(args.out_dir, "shrinkage.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(hdr) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k]
                                 for k in hdr) + "\n")
        print("saved ->", csv_path)

    print()
    for r in rows:
        print("%20s  pooled slope %.3f   per-arm slope %.3f   s %.3f   roots %.3f / %.3f"
              % (LABEL[r["model"]], r["b_pooled"], r["b_truth_on_mean"],
                 r["s_measured"], r["root_low"], r["root_high"]))


if __name__ == "__main__":
    main()
