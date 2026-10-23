"""Prior predictive check (PPC, project plan step 4): every real cohort vs one simulated dataset.

Usage (from ``code/src``)::

    python -m ppc.run_ppc --sim <sim_root> --label F10 [--out <dir>] [--figdir <dir>] [--n-perm 100]
                          [--primary auto|all|one]

Primary sample (pre-registered 2026-10-23): ``--primary auto`` (default) uses A-one (first completed
trial per theta, independent prior draws) when the dataset has >= 100 thetas, else A-all. The primary
sample drives stages 3, 4, 5 and 7 below (and B follows it); A-all always feeds the variance split,
and with primary A-one it is also checked as ``marginal_all_secondary.csv``.

Stages (one progress line each), all outputs under ``--out`` (default
``reports/ppc/<label>``) and figures under ``--figdir`` (default ``reports/figures``,
``ppc_<label>_*.png``):

1. load the 18 step-1 real cohorts (:func:`ppc.step1_change_sizes.load_cohorts`), plus PCAWG
   ``block_cn_cw``; load the simulation: version A = largest clone per trial (X_cw = bulk), all
   trials and trial 1 per theta; version B = :func:`ppc.stats.sim_observed_profile` per trial
   (X_cw = bulk). The structural mask is applied to every profile (int and float) first.
2. :func:`ppc.stats.cohort_stats` per cohort -> ``stats_<cohort>.csv`` + ``stats_summary.csv``.
3. marginal checks vs sim A-all and sim B-all -> ``marginal_<cohort>_<A|B>.csv``,
   ``marginal_all.csv``, ``marginal_flags_<A|B>.md``.
4. joint tests per target cohort vs sim A-all -> ``joint_<cohort>.json``, ``joint_summary.csv``.
5. per-arm / per-block frequencies -> ``arm_freq.csv``, ``arm_agreement.csv``, ``block_freq_*.csv``.
6. variance split of sim A-all over theta -> ``variance_split.csv``.
7. decisions -> ``decisions.csv``, ``flag_counts.csv``, the gate line.
8. figures; 9. the factual note ``ppc_results.md``.

Baseline CN is 2 for every profile (non-WGD real cohorts; simulated largest clones are expected to
have nWGD = 0, which is checked and reported as a caveat if violated).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ppc import ppc_tests as pt
from ppc.grid import Grid, load_grid
from ppc.real_loaders import load_pcawg
from ppc.sim_loaders import load_sim_cohort
from ppc.stats import (CW_COLUMNS, STAT_COLUMNS, STAT_KNOBS, apply_structural_mask, arm_frequencies,
                       block_frequencies, cohort_stats, sim_observed_profile)
from ppc.step1_change_sizes import (PCAWG_CONTRAST, PCAWG_TARGETS, PROJECT, REAL_ROOT, TCGA_CONTRAST,
                                    TCGA_TARGETS, TYPE_COLOURS, _ecdf, _style, load_cohorts, md_table)

MASK_PATH = f"{PROJECT}/reports/step1/structural_mask.npy"
FIG_DIR = f"{PROJECT}/reports/figures"
SEED = 0
NON_CW = [c for c in STAT_COLUMNS if c not in CW_COLUMNS]
CN_STATES = ["cn_frac_0", "cn_frac_1", "cn_frac_2", "cn_frac_3", "cn_frac_4", "cn_frac_5plus"]
PRIMARY_MIN_THETA = 100  # auto rule: A-one is the primary sample from this many thetas on


def resolve_primary(choice: str, n_theta: int) -> str:
    """'one' or 'all' for ``--primary`` ``choice``; 'auto' = 'one' iff n_theta >= PRIMARY_MIN_THETA."""
    if choice == "auto":
        return "one" if n_theta >= PRIMARY_MIN_THETA else "all"
    return choice


# ------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the CLI (``--sim``, ``--label``, ``--out``, ``--figdir``, ``--n-perm``)."""
    ap = argparse.ArgumentParser(prog="python -m ppc.run_ppc",
                                 description="Prior predictive check: real cohorts vs a simulated dataset.")
    ap.add_argument("--sim", required=True, help="simulated dataset root (has manifest.json)")
    ap.add_argument("--label", required=True, help="dataset label used in output names, e.g. F10")
    ap.add_argument("--out", default=None, help="output folder (default reports/ppc/<label>)")
    ap.add_argument("--figdir", default=FIG_DIR, help="figure folder (default reports/figures)")
    ap.add_argument("--n-perm", type=int, default=100, help="C2ST label permutations (default 100)")
    ap.add_argument("--primary", choices=("auto", "all", "one"), default="auto",
                    help=f"simulated sample for marginals / joint tests / arm agreement / decisions: A-all "
                         f"(every trial) or A-one (one trial per theta); auto = one iff >= {PRIMARY_MIN_THETA} "
                         "thetas (default auto)")
    return ap


# ------------------------------------------------------------------------------------- helpers
def flag_code(flag_q: bool, flag_z: bool) -> str:
    """'q', 'z', 'qz' or '' for one marginal cell."""
    return ("q" if flag_q else "") + ("z" if flag_z else "")


def build_flag_matrix(marginal_long: pd.DataFrame, cohorts: list[str] | None = None,
                      columns: list[str] | None = None) -> pd.DataFrame:
    """Cohort x statistic matrix of flag codes from a long marginal table.

    ``marginal_long`` needs columns cohort, statistic, flag_q, flag_z (one row per cohort and
    statistic). Cells absent from the table (e.g. cw statistics for TCGA) are ''. Row / column
    order follows ``cohorts`` / ``columns`` (default: order of first appearance).
    """
    cohorts = cohorts or list(dict.fromkeys(marginal_long["cohort"]))
    columns = columns or list(dict.fromkeys(marginal_long["statistic"]))
    M = pd.DataFrame("", index=pd.Index(cohorts, name="cohort"), columns=columns, dtype=object)
    for _, r in marginal_long.iterrows():
        if r["cohort"] in M.index and r["statistic"] in M.columns:
            M.loc[r["cohort"], r["statistic"]] = flag_code(bool(r["flag_q"]), bool(r["flag_z"]))
    return M


DEGENERATE_LABEL = "degenerate (sd_sim = 0)"


def degenerate_stats(sim_df: pd.DataFrame, columns) -> list[str]:
    """Statistics whose finite simulated values have sd 0 (constant, or fewer than 2 finite values).

    For these ``marginal_checks`` returns z = +-inf whenever a real mean differs, so their z flags
    are kept but reported separately ("degenerate (sd_sim = 0)"), not in the flag totals.
    """
    out = []
    for c in columns:
        v = sim_df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2 or float(v.std(ddof=1)) == 0.0:
            out.append(c)
    return out


def flag_matrix_md(M: pd.DataFrame, abbrev: dict[str, str] | None = None,
                   degenerate: list[str] | tuple = ()) -> str:
    """Markdown rendering of a flag matrix (cohort rows, statistic columns, '.' for no flag).

    Columns in ``degenerate`` get the header suffix "degenerate (sd_sim = 0)"; their flags are shown
    but counted in a separate "n degenerate flagged" column, not in "n flagged".
    """
    cols = list(M.columns)
    deg = set(degenerate)
    head = [(abbrev.get(c, c) if abbrev else c) + (f" [{DEGENERATE_LABEL}]" if c in deg else "") for c in cols]
    extra = " n degenerate flagged |" if deg else ""
    lines = ["| cohort | " + " | ".join(head) + " | n flagged |" + extra,
             "|---|" + "---|" * (len(cols) + 1 + bool(deg))]
    for coh, r in M.iterrows():
        cells = [v if v else "." for v in r.tolist()]
        n_ok = sum(1 for c, v in r.items() if v and c not in deg)
        n_dg = sum(1 for c, v in r.items() if v and c in deg)
        lines.append(f"| {coh} | " + " | ".join(cells) + f" | {n_ok} |" + (f" {n_dg} |" if deg else ""))
    return "\n".join(lines)


def _jsonable(o):
    """Recursively convert numpy / pandas objects to JSON-serialisable Python objects."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summarise_stats(df: pd.DataFrame, cohort: str, role: str) -> list[dict]:
    """Long summary rows (median, mean, p10, p90, n_finite) of every statistic of one cohort."""
    rows = []
    for c in STAT_COLUMNS:
        v = df[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        nan = float("nan")
        rows.append({"cohort": cohort, "role": role, "n": len(df), "statistic": c, "n_finite": int(v.size),
                     "median": float(np.median(v)) if v.size else nan,
                     "mean": float(v.mean()) if v.size else nan,
                     "p10": float(np.percentile(v, 10)) if v.size else nan,
                     "p90": float(np.percentile(v, 90)) if v.size else nan})
    return rows


# ------------------------------------------------------------------------------------- stage 1
def load_real(grid: Grid, mask: np.ndarray) -> dict[str, dict]:
    """The 18 step-1 real cohorts, masked: {label: {X, X_cw (PCAWG only, else None), role, source}}."""
    coh = load_cohorts(grid)
    pc = load_pcawg(f"{REAL_ROOT}/pcawg", grid)
    out = {}
    for lab, (X, role) in coh.items():
        if role == "sim":
            continue
        cw = None
        if lab.startswith("PCAWG_"):
            sub = pc.subset((pc.meta["cancer_type"] == lab[6:]).to_numpy())
            assert np.array_equal(sub.block_cn_int, X), f"{lab}: PCAWG cw rows not aligned with step 1"
            cw = apply_structural_mask(sub.block_cn_cw, mask)
        out[lab] = {"X": apply_structural_mask(X, mask), "X_cw": cw, "role": role,
                    "source": lab.split("_")[0]}
    return out


def load_sim(root: str, grid: Grid, mask: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Simulated version A (largest clone) and B (cell-weighted modal profile), masked.

    B is built one theta at a time (``keep_clones`` for one ``sim_id`` only) to bound memory.
    Returns ({'A': (X, X_cw), 'B': (X, X_cw)}, meta of A-all).
    """
    A = load_sim_cohort(root, grid)
    meta = A.meta.reset_index(drop=True)
    B = np.empty_like(A.block_cn_int, dtype=np.int64)
    for sid in pd.unique(meta["sim_id"]):
        s = load_sim_cohort(root, grid, keep_clones=True, sim_ids=[int(sid)])
        rows = np.flatnonzero(meta["sim_id"].to_numpy() == sid)
        assert s.meta["trial"].tolist() == meta.loc[rows, "trial"].tolist(), f"sim{sid}: trial order differs"
        for r, cl in zip(rows, s.clones):
            B[r] = sim_observed_profile(cl["block_cn"], cl["popsize"])
        del s
    meta["B_blocks_differ"] = (B != A.block_cn_int).sum(axis=1)
    cw = apply_structural_mask(A.block_cn_mean, mask)
    return ({"A": (apply_structural_mask(A.block_cn_int, mask), cw),
             "B": (apply_structural_mask(B, mask), cw)}, meta)


# ------------------------------------------------------------------------------------- stage 4
def run_joint(real_df: pd.DataFrame, simA: pd.DataFrame, simA_one: pd.DataFrame, cols: list[str],
              n_perm: int) -> dict:
    """joint_tests vs ``simA`` (the primary sample) on ``cols`` + energy distance vs sim A-one.

    ``cols`` must already exclude ``CW_COLUMNS`` for cohorts without a cell-weighted profile
    (TCGA / ASCAT); ``standardise`` would drop them anyway as all-NaN on the real side. Adds
    ``n_real_raw`` (rows before dropping) to the result.
    """
    j = pt.joint_tests(real_df, simA, cols, n_perm_c2st=n_perm, seed=SEED)
    za, zb, info = pt.standardise(simA_one, real_df, cols)
    j["energy_one"] = {**pt.energy_distance_test(za, zb, seed=SEED),
                       "columns_dropped": info["columns_dropped"]}
    j["n_real_raw"] = int(len(real_df))
    return j


def joint_row(cohort: str, variant: str, j: dict) -> dict:
    """One joint_summary.csv row from a run_joint dict."""
    c = j["c2st"]
    st = j["standardise"]
    return {"cohort": cohort, "variant": variant, "n_real_raw": j.get("n_real_raw", j["n_real"]),
            "n_real_rows_dropped": st["n_real_rows_dropped"], "n_real": j["n_real"],
            "n_sim_used": j["n_sim"], "n_cols_used": len(st["columns_used"]),
            "cols_dropped": ";".join(map(str, st["columns_dropped"])),
            "cols_dropped_real_nan": ";".join(map(str, st.get("columns_dropped_real_nan", []))),
            "energy_p": j["energy"]["p_value"], "mmd_p": j["mmd"]["p_value"],
            "c2st_acc_lr": c["acc_lr"], "c2st_acc_rf": c["acc_rf"], "c2st_best": c["best"],
            "c2st_acc": c["accuracy"], "c2st_n_per_side": c["n_sim"], "c2st_p": c["p_value"],
            "pca_outside_frac": j["pca"]["outside_frac"],
            "energy_p_A_one": j["energy_one"]["p_value"], "n_sim_A_one": j["energy_one"]["n_a"]}


# ------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clean(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_ecdf(stats: dict, roles: dict, simA: str, simB: str, label: str, path: str) -> None:
    """Grid of ECDFs (5 per row), one panel per statistic (sim A thick black + 2.5-97.5% band, sim B grey)."""
    plt = _plt()
    import matplotlib.patches as mpatches
    nrow = -(-len(STAT_COLUMNS) // 5)
    fig, axes = plt.subplots(nrow, 5, figsize=(24, 4.2 * nrow), dpi=150)
    axs = axes.ravel()
    for ax, s in zip(axs, STAT_COLUMNS):
        m = stats[simA][s].to_numpy(dtype=float)
        m = m[np.isfinite(m)]
        if m.size:
            lo, hi = np.percentile(m, [2.5, 97.5])
            ax.axvspan(lo, hi, color="#d6d5cf", alpha=0.6, lw=0, zorder=0)
        for lab, role in roles.items():
            if role == "sim":
                continue
            v = stats[lab][s].to_numpy(dtype=float)
            if not np.isfinite(v).any():
                continue
            x, y = _ecdf(v)
            st = _style(lab) if role == "target" else dict(color="#8a8984", ls="--", lw=1.2)
            ax.step(x, y, where="post", label=f"{lab} (n={len(v)})", **st)
        for lab, col in ((simB, "#8a8984"), (simA, "black")):
            v = stats[lab][s].to_numpy(dtype=float)
            if np.isfinite(v).any():
                x, y = _ecdf(v)
                ax.step(x, y, where="post", color=col, lw=3.0, zorder=5,
                        label=f"{lab} (n={int(np.isfinite(v).sum())})")
        ax.set_title(s, fontsize=12)
        ax.set_ylabel("ECDF")
        _clean(ax)
    for ax in axs[len(STAT_COLUMNS):]:
        ax.axis("off")
    h, l = axs[0].get_legend_handles_labels()
    h.append(mpatches.Patch(color="#d6d5cf", alpha=0.6)); l.append(f"{simA} 2.5-97.5% range")
    axs[-1].legend(h, l, loc="center", fontsize=8.5, frameon=False)
    fig.suptitle(f"PPC {label}: per-tumour statistics, real cohorts vs simulated (A = largest clone, "
                 "B = observed modal profile). TCGA solid, PCAWG dotted, contrast grey dashed", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def fig_cn_states(stats: dict, order: list[str], label: str, path: str) -> None:
    """Mean cn_frac per cohort: panel 1 cn_frac_2, panel 2 grouped bars of the other five states."""
    plt = _plt()
    means = pd.DataFrame({lab: stats[lab][CN_STATES].mean() for lab in order}).T
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(18, 10), dpi=150, gridspec_kw={"height_ratios": [1, 1.6]})
    x = np.arange(len(order))
    a1.bar(x, means["cn_frac_2"], color="#8a8984", width=0.7)
    for i, v in enumerate(means["cn_frac_2"]):
        a1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    a1.set_ylabel("mean cn_frac_2"); a1.set_ylim(0, 1.08); a1.set_xticks(x); a1.set_xticklabels([])
    others = [c for c in CN_STATES if c != "cn_frac_2"]
    w = 0.16
    for k, c in enumerate(others):
        a2.bar(x + (k - 2) * w, means[c], width=w, color=TYPE_COLOURS[k], label=c)
    a2.set_ylabel("mean fraction of unmasked blocks")
    a2.set_xticks(x); a2.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
    a2.legend(frameon=False, ncol=5)
    for ax in (a1, a2):
        _clean(ax)
    fig.suptitle(f"PPC {label}: mean fraction of blocks per total-CN state", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_arm_freq(freq: dict, order: list[str], label: str, path: str) -> None:
    """Two annotated heatmaps (gain, loss frequency): 39 arms x cohorts."""
    plt = _plt()
    arms = freq[order[0]].index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(26, 15), dpi=150)
    for ax, col, cmap in ((axes[0], "gain_freq", "Reds"), (axes[1], "loss_freq", "Blues")):
        M = np.array([freq[lab][col].reindex(arms).to_numpy(dtype=float) for lab in order]).T
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=max(0.5, float(np.nanmax(M))))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] > 0.6 * im.get_clim()[1] else "black")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=8)
        ax.set_title(f"arm {col.split('_')[0]} frequency (%)", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f"PPC {label}: per-arm Taylor gain / loss call frequency", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_block_freq(bf: dict, grid: Grid, label: str, path: str) -> None:
    """Per-block gain / loss frequency along the genome for sim A and pooled real targets."""
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(20, 9), dpi=150, sharex=True)
    styles = [("#000000", 2.0), ("#2a78d6", 1.3), ("#eb6834", 1.3)]
    for ax, col in zip(axes, ("gain_freq", "loss_freq")):
        for (name, df), (c, lw) in zip(bf.items(), styles):
            ax.plot(np.arange(grid.D), df[col].to_numpy(dtype=float), color=c, lw=lw, label=name)
        for ch in grid.chrom_order:
            ax.axvline(grid.offsets[ch] - 0.5, color="#b5b4ad", lw=0.6)
        ax.set_ylabel(col)
        _clean(ax)
    mids = [grid.offsets[c] + grid.nregions[c] / 2 for c in grid.chrom_order]
    axes[1].set_xticks(mids); axes[1].set_xticklabels([c[3:] for c in grid.chrom_order], fontsize=8)
    axes[1].set_xlabel("chromosome (5 Mb blocks; masked blocks are gaps)")
    axes[0].legend(frameon=False)
    fig.suptitle(f"PPC {label}: per-block gain / loss frequency", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_pca(simA: pd.DataFrame, reals: dict, cols: list[str], label: str, path: str) -> None:
    """PC1/PC2 of sim A (standardised with the sim mean / sd) with selected real cohorts projected."""
    plt = _plt()
    from sklearn.decomposition import PCA
    za, _, info = pt.standardise(simA, simA, cols)
    pca = PCA(n_components=2).fit(za)
    S = pca.transform(za)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    ax.scatter(S[:, 0], S[:, 1], s=12, color="#8a8984", alpha=0.6, label=f"sim A (n={len(S)})")
    for k, (lab, df) in enumerate(reals.items()):
        _, zb, _ = pt.standardise(simA, df, info["columns_used"])
        R = pca.transform(zb)
        ax.scatter(R[:, 0], R[:, 1], s=12, alpha=0.7, color=TYPE_COLOURS[k], label=f"{lab} (n={len(R)})")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% of sim variance)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% of sim variance)")
    ax.legend(frameon=False)
    _clean(ax)
    ax.set_title(f"PPC {label}: PCA of {len(info['columns_used'])} standardised statistics (fit on sim A)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_variance_split(vs: pd.DataFrame, label: str, path: str) -> None:
    """Bar chart of the between-theta variance share per statistic (sorted)."""
    plt = _plt()
    d = vs.sort_values("between_share", ascending=False, na_position="last")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    x = np.arange(len(d))
    ax.bar(x, d["between_share"].fillna(0), color="#2a78d6", width=0.7)
    for i, v in enumerate(d["between_share"]):
        ax.text(i, (0 if np.isnan(v) else v) + 0.01, "NaN" if np.isnan(v) else f"{v:.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(d["statistic"], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("between-theta share of variance"); ax.set_ylim(0, 1.05)
    _clean(ax)
    ax.set_title(f"PPC {label}: variance split of sim A over theta (one-way ANOVA, group = sim_id)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------- note
def write_note(path: str, ctx: dict) -> None:
    """The factual results note (no interpretation)."""
    L, p, sec = ctx["label"], ctx["primary"], ctx["sec"]
    pa = f"A-{p}"
    if p == "one":
        rep = (f"The {ctx['n_simA_one']} simulated rows are one trial per theta (independent prior draws); "
               "`energy_p_A_one` repeats the energy test on that same sample.")
    else:
        rep = (f"The {ctx['n_simA']} simulated rows are {ctx['n_rep']} replicates of {ctx['n_theta']} thetas, "
               "not independent draws from the prior; `energy_p_A_one` is the energy-distance p-value against "
               f"A-one ({ctx['n_simA_one']} rows, one trial per theta, standardised with A-one's own mean / sd).")
    sec_md = [] if sec is None else [
        "### Secondary: vs sim A-all (pseudo-replicated; not used for decisions)", "",
        f"`marginal_all_secondary.csv` repeats the A marginal checks against all {ctx['n_simA']} A-all rows. "
        f"Flagged cohort x statistic cells excluding degenerate statistics: {sec['n_flag']} (primary A-one: "
        f"{ctx['ncell']['A'][0]}); flags on statistics degenerate in A-all "
        f"({', '.join(sec['degen']) or 'none'}): {sec['n_deg']}; cells whose flag code differs from the "
        f"A-one matrix: {sec['n_diff']} of {sec['n_cells']}.", ""]
    lines = [f"# PPC (step 4) results: {L}", "",
             f"Generated by `python -m ppc.run_ppc --sim {ctx['sim_root']} --label {L} --n-perm {ctx['n_perm']} "
             f"--primary {ctx['primary_choice']}` "
             f"on {time.strftime('%Y-%m-%d')} ({ctx['runtime']:.0f} s). Outputs in `{ctx['out']}`.", "",
             f"## Data used (primary sample, which decided: sim {pa})", "",
             f"Primary sample: sim {pa} (`--primary {ctx['primary_choice']}`; auto = A-one iff n_theta >= "
             f"{PRIMARY_MIN_THETA}; here n_theta = {ctx['n_theta']}). It drives the marginal checks (B follows "
             f"it: B-{p}), the joint tests, the per-arm agreement and the decisions; A-all feeds the variance "
             "split" + (" and the secondary marginal table." if sec is not None else "."),
             f"Structural mask `{MASK_PATH}` ({ctx['n_mask']} blocks) applied to every profile (int and "
             "cell-weighted) before any statistic. Baseline CN 2 for every profile.",
             "Real cohorts: the 18 step-1 cohorts (non-WGD, purity >= 0.3, one sample per donor; TCGA = ASCAT "
             "integer profiles, no cell-weighted profile; PCAWG = consensus clonal total CN plus Battenberg "
             "cell-weighted `block_cn_cw` as X_cw).",
             f"Simulated: `{ctx['sim_root']}`; A-all = largest clone of every completed trial "
             f"({ctx['n_simA']} rows = {ctx['n_theta']} thetas x replicates), X_cw = bulk; A-one = trial 1 per "
             f"theta ({ctx['n_simA_one']} rows); B-all = cell-weighted modal profile per trial "
             "(`sim_observed_profile`), X_cw = bulk.", "",
             md_table(ctx["ncoh"]), "",
             "## Statistics and the prior knobs they point at", "",
             md_table(pd.DataFrame({"statistic": list(STAT_COLUMNS),
                                    "knobs": [", ".join(STAT_KNOBS[s]) for s in STAT_COLUMNS],
                                    "cw_only": [s in CW_COLUMNS for s in STAT_COLUMNS]})), "",
             "Per-cohort median / mean / p10 / p90: `stats_summary.csv`.", "",
             "## Marginal checks", "",
             "Flag codes: q = real median outside the simulated 2.5-97.5% quantile range; z = |mean_real - "
             "mean_sim| / sd_sim > 3; '.' = no flag. cw statistics are tested for PCAWG cohorts only.", "",
             f"Statistics marked [{DEGENERATE_LABEL}] are constant in that simulation (sim A: "
             f"{', '.join(ctx['degenA']) or 'none'}; sim B: {', '.join(ctx['degenB']) or 'none'}): any real "
             "cohort whose mean differs gets z = +-inf. Their flags are kept in the matrices but counted "
             "separately ('n degenerate flagged') and are excluded from the flag totals and the knob ranking.",
             "",
             f"Flagged cohort x statistic cells excluding degenerate statistics: A {ctx['ncell']['A'][0]}, "
             f"B {ctx['ncell']['B'][0]}; flags on degenerate statistics: A {ctx['ncell']['A'][1]}, "
             f"B {ctx['ncell']['B'][1]}.", "",
             f"### vs sim {pa} (largest clone)", "", flag_matrix_md(ctx["flagsA"], degenerate=ctx["degenA"]), "",
             f"### vs sim B-{p} (observed modal profile)", "",
             flag_matrix_md(ctx["flagsB"], degenerate=ctx["degenB"]), ""] + sec_md + [
             "### Flag counts per statistic, knob ranking (target cohorts, out of 14; all = out of 18; "
             "degenerate statistics excluded)", "",
             md_table(ctx["flag_counts"][~ctx["flag_counts"]["degenerate_A"]]), "",
             f"### Flag counts of degenerate statistics ({DEGENERATE_LABEL} in sim A; not ranked)", "",
             md_table(ctx["flag_counts"][ctx["flag_counts"]["degenerate_A"]]), "",
             f"## Joint tests (target cohorts vs sim {pa})", "",
             f"Columns: the {len(NON_CW)} non-cw statistics (`no_cw`); PCAWG cohorts also with the 2 cw "
             "statistics (`with_cw`); TCGA cohorts never get the cw statistics. Standardised with the "
             "simulated mean / sd; columns constant / all-NaN in the simulation or all-NaN in the real "
             "cohort are dropped (`cols_dropped`; the real-side ones also in `cols_dropped_real_nan`), "
             "then rows with any NaN are dropped (`n_real_raw` - `n_real_rows_dropped` = `n_real`). "
             f"Real rows dropped per cohort / variant: {ctx['row_drops']}. Energy / MMD: 200 permutations; "
             f"C2ST: {ctx['n_perm']} permutations, larger side subsampled to the smaller; "
             "the C2ST statistic is max(LR, RF) accuracy and every permutation also takes the max of both "
             "classifiers (selection inside the null; the null RF uses 50 trees). " + rep,
             "", md_table(ctx["joint"].drop(columns=["c2st_n_per_side"])), "",
             f"## Per-arm frequency agreement (each real cohort vs sim {pa})", "",
             md_table(ctx["arm"]), "",
             "Per-block frequencies: `block_freq_sim_A.csv`, `block_freq_TCGA_targets.csv`, "
             "`block_freq_PCAWG_targets.csv`.", "",
             "## Variance split of sim A-all over theta", "",
             md_table(ctx["vs"][["statistic", "n_rows", "n_groups", "between_share", "within_share",
                                 "icc_anova"]]), "",
             f"## Decisions (decided on sim {pa}; ppc_tests.decision; A-marginals, no_cw joint)", "",
             md_table(ctx["dec"]), "",
             f"**Gate: {ctx['gate']}**", "",
             "## Figures", ""] + [f"- `{p}`" for p in ctx["figs"]] + ["", "## Caveats observed", ""] + \
            [f"- {c}" for c in ctx["caveats"]] + [""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def collect_caveats(stats: dict, roles: dict, joint: pd.DataFrame, meta: pd.DataFrame,
                    simA: str, simB: str) -> list[str]:
    """Data-driven caveat list: small / degenerate cohorts, NaN columns, dropped columns, n dependence."""
    cav = []
    n_wgd = int((meta["nWGD_largest"] > 0).sum())
    if n_wgd:
        cav.append(f"{n_wgd} simulated largest clones have nWGD > 0 but baseline 2 was used for them.")
    for lab, role in roles.items():
        df = stats[lab]
        if role != "sim" and len(df) < 30:
            cav.append(f"{lab} is small (n = {len(df)}).")
        const = [c for c in STAT_COLUMNS if df[c].notna().any() and df[c].nunique(dropna=True) == 1]
        if const:
            cav.append(f"{lab}: constant statistics: " + ", ".join(f"{c}={df[c].dropna().iloc[0]:g}" for c in const))
        nan_cols = [c for c in STAT_COLUMNS if df[c].isna().all()]
        if nan_cols and set(nan_cols) != set(CW_COLUMNS):
            cav.append(f"{lab}: all-NaN statistics: {', '.join(nan_cols)}.")
        part = [c for c in STAT_COLUMNS if df[c].isna().any() and not df[c].isna().all()]
        if part:
            cav.append(f"{lab}: some NaN rows in {', '.join(f'{c} ({int(df[c].isna().sum())})' for c in part)} "
                       "(dropped per test, never imputed).")
    cav.append(f"TCGA cohorts have no cell-weighted profile: {', '.join(CW_COLUMNS)} are NaN and are "
               "excluded from their marginal checks and joint tests.")
    cav.append(f"Non-gating (pre-registered 2026-10-22): {', '.join(sorted(pt.NON_GATING))} are reported but "
               "excluded from ppc_tests.decision (column flagged_non_gating in decisions.csv).")
    drops = sorted(set(";".join(joint["cols_dropped"].fillna("")).split(";")) - {""})
    rdrops = sorted(set(";".join(joint["cols_dropped_real_nan"].fillna("")).split(";")) - {""})
    sdrops = [c for c in drops if c not in rdrops]
    if sdrops:
        cav.append(f"Columns dropped from the joint tests as constant / all-NaN in sim A: {', '.join(sdrops)}.")
    if rdrops:
        cav.append(f"Columns dropped from some joint tests as all-NaN in the real cohort: {', '.join(rdrops)}.")
    cav.append("C2ST balances classes by subsampling the larger side, so n_per_side = min(n_real, n_sim); "
               "the energy / MMD / C2ST p-values depend on n (large TCGA cohorts have more power than "
               "PCAWG cohorts of n = 14-93); the marginal q rule uses only the real median and does not "
               "depend on n_real. C2ST n_per_side per cohort (no_cw): "
               + "; ".join(f"{r.cohort} {r.c2st_n_per_side}" for r in joint[joint["variant"] == "no_cw"]
                           .itertuples()) + ".")
    d = (stats[simA][NON_CW] - stats[simB][NON_CW]).abs()
    same = [c for c in NON_CW if (d[c].fillna(0) == 0).all()]
    simsd0 = [c for c in STAT_COLUMNS if stats[simA][c].notna().any() and stats[simA][c].nunique() == 1]
    for c in simsd0:
        nz = [l for l, r in roles.items() if r != "sim" and len(set(stats[l][c].dropna()) - {stats[simA][c].iloc[0]})]
        cav.append(f"`{c}` is constant ({stats[simA][c].iloc[0]:g}) in every sim A trial (sd_sim = 0): "
                   f"marginal_checks returns z = +-inf for every real cohort whose mean differs ({len(nz)} of "
                   f"{sum(1 for r in roles.values() if r != 'sim')}), so these cohorts carry a z flag on `{c}`; "
                   f"it is labelled {DEGENERATE_LABEL} and excluded from the flag totals and the knob ranking "
                   "(the flag still counts in ppc_tests.decision, so it still blocks overall_pass).")
    nd = meta["B_blocks_differ"].to_numpy()
    cav.append(f"Sim B differs from sim A (before masking) in >= 1 block in {int((nd > 0).sum())} of {len(nd)} "
               f"trials (median {np.median(nd):g}, max {nd.max()} blocks).")
    p_floor = 1 / 201
    nfl = int((joint[["energy_p", "mmd_p"]] <= p_floor + 1e-12).to_numpy().sum())
    cav.append(f"Energy / MMD use 200 permutations, so the smallest attainable p-value is 1/201 = {p_floor:.5f}; "
               f"{nfl} of {2 * len(joint)} energy / MMD p-values sit at that floor. The C2ST p-value floor is "
               "1/(n_perm + 1).")
    cav.append(f"Sim A vs B: {len(same)} of {len(NON_CW)} non-cw statistics are identical in every trial"
               + (f" ({', '.join(same)})" if same else "") + ".")
    return cav


# ------------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    """CLI entry point; see the module docstring."""
    args = build_parser().parse_args(argv)
    t0 = time.time()
    L = args.label
    out = args.out or f"{PROJECT}/reports/ppc/{L}"
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    grid = load_grid(os.path.join(args.sim, "manifest.json"))
    mask = np.load(MASK_PATH).astype(bool)
    assert mask.shape == (grid.D,), "structural mask does not match the grid"

    # 1 -- load
    real = load_real(grid, mask)
    sim, meta = load_sim(args.sim, grid, mask)
    simA, simA1, simB = f"SIM_{L}_A_all", f"SIM_{L}_A_one", f"SIM_{L}_B_all"
    roles = {lab: d["role"] for lab, d in real.items()}
    roles.update({simA: "sim", simA1: "sim", simB: "sim"})
    targets = [l for l, r in roles.items() if r == "target"]
    contrasts = [l for l, r in roles.items() if r == "contrast"]
    primary = resolve_primary(args.primary, int(meta["sim_id"].nunique()))
    print(f"[1] loaded {len(real)} real cohorts ({sum(len(d['X']) for d in real.values())} tumours), "
          f"sim A/B {len(meta)} trials over {meta['sim_id'].nunique()} thetas; primary sample = A-{primary} "
          f"(--primary {args.primary}) ({time.time() - t0:.0f} s)")

    # 2 -- statistics
    stats = {lab: cohort_stats(d["X"], grid, 2, d["X_cw"]) for lab, d in real.items()}
    stats[simA] = cohort_stats(sim["A"][0], grid, 2, sim["A"][1])
    one = (meta["trial"] == meta.groupby("sim_id")["trial"].transform("min")).to_numpy()  # first completed
    stats[simA1] = stats[simA].loc[one].reset_index(drop=True)
    stats[simB] = cohort_stats(sim["B"][0], grid, 2, sim["B"][1])
    P, PB, prows = simA, simB, np.ones(len(meta), dtype=bool)   # primary A / B labels and sim rows
    if primary == "one":
        P, PB, prows = simA1, f"SIM_{L}_B_one", one
        stats[PB] = stats[simB].loc[one].reset_index(drop=True)
        roles[PB] = "sim"
    summ = []
    for lab, df in stats.items():
        df.to_csv(os.path.join(out, f"stats_{lab}.csv"), index=False)
        summ += summarise_stats(df, lab, roles[lab])
    pd.DataFrame(summ).to_csv(os.path.join(out, "stats_summary.csv"), index=False)
    print(f"[2] statistics for {len(stats)} cohorts ({time.time() - t0:.0f} s)")

    # 3 -- marginal checks
    marg, mlong = {"A": {}, "B": {}}, []
    mcols = {lab: list(STAT_COLUMNS) if real[lab]["X_cw"] is not None else NON_CW for lab in real}
    for lab in real:
        cols = mcols[lab]
        for v, sl in (("A", P), ("B", PB)):
            m = pt.marginal_checks(stats[lab], stats[sl], cols)
            m.to_csv(os.path.join(out, f"marginal_{lab}_{v}.csv"), index=False)
            marg[v][lab] = m
            mlong.append(m.assign(cohort=lab, role=roles[lab], sim=v))
    mlong = pd.concat(mlong, ignore_index=True)
    mlong = mlong[["cohort", "role", "sim"] + [c for c in mlong.columns if c not in ("cohort", "role", "sim")]]
    mlong.to_csv(os.path.join(out, "marginal_all.csv"), index=False)
    flags, degen = {}, {"A": degenerate_stats(stats[P], STAT_COLUMNS),
                        "B": degenerate_stats(stats[PB], STAT_COLUMNS)}
    ncell = {}
    for v in ("A", "B"):
        flags[v] = build_flag_matrix(mlong[mlong["sim"] == v], list(real), list(STAT_COLUMNS))
        with open(os.path.join(out, f"marginal_flags_{v}.md"), "w") as f:
            f.write(f"# Marginal flags vs sim {v}-{primary} ({L})\n\n" + flag_matrix_md(flags[v], degenerate=degen[v])
                    + "\n\nColumns marked [" + DEGENERATE_LABEL + "] are constant in the simulation; their "
                    "flags are counted separately.\n")
        ok_cols = [c for c in STAT_COLUMNS if c not in degen[v]]
        ncell[v] = (int((flags[v][ok_cols] != "").values.sum()),
                    int((flags[v][degen[v]] != "").values.sum()) if degen[v] else 0)
    print(f"[3] marginal checks: A {ncell['A'][0]} / B {ncell['B'][0]} flagged cohort x statistic cells "
          f"excluding degenerate (sd_sim = 0) statistics; degenerate-statistic flags A {ncell['A'][1]} "
          f"({', '.join(degen['A']) or 'none'}) / B {ncell['B'][1]} ({', '.join(degen['B']) or 'none'}) "
          f"({time.time() - t0:.0f} s)")
    sec = None   # secondary A-all marginal table when A-one is primary
    if primary == "one":
        slong = pd.concat([pt.marginal_checks(stats[lab], stats[simA], mcols[lab]).assign(
            cohort=lab, role=roles[lab], sim="A_all") for lab in real], ignore_index=True)
        slong = slong[["cohort", "role", "sim"] + [c for c in slong.columns if c not in ("cohort", "role", "sim")]]
        slong.to_csv(os.path.join(out, "marginal_all_secondary.csv"), index=False)
        sM, sdeg = build_flag_matrix(slong, list(real), list(STAT_COLUMNS)), degenerate_stats(stats[simA], STAT_COLUMNS)
        sec = {"degen": sdeg, "n_flag": int((sM[[c for c in STAT_COLUMNS if c not in sdeg]] != "").values.sum()),
               "n_deg": int((sM[sdeg] != "").values.sum()) if sdeg else 0,
               "n_diff": int((sM != flags["A"]).values.sum()), "n_cells": int(sM.size)}
        print(f"[3b] secondary marginal checks vs A-all: {sec['n_flag']} flagged non-degenerate cells, "
              f"{sec['n_diff']} cells differ from A-one")

    # 4 -- joint tests
    jrows, jdict = [], {}
    for lab in targets:
        # no_cw: never the cw statistics; with_cw only for cohorts that have a cw profile (PCAWG)
        variants = {"no_cw": NON_CW}
        if real[lab]["X_cw"] is not None:
            variants["with_cw"] = list(STAT_COLUMNS)
        res = {}
        for vname, vcols in variants.items():
            try:
                res[vname] = run_joint(stats[lab], stats[P], stats[simA1], vcols, args.n_perm)
            except ValueError as e:
                raise ValueError(f"joint tests {lab} ({vname}): {e}") from e
        jdict[lab] = res
        with open(os.path.join(out, f"joint_{lab}.json"), "w") as f:
            json.dump(_jsonable(res), f, indent=1)
        jrows += [joint_row(lab, k, j) for k, j in res.items()]
    joint = pd.DataFrame(jrows).assign(primary_sample=f"A_{primary}")
    joint.to_csv(os.path.join(out, "joint_summary.csv"), index=False)
    print(f"[4] joint tests: {len(joint)} runs, C2ST n_perm={args.n_perm} ({time.time() - t0:.0f} s)")

    # 5 -- arm / block frequencies
    freq = {lab: arm_frequencies(d["X"], grid) for lab, d in real.items()}
    freq[P] = arm_frequencies(sim["A"][0][prows], grid)
    freq[PB] = arm_frequencies(sim["B"][0][prows], grid)
    pd.concat({lab: f for lab, f in freq.items()}, names=["cohort"]).reset_index().to_csv(
        os.path.join(out, "arm_freq.csv"), index=False)
    arows = []
    for lab in real:
        a = pt.arm_frequency_agreement(freq[lab], freq[P])
        arows.append({"cohort": lab, "role": roles[lab], "n_arms": a["n_arms_common"],
                      **{f"{k.split('_')[0]}_{m}": a[k][m] for k in ("gain_freq", "loss_freq")
                         for m in ("spearman", "pearson", "mean_abs_diff")}})
    arm = pd.DataFrame(arows)
    arm.to_csv(os.path.join(out, "arm_agreement.csv"), index=False)
    bf = {"sim_A": block_frequencies(sim["A"][0][prows], grid, mask=mask),
          "TCGA_targets": block_frequencies(np.vstack([real[l]["X"] for l in targets if l.startswith("TCGA_")]),
                                            grid, mask=mask),
          "PCAWG_targets": block_frequencies(np.vstack([real[l]["X"] for l in targets if l.startswith("PCAWG_")]),
                                             grid, mask=mask)}
    for k, df in bf.items():
        df.to_csv(os.path.join(out, f"block_freq_{k}.csv"))
    print(f"[5] arm agreement for {len(arm)} cohorts, block frequencies for {len(bf)} sets ({time.time() - t0:.0f} s)")

    # 6 -- variance split
    vs = pt.variance_split(stats[simA].assign(sim_id=meta["sim_id"].to_numpy()), STAT_COLUMNS)
    vs.to_csv(os.path.join(out, "variance_split.csv"), index=False)
    print(f"[6] variance split over {meta['sim_id'].nunique()} thetas ({time.time() - t0:.0f} s)")

    # 7 -- decisions
    drows = []
    for lab in targets:
        d = pt.decision(lab, marg["A"][lab], jdict[lab]["no_cw"])
        fl = d["flagged_statistics"]
        drows.append({**d, "n_flagged": len(fl),
                      "n_flagged_non_degenerate": sum(1 for x in fl if x not in degen["A"]),
                      "flagged_statistics": ";".join(fl),
                      "flagged_non_gating": ";".join(d["flagged_non_gating"]),
                      "flagged_degenerate": ";".join(x for x in fl if x in degen["A"])})
    dec = pd.DataFrame(drows).assign(primary_sample=f"A_{primary}")
    dec.to_csv(os.path.join(out, "decisions.csv"), index=False)
    k = int(dec["overall_pass"].sum())
    best = dec.loc[dec["c2st_accuracy"].idxmin()]
    k_nd = int((dec["c2st_pass"] & (dec["n_flagged_non_degenerate"] == 0) & (dec["n_real"] > 0)).sum())
    gate = (f"overall_pass for {k} of {len(dec)} target cohorts; best (lowest) C2ST accuracy = "
            f"{best['c2st_accuracy']:.3f} ({best['cohort']}); ignoring flags on degenerate (sd_sim = 0) "
            f"statistics ({', '.join(degen['A']) or 'none'}): {k_nd} of {len(dec)}")
    fc = []
    for s in STAT_COLUMNS:
        fA_t = flags["A"].loc[targets, s]
        fc.append({"statistic": s, "knobs": ", ".join(STAT_KNOBS[s]),
                   "n_target_flagged_A": int((fA_t != "").sum()),
                   "n_target_q_A": int(fA_t.str.contains("q").sum()),
                   "n_target_z_A": int(fA_t.str.contains("z").sum()),
                   "n_all_flagged_A": int((flags["A"][s] != "").sum()),
                   "n_target_flagged_B": int((flags["B"].loc[targets, s] != "").sum()),
                   "n_tested_targets": int(sum(1 for l in targets if s not in CW_COLUMNS
                                               or real[l]["X_cw"] is not None)),
                   "degenerate_A": s in degen["A"], "degenerate_B": s in degen["B"]})
    # knob ranking: non-degenerate statistics first (ranked), degenerate (sd_sim = 0) ones after
    flag_counts = pd.DataFrame(fc).sort_values(["degenerate_A", "n_target_flagged_A", "n_all_flagged_A"],
                                               ascending=[True, False, False], kind="stable")
    flag_counts.to_csv(os.path.join(out, "flag_counts.csv"), index=False)
    print(f"[7] decisions: {gate} ({time.time() - t0:.0f} s)")

    # 8 -- figures
    fp = lambda n: os.path.join(args.figdir, f"ppc_{L}_{n}.png")  # noqa: E731
    order = targets + contrasts + [P, PB]
    fig_ecdf(stats, roles, P, PB, L, fp("ecdf"))
    fig_cn_states(stats, order, L, fp("cn_states"))
    fig_arm_freq(freq, list(real) + [P], L, fp("arm_freq"))
    fig_block_freq({f"sim A ({L})": bf["sim_A"], "TCGA targets pooled": bf["TCGA_targets"],
                    "PCAWG targets pooled": bf["PCAWG_targets"]}, grid, L, fp("block_freq"))
    fig_pca(stats[P], {l: stats[l] for l in ("TCGA_OV", "TCGA_BRCA", "TCGA_GBM") if l in stats},
            NON_CW, L, fp("pca"))
    fig_variance_split(vs, L, fp("variance_split"))
    figs = [fp(n) for n in ("ecdf", "cn_states", "arm_freq", "block_freq", "pca", "variance_split")]
    print(f"[8] figures: {len(figs)} written to {args.figdir} ({time.time() - t0:.0f} s)")

    # 9 -- note
    ncoh = pd.DataFrame([{"cohort": l, "role": roles[l], "n": len(stats[l]),
                          "has_cw": bool(l in real and real[l]["X_cw"] is not None) or roles[l] == "sim"}
                         for l in stats])
    n_rep = int(meta.groupby("sim_id").size().max())
    ctx = dict(label=L, sim_root=args.sim, n_perm=args.n_perm, out=out, runtime=time.time() - t0,
               primary=primary, primary_choice=args.primary, sec=sec,
               n_mask=int(mask.sum()), n_simA=len(meta), n_simA_one=int(one.sum()),
               n_theta=int(meta["sim_id"].nunique()), n_rep=n_rep, ncoh=ncoh,
               flagsA=flags["A"], flagsB=flags["B"], flag_counts=flag_counts, joint=joint, arm=arm, vs=vs,
               degenA=degen["A"], degenB=degen["B"], ncell=ncell,
               dec=dec.drop(columns=["energy_p", "mmd_p", "pca_outside_frac"]), gate=gate, figs=figs,
               row_drops="; ".join(f"{r.cohort} {r.variant} {r.n_real_rows_dropped}/{r.n_real_raw}"
                                   for r in joint.itertuples()),
               caveats=collect_caveats(stats, roles, joint, meta, P, PB))
    note = os.path.join(out, "ppc_results.md")
    write_note(note, ctx)
    print(f"[9] note: {note}; done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
