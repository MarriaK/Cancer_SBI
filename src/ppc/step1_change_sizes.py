"""PPC step 1: does the mix of change sizes look real?

Counts whole-chromosome, arm-level and focal copy-number changes per tumour with ONE rule
(``ppc.decompose``) in simulated tumours (F10 calibration set) and in real non-WGD tumours (TCGA
ASCAT, PCAWG), validates the rule twice, and writes tables, two figures and a factual note.

  A. Validation 1 - decomposition vs the simulator's TRUE event history (events_truth set).
  B. Validation 2 - our arm calls vs Taylor et al. 2018 (TCGA ASCAT, all QC-pass samples).
  C. Per-tumour counts per cohort (CSV per cohort + summary CSV).
  D. Comparison of each real cohort with the simulated F10 distribution (pilot PPC marginal rule).
  E. Figures (ECDF 2x3 panel; grouped bar chart of median counts).
  F. Markdown note.

Run from code/src:
    python -m ppc.step1_change_sizes --out /Users/maryamkafi/Documents/PhD_Project/cancer/reports/step1
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

from ppc.decompose import decompose, decompose_cohort
from ppc.grid import Grid, load_grid
from ppc.real_loaders import ARM_NAMES, load_ascat_tcga, load_pcawg, load_taylor_arm_calls
from ppc.sim_loaders import load_sim_cohort

PROJECT = "/Users/maryamkafi/Documents/PhD_Project/cancer"
REAL_ROOT = f"{PROJECT}/real_data/bulk_cnv"
F10_ROOT = f"{PROJECT}/data_sim/gene_calib_f10"
TRUTH_ROOT = f"{PROJECT}/data_sim/events_truth"
FIG_DIR = f"{PROJECT}/reports/figures"

# (label, source, code, role); TCGA and PCAWG codes of the same tumour type share a position
TCGA_TARGETS = ["OV", "BRCA", "LUSC", "HNSC", "ESCA", "GBM", "STAD"]
PCAWG_TARGETS = ["Ovary-AdenoCA", "Breast-AdenoCA", "Lung-SCC", "Head-SCC", "Eso-AdenoCA",
                 "CNS-GBM", "Stomach-AdenoCA"]
TCGA_CONTRAST = ["THCA", "PRAD"]
PCAWG_CONTRAST = ["Thy-AdenoCA", "Prost-AdenoCA"]

STATS = ["n_chrom", "n_arm", "n_focal", "n_focal_strict", "n_breakpoints", "fga", "ploidy"]
SUMMARY_STATS = ["n_chrom", "n_arm", "n_focal", "n_focal_strict", "n_broad", "n_breakpoints",
                 "fga", "ploidy"]
SIM_ALL = "SIM_F10_all"
SIM_ONE = "SIM_F10_one_per_theta"

# SISTEM event categories, column order of clones.npz["event_counts"] (simulate_gene_selection.py)
EV_FOCAL_AMP, EV_FOCAL_DEL, EV_ARM_DUP, EV_ARM_DEL, EV_CHR_DUP, EV_CHR_DEL, EV_WGD = range(7)


# ------------------------------------------------------------------------------------- helpers
def derive_per_tumour(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tumour summary counts from a :func:`ppc.decompose.decompose_cohort` DataFrame.

    n_chrom = whole_chrom_gain + whole_chrom_loss; n_arm = arm_gain + arm_loss;
    n_focal = focal_gain + focal_loss; n_focal_strict = n_focal minus the two focal
    ``*_orig_baseline`` subsets; n_broad = broad_gain + broad_loss; ploidy, fga and
    n_breakpoints are copied through.
    """
    out = pd.DataFrame(index=df.index)
    out["n_chrom"] = df["whole_chrom_gain"] + df["whole_chrom_loss"]
    out["n_arm"] = df["arm_gain"] + df["arm_loss"]
    out["n_focal"] = df["focal_gain"] + df["focal_loss"]
    out["n_focal_strict"] = out["n_focal"] - df["focal_gain_orig_baseline"] - df["focal_loss_orig_baseline"]
    out["n_broad"] = df["broad_gain"] + df["broad_loss"]
    for c in ("ploidy", "fga", "n_breakpoints"):
        out[c] = df[c]
    return out


def quantile_position(sim: np.ndarray, value: float) -> float:
    """Mid-rank quantile position of ``value`` within the sample ``sim``.

    Returns (#sim < value + 0.5 * #sim == value) / n, so a value equal to every simulated value
    sits at 0.5 and a value below all of them at 0.0. NaNs in ``sim`` are dropped.
    """
    s = np.asarray(sim, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0 or np.isnan(value):
        return float("nan")
    return float((np.sum(s < value) + 0.5 * np.sum(s == value)) / s.size)


def structural_mask(sources: dict[str, list[np.ndarray]], threshold: float = 0.95) -> dict[str, np.ndarray]:
    """Structural mask from real profiles grouped by data source (e.g. {'TCGA': [...], 'PCAWG': [...]}).

    For each source, pools its cohorts and flags blocks masked (-1) in at least ``threshold`` of the
    profiles. Returns {'<source>': mask, ..., 'pooled_all': mask over all sources pooled together,
    'union': OR of the per-source masks}. The union is the mask applied (see the note for why).
    """
    out, allX = {}, []
    for name, profiles in sources.items():
        X = np.concatenate([np.asarray(p) for p in profiles if len(p)], axis=0)
        allX.append(X)
        out[name] = (X == -1).mean(axis=0) >= threshold
    out["pooled_all"] = (np.concatenate(allX, axis=0) == -1).mean(axis=0) >= threshold
    out["union"] = np.logical_or.reduce([out[n] for n in sources])
    return out


def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of ``X`` (1-D profile or [n, D]) with the blocks in ``mask`` set to -1 (masked)."""
    Y = np.array(X, dtype=np.int64, copy=True)
    Y[..., mask] = -1
    return Y


def summarise(per: pd.DataFrame, cohort: str) -> dict:
    """One summary row: n and median/mean/p10/p90 of every statistic in SUMMARY_STATS."""
    row = {"cohort": cohort, "n": len(per)}
    for c in SUMMARY_STATS:
        v = per[c].to_numpy(dtype=float)
        row[f"{c}_median"] = float(np.nanmedian(v))
        row[f"{c}_mean"] = float(np.nanmean(v))
        row[f"{c}_p10"] = float(np.nanpercentile(v, 10))
        row[f"{c}_p90"] = float(np.nanpercentile(v, 90))
    return row


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------- A. validation 1
def focal_overlap_stats(events: np.ndarray) -> dict:
    """Overlap among the TRUE focal events of a trial (``clones.npz['events']``, all lineages).

    Two focal events (categories focal_amplification / focal_deletion) overlap when they are on the
    same chromosome and their [first_block, last_block] block ranges intersect. Returns the number
    of focal events, the number with block ranges, the number of distinct (chrom, first, last)
    footprints, the number overlapping at least one other focal event, and the number of
    overlapping pairs.
    """
    ev = np.asarray(events)
    foc = ev[np.isin(ev[:, 0], (EV_FOCAL_AMP, EV_FOCAL_DEL))] if ev.size else ev.reshape(0, 9)
    with_range = foc[(foc[:, 3] >= 0) & (foc[:, 4] >= 0)]
    n = len(with_range)
    overl = np.zeros(n, dtype=bool)
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = with_range[i], with_range[j]
            if a[1] == b[1] and a[3] <= b[4] and b[3] <= a[4]:
                overl[i] = overl[j] = True
                pairs += 1
    footprints = len({(int(r[1]), int(r[3]), int(r[4])) for r in with_range})
    return {"true_focal_events_all_lineages": int(len(foc)), "focal_with_block_range": int(n),
            "distinct_focal_footprints": int(footprints),
            "focal_overlapping_another": int(overl.sum()), "focal_overlap_pairs": int(pairs)}


def validation1(truth_root: str, grid: Grid, mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decomposition of the largest clone vs its TRUE lineage event counts, per completed trial.

    The largest clone's profile has the structural ``mask`` applied before decomposition, so true
    events that fall only on masked blocks are invisible by construction.

    Returns (per-trial table, pooled table with visible fraction = sum inferred / sum true).
    """
    paths = sorted(glob.glob(os.path.join(truth_root, "simulation_outputs", "sim*", "*", "clones.npz")))
    rows = []
    for p in paths:
        z = np.load(p)
        if "event_counts" not in z.files:
            continue
        i = int(np.argmax(z["popsize"]))
        ec = z["event_counts"][i]
        nwgd = int(z["nWGD"][i]) if "nWGD" in z.files else 0
        dec = decompose(apply_mask(z["block_cn"][i], mask), grid, baseline=2 * 2 ** nwgd)
        foc_ob = dec.focal_gain_orig_baseline + dec.focal_loss_orig_baseline
        brd_ob = dec.broad_gain_orig_baseline + dec.broad_loss_orig_baseline
        trial = os.path.basename(os.path.dirname(p))
        sim = os.path.basename(os.path.dirname(os.path.dirname(p)))
        row = {
            "trial": f"{sim}/{trial}", "n_clones": int(len(z["popsize"])), "nWGD_largest": nwgd,
            "true_chrom": int(ec[EV_CHR_DUP] + ec[EV_CHR_DEL]),
            "inf_chrom": dec.whole_chrom_gain + dec.whole_chrom_loss,
            "true_arm": int(ec[EV_ARM_DUP] + ec[EV_ARM_DEL]),
            "inf_arm": dec.arm_gain + dec.arm_loss,
            "true_focal": int(ec[EV_FOCAL_AMP] + ec[EV_FOCAL_DEL]),
            "inf_focal": dec.focal_gain + dec.focal_loss,
            "inf_focal_strict": dec.focal_gain + dec.focal_loss - foc_ob,
            "inf_broad": dec.broad_gain + dec.broad_loss,
            "inf_broad_strict": dec.broad_gain + dec.broad_loss - brd_ob,
            "true_wgd": int(ec[EV_WGD]),
        }
        row.update(focal_overlap_stats(z["events"]) if "events" in z.files else {})
        rows.append(row)
    per = pd.DataFrame(rows)
    if per.empty:
        return per, pd.DataFrame()
    pooled = []
    for cls, inf_cols in (("chrom", ["inf_chrom"]), ("arm", ["inf_arm"]),
                          ("focal", ["inf_focal", "inf_focal_strict"])):
        t = per[f"true_{cls}"].sum()
        for ic in inf_cols:
            s = per[ic].sum()
            pooled.append({"class": cls, "inferred_column": ic, "sum_true": int(t), "sum_inferred": int(s),
                           "visible_fraction": float(s / t) if t else float("nan")})
    return per, pd.DataFrame(pooled)


# ------------------------------------------------------------------------------- B. validation 2
def validation2(grid: Grid, mask: np.ndarray) -> dict:
    """Our R2 arm calls on TCGA ASCAT (QC Pass, any WGD, baseline = round(ploidy)) vs Taylor 2018.

    Join: ASCAT ``meta.donor`` == Taylor ``case_id``; for a case with several Taylor rows the sample
    id ending '-01' is used, else the first row. Returns a dict of DataFrames / numbers for the
    whole set ('all') and the non-WGD subset ('nonwgd'). The structural ``mask`` is applied first.
    """
    coh = load_ascat_tcga(f"{REAL_ROOT}/ascat", grid, wgd=None, qc=("Pass",), min_purity=None)
    tl = load_taylor_arm_calls(f"{REAL_ROOT}/tcga/arm.txt").reset_index()
    tl["_pref"] = ~tl["sample"].str.endswith("-01")
    tl = tl.sort_values(["case_id", "_pref"], kind="stable").drop_duplicates("case_id", keep="first")
    tl = tl.set_index("case_id")
    meta = coh.meta
    keep = np.flatnonzero(meta["donor"].isin(tl.index).to_numpy())
    baselines = np.round(meta["ploidy"].to_numpy(dtype=float)).astype(np.int64)
    ours = np.zeros((len(keep), len(ARM_NAMES)))
    for k, r in enumerate(keep):
        dec = decompose(apply_mask(coh.block_cn_int[r], mask), grid, baseline=int(baselines[r]))
        ours[k] = [dec.arm_calls[a] for a in ARM_NAMES]
    theirs = tl.loc[meta["donor"].iloc[keep], ARM_NAMES].to_numpy(dtype=float)
    their_score = tl.loc[meta["donor"].iloc[keep], "aneuploidy_score"].to_numpy(dtype=float)
    wgd = meta["wgd"].iloc[keep].to_numpy() != "0"
    n_baseline = pd.Series(baselines[keep]).value_counts().sort_index()

    def _summ(sel: np.ndarray) -> dict:
        o, t, sc = ours[sel], theirs[sel], their_score[sel]
        valid = ~np.isnan(t)
        agree = (o == t) & valid
        per_arm = pd.DataFrame({"arm": ARM_NAMES, "n_compared": valid.sum(0),
                                "agreement": agree.sum(0) / np.maximum(valid.sum(0), 1)})
        conf = pd.crosstab(pd.Series(o[valid].astype(int), name="ours"),
                           pd.Series(t[valid].astype(int), name="taylor"))
        our_score = (o != 0).sum(1)
        ok = ~np.isnan(sc)
        from scipy.stats import spearmanr
        rho = spearmanr(our_score[ok], sc[ok]).statistic if ok.sum() > 2 else float("nan")
        return {"n_samples": int(sel.sum()), "n_calls": int(valid.sum()),
                "overall": float(agree.sum() / valid.sum()), "per_arm": per_arm, "confusion": conf,
                "score_rho": float(rho), "score_mad": float(np.mean(np.abs(our_score[ok] - sc[ok]))),
                "score_mean_ours": float(our_score[ok].mean()), "score_mean_taylor": float(sc[ok].mean())}

    return {"all": _summ(np.ones(len(keep), dtype=bool)), "nonwgd": _summ(~wgd),
            "n_ascat_pass": len(meta), "n_joined": len(keep), "baselines": n_baseline}


# ------------------------------------------------------------------------------- C. counts
def load_cohorts(grid: Grid) -> dict[str, tuple[np.ndarray, str]]:
    """All cohorts as {label: (block_cn_int, role)}; role in {'target', 'contrast', 'sim'}."""
    out = {}
    tcga = load_ascat_tcga(f"{REAL_ROOT}/ascat", grid)
    for codes, role in ((TCGA_TARGETS, "target"), (TCGA_CONTRAST, "contrast")):
        for c in codes:
            sub = tcga.subset((tcga.meta["cancer_type"] == c).to_numpy())
            out[f"TCGA_{c}"] = (sub.block_cn_int, role)
    pc = load_pcawg(f"{REAL_ROOT}/pcawg", grid)
    for codes, role in ((PCAWG_TARGETS, "target"), (PCAWG_CONTRAST, "contrast")):
        for c in codes:
            sub = pc.subset((pc.meta["cancer_type"] == c).to_numpy())
            out[f"PCAWG_{c}"] = (sub.block_cn_int, role)
    sim = load_sim_cohort(F10_ROOT, grid)
    assert (sim.meta["nWGD_largest"] == 0).all(), "F10 has WGD largest clones; baseline 2 would be wrong"
    out[SIM_ALL] = (sim.block_cn_int, "sim")
    one = load_sim_cohort(F10_ROOT, grid, one_trial_per_theta=1)
    out[SIM_ONE] = (one.block_cn_int, "sim")
    return out


def count_cohorts(cohorts: dict, grid: Grid, outdir: str, mask: np.ndarray
                  ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Decompose every cohort (baseline 2) after applying the structural ``mask``; save per-tumour
    CSVs and the summary CSV. Also returns the per-cohort change in the mean of every statistic
    caused by the mask (masked minus unmasked)."""
    per, rows, eff = {}, [], []
    os.makedirs(os.path.join(outdir, "per_tumour"), exist_ok=True)
    for lab, (X, _) in cohorts.items():
        df = decompose_cohort(apply_mask(X, mask), grid, 2)
        p = pd.concat([derive_per_tumour(df), df], axis=1)
        p = p.loc[:, ~p.columns.duplicated()]
        if len(X):
            raw = derive_per_tumour(decompose_cohort(X, grid, 2))
            eff.append({"cohort": lab, **{s: float(p[s].mean() - raw[s].mean()) for s in SUMMARY_STATS}})
        p = p.loc[:, ~p.columns.duplicated()]
        per[lab] = p
        p.to_csv(os.path.join(outdir, "per_tumour", f"{lab}.csv"), index_label="row")
        rows.append(summarise(p, lab))
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(outdir, "counts_summary.csv"), index=False)
    eff = pd.DataFrame(eff)
    eff.to_csv(os.path.join(outdir, "mask_effect_on_means.csv"), index=False)
    return per, summ, eff


# ------------------------------------------------------------------------------- D. comparison
def compare(per: dict[str, pd.DataFrame], cohorts: dict) -> pd.DataFrame:
    """Each real cohort vs SIM_F10_all, per statistic: median quantile position, z, KS, W1."""
    from scipy.stats import ks_2samp, wasserstein_distance
    sim = per[SIM_ALL]
    rows = []
    for lab, (_, role) in cohorts.items():
        if role == "sim":
            continue
        for s in STATS:
            r = per[lab][s].dropna().to_numpy(dtype=float)
            m = sim[s].dropna().to_numpy(dtype=float)
            if r.size == 0:
                rows.append({"cohort": lab, "role": role, "stat": s, "n_real": 0})
                continue
            q = quantile_position(m, float(np.median(r)))
            sd = m.std(ddof=1)
            z = (r.mean() - m.mean()) / sd if sd > 0 else float("inf") * np.sign(r.mean() - m.mean())
            rows.append({"cohort": lab, "role": role, "stat": s, "n_real": int(r.size),
                         "real_median": float(np.median(r)), "sim_median": float(np.median(m)),
                         "q_pos": q, "q_flag": bool(q < 0.025 or q > 0.975),
                         "z": float(z), "z_flag": bool(abs(z) > 3),
                         "KS": float(ks_2samp(r, m).statistic), "W1": float(wasserstein_distance(r, m))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- E. figures
# validated categorical order (dataviz reference palette, light mode), fixed per tumour type
TYPE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def _style(lab: str) -> dict:
    """Line style per cohort: colour = tumour type; TCGA solid, PCAWG dotted; contrast grey dashed."""
    if lab.startswith("TCGA_"):
        c = lab[5:]
        if c in TCGA_TARGETS:
            return dict(color=TYPE_COLOURS[TCGA_TARGETS.index(c)], ls="-", lw=1.4)
        return dict(color="#52514e" if c == "THCA" else "#a3a29b", ls="--", lw=1.4)
    c = lab[6:]
    if c in PCAWG_TARGETS:
        return dict(color=TYPE_COLOURS[PCAWG_TARGETS.index(c)], ls=":", lw=1.6)
    return dict(color="#52514e" if c == "Thy-AdenoCA" else "#a3a29b", ls="-.", lw=1.4)


def _ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(v[~np.isnan(v)])
    return v, np.arange(1, v.size + 1) / v.size


def figure_ecdf(per: dict, cohorts: dict, path: str) -> None:
    """2x3 ECDF panel (n_chrom, n_arm, n_focal_strict / n_breakpoints, fga, ploidy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = ["n_chrom", "n_arm", "n_focal_strict", "n_breakpoints", "fga", "ploidy"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=150)
    sim = per[SIM_ALL]
    for ax, s in zip(axes.ravel(), panels):
        m = sim[s].dropna().to_numpy(dtype=float)
        lo, hi = np.percentile(m, [2.5, 97.5])
        ax.axvspan(lo, hi, color="#d6d5cf", alpha=0.6, lw=0, zorder=0)
        for lab, (_, role) in cohorts.items():
            if role == "sim" or len(per[lab]) == 0:
                continue
            x, y = _ecdf(per[lab][s].to_numpy(dtype=float))
            ax.step(x, y, where="post", label=f"{lab} (n={len(per[lab])})", **_style(lab))
        x, y = _ecdf(m)
        ax.step(x, y, where="post", color="black", lw=3.0, label=f"{SIM_ALL} (n={len(m)})", zorder=5)
        ax.set_title(s, fontsize=12)
        ax.set_xlabel("per tumour" if s in ("fga", "ploidy") else "count per tumour")
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    import matplotlib.patches as mpatches
    h.append(mpatches.Patch(color="#d6d5cf", alpha=0.6)); l.append("simulated 2.5-97.5% range")
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle("Step 1: change sizes per tumour, real non-WGD cohorts vs simulated F10 "
                 "(TCGA solid, PCAWG dotted, contrast grey dashed/dash-dot)", fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def figure_ratio(summ: pd.DataFrame, path: str) -> None:
    """Grouped bars: median per-tumour n_chrom / n_arm / n_focal_strict per cohort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("n_chrom_median", "whole chromosome", "#2a78d6"), ("n_arm_median", "arm", "#eb6834"),
            ("n_focal_strict_median", "focal (strict)", "#1baf7a")]
    labs = summ["cohort"].tolist()
    x = np.arange(len(labs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=150)
    for k, (c, name, col) in enumerate(cols):
        ax.bar(x + (k - 1) * w, summ[c], width=w - 0.03, color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labs, summ["n"])], rotation=60, ha="right",
                       fontsize=8)
    for t, l in zip(ax.get_xticklabels(), labs):
        if l.startswith("SIM"):
            t.set_fontweight("bold")
    ax.set_ylabel("median count per tumour")
    ax.set_title("Median per-tumour event counts by size class (real cohorts vs simulated F10)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------- F. note
def write_note(path: str, v1: tuple, v2: dict, summ: pd.DataFrame, comp: pd.DataFrame,
               outdir: str, figs: list[str], caveats: list[str], n_truth_expected: int,
               masks: dict, eff: pd.DataFrame, grid: Grid) -> None:
    """Write the factual markdown note."""
    blocks = np.flatnonzero(masks["union"])
    fmt = lambda m: ", ".join(map(str, np.flatnonzero(m))) or "none"  # noqa: E731
    arm_of = {}
    for a in grid.arms:
        for b in range(a.start, a.end):
            arm_of[b] = a.name
    per_arm = pd.Series([arm_of[int(b)] for b in blocks]).value_counts(sort=False)
    arm_tot = {a.name: a.n_blocks for a in grid.arms}
    mask_lines = ["## Structural mask", "",
                  "Rule as requested: blocks masked (copy number -1) in >= 95% of the pooled filtered real "
                  f"profiles. Pooling all 18 real cohorts gives only {int(masks['pooled_all'].sum())} blocks "
                  f"({fmt(masks['pooled_all'])}), because PCAWG (whole-genome) covers the acrocentric p arms in "
                  "most samples while TCGA ASCAT (SNP array) never does, and TCGA is ~80% of the pool. Per "
                  f"source: TCGA {int(masks['TCGA'].sum())} blocks ({fmt(masks['TCGA'])}); PCAWG "
                  f"{int(masks['PCAWG'].sum())} blocks ({fmt(masks['PCAWG'])}). The mask APPLIED is the union "
                  "of the per-source masks (>= 95% within either source), so that no cohort can carry events "
                  f"the other source cannot see: {len(blocks)} blocks, saved as `{outdir}/structural_mask.npy` "
                  f"(bool, length {grid.D}). It is applied to EVERY profile (real, simulated F10, truth trials, and the "
                  "ASCAT profiles of validation 2) before decomposition, so both sides use the same block set.", "",
                  "Blocks (flat indices): " + ", ".join(map(str, blocks)) + ".", "",
                  "| arm | masked blocks | arm blocks |", "|---|---|---|"]
    mask_lines += [f"| {a} | {n} | {arm_tot[a]} |" for a, n in per_arm.items()]
    mask_lines += ["", "Change in the cohort mean of each statistic caused by the mask (masked minus unmasked; "
                   "`mask_effect_on_means.csv`):", "", md_table(eff), ""]
    per1, pooled1 = v1
    L = ["# Step 1 - Does the mix of change sizes look real? (results)", "",
         f"Generated by `code/src/ppc/step1_change_sizes.py` on {time.strftime('%Y-%m-%d %H:%M')}.", "",
         "## Data used", "",
         "Real cohorts: non-WGD, QC pass (TCGA), purity >= 0.3, one sample per donor (loader filters; PCAWG: is_preferred first, then lowest aliquot id). "
         "Simulated: largest clone per trial of the F10 calibration set (21 thetas x 25 trials); all "
         "profiles decomposed with baseline 2.", "",
         md_table(summ[["cohort", "n"]]), "",
         f"Truth trials used for validation 1: {len(per1)} of {n_truth_expected} expected "
         "(3 thetas x 2 trials).", "",
         ] + mask_lines + ["## Validation 1 - decomposition vs true lineage events (largest clone)", "",
         "The structural mask is applied to the largest clone's profile, so true events lying only on "
         "masked blocks are invisible by construction.", ""]
    if per1.empty:
        L += ["No completed truth trial was available.", ""]
    else:
        L += [md_table(per1), "", "Pooled visible fraction (sum inferred / sum true):", "",
              md_table(pooled1), "",
              "`true_*` counts come from `event_counts[argmax popsize]` (lineage of the largest clone); "
              "`focal_overlapping_another` / `focal_overlap_pairs` are over the trial's `events` table, "
              "which holds the unique events of ALL surviving lineages (same chromosome, intersecting "
              "[first_block, last_block]).", ""]
    a, nw = v2["all"], v2["nonwgd"]
    worst = a["per_arm"].sort_values("agreement").head(5)
    L += ["## Validation 2 - arm calls vs Taylor et al. 2018 (TCGA ASCAT, QC Pass)", "",
          f"ASCAT QC-pass samples: {v2['n_ascat_pass']}; joined to a Taylor row: {v2['n_joined']}. "
          "Baseline = round(ploidy) per sample; baseline counts: "
          + ", ".join(f"{int(b)}: {int(n)}" for b, n in v2["baselines"].items()) + ".", "",
          "| set | samples | arm calls compared | overall agreement | aneuploidy-score Spearman rho | "
          "mean abs diff | mean score ours | mean score Taylor |", "|---|---|---|---|---|---|---|---|"]
    for name, d in (("all", a), ("non-WGD", nw)):
        L.append(f"| {name} | {d['n_samples']} | {d['n_calls']} | {d['overall']:.3f} | {d['score_rho']:.3f} | "
                 f"{d['score_mad']:.2f} | {d['score_mean_ours']:.2f} | {d['score_mean_taylor']:.2f} |")
    L += ["", "Worst 5 arms (all samples):", "", md_table(worst), "",
          "Confusion (all samples; rows = ours, columns = Taylor; NaN excluded):", "",
          md_table(a["confusion"].reset_index().rename(columns=str)), "",
          "Per-arm table: `validation2_per_arm.csv`.", "",
          "## Counts summary (per-tumour statistics)", "",
          md_table(summ[["cohort", "n"] + [f"{s}_{k}" for s in
                                           ("n_chrom", "n_arm", "n_focal", "n_focal_strict", "n_broad",
                                            "n_breakpoints", "fga", "ploidy") for k in ("median",)]]), "",
          "Means and 10th/90th percentiles: `counts_summary.csv`.", "",
          "## Comparison with simulated F10 (all 525 trials)", "",
          "q_pos = mid-rank quantile of the real cohort median within the simulated distribution "
          "(flag < 0.025 or > 0.975); z = (mean_real - mean_sim) / sd_sim (flag |z| > 3); KS and W1 "
          "are two-sample statistics.", "",
          md_table(comp.drop(columns=["role"])), "",
          "Flag totals per statistic (cohorts flagged by q_pos / by z, out of "
          f"{comp['cohort'].nunique()}):", "",
          md_table(comp.groupby("stat", sort=False)[["q_flag", "z_flag"]].sum().reset_index()), "",
          "## Files", ""]
    L += [f"- `{f}`" for f in figs]
    L += [f"- `{outdir}/counts_summary.csv`, `{outdir}/comparison.csv`, `{outdir}/comparison.md`",
          f"- `{outdir}/validation1_per_trial.csv`, `{outdir}/validation1_pooled.csv`",
          f"- `{outdir}/validation2_per_arm.csv`, `{outdir}/validation2_confusion.csv`, "
          f"`{outdir}/validation2_summary.csv`",
          f"- `{outdir}/per_tumour/<cohort>.csv` (one per cohort)",
          f"- `{outdir}/structural_mask.npy`, `{outdir}/mask_effect_on_means.csv`", "",
          "## Caveats observed", ""]
    L += [f"- {c}" for c in caveats]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def collect_caveats(cohorts: dict, per: dict, v1: tuple, v2: dict, n_truth_expected: int) -> list[str]:
    """Data-driven list of surprising observations (masked arms, small or degenerate cohorts...)."""
    cav = []
    grid_arms = None
    for lab, (X, role) in cohorts.items():
        n = len(X)
        if role != "sim" and n < 20:
            cav.append(f"{lab} is small (n = {n}); its quantile/z/KS values are noisy.")
        if n and role != "sim":
            frac_masked = float((X == -1).mean())
            if frac_masked > 0.10:
                cav.append(f"{lab}: {100 * frac_masked:.1f}% of blocks masked on average.")
        p = per[lab]
        for s in STATS:
            if n and p[s].nunique() == 1:
                cav.append(f"{lab}: `{s}` is constant ({p[s].iloc[0]}) across all tumours.")
    for lab, (X, role) in cohorts.items():
        if role == "sim" or len(X) == 0:
            continue
        if grid_arms is None:
            from ppc.grid import load_grid as _lg
            grid_arms = _lg(os.path.join(F10_ROOT, "manifest.json")).called_arms
        allm = [a.name for a in grid_arms if (X[:, a.start:a.end] == -1).all(axis=1).mean() > 0.5]
        if allm:
            cav.append(f"{lab}: arms fully masked in > 50% of tumours: {', '.join(allm)}.")
    per1 = v1[0]
    if len(per1) < n_truth_expected:
        cav.append(f"Only {len(per1)} of {n_truth_expected} truth trials had completed; validation 1 is "
                   "based on those.")
    tot_ob = sum(int((per[l]["n_focal"] - per[l]["n_focal_strict"]).sum()) for l in per)
    cav.append(f"Focal runs that exist only because of a chrom/arm subtraction (orig_baseline), summed over "
               f"all cohorts: {tot_ob}.")
    if v2["all"]["overall"] < 0.8:
        cav.append(f"Overall arm-call agreement with Taylor is below 80% ({v2['all']['overall']:.3f}).")
    return cav


# ------------------------------------------------------------------------------- main
def main(argv=None) -> None:
    """CLI entry point: run A-F and write everything under ``--out`` and the figures folder."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"{PROJECT}/reports/step1")
    ap.add_argument("--figdir", default=FIG_DIR)
    ap.add_argument("--truth-root", default=TRUTH_ROOT)
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    grid = load_grid(os.path.join(F10_ROOT, "manifest.json"))
    tgrid = load_grid(os.path.join(args.truth_root, "manifest.json"))
    assert tgrid.D == grid.D and tgrid.nregions == grid.nregions, "truth grid differs from F10 grid"

    cohorts = load_cohorts(grid)
    masks = structural_mask({src: [X for lab, (X, role) in cohorts.items() if role != "sim" and lab.startswith(src)]
                             for src in ("TCGA", "PCAWG")})
    mask = masks["union"]
    np.save(os.path.join(args.out, "structural_mask.npy"), mask)
    print(f"[mask] structural mask: {int(mask.sum())} blocks (TCGA {int(masks['TCGA'].sum())}, "
          f"PCAWG {int(masks['PCAWG'].sum())}, all-pooled {int(masks['pooled_all'].sum())})")

    v1 = validation1(args.truth_root, tgrid, mask)
    v1[0].to_csv(os.path.join(args.out, "validation1_per_trial.csv"), index=False)
    v1[1].to_csv(os.path.join(args.out, "validation1_pooled.csv"), index=False)
    print(f"[A] validation 1: {len(v1[0])} truth trials")

    v2 = validation2(grid, mask)
    v2["all"]["per_arm"].merge(v2["nonwgd"]["per_arm"], on="arm", suffixes=("_all", "_nonwgd")).to_csv(
        os.path.join(args.out, "validation2_per_arm.csv"), index=False)
    v2["all"]["confusion"].to_csv(os.path.join(args.out, "validation2_confusion.csv"))
    pd.DataFrame([{"set": k, **{m: v2[k][m] for m in ("n_samples", "n_calls", "overall", "score_rho",
                                                      "score_mad", "score_mean_ours", "score_mean_taylor")}}
                  for k in ("all", "nonwgd")]).to_csv(os.path.join(args.out, "validation2_summary.csv"),
                                                      index=False)
    print(f"[B] validation 2: agreement all={v2['all']['overall']:.3f} (n={v2['all']['n_samples']}), "
          f"non-WGD={v2['nonwgd']['overall']:.3f}; score rho={v2['all']['score_rho']:.3f}")

    per, summ, eff = count_cohorts(cohorts, grid, args.out, mask)
    print(f"[C] counts: {len(cohorts)} cohorts, n = " + ", ".join(str(len(v[0])) for v in cohorts.values()))

    comp = compare(per, cohorts)
    comp.to_csv(os.path.join(args.out, "comparison.csv"), index=False)
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write(md_table(comp.drop(columns=["role"])) + "\n")
    print(f"[D] comparison: {int(comp['q_flag'].sum())} q-flags, {int(comp['z_flag'].sum())} z-flags "
          f"of {len(comp)} cohort x stat cells")

    f1 = os.path.join(args.figdir, "step1_change_sizes.png")
    f2 = os.path.join(args.figdir, "step1_focal_arm_chrom_ratio.png")
    figure_ecdf(per, cohorts, f1)
    figure_ratio(summ, f2)
    print(f"[E] figures: {f1}, {f2}")

    n_truth_expected = 6
    cav = collect_caveats(cohorts, per, v1, v2, n_truth_expected)
    note = os.path.join(args.out, "step1_change_sizes_results.md")
    write_note(note, v1, v2, summ, comp, args.out, [f1, f2], cav, n_truth_expected, masks, eff, grid)
    print(f"[F] note: {note}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
