"""Every evaluation figure, one function each.

This module replaces four near-duplicate copies of the same plotting code. The
numbers, colours, bin counts, figure sizes and axis limits are reproduced
exactly: the figures these functions draw are in a manuscript under review, so a
"nicer" plot is a regression.

Every function takes already-computed arrays (see
:mod:`cancer_sbi.evaluation.diagnostics`) plus an explicit ``out_dir``. No
function samples a posterior, loads a checkpoint, or defaults its output
directory to the string ``"figures"`` the way the originals did.

Originals:
    * ``Base_NPE/z-score_violin.py`` -- the source of the paper's figures:
      global z histogram (212-245), the three per-parameter bar charts (252-290),
      ``zscore_summary.txt`` (293-302), the two violin figures (314-481), the two
      true-vs-posterior-mean scatter grids (499-666) and the correlation
      spreadsheet (682-738).
    * ``Plain_NPE/z-score_violin.py`` -- the same maths with a different
      data-loading head; its plotting code differs from the above only in
      whitespace, comments and a ``for ext in ("png", "pdf")`` save loop, so it
      needs no separate implementation.
    * ``Base_NPE/z-score.py`` -- the earlier, smaller version of the same bar
      charts and summary file.
    * ``SetTransformer_NPE/kde_prior_vs_posterior.py`` -- the prior-vs-posterior
      KDE grids (204-509).
    * ``Base_NPE/ppc-plot.py`` -- the sbi pairplot (147-166).
    * ``Base_NPE/test.ipynb`` -- SBC rank plots (cells 14-27) and the three SBC
      bar charts (cells 29-31).
    * ``Base_NPE/PriorPredictiveCheck.ipynb`` -- the L2 bar chart (cell 12) and
      the per-parameter z-vs-index panels (cells 9 and 10).
"""

from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde

import cancer_sbi.evaluation.diagnostics as diagnostics  # robust under circular init
from cancer_sbi.arms import ARM_LABELS, arm_index_halves
from cancer_sbi.evaluation.style import (
    BIASED_RED,
    CASE_COLORS,
    FOOTNOTE_GREY,
    GRID_GREY,
    KDE_GRID_GREY,
    OVERCONFIDENT_ORANGE,
    PRIOR_FILL_BLUE,
    R2_GOOD_GREEN,
    R2_POOR_ORANGE,
    REF_TWO_SIGMA_ORANGE,
    REF_ZERO_RED,
    SBC_FAIL_RED,
    SBC_PASS_BLUE,
    SPINE_GREY,
    WELL_CALIBRATED_BLUE,
)

__all__ = [
    "DEFAULT_PRIOR_MU",
    "DEFAULT_PRIOR_STD",
    "LEGACY_PRIOR_STD",
    "plot_global_z_histogram",
    "plot_per_parameter_bar",
    "plot_per_parameter_z_bars",
    "write_zscore_summary",
    "plot_violin_zscores",
    "plot_true_vs_postmean_scatter",
    "write_correlation_excel",
    "plot_kde_pooled_only",
    "plot_kde_full",
    "plot_posterior_pairplot",
    "plot_sbc_rank_slice",
    "plot_sbc_rank_hist_panels",
    "plot_sbc_rank_cdf",
    "plot_sbc_ks_pvalues",
    "plot_sbc_c2st_bar",
    "plot_sbc_c2st_ranks",
    "plot_sbc_c2st_dap",
    "plot_l2_per_case",
    "plot_abs_z_vs_index",
    "plot_z_vs_index_with_bands",
]


#: Mean of the simulator's prior over selection coefficients.
DEFAULT_PRIOR_MU: float = 0.0

#: Standard deviation of the simulator's prior over selection coefficients.
#:
#: This is the CORRECT value. See :func:`plot_kde_pooled_only` for what the
#: original scripts drew instead and what changes because of it.
DEFAULT_PRIOR_STD: float = 0.2

#: The prior standard deviation the six original evaluation scripts used:
#: ``math.sqrt(0.2) = 0.4472...``, i.e. 2.24x too wide. Recorded here only so the
#: old figures can be reproduced deliberately for comparison. Never the default.
LEGACY_PRIOR_STD: float = 0.4472135954999579

#: The two 22-arm halves the paper figures are split into, taken from the single
#: definition in :mod:`cancer_sbi.arms` so the split lives in one place. Verified
#: equal to the originals' hard-coded ``range(0, 22)`` / ``range(22, 44)``.
#:
#: The subtitle strings are NOT shared across figures: the violins use an en dash
#: and the scatter and KDE grids use a hyphen. That difference is in the published
#: figures, so the strings stay where they are, per figure, rather than being
#: unified here.
_FIRST_HALF, _SECOND_HALF = (list(half) for half in arm_index_halves())


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #


def _save_png_pdf(
    fig: Figure,
    out_dir: Path,
    stem: str,
    *,
    dpi: int = 300,
    facecolor: str | None = None,
) -> dict[str, Path]:
    """Save a figure as both PNG and PDF and close it.

    Preserved from every original save site: the PNG carries an explicit ``dpi``
    and the PDF does not (it inherits ``figure.dpi``, which matters only for the
    rasterised scatter points). Both use ``bbox_inches="tight"``.

    Args:
        fig: The figure to save.
        out_dir: Directory to write into; created if missing.
        stem: File name without extension.
        dpi: PNG resolution.
        facecolor: Passed to ``savefig`` when not None. The originals pass
            ``"white"`` for the figures that set a white figure patch and omit it
            for the rest, which changes the saved background.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    png_kwargs: dict[str, Any] = {"dpi": dpi, "bbox_inches": "tight"}
    pdf_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
    if facecolor is not None:
        png_kwargs["facecolor"] = facecolor
        pdf_kwargs["facecolor"] = facecolor

    fig.savefig(png_path, **png_kwargs)
    fig.savefig(pdf_path, **pdf_kwargs)
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach a torch tensor to CPU numpy, or pass an array through.

    Args:
        array: A torch tensor or anything numpy can wrap.

    Returns:
        A numpy array.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


# --------------------------------------------------------------------------- #
# Global z-score histogram and per-parameter bar charts
# --------------------------------------------------------------------------- #


def plot_global_z_histogram(
    z_all: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    bins: int = 60,
    stem: str = "global_zscore_histogram",
) -> dict[str, Path]:
    """Histogram of every z-score with a standard-normal overlay.

    Preserved from ``Base_NPE/z-score_violin.py:231-245``: 60 bins, ``density=True``,
    an N(0,1) curve evaluated on ``linspace(-5, 5, 500)``, figure size (8, 5), and
    default matplotlib colours for both the bars and the curve (no colour is
    specified in the original, so the property cycle supplies C0 and C1).

    Args:
        z_all: z-scores, shape (N, D); pooled to (N*D,) before binning.
        out_dir: Directory to write ``<stem>.png`` and ``<stem>.pdf`` into.
        bins: Histogram bin count.
        stem: File name without extension.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    z_pooled = _to_numpy(z_all).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z_pooled, bins=bins, density=True)

    xgrid = np.linspace(-5, 5, 500)
    normal_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xgrid**2)
    ax.plot(xgrid, normal_pdf)

    ax.set_title("Global Z-score Distribution (All 44 Thetas Pooled)")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    fig.tight_layout()

    return _save_png_pdf(fig, out_dir, stem)


def plot_per_parameter_bar(
    values: np.ndarray,
    out_dir: Path,
    *,
    title: str,
    ylabel: str,
    stem: str,
    labels: Sequence[str] = ARM_LABELS,
    hline: float | None = None,
) -> dict[str, Path]:
    """One per-arm bar chart.

    Preserved from the ``_save_bar`` closure at ``Base_NPE/z-score_violin.py:252-266``:
    figure size (14, 5), default bar colour, x tick labels rotated 90 degrees, and
    an optional unstyled ``axhline`` whose colour therefore comes from the axes
    property cycle rather than being black.

    Args:
        values: Bar heights, shape (D,).
        out_dir: Directory to write into.
        title: Axes title.
        ylabel: y-axis label.
        stem: File name without extension.
        labels: Arm names used as x tick labels, length D.
        hline: If given, a horizontal reference line at this y value.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(list(labels), values)
    if hline is not None:
        ax.axhline(hline)
    # Matches plt.xticks(rotation=90): rotate the existing tick labels in place.
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    return _save_png_pdf(fig, out_dir, stem)


def plot_per_parameter_z_bars(
    per_parameter: diagnostics.PerParameterZSummary,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
) -> dict[str, dict[str, Path]]:
    """The three per-arm z-score bar charts.

    Preserved from ``Base_NPE/z-score_violin.py:268-290``, including which chart
    gets a reference line: mean |Z| gets none, mean Z gets one at 0.0, std(Z) gets
    one at 1.0.

    Args:
        per_parameter: Output of
            :func:`cancer_sbi.evaluation.diagnostics.per_parameter_z_summary`.
        out_dir: Directory to write into.
        labels: Arm names, length D.

    Returns:
        A dict keyed by file stem, each value ``{"png": ..., "pdf": ...}``.
    """
    return {
        "mean_abs_z_per_parameter": plot_per_parameter_bar(
            per_parameter.mean_abs_z,
            out_dir,
            title="Per-Parameter Calibration Error (Mean |Z|)",
            ylabel="Mean |Z|",
            stem="mean_abs_z_per_parameter",
            labels=labels,
        ),
        "mean_z_per_parameter": plot_per_parameter_bar(
            per_parameter.mean_z,
            out_dir,
            title="Per-Parameter Bias (Mean Z-score)",
            ylabel="Mean Z",
            stem="mean_z_per_parameter",
            labels=labels,
            hline=0.0,
        ),
        "std_z_per_parameter": plot_per_parameter_bar(
            per_parameter.std_z,
            out_dir,
            title="Per-Parameter Calibration Width (Std of Z-score)",
            ylabel="Std(Z)",
            stem="std_z_per_parameter",
            labels=labels,
            hline=1.0,
        ),
    }


def write_zscore_summary(
    pooled: diagnostics.PooledZSummary,
    per_parameter: diagnostics.PerParameterZSummary,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    filename: str = "zscore_summary.txt",
    l2: np.ndarray | None = None,
) -> Path:
    """Write the plain-text z-score summary.

    Preserved from ``Base_NPE/z-score_violin.py:293-302``: same headings, same
    6-decimal formatting, same 2-decimal percentage, same tab-separated table.
    (``Base_NPE/z-score.py:268`` writes the heading as "Per-parameter summaries
    (ordered by label)"; the violin script's shorter heading is the one used
    here, since that script produced the paper's artefacts.)

    The L2 block is the one deliberate addition. ``l2`` was computed and thrown
    away in every original. Leave ``l2=None`` and the file is byte-identical to
    the original; pass it and a short block is appended after the table.

    Args:
        pooled: Output of
            :func:`cancer_sbi.evaluation.diagnostics.pooled_z_summary`.
        per_parameter: Output of
            :func:`cancer_sbi.evaluation.diagnostics.per_parameter_z_summary`.
        out_dir: Directory to write into; created if missing.
        labels: Arm names, length D.
        filename: Name of the text file.
        l2: Optional per-case L2 distances, shape (N,).

    Returns:
        The path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / filename

    with open(summary_path, "w") as f:
        f.write("Global Z-score summary (pooled across all 44 thetas)\n")
        f.write(f"Mean(z): {pooled.mean_z:.6f}\n")
        f.write(f"Std(z):  {pooled.std_z:.6f}\n")
        f.write(f"% within [-2,2]: {pooled.pct_within_2:.2f}%\n\n")
        f.write("Per-parameter summaries\n")
        f.write("label\tmean_abs_z\tmean_z\tstd_z\n")
        # strict=True stands in for the shape assertions at
        # z-score_violin.py:218-220: a labels/values length mismatch must fail
        # loudly rather than silently truncate the table.
        for lab, a, m, s in zip(
            labels,
            per_parameter.mean_abs_z,
            per_parameter.mean_z,
            per_parameter.std_z,
            strict=True,
        ):
            f.write(f"{lab}\t{a:.6f}\t{m:.6f}\t{s:.6f}\n")

        if l2 is not None:
            l2_np = np.asarray(l2, dtype=np.float64)
            f.write("\nL2 distance ||theta_true - posterior mean||_2\n")
            f.write(f"Mean L2:   {float(np.mean(l2_np)):.6f}\n")
            f.write(f"Median L2: {float(np.median(l2_np)):.6f}\n")
            f.write(f"Std L2:    {float(np.std(l2_np, ddof=0)):.6f}\n")
            f.write(f"Min L2:    {float(np.min(l2_np)):.6f}\n")
            f.write(f"Max L2:    {float(np.max(l2_np)):.6f}\n")

    return summary_path


# --------------------------------------------------------------------------- #
# Violin figures
# --------------------------------------------------------------------------- #


def _draw_violin_panel(
    ax: Any,
    panel_z: np.ndarray,
    panel_labels: Sequence[str],
    panel_palette: Sequence[str],
    subtitle: str,
) -> None:
    """Draw one 22-arm violin panel onto an existing Axes.

    Preserved from ``Base_NPE/z-score_violin.py:314-358``. The seaborn call keeps
    ``hue="Parameter"`` (required from seaborn 0.14 for a per-category palette),
    ``inner="quartile"``, ``density_norm="width"``, ``linewidth=0.7`` and
    ``saturation=0.90``.

    Args:
        ax: Target axes.
        panel_z: z-scores for this panel, shape (N, 22).
        panel_labels: The 22 arm names, in plotting order.
        panel_palette: One colour per arm, same order.
        subtitle: Italic axes title.
    """
    # Imported here rather than at module scope so that importing this module
    # does not pull in seaborn's own rcParams handling.
    import seaborn as sns

    df = pd.DataFrame(panel_z, columns=list(panel_labels))
    df_long = df.melt(var_name="Parameter", value_name="Z-score")

    sns.violinplot(
        data=df_long,
        x="Parameter",
        y="Z-score",
        ax=ax,
        hue="Parameter",  # required in seaborn >= 0.14 for custom palette
        palette=list(panel_palette),
        order=list(panel_labels),
        legend=False,
        inner="quartile",  # draws Q1 / median / Q3 lines inside each violin
        density_norm="width",  # all violins same width -> easier comparison
        linewidth=0.7,
        saturation=0.90,
    )

    # Reference lines.
    ax.axhline(
        0,
        color=REF_ZERO_RED,
        linestyle="--",
        linewidth=1.2,
        label="z = 0  (unbiased)",
        zorder=5,
    )
    ax.axhline(
        2,
        color=REF_TWO_SIGMA_ORANGE,
        linestyle=":",
        linewidth=1.1,
        label="z = ±2  (95% CI)",
        zorder=5,
    )
    ax.axhline(-2, color=REF_TWO_SIGMA_ORANGE, linestyle=":", linewidth=1.1, zorder=5)

    # White background, publication-style ticks. Font sizes come from rcParams.
    ax.set_facecolor("white")
    ax.tick_params(axis="x", rotation=90, colors="black")
    ax.tick_params(axis="y", colors="black")
    ax.set_xlabel("Chromosome Arm", color="black", labelpad=8)
    ax.set_ylabel("Posterior Z-score", color="black", labelpad=8)
    ax.set_title(subtitle, color="black", pad=6, style="italic")

    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_GREY)
        spine.set_linewidth(0.8)

    ax.yaxis.grid(True, color=GRID_GREY, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def plot_violin_zscores(
    z_all: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    bias_threshold: float = diagnostics.BIAS_THRESHOLD,
    overconfidence_threshold: float = diagnostics.OVERCONFIDENCE_THRESHOLD,
) -> dict[str, dict[str, Path]]:
    """The two paper violin figures, 22 chromosome arms each.

    Preserved from ``Base_NPE/z-score_violin.py:362-481``.

    Violins are coloured by calibration class:
        * blue ``#4A90D9`` -- well calibrated,
        * red ``#C0504D`` -- biased, |mean z| > 1.0,
        * orange ``#E07B54`` -- overconfident, std z > 1.5,

    with overconfidence tested first, so an arm that is both comes out orange
    (see :func:`cancer_sbi.evaluation.diagnostics.classify_calibration`).

    Args:
        z_all: z-scores, shape (N, 44).
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        bias_threshold: |mean z| above which an arm is red. Also interpolated
            into the legend text, as in the original.
        overconfidence_threshold: std z above which an arm is orange.

    Returns:
        A dict keyed by file stem (``violin_zscore_chr1_11``,
        ``violin_zscore_chr12_22``), each value ``{"png": ..., "pdf": ...}``.
    """
    z_np = _to_numpy(z_all)

    # Per-parameter stats for colour-coding; ddof=0, as in the original.
    mean_z = z_np.mean(axis=0)
    std_z = z_np.std(axis=0, ddof=0)

    classes = diagnostics.classify_calibration(
        mean_z, std_z, bias_threshold, overconfidence_threshold
    )
    class_colour = {
        "overconfident": OVERCONFIDENT_ORANGE,
        "biased": BIASED_RED,
        "well_calibrated": WELL_CALIBRATED_BLUE,
    }
    palette_all = [class_colour[c] for c in classes]

    panels = [
        {
            "indices": _FIRST_HALF,
            # En dash, as published. The scatter and KDE grids use a hyphen.
            "subtitle": "Chromosomes 1–11",
            "fname": "violin_zscore_chr1_11",
        },
        {
            "indices": _SECOND_HALF,
            "subtitle": "Chromosomes 12–22",
            "fname": "violin_zscore_chr12_22",
        },
    ]

    # Shared legend -- identical on both figures for consistency.
    legend_handles = [
        mpatches.Patch(color=WELL_CALIBRATED_BLUE, label="Well-calibrated"),
        mpatches.Patch(
            color=BIASED_RED, label=f"Biased  (|mean z| > {bias_threshold})"
        ),
        mpatches.Patch(
            color=OVERCONFIDENT_ORANGE,
            label=f"Overconfident  (std z > {overconfidence_threshold})",
        ),
        Line2D(
            [0],
            [0],
            color=REF_ZERO_RED,
            linestyle="--",
            linewidth=1.2,
            label="z = 0  (unbiased)",
        ),
        Line2D(
            [0],
            [0],
            color=REF_TWO_SIGMA_ORANGE,
            linestyle=":",
            linewidth=1.1,
            label="z = ±2  (95% CI)",
        ),
    ]

    saved_paths: dict[str, dict[str, Path]] = {}

    for panel in panels:
        idx: list[int] = panel["indices"]  # type: ignore[assignment]
        panel_labels = [labels[j] for j in idx]
        panel_palette = [palette_all[j] for j in idx]
        panel_z = z_np[:, idx]  # (N, 22)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("white")

        _draw_violin_panel(
            ax,
            panel_z,
            panel_labels,
            panel_palette,
            subtitle=str(panel["subtitle"]),
        )

        fig.suptitle(
            "Posterior Calibration: Per-Parameter Z-Score Distribution "
            "of Chromosomal Copy-Number Coefficients",
            fontweight="bold",
            color="black",
            y=1.03,
        )

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            framealpha=0.9,
            facecolor="white",
            edgecolor=SPINE_GREY,
        )

        fig.tight_layout()

        saved_paths[str(panel["fname"])] = _save_png_pdf(
            fig, out_dir, str(panel["fname"]), facecolor="white"
        )

    return saved_paths


# --------------------------------------------------------------------------- #
# True vs posterior mean scatter grids
# --------------------------------------------------------------------------- #


def plot_true_vs_postmean_scatter(
    theta_all: torch.Tensor | np.ndarray,
    post_mean_all: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    r2_good_threshold: float = 0.70,
) -> dict[str, dict[str, Path]]:
    """The two 4x6 true-theta vs posterior-mean scatter grids.

    Preserved from ``Base_NPE/z-score_violin.py:499-666``:

    * shared axis limit ``lim = |theta|.max() * 1.15`` across every panel, and
      ticks at ``round(linspace(-0.85*lim, 0.85*lim, 5), 2)``;
    * points coloured by absolute error, min-max normalised per panel with a
      ``+1e-8`` guard, through the ``RdYlBu_r`` colormap (blue = low error);
    * a black dashed ``y = x`` diagonal and a red ``#CC2222`` fitted regression
      line from ``scipy.stats.linregress``;
    * the panel title shows R^2 and is green ``#1A7A1A`` when R^2 >= 0.70, orange
      ``#CC5500`` otherwise -- but the number is now the coefficient of
      determination ``1 - SSE/SST``, with Pearson ``r`` beside it, where the
      original printed the squared correlation under the name R^2;
    * 4 rows x 6 columns = 24 slots for 22 arms; the 2 spare axes are hidden.

    Args:
        theta_all: True parameters, shape (N, 44).
        post_mean_all: Posterior means, shape (N, 44).
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        r2_good_threshold: true R^2 (1 - SSE/SST) at or above which a panel title
            turns green.

    Returns:
        A dict keyed by file stem (``scatter_true_vs_postmean_chr1_11``,
        ``scatter_true_vs_postmean_chr12_22``), each value ``{"png", "pdf"}``.
    """
    theta_np = _to_numpy(theta_all)
    mean_np = _to_numpy(post_mean_all)

    # Global axis limits -- same across all panels for fair comparison.
    lim = float(np.abs(theta_np).max() * 1.15)
    ticks = np.round(np.linspace(-lim * 0.85, lim * 0.85, 5), 2).tolist()

    legend_elems = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="Ideal ($y = x$)",
        ),
        Line2D(
            [0],
            [0],
            color=REF_ZERO_RED,
            linestyle="-",
            linewidth=0.9,
            label="Fitted regression",
        ),
    ]

    panels = [
        {
            "indices": _FIRST_HALF,
            "fname": "scatter_true_vs_postmean_chr1_11",
            "title": "Chromosomes 1-11",
        },
        {
            "indices": _SECOND_HALF,
            "fname": "scatter_true_vs_postmean_chr12_22",
            "title": "Chromosomes 12-22",
        },
    ]

    saved_paths: dict[str, dict[str, Path]] = {}

    for panel in panels:
        idx: list[int] = panel["indices"]  # type: ignore[assignment]
        n_cols = 6
        n_rows = 4  # 4 rows x 6 cols = 24 slots for 22 arms

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 8))
        fig.patch.set_facecolor("white")
        axes_flat = axes.flatten()

        for k, j in enumerate(idx):
            ax = axes_flat[k]
            theta_j = theta_np[:, j]  # (N,)
            mean_j = mean_np[:, j]  # (N,)

            slope, intercept, r, _, _ = scipy_stats.linregress(theta_j, mean_j)
            # The coefficient of determination, NOT the squared correlation: the two
            # differ by exactly the affine error the posterior mean makes, and this
            # panel used to print r^2 under the name R^2 (+0.056 on average, +0.224
            # on 12p for the published CloneMLP). See docs/EVALUATION_PRACTICE_REVIEW.md.
            sse = float(((theta_j - mean_j) ** 2).sum())
            sst = float(((theta_j - theta_j.mean()) ** 2).sum())
            r2_true = float("nan") if sst == 0 else 1.0 - sse / sst

            # Colour each point by absolute error: blue = low, red = high.
            err = np.abs(mean_j - theta_j)
            norm_err = (err - err.min()) / (err.max() - err.min() + 1e-8)
            colors = plt.cm.RdYlBu_r(norm_err)

            ax.scatter(
                theta_j,
                mean_j,
                c=colors,
                s=5,
                alpha=0.55,
                linewidths=0,
                rasterized=True,
            )

            ax.plot(
                [-lim, lim],
                [-lim, lim],
                color="black",
                linewidth=0.8,
                linestyle="--",
                zorder=5,
            )

            x_line = np.array([-lim, lim])
            ax.plot(
                x_line,
                slope * x_line + intercept,
                color=REF_ZERO_RED,
                linewidth=0.9,
                linestyle="-",
                zorder=6,
            )

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_aspect("equal")

            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_GREY)
                spine.set_linewidth(0.5)
            ax.tick_params(axis="both", length=2, pad=2)

            r2_color = R2_GOOD_GREEN if r2_true >= r2_good_threshold else R2_POOR_ORANGE
            ax.set_title(
                f"{labels[j]},  $R^2 = {r2_true:.2f}$  ($r = {r:.2f}$)",
                fontsize=7,
                pad=3,
                color=r2_color,
                fontweight="bold",
            )

            # Axis labels only on outer edges to avoid clutter.
            if k % n_cols == 0:
                ax.set_ylabel("Post. mean", labelpad=3)
            else:
                ax.set_yticklabels([])

            if k >= (n_rows - 1) * n_cols:
                ax.set_xlabel("True $\\theta$", labelpad=3)
            else:
                ax.set_xticklabels([])

        # Hide unused subplots (24 slots - 22 used = 2 empty).
        for ax in axes_flat[len(idx) :]:
            ax.set_visible(False)

        fig.legend(
            handles=legend_elems,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.01),
            framealpha=0.9,
            edgecolor=SPINE_GREY,
        )

        fig.suptitle(
            "True vs. Posterior Mean: Recovery of Chromosomal "
            "Copy-Number Coefficients  —  " + str(panel["title"]),
            fontweight="bold",
            y=1.02,
        )

        fig.text(
            0.5,
            -0.01,
            "Points coloured by absolute error (blue = low, red = high). "
            "Black dashed = ideal ($y=x$). Red line = fitted regression.",
            ha="center",
            fontsize=7,
            color=FOOTNOTE_GREY,
            style="italic",
        )

        fig.tight_layout(h_pad=1.0, w_pad=0.5)

        saved_paths[str(panel["fname"])] = _save_png_pdf(
            fig, out_dir, str(panel["fname"]), facecolor="white"
        )

    return saved_paths


# --------------------------------------------------------------------------- #
# Correlation spreadsheet
# --------------------------------------------------------------------------- #


def write_correlation_excel(
    table: pd.DataFrame,
    out_dir: Path,
    *,
    filename: str = "correlation_true_vs_postmean.xlsx",
    sheet_name: str = "Correlations",
) -> Path:
    """Write the per-arm correlation table to an Excel workbook.

    Preserved from ``Base_NPE/z-score_violin.py:724-735``: the openpyxl engine,
    the single sheet named "Correlations", no index column, and the column
    auto-width rule ``max(len(str(cell)) for cell in column) + 3``.

    The 6-decimal rounding happens upstream, in
    :func:`cancer_sbi.evaluation.diagnostics.correlation_table`.

    Args:
        table: The correlation DataFrame to write.
        out_dir: Directory to write into; created if missing.
        filename: Workbook file name.
        sheet_name: Worksheet name.

    Returns:
        The path written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_path = out_dir / filename

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name=sheet_name)

        # Auto-fit column widths for readability.
        ws = writer.sheets[sheet_name]
        for col in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0 for cell in col
            )
            ws.column_dimensions[col[0].column_letter].width = max_len + 3

    return excel_path


# --------------------------------------------------------------------------- #
# Prior vs posterior KDE grids
# --------------------------------------------------------------------------- #


def _draw_kde_panel(
    ax: Any,
    j: int,
    x_grid: np.ndarray,
    prior_pdf: np.ndarray,
    pooled_samples: np.ndarray,
    selected_samples: Sequence[np.ndarray] | None,
    selected_indices: Sequence[int] | None,
    case_colors: Sequence[str] | None,
    theta_all: torch.Tensor | np.ndarray | None,
    labels: Sequence[str],
    mode: str,
) -> None:
    """Draw one chromosome-arm KDE panel onto an existing Axes.

    Preserved from ``SetTransformer_NPE/kde_prior_vs_posterior.py:204-273``,
    including the broad ``except Exception`` around each ``gaussian_kde`` call:
    a degenerate marginal must skip its curve rather than kill the whole figure.

    Args:
        ax: Target axes.
        j: Index of the arm to draw.
        x_grid: Evaluation grid, shape (500,).
        prior_pdf: Prior density on ``x_grid``, shape (500,).
        pooled_samples: All posterior draws pooled, shape (N*S, 44).
        selected_samples: Per-case draws, 5 arrays of shape (S, 44). Only used in
            "full" mode.
        selected_indices: The 5 test-case indices. Only used in "full" mode.
        case_colors: One colour per selected case. Only used in "full" mode.
        theta_all: True parameters, shape (N, 44). Only used in "full" mode.
        labels: Arm names, length 44.
        mode: ``"pooled_only"`` for prior + pooled posterior, ``"full"`` to add
            the 5 per-case curves and their true-theta verticals.
    """
    ax.set_facecolor("white")

    # 1. Prior: filled blue area with a solid outline. Identical in every panel.
    ax.fill_between(
        x_grid, prior_pdf, alpha=0.35, color=PRIOR_FILL_BLUE, linewidth=0, zorder=1
    )
    ax.plot(
        x_grid, prior_pdf, color=WELL_CALIBRATED_BLUE, linewidth=1.0, linestyle="-",
        zorder=2,
    )

    # 2. Pooled posterior: thick black dashed. The average posterior behaviour.
    try:
        kde_pooled = gaussian_kde(pooled_samples[:, j], bw_method="scott")
        ax.plot(
            x_grid,
            kde_pooled(x_grid),
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=8,
            alpha=0.85,
        )
    except Exception as exc:  # noqa: BLE001 -- preserved from the original
        print(f"Pooled KDE failed for {labels[j]}: {exc}")

    # 3. Selected cases, "full" mode only.
    if mode == "full":
        assert selected_samples is not None
        assert selected_indices is not None
        assert case_colors is not None
        assert theta_all is not None
        theta_np = _to_numpy(theta_all)

        for ci, (color, samp_ci) in enumerate(zip(case_colors, selected_samples)):
            samples_j = samp_ci[:, j]  # (S,) for this arm

            try:
                kde_case = gaussian_kde(samples_j, bw_method="scott")
                ax.plot(
                    x_grid,
                    kde_case(x_grid),
                    color=color,
                    linewidth=1.0,
                    linestyle="-",
                    alpha=0.80,
                    zorder=3 + ci,
                )
            except Exception as exc:  # noqa: BLE001 -- preserved from the original
                print(f"Case KDE failed for {labels[j]}, case {ci}: {exc}")

            # True theta vertical line, same colour as its posterior curve.
            theta_true_j = float(theta_np[selected_indices[ci], j])
            ax.axvline(
                theta_true_j,
                color=color,
                linewidth=0.8,
                linestyle=":",
                alpha=0.9,
                zorder=10 + ci,
            )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(bottom=0)
    ax.set_title(labels[j], fontsize=7, pad=3, fontweight="bold", color="black")

    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_GREY)
        spine.set_linewidth(0.5)

    ax.yaxis.grid(True, color=KDE_GRID_GREY, linewidth=0.4, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, pad=2)


def _save_kde_figure(
    panel: dict[str, Any],
    x_grid: np.ndarray,
    prior_pdf: np.ndarray,
    pooled_samples: np.ndarray,
    selected_samples: Sequence[np.ndarray] | None,
    selected_indices: Sequence[int] | None,
    case_colors: Sequence[str] | None,
    theta_all: torch.Tensor | np.ndarray | None,
    labels: Sequence[str],
    legend_handles: list[Any],
    suptitle_text: str,
    footnote_text: str,
    out_dir: Path,
    mode: str,
) -> dict[str, Path]:
    """Build and save one 4x6 KDE grid.

    Preserved from ``SetTransformer_NPE/kde_prior_vs_posterior.py:280-347``:
    figure size (13, 9), 24 slots for 22 arms with the 2 spares hidden, y labels
    only on the left column, x labels only on the bottom row, a legend anchored
    at (0.98, 0.01), and ``tight_layout(h_pad=1.2, w_pad=0.5)``.

    Args:
        panel: ``{"indices": [...], "subtitle": str, "fname": str}``.
        x_grid: Evaluation grid, shape (500,).
        prior_pdf: Prior density on ``x_grid``, shape (500,).
        pooled_samples: Pooled posterior draws, shape (N*S, 44).
        selected_samples: Per-case draws, or None in "pooled_only" mode.
        selected_indices: The 5 case indices, or None.
        case_colors: The 5 case colours, or None.
        theta_all: True parameters, or None.
        labels: Arm names, length 44.
        legend_handles: Legend artists, built by the caller.
        suptitle_text: Bold figure title.
        footnote_text: Italic caption under the figure.
        out_dir: Directory to write into.
        mode: ``"pooled_only"`` or ``"full"``.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    idx_list: list[int] = panel["indices"]
    n_cols, n_rows = 6, 4  # 24 slots for 22 arms

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 9))
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    for k, j in enumerate(idx_list):
        ax = axes_flat[k]

        _draw_kde_panel(
            ax,
            j,
            x_grid,
            prior_pdf,
            pooled_samples,
            selected_samples,
            selected_indices,
            case_colors,
            theta_all,
            labels,
            mode=mode,
        )

        if k % n_cols == 0:
            ax.set_ylabel("Density", labelpad=3)
        else:
            ax.set_yticklabels([])

        if k >= (n_rows - 1) * n_cols:
            ax.set_xlabel("$\\theta$", labelpad=3)
        else:
            ax.set_xticklabels([])

    # Hide unused slots (24 - 22 = 2).
    for ax in axes_flat[len(idx_list) :]:
        ax.set_visible(False)

    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.01),
        framealpha=0.92,
        edgecolor=SPINE_GREY,
        fontsize=7,
    )

    fig.suptitle(suptitle_text, fontweight="bold", y=1.02)

    fig.text(
        0.5,
        -0.005,
        footnote_text,
        ha="center",
        fontsize=6.5,
        color=FOOTNOTE_GREY,
        style="italic",
    )

    fig.tight_layout(h_pad=1.2, w_pad=0.5)

    return _save_png_pdf(fig, out_dir, str(panel["fname"]), facecolor="white")


def _prior_pdf(
    x_grid: np.ndarray, prior_mu: float, prior_std: float
) -> np.ndarray:
    """Evaluate the analytical Gaussian prior on a grid.

    Args:
        x_grid: Evaluation points, shape (G,).
        prior_mu: Prior mean.
        prior_std: Prior standard deviation.

    Returns:
        Prior density, shape (G,).
    """
    return (1.0 / (prior_std * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_grid - prior_mu) / prior_std) ** 2
    )


def plot_kde_pooled_only(
    pooled_samples: np.ndarray,
    n_test: int,
    out_dir: Path,
    *,
    prior_mu: float = DEFAULT_PRIOR_MU,
    prior_std: float = DEFAULT_PRIOR_STD,
    labels: Sequence[str] = ARM_LABELS,
) -> dict[str, dict[str, Path]]:
    """Prior vs pooled posterior: the clean two-curve KDE grids.

    Preserved from ``SetTransformer_NPE/kde_prior_vs_posterior.py:361-418``:
    ``x_grid = linspace(-1.5, 1.5, 500)``, prior drawn as a filled ``#AEC6E8``
    area with a ``#4A90D9`` outline, pooled posterior as a 1.5pt black dashed
    line, two figures of 22 arms each.

    **The prior standard deviation changed, on purpose.** The original scripts
    built this reference curve with ``coeff_std = math.sqrt(0.2) = 0.4472``
    (``kde_prior_vs_posterior.py:110``) while the simulator drew coefficients
    with standard deviation 0.2 -- the sqrt was applied to a value that was
    already a standard deviation, not a variance. The plotted prior was therefore
    2.24x too wide and about 2.24x too low at its peak, which made the posterior
    look far less informative than it is: the published KDE figures understate
    how much the model learned. ``prior_std`` defaults to the correct 0.2 here,
    so regenerating ``kde_pooled_only_chr1_11``, ``kde_pooled_only_chr12_22``,
    ``kde_full_chr1_11`` and ``kde_full_chr12_22`` will produce a visibly
    narrower, taller blue prior than the versions currently on disk. Nothing else
    in those figures moves, and no other reported number depends on this value.
    Pass :data:`LEGACY_PRIOR_STD` to reproduce the old figures for comparison.

    Note that the legend and footnote in the originals already *claimed*
    ``N(0, 0.2)`` while drawing 0.447, so with the default the text and the curve
    finally agree.

    Args:
        pooled_samples: Posterior draws from every test case, concatenated,
            shape (N*S, 44).
        n_test: Number of test cases, interpolated into the legend and footnote.
        out_dir: Directory to write into.
        prior_mu: Prior mean; 0.0 in every original.
        prior_std: Prior standard deviation; see above.
        labels: Arm names, length 44.

    Returns:
        A dict keyed by file stem (``kde_pooled_only_chr1_11``,
        ``kde_pooled_only_chr12_22``), each value ``{"png", "pdf"}``.
    """
    x_grid = np.linspace(-1.5, 1.5, 500)
    prior_pdf = _prior_pdf(x_grid, prior_mu, prior_std)

    legend_handles = [
        Patch(
            facecolor=PRIOR_FILL_BLUE,
            edgecolor=WELL_CALIBRATED_BLUE,
            linewidth=0.8,
            alpha=0.8,
            label=f"Prior  $\\mathcal{{N}}(0,\\ {prior_std:g})$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Pooled posterior  (all {n_test} test cases)",
        ),
    ]

    panels = [
        {
            "indices": _FIRST_HALF,
            "subtitle": "Chromosomes 1-11",
            "fname": "kde_pooled_only_chr1_11",
        },
        {
            "indices": _SECOND_HALF,
            "subtitle": "Chromosomes 12-22",
            "fname": "kde_pooled_only_chr12_22",
        },
    ]

    saved: dict[str, dict[str, Path]] = {}
    for panel in panels:
        saved[str(panel["fname"])] = _save_kde_figure(
            panel=panel,
            x_grid=x_grid,
            prior_pdf=prior_pdf,
            pooled_samples=pooled_samples,
            selected_samples=None,
            selected_indices=None,
            case_colors=None,
            theta_all=None,
            labels=labels,
            legend_handles=legend_handles,
            suptitle_text=(
                "Prior vs. Pooled Posterior: KDE of Chromosomal "
                "Copy-Number Coefficients  —  " + str(panel["subtitle"])
            ),
            footnote_text=(
                f"Blue filled = prior $\\mathcal{{N}}(0, {prior_std:g})$.  "
                f"Black dashed = pooled posterior KDE ({n_test} test cases pooled)."
            ),
            out_dir=out_dir,
            mode="pooled_only",
        )
    return saved


def plot_kde_full(
    selected_samples: Sequence[np.ndarray],
    pooled_samples: np.ndarray,
    theta_all: torch.Tensor | np.ndarray,
    selected_indices: Sequence[int],
    n_test: int,
    out_dir: Path,
    *,
    case_colors: Sequence[str] = CASE_COLORS,
    prior_mu: float = DEFAULT_PRIOR_MU,
    prior_std: float = DEFAULT_PRIOR_STD,
    labels: Sequence[str] = ARM_LABELS,
) -> dict[str, dict[str, Path]]:
    """Prior + pooled posterior + five individual posteriors, as KDE grids.

    Preserved from ``SetTransformer_NPE/kde_prior_vs_posterior.py:434-509``. The
    legend gains one entry per selected case, labelled with that case's mean true
    theta to 2 decimals, plus a grey dotted entry for the true-theta verticals.

    The same prior-standard-deviation correction described in
    :func:`plot_kde_pooled_only` applies to these two figures. Read that
    docstring before regenerating them.

    Args:
        selected_samples: Posterior draws for the 5 selected cases, each of
            shape (S, 44).
        pooled_samples: Posterior draws from every test case, shape (N*S, 44).
        theta_all: True parameters, shape (N, 44).
        selected_indices: The 5 test-case indices, from
            :func:`cancer_sbi.evaluation.diagnostics.select_diverse_test_cases`.
        n_test: Number of test cases, interpolated into the legend and footnote.
        out_dir: Directory to write into.
        case_colors: One colour per selected case.
        prior_mu: Prior mean.
        prior_std: Prior standard deviation; see :func:`plot_kde_pooled_only`.
        labels: Arm names, length 44.

    Returns:
        A dict keyed by file stem (``kde_full_chr1_11``, ``kde_full_chr12_22``),
        each value ``{"png", "pdf"}``.
    """
    x_grid = np.linspace(-1.5, 1.5, 500)
    prior_pdf = _prior_pdf(x_grid, prior_mu, prior_std)
    theta_np = _to_numpy(theta_all)

    legend_handles: list[Any] = [
        Patch(
            facecolor=PRIOR_FILL_BLUE,
            edgecolor=WELL_CALIBRATED_BLUE,
            linewidth=0.8,
            alpha=0.8,
            label=f"Prior  $\\mathcal{{N}}(0,\\ {prior_std:g})$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            linestyle="--",
            label=f"Pooled posterior  (all {n_test} test cases)",
        ),
    ]
    for ci, (color, idx) in enumerate(zip(case_colors, selected_indices)):
        mean_theta = float(theta_np[idx].mean())
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.0,
                linestyle="-",
                label=f"Case {ci + 1}  "
                f"($\\bar{{\\theta}}_{{true}}={mean_theta:.2f}$)",
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=0.8,
            linestyle=":",
            label="True $\\theta$ (per selected case)",
        )
    )

    panels = [
        {
            "indices": _FIRST_HALF,
            "subtitle": "Chromosomes 1-11",
            "fname": "kde_full_chr1_11",
        },
        {
            "indices": _SECOND_HALF,
            "subtitle": "Chromosomes 12-22",
            "fname": "kde_full_chr12_22",
        },
    ]

    saved: dict[str, dict[str, Path]] = {}
    for panel in panels:
        saved[str(panel["fname"])] = _save_kde_figure(
            panel=panel,
            x_grid=x_grid,
            prior_pdf=prior_pdf,
            pooled_samples=pooled_samples,
            selected_samples=selected_samples,
            selected_indices=selected_indices,
            case_colors=case_colors,
            theta_all=theta_all,
            labels=labels,
            legend_handles=legend_handles,
            suptitle_text=(
                "Prior vs. Learned Posterior: KDE of Chromosomal "
                "Copy-Number Coefficients  —  " + str(panel["subtitle"])
            ),
            footnote_text=(
                f"Blue filled = prior $\\mathcal{{N}}(0, {prior_std:g})$.  "
                f"Black dashed = pooled posterior ({n_test} test cases).  "
                "Coloured lines = posterior for 5 selected test cases.  "
                "Dotted verticals = true $\\theta$."
            ),
            out_dir=out_dir,
            mode="full",
        )
    return saved


# --------------------------------------------------------------------------- #
# sbi pairplot
# --------------------------------------------------------------------------- #


def plot_posterior_pairplot(
    samples: torch.Tensor,
    theta_true: torch.Tensor,
    out_dir: Path,
    *,
    test_index: int,
    dims: Sequence[int] | None = None,
    limits: tuple[float, float] = (-2.0, 5.0),
    labels: Sequence[str] = ARM_LABELS,
    figsize: tuple[float, float] = (12, 12),
) -> dict[str, Path]:
    """The sbi corner plot of one test case's posterior.

    Preserved from ``Base_NPE/ppc-plot.py:137-166``: the scatter upper triangle
    with ``marker="."`` and ``s=5``, the true parameters drawn as red ``+``
    markers of size 10, the file stem
    ``pairplot_test{i}_dims_{first}-{last}``, and the suptitle at ``y=1.02``.

    The default ``limits`` of (-2.0, 5.0) are the original's and are worth a
    second look: they are asymmetric and far wider than the prior, so every
    marginal is squeezed into the left of its panel. Preserved as-is because the
    existing pairplot in ``figures/ppc/`` was made with them.

    Args:
        samples: Posterior draws, shape (S, 44) or (S, 1, 44).
        theta_true: True parameters for this case, shape (1, 44).
        out_dir: Directory to write into. The original hard-coded
            ``"figures/ppc"``.
        test_index: Index of the test case, used in the file name and title.
        dims: Which parameter indices to plot; defaults to all 44.
        limits: ``(low, high)`` applied to every plotted dimension.
        labels: Arm names, length 44.
        figsize: Figure size passed to sbi.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    from sbi.analysis import pairplot  # local import: sbi is optional here

    idx = list(range(len(labels))) if dims is None else list(dims)

    if samples.dim() == 3:
        samples = samples.squeeze(1)

    samples_sel = samples[:, idx].detach().cpu()
    points_sel = theta_true[:, idx].detach().cpu()

    # One [low, high] row per plotted dimension.
    limits_tensor = torch.tensor([[limits[0], limits[1]]] * len(idx))
    plot_labels = [labels[j] for j in idx]

    fig, _axes = pairplot(
        samples=samples_sel,
        points=points_sel,
        limits=limits_tensor,
        figsize=figsize,
        upper="scatter",
        upper_kwargs=dict(marker=".", s=5),
        fig_kwargs=dict(
            points_offdiag=dict(marker="+", markersize=10),
            points_colors="red",
        ),
        labels=plot_labels,
    )

    fig.suptitle(f"Posterior Pairplot (test idx={test_index})", y=1.02)
    fig.tight_layout()

    stem = f"pairplot_test{test_index}_dims_{idx[0]}-{idx[-1]}"
    return _save_png_pdf(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# Simulation-based calibration (SBC)
#
# Ported from Base_NPE/test.ipynb, the only place SBC exists in the repository.
# The notebook displayed these inline and saved exactly one of them, so the file
# stems below are new; only the drawing parameters are inherited.
# --------------------------------------------------------------------------- #


def plot_sbc_rank_slice(
    ranks: torch.Tensor,
    num_posterior_samples: int,
    out_dir: Path,
    *,
    start: int,
    end: int,
    plot_type: str = "hist",
    num_bins: int | None = 20,
    figsize: tuple[float, float] = (10, 10),
    labels: Sequence[str] = ARM_LABELS,
    stem: str | None = None,
) -> dict[str, Path]:
    """One sbi SBC rank plot over a contiguous slice of parameters.

    Preserved from ``Base_NPE/test.ipynb`` cells 14-27: ``num_bins=20`` and the
    ``ranks[:, s:e].reshape(-1, e - s)`` reshape, which is a no-op for a
    contiguous slice but is kept so the call matches the notebook exactly.

    Args:
        ranks: SBC ranks, shape (N, 44).
        num_posterior_samples: The value SBC was run with; sbi uses it to place
            the uniform reference band.
        out_dir: Directory to write into.
        start: First parameter index, inclusive.
        end: Last parameter index, exclusive.
        plot_type: ``"hist"`` or ``"cdf"``.
        num_bins: Bin count; ``None`` lets sbi apply its own heuristic.
        figsize: Figure size.
        labels: Arm names, length 44.
        stem: File name without extension; defaults to
            ``sbc_rank_{plot_type}_{start}_{end}``.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    from sbi.analysis.plot import sbc_rank_plot  # local import: sbi is optional

    fig, _ax = sbc_rank_plot(
        ranks[:, start:end].reshape(-1, (end - start)),
        num_posterior_samples,
        plot_type=plot_type,
        num_bins=num_bins,
        figsize=figsize,
        parameter_labels=[labels[i] for i in range(start, end)],
    )

    stem = stem or f"sbc_rank_{plot_type}_{start}_{end}"
    return _save_png_pdf(fig, out_dir, stem)


def plot_sbc_rank_hist_panels(
    ranks: torch.Tensor,
    num_posterior_samples: int,
    out_dir: Path,
    *,
    chunk_size: int = 10,
    labels: Sequence[str] = ARM_LABELS,
    figsize: tuple[float, float] = (10, 10),
    last_figsize: tuple[float, float] = (10, 5),
) -> dict[str, dict[str, Path]]:
    """The SBC rank histograms, ten arms at a time.

    Preserved from ``Base_NPE/test.ipynb`` cells 14-18: chunks 0-10, 10-20,
    20-30 and 30-40 at figure size (10, 10), and the short final chunk 40-44 at
    (10, 5).

    Args:
        ranks: SBC ranks, shape (N, 44).
        num_posterior_samples: The value SBC was run with.
        out_dir: Directory to write into.
        chunk_size: Parameters per figure.
        labels: Arm names, length 44.
        figsize: Figure size for a full chunk.
        last_figsize: Figure size for a final short chunk.

    Returns:
        A dict keyed by file stem, each value ``{"png", "pdf"}``.
    """
    n_params = ranks.shape[1]
    saved: dict[str, dict[str, Path]] = {}

    for start in range(0, n_params, chunk_size):
        end = min(start + chunk_size, n_params)
        size = figsize if (end - start) == chunk_size else last_figsize
        stem = f"sbc_rank_hist_{start}_{end}"
        saved[stem] = plot_sbc_rank_slice(
            ranks,
            num_posterior_samples,
            out_dir,
            start=start,
            end=end,
            plot_type="hist",
            num_bins=20,
            figsize=size,
            labels=labels,
            stem=stem,
        )
    return saved


def plot_sbc_rank_cdf(
    ranks: torch.Tensor,
    num_posterior_samples: int,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    figsize: tuple[float, float] = (20, 20),
    stem: str = "sbc_rank_plot_chromosomes_SetTransformer",
) -> dict[str, Path]:
    """The all-44-arm SBC rank CDF plot.

    Preserved from ``Base_NPE/test.ipynb`` cells 19-20: all 44 parameters in one
    (20, 20) figure, ``num_bins=20``, saved at dpi 300. The default stem is the
    notebook's literal file name, "SetTransformer" included, even though the
    notebook lives in ``Base_NPE/`` -- it loads
    ``SetTransformer_NPE_Freq_mean.pkl`` (cell 5), so the name describes the
    model, not the folder. Override ``stem`` for a different model.

    Args:
        ranks: SBC ranks, shape (N, 44).
        num_posterior_samples: The value SBC was run with.
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        figsize: Figure size.
        stem: File name without extension.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    return plot_sbc_rank_slice(
        ranks,
        num_posterior_samples,
        out_dir,
        start=0,
        end=len(labels),
        plot_type="cdf",
        num_bins=20,
        figsize=figsize,
        labels=labels,
        stem=stem,
    )


def _plot_sbc_bar(
    heights: np.ndarray,
    colors: list[str],
    out_dir: Path,
    *,
    ylabel: str,
    title: str,
    stem: str,
    labels: Sequence[str],
    threshold: Callable[[Any], None] | None = None,
) -> dict[str, Path]:
    """Shared body of the three SBC bar charts.

    Preserved from ``Base_NPE/test.ipynb`` cells 29-31: figure size (16, 6) at
    dpi 450, ``width=0.8`` bars, x tick labels rotated 90 degrees, and the
    misspelled x label "selection coeeficents parameters", which is kept because
    it is what the existing plots say.

    Args:
        heights: Bar heights, shape (44,).
        colors: One colour per bar.
        out_dir: Directory to write into.
        ylabel: y-axis label.
        title: Axes title.
        stem: File name without extension.
        labels: Arm names, length 44.
        threshold: Optional callable that draws the acceptance line or band onto
            the axes. The notebook drew neither -- it encoded the threshold in
            the bar colours only -- so passing None reproduces it exactly.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    x_indices = np.arange(len(heights))

    fig, ax = plt.subplots(figsize=(16, 6), dpi=450)
    ax.bar(x_indices, heights, color=colors, width=0.8)
    if threshold is not None:
        threshold(ax)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(list(labels), rotation=90)
    ax.set_xlabel("selection coeeficents parameters")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # dpi matches the figure's own 450, which the notebook chose for these.
    return _save_png_pdf(fig, out_dir, stem, dpi=450)


def plot_sbc_ks_pvalues(
    ks_pvals: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    alpha: float = 0.05,
    show_threshold: bool = True,
    stem: str = "sbc_ks_pvalues",
) -> dict[str, Path]:
    """Bar chart of the per-arm Kolmogorov-Smirnov p-values.

    Preserved from ``Base_NPE/test.ipynb`` cell 29: a bar is red when
    ``p <= 0.05`` and blue otherwise.

    The notebook conveyed the 0.05 threshold through bar colour alone. The
    horizontal line is the one addition, because the specification for this port
    asks for it; pass ``show_threshold=False`` to reproduce the notebook exactly.

    Args:
        ks_pvals: Per-arm p-values, shape (44,).
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        alpha: Significance threshold used for colouring and the line.
        show_threshold: Draw the dashed line at ``alpha``.
        stem: File name without extension.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    heights = _to_numpy(ks_pvals)
    colors = [SBC_FAIL_RED if h <= alpha else SBC_PASS_BLUE for h in heights]

    def _draw_threshold(ax: Any) -> None:
        ax.axhline(alpha, color="black", linestyle="--", linewidth=0.8)

    return _plot_sbc_bar(
        heights,
        colors,
        out_dir,
        ylabel="p-values",
        title="kolmogorov-smirnov p-values",
        stem=stem,
        labels=labels,
        threshold=_draw_threshold if show_threshold else None,
    )


def plot_sbc_c2st_bar(
    values: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    ylabel: str,
    title: str,
    stem: str,
    labels: Sequence[str] = ARM_LABELS,
    band: tuple[float, float] = (0.45, 0.55),
    show_band: bool = True,
) -> dict[str, Path]:
    """Bar chart of a per-arm c2st accuracy.

    Preserved from ``Base_NPE/test.ipynb`` cells 30-31: a bar is red when the
    accuracy falls outside ``(0.45, 0.55)`` and blue when inside -- 0.5 is the
    chance level a well-calibrated posterior should sit at.

    The notebook conveyed the band through bar colour alone. The shaded band is
    the one addition, because the specification for this port asks for it; pass
    ``show_band=False`` to reproduce the notebook exactly.

    Args:
        values: Per-arm c2st accuracies, shape (44,).
        out_dir: Directory to write into.
        ylabel: y-axis label.
        title: Axes title.
        stem: File name without extension.
        labels: Arm names, length 44.
        band: ``(low, high)`` acceptance band.
        show_band: Shade the band.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    heights = _to_numpy(values)
    low, high = band
    colors = [
        SBC_FAIL_RED if (h > high or h < low) else SBC_PASS_BLUE for h in heights
    ]

    def _draw_band(ax: Any) -> None:
        ax.axhspan(low, high, color="grey", alpha=0.15, zorder=0)

    return _plot_sbc_bar(
        heights,
        colors,
        out_dir,
        ylabel=ylabel,
        title=title,
        stem=stem,
        labels=labels,
        threshold=_draw_band if show_band else None,
    )


def plot_sbc_c2st_ranks(
    c2st_ranks: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    band: tuple[float, float] = (0.45, 0.55),
    show_band: bool = True,
) -> dict[str, Path]:
    """The c2st-on-ranks bar chart (``Base_NPE/test.ipynb`` cell 30).

    Args:
        c2st_ranks: ``check_sbc`` output ``c2st_ranks``, shape (44,).
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        band: ``(low, high)`` acceptance band.
        show_band: Shade the band; False reproduces the notebook exactly.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    return plot_sbc_c2st_bar(
        c2st_ranks,
        out_dir,
        ylabel="c2st",
        title="c2st accuracies",
        stem="sbc_c2st_ranks",
        labels=labels,
        band=band,
        show_band=show_band,
    )


def plot_sbc_c2st_dap(
    c2st_dap: torch.Tensor | np.ndarray,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    band: tuple[float, float] = (0.45, 0.55),
    show_band: bool = True,
) -> dict[str, Path]:
    """The c2st-on-data-averaged-posterior bar chart (cell 31).

    Args:
        c2st_dap: ``check_sbc`` output ``c2st_dap``, shape (44,).
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        band: ``(low, high)`` acceptance band.
        show_band: Shade the band; False reproduces the notebook exactly.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    return plot_sbc_c2st_bar(
        c2st_dap,
        out_dir,
        ylabel="c2st_dap",
        title="c2st_dap accuracies",
        stem="sbc_c2st_dap",
        labels=labels,
        band=band,
        show_band=show_band,
    )


# --------------------------------------------------------------------------- #
# Prior predictive check panels
#
# Ported from Base_NPE/PriorPredictiveCheck.ipynb. Only the two panels that are
# not already covered by z-score_violin.py are kept.
# --------------------------------------------------------------------------- #


def plot_l2_per_case(
    l2: np.ndarray,
    out_dir: Path,
    *,
    stem: str = "l2_per_test_case",
) -> dict[str, Path]:
    """Bar chart of the L2 distance for every test case.

    Preserved from ``Base_NPE/PriorPredictiveCheck.ipynb`` cell 12, which is the
    whole of that cell: ``plt.bar(np.arange(len(l2_all)), l2_all)`` with default
    colours and no title. Titles and labels are added here because the notebook
    relied on the surrounding prose; the bars themselves are unchanged.

    Args:
        l2: Per-case L2 distances, shape (N,).
        out_dir: Directory to write into.
        stem: File name without extension.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    values = np.asarray(l2)
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(values)), values)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("$\\|\\theta_{true} - \\mu_{post}\\|_2$")
    ax.set_title("L2 distance per test case")
    fig.tight_layout()

    return _save_png_pdf(fig, out_dir, stem)


def plot_abs_z_vs_index(
    z_all: torch.Tensor | np.ndarray,
    parameter_index: int,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    stem: str | None = None,
) -> dict[str, Path]:
    """|z| against test-sample index for one arm, with 1 and 2 SD lines.

    Preserved from ``Base_NPE/PriorPredictiveCheck.ipynb`` cell 9: figure size
    (6, 3), dot markers of size 3, a red dashed line at 1 and an orange dashed
    line at 2.

    Args:
        z_all: z-scores, shape (N, 44).
        parameter_index: Which arm to plot.
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        stem: File name without extension; defaults to
            ``abs_zscore_{label}``.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    z = _to_numpy(z_all)[:, parameter_index]
    abs_z = np.abs(z)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(abs_z, ".", markersize=3)
    ax.axhline(1, color="red", linestyle="--", label="1 SD")
    ax.axhline(2, color="orange", linestyle="--", label="2 SD")
    ax.set_title(f"Absolute Z-scores for Parameter {labels[parameter_index]}")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("|Z-score|")
    ax.legend()
    fig.tight_layout()

    stem = stem or f"abs_zscore_{labels[parameter_index]}"
    return _save_png_pdf(fig, out_dir, stem)


def plot_z_vs_index_with_bands(
    z_all: torch.Tensor | np.ndarray,
    parameter_index: int,
    out_dir: Path,
    *,
    labels: Sequence[str] = ARM_LABELS,
    stem: str | None = None,
) -> dict[str, Path]:
    """Signed z against test-sample index for one arm, with +/-1 and +/-2 bands.

    Preserved from ``Base_NPE/PriorPredictiveCheck.ipynb`` cell 10: figure size
    (8, 5), a green ``axhspan(-1, 1)`` at alpha 0.15, a yellow ``axhspan(-2, 2)``
    at alpha 0.10 drawn on top of it, steelblue dots of size 4, a black zero
    line, and the counts box pinned at axes coordinates (0.02, 0.98).

    Note that the two bands are drawn in that order, so the +/-1 region shows
    green tinted by yellow rather than green alone -- that is what the notebook
    produced.

    Args:
        z_all: z-scores, shape (N, 44).
        parameter_index: Which arm to plot.
        out_dir: Directory to write into.
        labels: Arm names, length 44.
        stem: File name without extension; defaults to ``zscore_{label}``.

    Returns:
        ``{"png": <path>, "pdf": <path>}``.
    """
    z = _to_numpy(z_all)[:, parameter_index]
    abs_z = np.abs(z)
    n = len(z)

    within_1sd = int((abs_z < 1).sum())
    within_2sd = int((abs_z < 2).sum())
    beyond_2sd = int((abs_z >= 2).sum())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhspan(-1, 1, color="green", alpha=0.15, label="±1 SD")
    ax.axhspan(-2, 2, color="yellow", alpha=0.10, label="±2 SD")

    ax.plot(z, ".", color="steelblue", markersize=4)
    ax.axhline(0, color="black", linewidth=1)

    ax.set_title(f"Z-scores for Parameter {labels[parameter_index]}")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Z-score")

    summary = (
        f"Total: {n}\n"
        f"|z| < 1: {within_1sd}  ({within_1sd / n:.1%})\n"
        f"|z| < 2: {within_2sd}  ({within_2sd / n:.1%})\n"
        f"|z| ≥ 2: {beyond_2sd} ({beyond_2sd / n:.1%})"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    stem = stem or f"zscore_{labels[parameter_index]}"
    return _save_png_pdf(fig, out_dir, stem)
