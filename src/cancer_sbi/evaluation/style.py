"""Matplotlib styling shared by every evaluation figure.

The paper figures were produced with an "Elsevier" rcParams block that was
copy-pasted into the top of several scripts. This module holds the single
authoritative copy of that block plus the colour literals that were repeated
inline across those scripts.

Originals:
    * ``Base_NPE/z-score_violin.py`` lines 22-43 -- the full-size preset used for
      the violin / scatter / bar / histogram figures in the paper.
    * ``SetTransformer_NPE/kde_prior_vs_posterior.py`` lines 9-26 -- the same
      preset, one point smaller, used for the dense 4x6 KDE grids.

Nothing here runs at import time: rcParams are only touched when
:func:`use_paper_style` is called, and the non-interactive backend is an
explicit opt-in via :func:`use_agg_backend`. The originals called
``matplotlib.use("Agg")`` at module scope, which silently broke interactive use
of anything that imported them.
"""

import math
from typing import Any

import matplotlib

__all__ = [
    "PAPER_RCPARAMS",
    "PAPER_RCPARAMS_SMALL",
    "SCALE_FULL",
    "SCALE_SMALL",
    "paper_rcparams",
    "use_paper_style",
    "use_agg_backend",
    "WELL_CALIBRATED_BLUE",
    "BIASED_RED",
    "OVERCONFIDENT_ORANGE",
    "REF_ZERO_RED",
    "REF_TWO_SIGMA_ORANGE",
    "SPINE_GREY",
    "GRID_GREY",
    "KDE_GRID_GREY",
    "FOOTNOTE_GREY",
    "R2_GOOD_GREEN",
    "R2_POOR_ORANGE",
    "PRIOR_FILL_BLUE",
    "CASE_COLORS",
    "SBC_FAIL_RED",
    "SBC_PASS_BLUE",
]


# --------------------------------------------------------------------------- #
# Colour constants
#
# Each of these was written as a bare hex literal, usually more than once, in
# the scripts named beside it. They are named here so that a future change is a
# one-line change rather than a grep-and-hope.
# --------------------------------------------------------------------------- #

#: Violin fill for a well-calibrated arm, and the prior outline in the KDE
#: grids (``Base_NPE/z-score_violin.py:399``, ``kde_prior_vs_posterior.py:222``).
WELL_CALIBRATED_BLUE: str = "#4A90D9"

#: Violin fill for a biased arm, |mean z| > 1.0 (``z-score_violin.py:397``).
BIASED_RED: str = "#C0504D"

#: Violin fill for an overconfident arm, std z > 1.5 (``z-score_violin.py:395``).
OVERCONFIDENT_ORANGE: str = "#E07B54"

#: The ``z = 0`` reference line in the violins and the fitted-regression line in
#: the scatter grids (``z-score_violin.py:339`` and ``:594``).
REF_ZERO_RED: str = "#CC2222"

#: The ``z = +/-2`` reference lines in the violins (``z-score_violin.py:341-343``).
REF_TWO_SIGMA_ORANGE: str = "#CC7700"

#: Axes spines and legend edges everywhere (``z-score_violin.py:355``, ``:457``).
SPINE_GREY: str = "#AAAAAA"

#: Horizontal grid in the violin panels (``z-score_violin.py:358``).
GRID_GREY: str = "#DDDDDD"

#: Horizontal grid in the KDE panels -- one shade lighter than the violins.
#: Preserved as a separate constant because the two figures genuinely differ
#: (``kde_prior_vs_posterior.py:271``); do not unify them.
KDE_GRID_GREY: str = "#EEEEEE"

#: Caption-style footnote text under the scatter and KDE grids
#: (``z-score_violin.py:651``, ``kde_prior_vs_posterior.py:335``).
FOOTNOTE_GREY: str = "#444444"

#: Scatter panel title colour when R^2 >= 0.70 (``z-score_violin.py:608``).
R2_GOOD_GREEN: str = "#1A7A1A"

#: Scatter panel title colour when R^2 < 0.70 (``z-score_violin.py:608``).
R2_POOR_ORANGE: str = "#CC5500"

#: Filled area under the prior curve in the KDE grids
#: (``kde_prior_vs_posterior.py:219``).
PRIOR_FILL_BLUE: str = "#AEC6E8"

#: The five-case palette for the per-case KDE curves
#: (``kde_prior_vs_posterior.py:157``).
CASE_COLORS: tuple[str, ...] = (
    "#E63946",
    "#F4A261",
    "#2A9D8F",
    "#457B9D",
    "#6A0572",
)

#: SBC bar colours. These are matplotlib *named* colours in the original
#: notebook, not hex (``Base_NPE/test.ipynb`` cells 29-31); kept verbatim so the
#: bars come out the exact same shade.
SBC_FAIL_RED: str = "red"
SBC_PASS_BLUE: str = "blue"


# --------------------------------------------------------------------------- #
# rcParams presets
# --------------------------------------------------------------------------- #

#: ``scale`` value that selects the full-size preset.
SCALE_FULL: float = 1.0

#: ``scale`` value that selects the one-point-smaller KDE preset.
SCALE_SMALL: float = 0.9

#: Verbatim copy of ``Base_NPE/z-score_violin.py`` lines 22-43.
#:
#: Times New Roman rather than Computer Modern because LaTeX is not installed on
#: the HPC nodes that produced the figures; Elsevier accepts it.
PAPER_RCPARAMS: dict[str, Any] = {
    # No LaTeX engine needed.
    "text.usetex": False,
    # Times New Roman -- closest system font to Elsevier / Computer Modern.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    # Font sizes (Elsevier standard: 10pt body).
    "font.size": 10,  # base size
    "axes.labelsize": 9,  # x/y axis labels
    "xtick.labelsize": 8,  # x tick labels (chr arm names)
    "ytick.labelsize": 8,  # y tick labels (z-score values)
    "legend.fontsize": 7,  # legend text
    "axes.titlesize": 9,  # panel subtitle (Chromosomes 1-11)
    "figure.titlesize": 10,  # suptitle
    # Clean figure style.
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
}

#: Verbatim copy of ``SetTransformer_NPE/kde_prior_vs_posterior.py`` lines 9-26.
#:
#: The KDE grids pack 22 panels into one figure, so every font drops one point
#: and the line weights drop from 0.8 to 0.6. Note that ``figure.titlesize`` and
#: ``legend.fontsize`` do *not* change -- that asymmetry is why this preset
#: cannot be expressed as a single multiplier over :data:`PAPER_RCPARAMS`.
PAPER_RCPARAMS_SMALL: dict[str, Any] = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.titlesize": 8,
    "figure.titlesize": 10,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
}


def paper_rcparams(scale: float = SCALE_FULL) -> dict[str, Any]:
    """Build the rcParams dict for a given figure scale.

    The two presets recorded above are the anchors: ``scale=1.0`` returns
    :data:`PAPER_RCPARAMS` and ``scale=0.9`` returns :data:`PAPER_RCPARAMS_SMALL`,
    both byte-for-byte as they appear in the originals. Any other value is a
    linear blend of the two anchors (extrapolated outside ``[0.9, 1.0]``), which
    is the only way one scalar can reproduce both presets: the small preset drops
    some fonts by a point while leaving ``figure.titlesize`` and
    ``legend.fontsize`` alone, so no single multiplier fits.

    Args:
        scale: Figure scale. Use 1.0 for the single-axes paper figures (violins,
            scatter grids, bar charts, histogram) and 0.9 for the dense 4x6 KDE
            grids. Non-anchor values may produce fractional font sizes, which
            matplotlib accepts.

    Returns:
        A fresh dict of rcParams keys to values; the caller owns it and may
        mutate it without affecting the module constants.
    """
    if math.isclose(scale, SCALE_FULL):
        return dict(PAPER_RCPARAMS)
    if math.isclose(scale, SCALE_SMALL):
        return dict(PAPER_RCPARAMS_SMALL)

    t = (scale - SCALE_SMALL) / (SCALE_FULL - SCALE_SMALL)
    blended: dict[str, Any] = {}
    for key, full_value in PAPER_RCPARAMS.items():
        small_value = PAPER_RCPARAMS_SMALL[key]
        # bool is a subclass of int, so exclude it explicitly: "text.usetex"
        # must stay a bool, not become 0.0.
        numeric = isinstance(full_value, (int, float)) and not isinstance(
            full_value, bool
        )
        if numeric:
            blended[key] = small_value + t * (full_value - small_value)
        else:
            blended[key] = full_value
    return blended


def use_paper_style(scale: float = SCALE_FULL) -> dict[str, Any]:
    """Apply the paper rcParams to the global matplotlib state.

    Call this once before building figures. It is a deliberate side effect and
    therefore a function, not module-level code as in the originals.

    Args:
        scale: See :func:`paper_rcparams`. 1.0 reproduces
            ``Base_NPE/z-score_violin.py:22-43``; 0.9 reproduces
            ``SetTransformer_NPE/kde_prior_vs_posterior.py:9-26``.

    Returns:
        The dict that was applied, so a caller can log or diff it.
    """
    params = paper_rcparams(scale)
    matplotlib.rcParams.update(params)
    return params


def use_agg_backend() -> None:
    """Switch matplotlib to the non-interactive ``Agg`` backend.

    Every original script ran ``matplotlib.use("Agg")`` at import time with the
    comment "Non-interactive backend for HPC". That is correct on a headless
    compute node and wrong everywhere else, so it is opt-in here. Call it from a
    CLI entry point or a batch script, never from library code.
    """
    matplotlib.use("Agg")
