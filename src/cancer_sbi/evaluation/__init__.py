"""Evaluation of a trained posterior: diagnostics, figures and styling.

This subpackage replaces the evaluation scripts that were duplicated across
``Base_NPE/``, ``SetTransformer_NPE/`` and ``Plain_NPE/``:

    ============================================  ==========================
    original                                      new home
    ============================================  ==========================
    ``<model>/z-score_violin.py``                 :mod:`.diagnostics` + :mod:`.figures`
    ``Base_NPE/z-score.py``                       :mod:`.diagnostics` + :mod:`.figures`
    ``SetTransformer_NPE/kde_prior_vs_posterior.py``  :mod:`.figures`
    ``<model>/ppc-plot.py``                       :mod:`.figures`
    ``Base_NPE/test.ipynb`` (SBC)                 :mod:`.diagnostics` + :mod:`.figures`
    ``Base_NPE/PriorPredictiveCheck.ipynb``       :mod:`.figures`
    the repeated "Elsevier" rcParams block        :mod:`.style`
    ============================================  ==========================

The split is deliberate: :mod:`.diagnostics` computes and returns, :mod:`.figures`
draws and writes, :mod:`.style` decides how things look. Loading a checkpoint and
drawing posterior samples lives in :mod:`cancer_sbi.evaluation.posterior`.

Importing this package has no side effects. In particular it does not touch
``matplotlib.rcParams`` and does not select a backend; call
:func:`cancer_sbi.evaluation.style.use_paper_style` and, on a headless machine,
:func:`cancer_sbi.evaluation.style.use_agg_backend` explicitly.

One number changed on purpose during the port: the prior standard deviation used
for the reference curve in the four ``kde_*`` figures. See
:func:`cancer_sbi.evaluation.figures.plot_kde_pooled_only`.
"""

from cancer_sbi.evaluation import diagnostics, figures, style

__all__ = ["diagnostics", "figures", "style"]
