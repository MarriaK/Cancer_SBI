"""The 44 chromosome-arm labels, defined once.

Replaces the hard-coded 44-element ``labels = [...]`` literal that appears in at
least ten places in the original repo, among them
``Base_NPE/z-score_violin.py:182-205``, ``SetTransformer_NPE/z-score_violin.py``,
``Plain_NPE/z-score_violin.py``, ``*/z-score.py``, ``*/ppc-plot.py``,
``SetTransformer_NPE/kde_prior_vs_posterior.py`` and the ``test.ipynb`` notebooks.
Every one of those copies had the same order, which is the order the simulator
writes theta in, so a single shared tuple is safe.

Two spellings of the same 44 arms exist in this repo and both are correct in
their own context:

* **Prefixed** -- ``"chr1p" ... "chr22q"``, defined here. This is what the
  figures, the z-score summaries and ``baselines/summary_stat_baseline.csv`` use,
  and it is the order of the 44 entries of ``theta``.
* **Un-prefixed** -- ``"1p" ... "22q"``, produced by
  ``utilities/chromosome_arms.py::bin_to_arm`` (see its ``return`` at
  ``utilities/chromosome_arms.py:31``), which maps a genomic bin to an arm while
  preprocessing real CHISEL/HMMcopy data.

They are NOT interchangeable: strip/add the ``"chr"`` prefix explicitly when
crossing between the two worlds. ``strip_chr_prefix`` below does that for the
whole table.
"""

from typing import Tuple

#: The 44 chromosome-arm labels in the fixed order used by ``theta``.
#: Index ``i`` of this tuple names element ``i`` of every 44-vector in the
#: package (true theta, posterior mean, z-score, per-arm correlation).
ARM_LABELS: Tuple[str, ...] = (
    "chr1p", "chr1q",
    "chr2p", "chr2q",
    "chr3p", "chr3q",
    "chr4p", "chr4q",
    "chr5p", "chr5q",
    "chr6p", "chr6q",
    "chr7p", "chr7q",
    "chr8p", "chr8q",
    "chr9p", "chr9q",
    "chr10p", "chr10q",
    "chr11p", "chr11q",
    "chr12p", "chr12q",
    "chr13p", "chr13q",
    "chr14p", "chr14q",
    "chr15p", "chr15q",
    "chr16p", "chr16q",
    "chr17p", "chr17q",
    "chr18p", "chr18q",
    "chr19p", "chr19q",
    "chr20p", "chr20q",
    "chr21p", "chr21q",
    "chr22p", "chr22q",
)

#: Number of parameters inferred per simulation. Equal to ``len(ARM_LABELS)``.
NUM_ARMS: int = len(ARM_LABELS)

#: Index of the first arm of chromosome 12, i.e. where the figures cut the 44
#: arms into two panels of 22. Matches ``list(range(0, 22))`` /
#: ``list(range(22, 44))`` at ``Base_NPE/z-score_violin.py:404,409`` (violins) and
#: ``:545,550`` (scatter grids).
HALF_SPLIT_INDEX: int = 22


def arm_label_halves() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Split the 44 arm labels into the two 22-arm halves the figures use.

    The violin and scatter figures are drawn as two panels, "Chromosomes 1-11"
    (``chr1p`` .. ``chr11q``) and "Chromosomes 12-22" (``chr12p`` .. ``chr22q``).
    Both halves happen to hold exactly 22 arms because chromosomes 1-11 and 12-22
    are eleven chromosomes each.

    Returns:
        A 2-tuple ``(first_half, second_half)``; ``first_half`` is
        ``ARM_LABELS[:22]`` and ``second_half`` is ``ARM_LABELS[22:]``. Element
        ``j`` of ``second_half`` corresponds to theta index ``22 + j``.
    """
    return ARM_LABELS[:HALF_SPLIT_INDEX], ARM_LABELS[HALF_SPLIT_INDEX:]


def arm_index_halves() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return the theta column indices behind each figure panel.

    Returns:
        A 2-tuple ``(range(0, 22), range(22, 44))`` materialised as tuples of
        ``int``, so callers can index a ``(N, 44)`` array column-wise exactly as
        ``Base_NPE/z-score_violin.py:404,409`` did.
    """
    return (
        tuple(range(0, HALF_SPLIT_INDEX)),
        tuple(range(HALF_SPLIT_INDEX, NUM_ARMS)),
    )


def strip_chr_prefix(labels: Tuple[str, ...] = ARM_LABELS) -> Tuple[str, ...]:
    """Convert prefixed labels to the un-prefixed form used for real data.

    ``utilities/chromosome_arms.py`` emits ``"1p"``-style labels when it maps
    genomic bins to arms; this package's theta vectors use ``"chr1p"``. Use this
    when joining a real-data table against ``ARM_LABELS``.

    Args:
        labels: Prefixed labels, defaults to the full :data:`ARM_LABELS` table.

    Returns:
        The same labels with a leading ``"chr"`` removed, e.g. ``("1p", "1q", ...)``.
    """
    return tuple(label[3:] if label.startswith("chr") else label for label in labels)
