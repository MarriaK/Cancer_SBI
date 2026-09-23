"""Posterior diagnostics: z-scores, L2, correlations, calibration, SBC.

Pure computation. Nothing here plots, prints or writes a file -- every function
takes arrays and returns arrays, a namedtuple or a :class:`pandas.DataFrame`.
The figures that consume these numbers live in :mod:`cancer_sbi.evaluation.figures`.

Originals:
    * ``Base_NPE/z-score_violin.py`` lines 140-177 (per-case posterior summary),
      227-250 (pooled and per-parameter z summaries), 385-399 and 473-479
      (calibration classification and counts), 705-722 (correlation table).
    * ``Base_NPE/z-score.py`` lines 98-135 -- the earlier, smaller version of the
      same loop; identical maths, so only one implementation is kept.
    * ``SetTransformer_NPE/kde_prior_vs_posterior.py`` lines 142-154 (selection
      of five diverse test cases).
    * ``Base_NPE/test.ipynb`` cells 8 and 28 -- ``run_sbc`` / ``check_sbc``. This
      is the only SBC analysis in the repository and existed nowhere as a ``.py``
      file before this module.
"""

from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

from cancer_sbi.arms import ARM_LABELS

__all__ = [
    "POST_STD_EPS",
    "BIAS_THRESHOLD",
    "OVERCONFIDENCE_THRESHOLD",
    "CasePosteriorSummary",
    "StackedCaseSummaries",
    "PooledZSummary",
    "PerParameterZSummary",
    "CalibrationCounts",
    "summarise_case",
    "stack_case_summaries",
    "pooled_z_summary",
    "per_parameter_z_summary",
    "correlation_table",
    "classify_calibration",
    "calibration_counts",
    "select_diverse_test_cases",
    "run_sbc_ranks",
    "check_sbc_stats",
]


#: Added to the posterior standard deviation before dividing, so that a
#: degenerate (zero-width) marginal cannot produce an infinite z-score.
#: Preserved from ``Base_NPE/z-score_violin.py:141,161``: the epsilon is inside
#: the published numbers, so it is part of the contract, not a detail.
POST_STD_EPS: float = 1e-8

#: An arm counts as biased when |mean z| exceeds this
#: (``Base_NPE/z-score_violin.py:389``).
BIAS_THRESHOLD: float = 1.0

#: An arm counts as overconfident when std(z) exceeds this
#: (``Base_NPE/z-score_violin.py:390``).
OVERCONFIDENCE_THRESHOLD: float = 1.5


class CasePosteriorSummary(NamedTuple):
    """Per-test-case summary of one posterior.

    Attributes:
        post_mean: Posterior mean, shape (D,), float32.
        post_std: Posterior standard deviation plus :data:`POST_STD_EPS`,
            shape (D,), float32.
        z: Standardised error ``(theta_true - post_mean) / post_std``,
            shape (D,), float32.
        l2: Euclidean distance ``||theta_true - post_mean||_2``, a Python float.
    """

    post_mean: torch.Tensor
    post_std: torch.Tensor
    z: torch.Tensor
    l2: float


class StackedCaseSummaries(NamedTuple):
    """The whole test set's summaries stacked into matrices.

    Attributes:
        z: Shape (N, D) -- one row of z-scores per test case.
        post_mean: Shape (N, D) -- one posterior mean per test case.
        l2: Shape (N,) float64 -- one L2 distance per test case.
        mean_abs_z_per_case: Shape (N,) float64 -- ``z.abs().mean()`` per case.
    """

    z: torch.Tensor
    post_mean: torch.Tensor
    l2: np.ndarray
    mean_abs_z_per_case: np.ndarray


class PooledZSummary(NamedTuple):
    """Global z-score summary, pooled over cases *and* parameters.

    Attributes:
        mean_z: Mean of all N*D z-scores.
        std_z: Population standard deviation (ddof=0) of all N*D z-scores.
        pct_within_2: Percentage (0-100) of z-scores with |z| <= 2.
    """

    mean_z: float
    std_z: float
    pct_within_2: float


class PerParameterZSummary(NamedTuple):
    """Per-chromosome-arm z-score summary.

    Attributes:
        mean_abs_z: Shape (D,) -- mean |z| per arm, the calibration error.
        mean_z: Shape (D,) -- mean z per arm, the bias.
        std_z: Shape (D,) -- population std (ddof=0) of z per arm, the width.
    """

    mean_abs_z: np.ndarray
    mean_z: np.ndarray
    std_z: np.ndarray


class CalibrationCounts(NamedTuple):
    """How many arms fall into each calibration class.

    Attributes:
        n_well_calibrated: ``D - n_biased - n_overconfident`` (see the warning in
            :func:`calibration_counts`).
        n_biased: Number of arms with |mean z| > the bias threshold.
        n_overconfident: Number of arms with std z > the overconfidence threshold.
    """

    n_well_calibrated: int
    n_biased: int
    n_overconfident: int


def summarise_case(
    samples: torch.Tensor,
    theta_true: torch.Tensor,
    eps: float = POST_STD_EPS,
) -> CasePosteriorSummary:
    """Reduce one case's posterior samples to mean, std, z-scores and L2.

    Preserved from ``Base_NPE/z-score_violin.py:156-167``. Two details are
    load-bearing and must not be "improved":

    * ``unbiased=False`` -- the population (1/N) standard deviation, not the
      sample (1/(N-1)) one.
    * ``+ eps`` is added to the standard deviation *before* the division, so
      every z-score is very slightly shrunk toward zero.

    Args:
        samples: Posterior samples, shape (S, D) or (S, 1, D). The middle
            singleton dimension that ``DirectPosterior.sample`` sometimes emits
            is squeezed away, exactly as the original did.
        theta_true: Ground-truth parameters for this case, shape (D,).
        eps: Epsilon added to the posterior standard deviation.

    Returns:
        A :class:`CasePosteriorSummary`; all tensors live on the same device as
        ``samples``.
    """
    if samples.dim() == 3:  # e.g. (S, 1, D) -> (S, D)
        samples = samples.squeeze(1)

    post_mean = samples.mean(dim=0)
    post_std = samples.std(dim=0, unbiased=False) + eps

    l2 = torch.norm(theta_true - post_mean, p=2).item()
    z = (theta_true - post_mean) / post_std

    return CasePosteriorSummary(post_mean=post_mean, post_std=post_std, z=z, l2=l2)


def stack_case_summaries(
    summaries: Iterable[CasePosteriorSummary],
) -> StackedCaseSummaries:
    """Stack per-case summaries into the matrices the figures expect.

    Preserved from ``Base_NPE/z-score_violin.py:166-175``: the per-case tensors
    are moved to CPU before stacking, so the result is always CPU-resident.

    Args:
        summaries: Per-case summaries, in test-set order.

    Returns:
        A :class:`StackedCaseSummaries` with z and post_mean of shape (N, D).
    """
    summaries = list(summaries)
    z = torch.stack([s.z.detach().cpu() for s in summaries], dim=0)
    post_mean = torch.stack([s.post_mean.detach().cpu() for s in summaries], dim=0)
    l2 = np.asarray([s.l2 for s in summaries], dtype=np.float64)
    mean_abs_z = np.asarray(
        [s.z.detach().abs().mean().item() for s in summaries], dtype=np.float64
    )
    return StackedCaseSummaries(
        z=z, post_mean=post_mean, l2=l2, mean_abs_z_per_case=mean_abs_z
    )


def pooled_z_summary(z_all: torch.Tensor | np.ndarray) -> PooledZSummary:
    """Summarise every z-score in the test set as one pooled sample.

    Preserved from ``Base_NPE/z-score_violin.py:224-229``. ``np.std`` defaults to
    ddof=0, which is what the original relied on; it is spelled out here so a
    reader does not have to know the default.

    Args:
        z_all: z-scores, shape (N, D). Torch tensors are detached and moved to
            CPU first.

    Returns:
        A :class:`PooledZSummary`.
    """
    z_pooled = _to_numpy(z_all).reshape(-1)
    return PooledZSummary(
        mean_z=float(np.mean(z_pooled)),
        std_z=float(np.std(z_pooled, ddof=0)),
        pct_within_2=float(np.mean(np.abs(z_pooled) <= 2.0) * 100.0),
    )


def per_parameter_z_summary(z_all: torch.Tensor | np.ndarray) -> PerParameterZSummary:
    """Summarise the z-scores arm by arm.

    Preserved from ``Base_NPE/z-score_violin.py:248-250``. The standard
    deviation is the population one (``unbiased=False`` / ``ddof=0``), matching
    both the bar chart and the violin colour-coding at ``:387``.

    Args:
        z_all: z-scores, shape (N, D).

    Returns:
        A :class:`PerParameterZSummary`, each field shape (D,).
    """
    z_np = _to_numpy(z_all)
    return PerParameterZSummary(
        mean_abs_z=np.abs(z_np).mean(axis=0),
        mean_z=z_np.mean(axis=0),
        std_z=z_np.std(axis=0, ddof=0),
    )


def correlation_table(
    theta_all: torch.Tensor | np.ndarray,
    post_mean_all: torch.Tensor | np.ndarray,
    labels: Sequence[str] = ARM_LABELS,
    decimals: int = 6,
) -> pd.DataFrame:
    """Per-arm Pearson, Spearman and Kendall correlation of truth vs posterior mean.

    Preserved from ``Base_NPE/z-score_violin.py:705-724``, including the rounding
    to 6 decimal places, which happens *before* the values reach the spreadsheet
    and is therefore part of the artefact rather than a display choice.

    Args:
        theta_all: True parameters, shape (N, D).
        post_mean_all: Posterior means, shape (N, D).
        labels: Arm names, length D.
        decimals: Decimal places to round every statistic and p-value to.

    Returns:
        A DataFrame with one row per arm and columns ``Chromosome_Arm``,
        ``Pearson_r``, ``Pearson_p``, ``Spearman_r``, ``Spearman_p``,
        ``Kendall_tau``, ``Kendall_p`` -- in that order, which is the column
        order of the published spreadsheet.
    """
    theta_np = _to_numpy(theta_all)
    mean_np = _to_numpy(post_mean_all)

    rows: list[dict[str, Any]] = []
    for j, arm in enumerate(labels):
        true_j = theta_np[:, j]
        pred_j = mean_np[:, j]

        pearson_r, pearson_p = scipy_stats.pearsonr(true_j, pred_j)
        spearman_r, spearman_p = scipy_stats.spearmanr(true_j, pred_j)
        kendall_tau, kendall_p = scipy_stats.kendalltau(true_j, pred_j)

        rows.append(
            {
                "Chromosome_Arm": arm,
                "Pearson_r": round(float(pearson_r), decimals),
                "Pearson_p": round(float(pearson_p), decimals),
                "Spearman_r": round(float(spearman_r), decimals),
                "Spearman_p": round(float(spearman_p), decimals),
                "Kendall_tau": round(float(kendall_tau), decimals),
                "Kendall_p": round(float(kendall_p), decimals),
            }
        )

    return pd.DataFrame(rows)


def classify_calibration(
    mean_z: np.ndarray,
    std_z: np.ndarray,
    bias_threshold: float = BIAS_THRESHOLD,
    overconfidence_threshold: float = OVERCONFIDENCE_THRESHOLD,
) -> list[str]:
    """Label each arm "overconfident", "biased" or "well_calibrated".

    Preserved from ``Base_NPE/z-score_violin.py:392-399``. The test order
    matters: overconfidence is checked *first*, so an arm that is both biased and
    overconfident is labelled overconfident and drawn orange. Reordering the
    branches would recolour violins in the published figure.

    Args:
        mean_z: Per-arm mean z, shape (D,).
        std_z: Per-arm std z, shape (D,).
        bias_threshold: |mean z| above which an arm is biased.
        overconfidence_threshold: std z above which an arm is overconfident.

    Returns:
        A list of D class names, one per arm.
    """
    classes: list[str] = []
    for j in range(len(mean_z)):
        if std_z[j] > overconfidence_threshold:
            classes.append("overconfident")
        elif abs(mean_z[j]) > bias_threshold:
            classes.append("biased")
        else:
            classes.append("well_calibrated")
    return classes


def calibration_counts(
    mean_z: np.ndarray,
    std_z: np.ndarray,
    bias_threshold: float = BIAS_THRESHOLD,
    overconfidence_threshold: float = OVERCONFIDENCE_THRESHOLD,
) -> CalibrationCounts:
    """Count arms per calibration class, the way the original printout did.

    Preserved from ``Base_NPE/z-score_violin.py:473-475``, quirk included: the
    two counts are computed independently, so an arm that is both biased and
    overconfident is counted twice and ``n_well_calibrated`` is correspondingly
    too small -- it can even go negative. This does not agree with
    :func:`classify_calibration`, which assigns each arm exactly one colour. Both
    behaviours are reproduced as they stand because the printed summary was read
    off the console into the manuscript's prose.

    Args:
        mean_z: Per-arm mean z, shape (D,).
        std_z: Per-arm std z, shape (D,).
        bias_threshold: |mean z| above which an arm is biased.
        overconfidence_threshold: std z above which an arm is overconfident.

    Returns:
        A :class:`CalibrationCounts`.
    """
    n_biased = int(np.sum(np.abs(mean_z) > bias_threshold))
    n_overconfident = int(np.sum(std_z > overconfidence_threshold))
    n_good = len(mean_z) - n_biased - n_overconfident
    return CalibrationCounts(
        n_well_calibrated=n_good,
        n_biased=n_biased,
        n_overconfident=n_overconfident,
    )


def select_diverse_test_cases(
    theta_all: torch.Tensor | np.ndarray,
    n_selected: int = 5,
) -> list[int]:
    """Pick evenly spaced test cases across the range of mean true theta.

    Preserved from ``SetTransformer_NPE/kde_prior_vs_posterior.py:145-150``. The
    cases are ranked by their mean theta across all arms and then sampled at
    ``i * (N // n_selected)``. Note that integer floor division means the last
    selected case sits well short of the top of the ranking (for N=152, n=5 the
    indices are 0, 30, 60, 90, 120 of 151), so the "highest theta" region of the
    prior is not represented. Preserved: these five cases are the coloured curves
    in the published KDE figures.

    Args:
        theta_all: True parameters, shape (N, D).
        n_selected: How many cases to pick.

    Returns:
        ``n_selected`` indices into the test set, in ascending order of mean theta.
    """
    theta_np = _to_numpy(theta_all)
    theta_mean_per_case = theta_np.mean(axis=1)  # (N,)
    sorted_indices = np.argsort(theta_mean_per_case)
    step = len(sorted_indices) // n_selected
    return [int(sorted_indices[i * step]) for i in range(n_selected)]


def run_sbc_ranks(
    theta: torch.Tensor,
    x: torch.Tensor,
    posterior: Any,
    num_posterior_samples: int = 1_000,
    num_workers: int = 1,
    use_sample_batched: bool = False,
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run simulation-based calibration over the test set.

    Ported from ``Base_NPE/test.ipynb`` cell 8, which is the only place in the
    repository where SBC is run. ``use_sample_batched=False`` is the notebook's
    value and is kept as the default: the notebook's own comment says batching
    "can give a speed-up, but might cause memory issues".

    ``sbi`` is imported inside the function so that the rest of this module stays
    importable on a machine without it.

    Args:
        theta: True parameters, shape (N, D).
        x: Observations, shape (N, ...) -- whatever the encoder consumes.
        posterior: A built ``DirectPosterior``.
        num_posterior_samples: Posterior draws per test case.
        num_workers: Worker processes for sbi.
        use_sample_batched: Passed straight through to ``run_sbc``.
        device: If given, ``theta`` and ``x`` are moved there first. The notebook
            hard-coded ``.to('cuda')``; this is an argument so the same code runs
            on a CPU box, and ``None`` (the default) moves nothing.

    Returns:
        ``(ranks, dap_samples)``: ranks of shape (N, D) and data-averaged
        posterior samples of shape (N, D).
    """
    from sbi.diagnostics import run_sbc  # local import: sbi is optional here

    if device is not None:
        theta = theta.to(device)
        x = x.to(device)

    ranks, dap_samples = run_sbc(
        theta,
        x,
        posterior,
        num_posterior_samples=num_posterior_samples,
        num_workers=num_workers,
        use_sample_batched=use_sample_batched,
    )
    return ranks, dap_samples


def check_sbc_stats(
    ranks: torch.Tensor,
    theta: torch.Tensor,
    dap_samples: torch.Tensor,
    num_posterior_samples: int = 1_000,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Turn SBC ranks into KS p-values and c2st accuracies.

    Ported from ``Base_NPE/test.ipynb`` cell 28.

    Args:
        ranks: SBC ranks from :func:`run_sbc_ranks`, shape (N, D).
        theta: True parameters, shape (N, D).
        dap_samples: Data-averaged posterior samples, shape (N, D).
        num_posterior_samples: The same value passed to :func:`run_sbc_ranks`.
        device: If given, ``theta`` is moved there first (the notebook used
            ``.to('cuda')``); ``None`` moves nothing.

    Returns:
        The dict sbi returns, with at least the keys ``ks_pvals``,
        ``c2st_ranks`` and ``c2st_dap``, each a tensor of shape (D,).
    """
    from sbi.diagnostics import check_sbc  # local import: sbi is optional here

    if device is not None:
        theta = theta.to(device)

    return check_sbc(
        ranks,
        theta,
        dap_samples,
        num_posterior_samples=num_posterior_samples,
    )


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a detached CPU numpy view of a torch tensor, or the array as-is.

    Args:
        array: A torch tensor or anything numpy can wrap.

    Returns:
        A numpy array.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)
