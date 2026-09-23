"""Load a trained model, build the posterior, draw samples.

Ported from the block that opens every evaluation script -- the canonical copy
is ``Base_NPE/z-score_violin.py:95-175``, repeated near-verbatim in
``Base_NPE/z-score.py``, ``SetTransformer_NPE/z-score_violin.py``,
``SetTransformer_NPE/z-score.py``, ``SetTransformer_NPE/kde_prior_vs_posterior.py``,
``Plain_NPE/z-score_violin.py`` and the two ``ppc-plot.py`` files.

The sequence is always the same:

1. rebuild the model exactly as training built it (see
   :func:`cancer_sbi.training.trainer.build_training_components`);
2. load ``best.pt`` into it, after stripping the RNG state -- see
   :func:`load_checkpoint_for_eval`;
3. wrap it in ``DirectPosterior`` with a 44-dimensional independent normal
   prior;
4. for each test case, draw ``num_posterior_samples`` samples and reduce them to
   a posterior mean, a posterior standard deviation, z-scores and an L2 error.

Step 4's arithmetic lives in :mod:`cancer_sbi.evaluation.diagnostics`; this
module only orchestrates it, so that "how a number is computed" has exactly one
home.

**One value changed on purpose.** ``prior_sd`` defaults to ``0.2`` here. See
:func:`build_prior`.

Nothing here runs at import time.
"""

import math
import os
import tempfile
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from sbi.inference.posteriors import DirectPosterior
from torch.distributions import Independent, Normal
from torch.utils.data import DataLoader

from cancer_sbi.config import DatasetKind
from cancer_sbi.evaluation import diagnostics
from cancer_sbi.training import checkpoints

PathLike = Union[str, os.PathLike]

#: Number of chromosome-arm selection coefficients, i.e. ``len(ARM_LABELS)``.
NUM_PARAMETERS = 44

#: Posterior samples drawn per test case (``Base_NPE/z-score_violin.py:140``).
DEFAULT_NUM_POSTERIOR_SAMPLES = 5000

#: Standard deviation of the prior over selection coefficients. This is the
#: simulator's actual value.
DEFAULT_PRIOR_SD = 0.2

#: What the original evaluation scripts used: ``math.sqrt(0.2) = 0.4472...``.
#: Kept only so the old prior curve can be redrawn deliberately. Trap 20.
LEGACY_PRIOR_SD = math.sqrt(0.2)


class TestSetPosteriors(NamedTuple):
    """Everything :func:`summarise_test_set` produces.

    Attributes:
        stacked: The per-case summaries stacked into matrices --
            ``z`` and ``post_mean`` of shape ``(N, 44)``, ``l2`` and
            ``mean_abs_z_per_case`` of shape ``(N,)``.
        summaries: The per-case summaries themselves, in test-set order.
        samples: Raw posterior samples per case, shape ``(S, 44)`` each, only
            when ``keep_samples=True`` was passed; otherwise ``None``. Five
            thousand samples times 44 parameters times ~150 cases is about
            130 MB in float32, which is why they are not kept by default.
    """

    stacked: diagnostics.StackedCaseSummaries
    summaries: List[diagnostics.CasePosteriorSummary]
    samples: Optional[List[torch.Tensor]]


def build_prior(
    prior_sd: float = DEFAULT_PRIOR_SD,
    dim: int = NUM_PARAMETERS,
    device: str = "cpu",
    prior_mu: float = 0.0,
) -> Independent:
    """Build the independent normal prior over the 44 selection coefficients.

    Args:
        prior_sd: Standard deviation of each coordinate. See the note.
        dim: Number of coordinates; 44 in every published experiment.
        device: Device the prior's tensors live on. It must match the device the
            density estimator is on, because ``DirectPosterior`` mixes them.
        prior_mu: Mean of each coordinate; 0.0 in every published experiment.

    Returns:
        ``Independent(Normal(mu * ones(dim), sd * ones(dim)), 1)`` -- a single
        44-dimensional distribution rather than 44 independent ones, which is
        what ``reinterpreted_batch_ndims=1`` buys.

    Note:
        **Trap 20, and the one number this port does not reproduce.** All the
        original scripts wrote ``coeff_mu, coeff_std = 0.0, math.sqrt(0.2)``
        (``Base_NPE/z-score_violin.py:125``), i.e. a standard deviation of
        0.447, while the simulator drew the true coefficients with a standard
        deviation of 0.2. The comment beside it -- "coeff_var=0.2, std =
        sqrt(var)" -- shows where it came from: 0.2 was read as a variance.

        The wrong value is inert for every number in the paper. ``DirectPosterior``
        uses the prior only to reject samples outside its support, and a normal's
        support is all of R^44, so nothing is ever rejected; ``log_prob`` is never
        called by any of these scripts. It is *not* inert for the prior reference
        curve drawn in the four ``kde_*`` figures, which was therefore 2.24 times
        too wide.

        The default here is the correct 0.2. Pass :data:`LEGACY_PRIOR_SD` to
        redraw the old figures exactly. See docs/REFACTOR_NOTES.md.

        ``validate_args=False`` is preserved from the originals: it skips the
        support check on every ``sample``/``log_prob`` call.
    """
    return Independent(
        Normal(
            torch.full((dim,), float(prior_mu), dtype=torch.float32, device=device),
            torch.full((dim,), float(prior_sd), dtype=torch.float32, device=device),
            validate_args=False,
        ),
        reinterpreted_batch_ndims=1,
    )


def build_posterior(density_estimator: torch.nn.Module, prior: Independent) -> DirectPosterior:
    """Wrap a trained flow and a prior into an sbi posterior.

    Args:
        density_estimator: The trained flow, already in ``eval()`` mode.
        prior: From :func:`build_prior`, on the same device.

    Returns:
        A ``DirectPosterior`` whose ``.sample((n,), x=x)`` draws from
        ``p(theta | x)``.

    Note:
        Preserved from ``Base_NPE/z-score_violin.py:135``: the default
        ``DirectPosterior`` settings are used, which means samples outside the
        prior's support would be rejected and redrawn. With a normal prior that
        never happens, so sampling is a plain forward pass through the flow.
    """
    return DirectPosterior(density_estimator, prior)


def load_checkpoint_for_eval(
    ckpt_path: PathLike,
    density_estimator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    history_val_key: str = "validation_loss",
) -> checkpoints.ResumedState:
    """Load a training checkpoint for evaluation, stripping the RNG state.

    Args:
        ckpt_path: Usually ``<some checkpoint dir>/best.pt``.
        density_estimator: Freshly built model with the same architecture.
        optimizer: Freshly built optimiser with the same parameter groups. It is
            not used for evaluation, but the checkpoint carries its state and
            :func:`cancer_sbi.training.checkpoints.load_checkpoint` restores it.
        device: ``map_location``.
        history_val_key: Per-model history key; see trap 14.

    Returns:
        The :class:`~cancer_sbi.training.checkpoints.ResumedState`, whose
        ``best_val_loss`` is what the scripts printed after loading.

    Note:
        Preserved from ``Base_NPE/z-score_violin.py:106-114`` and the comment
        block at ``SetTransformer_NPE/kde_prior_vs_posterior.py:69-73``, which
        explains the problem: a checkpoint saved on the GPU holds a CUDA RNG
        tensor, and ``torch.set_rng_state`` raises a ``TypeError`` when that is
        restored on a CPU-only machine. The fix the author found was to drop the
        RNG keys and write the rest to a temporary file, then load that file
        through the normal path -- so the temporary file is reproduced here
        rather than replaced by an in-memory load, because the normal path is
        what tests the state dicts for strict compatibility.

        Both spellings of the CUDA key are popped: ``cuda_rng_state``, which the
        original scripts popped and which no checkpoint actually contains, and
        ``cuda_rng_state_all``, which is the key
        :func:`cancer_sbi.training.checkpoints.build_checkpoint` really writes.
        ``Plain_NPE/z-score_violin.py:100`` popped the second one, the other
        scripts the first. Popping both is a strict superset of either, and
        changes nothing: neither key is needed for evaluation.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt.pop("rng_state", None)
    ckpt.pop("cuda_rng_state", None)
    ckpt.pop("cuda_rng_state_all", None)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(ckpt, tmp_path)
        resumed = checkpoints.load_checkpoint(
            tmp_path,
            density_estimator,
            optimizer,
            device=device,
            history_val_key=history_val_key,
        )
    finally:
        # The original called os.remove unconditionally on the next line; the
        # try/finally only makes sure a failed load does not leave the file
        # behind. Same effect on the success path.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return resumed


def collect_test_tensors(
    loader: DataLoader,
    dataset: DatasetKind = "clone_sets",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate a whole loader into one context tensor and one theta tensor.

    Args:
        loader: The test loader, built with ``shuffle=False`` so the row order is
            the dataset's order.
        dataset: ``"clone_sets"`` or ``"dominant_clone"``; decides the batch
            layout.

    Returns:
        ``(x_all, theta_all)``. ``x_all`` is ``(N, T, K, 45)`` for the clone-set
        models and ``(N, T, 44)`` for DominantClone; ``theta_all`` is ``(N, 44)``.
        Both stay on the CPU, as in the originals -- they are indexed one row at
        a time and moved to the device inside the sampling loop.

    Note:
        Preserved from ``Base_NPE/z-score_violin.py:80-90`` and
        ``Plain_NPE/z-score_violin.py:73-82``: the dominant-clone version skips
        ``None`` batches, the clone-set version does not have to. The whole test
        set is materialised in memory, which is what makes the row index ``i``
        used throughout the figures meaningful.
    """
    all_x: List[torch.Tensor] = []
    all_theta: List[torch.Tensor] = []

    for batch in loader:
        if dataset == "dominant_clone":
            if batch is None:
                continue
            theta_batch, x_batch = batch
        else:
            x_batch, _, theta_batch = batch
        all_x.append(x_batch)
        all_theta.append(theta_batch)

    return torch.cat(all_x, dim=0), torch.cat(all_theta, dim=0)


@torch.no_grad()
def draw_posterior_samples(
    posterior: Any,
    x: torch.Tensor,
    num_posterior_samples: int = DEFAULT_NUM_POSTERIOR_SAMPLES,
) -> torch.Tensor:
    """Draw posterior samples for a single observation.

    Args:
        posterior: A ``DirectPosterior`` from :func:`build_posterior`.
        x: One observation **with its batch dimension kept**: ``(1, T, K, 45)``
            or ``(1, T, 44)``. Slicing as ``x_all[i:i+1]`` is what keeps it.
        num_posterior_samples: How many samples to draw. 5000 in every published
            figure.

    Returns:
        ``(num_posterior_samples, 44)`` float32 on the posterior's device.

    Note:
        Preserved from ``Base_NPE/z-score_violin.py:156-158``: sbi sometimes
        returns ``(S, 1, 44)`` and sometimes ``(S, 44)`` depending on version and
        on the shape of ``x``, so the middle singleton is squeezed away when it
        is there. ``.detach()`` is kept even though the whole call is already
        under ``no_grad``.
    """
    samples = posterior.sample((num_posterior_samples,), x=x).detach()
    if samples.dim() == 3:  # (S, 1, 44) -> (S, 44)
        samples = samples.squeeze(1)
    return samples


@torch.no_grad()
def evaluate_case(
    posterior: Any,
    x: torch.Tensor,
    theta_true: torch.Tensor,
    num_posterior_samples: int = DEFAULT_NUM_POSTERIOR_SAMPLES,
) -> Tuple[diagnostics.CasePosteriorSummary, torch.Tensor]:
    """Sample one test case and reduce it to the four published quantities.

    Args:
        posterior: A ``DirectPosterior``.
        x: One observation with its batch dimension, ``(1, ...)``.
        theta_true: That case's ground truth, ``(44,)``.
        num_posterior_samples: Samples to draw.

    Returns:
        ``(summary, samples)`` where ``summary`` carries ``post_mean``,
        ``post_std``, ``z`` and ``l2`` -- computed by
        :func:`cancer_sbi.evaluation.diagnostics.summarise_case`, so the epsilon
        and the population standard deviation are the published ones -- and
        ``samples`` is the ``(S, 44)`` draw they came from.
    """
    samples = draw_posterior_samples(posterior, x, num_posterior_samples)
    summary = diagnostics.summarise_case(samples, theta_true)
    return summary, samples


@torch.no_grad()
def summarise_test_set(
    posterior: Any,
    x_all: torch.Tensor,
    theta_all: torch.Tensor,
    num_posterior_samples: int = DEFAULT_NUM_POSTERIOR_SAMPLES,
    device: str = "cpu",
    progress_every: int = 50,
    keep_samples: bool = False,
    indices: Optional[Sequence[int]] = None,
) -> TestSetPosteriors:
    """Run the whole test set through the posterior, one case at a time.

    Args:
        posterior: A ``DirectPosterior``.
        x_all: All test contexts, ``(N, ...)``, CPU.
        theta_all: All ground truths, ``(N, 44)``, CPU.
        num_posterior_samples: Samples per case.
        device: Device each case is moved to before sampling.
        progress_every: Print a progress line every this many cases; ``0``
            silences it.
        keep_samples: Keep the raw samples of every case (memory-hungry; needed
            for the KDE figures).
        indices: Evaluate only these rows of ``x_all``. ``None`` means all of
            them, which is what every published figure used.

    Returns:
        A :class:`TestSetPosteriors`.

    Note:
        Preserved from ``Base_NPE/z-score_violin.py:150-171``: the cases are
        processed **one at a time**, not batched. That is slow (5000 samples per
        case, ~150 cases) and it is what the published numbers came from;
        batching would change the order in which the flow's RNG is consumed and
        therefore every individual sample.
    """
    row_indices = list(range(x_all.shape[0])) if indices is None else list(indices)
    total = len(row_indices)

    summaries: List[diagnostics.CasePosteriorSummary] = []
    samples_out: Optional[List[torch.Tensor]] = [] if keep_samples else None

    # `processed` counts from 0 and equals the row index in the default case, so
    # the progress line is the original's (Base_NPE/z-score_violin.py:170-171).
    for processed, i in enumerate(row_indices):
        # Keep the batch dimension: (1, T, K, 45) / (1, T, 44).
        x = x_all[i : i + 1].to(device)
        theta_true = theta_all[i].to(device)

        summary, samples = evaluate_case(posterior, x, theta_true, num_posterior_samples)
        summaries.append(summary)
        if samples_out is not None:
            samples_out.append(samples.cpu())

        if progress_every and processed % progress_every == 0:
            print(f"Processed {processed}/{total} test points...")

    return TestSetPosteriors(
        stacked=diagnostics.stack_case_summaries(summaries),
        summaries=summaries,
        samples=samples_out,
    )


__all__ = [
    "NUM_PARAMETERS",
    "DEFAULT_NUM_POSTERIOR_SAMPLES",
    "DEFAULT_PRIOR_SD",
    "LEGACY_PRIOR_SD",
    "TestSetPosteriors",
    "build_prior",
    "build_posterior",
    "load_checkpoint_for_eval",
    "collect_test_tensors",
    "draw_posterior_samples",
    "evaluate_case",
    "summarise_test_set",
]
