"""The neural spline flow (NSF), with every architecture argument pinned.

Replaces the single ``build_nsf(...)`` call that appears three times in the
originals:

* ``Base_NPE/inference_model.py:91-99``
* ``SetTransformer_NPE/inference_model.py:87``
* ``Plain_NPE/model.py:64``

All three relied on sbi's *defaults* for the flow architecture -- they passed
only ``z_score_x``, ``z_score_y``, ``exclude_invalid_y``, ``embedding_net`` and
``dropout_probability``. That makes the published architecture a function of
whichever sbi version happens to be installed. :func:`build_flow` writes the
seven architecture defaults out explicitly so it no longer is.

**Writing them out is a documentation change, not a behaviour change.** The
values in :class:`~cancer_sbi.config.FlowConfig` are exactly sbi's own defaults:
``hidden_features=50, num_transforms=5, num_bins=10, num_blocks=2,
tail_bound=3.0, hidden_layers_spline_context=1, use_batch_norm=False``. Verified
identical in sbi 0.23.3 and sbi 0.25.0, and re-checked here against the
installed sbi 0.25.0 at
``site-packages/sbi/neural_nets/net_builders/flow.py:306-324``. A model built
with this function under either version is the same model the originals built.

Nothing here runs at import time.
"""

from typing import Any

from sbi.neural_nets.net_builders import build_nsf
from torch import Tensor, nn

from cancer_sbi.config import FlowConfig


def build_flow(
    theta_batch: Tensor,
    condition_batch: Tensor,
    embedding_net: nn.Module,
    cfg: FlowConfig,
) -> Any:
    """Build the conditional NSF density estimator ``p(theta | x)``.

    sbi infers the flow's input dimensionality, and the shapes for any
    z-scoring, from the two example batches, so both must come from the real
    training loader -- the originals used ``next(iter(train_loader))`` for
    exactly this.

    Args:
        theta_batch: ``(B, 44)`` float32 batch of parameter vectors. Passed as
            sbi's ``batch_x``, the variable the flow models.
        condition_batch: The raw conditioning batch, passed as sbi's ``batch_y``
            *before* the embedding net runs: ``(B, T, K, 45)`` float32 for the
            clone-set models, ``(B, T, 44)`` float32 for DominantClone.
        embedding_net: Module mapping ``condition_batch`` to a context vector --
            :class:`~cancer_sbi.models.trials.TrialsSBIEmbedding` for CloneMLP
            and CloneAtt, :class:`~cancer_sbi.models.deep_set.DeepSet` for
            DominantClone. Its parameters become part of the returned estimator,
            which is why the two-group optimiser (trap 2) has to separate them
            out again by ``id()``.
        cfg: Flow settings, normally ``preset.flow``.

    Returns:
        The sbi density estimator (an ``NFlowsFlow``). It is returned on
        whatever device its inputs implied; the caller does the ``.to(device)``,
        as all three originals did.

    Note:
        ``exclude_invalid_y`` is forwarded because all three originals passed it,
        but ``build_nsf`` has no such parameter: it is absorbed by ``**kwargs``
        and ignored (``sbi/neural_nets/net_builders/flow.py:324``). It is kept so
        the call site still reads like the original and so nothing appears to
        have been silently removed.
    """
    return build_nsf(
        theta_batch,
        condition_batch,
        embedding_net=embedding_net,
        # Preserved from Base_NPE/inference_model.py:95-96 and
        # SetTransformer_NPE/inference_model.py:87 ("none") versus
        # Plain_NPE/model.py:64 ("structured"). Trap 1: only DominantClone
        # whitens theta and the context. This looks wrong to unify but it is
        # what the published models do; changing it changes the results.
        # See docs/REFACTOR_NOTES.md.
        z_score_x=cfg.z_score_x,
        z_score_y=cfg.z_score_y,
        exclude_invalid_y=cfg.exclude_invalid_y,
        # Passed by all three originals; 0.2 via */main.py:35.
        dropout_probability=cfg.dropout_probability,
        # --- sbi's own defaults, pinned so the architecture stops depending on
        # --- the installed sbi version. Same values in 0.23.3 and 0.25.0.
        hidden_features=cfg.hidden_features,
        num_transforms=cfg.num_transforms,
        num_bins=cfg.num_bins,
        num_blocks=cfg.num_blocks,
        tail_bound=cfg.tail_bound,
        hidden_layers_spline_context=cfg.hidden_layers_spline_context,
        use_batch_norm=cfg.use_batch_norm,
    )


__all__ = ["build_flow"]
