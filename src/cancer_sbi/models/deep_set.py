"""DeepSet embedding net: the DominantClone-NPE encoder.

Ported from ``Plain_NPE/net_builder.py:5-98``.

Unlike the other two models there is no separate trial encoder here: this single
module *is* the embedding net handed to ``build_nsf``. It maps one sim's
``(T, 44)`` matrix of dominant-clone profiles to a single vector, masking the
NaN-padded trials and appending the trial count as an extra feature.

``Plain_NPE/NN_Utils.py`` constructs a ``DeepSet`` at import time. That is a
module-level side effect and is deliberately **not** reproduced anywhere in this
package; construct one explicitly instead.
"""

from typing import Optional

import torch
from torch import Tensor, nn


class DeepSet(nn.Module):
    """Permutation-invariant encoder over a sim's trials.

    Ported from ``Plain_NPE/net_builder.py:5-98``.

    Similar to ``sbi.neural_nets.embedding_nets.PermutationInvariantEmbedding``,
    with two differences kept from the original: self-attention is available as
    an aggregation function, and phi/rho are built here rather than supplied.
    """

    def __init__(
        self,
        aggregation_fn: Optional[str] = "mean",
        aggregation_dim: int = 1,
        input_dim: int = 44,
        hidden_dim_phi: int = 100,
        hidden_dim_rho: int = 100,
        num_layers: int = 2,
        output_dim: int = 20,
        num_heads: int = 4,
    ) -> None:
        """Build phi, the optional attention head and rho.

        Args:
            aggregation_fn: ``"mean"``, ``"sum"`` or ``"attention"``. The
                published model uses ``"mean"``.
            aggregation_dim: Dimension holding the trials. 1 for ``(B, T, F)``.
            input_dim: Feature width per trial, 44 here.
            hidden_dim_phi: Hidden width of phi. ``Plain_NPE/model.py:18`` passes
                44, not the default 100.
            hidden_dim_rho: Hidden width of rho. ``Plain_NPE/model.py:19`` passes
                44, not the default 100.
            num_layers: Depth of phi (and, minus two, of rho's hidden stack).
                With ``num_layers=2`` rho is a single
                ``Linear(hidden_dim_phi + 1 -> hidden_dim_rho) + ReLU`` followed
                by ``Linear(hidden_dim_rho -> output_dim)``.
            output_dim: Width of the context vector. ``Plain_NPE/model.py:20``
                passes 128, not the default 20.
            num_heads: Heads for the attention aggregation. Unused for ``"mean"``.

        Raises:
            AssertionError: If ``aggregation_fn`` is not one of the three names.
        """
        super().__init__()

        assert aggregation_fn in [
            "mean",
            "sum",
            "attention",
        ], "aggregation_fn must be 'mean', 'sum', or 'attention'."
        self.aggregation_fn = aggregation_fn
        self.aggregation_dim = aggregation_dim

        # Single-trial feature extractor.
        phi_layers = [nn.Linear(input_dim, hidden_dim_phi), nn.ReLU()]
        for _ in range(num_layers - 1):
            phi_layers.append(nn.Linear(hidden_dim_phi, hidden_dim_phi))
            phi_layers.append(nn.ReLU())
        self.phi = nn.Sequential(*phi_layers)

        if self.aggregation_fn == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim_phi, num_heads=num_heads, batch_first=True
            )
        else:
            self.attention = None

        # Output function. The "+ 1" makes room for the trial count that
        # forward() concatenates onto the pooled embedding.
        rho_layers = [nn.Linear(hidden_dim_phi + 1, hidden_dim_rho), nn.ReLU()]
        for _ in range(num_layers - 2):
            rho_layers.append(nn.Linear(hidden_dim_rho, hidden_dim_rho))
            rho_layers.append(nn.ReLU())
        rho_layers.append(nn.Linear(hidden_dim_rho, output_dim))
        self.rho = nn.Sequential(*rho_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Pool a batch of trial matrices into context vectors.

        Args:
            x: ``(batch_size, num_trials, input_dim)`` float32. A trial that was
                missing on disk is an all-NaN row (see
                :class:`~cancer_sbi.data.dominant_clone.SimulationDataset`).

        Returns:
            ``(batch_size, output_dim)`` float32 context vectors.

        Raises:
            AssertionError: If the pooled embedding still contains NaNs, which
                happens when a sim has zero valid trials.
        """
        num_batch, max_num_trials = x.shape[0], x.shape[self.aggregation_dim]

        # Preserved from Plain_NPE/net_builder.py:66-71. Trap 7: after
        # .sum(dim=1) the counts have shape (batch, features); .reshape(-1)
        # flattens them and [:num_batch] then takes the FIRST `num_batch`
        # entries, which are batch-element 0's per-feature counts -- not one
        # count per batch element. The comment "counts are the same across data
        # dims" is true within a sim (a missing trial is NaN in all 44 columns)
        # but not across sims, so every sim in the batch is told it has sim 0's
        # trial count. This is live and affects DominantClone's results. This
        # looks wrong but it is what the published model does; fixing the
        # indexing changes the results. See docs/REFACTOR_NOTES.md.
        nan_counts = (
            torch.isnan(x)
            .sum(dim=self.aggregation_dim)  # count nans over trial dimension
            .reshape(-1)[:num_batch]  # counts are the same across data dims
            .unsqueeze(-1)  # make it (batch, 1) to match embeddings below
        )
        # Number of non-nan trials.
        trial_counts = max_num_trials - nan_counts

        is_nan = torch.isnan(x)
        # Apply the trial net with NaN entries replaced by 0, then zero the
        # embeddings of the trials that were NaN so they contribute nothing.
        masked_x = torch.nan_to_num(x, nan=0.0)
        trial_embeddings = self.phi(masked_x)
        trial_embeddings = trial_embeddings * (~is_nan.all(-1, keepdim=True)).float()

        # Take the mean over the permutation dimension by dividing by the number
        # of trials (instead of torch.mean) to account for masking.
        if self.aggregation_fn == "mean":
            combined_embedding = trial_embeddings.sum(dim=self.aggregation_dim) / trial_counts
        elif self.aggregation_fn == "sum":
            combined_embedding = trial_embeddings.sum(dim=self.aggregation_dim)
        else:
            attn_output, _ = self.attention(
                trial_embeddings, trial_embeddings, trial_embeddings
            )  # Self-attention
            combined_embedding = attn_output.mean(dim=self.aggregation_dim)

        assert not torch.isnan(combined_embedding).any(), "NaNs in embedding."

        # Add the number of trials as an additional input.
        return self.rho(torch.cat([combined_embedding, trial_counts], dim=1))


__all__ = ["DeepSet"]
