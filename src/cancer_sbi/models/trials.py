"""Trial-level permutation-invariant wrapper shared by CloneMLP and CloneAtt.

There were two copies of this class in the original repo:
``Base_NPE/MLP_encoder.py:94-166`` and
``SetTransformer_NPE/set_transformer.py:151-223``. They are identical except for
the type annotation on ``trial_encoder`` (``BaselineCloneEmbedding`` vs
``CloneSetEmbedding``) and two comments naming that class. Only one copy is
ported; the annotation below is the union of the two, which is what the code
always did at runtime -- it only ever reads ``trial_encoder.d_model`` and calls
it.

The wrapper turns a *nested* set -- trials of clones -- into one vector: it
embeds each trial with the given clone encoder, marks fully-NaN trials, and hands
the ``(B, T, d_model)`` stack to sbi's ``PermutationInvariantEmbedding`` to pool
over ``T``.

Nothing here runs at import time.
"""

from typing import Union

import torch
from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding
from torch import Tensor, nn

from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding
from cancer_sbi.models.set_transformer import CloneSetEmbedding

#: Either clone-set encoder can be wrapped; both expose ``d_model`` and map
#: ``(B, K, 45) -> (B, d_model)``.
CloneEncoder = Union[BaselineCloneEmbedding, CloneSetEmbedding]


class TrialsSBIEmbedding(nn.Module):
    """Pool per-trial clone embeddings into one context vector per sim.

    Replacement for the repo's earlier ``TrialsPermutationInvariantEmbed``, using
    sbi's ``PermutationInvariantEmbedding`` on top of a clone-set encoder.
    """

    def __init__(
        self,
        trial_encoder: CloneEncoder,
        aggregation_fn: str = "sum",
        num_hiddens: int = 256,
        num_layers: int = 2,
        output_dim: int = 256,
    ) -> None:
        """Wrap a clone-set encoder in sbi's permutation-invariant pooling.

        Args:
            trial_encoder: :class:`~cancer_sbi.models.mlp_encoder.BaselineCloneEmbedding`
                (CloneMLP) or
                :class:`~cancer_sbi.models.set_transformer.CloneSetEmbedding`
                (CloneAtt). Must expose ``d_model``.
            aggregation_fn: How sbi pools over trials. The default is ``"sum"``,
                but both published models pass ``"mean"``
                (``Base_NPE/inference_model.py:81``,
                ``SetTransformer_NPE/inference_model.py:77``).
            num_hiddens: Hidden width of sbi's post-pooling MLP.
            num_layers: Depth of sbi's post-pooling MLP.
            output_dim: Width of the context vector handed to the flow.
        """
        super().__init__()
        self.trial_encoder = trial_encoder
        d_model = trial_encoder.d_model

        # The clone encoder already ran, so sbi's per-trial net is the identity:
        # this wrapper only borrows sbi's NaN-aware pooling over the T dimension.
        self.perm_embed = PermutationInvariantEmbedding(
            trial_net=nn.Identity(),
            trial_net_output_dim=d_model,
            aggregation_fn=aggregation_fn,
            num_hiddens=num_hiddens,
            num_layers=num_layers,
            output_dim=output_dim,
            aggregation_dim=1,  # aggregate over the trials dimension T
        )

    @staticmethod
    def _mask_invalid_trials(X: Tensor, trial_embeddings: Tensor) -> Tensor:
        """Set the embedding of every fully-NaN trial to NaN.

        sbi's ``PermutationInvariantEmbedding`` identifies trials to skip by
        looking for NaNs in the embedding it is given, so the NaN sentinel has to
        be re-introduced after the clone encoder (which strips NaNs with
        ``nan_to_num``).

        Args:
            X: ``(B, T, K, 45)`` float32, the raw input, still carrying NaNs.
            trial_embeddings: ``(B, T, d_model)`` float32 from the clone encoder.

        Returns:
            ``(B, T, d_model)`` float32 with invalid trials' rows set to NaN.
        """
        # A trial is invalid iff its whole (K, 45) matrix is NaN, which is how
        # CNASimsDataset.__getitem__ marks a slot it never filled.
        all_nan = torch.isnan(X).all(dim=(2, 3))  # (B, T) bool

        trial_valid = ~all_nan                     # (B, T)
        mask = ~trial_valid.unsqueeze(-1)          # (B, T, 1), True = invalid

        return trial_embeddings.masked_fill(mask, float("nan"))

    def forward(self, X: Tensor) -> Tensor:
        """Embed a batch of sims.

        Args:
            X: ``(B, T, K, 45)`` float32: ``B`` sims, ``T`` trials, ``K`` clones,
                44 CNA features plus a frequency column.

        Returns:
            ``(B, output_dim)`` float32 context vectors for the flow.
        """
        batch_size, num_trials, num_clones, feature_dim = X.shape

        # 1) Embed each trial (a set of clones) with the clone encoder. Trials
        #    are folded into the batch dimension so the encoder sees (B*T, K, 45).
        X_flat = X.reshape(batch_size * num_trials, num_clones, feature_dim)
        trial_emb_flat = self.trial_encoder(X_flat)            # (B*T, d_model)
        trial_emb = trial_emb_flat.view(batch_size, num_trials, -1)  # (B, T, d_model)

        # 2) Re-mark invalid trials as NaN so sbi's pooling can ignore them.
        trial_emb = self._mask_invalid_trials(X, trial_emb)

        # 3) Pool across trials.
        return self.perm_embed(trial_emb)  # (B, output_dim)


__all__ = ["TrialsSBIEmbedding", "CloneEncoder"]
