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

from typing import Optional, Union

import torch
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from torch import Tensor, nn

from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding
from cancer_sbi.models.set_transformer import PMA, CloneSetEmbedding

#: Attention heads used by :class:`AttentionTrialPooling` when the encoder has
#: no ``n_heads`` of its own. CloneMLP is that case -- its ``EncoderConfig``
#: leaves ``n_heads`` at ``None`` because the MLP encoder has no attention --
#: and ``--trial-pool attention`` still has to build a PMA for it. 8 is
#: CloneAtt's published head count, and 128 (both encoders' ``d_model``) is
#: divisible by it.
DEFAULT_TRIAL_POOL_HEADS = 8

#: Either clone-set encoder can be wrapped; both expose ``d_model`` and map
#: ``(B, K, 45) -> (B, d_model)``.
CloneEncoder = Union[BaselineCloneEmbedding, CloneSetEmbedding]



class AttentionTrialPooling(nn.Module):
    """Pool ``(B, T, d_model)`` trial embeddings with a PMA instead of a mean.

    Matrix 4, run R20. The published wrapper hands the trial embeddings to sbi's
    ``PermutationInvariantEmbedding``, which takes a NaN-masked **mean** over
    the 25 trials and then runs a small MLP on
    ``[pooled, number of valid trials]``. A mean gives every trial the same
    weight, which is the modelling assumption this switch questions: the trials
    of one sim are 25 draws of the same process but not equally informative.

    This module keeps the second half of that computation exactly -- the same
    ``FCEmbedding``, the same ``d_model + 1`` input, the same ``output_dim`` --
    and replaces only the mean, so **the flow's context width does not move**
    and a ``--trial-pool attention`` run is comparable with its ``mean``
    counterpart on everything else.

    Invalid trials (a sim's unfilled slots, marked NaN by
    :meth:`TrialsSBIEmbedding._mask_invalid_trials`) are masked out of the
    attention rather than fed to it: a NaN key would poison every logit, and a
    zeroed one would still take softmax mass away from the real trials. With a
    pre-built clone cache every trial is present and the mask is all-False, but
    the uncached path can have NaN slots.

    Attributes:
        output_dim: Width of the context vector, i.e. what the flow sees.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = DEFAULT_TRIAL_POOL_HEADS,
        num_hiddens: int = 256,
        num_layers: int = 2,
        output_dim: int = 256,
        ln: bool = False,
        attn_scale: str = "published",
    ) -> None:
        """Build the one-seed PMA and the post-pooling MLP.

        Args:
            d_model: Width of the per-trial embeddings, i.e. the clone
                encoder's output width.
            n_heads: Heads of the pooling PMA. Must divide ``d_model``.
            num_hiddens: Hidden width of the post-pooling MLP, as sbi's
                ``PermutationInvariantEmbedding`` uses it.
            num_layers: Depth of the post-pooling MLP, likewise.
            output_dim: Width of the context vector, likewise. Keep it equal to
                the ``mean`` path's or the flow changes shape.
            ln: ``LayerNorm`` inside the PMA, following ``attn_ln`` so the
                pooling matches the encoder it sits on top of.
            attn_scale: Logit scaling of the PMA, following ``attn_scale`` for
                the same reason.
        """
        super().__init__()
        self.output_dim = output_dim
        self.pma = PMA(
            dim=d_model,
            num_heads=n_heads,
            num_seeds=1,
            ln=ln,
            attn_scale=attn_scale,
        )
        # Byte-for-byte the subnet PermutationInvariantEmbedding builds
        # (sbi/neural_nets/embedding_nets/__init__.py): input_dim is
        # trial_net_output_dim + 1, the +1 being the number of valid trials,
        # which is appended in forward() below exactly as sbi appends it.
        self.fc_subnet = FCEmbedding(
            input_dim=d_model + 1,
            output_dim=output_dim,
            num_layers=num_layers,
            num_hiddens=num_hiddens,
        )

    def forward(self, trial_embeddings: Tensor) -> Tensor:
        """Pool the trials of each sim into one context vector.

        Args:
            trial_embeddings: ``(B, T, d_model)`` float32, with a fully-NaN row
                marking an invalid trial -- the convention sbi's pooling uses
                and :meth:`TrialsSBIEmbedding._mask_invalid_trials` writes.

        Returns:
            ``(B, output_dim)`` float32.
        """
        # A trial is invalid iff its whole embedding row is NaN, which is the
        # sentinel _mask_invalid_trials writes and the one sbi's pooling reads.
        invalid = torch.isnan(trial_embeddings).all(dim=-1)   # (B, T)
        trial_counts = (~invalid).sum(dim=1, keepdim=True).to(trial_embeddings.dtype)

        # NaNs must leave the tensor before the projections inside the PMA see
        # them: a masked softmax kills a NaN key's *weight*, not the NaN itself.
        clean = torch.nan_to_num(trial_embeddings, nan=0.0)
        pooled = self.pma(clean, key_mask=invalid)[:, 0, :]   # (B, d_model)

        # The same [pooled, trial_counts] concatenation sbi does, so the MLP
        # sees the input it was sized for.
        return self.fc_subnet(torch.cat([pooled, trial_counts], dim=1))


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
        trial_pool: str = "mean",
        n_heads: Optional[int] = None,
        attn_ln: bool = False,
        attn_scale: str = "published",
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
            trial_pool: Matrix 4, run R20. ``"mean"`` -- the default and the
                published behaviour -- pools the trials with sbi's masked mean.
                ``"attention"`` pools them with :class:`AttentionTrialPooling`
                instead; ``num_hiddens``, ``num_layers`` and ``output_dim``
                keep their meaning there, so the flow's context width is the
                same either way. Only the module that does the pooling
                changes, which is why the two builds have different
                ``state_dict`` keys and a checkpoint of one cannot load into
                the other.
            n_heads: Heads of the pooling PMA, ``"attention"`` only.
                ``None`` -- which is what CloneMLP's encoder config carries,
                having no attention of its own -- falls back to
                :data:`DEFAULT_TRIAL_POOL_HEADS`.
            attn_ln: ``LayerNorm`` inside the pooling PMA, ``"attention"``
                only. Follows the encoder's ``attn_ln`` so the two halves of
                the network agree.
            attn_scale: Logit scaling of the pooling PMA, ``"attention"``
                only. Follows the encoder's ``attn_scale``, likewise.

        Raises:
            ValueError: If ``trial_pool`` is neither ``"mean"`` nor
                ``"attention"``.
        """
        super().__init__()
        if trial_pool not in ("mean", "attention"):
            raise ValueError(
                f"trial_pool must be 'mean' or 'attention', got {trial_pool!r}."
            )
        self.trial_encoder = trial_encoder
        self.trial_pool = trial_pool
        d_model = trial_encoder.d_model

        # `perm_embed` exists only on the published path, and `pool_attn` only
        # on the new one -- not both with one left unused. An unused submodule
        # would put keys in every checkpoint's state_dict that nothing reads,
        # and the optimiser would carry its parameters.
        self.perm_embed = None
        self.pool_attn = None
        if trial_pool == "attention":
            self.pool_attn = AttentionTrialPooling(
                d_model=d_model,
                n_heads=n_heads if n_heads else DEFAULT_TRIAL_POOL_HEADS,
                num_hiddens=num_hiddens,
                num_layers=num_layers,
                output_dim=output_dim,
                ln=attn_ln,
                attn_scale=attn_scale,
            )
        else:
            # The clone encoder already ran, so sbi's per-trial net is the
            # identity: this wrapper only borrows sbi's NaN-aware pooling over
            # the T dimension.
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

        # 3) Pool across trials. Exactly one of the two modules exists; see
        #    __init__.
        if self.pool_attn is not None:
            return self.pool_attn(trial_emb)  # (B, output_dim)
        return self.perm_embed(trial_emb)  # (B, output_dim)


__all__ = [
    "TrialsSBIEmbedding",
    "AttentionTrialPooling",
    "CloneEncoder",
    "DEFAULT_TRIAL_POOL_HEADS",
]
