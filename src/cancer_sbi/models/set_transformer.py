"""Set-transformer blocks and the CloneAtt-NPE trial embedding.

Ported from ``SetTransformer_NPE/set_transformer.py:13-148``. The blocks follow
Lee et al. (2019), "Set Transformer": MAB (multihead attention block), ISAB
(induced set attention block) and PMA (pooling by multihead attention).

The original also defines ``SAB`` (``set_transformer.py:42-48``), a self-attention
block. It is **dropped here because it is never instantiated**: grepping the
three model folders finds no ``SAB(`` call site -- ``CloneSetEmbedding`` stacks
ISABs and pools with PMA. Dropping it removes no behaviour; re-add it from the
original if a future model needs it.

Nothing here runs at import time.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MAB(nn.Module):
    """Multihead attention block: attends a query set to a key set.

    Ported from ``SetTransformer_NPE/set_transformer.py:13-40``.
    """

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        ln: bool = True,
    ) -> None:
        """Build the four projections and, optionally, the two LayerNorms.

        Args:
            dim_Q: Width of the query vectors.
            dim_K: Width of the key/value vectors.
            dim_V: Width of the output, split evenly across ``num_heads``.
            num_heads: Number of attention heads. Must divide ``dim_V``.
            ln: Add LayerNorm after each residual. Defaults to True, but every
                call site in this package passes False -- see the trap-5 note.
        """
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        # Preserved from SetTransformer_NPE/set_transformer.py:21-23. Trap 5:
        # when ln is False the attributes ln0 and ln1 are never created at all,
        # which is why forward() probes them with getattr(..., None) rather than
        # testing a flag. Do not replace this with `self.ln0 = None` or with an
        # nn.Identity: the published checkpoints have no such keys in their
        # state_dict, and adding them breaks loading.
        # See docs/REFACTOR_NOTES.md.
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        """Attend ``Q`` to ``K``.

        Heads are emulated by splitting the feature dimension and stacking the
        pieces along the batch dimension, so the ``bmm`` below is over
        ``(num_heads * B)`` matrices.

        Args:
            Q: ``(B, n_q, dim_Q)`` float32 query set.
            K: ``(B, n_k, dim_K)`` float32 key/value set.

        Returns:
            ``(B, n_q, dim_V)`` float32.
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        # Preserved from SetTransformer_NPE/set_transformer.py:35. Trap 6: the
        # logits are scaled by sqrt(dim_V) = sqrt(128) = 11.31, not by the
        # standard sqrt(dim_V / num_heads) = sqrt(16) = 4. The attention is
        # therefore ~2.83x flatter than textbook scaled dot-product attention.
        # This looks wrong but it is what the published model does; changing it
        # changes the results. See docs/REFACTOR_NOTES.md.
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    """Induced set attention block: attention through a learned bottleneck.

    Ported from ``SetTransformer_NPE/set_transformer.py:50-60``. Two MABs and
    ``num_inds`` learned inducing points make the cost linear in the set size
    instead of quadratic, which matters at ``K = 100`` clones per trial.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inds: int,
        ln: bool = True,
    ) -> None:
        """Build the inducing points and the two attention blocks.

        Args:
            dim_in: Width of the input set elements.
            dim_out: Width of the output set elements and of the inducing points.
            num_heads: Attention heads for both MABs.
            num_inds: Number of learned inducing points.
            ln: Forwarded to both MABs. False in the published model (trap 5).
        """
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X: Tensor) -> Tensor:
        """Run the two-stage induced attention.

        Args:
            X: ``(B, n, dim_in)`` float32 set.

        Returns:
            ``(B, n, dim_out)`` float32, permutation-equivariant in ``n``.
        """
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by multihead attention: a set to ``num_seeds`` vectors.

    Ported from ``SetTransformer_NPE/set_transformer.py:62-70``. This is the step
    that makes the encoder permutation-*invariant*.
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = True) -> None:
        """Build the learned seed vectors and the attention block.

        Args:
            dim: Width of the input elements, the seeds and the output.
            num_heads: Attention heads.
            num_seeds: Number of output vectors. 1 in the published model.
            ln: Forwarded to the MAB. False in the published model (trap 5).
        """
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: Tensor) -> Tensor:
        """Pool a set into ``num_seeds`` vectors.

        Args:
            X: ``(B, n, dim)`` float32 set.

        Returns:
            ``(B, num_seeds, dim)`` float32.
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class CloneSetEmbedding(nn.Module):
    """Embed one trial's clone set as a ``d_model`` vector (attention).

    Ported from ``SetTransformer_NPE/set_transformer.py:76-148``. This is the
    CloneAtt-NPE trial encoder: project clones to ``d_model``, scale each token
    by its frequency, run a stack of ISABs, pool with a one-seed PMA.

    Attributes:
        d_model: Output width. Read by
            :class:`~cancer_sbi.models.trials.TrialsSBIEmbedding`.
    """

    def __init__(
        self,
        in_dim: int = 45,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 3,
        num_inducing: int = 32,
        dropout: float = 0.1,
        freq_as_weight: bool = True,
    ) -> None:
        """Build the input projection, the ISAB stack and the PMA head.

        Args:
            in_dim: Declared input width. Accepted and never read -- the
                projection below is hard-coded to 44 inputs, so passing a
                different ``in_dim`` has no effect.
            d_model: Token width throughout the stack and the output width.
            n_heads: Attention heads in every ISAB and in the PMA.
            num_layers: Number of stacked ISABs.
            dropout: Accepted and IGNORED -- see the trap-4 note below.
            freq_as_weight: Multiply each token by its clone frequency.
        """
        super().__init__()
        self.freq_as_weight = freq_as_weight
        self.input_proj = nn.Linear(44, d_model)
        self.d_model = d_model

        # Preserved from SetTransformer_NPE/set_transformer.py:88 and :91-103.
        # Trap 4: `dropout` is accepted here and never used -- no nn.Dropout is
        # constructed anywhere in this class, so CloneAtt has NO encoder dropout
        # even though SetTransformer_NPE/inference_model.py:69 passes the same
        # 0.2 that CloneMLP genuinely applies. The parameter stays in the
        # signature because removing it would change the call sites. This looks
        # wrong but it is what the published model does; wiring dropout up
        # changes the results. See docs/REFACTOR_NOTES.md.

        # Stack of ISABs -> permutation-equivariant encoder.
        # Preserved from SetTransformer_NPE/set_transformer.py:98 (trap 5): ln=False.
        self.layers = nn.ModuleList([
            ISAB(
                dim_in=d_model,
                dim_out=d_model,
                num_heads=n_heads,
                num_inds=num_inducing,
                ln=False,
            )
            for _ in range(num_layers)
        ])

        # PMA -> permutation-invariant pooling.
        # Preserved from SetTransformer_NPE/set_transformer.py:103 (trap 5): ln=False.
        self.pma = PMA(dim=d_model, num_heads=n_heads, num_seeds=1, ln=False)

    def forward(self, x: Tensor) -> Tensor:
        """Embed and pool one batch of clone sets.

        Args:
            x: ``(B, K, 45)`` float32. NaN rows are padding; zero rows are data.

        Returns:
            ``(B, d_model)`` float32 pooled embedding.

        Raises:
            AssertionError: If the last dimension is smaller than 45.
        """
        _, _, feature_dim = x.shape
        assert feature_dim >= 45

        # Compute the padding mask from NaNs *before* modifying x.
        # Preserved from SetTransformer_NPE/set_transformer.py:118. Trap 8: as in
        # the MLP encoder, only whole-NaN rows count as padding, so zero-padded
        # clone rows are processed as real clones.
        pad_mask = torch.isnan(x).all(dim=-1)  # (B, K)

        x_clean = torch.nan_to_num(x, nan=0.0)

        feats = x_clean[..., :44]  # (B, K, 44)
        freq = x_clean[..., 44]    # (B, K)

        h = self.input_proj(feats)  # (B, K, d_model)

        if self.freq_as_weight:
            freq_masked = freq.masked_fill(pad_mask, 0.0)  # (B, K)
            # Preserved from SetTransformer_NPE/set_transformer.py:133-135.
            # Trap 19: the token embeddings are multiplied by the RAW top-K
            # frequency, which sums to less than 1 (trap 18), so every token is
            # scaled down by ~1/K before attention. The two normalising lines
            # that would have divided by the frequency sum are commented out in
            # the original and are deliberately NOT restored here. This looks
            # wrong but it is what the published model does; normalising changes
            # the results. See docs/REFACTOR_NOTES.md.
            h = h * freq_masked.unsqueeze(-1)  # (B, K, d_model)

        # Note: pad_mask is not passed to the attention layers. Padded clones
        # enter attention as zero-frequency, zero-scaled tokens rather than being
        # masked out, which is the behaviour the published model was trained
        # with (SetTransformer_NPE/set_transformer.py:137-143).
        z = h
        for layer in self.layers:
            z = layer(z)  # (B, K, d_model)

        pooled = self.pma(z)[:, 0, :]  # (B, d_model)
        return pooled


__all__ = ["MAB", "ISAB", "PMA", "CloneSetEmbedding"]
