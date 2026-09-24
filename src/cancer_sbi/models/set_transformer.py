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
        freq_renorm: bool = False,
        input_space: str = "log2",
        freq_mode: str = "weight",
        attn_ln: bool = False,
        attn_dropout_active: bool = False,
    ) -> None:
        """Build the input projection, the ISAB stack and the PMA head.

        Args:
            in_dim: Declared input width. Accepted and never read -- the
                projection below is sized by ``freq_mode`` (44 inputs in the
                published ``"weight"`` mode), so passing a different ``in_dim``
                has no effect.
            d_model: Token width throughout the stack and the output width.
            n_heads: Attention heads in every ISAB and in the PMA.
            num_layers: Number of stacked ISABs.
            dropout: IGNORED unless ``attn_dropout_active`` -- see trap 4 below.
            freq_as_weight: Multiply each token by its clone frequency. Read
                only in ``freq_mode="weight"``; ``"feature"`` never multiplies.
            freq_renorm: Repair R4. ``False`` is the published behaviour -- the
                tokens are multiplied by the RAW top-K frequencies, which sum to
                well under 1 (trap 18/19), so every token is shrunk by roughly
                ``1/K`` before attention and the logits collapse towards uniform.
                ``True`` renormalises the masked frequencies to sum to 1 per set
                before the multiply, which restores the token scale while keeping
                the relative frequency weighting. Only meaningful when
                ``freq_as_weight`` is True.

                ``ln`` stays ``False`` in both cases, deliberately: the
                architecture review's T3 pairs this renormalisation with
                ``ln=True``, but ``ln0`` renormalises every token to unit scale on
                the way out of the first ISAB, which erases the frequency
                weighting for ISABs 2-3 and the PMA -- so the two halves cancel
                and a joint run cannot be read. See
                ``docs/MODEL_IMPROVEMENT_PLAN.md`` §5 step 5c.
            input_space: Representation the 44 CNA columns are fed to the input
                projection in, with exactly the formula
                :class:`~cancer_sbi.models.mlp_encoder.BaselineCloneEmbedding`
                uses (repair T2, there for run R2 and here for runs R6-R8).
                ``"log2"`` is the published pass-through. The frequency column
                is never touched by this setting.
            freq_mode: What the clone frequency is *for*. ``"weight"`` is the
                published behaviour: the 44 CNA columns are projected and each
                token is multiplied by the frequency (see ``freq_renorm``).
                ``"feature"`` (runs R5-R8) removes the multiply entirely and
                feeds ``log10(clamp(freq, 1e-6))`` to the projection as a 45th
                input column instead, so the frequency informs the token without
                rescaling it. R4 showed the multiply is the reason CloneAtt
                stalls: even renormalised, 100 clones share a unit of mass, so
                every token is still ~0.01 of its scale with no LayerNorm to
                rescale it. ``freq_renorm`` has nothing to act on in this mode
                and ``cli/train.py`` refuses the pair.
            attn_ln: Build every MAB/ISAB/PMA with ``ln=True`` (runs R5-R8).
                ``False`` -- the default -- is trap 5, no LayerNorm anywhere.
                Unlike under ``freq_mode="weight"`` there is no cancellation to
                worry about here: ``"feature"`` does not scale the tokens, so a
                LayerNorm has no frequency weighting left to erase.
            attn_dropout_active: Wire ``dropout`` up (run R8). ``False`` -- the
                default -- is trap 4: the value is accepted and discarded.

        Raises:
            ValueError: If ``input_space`` or ``freq_mode`` is not one of its
                two allowed values.
        """
        super().__init__()

        if input_space not in ("log2", "copy"):
            raise ValueError(
                f"input_space must be 'log2' or 'copy', got {input_space!r}."
            )
        if freq_mode not in ("weight", "feature"):
            raise ValueError(
                f"freq_mode must be 'weight' or 'feature', got {freq_mode!r}."
            )

        self.freq_as_weight = freq_as_weight
        self.freq_renorm = freq_renorm
        self.input_space = input_space
        self.freq_mode = freq_mode
        self.attn_ln = attn_ln
        self.attn_dropout_active = attn_dropout_active
        # 44 in the published "weight" mode, where the frequency is a multiplier
        # and never reaches the projection; 45 in "feature" mode, where the
        # log10 frequency is the extra column. The width therefore moves only
        # when freq_mode does, which keeps every published state_dict loadable.
        self.input_proj = nn.Linear(45 if freq_mode == "feature" else 44, d_model)
        self.d_model = d_model

        # Preserved from SetTransformer_NPE/set_transformer.py:88 and :91-103.
        # Trap 4: `dropout` is accepted here and never used -- no nn.Dropout is
        # constructed anywhere in this class, so CloneAtt has NO encoder dropout
        # even though SetTransformer_NPE/inference_model.py:69 passes the same
        # 0.2 that CloneMLP genuinely applies. The parameter stays in the
        # signature because removing it would change the call sites. This looks
        # wrong but it is what the published model does; wiring dropout up
        # changes the results. See docs/REFACTOR_NOTES.md.
        #
        # Run R8 (attn_dropout_active=True) is trap 4 put right, opt-in: one
        # nn.Dropout applied to the output of every ISAB and of the PMA, which
        # is where the MLP encoder's dropout sits relative to its own blocks.
        # None -- the default -- constructs no module at all, so the trap-4
        # forward pass is untouched. nn.Dropout holds no parameters, so neither
        # branch consumes RNG or changes the state_dict.
        self.attn_dropout = nn.Dropout(dropout) if attn_dropout_active else None

        # Stack of ISABs -> permutation-equivariant encoder.
        # Preserved from SetTransformer_NPE/set_transformer.py:98 (trap 5):
        # ln=False, which is what attn_ln defaults to.
        self.layers = nn.ModuleList([
            ISAB(
                dim_in=d_model,
                dim_out=d_model,
                num_heads=n_heads,
                num_inds=num_inducing,
                ln=attn_ln,
            )
            for _ in range(num_layers)
        ])

        # PMA -> permutation-invariant pooling.
        # Preserved from SetTransformer_NPE/set_transformer.py:103 (trap 5):
        # ln=False, which is what attn_ln defaults to.
        self.pma = PMA(dim=d_model, num_heads=n_heads, num_seeds=1, ln=attn_ln)

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

        if self.input_space == "copy":
            # Repair T2, the same formula as
            # BaselineCloneEmbedding.forward (models/mlp_encoder.py): undo the
            # log, clamp to [0, 8] so the -10.966 "arm completely lost"
            # sentinel lands on 0, then recentre on diploid and halve. Kept
            # literally identical to that line so the two encoders cannot drift
            # apart. The frequency column is NOT converted.
            feats = (torch.clamp(2.0 ** (feats + 1.0), 0, 8) - 2.0) / 2.0

        if self.freq_mode == "feature":
            # Runs R5-R8. No multiply anywhere: the frequency enters as a 45th
            # input column, on a log scale so that the 1e-2..1e-6 range the top
            # 100 clones span is spread out rather than crushed against 0.
            log_freq = torch.log10(freq.clamp_min(1e-6))  # (B, K)
            h = self.input_proj(torch.cat([feats, log_freq.unsqueeze(-1)], dim=-1))
            # A padded row's frequency is 0, so its log10 is the -6 floor, which
            # is a perfectly ordinary token value -- without this the padding
            # would enter attention as data. In "weight" mode the multiply by a
            # masked-to-zero frequency is what zeroes those tokens; this is the
            # same zeroing, done explicitly because there is no multiply left.
            h = h * (~pad_mask).unsqueeze(-1).to(h.dtype)  # (B, K, d_model)
        else:
            h = self.input_proj(feats)  # (B, K, d_model)

        if self.freq_mode == "weight" and self.freq_as_weight:
            freq_masked = freq.masked_fill(pad_mask, 0.0)  # (B, K)
            # Preserved from SetTransformer_NPE/set_transformer.py:133-135.
            # Trap 19: the token embeddings are multiplied by the RAW top-K
            # frequency, which sums to less than 1 (trap 18), so every token is
            # scaled down by ~1/K before attention. The two normalising lines
            # that would have divided by the frequency sum are commented out in
            # the original and are deliberately NOT restored here. This looks
            # wrong but it is what the published model does; normalising changes
            # the results. See docs/REFACTOR_NOTES.md.
            #
            # Repair R4 (freq_renorm=True) is exactly those two commented-out
            # lines put back: divide by the masked frequency sum so the weights
            # form a per-set distribution summing to 1. Padded rows carry
            # frequency 0, so they add nothing to the sum and keep weight 0.
            # clamp_min(1e-8) guards a set whose kept clones all have frequency 0.
            if self.freq_renorm:
                w = freq_masked / freq_masked.sum(dim=1, keepdim=True).clamp_min(1e-8)
                h = h * w.unsqueeze(-1)  # (B, K, d_model)
            else:
                h = h * freq_masked.unsqueeze(-1)  # (B, K, d_model)

        # Note: pad_mask is not passed to the attention layers. Padded clones
        # enter attention as zero-frequency, zero-scaled tokens rather than being
        # masked out, which is the behaviour the published model was trained
        # with (SetTransformer_NPE/set_transformer.py:137-143).
        z = h
        for layer in self.layers:
            z = layer(z)  # (B, K, d_model)
            if self.attn_dropout is not None:
                z = self.attn_dropout(z)

        pooled = self.pma(z)[:, 0, :]  # (B, d_model)
        if self.attn_dropout is not None:
            pooled = self.attn_dropout(pooled)
        return pooled


__all__ = ["MAB", "ISAB", "PMA", "CloneSetEmbedding"]
