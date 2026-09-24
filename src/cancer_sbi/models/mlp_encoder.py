"""Per-clone MLP encoder: the CloneMLP-NPE trial embedding.

Ported from ``Base_NPE/MLP_encoder.py:12-91``.

This is the "no attention" baseline that became the paper's main model. It
embeds every clone independently with a small MLP, then pools the clones of one
trial into a single vector with a frequency-weighted mean.

Nothing here runs at import time.
"""

import torch
from torch import Tensor, nn


class BaselineCloneEmbedding(nn.Module):
    """Embed one trial's clone set as a ``d_model`` vector (no attention).

    Ported from ``Base_NPE/MLP_encoder.py:12-91``.

    Input layout, ``x`` of shape ``(B, K, 45)``:

    * ``x[..., :44]`` -- the clone's copy-number features.
    * ``x[..., 44]`` -- the clone's frequency within the trial.
    * A row that is NaN in *every* column is a padded trial slot.
    * A row that is all zeros is *valid data* as far as this module is
      concerned -- see the trap-8 note in
      :func:`cancer_sbi.data.clone_sets.top_frequent_rows_tensor`.

    Attributes:
        d_model: Output width. Read by
            :class:`~cancer_sbi.models.trials.TrialsSBIEmbedding`.
    """

    def __init__(
        self,
        in_dim: int = 45,
        d_model: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        freq_as_weight: bool = True,
        include_freq_in_mlp: bool = False,
        input_space: str = "log2",
        freq_mode: str = "weight",
    ) -> None:
        """Build the per-clone MLP and the output LayerNorm.

        Args:
            in_dim: Declared input width. Only checked, never used to size a
                layer -- the MLP's input width comes from ``include_freq_in_mlp``.
            d_model: Output width of the MLP and of the pooled embedding.
            hidden_dim: Width of the hidden layers.
            num_layers: Number of ``Linear`` layers, so 3 gives
                ``44 -> 256 -> 256 -> 128`` with ReLU+Dropout between them.
            dropout: Dropout probability between hidden layers. Unlike
                CloneAtt's encoder, this one really uses it (trap 4).
            freq_as_weight: Pool by frequency-weighted mean when True, by plain
                masked mean when False. The published model uses True.
            include_freq_in_mlp: Feed the frequency column to the MLP as a 45th
                input. ``False`` in the published model, which keeps the
                projection at 44 inputs to match CloneAtt's ``input_proj``.
            input_space: Representation the 44 CNA columns are fed to the MLP in.
                ``"log2"`` is the published behaviour: the columns are passed
                through untouched, so the forward pass is byte-identical to the
                pre-repair model. ``"copy"`` is repair T2 / run R2 -- the columns
                are converted from log2 ratio to a centred copy-number scale by
                ``(clamp(2**(x+1), 0, 8) - 2) / 2`` inside :meth:`forward`. The
                frequency column is never touched by either setting.

                Why here and not in the data loader: one loader feeds both
                CloneMLP and CloneAtt, so a loader-side transform would leak into
                the CloneAtt R4 comparison, and the preprocessed cache stores
                pre-transform data. See ``docs/CODEBASE_IMPROVEMENT_PLAN.md``,
                "The T2 edit must go in the encoder, not the loader".
            freq_mode: What the clone frequency is *for*, matrix 3 / run R10.
                ``"weight"`` is the published behaviour: the frequency reaches
                the network only as the pooling weight below, and the MLP sees
                44 inputs. ``"feature"`` additionally feeds
                ``log10(clamp(freq, 1e-6))`` to the MLP as a 45th input column,
                exactly as ``CloneSetEmbedding`` does in its own ``"feature"``
                mode -- the same clamp, the same base, so the two encoders
                cannot drift apart. The frequency-weighted pooling is **kept**:
                unlike CloneAtt's token multiply (trap 19), CloneMLP's use of
                the frequency is a *normalised* weighted mean, which shrinks
                nothing, so there is nothing there to remove.

                This is a different column from ``include_freq_in_mlp``, which
                appends the RAW frequency. Asking for both is refused rather
                than silently giving the MLP 46 inputs.

        Raises:
            AssertionError: If ``in_dim`` is smaller than 45.
            ValueError: If ``input_space`` is not ``"log2"`` or ``"copy"``, if
                ``freq_mode`` is not ``"weight"`` or ``"feature"``, or if
                ``freq_mode="feature"`` is combined with
                ``include_freq_in_mlp=True``.
        """
        super().__init__()
        assert in_dim >= 45, "Expected at least 45 features (44 CNA + freq)."

        if input_space not in ("log2", "copy"):
            raise ValueError(
                f"input_space must be 'log2' or 'copy', got {input_space!r}."
            )
        if freq_mode not in ("weight", "feature"):
            raise ValueError(
                f"freq_mode must be 'weight' or 'feature', got {freq_mode!r}."
            )
        if freq_mode == "feature" and include_freq_in_mlp:
            raise ValueError(
                "freq_mode='feature' and include_freq_in_mlp=True both append a "
                "frequency column to the MLP's input -- the raw value in one "
                "case, log10 of it in the other. Pick one; together they would "
                "give the MLP 46 inputs and two versions of the same number."
            )

        self.d_model = d_model
        self.freq_as_weight = freq_as_weight
        self.include_freq_in_mlp = include_freq_in_mlp
        self.input_space = input_space
        self.freq_mode = freq_mode

        # Only ONE of the two flags can be on (refused above), so the width is
        # 44 in the published configuration and 45 in either extended one.
        mlp_in = 44 + (1 if (include_freq_in_mlp or freq_mode == "feature") else 0)

        layers = []
        dims = [mlp_in] + [hidden_dim] * (max(0, num_layers - 1)) + [d_model]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # No ReLU/Dropout after the final projection, so the embedding is
            # linear before the LayerNorm below.
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        self.ln = nn.LayerNorm(d_model)

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

        # Preserved from Base_NPE/MLP_encoder.py:65. Trap 8: this is the ONLY
        # padding test, and it only fires for whole-NaN rows. Clone rows that
        # top_frequent_rows_tensor zero-padded are therefore treated as real
        # clones here. Do not add clone-level masking; it changes the results.
        # See docs/REFACTOR_NOTES.md.
        pad_mask = torch.isnan(x).all(dim=-1)  # (B, K) True = padded/invalid
        x_clean = torch.nan_to_num(x, nan=0.0)

        feats = x_clean[..., :44]  # (B, K, 44)
        freq = x_clean[..., 44]    # (B, K)

        if self.input_space == "copy":
            # Repair T2 / run R2. The CNA columns arrive as log2 ratios against a
            # diploid baseline, where 16-21% of the values are the single
            # "arm completely lost" sentinel -10.966, about 15 sd from everything
            # else. Undo the log: 2**(x+1) is the absolute copy number, clamped to
            # [0, 8] so the sentinel lands on 0 and amplifications saturate, then
            # recentred on diploid and halved. The sentinel therefore maps to
            # -1.0 instead of -10.966. The frequency column is NOT converted.
            feats = (torch.clamp(2.0 ** (feats + 1.0), 0, 8) - 2.0) / 2.0

        if self.include_freq_in_mlp:
            feats = torch.cat([feats, freq.unsqueeze(-1)], dim=-1)  # (B, K, 45)
        elif self.freq_mode == "feature":
            # Matrix 3 / run R10. The log scale is the point: the top-100
            # frequencies span 1e-2..1e-6 (trap 18 leaves them unnormalised), so
            # the raw column include_freq_in_mlp would append is crushed against
            # 0. Written exactly as CloneSetEmbedding.forward writes it.
            log_freq = torch.log10(freq.clamp_min(1e-6))  # (B, K)
            feats = torch.cat([feats, log_freq.unsqueeze(-1)], dim=-1)  # (B, K, 45)

        h = self.mlp(feats)  # (B, K, d_model)
        h = self.ln(h)

        valid = (~pad_mask).float()  # (B, K)

        if self.freq_as_weight:
            # Frequency-weighted mean. The weights are the *unnormalised*
            # top-K frequencies (trap 18), so this division by their sum is the
            # only place they get normalised -- and it normalises the pooling,
            # not the per-clone scale.
            w = freq.masked_fill(pad_mask, 0.0)  # (B, K)
            # Preserved from Base_NPE/MLP_encoder.py:84: clamp_min(1e-8) guards
            # a trial whose kept clones all have frequency 0, which zero-padded
            # rows can produce.
            w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)  # (B, 1)
            pooled = (h * w.unsqueeze(-1)).sum(dim=1) / w_sum  # (B, d_model)
        else:
            v_sum = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
            pooled = (h * valid.unsqueeze(-1)).sum(dim=1) / v_sum  # (B, d_model)

        return pooled


__all__ = ["BaselineCloneEmbedding"]
