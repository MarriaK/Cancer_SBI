"""ArmToken-NPE: an arm-equivariant encoder over per-arm summary tokens.

Matrix 5. The three published encoders all tokenise a *clone* -- a 44-vector of
log2 ratios -- and pool the clones away, so the only thing the flow ever sees of
arm 17 is whatever survived a 128-wide projection that mixed all 44 arms
together on the first layer. The 44 fitness coefficients this project infers are
per-arm, and nothing in those architectures ties output arm 17 to input arm 17.

This module inverts the set: the *arms* are the tokens and the clones are
summarised away. Eight frequency-weighted moments per (trial, arm) describe the
distribution of that arm's copy number across the clone population; the moments
are pooled over trials, projected by one MLP **shared across all 44 arms**, run
through a stack of ISABs over the 44-element arm set, and read out to
``d_arm`` numbers per arm. The context vector is the 44 per-arm blocks
concatenated in arm order, plus one global block.

The consequence is the property the module exists for: permuting the 44 input
columns permutes the 44 output blocks, exactly. Arm identity reaches the flow
**only** through the concatenation order -- every weight is shared, so the
network cannot learn "arm 17 is special" and cannot mis-align arm 17's evidence
with arm 17's coefficient.

Nothing here runs at import time.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from cancer_sbi.models.set_transformer import ISAB, PMA

#: Number of chromosome-arm columns in a clone row. The 45th column is the
#: clone's frequency, which is never one of these.
N_ARMS = 44

#: Moments computed per (trial, arm) by :func:`arm_moments`.
N_MOMENTS = 8

#: Global per-trial scalars: log10 of the kept frequency mass and log10 of the
#: largest single clone frequency.
N_GLOBAL_SCALARS = 2

#: Published defaults of :class:`ArmTokenEmbedding`, repeated here so that
#: ``build_embedding_net`` can fall back to them for an ``EncoderConfig`` whose
#: new optional fields are ``None`` -- which is every checkpoint written before
#: matrix 5.
DEFAULT_D_TOKEN = 64
DEFAULT_D_ARM = 8
DEFAULT_D_GLOBAL = 64
DEFAULT_N_ARM_LAYERS = 1
DEFAULT_ARM_NUM_INDUCING = 16
DEFAULT_N_HEADS = 4

#: Hard cap on ``44 * d_arm + d_global``. The flow's context is the embedding's
#: output, and ``build_nsf`` puts a ``50``-wide residual net on it; a context
#: much wider than the published 256 makes the flow's first layer the biggest
#: thing in the network again, which is the opposite of what this encoder is
#: for. ``cli/train.py`` refuses a ``--d-arm`` that breaks it.
MAX_CONTEXT_WIDTH = 512

#: Copy-space cut points of moments 2, 3 and 4: "meaningfully lost", "gained"
#: and "deeply lost" relative to diploid, in the same units
#: ``CloneSetEmbedding`` produces under ``input_space="copy"`` (0 == diploid,
#: -1 == nothing left, +3 == the clamp at 8 copies).
COPY_CUT_LOSS = -0.25
COPY_CUT_GAIN = 0.25
COPY_CUT_DEEP_LOSS = -0.75


def _log2_cut(copy_cut: float) -> float:
    """The log2-ratio value that maps to ``copy_cut`` in copy space.

    ``c = (clamp(2 ** (x + 1), 0, 8) - 2) / 2`` is monotone non-decreasing in
    ``x``, so a threshold test on ``c`` is the same test on ``x`` at the
    pre-image of the threshold. Inverting the unclamped branch,
    ``x = log2(2 * c + 2) - 1``.

    Args:
        copy_cut: A cut point in copy space, strictly inside ``(-1, 3)``.

    Returns:
        The equivalent cut point in log2 space.
    """
    return math.log2(2.0 * copy_cut + 2.0) - 1.0


#: The three cut points above, in log2 space, for ``input_space="log2"``.
LOG2_CUT_LOSS = _log2_cut(COPY_CUT_LOSS)          # -0.415037...
LOG2_CUT_GAIN = _log2_cut(COPY_CUT_GAIN)          # +0.321928...
LOG2_CUT_DEEP_LOSS = _log2_cut(COPY_CUT_DEEP_LOSS)  # -2.0 exactly


def arm_moments(
    X: Tensor,
    input_space: str = "copy",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Summarise each (trial, arm) by eight frequency-weighted moments.

    The clone axis is reduced here and never reappears: everything downstream
    works on ``(B, T, 44, 8)``. Reducing it with *weighted* statistics rather
    than attention is deliberate -- trap 19 is that CloneAtt's frequency
    multiply shrinks every token to ~1/K of its scale, and a weighted mean
    cannot have that failure mode because its weights are renormalised to sum
    to 1 by construction.

    The eight, in order:

    0. weighted mean of the arm's copy number over the clone population
    1. weighted standard deviation of it -- subclonal heterogeneity at this arm
    2. weighted fraction of the population with a meaningful loss
    3. weighted fraction with a meaningful gain
    4. weighted fraction with a deep loss (near-total)
    5. maximum over the kept clones, weighted by nothing -- presence, not mass
    6. minimum over the kept clones, likewise
    7. the value in the single most frequent clone -- the dominant clone's copy
       number, which is the whole of what DominantClone-NPE ever sees

    Args:
        X: ``(B, T, K, 45)`` float32. A fully-NaN ``(K, 45)`` slice is a trial
            slot the dataset never filled; a fully-NaN row inside a trial is
            clone padding (trap 8: a zero row is data, not padding).
        input_space: ``"copy"`` -- the default and what the preset uses --
            converts the 44 columns with exactly the line
            ``CloneSetEmbedding.forward`` uses, so the two encoders cannot
            drift apart. ``"log2"`` leaves them alone and moves the three cut
            points of moments 2-4 to the log2 pre-images of the same copy-space
            thresholds, so the two spaces ask the same question of the data.

    Returns:
        ``(moments, trial_valid, glob)``:

        * ``moments`` -- ``(B, T, 44, 8)`` float32, all-zero on an invalid trial.
        * ``trial_valid`` -- ``(B, T)`` bool, ``False`` on an invalid trial.
        * ``glob`` -- ``(B, T, 2)`` float32: ``log10`` of the kept frequency
          mass and ``log10`` of the largest single clone frequency, all-zero on
          an invalid trial.

    Raises:
        ValueError: If ``input_space`` is neither ``"log2"`` nor ``"copy"``.
        AssertionError: If ``X`` is not 4-dimensional with at least 45 columns.
    """
    if input_space not in ("log2", "copy"):
        raise ValueError(
            f"input_space must be 'log2' or 'copy', got {input_space!r}."
        )
    assert X.dim() == 4 and X.shape[-1] >= 45, (
        f"arm_moments expects (B, T, K, 45); got {tuple(X.shape)}."
    )

    # Masks come from the NaNs *before* anything touches the tensor, exactly as
    # in CloneSetEmbedding.forward.
    pad_mask = torch.isnan(X).all(dim=-1)                 # (B, T, K) True = pad
    trial_valid = ~pad_mask.all(dim=-1)                   # (B, T)

    x_clean = torch.nan_to_num(X, nan=0.0)
    feats = x_clean[..., :N_ARMS]                         # (B, T, K, 44)
    freq = x_clean[..., 44]                               # (B, T, K)

    if input_space == "copy":
        # Copied verbatim from CloneSetEmbedding.forward (repair T2): undo the
        # log, clamp to [0, 8] so the -10.966 "arm completely lost" sentinel
        # lands on 0, then recentre on diploid and halve.
        feats = (torch.clamp(2.0 ** (feats + 1.0), 0, 8) - 2.0) / 2.0
        cut_loss, cut_gain, cut_deep = (
            COPY_CUT_LOSS,
            COPY_CUT_GAIN,
            COPY_CUT_DEEP_LOSS,
        )
    else:
        cut_loss, cut_gain, cut_deep = (
            LOG2_CUT_LOSS,
            LOG2_CUT_GAIN,
            LOG2_CUT_DEEP_LOSS,
        )

    # Renormalised clone weights. Trap 18/19 put right at the source: the top-K
    # frequencies sum to well under 1, so a raw-frequency weight would shrink
    # every moment by ~1/K. Dividing by the masked sum makes w a distribution
    # over the kept clones, which is what "the fraction of the population with
    # a loss at this arm" has to mean for the number to be readable at all.
    freq_masked = freq.masked_fill(pad_mask, 0.0)          # (B, T, K)
    freq_sum = freq_masked.sum(dim=-1, keepdim=True)       # (B, T, 1)
    # clamp_min guards a trial whose kept clones all carry frequency 0, which
    # then gets w == 0 everywhere and hence all-zero weighted moments. That is
    # a finite, honest answer: nothing is known about the population's mass.
    w = freq_masked / freq_sum.clamp_min(1e-12)            # (B, T, K)
    w_arm = w.unsqueeze(-1)                                # (B, T, K, 1)

    mean = (w_arm * feats).sum(dim=2)                      # (B, T, 44)
    var = (w_arm * (feats - mean.unsqueeze(2)) ** 2).sum(dim=2)
    # sqrt(0) has an infinite gradient, so the variance gets an epsilon rather
    # than a clamp. A genuinely homogeneous arm therefore reports sd == 1e-6,
    # not 0; that is four orders of magnitude below the smallest real spread
    # and cannot be confused with one.
    sd = torch.sqrt(var.clamp_min(0.0) + 1e-12)            # (B, T, 44)

    frac_loss = (w_arm * (feats < cut_loss)).sum(dim=2)
    frac_gain = (w_arm * (feats > cut_gain)).sum(dim=2)
    frac_deep_loss = (w_arm * (feats < cut_deep)).sum(dim=2)

    # Presence statistics: over the kept clones, unweighted. A padded clone
    # must not win either extremum, so it is pushed outside any real value --
    # the copy space is bounded by [-1, 3] and log2 by the -10.966 sentinel.
    big = torch.finfo(feats.dtype).max / 4.0
    pad_arm = pad_mask.unsqueeze(-1)                       # (B, T, K, 1)
    max_c = feats.masked_fill(pad_arm, -big).amax(dim=2)   # (B, T, 44)
    min_c = feats.masked_fill(pad_arm, big).amin(dim=2)    # (B, T, 44)
    # A trial with no kept clone at all would otherwise report +-big. It is an
    # invalid trial and gets zeroed below anyway; zeroing here as well keeps
    # the intermediate finite, which matters because NaN/inf * 0 is NaN.
    any_clone = (~pad_mask).any(dim=-1, keepdim=True)      # (B, T, 1)
    max_c = torch.where(any_clone, max_c, torch.zeros_like(max_c))
    min_c = torch.where(any_clone, min_c, torch.zeros_like(min_c))

    # The dominant clone's profile: the row with the largest frequency. argmax
    # on an all-zero row returns 0, which is a kept clone whenever there is one.
    dom_idx = freq_masked.argmax(dim=-1)                   # (B, T)
    dom_index = dom_idx.view(*dom_idx.shape, 1, 1).expand(
        *dom_idx.shape, 1, N_ARMS
    )
    dominant = torch.gather(feats, 2, dom_index).squeeze(2)  # (B, T, 44)

    moments = torch.stack(
        [mean, sd, frac_loss, frac_gain, frac_deep_loss, max_c, min_c, dominant],
        dim=-1,
    )                                                      # (B, T, 44, 8)

    # The two global scalars. 1e-6 is the same floor CloneSetEmbedding uses for
    # its log10 frequency feature, so a zero frequency reads -6 in both places.
    total_freq = torch.log10(freq_sum.squeeze(-1).clamp_min(1e-6))   # (B, T)
    max_freq = torch.log10(freq_masked.amax(dim=-1).clamp_min(1e-6))  # (B, T)
    glob = torch.stack([total_freq, max_freq], dim=-1)     # (B, T, 2)

    # Invalid trials are zeroed rather than left as whatever nan_to_num made of
    # them: they are excluded by `trial_valid` everywhere downstream, and a
    # zero is the value that survives a masked sum untouched.
    valid_f = trial_valid.to(moments.dtype)
    moments = moments * valid_f.view(*valid_f.shape, 1, 1)
    glob = glob * valid_f.unsqueeze(-1)

    return moments, trial_valid, glob


def _masked_mean_sd_over_trials(
    values: Tensor,
    trial_valid: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Mean and standard deviation over the valid trials of each sim.

    Args:
        values: ``(B, T, 44, 8)`` float32, already zeroed on invalid trials.
        trial_valid: ``(B, T)`` bool.

    Returns:
        ``(mean, sd)``, both ``(B, 44, 8)``. A sim with no valid trial at all
        gets zeros rather than a division by zero.
    """
    valid = trial_valid.to(values.dtype).view(*trial_valid.shape, 1, 1)
    count = valid.sum(dim=1).clamp_min(1.0)                # (B, 1, 1)
    mean = (values * valid).sum(dim=1) / count             # (B, 44, 8)
    centred = (values - mean.unsqueeze(1)) * valid
    var = (centred ** 2).sum(dim=1) / count
    # Same epsilon-not-clamp reasoning as in arm_moments: a sim with one valid
    # trial has zero spread and must still be differentiable.
    return mean, torch.sqrt(var.clamp_min(0.0) + 1e-12)


class ArmTokenEmbedding(nn.Module):
    """Embed a sim as 44 per-arm blocks plus one global block.

    Unlike :class:`~cancer_sbi.models.trials.TrialsSBIEmbedding`, this module is
    the whole embedding net: it takes ``(B, T, K, 45)`` and returns
    ``(B, d_model)``. It must **not** be wrapped in ``TrialsSBIEmbedding`` --
    that wrapper's ``FCEmbedding`` is a dense MLP over the pooled vector, and
    running it on this output would mix the 44 per-arm blocks back together,
    destroying the one property this encoder exists for.

    Attributes:
        d_model: ``n_arms * d_arm + d_global``. This is the flow's context
            width; ``build_training_components`` never reads it, inferring the
            width from a real forward pass instead, but it is the number that
            forward pass will produce.
        trial_pool: ``"mean"`` or ``"attention"``.
        input_space: ``"copy"`` or ``"log2"``, forwarded to :func:`arm_moments`.
    """

    def __init__(
        self,
        in_dim: int = 45,
        n_arms: int = N_ARMS,
        d_token: int = DEFAULT_D_TOKEN,
        d_arm: int = DEFAULT_D_ARM,
        d_global: int = DEFAULT_D_GLOBAL,
        n_arm_layers: int = DEFAULT_N_ARM_LAYERS,
        n_heads: int = DEFAULT_N_HEADS,
        num_inducing: int = DEFAULT_ARM_NUM_INDUCING,
        trial_pool: str = "mean",
        input_space: str = "copy",
        attn_ln: bool = True,
        dropout: float = 0.2,
        attn_dropout_active: bool = False,
        attn_scale: str = "published",
    ) -> None:
        """Build the per-arm stack, the arm head and the global head.

        Args:
            in_dim: Declared clone-row width. Accepted and never read -- the
                clone axis is reduced by :func:`arm_moments`, which sizes
                itself -- and kept so the signature matches the other encoders.
            n_arms: Number of arm tokens. 44 everywhere in this project.
            d_token: Width of an arm token through the whole stack.
            d_arm: Numbers read out per arm. The flow sees ``n_arms * d_arm``
                of them, in arm order.
            d_global: Width of the global block appended to those.
            n_arm_layers: Number of ISABs over the 44-arm set. ``0`` is a
                meaningful setting: the arms are then processed completely
                independently of one another, which is the strictest form of
                the equivariance and the ablation run AT3 tests.
            n_heads: Attention heads in every ISAB and PMA. Must divide
                ``d_token``.
            num_inducing: Inducing points per arm ISAB. 16 is already more than
                a third of the 44-element set, so the bottleneck is mild.
            trial_pool: How the ``T`` trials are reduced. ``"mean"`` -- the
                default -- takes the masked mean **and** standard deviation of
                the moments over the valid trials, so the per-arm MLP sees
                ``2 * 8 = 16`` numbers and knows how much the arm varied
                between replicates. ``"attention"`` instead projects each
                trial's 8 moments to ``d_token`` and pools the trials with a
                one-seed PMA, shared across arms, with invalid trials
                key-masked out.
            input_space: Forwarded to :func:`arm_moments`.
            attn_ln: ``LayerNorm`` in every MAB/ISAB/PMA. ``True`` here, unlike
                the published CloneAtt (trap 5): there is no frequency multiply
                in this encoder for a LayerNorm to erase -- the moments are
                already renormalised -- so the reason trap 5 was left in place
                for ``freq_mode="weight"`` does not apply.
            dropout: Probability used by ``attn_dropout_active``.
            attn_dropout_active: ``True`` applies ``nn.Dropout(dropout)`` after
                each arm layer. ``False`` -- the default -- constructs no
                module at all, so the graph is untouched. Dropout is opt-in
                here for the same reason as in ``CloneSetEmbedding``: the
                per-arm blocks are only ``d_arm`` wide, and dropping a fifth of
                eight numbers is a much blunter instrument than dropping a
                fifth of 128.
            attn_scale: Forwarded to every MAB. ``"published"`` is trap 6's
                ``sqrt(dim_V)``; ``"standard"`` is ``sqrt(dim_V / num_heads)``.

        Raises:
            ValueError: If ``trial_pool`` is not one of its two values, if
                ``d_token`` is not divisible by ``n_heads``, or if
                ``n_arms * d_arm + d_global`` exceeds
                :data:`MAX_CONTEXT_WIDTH`.
        """
        super().__init__()
        if trial_pool not in ("mean", "attention"):
            raise ValueError(
                f"trial_pool must be 'mean' or 'attention', got {trial_pool!r}."
            )
        if d_token % n_heads:
            raise ValueError(
                f"d_token {d_token} is not divisible by n_heads {n_heads}: the "
                f"attention splits the token evenly across the heads, so an "
                f"indivisible pair would silently discard {d_token % n_heads} "
                f"of every token's features."
            )
        width = n_arms * d_arm + d_global
        if width > MAX_CONTEXT_WIDTH:
            raise ValueError(
                f"n_arms * d_arm + d_global = {n_arms} * {d_arm} + {d_global} "
                f"= {width}, which exceeds the {MAX_CONTEXT_WIDTH} cap on the "
                f"flow's context width. Lower d_arm or d_global."
            )
        # input_space is validated by arm_moments on the first forward pass;
        # doing it here too means a bad value fails at build time, where the
        # flag that caused it is still named.
        if input_space not in ("log2", "copy"):
            raise ValueError(
                f"input_space must be 'log2' or 'copy', got {input_space!r}."
            )

        self.in_dim = in_dim
        self.n_arms = n_arms
        self.d_token = d_token
        self.d_arm = d_arm
        self.d_global = d_global
        self.n_arm_layers = n_arm_layers
        self.trial_pool = trial_pool
        self.input_space = input_space
        self.attn_ln = attn_ln
        self.attn_dropout_active = attn_dropout_active
        self.attn_scale = attn_scale
        self.d_model = width

        # The attention pooling path's two extra modules exist only on that
        # path -- an unused submodule would put keys in every checkpoint that
        # nothing reads, and the optimiser would carry its parameters. Same
        # rule as TrialsSBIEmbedding's perm_embed/pool_attn pair.
        self.moment_proj: Optional[nn.Module] = None
        self.trial_pma: Optional[nn.Module] = None
        if trial_pool == "attention":
            self.moment_proj = nn.Linear(N_MOMENTS, d_token)
            self.trial_pma = PMA(
                dim=d_token,
                num_heads=n_heads,
                num_seeds=1,
                ln=attn_ln,
                attn_scale=attn_scale,
            )
            arm_mlp_in = d_token
        else:
            # [mean over trials, sd over trials] of the 8 moments.
            arm_mlp_in = 2 * N_MOMENTS

        # One MLP for all 44 arms. This is where the equivariance is bought:
        # there is no per-arm parameter anywhere in this module.
        self.arm_mlp = nn.Sequential(
            nn.Linear(arm_mlp_in, d_token),
            nn.ReLU(),
            nn.Linear(d_token, d_token),
        )

        # Equivariant mixing across arms: an ISAB maps a set to a set, so arm
        # 17's output moves with arm 17 when the set is permuted. This is what
        # lets the encoder say "this arm is lost *and* its neighbour is not"
        # without ever learning which arm is which.
        self.arm_layers = nn.ModuleList(
            [
                ISAB(
                    dim_in=d_token,
                    dim_out=d_token,
                    num_heads=n_heads,
                    num_inds=num_inducing,
                    ln=attn_ln,
                    attn_scale=attn_scale,
                )
                for _ in range(n_arm_layers)
            ]
        )
        self.arm_dropout = nn.Dropout(dropout) if attn_dropout_active else None

        self.arm_head = nn.Linear(d_token, d_arm)

        # The global block: a PMA over the same arm tokens (invariant, so it
        # carries no arm identity either) concatenated with the two per-sim
        # frequency scalars.
        self.global_pma = PMA(
            dim=d_token,
            num_heads=n_heads,
            num_seeds=1,
            ln=attn_ln,
            attn_scale=attn_scale,
        )
        self.global_head = nn.Linear(d_token + N_GLOBAL_SCALARS, d_global)

    def forward(self, X: Tensor) -> Tensor:
        """Embed a batch of sims.

        Args:
            X: ``(B, T, K, 45)`` float32: ``B`` sims, ``T`` trials, ``K``
                clones, 44 arm columns plus a frequency column. NaNs mark
                unfilled trial slots and clone padding.

        Returns:
            ``(B, n_arms * d_arm + d_global)`` float32. The first
            ``n_arms * d_arm`` entries are the per-arm blocks in arm order.
        """
        moments, trial_valid, glob = arm_moments(X, input_space=self.input_space)
        batch = moments.shape[0]

        if self.trial_pool == "attention":
            # (B, T, 44, 8) -> (B * 44, T, d_token): each arm's trials are a
            # set, pooled by one PMA whose weights every arm shares.
            tokens = self.moment_proj(moments)                     # (B,T,44,dt)
            tokens = tokens.permute(0, 2, 1, 3).contiguous()        # (B,44,T,dt)
            tokens = tokens.reshape(batch * self.n_arms, -1, self.d_token)
            # True where a key must be ignored, matching MAB.forward's
            # convention. A sim with no valid trial at all masks every key;
            # MAB un-masks such a row rather than producing an all -inf
            # softmax, and its keys are zeros by then, so the result is finite.
            key_mask = (
                (~trial_valid)
                .unsqueeze(1)
                .expand(batch, self.n_arms, trial_valid.shape[1])
                .reshape(batch * self.n_arms, -1)
            )
            pooled = self.trial_pma(tokens, key_mask=key_mask)[:, 0, :]
            arm_in = pooled.view(batch, self.n_arms, self.d_token)
        else:
            # The masked mean, plus the masked sd so that the between-replicate
            # spread -- which is the only thing 25 trials of the same sim add
            # over one trial -- is not thrown away by the pooling.
            mean_t, sd_t = _masked_mean_sd_over_trials(moments, trial_valid)
            arm_in = torch.cat([mean_t, sd_t], dim=-1)             # (B, 44, 16)

        z = self.arm_mlp(arm_in)                                   # (B, 44, dt)
        for layer in self.arm_layers:
            z = layer(z)
            if self.arm_dropout is not None:
                z = self.arm_dropout(z)

        arm_block = self.arm_head(z)                               # (B, 44, da)

        # The global block. The PMA is permutation-*invariant* over the arms,
        # so nothing about arm order leaks into it; the two scalars are per
        # trial and are averaged over the valid ones.
        valid_f = trial_valid.to(glob.dtype).unsqueeze(-1)         # (B, T, 1)
        glob_mean = (glob * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        global_block = self.global_head(
            torch.cat([self.global_pma(z)[:, 0, :], glob_mean], dim=-1)
        )                                                          # (B, d_global)

        return torch.cat(
            [arm_block.reshape(batch, self.n_arms * self.d_arm), global_block],
            dim=-1,
        )


__all__ = [
    "ArmTokenEmbedding",
    "arm_moments",
    "N_ARMS",
    "N_MOMENTS",
    "N_GLOBAL_SCALARS",
    "MAX_CONTEXT_WIDTH",
    "DEFAULT_D_TOKEN",
    "DEFAULT_D_ARM",
    "DEFAULT_D_GLOBAL",
    "DEFAULT_N_ARM_LAYERS",
    "DEFAULT_ARM_NUM_INDUCING",
    "DEFAULT_N_HEADS",
    "COPY_CUT_LOSS",
    "COPY_CUT_GAIN",
    "COPY_CUT_DEEP_LOSS",
    "LOG2_CUT_LOSS",
    "LOG2_CUT_GAIN",
    "LOG2_CUT_DEEP_LOSS",
]
