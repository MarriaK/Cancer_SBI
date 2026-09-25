"""Hybrid-NPE: ArmToken and CloneAtt side by side, contexts concatenated.

Matrix 6. Matrix 5 established that an arm-equivariant encoder over per-arm
moments (``ArmTokenEmbedding``, run AT0) reaches R^2 0.898, while the best
clone-level attention encoder the first four matrices could find (CloneAtt's
run R26) reaches 0.568. Those are two different summaries of the same
``(B, T, K, 45)`` tensor: the moments reduce the clone axis with eight weighted
statistics per arm, and CloneAtt reduces it with attention over raw clone rows.
The moments are the better summary by a wide margin, but they are *lossy* in a
specific way -- every statistic is taken per arm, so nothing about the joint
pattern of two arms within one clone survives.

This module asks whether that lost joint structure is worth anything. It runs
both encoders on the same input and hands the flow both contexts::

    forward(X) = cat([arm_branch(X), clone_branch(X)], -1)
                    (B, 416)        (B, 256)          -> (B, 672)

It is a **late** fusion on purpose: the two branches share no weights and never
see each other's activations, so the arm branch is bit-for-bit the module
matrix 5 measured, and a hybrid run that scores like AT0 says the clone branch
added nothing rather than leaving it unclear which half was to blame.

The symmetry is correspondingly partial, and that is the honest statement of
what this module is:

* the first ``44 * d_arm`` outputs (352) are exactly arm-equivariant --
  permuting the 44 arm columns of the input permutes those 44 blocks;
* the next ``d_global`` (64) are exactly arm-invariant;
* the last 256 -- the clone branch's context -- are **neither**.
  ``CloneSetEmbedding`` projects all 44 arms together on its first layer, so a
  permutation changes that block arbitrarily. That is the whole point of
  including it, and it is why the arm branch is kept separate rather than the
  two being fused early.

Nothing here runs at import time.
"""

import torch
from torch import Tensor, nn

from cancer_sbi.config import EncoderConfig


class HybridEmbedding(nn.Module):
    """One embedding net owning an ArmToken branch and a full CloneAtt branch.

    Like :class:`~cancer_sbi.models.arm_tokens.ArmTokenEmbedding` and unlike the
    two clone-set encoders, this module is the *whole* embedding net: it takes
    ``(B, T, K, 45)`` and returns ``(B, d_model)``, and
    :func:`~cancer_sbi.training.trainer.build_embedding_net` returns it without
    wrapping it further. The clone branch carries its **own**
    ``TrialsSBIEmbedding`` inside it -- that wrapper is what turns a per-trial
    ``CloneSetEmbedding`` into the 256-wide per-sim context CloneAtt's flow
    sees, so leaving it out would not be "CloneAtt's context" at all.

    Both branches are built by the same three functions
    ``build_embedding_net`` uses for the ``"armtoken"`` and ``"attention"``
    kinds, so a switch added to either encoder reaches this one too and the
    branches cannot drift away from the models they are supposed to reproduce.

    Attributes:
        arm_branch: The ``ArmTokenEmbedding``; ``(B, T, K, 45) -> (B, 416)``.
        clone_branch: The wrapped ``CloneSetEmbedding``;
            ``(B, T, K, 45) -> (B, 256)``.
        d_model: ``arm_width + clone_width`` -- 672 for the ``hybrid`` preset.
            This is the flow's context width.
        arm_width: Width of the arm branch's output, ``44 * d_arm + d_global``.
        clone_width: Width of the clone branch's output, ``trials_output_dim``.
        n_arms / d_arm: Repeated from the arm branch so that a caller can split
            the output into its blocks without reaching into the submodule.
    """

    def __init__(self, cfg: EncoderConfig, device: str = "cpu") -> None:
        """Build both branches from one :class:`EncoderConfig`.

        Args:
            cfg: The encoder settings. This is the one ``kind`` that reads
                *both* field groups: ``d_token``/``d_arm``/``d_global``/
                ``n_arm_layers``/``arm_num_inducing`` size the arm branch and
                ``d_model``/``num_inducing``/``freq_mode``/``freq_renorm`` plus
                the ``trials_*`` block size the clone branch. ``n_heads``,
                ``attn_ln``, ``attn_scale``, ``input_space``, ``dropout``,
                ``attn_dropout_active`` and ``trial_pool`` are read by both, so
                one flag moves both branches -- which is deliberate: a hybrid
                whose two halves disagreed about the input space would be two
                different experiments at once.
            device: Device both branches are moved to.

        Raises:
            ValueError: Whatever either branch raises for its own fields --
                ``d_token`` not divisible by ``n_heads``, a context over the
                cap, an unknown ``input_space``. The clone branch's own
                ``d_model``/``n_heads`` divisibility is checked by
                ``cli/train.py`` before anything is built.

        Note:
            The import below is local because
            :mod:`cancer_sbi.training.trainer` imports *this* module (its
            ``"hybrid"`` branch constructs the class), so a module-level import
            here would close a cycle. By the time any ``HybridEmbedding`` is
            constructed, ``trainer`` is fully imported.
        """
        super().__init__()
        from cancer_sbi.training.trainer import (
            build_arm_token_encoder,
            build_clone_attention_encoder,
            wrap_trial_encoder,
        )

        self.arm_branch = build_arm_token_encoder(cfg, device)
        # Wrapped exactly as build_embedding_net wraps it for kind
        # "attention" -- same helper, same arguments -- so the branch produces
        # the same 256-wide context CloneAtt's flow is given.
        self.clone_branch = wrap_trial_encoder(
            build_clone_attention_encoder(cfg, device), cfg, device
        )

        self.arm_width: int = int(self.arm_branch.d_model)
        # TrialsSBIEmbedding exposes the pooled width under `output_dim`; the
        # config's trials_output_dim is the value it was built from and is the
        # fallback for a wrapper that does not carry the attribute.
        self.clone_width: int = int(
            getattr(self.clone_branch, "output_dim", cfg.trials_output_dim)
        )
        self.d_model: int = self.arm_width + self.clone_width

        # Repeated for callers (and tests) that split the output into blocks.
        self.n_arms: int = int(self.arm_branch.n_arms)
        self.d_arm: int = int(self.arm_branch.d_arm)
        self.d_global: int = int(self.arm_branch.d_global)

    def forward(self, X: Tensor) -> Tensor:
        """Embed a batch of sims with both encoders and concatenate.

        Args:
            X: ``(B, T, K, 45)`` float32: ``B`` sims, ``T`` trials, ``K``
                clones, 44 arm columns plus a frequency column. NaNs mark
                unfilled trial slots and clone padding, and both branches
                handle them themselves -- nothing is masked here.

        Returns:
            ``(B, arm_width + clone_width)`` float32. The first
            ``n_arms * d_arm`` entries are the per-arm blocks in arm order, the
            next ``d_global`` are the arm branch's global block, and the last
            ``clone_width`` are the clone branch's context.
        """
        return torch.cat([self.arm_branch(X), self.clone_branch(X)], dim=-1)


__all__ = ["HybridEmbedding"]
