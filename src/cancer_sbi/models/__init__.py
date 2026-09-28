"""Encoders and the conditional flow.

Three trial encoders, one per published model, plus the wrapper and the flow:

* :class:`~cancer_sbi.models.mlp_encoder.BaselineCloneEmbedding` -- CloneMLP-NPE.
* :class:`~cancer_sbi.models.set_transformer.CloneSetEmbedding` (with ``MAB``,
  ``ISAB``, ``PMA``) -- CloneAtt-NPE.
* :class:`~cancer_sbi.models.deep_set.DeepSet` -- DominantClone-NPE; this one is
  the embedding net itself and needs no wrapper.
* :class:`~cancer_sbi.models.arm_tokens.ArmTokenEmbedding` -- ArmToken-NPE
  (matrix 5); likewise the embedding net itself, and deliberately so: the
  wrapper's MLP would blend its per-arm blocks together.
* :class:`~cancer_sbi.models.trials.TrialsSBIEmbedding` -- pools per-trial
  embeddings for the two clone-set models.
* :func:`~cancer_sbi.models.flow.build_flow` -- the ``build_nsf`` call, with
  every architecture argument written out.

Importing this module pulls in torch and sbi; it constructs nothing.
"""

from cancer_sbi.models.arm_tokens import ArmTokenEmbedding, arm_moments
from cancer_sbi.models.deep_set import DeepSet
from cancer_sbi.models.flow import build_flow
from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding
from cancer_sbi.models.set_transformer import ISAB, MAB, PMA, CloneSetEmbedding
from cancer_sbi.models.trials import CloneEncoder, TrialsSBIEmbedding

__all__ = [
    "ArmTokenEmbedding",
    "arm_moments",
    "BaselineCloneEmbedding",
    "CloneSetEmbedding",
    "DeepSet",
    "MAB",
    "ISAB",
    "PMA",
    "TrialsSBIEmbedding",
    "CloneEncoder",
    "build_flow",
]
