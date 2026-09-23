"""cancer_sbi: neural posterior estimation of per-arm selection coefficients.

This package is a behaviour-preserving refactor of three near-duplicate pipelines
that live beside it and are *not* modified:

===================  ==================  =========================================
Paper name           Original folder     What it is
===================  ==================  =========================================
CloneMLP-NPE         ``Base_NPE/``       per-clone MLP encoder (the main model)
CloneAtt-NPE         ``SetTransformer_NPE/``  set-transformer (ISAB/PMA) encoder
DominantClone-NPE    ``Plain_NPE/``      DeepSet over the dominant clone per trial
===================  ==================  =========================================

The method: a forward simulator produces, for one simulated tumour ("sim"), a
vector of 44 chromosome-arm selection coefficients (theta) and a set of ~25
independent evolutionary replicates ("trials"). Each trial is summarised as a set
of clonal copy-number-alteration profiles. A permutation-invariant embedding net
turns that nested set (trials of clones) into a fixed-length context vector, and a
neural spline flow (sbi's ``build_nsf``) is trained by maximum likelihood to
approximate p(theta | x). The three models differ only in how a single trial is
embedded.

**Behaviour is frozen.** Every quirk of the originals that affects a number in the
paper is reproduced here verbatim and carries a ``# Preserved from <file>:<line>``
comment. See ``docs/REFACTOR_NOTES.md`` for the full list and why each one stays.

Nothing in this package has module-level side effects: importing it builds no
model, reads no file and mutates no global state.
"""

__version__ = "0.1.0"

# Only the torch-free label table is re-exported here, so that
# ``import cancer_sbi`` stays cheap and usable without torch/sbi installed.
from cancer_sbi.arms import ARM_LABELS

__all__ = ["ARM_LABELS", "__version__"]
