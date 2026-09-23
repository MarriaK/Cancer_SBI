"""Training: the loop, the optimiser and the checkpoint files.

Replaces the ``train()`` / ``compute_nltp()`` / ``save_checkpoint()`` trio that
was copied, with small deliberate edits, into three files:

    ==========================================  ==========================
    original                                    new home
    ==========================================  ==========================
    ``Base_NPE/inference_model.py``             :mod:`.trainer`, :mod:`.checkpoints`
    ``SetTransformer_NPE/inference_model.py``   :mod:`.trainer`, :mod:`.checkpoints`
    ``Plain_NPE/model.py``                      :mod:`.trainer`, :mod:`.checkpoints`
    ``*/main.py``                               :mod:`cancer_sbi.cli.train`
    ==========================================  ==========================

There is exactly one :class:`~cancer_sbi.training.trainer.Trainer`. Which of the
three published models it reproduces is decided entirely by the
:class:`~cancer_sbi.config.OptimConfig` and
:class:`~cancer_sbi.config.TrainConfig` it is given, plus the two per-folder
details in :class:`~cancer_sbi.training.trainer.TrainerQuirks`.

Importing this package pulls in torch and sbi; it builds nothing and writes
nothing.
"""

# checkpoints first: trainer imports it, and importing it here before trainer
# keeps that submodule import off the partially-initialised-package path.
from cancer_sbi.training import checkpoints
from cancer_sbi.training.checkpoints import (
    ResumedState,
    build_checkpoint,
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    try_resume,
)
from cancer_sbi.training.trainer import (
    MODEL_QUIRKS,
    Trainer,
    TrainerQuirks,
    TrainingComponents,
    build_embedding_net,
    build_optimizer,
    build_training_components,
    quirks_for,
    seed_everything,
    unpack_batch,
)

__all__ = [
    "checkpoints",
    "ResumedState",
    "build_checkpoint",
    "latest_checkpoint_path",
    "load_checkpoint",
    "save_checkpoint",
    "try_resume",
    "Trainer",
    "TrainerQuirks",
    "TrainingComponents",
    "MODEL_QUIRKS",
    "build_embedding_net",
    "build_optimizer",
    "build_training_components",
    "quirks_for",
    "seed_everything",
    "unpack_batch",
]
