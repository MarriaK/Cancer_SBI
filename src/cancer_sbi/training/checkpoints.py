"""Checkpoint writing, loading and resuming.

Ported from the three near-identical implementations:

* ``Base_NPE/inference_model.py:118-173``
* ``SetTransformer_NPE/inference_model.py:117-179``
* ``Plain_NPE/model.py:73-128``

The three differed only in the default value of the history key (trap 14) and in
the directory they wrote to (trap 11); the payload, the file names and the
resume logic were identical, so they collapse into the functions below without
any behaviour change.

A checkpoint is a plain ``dict`` with these nine keys, in this order::

    epoch                          int, 1-based, the epoch just finished
    model_state                    density_estimator.state_dict()
    optimizer_state                optimizer.state_dict()
    best_val_loss                  float
    best_model_state_dict          state dict or None (see Trainer, trap 12)
    history                        {"training_loss": [...], <val key>: [...]}
    epochs_since_last_improvement  int, the early-stopping counter
    rng_state                      torch.get_rng_state()
    cuda_rng_state_all             torch.cuda.get_rng_state_all() or None

plus, since 2026-09-24, a tenth::

    effective_config               cancer_sbi.config.config_to_dict(cfg)

and an eleventh, written only by a ``--lr-plateau`` run (matrix 6)::

    scheduler_state                ReduceLROnPlateau.state_dict()

without which a resumed run would restart at the full learning rate and lose
every halving the first attempt had earned.

which is the config the run was *actually* trained with, flags folded in. It is
what lets evaluation rebuild the same network instead of guessing at
``get_preset`` defaults -- see :func:`read_effective_config`. Every checkpoint
written before that date lacks it, so it is optional on both sides.

Three files are written per epoch, all with the same payload:
``ckpt_epoch_<epoch:04d>.pt`` (the per-epoch archive), ``latest.pt`` (what a
resume reads) and, on an improvement, ``best.pt`` (what evaluation reads).

Nothing here runs at import time.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch

from cancer_sbi.config import EFFECTIVE_CONFIG_KEY

PathLike = Union[str, os.PathLike]

#: The keys every checkpoint carries, in the order the originals wrote them.
#: Kept as data so that :mod:`cancer_sbi.evaluation.posterior` and any future
#: inspection tool agree with the writer about the payload.
CHECKPOINT_KEYS = (
    "epoch",
    "model_state",
    "optimizer_state",
    "best_val_loss",
    "best_model_state_dict",
    "history",
    "epochs_since_last_improvement",
    "rng_state",
    "cuda_rng_state_all",
)

#: The optional tenth key, written since 2026-09-24. Kept out of
#: :data:`CHECKPOINT_KEYS` because that tuple documents the payload the three
#: originals wrote, and a checkpoint without this key is still a valid one.
#: Key under which a ``--lr-plateau`` run stores its scheduler state.
SCHEDULER_STATE_KEY = "scheduler_state"

OPTIONAL_CHECKPOINT_KEYS = (EFFECTIVE_CONFIG_KEY, SCHEDULER_STATE_KEY)

#: File name of the pointer a resume reads.
LATEST_FILENAME = "latest.pt"

#: File name of the pointer evaluation reads.
BEST_FILENAME = "best.pt"


class ResumedState(NamedTuple):
    """Everything a checkpoint restores that is not a module's own state.

    ``model_state`` and ``optimizer_state`` are loaded straight into the objects
    passed to :func:`load_checkpoint`, so they are not repeated here.

    Attributes:
        epoch: Last finished epoch, 1-based. Training continues from
            ``epoch + 1``.
        best_val_loss: Lowest validation loss seen so far.
        best_model_state_dict: Snapshot of the weights that achieved it, or
            ``None`` if no epoch has improved yet.
        history: ``{"training_loss": [...], <history_val_key>: [...]}``.
        epochs_since_last_improvement: The early-stopping counter as stored.
        path: The file this state came from.
        scheduler_state: The LR scheduler's ``state_dict()``, or ``None`` for a
            checkpoint written without ``--lr-plateau`` (which is every
            checkpoint before matrix 6). Returned rather than loaded here
            because :func:`load_checkpoint` is given a model and an optimizer,
            not a scheduler; :class:`~cancer_sbi.training.trainer.Trainer`
            applies it to whichever scheduler it built.
    """

    epoch: int
    best_val_loss: float
    best_model_state_dict: Optional[Dict[str, torch.Tensor]]
    history: Dict[str, List[float]]
    epochs_since_last_improvement: int
    path: Path
    scheduler_state: Optional[Dict[str, Any]] = None


def latest_checkpoint_path(ckpt_dir: PathLike) -> Optional[Path]:
    """Return ``<ckpt_dir>/latest.pt`` if it exists.

    Args:
        ckpt_dir: Directory holding the checkpoints of one run.

    Returns:
        The path, or ``None`` when there is nothing to resume from.

    Note:
        Preserved from ``Base_NPE/inference_model.py:118-120``: existence is the
        only test. A truncated or unreadable ``latest.pt`` still counts as
        "found" here and is caught later by :func:`try_resume`.
    """
    latest = Path(ckpt_dir) / LATEST_FILENAME
    return latest if latest.exists() else None


def build_checkpoint(
    epoch: int,
    density_estimator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    best_model_state_dict: Optional[Dict[str, torch.Tensor]],
    history: Dict[str, List[float]],
    epochs_since_last_improvement: int,
    effective_config: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the checkpoint payload.

    Args:
        epoch: Epoch just finished, 1-based.
        density_estimator: The sbi flow; its ``state_dict()`` includes the
            embedding net's parameters, because the embedding net is a submodule
            of the flow.
        optimizer: The Adam instance, including its per-group learning rates and
            its exponential-moving-average buffers.
        best_val_loss: Lowest validation loss so far; cast to ``float`` because
            the originals did and because ``np.inf`` must survive the round trip.
        best_model_state_dict: Best weights so far, or ``None``.
        history: The loss curves.
        epochs_since_last_improvement: Early-stopping counter; cast to ``int``.
        effective_config: :func:`cancer_sbi.config.config_to_dict` of the config
            this run is training with, flags folded in. ``None`` omits the key
            entirely, which is what a checkpoint written before 2026-09-24 looks
            like.
        scheduler_state: ``ReduceLROnPlateau.state_dict()`` for a
            ``--lr-plateau`` run (matrix 6). ``None`` omits the key, which is
            what a run with no scheduler writes.

    Returns:
        A ``dict`` with the keys listed in :data:`CHECKPOINT_KEYS`, plus
        ``effective_config`` when one was given.

    Note:
        The RNG states are captured here, at save time, exactly as in
        ``Base_NPE/inference_model.py:131-132``. ``cuda_rng_state_all`` is
        ``None`` on a machine without CUDA, which is why every reader has to
        guard on it.
    """
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": density_estimator.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": float(best_val_loss),
        "best_model_state_dict": best_model_state_dict,
        "history": history,
        "epochs_since_last_improvement": int(epochs_since_last_improvement),
        # Preserved from Base_NPE/inference_model.py:131-132. The originals
        # called these "optional reproducibility"; nothing ever seeded the RNG
        # (trap 21), so what is stored is whatever state the process happened to
        # be in. It is still restored on resume, so a resumed run is
        # reproducible from its own checkpoint even though a fresh run is not.
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    if effective_config is not None:
        payload[EFFECTIVE_CONFIG_KEY] = effective_config
    if scheduler_state is not None:
        payload[SCHEDULER_STATE_KEY] = scheduler_state
    return payload


def read_effective_config(
    path: PathLike, device: str = "cpu"
) -> Optional[Dict[str, Any]]:
    """Read only the config a checkpoint was written with.

    Args:
        path: The ``.pt`` file.
        device: ``map_location`` for :func:`torch.load`.

    Returns:
        The stored ``effective_config`` dict, or ``None`` for a checkpoint
        written before 2026-09-24 (which is every checkpoint currently on the
        cluster). ``None`` means "fall back to the preset and say so", never
        "assume the defaults were used".

    Note:
        ``weights_only`` is left at its default for the same reason as
        :func:`load_checkpoint`: these payloads hold numpy scalars, which the
        safe loader refuses. The whole file is read, because ``torch.load``
        cannot fetch one key -- for these models that is a few MB.
    """
    ckpt = torch.load(path, map_location=device)
    stored = ckpt.get(EFFECTIVE_CONFIG_KEY)
    return stored if isinstance(stored, dict) else None


def save_checkpoint(
    ckpt_dir: PathLike,
    epoch: int,
    payload: Dict[str, Any],
    is_best: bool = False,
) -> Path:
    """Write one checkpoint to the three file names the originals used.

    Args:
        ckpt_dir: Destination directory; created if missing.
        epoch: Epoch number used in the per-epoch file name.
        payload: The dict from :func:`build_checkpoint`.
        is_best: Also write ``best.pt``. Evaluation reads only that file.

    Returns:
        The path of the per-epoch file.

    Note:
        Preserved from ``Base_NPE/inference_model.py:135-143``: the same payload
        is written up to three times rather than hard-linked or copied, and the
        per-epoch file is never pruned. A 200-epoch run leaves 202 files behind.
        This is wasteful but it is what produced the runs on the cluster, and a
        rolling-window policy would delete files the author may still have.
    """
    out_dir = Path(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_path = out_dir / f"ckpt_epoch_{epoch:04d}.pt"
    torch.save(payload, epoch_path)

    torch.save(payload, out_dir / LATEST_FILENAME)

    if is_best:
        torch.save(payload, out_dir / BEST_FILENAME)

    return epoch_path


def load_checkpoint(
    path: PathLike,
    density_estimator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    history_val_key: str = "validation_loss",
    verbose: bool = True,
) -> ResumedState:
    """Restore a checkpoint into a model and optimizer.

    Args:
        path: The ``.pt`` file to read.
        density_estimator: Model to load ``model_state`` into. ``load_state_dict``
            is strict, so the architecture must match the one that was saved.
        optimizer: Optimizer to load ``optimizer_state`` into. Its parameter
            groups must match the saved ones, which is why a checkpoint from a
            two-group run cannot be loaded into a one-group optimizer.
        device: ``map_location`` for :func:`torch.load`.
        history_val_key: Key used when a checkpoint has no ``history`` at all --
            ``"validation_loss"`` for CloneMLP/CloneAtt, ``"test_loss"`` for
            DominantClone. Trap 14.
        verbose: Print the ``[checkpoint] Resumed from ...`` line, as the
            originals unconditionally did.

    Returns:
        A :class:`ResumedState`.

    Raises:
        RuntimeError: If the state dicts do not match the given objects.
        KeyError: If ``model_state`` or ``optimizer_state`` is missing.

    Note:
        Preserved from ``Base_NPE/inference_model.py:145-161``. Two details are
        load-bearing. First, ``torch.load`` is called without
        ``weights_only=True``: these checkpoints hold numpy scalars and plain
        Python containers, and newer torch refuses them under the safe loader.
        Second, the RNG state is restored *if present*, which is why every
        evaluation script strips those keys before loading a GPU checkpoint on a
        CPU machine -- see
        :func:`cancer_sbi.evaluation.posterior.load_checkpoint_for_eval`.
    """
    ckpt = torch.load(path, map_location=device)

    density_estimator.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    epoch = int(ckpt.get("epoch", 0))
    best_val_loss = float(ckpt.get("best_val_loss", np.inf))
    best_model_state_dict = ckpt.get("best_model_state_dict", None)
    # Preserved from Base_NPE/inference_model.py:153 and Plain_NPE/model.py:108.
    # Trap 14: the fallback history carries the model's own validation key, so a
    # CloneMLP checkpoint falls back to "validation_loss" and a DominantClone one
    # to "test_loss". utilities/plot_losses.py:30-44 identifies the folder a
    # history came from by exactly this key. This looks like something to unify
    # but it is what the published models do; renaming it breaks those plots and
    # every checkpoint already on the cluster. See docs/REFACTOR_NOTES.md.
    history = ckpt.get("history", {"training_loss": [], history_val_key: []})
    epochs_since_last_improvement = int(ckpt.get("epochs_since_last_improvement", 0))
    # Matrix 6: absent from every checkpoint written without --lr-plateau, so
    # the fallback is None and the Trainer simply leaves its scheduler fresh.
    scheduler_state = ckpt.get(SCHEDULER_STATE_KEY)
    if not isinstance(scheduler_state, dict):
        scheduler_state = None

    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

    if verbose:
        print(f"[checkpoint] Resumed from {path} (epoch={epoch})")

    return ResumedState(
        epoch=epoch,
        best_val_loss=best_val_loss,
        best_model_state_dict=best_model_state_dict,
        history=history,
        epochs_since_last_improvement=epochs_since_last_improvement,
        path=Path(path),
        scheduler_state=scheduler_state,
    )


def try_resume(
    ckpt_dir: PathLike,
    density_estimator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    history_val_key: str = "validation_loss",
) -> Optional[ResumedState]:
    """Resume from ``<ckpt_dir>/latest.pt`` if that is possible.

    Args:
        ckpt_dir: Directory to look in.
        density_estimator: Model to restore into.
        optimizer: Optimizer to restore into.
        device: ``map_location``.
        history_val_key: See :func:`load_checkpoint`.

    Returns:
        The :class:`ResumedState`, or ``None`` when there was no checkpoint or
        it could not be read.

    Note:
        Preserved from ``Base_NPE/inference_model.py:163-173``: **every**
        exception is caught, reported as a message and turned into "start from
        scratch". That is deliberate and it is also the single most dangerous
        line in the original pipeline -- a checkpoint saved by a different
        architecture, or one truncated by a killed job, silently restarts
        training at epoch 0 instead of stopping. It is preserved because a run
        that used to survive a bad checkpoint must still survive it; the message
        it prints is the only warning there is. See docs/REFACTOR_NOTES.md.
    """
    latest = latest_checkpoint_path(ckpt_dir)
    if latest is None:
        print("[checkpoint] No checkpoint found. Starting from scratch.")
        return None
    try:
        return load_checkpoint(
            latest,
            density_estimator,
            optimizer,
            device=device,
            history_val_key=history_val_key,
        )
    except Exception as exc:  # noqa: BLE001 - see the Note above.
        print(f"[checkpoint] Found checkpoint but failed to load: {exc}")
        return None


__all__ = [
    "CHECKPOINT_KEYS",
    "OPTIONAL_CHECKPOINT_KEYS",
    "SCHEDULER_STATE_KEY",
    "read_effective_config",
    "LATEST_FILENAME",
    "BEST_FILENAME",
    "ResumedState",
    "latest_checkpoint_path",
    "build_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "try_resume",
]
