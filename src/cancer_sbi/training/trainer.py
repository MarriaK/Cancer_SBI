"""One training loop for all three published models.

Ported from three files that are 90% the same and 10% deliberately different:

* ``Base_NPE/inference_model.py`` -- ``InferenceModelBaseline`` (CloneMLP-NPE)
* ``SetTransformer_NPE/inference_model.py`` -- ``InferenceModel`` (CloneAtt-NPE)
* ``Plain_NPE/model.py`` -- ``InferenceModel`` (DominantClone-NPE)

Everything that differs between them is a field of
:class:`~cancer_sbi.config.OptimConfig` or
:class:`~cancer_sbi.config.TrainConfig`, so :class:`Trainer` itself contains no
``if model == ...`` branch. The three differences that have no config field --
because :mod:`cancer_sbi.config` records the differences the author knows about,
and these two were found while porting -- live in :class:`TrainerQuirks`.

What the loop does, in one paragraph: for each epoch, run the training set once
(forward, backward, optional gradient clipping, Adam step), accumulate the
per-sample losses, then run the validation set under ``no_grad`` and average the
same way. If the validation average is strictly lower than the best so far,
snapshot the weights and write ``best.pt``; otherwise increment a patience
counter. Stop when the counter reaches ``stop_after_epochs`` (and, for two of the
three models, only once ``min_epochs`` have passed), or when ``max_epochs`` is
reached. What happens to the weights at the end depends on ``reload_best``.

Nothing here runs at import time.
"""

import logging
import os
import pickle as pkl
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from cancer_sbi.config import (
    DatasetKind,
    EncoderConfig,
    ModelPreset,
    OptimConfig,
    TrainConfig,
)
from cancer_sbi.models.deep_set import DeepSet
from cancer_sbi.models.flow import build_flow
from cancer_sbi.models.mlp_encoder import BaselineCloneEmbedding
from cancer_sbi.models.set_transformer import CloneSetEmbedding
from cancer_sbi.models.trials import TrialsSBIEmbedding
from cancer_sbi.training import checkpoints

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Per-folder details that config.py does not (yet) record.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainerQuirks:
    """Two per-folder behaviours that have no field in :mod:`cancer_sbi.config`.

    Both were found while porting the three ``train()`` methods line by line.
    They are kept here rather than being "tidied away" because each one is
    visible in a saved checkpoint, and because unifying them would change what a
    resumed or reloaded run does. See docs/REFACTOR_NOTES.md, issues 22 and 23.

    Attributes:
        best_state_on_device: How the best-so-far weights are snapshotted.
            ``False`` -> ``{k: v.detach().cpu() for ...}``
            (``Base_NPE/inference_model.py:232``, ``Plain_NPE/model.py:174``).
            ``True`` -> ``deepcopy(state_dict())``
            (``SetTransformer_NPE/inference_model.py:229``), which keeps the
            snapshot on whatever device the model is on.
        save_before_counter_update: Whether ``save_checkpoint`` runs *before*
            the patience counter is copied onto ``self``. ``True`` for CloneMLP
            (``Base_NPE/inference_model.py:234,237`` then ``:239``) and
            DominantClone (``Plain_NPE/model.py:176,179`` then ``:181``), so the
            counter stored in the checkpoint is the one from the *previous*
            epoch. ``False`` for CloneAtt
            (``SetTransformer_NPE/inference_model.py:235-236``), which stores the
            current value.
    """

    best_state_on_device: bool = False
    save_before_counter_update: bool = True


#: The quirks of each published model, keyed by ``ModelPreset.name``.
MODEL_QUIRKS: Dict[str, TrainerQuirks] = {
    # Base_NPE/inference_model.py:232 (.detach().cpu()) and :234-239 (save first).
    "clonemlp": TrainerQuirks(best_state_on_device=False, save_before_counter_update=True),
    # SetTransformer_NPE/inference_model.py:229 (deepcopy) and :235-236 (count first).
    "cloneatt": TrainerQuirks(best_state_on_device=True, save_before_counter_update=False),
    # Plain_NPE/model.py:174 (.detach().cpu()) and :176-181 (save first).
    "dominantclone": TrainerQuirks(best_state_on_device=False, save_before_counter_update=True),
}


def quirks_for(preset_name: str) -> TrainerQuirks:
    """Look up the :class:`TrainerQuirks` of a published model.

    Args:
        preset_name: ``"clonemlp"``, ``"cloneatt"`` or ``"dominantclone"``.

    Returns:
        The quirks of that folder. Unknown names get the CloneMLP/DominantClone
        behaviour, which is the majority of the three.
    """
    return MODEL_QUIRKS.get(preset_name.strip().lower(), TrainerQuirks())


# ---------------------------------------------------------------------------
# Seeding (trap 21).
# ---------------------------------------------------------------------------


def seed_everything(seed: Optional[int]) -> None:
    """Seed torch, CUDA, numpy and Python's ``random`` -- if a seed was asked for.

    Args:
        seed: The seed, or ``None`` to leave every RNG untouched.

    Note:
        Preserved from all three originals, none of which ever calls
        ``torch.manual_seed`` or ``numpy.random.seed``. Trap 21: training is
        unseeded, so weight initialisation, dropout masks and the training
        loader's shuffling differ between runs, and the published numbers come
        from one particular unrepeatable run. ``seed=None`` (the default in
        :class:`~cancer_sbi.config.TrainConfig`) reproduces exactly that. Passing
        a seed is a new capability for future runs; it does not reproduce an old
        one. See docs/REFACTOR_NOTES.md.
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    # Added 2026-09-24 (WP-A). Nothing in the published loop draws from Python's
    # own RNG, so seeding it cannot change a published number; it is here so
    # that anything added later (a shuffle, a subsample) is covered too.
    random.seed(seed)


# ---------------------------------------------------------------------------
# Batch handling. The two dataset families yield different tuples.
# ---------------------------------------------------------------------------


def unpack_batch(
    batch: Any,
    dataset: DatasetKind,
    device: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Turn one loader batch into ``(theta, condition)`` on ``device``.

    Args:
        batch: Whatever the loader yielded.
        dataset: ``"clone_sets"`` -> the 3-tuple ``(X, trial_mask, theta)``;
            ``"dominant_clone"`` -> the 2-tuple ``(theta, x)``, or ``None``.
        device: Target device for both tensors.

    Returns:
        ``(theta, condition)``, or ``None`` when the whole batch was dropped by
        :func:`cancer_sbi.data.dominant_clone.collate_skip_none`.

    Note:
        Preserved from ``Plain_NPE/model.py:146-147`` and ``:195-196``: the
        ``if batch is None: continue`` guard exists **only** on the
        dominant-clone path, because only that path has a collate function that
        can return ``None``. The clone-set path
        (``Base_NPE/inference_model.py:205``) unpacks the tuple unconditionally
        and would raise on a ``None`` batch. Adding the guard there too would be
        harmless today and is still not done: a ``None`` batch from
        ``CNASimsDataset`` would mean something is badly wrong, and the original
        crashes rather than silently training on fewer sims.

        The middle element of the clone-set tuple -- ``trial_mask`` -- is dropped
        here, exactly as ``Base_NPE/inference_model.py:205`` drops it into ``_``.
        Trap 9: it is computed and returned by the dataset and never used by
        anything. It stays part of the tuple because the tuple shape is the
        contract between the dataset and every script that reads it.
    """
    if dataset == "dominant_clone":
        if batch is None:
            return None
        theta_batch, x_batch = batch
        return theta_batch.to(device), x_batch.to(device)

    x_batch, _, theta_batch = batch
    return theta_batch.to(device), x_batch.to(device)


# ---------------------------------------------------------------------------
# Optimiser (traps 2, 3).
# ---------------------------------------------------------------------------


def build_optimizer(
    density_estimator: torch.nn.Module,
    embedding_net: Optional[torch.nn.Module],
    cfg: OptimConfig,
) -> torch.optim.Adam:
    """Build the Adam instance the configured model was trained with.

    Two shapes are possible and they are not interchangeable:

    * **Two groups** (``use_param_groups=True``, CloneMLP and CloneAtt): the
      flow's parameters get ``lr=1e-3, weight_decay=1e-4`` and the embedding
      net's get ``lr=1e-4, weight_decay=0.0``.
    * **One group** (``use_param_groups=False``, DominantClone): every parameter
      gets ``learning_rate`` and ``weight_decay``.

    Args:
        density_estimator: The sbi flow. Its ``parameters()`` already include the
            embedding net's, because the embedding net is a submodule.
        embedding_net: The embedding net, needed only in two-group mode to
            separate its parameters out again. May be ``None`` in one-group mode.
        cfg: The optimiser settings, normally ``preset.optim``.

    Returns:
        A configured :class:`torch.optim.Adam`.

    Raises:
        ValueError: If ``use_param_groups`` and ``learning_rate_is_used``
            contradict each other, or if two-group mode is requested without an
            embedding net. Neither can happen with the three shipped presets;
            the checks exist so a hand-built config fails loudly instead of
            silently training with the wrong learning rate.
    """
    if cfg.use_param_groups:
        # Preserved from Base_NPE/inference_model.py:58 and
        # SetTransformer_NPE/inference_model.py:49, which store `learning_rate`
        # on self and never read it again. Trap 3: `learning_rate` is DEAD for
        # these two models -- the rates that matter are the per-group ones below.
        # This looks wrong but it is what the published models do; wiring it up
        # would change the results. See docs/REFACTOR_NOTES.md.
        if cfg.learning_rate_is_used:
            raise ValueError(
                "use_param_groups=True means the two group learning rates are "
                "used and OptimConfig.learning_rate is ignored (trap 3), but "
                "learning_rate_is_used=True says the opposite. Set "
                "learning_rate_is_used=False, or use_param_groups=False."
            )
        if embedding_net is None:
            raise ValueError(
                "use_param_groups=True needs the embedding net so its "
                "parameters can be put in their own group."
            )

        # Preserved verbatim from Base_NPE/inference_model.py:102-104 (identical
        # to SetTransformer_NPE/inference_model.py:97-101). The split is by
        # object identity, not by name: the embedding net is a submodule of the
        # flow, so `density_estimator.parameters()` yields its tensors too, and
        # `id(p)` is what keeps them out of the flow group. Any rewrite using
        # named_parameters() and a name prefix would be a different split.
        embed_params = list(embedding_net.parameters())
        embed_ids = {id(p) for p in embed_params}
        flow_params = [
            p for p in density_estimator.parameters() if id(p) not in embed_ids
        ]

        # Preserved from Base_NPE/inference_model.py:106-113. Trap 2:
        # SetTransformer_NPE/inference_model.py:94 keeps the single-group version
        # of this line commented out with the note "this makes overfitting", so
        # the two groups are a deliberate, load-bearing choice. Collapsing them
        # into one group changes the results. See docs/REFACTOR_NOTES.md.
        return torch.optim.Adam(
            [
                # Flow: normal LR, small weight decay.
                {
                    "params": flow_params,
                    "lr": cfg.flow_lr,
                    "weight_decay": cfg.flow_weight_decay,
                },
                # Embedding: smaller LR, no weight decay.
                {
                    "params": embed_params,
                    "lr": cfg.embed_lr,
                    "weight_decay": cfg.embed_weight_decay,
                },
            ],
            betas=cfg.betas,
            eps=cfg.eps,
        )

    # One group. Preserved from Plain_NPE/model.py:67, which passes
    # `list(self.density_estimator.parameters())` -- i.e. the embedding net's
    # parameters get the same learning rate and the same weight decay as the
    # flow -- and leaves betas and eps at torch's defaults ((0.9, 0.999), 1e-8).
    # Those happen to equal the values the two-group models pass explicitly, so
    # `cfg.betas` and `cfg.eps` are deliberately NOT forwarded here: the original
    # call did not pass them, and passing them would only look different.
    if not cfg.learning_rate_is_used:
        raise ValueError(
            "use_param_groups=False means OptimConfig.learning_rate IS the "
            "learning rate, but learning_rate_is_used=False says it is dead "
            "(trap 3). Set learning_rate_is_used=True for a single-group model."
        )
    return torch.optim.Adam(
        list(density_estimator.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


# ---------------------------------------------------------------------------
# Model assembly: what each original __init__ did before the loop started.
# ---------------------------------------------------------------------------


class TrainingComponents(NamedTuple):
    """The three objects a :class:`Trainer` needs.

    Attributes:
        embedding_net: The module whose parameters form the second optimiser
            group. ``TrialsSBIEmbedding`` for the clone-set models, ``DeepSet``
            for DominantClone.
        density_estimator: The sbi flow, with ``embedding_net`` inside it.
        example_batch: The ``(theta, condition)`` pair drawn from the training
            loader to shape the flow. Returned so callers can print or check it.
    """

    embedding_net: torch.nn.Module
    density_estimator: torch.nn.Module
    example_batch: Tuple[torch.Tensor, torch.Tensor]


def build_embedding_net(cfg: EncoderConfig, device: str) -> torch.nn.Module:
    """Build the embedding net described by an :class:`EncoderConfig`.

    Args:
        cfg: The encoder settings, normally ``preset.encoder``.
        device: Device the module is moved to. DominantClone's DeepSet is the
            one exception -- see the note.

    Returns:
        ``TrialsSBIEmbedding`` wrapping a per-trial clone encoder (``"mlp"`` or
        ``"attention"``), or a bare ``DeepSet`` (``"deepset"``).

    Raises:
        ValueError: On an unknown ``cfg.kind``.

    Note:
        Preserved from ``Plain_NPE/model.py:55``, which constructs ``DeepSet``
        on the CPU and never moves it: the module reaches the GPU only later, as
        a submodule, when ``self.density_estimator.to(self.device)`` runs
        (``Plain_NPE/model.py:65``). The two clone-set models move their encoder
        and wrapper immediately (``Base_NPE/inference_model.py:75,84``). The end
        state is the same; the order is preserved so that anyone comparing the
        two files sees the same sequence of calls.
    """
    if cfg.kind == "mlp":
        # Base_NPE/inference_model.py:67-75.
        trial_encoder: torch.nn.Module = BaselineCloneEmbedding(
            in_dim=cfg.in_dim,
            d_model=cfg.d_model,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            # Genuinely used by this encoder, unlike CloneAtt's (trap 4).
            dropout=cfg.dropout,
            freq_as_weight=cfg.freq_as_weight,
            include_freq_in_mlp=cfg.include_freq_in_mlp,
            # Repair T2 / run R2 (WP-A + WP-C). "log2" is the published
            # behaviour and the default, so this argument changes nothing
            # unless --input-space copy is passed.
            input_space=cfg.input_space,
            # Matrix 3 / run R10. "weight" is the published behaviour for this
            # encoder too, so this argument changes nothing unless
            # --freq-mode feature is passed. Forwarded here, not only at
            # training time: evaluation rebuilds the encoder from the
            # checkpoint's effective config through this same function, and a
            # missing argument would rebuild a 44-input MLP for a 45-input
            # state_dict.
            freq_mode=cfg.freq_mode,
        ).to(device)
    elif cfg.kind == "attention":
        # SetTransformer_NPE/inference_model.py:63-71. Trap 4: `dropout` is
        # accepted by CloneSetEmbedding and never used
        # (SetTransformer_NPE/set_transformer.py:88), so CloneAtt has no encoder
        # dropout at all. EncoderConfig.encoder_dropout_is_used records that.
        # This looks wrong but it is what the published model does; adding the
        # dropout would change the results. See docs/REFACTOR_NOTES.md.
        trial_encoder = CloneSetEmbedding(
            in_dim=cfg.in_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            num_layers=cfg.num_layers,
            num_inducing=cfg.num_inducing,
            dropout=cfg.dropout,
            freq_as_weight=cfg.freq_as_weight,
            # Run R4 (WP-A + WP-C). False is the published raw-frequency
            # multiply, so this argument changes nothing unless --freq-renorm
            # is passed. Deliberately NOT forwarded to DeepSet below, which has
            # no per-clone frequency weighting at all.
            freq_renorm=cfg.freq_renorm,
            # Matrix 2 (runs R5-R8). Every one of these four defaults to the
            # published behaviour, so an untouched cloneatt preset builds the
            # same network it always did. They must be forwarded here and not
            # only at training time: evaluation rebuilds the encoder from the
            # checkpoint's effective config through this same function, and a
            # missing argument would silently rebuild a different network --
            # the failure mode M1 exists to prevent.
            input_space=cfg.input_space,
            freq_mode=cfg.freq_mode,
            attn_ln=cfg.attn_ln,
            attn_dropout_active=cfg.attn_dropout_active,
            # Matrix 4 / run R17, trap 6 made opt-out. "published" is the
            # published sqrt(d_model) divisor, so this argument changes nothing
            # unless --attn-scale standard is passed. Forwarded here for the
            # same reason as the four above: evaluation rebuilds through this
            # function, and a missing argument would score a run as a network
            # it never was.
            attn_scale=cfg.attn_scale,
        ).to(device)
    elif cfg.kind == "deepset":
        # Plain_NPE/model.py:55 passes only the three dimensions; every other
        # argument keeps the class default from Plain_NPE/net_builder.py:14-24.
        # Not moved to `device` here -- see the note in this docstring.
        return DeepSet(
            aggregation_fn=cfg.aggregation_fn,
            aggregation_dim=cfg.aggregation_dim,
            input_dim=cfg.input_dim,
            hidden_dim_phi=cfg.hidden_dim_phi,
            hidden_dim_rho=cfg.hidden_dim_rho,
            num_layers=cfg.num_layers,
            output_dim=cfg.output_dim,
            num_heads=cfg.num_heads_deepset,
        )
    else:
        raise ValueError(f"Unknown encoder kind: {cfg.kind!r}")

    # Base_NPE/inference_model.py:78-84 and
    # SetTransformer_NPE/inference_model.py:74-80 -- byte-identical calls.
    return TrialsSBIEmbedding(
        trial_encoder=trial_encoder,
        aggregation_fn=cfg.trials_aggregation_fn,
        num_hiddens=cfg.trials_num_hiddens,
        num_layers=cfg.trials_num_layers,
        output_dim=cfg.trials_output_dim,
        # Matrix 4 / run R20. "mean" is the published pooling, so these four
        # arguments change nothing unless --trial-pool attention is passed.
        # The three attention settings follow the encoder's so that the
        # pooling PMA matches the stack under it; n_heads is None for the MLP
        # encoder, where TrialsSBIEmbedding falls back to its own default.
        trial_pool=cfg.trial_pool,
        n_heads=cfg.n_heads,
        attn_ln=cfg.attn_ln,
        attn_scale=cfg.attn_scale,
    ).to(device)


def build_training_components(
    preset: ModelPreset,
    train_loader: DataLoader,
    device: str = "cpu",
    log_progress: bool = True,
) -> TrainingComponents:
    """Assemble encoder + flow the way the original ``__init__`` did.

    The flow's input and context shapes are inferred by sbi from one real batch,
    so this function consumes the first batch of ``train_loader`` exactly as
    ``Base_NPE/inference_model.py:87`` and ``Plain_NPE/model.py:57`` did.

    Args:
        preset: The model to build.
        train_loader: Training loader; its first batch shapes the flow.
        device: ``"cuda"`` or ``"cpu"``.
        log_progress: Print the one-line build summary.

    Returns:
        A :class:`TrainingComponents`.

    Note:
        This is also where ``TrainConfig.seed`` is applied, because seeding has
        to happen before any weight is initialised. With the default
        ``seed=None`` nothing is seeded and the behaviour is the original's
        (trap 21).

        ``next(iter(train_loader))`` on the dominant-clone path is deliberately
        **not** guarded against a ``None`` batch, matching
        ``Plain_NPE/model.py:57``. If the first batch of the epoch happened to be
        entirely unusable, the original crashed here, and so does this.
    """
    seed_everything(preset.train.seed)

    embedding_net = build_embedding_net(preset.encoder, device)

    # Base_NPE/inference_model.py:87-89 / Plain_NPE/model.py:57-58.
    example = next(iter(train_loader))
    unpacked = unpack_batch(example, preset.data.dataset, device)
    if unpacked is None:
        raise RuntimeError(
            "The first training batch was empty, so the flow's input shapes "
            "cannot be inferred. The original code raised here too."
        )
    theta_batch, condition_batch = unpacked

    if log_progress:
        # Plain_NPE/model.py:60-63 printed the devices and sizes of this batch.
        # The z-scoring modes are printed with them because they are the single
        # most surprising per-model difference (trap 1) and are otherwise
        # invisible in the output of a run.
        print(
            f"[build] {preset.paper_name} ({preset.origin}) on {device}: "
            f"theta {tuple(theta_batch.shape)}, "
            f"condition {tuple(condition_batch.shape)}, "
            f"z_score_x={preset.flow.z_score_x}, z_score_y={preset.flow.z_score_y}"
        )

    density_estimator = build_flow(
        theta_batch, condition_batch, embedding_net, preset.flow
    ).to(device)

    return TrainingComponents(
        embedding_net=embedding_net,
        density_estimator=density_estimator,
        example_batch=(theta_batch, condition_batch),
    )


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


class Trainer:
    """Train one density estimator, with early stopping and checkpointing.

    Construction has a side effect, inherited from all three originals: it
    creates ``ckpt_dir`` and then tries to resume from ``<ckpt_dir>/latest.pt``.
    A leftover checkpoint from a previous run is therefore picked up silently.
    See :func:`cancer_sbi.training.checkpoints.try_resume`.

    Attributes:
        density_estimator: The flow being trained.
        embedding_net: Its embedding submodule, or ``None``.
        optimizer: The Adam instance from :func:`build_optimizer`.
        ckpt_dir: Directory the checkpoints go to.
        epoch: Epochs finished so far (restored from a checkpoint if resuming).
        best_val_loss: Lowest validation loss so far.
        best_model_state_dict: Weights that achieved it, or ``None``.
        history: ``{"training_loss": [...], <history_val_key>: [...]}``.
        epochs_since_last_improvement: The patience counter.
    """

    def __init__(
        self,
        density_estimator: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optim_cfg: OptimConfig,
        train_cfg: TrainConfig,
        embedding_net: Optional[torch.nn.Module] = None,
        dataset: DatasetKind = "clone_sets",
        device: str = "cpu",
        ckpt_dir: Optional[PathLike] = None,
        quirks: Optional[TrainerQuirks] = None,
        final_pickle_path: Optional[PathLike] = None,
        effective_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Wire the loop up and try to resume.

        Args:
            density_estimator: The sbi flow, already on ``device``.
            train_loader: Training loader.
            val_loader: Validation loader. All three originals passed the *test*
                loader here, so "validation" and "test" are the same set of sims
                throughout this project -- see docs/REFACTOR_NOTES.md.
            optim_cfg: Optimiser settings, normally ``preset.optim``.
            train_cfg: Loop settings, normally ``preset.train``.
            embedding_net: Needed when ``optim_cfg.use_param_groups`` is set.
            dataset: Which batch layout the loaders yield; normally
                ``preset.data.dataset``.
            device: Device the tensors are moved to each step.
            ckpt_dir: Overrides ``train_cfg.ckpt_dir``. The CLI passes an
                absolute path; ``train_cfg.ckpt_dir`` alone is a bare directory
                name that resolves against the current directory, as in the
                originals (trap 11).
            quirks: The per-folder details of :class:`TrainerQuirks`. Defaults
                are derived from ``train_cfg.reload_best``, which is the only
                config field that separates CloneAtt from the other two; pass
                :func:`quirks_for` explicitly to be unambiguous.
            final_pickle_path: If given, ``pickle.dump`` the density estimator
                there when training ends. Only CloneAtt did this
                (``SetTransformer_NPE/inference_model.py:246-247``, which wrote
                ``SetTransformer_NPE_Freq_mean.pkl`` into the current directory).
            effective_config: :func:`cancer_sbi.config.config_to_dict` of the
                config this run is training with -- the CLI's, after the flags
                are folded in, not ``get_preset``'s. Written into every
                checkpoint so that evaluation rebuilds *this* network rather
                than the preset's. ``None`` writes checkpoints in the
                pre-2026-09-24 shape.
        """
        self.density_estimator = density_estimator
        self.embedding_net = embedding_net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim_cfg = optim_cfg
        self.train_cfg = train_cfg
        self.dataset: DatasetKind = dataset
        self.device = device
        self.final_pickle_path = (
            Path(final_pickle_path) if final_pickle_path is not None else None
        )
        self.effective_config = effective_config
        self.quirks = (
            quirks
            if quirks is not None
            # CloneAtt is the only model that reloads on early stop, so
            # reload_best identifies it when the caller did not say.
            else TrainerQuirks(
                best_state_on_device=train_cfg.reload_best == "on_early_stop",
                save_before_counter_update=train_cfg.reload_best != "on_early_stop",
            )
        )

        # Base_NPE/inference_model.py:47-55: the loop's own state.
        self.epoch = 0
        self.best_val_loss: float = np.inf
        self.best_model_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.epochs_since_last_improvement = 0

        # Preserved from Base_NPE/inference_model.py:64, :41 and
        # Plain_NPE/model.py:50-52, :46-47. Traps 14 and 11: the validation key
        # and the checkpoint directory are per-model, and the directory is
        # created eagerly at construction, before a single batch is read.
        self.history: Dict[str, List[float]] = {
            "training_loss": [],
            train_cfg.history_val_key: [],
        }
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else Path(train_cfg.ckpt_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.optimizer = build_optimizer(density_estimator, embedding_net, optim_cfg)

        # Preserved from Base_NPE/inference_model.py:115: the resume attempt is
        # the LAST thing the constructor does, after the optimiser exists,
        # because the optimiser state is part of the checkpoint.
        #
        # `resumed` records whether a checkpoint was actually restored, so that
        # callers can tell a fresh run from a resumed one -- cli/train.py reads
        # it to decide whether re-seeding would clobber the restored RNG state.
        self.resumed: bool = self._try_resume()

    # -- checkpointing ----------------------------------------------------

    def _try_resume(self) -> bool:
        """Resume from ``<ckpt_dir>/latest.pt`` if there is one.

        Returns:
            ``True`` if state was restored.
        """
        resumed = checkpoints.try_resume(
            self.ckpt_dir,
            self.density_estimator,
            self.optimizer,
            device=self.device,
            history_val_key=self.train_cfg.history_val_key,
        )
        if resumed is None:
            return False
        self.epoch = resumed.epoch
        self.best_val_loss = resumed.best_val_loss
        self.best_model_state_dict = resumed.best_model_state_dict
        self.history = resumed.history
        self.epochs_since_last_improvement = resumed.epochs_since_last_improvement
        return True

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        """Write this epoch's checkpoint.

        Args:
            epoch: Epoch number, 1-based.
            is_best: Also refresh ``best.pt``.

        Returns:
            The per-epoch file's path.
        """
        payload = checkpoints.build_checkpoint(
            epoch=epoch,
            density_estimator=self.density_estimator,
            optimizer=self.optimizer,
            best_val_loss=self.best_val_loss,
            best_model_state_dict=self.best_model_state_dict,
            history=self.history,
            epochs_since_last_improvement=self.epochs_since_last_improvement,
            effective_config=self.effective_config,
        )
        return checkpoints.save_checkpoint(self.ckpt_dir, epoch, payload, is_best=is_best)

    def load_checkpoint(self, path: PathLike) -> checkpoints.ResumedState:
        """Restore a specific checkpoint into this trainer.

        Args:
            path: The ``.pt`` file.

        Returns:
            The :class:`~cancer_sbi.training.checkpoints.ResumedState` that was
            applied.

        Note:
            This is the method every evaluation script called on a freshly
            constructed model to pull in ``best.pt``
            (``Base_NPE/z-score_violin.py:113``). It is kept as a method for
            exactly that reason.
        """
        resumed = checkpoints.load_checkpoint(
            path,
            self.density_estimator,
            self.optimizer,
            device=self.device,
            history_val_key=self.train_cfg.history_val_key,
        )
        self.epoch = resumed.epoch
        self.best_val_loss = resumed.best_val_loss
        self.best_model_state_dict = resumed.best_model_state_dict
        self.history = resumed.history
        self.epochs_since_last_improvement = resumed.epochs_since_last_improvement
        return resumed

    # -- validation -------------------------------------------------------

    @torch.no_grad()
    def compute_nltp(self) -> float:
        """Average negative log posterior over the validation set.

        Returns:
            ``sum(per-sample losses) / max(1, number of samples)``.

        Note:
            Preserved from ``Base_NPE/inference_model.py:176-190``: the average
            is over *samples*, not over batches, so a short last batch is
            weighted correctly -- and ``max(1, n_samples)`` makes an empty loader
            return ``0.0`` instead of raising.

            ``Plain_NPE/model.py:200`` passes the context positionally
            (``loss(theta_batch, x_batch)``) while the other two use
            ``condition=``. sbi's signature is
            ``loss(self, input, condition, **kwargs)``
            (``sbi/neural_nets/estimators/base.py``), so the second positional
            argument *is* ``condition`` and the two spellings are the same call.
        """
        self.density_estimator.eval()
        loss_sum = 0.0
        n_samples = 0

        for batch in self.val_loader:
            unpacked = unpack_batch(batch, self.dataset, self.device)
            if unpacked is None:
                continue
            theta_batch, condition = unpacked

            loss_vec = self.density_estimator.loss(theta_batch, condition=condition)
            loss_sum += loss_vec.sum().item()
            n_samples += loss_vec.numel()

        return loss_sum / max(1, n_samples)

    # -- training ---------------------------------------------------------

    def train(self) -> torch.nn.Module:
        """Run the epoch loop until early stopping or ``max_epochs``.

        Returns:
            The density estimator. Whether its weights are the last epoch's or
            the best epoch's depends on ``TrainConfig.reload_best`` (trap 12).

        Note:
            Ported from ``Base_NPE/inference_model.py:192-243``,
            ``SetTransformer_NPE/inference_model.py:181-249`` and
            ``Plain_NPE/model.py:132-187``. Every difference between those three
            is a config field or a :class:`TrainerQuirks` flag; the control flow
            below is the same in all three.
        """
        cfg = self.train_cfg
        val_key = cfg.history_val_key
        # "val" for the two clone-set models, "test" for DominantClone -- the
        # originals' per-epoch line said whichever matched their history key.
        val_label = "test" if val_key == "test_loss" else "val"

        if cfg.log_progress:
            logging.info("[%s] Begin Training", datetime.now())

        converged = False
        # Preserved from SetTransformer_NPE/inference_model.py:188-189: the
        # counter is seeded from the resumed value, so patience survives a
        # restart instead of silently resetting to zero.
        epochs_since_last_improvement = self.epochs_since_last_improvement

        while self.epoch < cfg.max_epochs and not converged:
            loss_sum = 0.0
            n_samples = 0
            self.epoch += 1

            self.density_estimator.train()
            for batch in self.train_loader:
                unpacked = unpack_batch(batch, self.dataset, self.device)
                if unpacked is None:
                    continue
                theta_batch, condition = unpacked

                # Preserved from Base_NPE/inference_model.py:209 and
                # SetTransformer_NPE/inference_model.py:201
                # (`zero_grad(set_to_none=True)`) versus Plain_NPE/model.py:152
                # (bare `zero_grad()`). Since torch 2.0 `set_to_none=True` IS the
                # default, so the two branches do the same thing today; under
                # torch 1.x they differ (grads become None rather than zeros).
                # The split follows `use_param_groups` because that flag marks
                # exactly the same two-versus-one folder split. Kept because the
                # instruction was to preserve, not to tidy.
                if self.optim_cfg.use_param_groups:
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.zero_grad()

                loss_vec = self.density_estimator.loss(theta_batch, condition=condition)
                loss = loss_vec.mean()
                loss.backward()

                # Preserved from Base_NPE/inference_model.py:214 and
                # SetTransformer_NPE/inference_model.py:207; absent from
                # Plain_NPE/model.py:157-158. Trap 16: DominantClone trains with
                # no gradient clipping at all. Note the norm is taken over
                # `density_estimator.parameters()`, which INCLUDES the embedding
                # net, so the clip is global, not per-group. This looks wrong to
                # unify but it is what the published models do; changing it
                # changes the results. See docs/REFACTOR_NOTES.md.
                if self.optim_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.density_estimator.parameters(),
                        max_norm=self.optim_cfg.grad_clip,
                    )

                self.optimizer.step()

                # Preserved from Base_NPE/inference_model.py:217-218. The
                # training loss reported is the average over samples of the
                # per-sample loss *before* the step that just ran, which is the
                # usual convention. Plain_NPE/model.py:155-156 accumulates these
                # two lines before backward() instead of after step(); the value
                # is identical because loss_vec is already computed.
                loss_sum += loss_vec.sum().item()
                n_samples += loss_vec.numel()

            train_loss_average = loss_sum / max(1, n_samples)
            self.history["training_loss"].append(train_loss_average)

            self.density_estimator.eval()
            val_loss_average = self.compute_nltp()
            self.history[val_key].append(val_loss_average)

            msg = (
                f"[{datetime.now()}] Epoch {self.epoch}: "
                f"train={train_loss_average:.6f}, {val_label}={val_loss_average:.6f}"
            )
            print(msg)
            if cfg.log_progress:
                logging.info(msg)

            # Preserved from Base_NPE/inference_model.py:230: the improvement
            # test is STRICT. An epoch that exactly ties the best loss counts as
            # no improvement and increments the patience counter.
            is_best = val_loss_average < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss_average
                self.best_model_state_dict = self._snapshot_best_state()
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

            if self.quirks.save_before_counter_update:
                # Preserved from Base_NPE/inference_model.py:234,237 and
                # Plain_NPE/model.py:176,179: the checkpoint is written BEFORE
                # self.epochs_since_last_improvement is refreshed, so the counter
                # inside the file is the previous epoch's. Issue 22 in
                # docs/REFACTOR_NOTES.md.
                self.save_checkpoint(self.epoch, is_best=is_best)
                self.epochs_since_last_improvement = epochs_since_last_improvement
            else:
                # SetTransformer_NPE/inference_model.py:235-236: the other order.
                self.epochs_since_last_improvement = epochs_since_last_improvement
                self.save_checkpoint(self.epoch, is_best=is_best)

            # Preserved from Base_NPE/inference_model.py:241 and
            # Plain_NPE/model.py:183 (`self.epoch >= self.min_epoch and ...`)
            # versus SetTransformer_NPE/inference_model.py:239, which drops the
            # min-epoch conjunct entirely. Trap 13: CloneAtt may stop before
            # epoch 50. `> stop_after_epochs - 1` there is the same integer test
            # as `>= stop_after_epochs` here. This looks wrong to unify but it is
            # what the published models do; changing it changes the results.
            # See docs/REFACTOR_NOTES.md.
            out_of_patience = epochs_since_last_improvement >= cfg.stop_after_epochs
            if cfg.enforce_min_epochs:
                out_of_patience = self.epoch >= cfg.min_epochs and out_of_patience

            if out_of_patience:
                print(
                    f"[early stop] No improvement for "
                    f"{epochs_since_last_improvement} epochs. "
                    f"Converged after {self.epoch} epochs."
                )
                # Preserved from SetTransformer_NPE/inference_model.py:240-241:
                # the reload happens INSIDE the early-stopping branch, so a run
                # that exhausts max_epochs keeps the last epoch's weights even
                # though a better epoch was seen. Trap 12.
                if cfg.reload_best == "on_early_stop":
                    if self.best_model_state_dict is not None:
                        self.density_estimator.load_state_dict(
                            self.best_model_state_dict
                        )
                converged = True

        # Preserved from Plain_NPE/model.py:187: unconditional, outside the
        # loop. Trap 12: CloneMLP does neither of these (`"never"`), so its
        # final weights are simply the last epoch's.
        #
        # Changed 2026-09-24 (WP-A): the original was NOT guarded against None,
        # so a run in which the validation loss never improved -- which is what
        # a one-epoch smoke test looks like after all the compute -- crashed
        # here with a TypeError. There are no weights to restore in that case,
        # so the only honest thing to do is say so and keep the last epoch's.
        # On every run that did improve this branch behaves exactly as before.
        if cfg.reload_best == "always":
            if self.best_model_state_dict is None:
                print(
                    "[warn] reload_best='always' but no epoch ever improved on "
                    "the initial validation loss, so there is no best snapshot "
                    "to restore. Keeping the final epoch's weights."
                )
            else:
                self.density_estimator.load_state_dict(self.best_model_state_dict)

        if self.final_pickle_path is not None:
            self._dump_final_pickle()

        return self.density_estimator

    # -- helpers ----------------------------------------------------------

    def _snapshot_best_state(self) -> Dict[str, torch.Tensor]:
        """Copy the current weights as the new best-so-far snapshot.

        Returns:
            A state dict, either a CPU copy or a deep copy on the model's device.

        Note:
            Preserved from ``Base_NPE/inference_model.py:232`` /
            ``Plain_NPE/model.py:174`` (``{k: v.detach().cpu() ...}``) versus
            ``SetTransformer_NPE/inference_model.py:229``
            (``deepcopy(state_dict())``). Issue 23 in docs/REFACTOR_NOTES.md:
            the original ``.detach().cpu()`` copies only when the model is
            on the GPU. On a CPU-only run it returned tensors that *shared
            storage* with the live parameters, so the "snapshot" kept changing
            as training continued and reloading it was a no-op. Every published
            run was on the GPU, where ``.cpu()`` already copies.

            Changed 2026-09-24 (WP-A): ``.clone()`` is appended, which makes the
            CPU path a real copy. On a GPU run ``.cpu()`` had already produced a
            fresh tensor, so the extra clone is a redundant memcpy and the
            resulting values are bit-for-bit what they were -- i.e. no published
            number moves, and the CPU path stops silently doing nothing.
        """
        if self.quirks.best_state_on_device:
            return deepcopy(self.density_estimator.state_dict())
        return {
            k: v.detach().cpu().clone()
            for k, v in self.density_estimator.state_dict().items()
        }

    def _dump_final_pickle(self) -> None:
        """Pickle the whole density estimator to ``final_pickle_path``.

        Note:
            Preserved from ``SetTransformer_NPE/inference_model.py:246-247``,
            which wrote ``SetTransformer_NPE_Freq_mean.pkl`` into the current
            directory after every CloneAtt run. ``Base_NPE/ppc-plot.py:81`` and
            the two Base_NPE notebooks load that file, which is how a CloneAtt
            model ends up being evaluated from inside the CloneMLP folder --
            open question (c) in docs/REFACTOR_NOTES.md.

            A pickle of a live module stores the *import path* of every class it
            contains, so a file written here references ``cancer_sbi.models.*``
            and cannot be unpickled by a script that expects the old top-level
            ``set_transformer`` module -- and vice versa. The old pickles on disk
            are still readable by the old scripts, and only by them.
        """
        self.final_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with self.final_pickle_path.open("wb") as handle:
            pkl.dump(self.density_estimator, handle)
        print(f"[pickle] Wrote density estimator to {self.final_pickle_path}")


__all__ = [
    "Trainer",
    "TrainerQuirks",
    "TrainingComponents",
    "MODEL_QUIRKS",
    "quirks_for",
    "seed_everything",
    "unpack_batch",
    "build_optimizer",
    "build_embedding_net",
    "build_training_components",
]
