"""Frozen configuration dataclasses and the three per-model presets.

The originals encoded their differences as literals scattered across
``*/main.py``, ``*/inference_model.py``, ``Plain_NPE/model.py`` and ``*/utils.py``.
This module collects them in one place *without changing any of them*: the three
presets below reproduce, field for field, what each folder actually did.

Several fields exist only to record a divergence between the three pipelines
(``enforce_min_epochs``, ``reload_best``, ``history_val_key``, ``grad_clip``,
``ckpt_dir``, ``z_score_x`` / ``z_score_y``, ``use_param_groups``,
``learning_rate_is_used``). They look like things that "should" be the same for
all three models. They are not, and unifying them would change published numbers.

All dataclasses are ``frozen=True``: a preset is a value, never mutated in place.
Use :func:`dataclasses.replace` to derive a variant.

Paths are ``None`` by default on purpose -- the originals hard-coded
``"../Guassian_Normal/simulation_outputs"`` relative to the folder they were run
from, which only worked from inside that folder. Callers (the CLI) must supply
them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

DatasetKind = Literal["clone_sets", "dominant_clone"]
EncoderKind = Literal["mlp", "attention", "deepset"]
ReloadBestPolicy = Literal["never", "on_early_stop", "always"]
ZScoreMode = Literal["none", "structured", "independent"]


@dataclass(frozen=True)
class DataConfig:
    """Which dataset class to build and how to feed it.

    Attributes:
        dataset: Selects the dataset family. ``"clone_sets"`` is
            ``CNASimsDataset`` (Base/SetT), ``"dominant_clone"`` is
            ``SimulationDataset`` (Plain). They filter the sims differently and
            therefore train on different subsets -- see ``num_trials``.
        root_dir: Directory holding the ``sim*/`` folders. No default: the
            originals hard-coded a relative path.
        split_path: Pickle with ``{"train_ids": ..., "test_ids": ...}``.
        top_k: Clones kept per trial by ``top_frequent_rows_tensor``; ``None``
            for the dominant-clone path, which keeps one profile per trial.
        batch_size: Mini-batch size.
        num_trials: Trials per sim.
        trial_filename: File read inside ``sim<N>/<t>/``.
        params_filename: File holding theta inside ``sim<N>/``.
        normalize_freq: Divide clone counts by the row total (see trap 18).
        pad_to_exact_k: Value stored on ``CNASimsDataset`` but never read; the
            per-trial call hard-codes ``True``. Kept for signature fidelity.
        drop_missing: Warn-and-skip rather than raise on unreadable sims.
        sim_regex: Which directory names count as a sim.
        use_bulk: ``SimulationDataset`` only -- pick ``CNratios_bulk`` (index 2)
            instead of ``CNratios_largest`` (index 1).
        pin_memory: Passed straight to ``DataLoader``.
    """

    dataset: DatasetKind
    root_dir: Optional[Path] = None
    split_path: Optional[Path] = None
    top_k: Optional[int] = 100
    batch_size: int = 32
    # Preserved from Base_NPE/utils.py:159-161 vs Plain_NPE/utils.py:86-92.
    # Trap 10: the two dataset classes react differently to a sim that has fewer
    # than `num_trials` trial files on disk. CNASimsDataset DROPS the sim
    # (639 train / 152 test); SimulationDataset NaN-pads it and keeps it
    # (718 train / 168 test). Same number here, deliberately different policy.
    # This looks wrong but it is what the published models do; changing it
    # changes the results. See docs/REFACTOR_NOTES.md.
    num_trials: int = 25
    trial_filename: str = "CNratios_all.pkl.gz"
    params_filename: str = "parameters.pkl"
    normalize_freq: bool = True
    pad_to_exact_k: bool = False
    drop_missing: bool = True
    sim_regex: str = r"^sim\d+$"
    use_bulk: bool = False
    pin_memory: bool = False


@dataclass(frozen=True)
class EncoderConfig:
    """Embedding-net architecture, for whichever of the three encoders is used.

    One dataclass covers all three encoders; fields that do not apply to a given
    ``kind`` are ``None`` and are never read. That keeps the preset table below
    readable as a single grid, which is how the paper describes the models.

    Attributes:
        kind: ``"mlp"`` -> ``BaselineCloneEmbedding`` (CloneMLP),
            ``"attention"`` -> ``CloneSetEmbedding`` (CloneAtt),
            ``"deepset"`` -> ``DeepSet`` (DominantClone).
        in_dim: Declared clone-row width (44 CNA features + 1 frequency).
        d_model: Per-trial embedding width for the clone-set encoders.
        hidden_dim: Hidden width of the per-clone MLP (``"mlp"`` only).
        num_layers: Depth. MLP: number of ``Linear`` layers. Attention: number of
            stacked ISABs. DeepSet: depth of phi.
        dropout: See ``encoder_dropout_is_used``.
        encoder_dropout_is_used: ``False`` for ``"attention"`` -- trap 4.
        freq_as_weight: Weight clones by their frequency when pooling.
        include_freq_in_mlp: ``"mlp"`` only; ``False`` keeps the 44->d projection.
        n_heads / num_inducing: Attention encoder only.
        layer_norm_in_attention: ``False`` everywhere -- trap 5.
        input_dim / hidden_dim_phi / hidden_dim_rho / output_dim / aggregation_fn
            / aggregation_dim / num_heads_deepset: DeepSet only.
        trials_*: The ``TrialsSBIEmbedding`` wrapper (clone-set models only);
            ``None`` for DeepSet, which *is* the embedding net.
    """

    kind: EncoderKind

    # --- shared by the two clone-set encoders -------------------------------
    in_dim: Optional[int] = 45
    d_model: Optional[int] = 128
    num_layers: Optional[int] = 3
    dropout: Optional[float] = None
    # Preserved from SetTransformer_NPE/set_transformer.py:88 (the `dropout`
    # parameter is accepted) and :91-103 (no nn.Dropout is ever constructed).
    # Trap 4: SetTransformer_NPE/main.py:35 passes dropout=0.2 exactly like
    # Base_NPE/main.py:35, but CloneAtt's encoder silently discards it, so
    # CloneAtt has NO encoder dropout at all. This looks wrong but it is what the
    # published model does; changing it changes the results.
    # See docs/REFACTOR_NOTES.md.
    encoder_dropout_is_used: bool = True
    freq_as_weight: Optional[bool] = True

    # --- BaselineCloneEmbedding (CloneMLP) ----------------------------------
    hidden_dim: Optional[int] = None
    include_freq_in_mlp: Optional[bool] = None

    # --- CloneSetEmbedding (CloneAtt) ---------------------------------------
    n_heads: Optional[int] = None
    num_inducing: Optional[int] = None
    # Preserved from SetTransformer_NPE/set_transformer.py:98 and :103, which
    # pass ln=False to every ISAB and to the PMA although MAB/ISAB/PMA all
    # default to ln=True. Trap 5: there is no LayerNorm anywhere in the
    # attention stack. This looks wrong but it is what the published model does;
    # changing it changes the results. See docs/REFACTOR_NOTES.md.
    layer_norm_in_attention: bool = False

    # --- DeepSet (DominantClone) --------------------------------------------
    input_dim: Optional[int] = None
    hidden_dim_phi: Optional[int] = None
    hidden_dim_rho: Optional[int] = None
    output_dim: Optional[int] = None
    aggregation_fn: Optional[str] = None
    aggregation_dim: Optional[int] = None
    num_heads_deepset: Optional[int] = None

    # --- TrialsSBIEmbedding wrapper ------------------------------------------
    trials_aggregation_fn: Optional[str] = None
    trials_num_hiddens: Optional[int] = None
    trials_num_layers: Optional[int] = None
    trials_output_dim: Optional[int] = None
    trials_aggregation_dim: Optional[int] = None


@dataclass(frozen=True)
class FlowConfig:
    """Arguments for the neural spline flow built by ``sbi``'s ``build_nsf``.

    The seven architecture fields (``hidden_features`` .. ``use_batch_norm``) are
    sbi's own defaults, written out explicitly here so the architecture stops
    depending on which sbi version is installed. See
    :func:`cancer_sbi.models.flow.build_flow`.

    Attributes:
        z_score_x: sbi's whitening mode for theta. Differs per model -- trap 1.
        z_score_y: sbi's whitening mode for the context. Differs per model.
        dropout_probability: Dropout inside the flow's residual blocks. All three
            mains pass 0.2 (``*/main.py:35``).
        exclude_invalid_y: ``False`` in all three originals; NaN-padded trials
            must survive into the embedding net, which handles them itself.
    """

    # Preserved from Base_NPE/inference_model.py:95-96 and
    # SetTransformer_NPE/inference_model.py:87 ("none") versus
    # Plain_NPE/model.py:64 ("structured"). Trap 1: the flow whitens theta and
    # the context for DominantClone but not for CloneMLP/CloneAtt. This looks
    # wrong but it is what the published models do; changing it changes the
    # results. See docs/REFACTOR_NOTES.md.
    z_score_x: ZScoreMode = "none"
    z_score_y: ZScoreMode = "none"
    dropout_probability: float = 0.2
    exclude_invalid_y: bool = False

    # sbi defaults, pinned. Identical in sbi 0.23.3 and 0.25.0.
    hidden_features: int = 50
    num_transforms: int = 5
    num_bins: int = 10
    num_blocks: int = 2
    tail_bound: float = 3.0
    hidden_layers_spline_context: int = 1
    use_batch_norm: bool = False


@dataclass(frozen=True)
class OptimConfig:
    """Adam configuration, which differs structurally between the models.

    Attributes:
        use_param_groups: ``True`` -> two groups (flow, embedding) with different
            learning rates and weight decays; ``False`` -> one group over all
            parameters. Trap 2.
        flow_lr / flow_weight_decay: Group 1, two-group mode only.
        embed_lr / embed_weight_decay: Group 2, two-group mode only.
        betas / eps: Explicit in two-group mode; torch defaults otherwise.
        learning_rate: Single-group learning rate.
        weight_decay: Single-group weight decay.
        learning_rate_is_used: Whether ``learning_rate`` reaches the optimiser.
            ``False`` for CloneMLP and CloneAtt -- trap 3.
        grad_clip: ``max_norm`` for ``clip_grad_norm_``, or ``None`` for no
            clipping. Trap 16.
    """

    # Preserved from Base_NPE/inference_model.py:106-113 and
    # SetTransformer_NPE/inference_model.py:104-114 (two groups) versus
    # Plain_NPE/model.py:67 (one group). Trap 2:
    # SetTransformer_NPE/inference_model.py:94 keeps the single-group line
    # commented out with the note "this makes overfitting", so the two-group
    # optimiser is a deliberate, load-bearing choice. This looks wrong to unify
    # but it is what the published models do; changing it changes the results.
    # See docs/REFACTOR_NOTES.md.
    use_param_groups: bool = True
    # Optional because the single-group models leave them unset rather than
    # carrying a value that is never applied.
    flow_lr: Optional[float] = 1e-3
    flow_weight_decay: Optional[float] = 1e-4
    embed_lr: Optional[float] = 1e-4
    embed_weight_decay: Optional[float] = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Preserved from Base_NPE/main.py:34 / SetTransformer_NPE/main.py:34, which
    # pass learning_rate=5e-4, versus Base_NPE/inference_model.py:58 and
    # SetTransformer_NPE/inference_model.py:49, which only store it on `self` and
    # never read it again. Trap 3: `learning_rate` is DEAD for CloneMLP and
    # CloneAtt; only Plain_NPE/model.py:67 actually uses it. This looks wrong but
    # it is what the published models do; wiring it up would change the results.
    # See docs/REFACTOR_NOTES.md.
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    learning_rate_is_used: bool = False

    # Preserved from Base_NPE/inference_model.py:214 and
    # SetTransformer_NPE/inference_model.py:207 (clip to 5.0) versus
    # Plain_NPE/model.py:157-158, which steps straight after backward() with no
    # clipping. Trap 16. This looks wrong but it is what the published models do;
    # changing it changes the results. See docs/REFACTOR_NOTES.md.
    grad_clip: Optional[float] = 5.0


@dataclass(frozen=True)
class TrainConfig:
    """Training loop, early stopping and checkpointing.

    Attributes:
        max_epochs / min_epochs / stop_after_epochs: The three mains all pass
            ``max_epochs=200, stop_after_epochs=50`` and inherit
            ``min_epochs=50`` (``*/main.py:36-37``).
        enforce_min_epochs: Whether ``min_epochs`` gates early stopping. Trap 13.
        reload_best: When the best weights are loaded back at the end. Trap 12.
        history_val_key: Key under which the validation curve is stored. Trap 14.
        ckpt_dir: Default checkpoint directory *of the original folder*. Trap 11.
        seed: ``None`` reproduces today's fully unseeded behaviour. Trap 21.
        log_progress: Mirror the per-epoch message into ``logging``.
    """

    max_epochs: int = 200
    min_epochs: int = 50
    stop_after_epochs: int = 50

    # Preserved from Base_NPE/inference_model.py:241 and Plain_NPE/model.py:183,
    # which both gate on `self.epoch >= self.min_epoch and ...`, versus
    # SetTransformer_NPE/inference_model.py:239, which drops that conjunct and
    # tests only `epochs_since_last_improvement > self.stop_after_epochs - 1`.
    # Trap 13: CloneAtt can stop before epoch 50. This looks wrong but it is what
    # the published model does; changing it changes the results.
    # See docs/REFACTOR_NOTES.md.
    enforce_min_epochs: bool = True

    # Preserved from Base_NPE/inference_model.py:243 (loop simply ends, weights
    # are whatever the last epoch left), SetTransformer_NPE/inference_model.py:
    # 240-241 (reloads best, but only inside the early-stopping branch) and
    # Plain_NPE/model.py:187 (reloads best unconditionally, after the loop).
    # Trap 12: three different answers to "which weights are saved at the end".
    # This looks wrong but it is what the published models do; changing it
    # changes the results. See docs/REFACTOR_NOTES.md.
    reload_best: ReloadBestPolicy = "never"

    # Preserved from Base_NPE/inference_model.py:64 and
    # SetTransformer_NPE/inference_model.py:59 ("validation_loss") versus
    # Plain_NPE/model.py:52 ("test_loss"). Trap 14: utilities/plot_losses.py
    # relies on the key telling it which folder a history came from. This looks
    # wrong but it is what the published models do; renaming it breaks the
    # existing plots and the saved checkpoints. See docs/REFACTOR_NOTES.md.
    history_val_key: str = "validation_loss"

    # Preserved from Base_NPE/inference_model.py:41 ("checkpoints_baseline"),
    # SetTransformer_NPE/inference_model.py:52 and Plain_NPE/model.py:29
    # ("checkpoints"). Trap 11: Base TRAINS into checkpoints_baseline/ but
    # Base_NPE/z-score_violin.py:105 EVALUATES from checkpoints/. Do not silently
    # pick one -- this default reproduces the training side, and the evaluation
    # side must pass its own. Open question for the author.
    # See docs/REFACTOR_NOTES.md.
    ckpt_dir: str = "checkpoints_baseline"

    # Preserved from Base_NPE/inference_model.py, SetTransformer_NPE/
    # inference_model.py and Plain_NPE/model.py: none of them ever calls
    # torch.manual_seed or numpy.random.seed. Trap 21: training is unseeded.
    # None keeps exactly that; pass an int only for new, explicitly-seeded runs.
    seed: Optional[int] = None
    log_progress: bool = True


@dataclass(frozen=True)
class ModelPreset:
    """One of the three published models, as a single immutable value.

    Attributes:
        name: Short CLI name (``clonemlp``, ``cloneatt``, ``dominantclone``).
        paper_name: Name used in the manuscript.
        origin: The original folder this preset reproduces.
        data / encoder / flow / optim / train: The five sub-configs.
        prior_sd: Standard deviation of the evaluation prior. Trap 20.
    """

    name: str
    paper_name: str
    origin: str
    data: DataConfig
    encoder: EncoderConfig
    flow: FlowConfig = field(default_factory=FlowConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # NOT preserved, deliberately. Trap 20: all seven evaluation scripts build
    # the reference prior with sd = math.sqrt(0.2) = 0.4472 --
    # Base_NPE/z-score_violin.py:125, Base_NPE/z-score.py:83,
    # Base_NPE/ppc-plot.py:94, SetTransformer_NPE/z-score_violin.py:125,
    # SetTransformer_NPE/z-score.py:83, SetTransformer_NPE/ppc-plot.py:94,
    # SetTransformer_NPE/kde_prior_vs_posterior.py:110 and
    # Plain_NPE/z-score_violin.py:116 -- while the simulator used sd = 0.2.
    # The wrong value is inert for every reported number -- the prior's support
    # is all of R^44 so DirectPosterior never rejects a sample, and log_prob is
    # never called -- EXCEPT for the prior curve drawn in the four kde_* figures,
    # which was therefore 2.24x too wide. The correct 0.2 is the default here;
    # figures.py must say so rather than silently reproducing the old curve.
    prior_sd: float = 0.2


# ---------------------------------------------------------------------------
# The three published models.
# ---------------------------------------------------------------------------

CLONEMLP = ModelPreset(
    name="clonemlp",
    paper_name="CloneMLP-NPE",
    origin="Base_NPE/",
    data=DataConfig(
        dataset="clone_sets",
        top_k=100,               # Base_NPE/main.py:25
        batch_size=32,           # Base_NPE/main.py:24
        trial_filename="CNratios_all.pkl.gz",  # Base_NPE/utils.py:99
    ),
    encoder=EncoderConfig(
        kind="mlp",
        # Base_NPE/inference_model.py:67-75
        in_dim=45,
        d_model=128,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,             # Base_NPE/main.py:35 -> :98; genuinely used
        encoder_dropout_is_used=True,
        freq_as_weight=True,
        include_freq_in_mlp=False,
        # Base_NPE/inference_model.py:78-84
        trials_aggregation_fn="mean",
        trials_num_hiddens=256,
        trials_num_layers=2,
        trials_output_dim=256,
        trials_aggregation_dim=1,
    ),
    flow=FlowConfig(z_score_x="none", z_score_y="none", dropout_probability=0.2),
    optim=OptimConfig(
        use_param_groups=True,
        grad_clip=5.0,
        learning_rate=5e-4,
        learning_rate_is_used=False,
    ),
    train=TrainConfig(
        max_epochs=200,
        min_epochs=50,
        stop_after_epochs=50,
        enforce_min_epochs=True,
        reload_best="never",
        history_val_key="validation_loss",
        ckpt_dir="checkpoints_baseline",
    ),
)

CLONEATT = ModelPreset(
    name="cloneatt",
    paper_name="CloneAtt-NPE",
    origin="SetTransformer_NPE/",
    data=DataConfig(
        dataset="clone_sets",
        top_k=100,               # SetTransformer_NPE/main.py:25
        batch_size=32,           # SetTransformer_NPE/main.py:24
        trial_filename="CNratios_all.pkl.gz",
    ),
    encoder=EncoderConfig(
        kind="attention",
        # SetTransformer_NPE/inference_model.py:63-71
        in_dim=45,
        d_model=128,
        n_heads=8,
        num_layers=3,
        num_inducing=32,
        dropout=0.2,
        # Trap 4 again, at the point where it bites: the value above is passed
        # (SetTransformer_NPE/inference_model.py:69) and dropped
        # (SetTransformer_NPE/set_transformer.py:88).
        encoder_dropout_is_used=False,
        freq_as_weight=True,
        layer_norm_in_attention=False,   # trap 5
        # SetTransformer_NPE/inference_model.py:74-80
        trials_aggregation_fn="mean",
        trials_num_hiddens=256,
        trials_num_layers=2,
        trials_output_dim=256,
        trials_aggregation_dim=1,
    ),
    flow=FlowConfig(z_score_x="none", z_score_y="none", dropout_probability=0.2),
    optim=OptimConfig(
        use_param_groups=True,
        grad_clip=5.0,
        learning_rate=5e-4,
        learning_rate_is_used=False,
    ),
    train=TrainConfig(
        max_epochs=200,
        min_epochs=50,
        stop_after_epochs=50,
        enforce_min_epochs=False,        # trap 13
        reload_best="on_early_stop",     # trap 12
        history_val_key="validation_loss",
        ckpt_dir="checkpoints",
    ),
)

DOMINANTCLONE = ModelPreset(
    name="dominantclone",
    paper_name="DominantClone-NPE",
    origin="Plain_NPE/",
    data=DataConfig(
        dataset="dominant_clone",
        top_k=None,              # no clone-set summarisation on this path
        batch_size=32,           # Plain_NPE/main.py:24
        trial_filename="results.pkl",   # Plain_NPE/utils.py:74
        use_bulk=False,          # Plain_NPE/utils.py:41,82 -> CNratios_largest
    ),
    encoder=EncoderConfig(
        kind="deepset",
        # Plain_NPE/model.py:55 with the defaults of Plain_NPE/net_builder.py:14-24
        in_dim=None,
        d_model=None,
        input_dim=44,
        hidden_dim_phi=44,       # Plain_NPE/model.py:18
        hidden_dim_rho=44,       # Plain_NPE/model.py:19
        num_layers=2,
        output_dim=128,          # Plain_NPE/model.py:20
        aggregation_fn="mean",
        aggregation_dim=1,
        num_heads_deepset=4,     # unused: aggregation_fn is "mean", not "attention"
        dropout=None,
        encoder_dropout_is_used=False,
        freq_as_weight=None,
        # DeepSet IS the embedding net here; there is no TrialsSBIEmbedding.
    ),
    # Trap 1: the one model that whitens.
    flow=FlowConfig(
        z_score_x="structured",
        z_score_y="structured",
        dropout_probability=0.2,  # Plain_NPE/main.py:34 -> Plain_NPE/model.py:64
    ),
    optim=OptimConfig(
        use_param_groups=False,   # trap 2: one group
        flow_lr=None,             # unused in one-group mode
        flow_weight_decay=None,   # unused in one-group mode
        embed_lr=None,            # unused in one-group mode
        embed_weight_decay=None,  # unused in one-group mode
        learning_rate=5e-4,       # Plain_NPE/main.py:33
        weight_decay=0.0,         # Plain_NPE/model.py:22 default
        learning_rate_is_used=True,   # trap 3: the only model that reads it
        grad_clip=None,           # trap 16: no clipping
    ),
    train=TrainConfig(
        max_epochs=200,
        min_epochs=50,
        stop_after_epochs=50,
        enforce_min_epochs=True,
        reload_best="always",            # trap 12
        history_val_key="test_loss",     # trap 14
        ckpt_dir="checkpoints",
    ),
)

#: Lookup by CLI name.
PRESETS = {
    CLONEMLP.name: CLONEMLP,
    CLONEATT.name: CLONEATT,
    DOMINANTCLONE.name: DOMINANTCLONE,
}


def get_preset(name: str) -> ModelPreset:
    """Look a preset up by its short name.

    Args:
        name: One of ``"clonemlp"``, ``"cloneatt"``, ``"dominantclone"``
            (case-insensitive).

    Returns:
        The frozen :class:`ModelPreset`.

    Raises:
        KeyError: If ``name`` is not one of the three published models.
    """
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown model {name!r}. Choose one of {sorted(PRESETS)}.")
    return PRESETS[key]


__all__ = [
    "DataConfig",
    "EncoderConfig",
    "FlowConfig",
    "OptimConfig",
    "TrainConfig",
    "ModelPreset",
    "CLONEMLP",
    "CLONEATT",
    "DOMINANTCLONE",
    "PRESETS",
    "get_preset",
]
