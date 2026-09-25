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

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

DatasetKind = Literal["clone_sets", "dominant_clone"]
EncoderKind = Literal["mlp", "attention", "deepset", "armtoken", "hybrid"]
ReloadBestPolicy = Literal["never", "on_early_stop", "always"]
ZScoreMode = Literal["none", "structured", "independent"]
InputSpace = Literal["log2", "copy"]
FreqMode = Literal["weight", "feature"]
AttnScale = Literal["published", "standard"]
TrialPool = Literal["mean", "attention"]
ArmFeatureNorm = Literal["none", "layernorm", "batchnorm"]


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
        num_workers: Worker processes per ``DataLoader``. ``0`` -- loading on
            the main process -- is what every published run did, and is the
            default here for that reason. Raising it only prefetches: shuffling
            stays on the main process's generator, so the RNG stream and hence
            the results are unchanged.
        cache_dir: Optional path to a pre-built clone cache (see
            ``src/utilities/build_clone_cache.py``). ``None`` -- the default and
            the published behaviour -- reads the gzipped trial files directly.
            Only the ``clone_sets`` path uses it; the dominant-clone builder
            ignores it.
        trial_subsample: Augmentation switch, matrix 3. ``None`` -- the default
            and the published behaviour -- serves every one of ``num_trials``
            trials. An int ``K < num_trials`` makes the *training* dataset draw
            a fresh random subset of ``K`` trials on every ``__getitem__``, so
            each epoch sees a different view of the same sim; the item is then
            ``(K, top_k, 45)`` with a ``(K,)`` mask. Validation and test
            datasets never subsample -- 25 trials is the published evaluation
            condition and the number every reported score is on. Read by the
            ``clone_sets`` path only; the dominant-clone builder ignores it
            (``cli/train.py`` warns).
        min_trials: Matrix 7. ``None`` -- the default and the published
            behaviour -- keeps only sims with all ``num_trials`` trial files
            (trap 10). An int ``K`` keeps every sim with at least ``K`` of
            them, NaN-padding the rest. Applied to the TRAINING and VALIDATION
            datasets only: the test set stays the published complete-sim set so
            every reported number remains comparable
            (``data/loaders.py``). Read by the ``clone_sets`` path only; the
            dominant-clone builder ignores it (``cli/train.py`` warns), because
            that family has its own NaN-pad rule already.
        require_all_trials: Restrict DominantClone to sims with every trial file
            present, i.e. the clone-set models' sim set. Off (the default) is
            the published behaviour: NaN-pad the missing trials and keep the
            sim. The ``clone_sets`` path already applies this rule and ignores
            the field.
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
    # Added 2026-09-24 (WP-A). 0 == the published behaviour; see the docstring.
    num_workers: int = 0
    # Added 2026-09-24 (WP-B). None == read the gzipped trial files; see above.
    cache_dir: Optional[str] = None
    # Added 2026-09-24. False == the published trap-10 behaviour; True makes the
    # dominant-clone path use the clone-set models' sim set, so that the three
    # models can be compared on the same simulations.
    require_all_trials: bool = False
    # Added 2026-09-24 (matrix 3). None == every trial, the published
    # behaviour; see the docstring. TRAINING loaders only.
    trial_subsample: Optional[int] = None
    # Added 2026-09-24 (matrix 7). None == the published trap-10 rule; see the
    # docstring. TRAINING and VALIDATION loaders only -- never the test set.
    min_trials: Optional[int] = None


@dataclass(frozen=True)
class EncoderConfig:
    """Embedding-net architecture, for whichever of the three encoders is used.

    One dataclass covers all three encoders; fields that do not apply to a given
    ``kind`` are ``None`` and are never read. That keeps the preset table below
    readable as a single grid, which is how the paper describes the models.

    Attributes:
        kind: ``"mlp"`` -> ``BaselineCloneEmbedding`` (CloneMLP),
            ``"attention"`` -> ``CloneSetEmbedding`` (CloneAtt),
            ``"deepset"`` -> ``DeepSet`` (DominantClone),
            ``"armtoken"`` -> ``ArmTokenEmbedding`` (ArmToken, matrix 5). The
            last one is the whole embedding net, like DeepSet and unlike the
            first two: its ``trials_*`` fields are ``None`` because wrapping it
            in ``TrialsSBIEmbedding`` would blend the per-arm blocks back
            together. ``"hybrid"`` -> ``HybridEmbedding`` (matrix 6): an
            ``ArmTokenEmbedding`` and a fully-wrapped CloneAtt stack side by
            side, their contexts concatenated. It reads BOTH sets of fields --
            the armtoken ones size the arm branch, ``d_model``/``n_heads``/
            ``num_inducing``/``freq_mode``/``attn_scale`` and the ``trials_*``
            block size the clone branch -- which is why this dataclass's
            "fields that do not apply are None" rule has one kind for which
            almost nothing is None.
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
        layer_norm_in_attention: ``False`` everywhere -- trap 5. Documentary
            only: ``attn_ln`` is the field that is actually wired.
        input_space: ``"mlp"`` and ``"attention"``. ``"log2"`` feeds the encoder
            the raw log2 ratios, which is what every published run did;
            ``"copy"`` converts them to copy-number space inside the encoder
            (repair T2 / runs R2, R6-R8).
        freq_renorm: ``"attention"`` only. ``False`` keeps CloneAtt's published
            raw-frequency multiply; ``True`` renormalises the per-clone weights
            so the tokens are not shrunk to ~0.003 of their scale (run R4).
        freq_mode: Both clone-set encoders. ``"weight"`` is the published
            behaviour for either one; ``"feature"`` feeds
            ``log10(clamp(freq, 1e-6))`` to the per-clone projection as a 45th
            input column. For ``"attention"`` that also *drops* the token
            multiply (runs R5-R8, modulated by ``freq_renorm``). For ``"mlp"``
            (matrix 3, run R10) there is no token multiply to drop: CloneMLP's
            frequency use is the normalised weighted mean in the pooling step,
            which is kept -- ``"feature"`` only widens the MLP's input from 44
            to 45. Note this is a *different* column from
            ``include_freq_in_mlp``, which appends the RAW frequency; the two
            are refused together by ``BaselineCloneEmbedding``.
        attn_ln: ``"attention"`` only. ``True`` builds every MAB/ISAB/PMA with
            ``ln=True`` (runs R5-R8). ``False`` -- the default -- is trap 5.
        attn_dropout_active: ``"attention"`` only. ``True`` makes
            ``CloneSetEmbedding`` apply ``nn.Dropout(dropout)`` after each ISAB
            and after the PMA (run R8). ``False`` -- the default -- is trap 4,
            where ``dropout`` is accepted and silently discarded.
        attn_scale: ``"attention"`` only, matrix 4. ``"published"`` -- the
            default -- is trap 6's ``sqrt(dim_V)`` logit scaling;
            ``"standard"`` is the textbook ``sqrt(dim_V / num_heads)`` (run
            R17).
        trial_pool: Both clone-set encoders, matrix 4. ``"mean"`` -- the
            default -- is sbi's masked mean over the 25 trial embeddings;
            ``"attention"`` pools them with a one-seed PMA instead (run R20).
        arm_feature_norm: ArmToken and the hybrid's arm branch, matrix 8.
            ``"none"`` -- the default -- is AT0: the eight per-arm moments and
            their across-trial spread reach ``arm_mlp`` un-normalised.
            ``"layernorm"`` and ``"batchnorm"`` normalise that ``(B, 44, P)``
            vector with one module shared by all 44 arms, so the equivariance
            holds either way. Warned about and ignored for the other kinds.
        arm_context_norm: ArmToken and the hybrid's arm branch, matrix 8.
            ``True`` LayerNorms the 416-wide context the arm branch hands the
            flow (the hybrid's clone branch is untouched). ``False`` -- the
            default -- is AT0. Not the same thing as ``FlowConfig.z_score_y``,
            which standardises the raw input tensor, not the context.
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
    # `layer_norm_in_attention` above is DOCUMENTARY -- verify_refactor.py:282
    # reads it to assert trap 5 is recorded, and nothing constructs a layer from
    # it. `attn_ln` below is the wired one: build_embedding_net forwards it to
    # CloneSetEmbedding, which passes it as `ln` to every MAB/ISAB/PMA.
    attn_ln: bool = False

    # --- ArmTokenEmbedding (ArmToken), matrix 5 ------------------------------
    # All five are Optional with a None default so that every checkpoint
    # written before matrix 5 round-trips through _block_from_dict unchanged:
    # a missing key keeps the default, and None is what the other encoders'
    # unused fields already look like. build_embedding_net substitutes the
    # published ArmToken values (models/arm_tokens.py) for a None.
    d_token: Optional[int] = None        # arm-token width through the stack
    d_arm: Optional[int] = None          # numbers read out per arm
    d_global: Optional[int] = None       # width of the global block
    n_arm_layers: Optional[int] = None   # ISABs over the 44-arm set
    # Named apart from `num_inducing` on purpose: that field is CloneAtt's and
    # is published at 32, while this one is 16 over a 44-element set. Sharing
    # one field would make --num-inducing silently reshape ArmToken's ISABs.
    arm_num_inducing: Optional[int] = None

    # --- repair switches, added 2026-09-24 (WP-A) ----------------------------
    # Every default is the PUBLISHED behaviour, so every preset below keeps
    # reproducing its original folder without naming them.
    input_space: InputSpace = "log2"      # both clone-set encoders
    freq_renorm: bool = False             # CloneSetEmbedding (CloneAtt)
    # --- repair switches, matrix 2 (added 2026-09-24) ------------------------
    # "weight" is the published token multiply; "feature" removes it and feeds
    # log10(freq) to the input projection instead (runs R5-R8). `freq_renorm`
    # has nothing to renormalise in "feature" mode -- cli/train.py refuses the
    # pair rather than silently ignoring one of them.
    # Matrix 3 widened this from CloneAtt-only to both clone-set encoders; the
    # default is still the published behaviour for either of them.
    freq_mode: FreqMode = "weight"        # both clone-set encoders
    # Trap 4 made opt-out: True wires `dropout` into CloneSetEmbedding, which
    # published CloneAtt never did. False keeps the value decorative (run R8).
    attn_dropout_active: bool = False     # CloneSetEmbedding (CloneAtt)
    # --- repair switches, matrix 4 (added 2026-09-24) ------------------------
    # Trap 6 made opt-out. "published" divides the attention logits by
    # sqrt(dim_V) -- sqrt(128) = 11.31 -- in every MAB, which is what
    # SetTransformer_NPE/set_transformer.py:35 does; "standard" divides by
    # sqrt(dim_V / num_heads) = sqrt(16) = 4, the textbook per-head scale, so
    # the logits (and hence the attention) are ~2.83x sharper (run R17).
    # CloneSetEmbedding only; the MLP encoder has no attention to scale.
    attn_scale: AttnScale = "published"   # CloneSetEmbedding (CloneAtt)
    # Both clone-set encoders. "mean" is the published pooling over the T=25
    # per-trial embeddings -- sbi's PermutationInvariantEmbedding, which takes
    # a NaN-masked mean. "attention" replaces that mean with a one-seed PMA
    # over the same (B, T, d_model) stack (run R20), keeping the post-pooling
    # MLP and therefore the flow's context width. See
    # cancer_sbi.models.trials.AttentionTrialPooling.
    trial_pool: TrialPool = "mean"        # both clone-set encoders
    # --- experiment switches, matrix 8 (added 2026-09-24) --------------------
    # ArmToken and the hybrid's ARM branch only. Both defaults reproduce AT0
    # exactly: "none" and False construct no module at all, so a default
    # ArmToken's state_dict is byte for byte the one matrix 5 measured.
    #
    # `arm_feature_norm` normalises the pooled moment vector (B, 44, P) just
    # before `arm_mlp`. The eight moments sit on four different scales -- a
    # weighted mean and two extrema in [-1, 3], an sd in [0, ~2], three
    # fractions in [0, 1] -- and matrix 5 fed them in raw, declaring that an
    # accepted risk and never testing it. "layernorm" z-scores each arm token
    # across its own P features; "batchnorm" is a learned per-feature z-scoring
    # over the (B * 44, P) view, with running statistics at eval time. One
    # shared module either way, so the arm-equivariance survives.
    #
    # `arm_context_norm` LayerNorms the 416-wide output instead. That vector is
    # the flow's context, and the preset gives the flow z_score_y="none" --
    # which is not an oversight to fix with sbi's own switch, because sbi
    # standardises the RAW input tensor before the embedding net, not the
    # context. This is the only place the context's scale can be set.
    arm_feature_norm: ArmFeatureNorm = "none"   # armtoken + hybrid's arm branch
    arm_context_norm: bool = False              # armtoken + hybrid's arm branch

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
            mains pass 0.2 (``*/main.py:35``), so 0.2 -- not 0.0 -- is the
            published value every preset below carries. ``--flow-dropout``
            (matrix 3, runs R11/R14) overrides it.
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
    # 5 is sbi's default and the published value; --flow-num-transforms
    # (matrix 3, runs R12/R16) shrinks the flow by lowering it. 50 was
    # deliberately left without a flag through matrices 1-5 because it is the
    # published width in every run; matrix 6 adds --flow-hidden-features to
    # WIDEN it (runs AT6, AT8, AT9), which is an experiment, not a repair. The
    # default is still 50, so an untouched command line is unchanged.
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
        lr_plateau: Matrix 6. ``False`` -- the default and the published
            behaviour -- builds no scheduler at all, so training is bitwise
            what it was before this field existed. ``True`` attaches
            ``ReduceLROnPlateau(mode="min", factor=0.5, patience=5)`` to the
            optimiser and steps it on the validation loss once per epoch; its
            state is written into every checkpoint and restored on resume, so a
            restarted run continues with the learning rate it had reached.
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
    # Added 2026-09-24 (matrix 6). False == no scheduler is constructed, which
    # is exactly what every published run did; see the docstring.
    lr_plateau: bool = False


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
#
# These three are frozen history. They reproduce, field for field, what
# Base_NPE/, SetTransformer_NPE/ and Plain_NPE/ actually did, and nothing in
# this file may change them again. Until 2026-09-25 they were also what
# ``--model clonemlp`` meant; they are now reached with ``--published`` or with
# ``--model clonemlp_published``, and the repaired presets below -- built from
# these with ``dataclasses.replace``, so the diff is exactly the changed fields
# -- are the defaults. See "The repaired defaults" further down.
# ---------------------------------------------------------------------------

#: CloneMLP-NPE as published (Base_NPE/); select with ``--published`` or
#: ``--model clonemlp_published``. True R^2 0.472, 39 of 44 arms failing SBC,
#: 95 % coverage 0.922, pooled std z 0.86, log p 21.96
#: (docs/CAMPAIGN_REPORT_2026-09-24.md, §3 and §4).
CLONEMLP_PUBLISHED = ModelPreset(
    name="clonemlp_published",
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

#: CloneAtt-NPE as published (SetTransformer_NPE/); select with ``--published``
#: or ``--model cloneatt_published``. True R^2 0.049, log p 9.12 -- the
#: frequency multiply shrinks every token to ~1/100 of its scale
#: (docs/CAMPAIGN_REPORT_2026-09-24.md, finding 4).
CLONEATT_PUBLISHED = ModelPreset(
    name="cloneatt_published",
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

#: DominantClone-NPE as published (Plain_NPE/); select with ``--published`` or
#: ``--model dominantclone_published``. True R^2 0.170 -- but on 707 test cases,
#: not the 651 every other model is scored on, because of trap 10 above
#: (docs/CAMPAIGN_REPORT_2026-09-24.md, finding 8).
DOMINANTCLONE_PUBLISHED = ModelPreset(
    name="dominantclone_published",
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

# ---------------------------------------------------------------------------
# The repaired defaults (2026-09-25).
#
# Policy change. ``--model clonemlp`` no longer means "the model as published";
# it means the best HONEST configuration the 2026-09-24 campaign found for that
# encoder. The published ones are preserved above, byte for byte, and are
# reached with ``--published`` or ``--model <name>_published``; every job script
# that reproduces a campaign run passes ``--published`` (see jobs/README.md).
#
# The rule used to choose each configuration, in this order:
#   1. calibration first. A model whose 95 % coverage is 0.945 and whose pooled
#      std of z is 1.00 beats a sharper one that fails SBC on 39 of 44 arms,
#      because a posterior that does not mean what it says is not an answer --
#      and the project's goal is real-data inference, where the error bars are
#      the product.
#   2. accuracy second, and only where the difference is OUTSIDE the seed band
#      the campaign measured: +-0.05 R^2 for the clone-set models, +-0.002 for
#      ArmToken (docs/CAMPAIGN_REPORT_2026-09-24.md, finding 3). Anything inside
#      that band is not adopted on its number alone.
#
# Each preset below is a ``replace`` on the published one above it, so the
# source diff IS the change: a field not named here still carries its published
# value, traps and all. Evidence for every number quoted:
# docs/CAMPAIGN_REPORT_2026-09-24.md -- §4 (the seed-averaged comparison) and
# findings 2, 4, 7 and 8.
# ---------------------------------------------------------------------------

#: CloneMLP-NPE, repaired: run **R2**.
#:
#: Published as true R^2 0.472 -- but with 39 of 44 arms failing SBC, 95 %
#: coverage 0.922 and pooled std z 0.86. It is sharp because it is
#: OVERCONFIDENT, and the 0.472 was itself selected on the test set it was
#: scored on (report §2.2). R2's seed family is 0.363 +- 0.053 with coverage
#: 0.945 and std z 1.00: a third of the headline R^2 traded for posteriors that
#: mean what they say. By rule 1 that is the trade this preset makes.
#:
#: Two fields change. Whitening theta is the solid half (report, finding 2).
#: Copy space is the weaker half and is kept on its mechanism rather than its
#: size: the +0.087 that a single-run R1-vs-R2 comparison suggested did NOT
#: replicate -- at the seed level the difference is +0.035, inside one seed sd
#: -- but copy space never hurt a single run in the campaign, it costs nothing,
#: and matrix 8 finally measured it at +0.006 in ArmToken, the one model whose
#: +-0.002 band is tight enough to see an effect that small (runs AT11/AT11s1).
CLONEMLP = replace(
    CLONEMLP_PUBLISHED,
    name="clonemlp",
    flow=replace(
        CLONEMLP_PUBLISHED.flow,
        # R1 -> R2. Trap 1 left theta unwhitened for this model. Whitening it
        # halves the SBC failures (37 -> 16 on the control seed) and puts the
        # pooled std of z on 1.0, at a cost of about 0.13 R^2. Report, finding 2.
        z_score_x="structured",
    ),
    encoder=replace(
        CLONEMLP_PUBLISHED.encoder,
        # R2. Converts the log2 ratios to copy-number space INSIDE the encoder,
        # which removes the sentinel that is 16-21 % of all input values and
        # sits ~15 sd from the rest (models/mlp_encoder.py:163-175). Report,
        # finding 2, and matrix 8 for its size.
        input_space="copy",
    ),
)

#: CloneAtt-NPE, repaired: run **R26**.
#:
#: Published at true R^2 0.049 and log p 9.12; R26 reaches 0.579 +- 0.015 over
#: two seeds with log p 29.97. This is the one model the campaign repaired
#: rather than re-balanced, and it took five measured steps, each a run in
#: report finding 4 and matrices 2-4b:
#:
#:   R5      log-frequency as a 45th input FEATURE, and LayerNorm on at last
#:           (the multiply is what forced trap 5)            0.025 -> 0.363
#:   R6      + copy-space input                                     -> 0.422
#:   R12     + a 3-transform flow: fewer parameters, the same fit, 8 of 44 SBC
#:           failures, and 40 % cheaper                              -> 0.413
#:   R18     + tail_bound 3 -> 5. The largest flow-side gain of the campaign,
#:           +0.09 replicated across three seeds at +-0.002     -> 0.507
#:   R21     + d_model 128 -> 256, on the R12 base (tail_bound still 3.0):
#:           +0.05 on its own                                   -> 0.455
#:   R26     + d_model 256 AND tail_bound 5 together: the two stack
#:                                              -> 0.568, and R26s1 -> 0.589
#:
#: Everything else is CloneAtt's published value: n_heads 8, num_inducing 32,
#: freq_renorm False (R4 showed renormalising the multiply changes nothing --
#: one unit of mass over 100 clones still leaves every token at ~1/100 scale),
#: attn_scale "published" (R17: no effect) and trial_pool "mean" (R20: the 25
#: replicates are exchangeable, so attention has nothing to select over).
#:
#: NOT fully calibrated, and this preset does not pretend otherwise: R26 is
#: slightly overconfident at 95 % coverage 0.930 and std z 1.08. The fix is
#: post-hoc rather than a preset field -- the per-arm affine recalibration of
#: jobs/recalibrate.sh takes R27rc to coverage 0.944 and 18 SBC failures from
#: 32, with R^2 unchanged (report, "post-hoc stages").
CLONEATT = replace(
    CLONEATT_PUBLISHED,
    name="cloneatt",
    flow=replace(
        CLONEATT_PUBLISHED.flow,
        # R1 -> R5 onwards. Trap 1 again; same reasoning as CLONEMLP's.
        z_score_x="structured",
        # R12. Fewer parameters, the same fit, and 8 of 44 SBC failures -- the
        # best-calibrated point of the CloneAtt ladder. Report, matrix 3.
        num_transforms=3,
        # R18. sbi's 3.0 defines the spline on +-3 STANDARDISED units, and with
        # theta of sd 0.2005 that was clipping the tails of exactly the
        # well-learned arms the SBC analysis had flagged. +0.104 R^2 and
        # +6.1 nats, replicated over three seeds at +-0.002. Report, finding 4.
        tail_bound=5.0,
    ),
    encoder=replace(
        CLONEATT_PUBLISHED.encoder,
        # R6. Same transform, same reason, as CLONEMLP's above.
        input_space="copy",
        # R5. Moves the frequency from a token MULTIPLIER to a 45th input
        # column, log10(clamp(freq, 1e-6)). This is what removes the dependence
        # on token magnitude -- and therefore what lets attn_ln be switched on.
        # Report, finding 4.
        freq_mode="feature",
        # R5. Trap 5 recorded that the published stack has no LayerNorm
        # anywhere; with freq_mode "feature" there is no longer a ~0.003-scale
        # multiply for a LayerNorm to fight, so it goes on. The DOCUMENTARY
        # field `layer_norm_in_attention` stays False: it records what the
        # published model did and verify_refactor.py reads it for that.
        attn_ln=True,
        # R21 (0.455, d_model 256 alone on the R12 base) and R26 (0.568, and
        # R26s1 0.589, with tail_bound 5): wider tokens through the ISAB stack,
        # +0.05 alone, and the gain stacks with the tail bound. More heads
        # (R22, 0.408) and more inducing points (R23 0.363, R24 0.431) did not
        # help, so those two keep their published values.
        d_model=256,
    ),
)

#: DominantClone-NPE, repaired: run **D0**.
#:
#: The only change is WHICH SIMULATIONS it is trained and scored on, and that
#: is a comparability repair, not a model repair. Trap 10 above: CNASimsDataset
#: drops a sim with fewer than 25 trial files while SimulationDataset NaN-pads
#: it and keeps it, so the published DominantClone was measured on 707 test
#: cases while every other model in this file was measured on 651.
#: ``require_all_trials=True`` puts it on the same 651, which is the only thing
#: that makes the ranking in report finding 8 mean anything: 0.177 +- 0.017 over
#: three seeds, against the published 0.170 on its larger set.
#:
#: **No model repair was ever applied to DeepSet.** No matrix in the campaign
#: touched this encoder, its flow or its optimiser -- theta whitening was
#: already on here (trap 1 points the other way for this one model), and
#: nothing else was tried. Its place at the bottom of finding 8's ranking is a
#: statement about the dominant clone as an INPUT, not a measurement of the best
#: DeepSet that could be built. Anyone quoting it should say so.
DOMINANTCLONE = replace(
    DOMINANTCLONE_PUBLISHED,
    name="dominantclone",
    data=replace(
        DOMINANTCLONE_PUBLISHED.data,
        # D0. The clone-set models' sim set, so the three models are compared
        # on identical data. Report, finding 8.
        require_all_trials=True,
    ),
)


# ---------------------------------------------------------------------------
# Matrix 5: a fourth model, not a published one.
# ---------------------------------------------------------------------------

#: ArmToken-NPE. The first encoder in this project whose architecture is tied
#: to the 44 per-arm coefficients the flow has to predict: the arms are the
#: tokens, every weight is shared across them, and arm identity reaches the
#: flow only through the order the 44 blocks are concatenated in. The data,
#: optimiser and training blocks are copied from CLONEATT so that a matrix-5
#: run is read against matrix 4 on everything except the encoder.
#:
#: NOT a published model: nothing here reproduces an original folder, so
#: `origin` names this module rather than a directory, and none of the traps
#: apply -- `attn_ln` is True (trap 5 has nothing to protect here; see
#: ArmTokenEmbedding.__init__) and `z_score_x` is "structured", which matrices
#: 2-4 established as better than the published "none" for every model.
#:
#: This preset IS run AT0: true R^2 0.898 +- 0.002 over three seeds, log p
#: 65.53 +- 0.02, 95 % coverage 0.939 (docs/CAMPAIGN_REPORT_2026-09-24.md, §4).
#: It has no `_published` twin, because it was never published -- `--published`
#: is an error for this model rather than a synonym for itself.
#:
#: Two things that belong to AT0 and are deliberately NOT fields here:
#:
#: * The **3-seed ensemble** AT0ens3 -- R^2 0.904, 95 % coverage 0.961, log p
#:   67.97, the best result in the campaign -- is an EVALUATION-time procedure,
#:   not a configuration. `jobs/ensemble.sh` pools three finished runs'
#:   posterior draws into one equal-weight mixture; no single training run can
#:   be it, and a preset field claiming it would be a lie about what one
#:   checkpoint contains.
#: * Matrix 8 closed this encoder's two open questions and both answers are
#:   "leave the default": copy space is worth +0.006 R^2 and +2.2 nats over
#:   log2 (AT11/AT11s1, replicated, 3x the seed band), and normalising the
#:   moment features does not help -- `arm_feature_norm` layernorm is within
#:   noise (AT12), batchnorm is slightly worse (AT13), and `arm_context_norm`
#:   costs a nat and 0.013 of coverage (AT14). Both stay off.
ARMTOKEN = ModelPreset(
    name="armtoken",
    paper_name="ArmToken-NPE",
    origin="cancer_sbi/models/arm_tokens.py (new, matrix 5)",
    data=DataConfig(
        dataset="clone_sets",
        top_k=100,
        batch_size=32,
        trial_filename="CNratios_all.pkl.gz",
    ),
    encoder=EncoderConfig(
        kind="armtoken",
        in_dim=45,
        # d_model is CloneMLP's and CloneAtt's per-trial width and has no
        # meaning here: ArmTokenEmbedding computes its own output width as
        # 44 * d_arm + d_global and exposes it as `.d_model`. None -- not 128
        # -- so that a value nothing reads cannot be mistaken for a choice,
        # and so --d-model has nothing to overwrite (cli/train.py warns).
        d_model=None,
        num_layers=None,
        d_token=64,
        d_arm=8,
        d_global=64,
        n_arm_layers=1,
        arm_num_inducing=16,
        n_heads=4,
        trial_pool="mean",
        input_space="copy",
        attn_ln=True,
        dropout=0.2,
        attn_dropout_active=False,
        attn_scale="published",
        freq_as_weight=None,     # the moments renormalise the weights themselves
        encoder_dropout_is_used=False,   # opt-in, like CloneAtt's (trap 4 style)
        # ArmTokenEmbedding IS the embedding net; there is no TrialsSBIEmbedding.
        trials_aggregation_fn=None,
        trials_num_hiddens=None,
        trials_num_layers=None,
        trials_output_dim=None,
        trials_aggregation_dim=None,
    ),
    # z_score_x "structured" is matrix 2-4's finding, not CloneAtt's published
    # "none"; z_score_y stays "none" as in CLONEATT, because the context is
    # this encoder's own output and whitening it would undo the per-arm scale
    # the blocks are meant to carry.
    flow=FlowConfig(
        z_score_x="structured",
        z_score_y="none",
        dropout_probability=0.2,
        num_transforms=3,        # matrix 3's R12 finding, kept
        hidden_features=50,      # the published width, in every run
        # AT0, added to the preset 2026-09-25. Matrix 4's R18 established 5.0
        # (+0.104 R^2 and +6.1 nats over tail_bound 3, replicated across three
        # seeds at +-0.002) and EVERY ArmToken run in matrices 5-8 passed
        # `--tail-bound 5` on the command line -- so sbi's 3.0 sitting here
        # described a run nobody ever made, and a command line that forgot the
        # flag was not comparable with AT0. Same value, same reason, as
        # HYBRID's below. Report, finding 4 and matrix 5.
        tail_bound=5.0,
    ),
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
        enforce_min_epochs=False,
        reload_best="on_early_stop",
        history_val_key="validation_loss",
        ckpt_dir="checkpoints",
    ),
)


# ---------------------------------------------------------------------------
# Matrix 6: a fifth model, the late fusion of the fourth and the second.
# ---------------------------------------------------------------------------

#: Hybrid-NPE. ArmToken reaches R^2 0.898 on per-arm moments alone; CloneAtt's
#: best (R26) reaches 0.568 on raw clone-level attention. This preset runs both
#: encoders side by side and concatenates their contexts -- 416 from the arm
#: branch, 256 from the clone branch, 672 in all -- so the flow can use either.
#: The question it asks is narrow and worth one matrix: does raw clone-level
#: attention carry ANYTHING the per-arm moments threw away?
#:
#: The two branches are each other's published best. The arm branch is
#: ARMTOKEN's encoder field for field; the clone branch is R26's CloneAtt
#: (`--z-score-x structured --input-space copy --freq-mode feature --attn-ln
#: --flow-num-transforms 3 --tail-bound 5 --d-model 256`), and the data,
#: optimiser and training blocks come from CLONEATT so a matrix-6 run is read
#: against matrices 4 and 5 on everything except the encoder.
#:
#: `n_heads` is shared by the two branches, as it is for every other preset:
#: 8 divides CloneAtt's d_model 256 and ArmToken's d_token 64, so R26's head
#: count is usable by the arm branch unchanged.
#:
#: **It never beat ArmToken, and it is kept for the record, not as a
#: recommendation.** H0 family 0.893 +- 0.002 against AT0 family
#: 0.898 +- 0.002, and <= ArmToken in 4 of 4 configurations
#: (docs/CAMPAIGN_REPORT_2026-09-24.md, findings 5 and 8.3). That is the
#: answer to the narrow question above, and a negative answer is why this
#: preset stays in the file: it is the cleanest evidence the campaign has that
#: within-clone cross-arm structure carries nothing the per-arm moments threw
#: away. Like ARMTOKEN it has no `_published` twin.
#:
#: NOT a published model, so none of the traps apply.
HYBRID = ModelPreset(
    name="hybrid",
    paper_name="Hybrid-NPE",
    origin="cancer_sbi/models/hybrid.py (new, matrix 6)",
    data=DataConfig(
        dataset="clone_sets",
        top_k=100,
        batch_size=32,
        trial_filename="CNratios_all.pkl.gz",
    ),
    encoder=EncoderConfig(
        kind="hybrid",
        in_dim=45,
        # --- arm branch: ARMTOKEN's encoder, field for field ------------------
        d_token=64,
        d_arm=8,
        d_global=64,
        n_arm_layers=1,
        arm_num_inducing=16,
        trial_pool="mean",
        # --- clone branch: R26's CloneAtt -------------------------------------
        d_model=256,             # R26's width, not CloneAtt's published 128
        n_heads=8,               # divides 256 and 64, so both branches use it
        num_layers=3,            # ISABs over the clone set
        num_inducing=32,         # CloneAtt's published value, kept by R26
        freq_mode="feature",     # R26
        attn_scale="published",
        freq_as_weight=True,
        # The clone branch IS wrapped in TrialsSBIEmbedding, exactly as
        # CloneAtt's is, so these five are read (unlike ARMTOKEN, where they
        # are all None). The wrapper is what makes the clone branch produce the
        # same 256-wide context CloneAtt's flow sees.
        trials_aggregation_fn="mean",
        trials_num_hiddens=256,
        trials_num_layers=2,
        trials_output_dim=256,
        trials_aggregation_dim=1,
        # --- shared by both branches ------------------------------------------
        input_space="copy",
        attn_ln=True,
        dropout=0.2,
        attn_dropout_active=False,
        encoder_dropout_is_used=False,   # opt-in on both branches (trap 4 style)
    ),
    # z_score_x "structured" is matrices 2-5's finding; z_score_y stays "none"
    # for the same reason as in ARMTOKEN -- the context is these encoders' own
    # output and whitening it would flatten the per-arm scale.
    #
    # tail_bound is 5.0 IN THE PRESET, unlike every preset above it, which
    # leaves sbi's 3.0 and lets `--tail-bound 5` name it per run. Matrix 4b and
    # matrix 5 both established 5 as the value for this theta, and a hybrid run
    # that forgot the flag would not be comparable with either AT0 or R26. The
    # jobs script still passes `--tail-bound 5` so the two agree visibly.
    flow=FlowConfig(
        z_score_x="structured",
        z_score_y="none",
        dropout_probability=0.2,
        num_transforms=3,        # matrix 3's R12 finding, kept by AT0 and R26
        hidden_features=50,      # the published width; --flow-hidden-features widens it
        tail_bound=5.0,
    ),
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
        enforce_min_epochs=False,
        reload_best="on_early_stop",
        history_val_key="validation_loss",
        ckpt_dir="checkpoints",
    ),
)


#: Lookup by CLI name. The three ``*_published`` entries are the models as
#: published; the bare names are the repaired defaults (2026-09-25).
PRESETS = {
    CLONEMLP.name: CLONEMLP,
    CLONEATT.name: CLONEATT,
    DOMINANTCLONE.name: DOMINANTCLONE,
    ARMTOKEN.name: ARMTOKEN,
    HYBRID.name: HYBRID,
    CLONEMLP_PUBLISHED.name: CLONEMLP_PUBLISHED,
    CLONEATT_PUBLISHED.name: CLONEATT_PUBLISHED,
    DOMINANTCLONE_PUBLISHED.name: DOMINANTCLONE_PUBLISHED,
}

#: Suffix naming the published twin of a repaired preset.
PUBLISHED_SUFFIX = "_published"

#: The presets whose defaults changed on 2026-09-25, and which therefore have a
#: ``<name>_published`` twin. ``armtoken`` and ``hybrid`` are absent on purpose:
#: neither was ever published, so there is no earlier configuration to fall back
#: to and ``--published`` is an error for them rather than a no-op.
REPAIRED_PRESETS = (CLONEMLP.name, CLONEATT.name, DOMINANTCLONE.name)


def has_published_twin(name: str) -> bool:
    """Whether ``name`` names a model that also exists in a published version.

    Args:
        name: A preset name, repaired or published (case-insensitive).

    Returns:
        ``True`` for the three published models and for their ``_published``
        twins (which are their own); ``False`` for ``armtoken`` and ``hybrid``.
    """
    key = name.strip().lower()
    return key in REPAIRED_PRESETS or key in {
        n + PUBLISHED_SUFFIX for n in REPAIRED_PRESETS
    }


def published_preset_name(name: str) -> str:
    """The name of ``name``'s published twin.

    Args:
        name: A preset name (case-insensitive). A name that already ends in
            ``_published`` is its own twin and is returned unchanged.

    Returns:
        ``"<name>_published"``.

    Raises:
        KeyError: If the model has no published version. ``armtoken`` and
            ``hybrid`` were introduced by the 2026-09-24 campaign and were never
            published, so there is nothing to fall back to and asking for it is
            a mistake worth naming rather than a request to be silently granted.
    """
    key = name.strip().lower()
    if key.endswith(PUBLISHED_SUFFIX):
        return key
    twin = key + PUBLISHED_SUFFIX
    if twin not in PRESETS:
        raise KeyError(
            f"{name!r} has no published version: it was introduced by the "
            f"2026-09-24 campaign and never published. --published applies to "
            f"{', '.join(REPAIRED_PRESETS)} only."
        )
    return twin


def resolve_model_name(name: str, published: bool = False) -> str:
    """Map ``--model`` and ``--published`` onto the preset actually wanted.

    One function for all three entry points (``cli.train``, ``cli.evaluate`` and
    ``evaluation.sample_posteriors``) so that ``--published`` cannot mean three
    slightly different things.

    Args:
        name: The ``--model`` value.
        published: Whether ``--published`` was passed.

    Returns:
        ``name`` unchanged when ``published`` is false, otherwise its
        ``_published`` twin. ``--model clonemlp_published`` works with or
        without the flag.

    Raises:
        KeyError: If ``name`` is not a preset, or if ``published`` is asked for
            a model that has no published version.
    """
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown model {name!r}. Choose one of {sorted(PRESETS)}.")
    return published_preset_name(key) if published else key


# ---------------------------------------------------------------------------
# Serialising an effective config into, and out of, a checkpoint.
# ---------------------------------------------------------------------------

#: Key under which a checkpoint carries the config the run was actually trained
#: with. Checkpoints written before 2026-09-24 -- every one now on the cluster --
#: do not have it, and every reader must fall back to the preset's defaults with
#: a warning rather than refusing them.
EFFECTIVE_CONFIG_KEY = "effective_config"


def _block_to_dict(block: Any) -> Dict[str, Any]:
    """One config dataclass as a plain, picklable, JSON-safe dict.

    Args:
        block: Any of the five frozen sub-configs.

    Returns:
        ``{field name: value}`` with ``Path`` turned into ``str`` and ``tuple``
        into ``list``, so the result survives ``json.dumps`` as well as
        ``torch.save``. Every value is a scalar, a string or a list of those.
    """
    out: Dict[str, Any] = {}
    for item in fields(block):
        value = getattr(block, item.name)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        out[item.name] = value
    return out


def config_to_dict(preset: ModelPreset) -> Dict[str, Any]:
    """Snapshot a preset -- normally the *effective* one -- as a dict.

    This is what training stores in every checkpoint so that evaluation can
    rebuild the same network. Without it, evaluation rebuilds from
    :func:`get_preset` and a run trained with ``--z-score-x structured`` fails
    to load (the flow has a standardising layer the fresh one does not), while
    a run trained with ``--input-space copy`` or ``--freq-renorm`` loads
    silently into the wrong encoder and reports wrong numbers.

    Args:
        preset: The config the run actually used, i.e. the output of
            ``cli.train.build_config``, not ``get_preset(name)``.

    Returns:
        ``{"model": name, "data"/"encoder"/"flow"/"optim"/"train": {...},
        "prior_sd": float}``.
    """
    return {
        "model": preset.name,
        "paper_name": preset.paper_name,
        "origin": preset.origin,
        "data": _block_to_dict(preset.data),
        "encoder": _block_to_dict(preset.encoder),
        "flow": _block_to_dict(preset.flow),
        "optim": _block_to_dict(preset.optim),
        "train": _block_to_dict(preset.train),
        "prior_sd": preset.prior_sd,
    }


def _block_from_dict(cls: Any, stored: Dict[str, Any]) -> Any:
    """Rebuild one config dataclass from :func:`_block_to_dict`'s output.

    Args:
        cls: The dataclass to build.
        stored: Its stored fields. Unknown keys are ignored, which is what makes
            a checkpoint written by a *newer* tree still loadable by this one;
            missing keys keep the dataclass default.

    Returns:
        An instance of ``cls``.
    """
    known = {item.name: item for item in fields(cls)}
    kwargs = {}
    for name, value in stored.items():
        if name not in known:
            continue
        if isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def preset_from_effective_config(effective: Dict[str, Any]) -> ModelPreset:
    """Rebuild the architecture-bearing config a checkpoint was written with.

    The published preset is the base, so anything the snapshot does not carry
    keeps its published value; the ``flow`` and ``encoder`` blocks -- the two
    that decide the shape of the ``state_dict`` and what the encoder computes --
    come from the snapshot.

    Args:
        effective: A dict from :func:`config_to_dict`.

    Returns:
        The :class:`ModelPreset` to hand to
        ``cancer_sbi.training.trainer.build_training_components``.

    Raises:
        KeyError: If the snapshot names no model, or names an unknown one.
    """
    name = effective.get("model")
    if name is None:
        raise KeyError("The stored effective config carries no 'model' key.")
    base = get_preset(name)
    return replace(
        base,
        flow=_block_from_dict(FlowConfig, effective.get("flow", {})),
        encoder=_block_from_dict(EncoderConfig, effective.get("encoder", {})),
    )


def get_preset(name: str) -> ModelPreset:
    """Look a preset up by its short name.

    Args:
        name: One of ``"clonemlp"``, ``"cloneatt"``, ``"dominantclone"`` --
            which since 2026-09-25 name the REPAIRED configurations, not the
            published ones -- or one of ``"clonemlp_published"``,
            ``"cloneatt_published"``, ``"dominantclone_published"``, which are
            the models as published, or ``"armtoken"`` (matrix 5) or
            ``"hybrid"`` (matrix 6), neither of which reproduces an original
            folder and neither of which has a published twin (case-insensitive).

    Returns:
        The frozen :class:`ModelPreset`.

    Raises:
        KeyError: If ``name`` is not one of the eight presets.
    """
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown model {name!r}. Choose one of {sorted(PRESETS)}.")
    return PRESETS[key]


__all__ = [
    "EFFECTIVE_CONFIG_KEY",
    "ArmFeatureNorm",
    "AttnScale",
    "TrialPool",
    "FreqMode",
    "InputSpace",
    "ZScoreMode",
    "DataConfig",
    "EncoderConfig",
    "FlowConfig",
    "OptimConfig",
    "TrainConfig",
    "ModelPreset",
    "CLONEMLP",
    "CLONEATT",
    "DOMINANTCLONE",
    "CLONEMLP_PUBLISHED",
    "CLONEATT_PUBLISHED",
    "DOMINANTCLONE_PUBLISHED",
    "ARMTOKEN",
    "HYBRID",
    "PRESETS",
    "PUBLISHED_SUFFIX",
    "REPAIRED_PRESETS",
    "get_preset",
    "has_published_twin",
    "published_preset_name",
    "resolve_model_name",
    "config_to_dict",
    "preset_from_effective_config",
]
