"""Train one of the three published models.

Replaces ``Base_NPE/main.py``, ``SetTransformer_NPE/main.py`` and
``Plain_NPE/main.py``, which were three copies of the same eleven lines with a
different import at the top.

Example:
    Train CloneMLP-NPE exactly as the paper did, writing into ``runs/clonemlp``::

        python -m cancer_sbi.cli.train \\
            --model clonemlp \\
            --data-root /path/to/simulation_outputs \\
            --split /path/to/train_test_split.pkl \\
            --out runs/clonemlp

    Training resumes automatically from ``<ckpt-dir>/latest.pt`` if that file
    exists, so re-running the same command continues the run rather than
    starting over. Point ``--out`` somewhere new to start from scratch.

Everything that is not a path keeps the value the published run used; the flags
that override them exist for new experiments, not for reproducing old ones.

Repair flags (added 2026-09-24, work package A). **Every one of them defaults to
the published behaviour**, so a command line that does not mention them produces
the run it produced before:

``--z-score-x {none,structured,independent}``
    sbi's whitening mode for theta (``FlowConfig.z_score_x``). This is the one
    value the R0-vs-R1 comparison turns on; before this flag existed the ``flow``
    block was never replaced, so it could only be changed by editing
    ``config.py``. Default: the preset's (``none`` for clonemlp/cloneatt,
    ``structured`` for dominantclone).
``--input-space {log2,copy}``
    Both clone-set models. ``copy`` converts the log2 ratios to copy-number
    space *inside the encoder* (repair T2; run R2 for CloneMLP, runs R6-R8 for
    CloneAtt) rather than in the loader, which they share. Default: ``log2``,
    the published behaviour.
``--freq-renorm``
    CloneAtt only. Renormalise the per-clone frequency weights instead of
    multiplying tokens by a raw ~0.003 frequency (run R4). ``ln`` stays off.
    Default: off, the published behaviour.
``--freq-mode {weight,feature}``
    CloneAtt only. ``feature`` drops the frequency multiply altogether and
    feeds ``log10(freq)`` to the input projection as a 45th column (runs
    R5-R8); R4 showed that even a renormalised multiply leaves every token at
    ~1/K of its scale. Refused together with ``--freq-renorm``. Default:
    ``weight``, the published behaviour.
``--attn-ln``
    CloneAtt only. LayerNorm in every MAB/ISAB/PMA (runs R5-R8). Default: off,
    which is trap 5.
``--encoder-dropout P``
    Embedding-net dropout. CloneMLP: overrides the published 0.2. CloneAtt:
    the value *and* the layers, which the published encoder never built (trap
    4, run R8). Ignored by DominantClone, whose DeepSet has no dropout.
``--embed-lr X`` / ``--embed-weight-decay X`` / ``--flow-weight-decay X``
    The two optimiser groups' rates and decays (runs R3, R7). Two-group models
    only; DominantClone optimises one group (trap 2). Default: the preset's.
``--require-all-trials``
    DominantClone only. Keep only the sims that have every trial file, i.e. the
    clone-set models' sim set, so the three models are compared on the same
    simulations. Default: off, the published NaN-pad-and-keep behaviour.
``--num-workers``
    ``DataLoader`` worker processes. Prefetch only -- shuffling stays on the
    main process's generator, so the RNG stream is unchanged. Default: 0.
``--cache-dir``
    Pre-built clone cache (``src/utilities/build_clone_cache.py``) to read the
    per-sim tensors from instead of the gzipped trial files. Clone-set models
    only; the dominant-clone builder ignores it. Default: unset.
``--deterministic``
    ``torch.use_deterministic_algorithms(True)`` plus
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, set before CUDA is initialised.
    Default: off.

Every checkpoint written here carries the *effective* config under the
``effective_config`` key (``config.config_to_dict`` plus the paths and the flags
as given), so ``evaluation/sample_posteriors.py`` and ``cli/evaluate.py`` rebuild
the network this run trained rather than the preset's defaults. Checkpoints from
before 2026-09-24 do not have it; those readers fall back to the preset and say
so.

Validation: if the split pickle carries a ``val_ids`` key, that third set is
used for early stopping and ``test_ids`` is never looked at during training. If
it does not, the historical behaviour is kept -- early stopping on the test set
-- and a warning says so on every run.

Heavy imports happen inside :func:`main` so that ``--help`` works without torch.
"""

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from cancer_sbi.cli import (
    DATA_ROOT_ENV,
    SPLIT_ENV,
    add_data_root_argument,
    add_device_argument,
    add_model_argument,
    add_split_argument,
    default_run_dir,
    require_path,
    resolve_device,
)
from cancer_sbi.config import ModelPreset, config_to_dict, get_preset

#: File name CloneAtt's original training loop pickled its density estimator to
#: (``SetTransformer_NPE/inference_model.py:246``). Kept so the artefact keeps
#: its familiar name; see the note in ``Trainer._dump_final_pickle`` about why
#: the new file is not interchangeable with the old one.
CLONEATT_PICKLE_NAME = "SetTransformer_NPE_Freq_mean.pkl"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for ``python -m cancer_sbi.cli.train``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cancer_sbi.cli.train",
        description=(
            "Train CloneMLP-NPE, CloneAtt-NPE or DominantClone-NPE on simulated "
            "tumour data. Reproduces the corresponding old folder exactly; the "
            "only thing that changed is that paths are arguments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_argument(parser)
    add_data_root_argument(parser)
    add_split_argument(parser)
    add_device_argument(parser)

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Run directory. Checkpoints go to <out>/<checkpoint dir name>. "
            "Defaults to runs/<model> under the current directory."
        ),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help=(
            "Where checkpoints are written, overriding <out>/<name>. The "
            "per-model default name is the one that model's original code used: "
            "'checkpoints_baseline' for clonemlp, 'checkpoints' for the other "
            "two. Note that clonemlp's evaluation scripts read 'checkpoints' "
            "even though its training wrote 'checkpoints_baseline' -- see the "
            "open questions in docs/REFACTOR_NOTES.md."
        ),
    )

    training = parser.add_argument_group(
        "training overrides",
        "Defaults are the published values. Change them only for new runs.",
    )
    training.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Hard stop on the number of epochs (published runs: 200).",
    )
    training.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help=(
            "Earliest epoch at which early stopping may fire (published runs: "
            "50). Ignored by cloneatt, whose original loop does not check it."
        ),
    )
    training.add_argument(
        "--stop-after-epochs",
        type=int,
        default=None,
        help="Patience: epochs without improvement before stopping (50).",
    )
    training.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Simulations per batch (32 in all three published runs).",
    )
    training.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Clones kept per trial, clonemlp and cloneatt only (100). Ignored "
            "for dominantclone, which keeps one profile per trial."
        ),
    )
    training.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed torch and numpy before anything is built. The published runs "
            "were NOT seeded, and leaving this unset reproduces that; passing a "
            "seed makes future runs repeatable but cannot reproduce an old one."
        ),
    )
    training.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader worker processes (published runs: 0, single-process). "
            "Workers only prefetch -- shuffling stays on the main process's "
            "generator -- so raising this does not move the results."
        ),
    )
    training.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Read clone tensors from a pre-built cache directory (see "
            "src/utilities/build_clone_cache.py) instead of the gzipped trial "
            "files. Bit-identical batches, just faster. Clone-set models only: "
            "the dominant-clone loader ignores it."
        ),
    )
    training.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Ask torch for deterministic kernels and set "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA starts. Off by "
            "default, which is what the published runs did."
        ),
    )
    training.add_argument(
        "--quiet",
        action="store_true",
        help="Do not mirror the per-epoch line into the logging module.",
    )

    repairs = parser.add_argument_group(
        "repair switches",
        "Added 2026-09-24. Each one defaults to the PUBLISHED behaviour, so "
        "omitting all of them reproduces the run this tree produced before.",
    )
    repairs.add_argument(
        "--z-score-x",
        choices=["none", "structured", "independent"],
        default=None,
        help=(
            "sbi's whitening mode for theta. The single value separating R0 "
            "from R1. Default: the preset's (none for clonemlp and cloneatt, "
            "structured for dominantclone)."
        ),
    )
    repairs.add_argument(
        "--input-space",
        choices=["log2", "copy"],
        default=None,
        help=(
            "Clone-set models only: the space the per-clone projection sees. "
            "'copy' applies the T2 conversion inside the encoder, identically "
            "for both (runs R2 and R6-R8). Default: log2, the published "
            "behaviour. Ignored by dominantclone."
        ),
    )
    repairs.add_argument(
        "--freq-mode",
        choices=["weight", "feature"],
        default=None,
        help=(
            "CloneAtt only: what the clone frequency is for. 'weight' is the "
            "published token multiply; 'feature' drops it and feeds "
            "log10(freq) to the input projection as a 45th column instead "
            "(runs R5-R8). Cannot be combined with --freq-renorm, which then "
            "has nothing to renormalise. Default: weight."
        ),
    )
    repairs.add_argument(
        "--attn-ln",
        action="store_true",
        help=(
            "CloneAtt only: build every MAB/ISAB/PMA with ln=True (runs "
            "R5-R8). Default: off, which is trap 5 -- no LayerNorm anywhere "
            "in the attention stack."
        ),
    )
    repairs.add_argument(
        "--encoder-dropout",
        type=float,
        default=None,
        help=(
            "Dropout probability inside the embedding net. clonemlp: "
            "overrides the published 0.2 between the MLP's hidden layers. "
            "cloneatt: the same value, AND wires it up at all -- the "
            "published encoder accepts dropout and discards it (trap 4), so "
            "this flag is what turns run R8 on. Ignored by dominantclone, "
            "whose DeepSet has no dropout layer. Default: the preset's."
        ),
    )
    repairs.add_argument(
        "--embed-lr",
        type=float,
        default=None,
        help=(
            "Learning rate of the embedding-net parameter group (published: "
            "1e-4, which is near-frozen for an attention encoder -- run R7 "
            "raises it). Two-group models only; ignored by dominantclone, "
            "which optimises every parameter in one group (trap 2)."
        ),
    )
    repairs.add_argument(
        "--embed-weight-decay",
        type=float,
        default=None,
        help=(
            "Weight decay of the embedding-net parameter group (published: "
            "0.0; run R3 uses 1e-4). Two-group models only."
        ),
    )
    repairs.add_argument(
        "--flow-weight-decay",
        type=float,
        default=None,
        help=(
            "Weight decay of the flow parameter group (published: 1e-4; run "
            "R3 uses 1e-3 against the epoch-15 overfit). Two-group models "
            "only."
        ),
    )
    repairs.add_argument(
        "--require-all-trials",
        action="store_true",
        help=(
            "DominantClone only: train, validate and test on the sims that "
            "have every trial file, i.e. exactly the clone-set models' sim "
            "set. Default: off, the published behaviour (missing trials are "
            "NaN-padded and the sim is kept). Ignored by the other two models, "
            "which already apply this rule."
        ),
    )
    repairs.add_argument(
        "--freq-renorm",
        action="store_true",
        default=None,
        help=(
            "CloneAtt only: renormalise the per-clone frequency weights "
            "instead of multiplying tokens by the raw frequency (run R4). "
            "LayerNorm stays off. Default: off, the published behaviour."
        ),
    )

    pickling = parser.add_mutually_exclusive_group()
    pickling.add_argument(
        "--final-pickle",
        type=Path,
        default=None,
        help=(
            "Pickle the trained density estimator here when training ends. "
            f"cloneatt did this by default, writing {CLONEATT_PICKLE_NAME} into "
            "the current directory; that default is kept, inside <out>."
        ),
    )
    pickling.add_argument(
        "--no-final-pickle",
        action="store_true",
        help="Skip the final pickle even for cloneatt.",
    )

    return parser


def build_config(
    args: argparse.Namespace,
    preset: "ModelPreset",
    data_root: Path,
    split_path: Path,
    ckpt_dir: Path,
) -> "ModelPreset":
    """Fold the command line into a preset, leaving unmentioned fields alone.

    This is the whole argument-to-config path, pulled out of :func:`main` so it
    can be tested without touching a GPU, a dataset or a checkpoint directory.
    That matters more than it looks: the two bugs this function fixes -- the
    unconditional ``seed=args.seed`` and the ``flow`` block never being replaced
    -- both lived *here*, in the mapping from flags to config, and a test that
    asserted on the presets alone would have passed while every run ignored the
    flags. See docs/CODEBASE_IMPROVEMENT_PLAN.md, "Testing".

    Args:
        args: Parsed arguments from :func:`build_parser`.
        preset: The published model to derive from, from
            :func:`cancer_sbi.config.get_preset`.
        data_root: Directory holding the ``sim*/`` folders.
        split_path: The split pickle.
        ckpt_dir: Where checkpoints are written (already resolved against
            ``--out`` by the caller).

    Returns:
        A new frozen :class:`~cancer_sbi.config.ModelPreset`. Every field the
        command line did not mention keeps the preset's published value.
    """
    # --- assemble the config -------------------------------------------------
    data_cfg = replace(
        preset.data,
        root_dir=data_root,
        split_path=split_path,
        batch_size=args.batch_size if args.batch_size is not None else preset.data.batch_size,
        top_k=(
            args.top_k
            if args.top_k is not None and preset.data.top_k is not None
            else preset.data.top_k
        ),
        num_workers=(
            args.num_workers if args.num_workers is not None else preset.data.num_workers
        ),
        cache_dir=(
            getattr(args, "cache_dir", None)
            if getattr(args, "cache_dir", None)
            else preset.data.cache_dir
        ),
        # Only the dominant-clone path has anything to opt into; recording the
        # flag on a clone-set preset would put a value in the checkpoint that
        # nothing applied. The warning below says so.
        require_all_trials=(
            bool(args.require_all_trials) and preset.data.dataset == "dominant_clone"
        )
        or preset.data.require_all_trials,
    )
    train_cfg = replace(
        preset.train,
        max_epochs=args.max_epochs if args.max_epochs is not None else preset.train.max_epochs,
        min_epochs=args.min_epochs if args.min_epochs is not None else preset.train.min_epochs,
        stop_after_epochs=(
            args.stop_after_epochs
            if args.stop_after_epochs is not None
            else preset.train.stop_after_epochs
        ),
        ckpt_dir=str(ckpt_dir),
        # Trap 21 / B1: this line used to be the bare `seed=args.seed`, which
        # overwrote the preset's seed with None whenever --seed was absent --
        # so TrainConfig.seed was a decorative field that no run could reach.
        seed=args.seed if args.seed is not None else preset.train.seed,
        log_progress=not args.quiet,
    )
    # The `flow` block was never replaced here, which is why z_score_x -- the
    # one value the R0-vs-R1 comparison turns on -- could previously only be
    # changed by editing config.py. Replaced only when the flag is given, so
    # the preset's value is still the default.
    flow_cfg = (
        replace(preset.flow, z_score_x=args.z_score_x)
        if args.z_score_x is not None
        else preset.flow
    )
    # --freq-renorm scales the weights of a multiply that --freq-mode feature
    # removes, so the pair describes no network at all. Refused rather than
    # resolved: silently dropping either half is how a run gets launched
    # believing it carries a repair it does not.
    if getattr(args, "freq_mode", None) == "feature" and args.freq_renorm:
        raise ValueError(
            "--freq-renorm and --freq-mode feature contradict each other: "
            "'feature' removes the frequency multiply entirely, so there are "
            "no per-clone weights left to renormalise. Pass one or the other."
        )

    encoder_cfg = preset.encoder
    # Both clone-set encoders read input_space now (the same formula in both);
    # only DeepSet ignores it, and a value recorded there would describe a
    # transform nothing applied.
    if args.input_space is not None and preset.encoder.kind != "deepset":
        encoder_cfg = replace(encoder_cfg, input_space=args.input_space)
    if args.freq_renorm:
        encoder_cfg = replace(encoder_cfg, freq_renorm=True)
    # The three attention-only switches, and the optimiser group overrides, are
    # applied only where they are read. A value recorded on a preset that
    # ignores it would ride into the checkpoint's effective config and describe
    # a network nothing built -- the same reason --require-all-trials is gated
    # on the dataset above.
    is_attention = preset.encoder.kind == "attention"
    if getattr(args, "freq_mode", None) is not None and is_attention:
        encoder_cfg = replace(encoder_cfg, freq_mode=args.freq_mode)
    if getattr(args, "attn_ln", False) and is_attention:
        encoder_cfg = replace(encoder_cfg, attn_ln=True)
    if getattr(args, "encoder_dropout", None) is not None and preset.encoder.kind != "deepset":
        encoder_cfg = replace(encoder_cfg, dropout=args.encoder_dropout)
        if is_attention:
            # Trap 4: CloneAtt's encoder accepts `dropout` and discards it, so
            # setting the probability is not enough -- the run has to say that
            # the layers should exist at all. clonemlp already applies it.
            encoder_cfg = replace(encoder_cfg, attn_dropout_active=True)
    optim_cfg = preset.optim
    if preset.optim.use_param_groups:
        if getattr(args, "embed_lr", None) is not None:
            optim_cfg = replace(optim_cfg, embed_lr=args.embed_lr)
        if getattr(args, "embed_weight_decay", None) is not None:
            optim_cfg = replace(optim_cfg, embed_weight_decay=args.embed_weight_decay)
        if getattr(args, "flow_weight_decay", None) is not None:
            optim_cfg = replace(optim_cfg, flow_weight_decay=args.flow_weight_decay)
    cfg = replace(
        preset,
        data=data_cfg,
        train=train_cfg,
        flow=flow_cfg,
        encoder=encoder_cfg,
        optim=optim_cfg,
    )

    if args.input_space is not None and preset.encoder.kind not in ("mlp", "attention"):
        print(f"[warn] --input-space is not used by {preset.name}; ignoring it.")
    if args.freq_renorm and preset.encoder.kind != "attention":
        print(f"[warn] --freq-renorm is not used by {preset.name}; ignoring it.")
    if getattr(args, "freq_mode", None) is not None and preset.encoder.kind != "attention":
        print(f"[warn] --freq-mode is not used by {preset.name}; ignoring it.")
    if getattr(args, "attn_ln", False) and preset.encoder.kind != "attention":
        print(f"[warn] --attn-ln is not used by {preset.name}; ignoring it.")
    # DeepSet builds no nn.Dropout at all (models/deep_set.py), so there is
    # nothing for the probability to reach on the dominant-clone path.
    if getattr(args, "encoder_dropout", None) is not None and preset.encoder.kind == "deepset":
        print(f"[warn] --encoder-dropout is not used by {preset.name}; ignoring it.")
    # Trap 2: dominantclone optimises every parameter in a single group, so the
    # two per-group fields are not read at all on that path.
    for flag, value in (
        ("--embed-lr", getattr(args, "embed_lr", None)),
        ("--embed-weight-decay", getattr(args, "embed_weight_decay", None)),
        ("--flow-weight-decay", getattr(args, "flow_weight_decay", None)),
    ):
        if value is not None and not preset.optim.use_param_groups:
            print(f"[warn] {flag} is not used by {preset.name}; ignoring it.")

    if args.require_all_trials and preset.data.dataset != "dominant_clone":
        print(f"[warn] --require-all-trials is not used by {preset.name}; ignoring it.")

    if args.top_k is not None and preset.data.top_k is None:
        print(f"[warn] --top-k is not used by {preset.name}; ignoring it.")

    return cfg


def effective_config_payload(
    cfg: "ModelPreset",
    args: argparse.Namespace,
    data_root: Path,
    split_path: Path,
) -> dict:
    """The config snapshot every checkpoint of this run carries.

    Evaluation rebuilds the network from this instead of from
    :func:`~cancer_sbi.config.get_preset`. That is not a convenience: an R1
    checkpoint (``z_score_x="structured"``) has a standardising layer in the
    flow that a preset-built network does not, so loading it raises on the
    state-dict keys; an R2 or R4 checkpoint loads *silently* into an encoder
    without the copy-space transform or the frequency renormalisation, and every
    number that comes out is wrong without anything saying so.

    Args:
        cfg: The effective preset from :func:`build_config`.
        args: The parsed command line, recorded verbatim so a checkpoint says
            which flags produced it.
        data_root: Resolved ``--data-root``.
        split_path: Resolved ``--split``.

    Returns:
        A picklable dict: the five config blocks, the model name and prior sd
        (all from :func:`~cancer_sbi.config.config_to_dict`), plus the paths and
        the repair flags as given.
    """
    payload = config_to_dict(cfg)
    payload["data_root"] = str(data_root)
    payload["split_path"] = str(split_path)
    payload["seed"] = cfg.train.seed
    payload["cli_flags"] = {
        "z_score_x": args.z_score_x,
        "input_space": args.input_space,
        "freq_renorm": bool(args.freq_renorm),
        # Matrix 2. Recorded as given, next to the blocks config_to_dict has
        # already snapshotted, so a checkpoint says both what was asked for and
        # what the run actually used.
        "freq_mode": getattr(args, "freq_mode", None),
        "attn_ln": bool(getattr(args, "attn_ln", False)),
        "encoder_dropout": getattr(args, "encoder_dropout", None),
        "embed_lr": getattr(args, "embed_lr", None),
        "embed_weight_decay": getattr(args, "embed_weight_decay", None),
        "flow_weight_decay": getattr(args, "flow_weight_decay", None),
        "require_all_trials": bool(args.require_all_trials),
        "num_workers": args.num_workers,
        "cache_dir": args.cache_dir,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "deterministic": bool(args.deterministic),
    }
    payload["written_by"] = "cancer_sbi.cli.train"
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    """Run the training command.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success.
    """
    args = build_parser().parse_args(argv)

    # Set before torch is imported, let alone before CUDA is initialised:
    # cuBLAS reads CUBLAS_WORKSPACE_CONFIG when its handle is created, and a
    # value set afterwards is ignored while
    # torch.use_deterministic_algorithms(True) still demands it. This is why
    # the assignment sits above the local imports rather than next to them.
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Local imports: see the module docstring. Everything below needs torch.
    import torch

    from cancer_sbi.data.loaders import (
        build_clone_set_dataloaders,
        build_dominant_clone_dataloaders,
    )
    from cancer_sbi.data.splits import load_split
    from cancer_sbi.training.trainer import (
        Trainer,
        build_training_components,
        quirks_for,
        seed_everything,
    )

    if args.deterministic:
        # Not cudnn.deterministic: there is no convolution anywhere in these
        # three models, so that knob would be decorative.
        torch.use_deterministic_algorithms(True)

    data_root = require_path(args.data_root, "--data-root", DATA_ROOT_ENV)
    split_path = require_path(args.split, "--split", SPLIT_ENV)
    device = resolve_device(args.device)

    preset = get_preset(args.model)
    out_dir = Path(args.out) if args.out is not None else default_run_dir(preset.name)
    # Trap 11 lives here: the directory NAME comes from the preset, which is the
    # name that model's own training code used. The parent is the user's --out,
    # so two runs of the same model never collide.
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / preset.train.ckpt_dir

    cfg = build_config(args, preset, data_root, split_path, ckpt_dir)

    # --- data ----------------------------------------------------------------
    print(f"Using device: {device}")
    split = load_split(split_path)
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    val_ids = split.get("val_ids")
    print(
        f"Split: {len(train_ids)} train sims, "
        f"{len(val_ids) if val_ids is not None else 0} val sims, "
        f"{len(test_ids)} test sims"
    )

    if cfg.data.dataset == "clone_sets":
        train_loader, val_loader, test_loader = build_clone_set_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            top_k=cfg.data.top_k,
            batch_size=cfg.data.batch_size,
            pin_memory=cfg.data.pin_memory,
            num_workers=cfg.data.num_workers,
            cache_dir=cfg.data.cache_dir,
        )
    else:
        train_loader, val_loader, test_loader = build_dominant_clone_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            batch_size=cfg.data.batch_size,
            pin_memory=cfg.data.pin_memory,
            num_workers=cfg.data.num_workers,
            require_all_trials=cfg.data.require_all_trials,
        )

    # Preserved from all three */main.py:33 only when there is no third split:
    # the TEST loader is then passed as the validation loader, so "validation
    # loss" and "test loss" are the same number and early stopping selects the
    # epoch on the very sims the score is reported on. With a `val_ids` key in
    # the split pickle that stops being true. See docs/REFACTOR_NOTES.md.
    if val_loader is None:
        val_loader = test_loader
        print(
            "[warn] The split has no 'val_ids', so early stopping runs on the "
            "TEST set -- best.pt is chosen on the sims the score is reported "
            "on. Rebuild the split with a validation set before comparing runs."
        )

    components = build_training_components(
        cfg, train_loader, device=device, log_progress=cfg.train.log_progress
    )

    # --- the final pickle, cloneatt only by default --------------------------
    final_pickle: Optional[Path] = None
    if args.final_pickle is not None:
        final_pickle = Path(args.final_pickle)
    elif not args.no_final_pickle and cfg.name == "cloneatt":
        final_pickle = out_dir / CLONEATT_PICKLE_NAME

    trainer = Trainer(
        density_estimator=components.density_estimator,
        train_loader=train_loader,
        val_loader=val_loader,
        optim_cfg=cfg.optim,
        train_cfg=cfg.train,
        embedding_net=components.embedding_net,
        dataset=cfg.data.dataset,
        device=device,
        ckpt_dir=ckpt_dir,
        quirks=quirks_for(cfg.name),
        final_pickle_path=final_pickle,
        # Every checkpoint this run writes carries the config it was trained
        # with, so evaluation rebuilds this network and not the preset's.
        effective_config=effective_config_payload(cfg, args, data_root, split_path),
    )

    # Second seeding, deliberately AFTER construction, per
    # CODEBASE_IMPROVEMENT_PLAN.md "Seeding -- the correct change", item 2.
    # seed_everything also runs inside build_training_components (before
    # build_embedding_net), which is where weight initialisation needs it and
    # where it must stay -- this call does not replace it.
    #
    # Fresh run: re-seed here so the training loop's RNG stream starts from the
    # seed. Resumed run: Trainer.__init__ has just restored the checkpoint's
    # saved RNG state (checkpoints.py:257-260), so re-seeding would clobber it
    # and replay the stream instead of continuing it -- skip it.
    if not trainer.resumed:
        seed_everything(cfg.train.seed)
    elif cfg.train.seed is not None:
        print(
            "[note] Resumed from a checkpoint: keeping its saved RNG state "
            "instead of re-seeding with --seed, so the run continues its "
            "stream rather than replaying it."
        )

    print(
        f"Training {cfg.paper_name} (was {cfg.origin}) -> {ckpt_dir}\n"
        f"  max_epochs={cfg.train.max_epochs}, "
        f"stop_after_epochs={cfg.train.stop_after_epochs}, "
        f"min_epochs={'enforced' if cfg.train.enforce_min_epochs else 'NOT enforced'}, "
        f"reload_best={cfg.train.reload_best}"
    )

    trainer.train()

    val_key = cfg.train.history_val_key
    print(
        f"Done after {trainer.epoch} epochs. "
        f"best {val_key}={trainer.best_val_loss:.6f}. "
        f"Checkpoints in {ckpt_dir} (best.pt, latest.pt, ckpt_epoch_*.pt)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
