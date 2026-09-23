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

Heavy imports happen inside :func:`main` so that ``--help`` works without torch.
"""

import argparse
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
from cancer_sbi.config import get_preset

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
        "--quiet",
        action="store_true",
        help="Do not mirror the per-epoch line into the logging module.",
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


def main(argv: Optional[List[str]] = None) -> int:
    """Run the training command.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success.
    """
    args = build_parser().parse_args(argv)

    # Local imports: see the module docstring. Everything below needs torch.
    from cancer_sbi.data.loaders import (
        build_clone_set_dataloaders,
        build_dominant_clone_dataloaders,
    )
    from cancer_sbi.data.splits import load_split
    from cancer_sbi.training.trainer import (
        Trainer,
        build_training_components,
        quirks_for,
    )

    data_root = require_path(args.data_root, "--data-root", DATA_ROOT_ENV)
    split_path = require_path(args.split, "--split", SPLIT_ENV)
    device = resolve_device(args.device)

    preset = get_preset(args.model)
    out_dir = Path(args.out) if args.out is not None else default_run_dir(preset.name)
    # Trap 11 lives here: the directory NAME comes from the preset, which is the
    # name that model's own training code used. The parent is the user's --out,
    # so two runs of the same model never collide.
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out_dir / preset.train.ckpt_dir

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
        seed=args.seed,
        log_progress=not args.quiet,
    )
    cfg = replace(preset, data=data_cfg, train=train_cfg)

    if args.top_k is not None and preset.data.top_k is None:
        print(f"[warn] --top-k is not used by {preset.name}; ignoring it.")

    # --- data ----------------------------------------------------------------
    print(f"Using device: {device}")
    train_ids, test_ids = load_split(split_path)
    print(f"Split: {len(train_ids)} train sims, {len(test_ids)} test sims")

    if cfg.data.dataset == "clone_sets":
        train_loader, test_loader = build_clone_set_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            test_ids=test_ids,
            top_k=cfg.data.top_k,
            batch_size=cfg.data.batch_size,
            pin_memory=cfg.data.pin_memory,
        )
    else:
        train_loader, test_loader = build_dominant_clone_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            test_ids=test_ids,
            batch_size=cfg.data.batch_size,
            pin_memory=cfg.data.pin_memory,
        )

    # Preserved from all three */main.py:33: the TEST loader is passed as the
    # validation loader. There is no third split -- "validation loss" and "test
    # loss" are the same number, computed on the sims the model never trains on.
    # Early stopping therefore selects on the test set. See docs/REFACTOR_NOTES.md.
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
        val_loader=test_loader,
        optim_cfg=cfg.optim,
        train_cfg=cfg.train,
        embedding_net=components.embedding_net,
        dataset=cfg.data.dataset,
        device=device,
        ckpt_dir=ckpt_dir,
        quirks=quirks_for(cfg.name),
        final_pickle_path=final_pickle,
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
