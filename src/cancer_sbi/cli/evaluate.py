"""Evaluate a trained model on the held-out simulations.

Replaces the evaluation scripts that were duplicated across the three model
folders -- ``<model>/z-score_violin.py`` (the source of the paper's figures),
``<model>/z-score.py``, ``SetTransformer_NPE/kde_prior_vs_posterior.py``,
``<model>/ppc-plot.py`` and the SBC cells of ``Base_NPE/test.ipynb``.

Example:
    Reproduce the z-score figures for CloneMLP-NPE::

        python -m cancer_sbi.cli.evaluate \\
            --model clonemlp \\
            --data-root /path/to/simulation_outputs \\
            --split /path/to/train_test_split.pkl \\
            --ckpt runs/clonemlp/checkpoints/best.pt \\
            --out-dir runs/clonemlp/evaluation

    Add ``--kde``, ``--pairplot`` or ``--sbc`` for the other figure families.
    ``--sbc`` is slow: it draws 1000 posterior samples for every test case.

What it writes (all names unchanged from the originals): the global z-score
histogram, the three per-arm bar charts, ``zscore_summary.txt``, the two violin
figures, the two true-versus-posterior-mean scatter grids and
``correlation_true_vs_postmean.xlsx``. Each figure is written as both .png and
.pdf, as before.

Heavy imports happen inside :func:`main` so that ``--help`` works without torch.
"""

import argparse
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

#: Checkpoint directory name every evaluation script read, on every model
#: (``Base_NPE/z-score_violin.py:105``, ``SetTransformer_NPE/z-score_violin.py``,
#: ``Plain_NPE/z-score_violin.py:42``). For clonemlp this is NOT the directory
#: training wrote to -- trap 11, and open question (b) in docs/REFACTOR_NOTES.md.
EVAL_CKPT_DIRNAME = "checkpoints"

#: Sub-directory of the run directory that the figures go to. The originals
#: wrote into a directory of their own next to the script; this default keeps a
#: run's outputs together instead.
DEFAULT_OUTPUT_DIRNAME = "evaluation"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for ``python -m cancer_sbi.cli.evaluate``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cancer_sbi.cli.evaluate",
        description=(
            "Draw posteriors for every held-out simulation and write the "
            "calibration figures and tables. The model architecture is rebuilt "
            "from --model and the weights come from --ckpt, so both must refer "
            "to the same model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_argument(parser)
    add_data_root_argument(parser)
    add_split_argument(parser)
    add_device_argument(parser)

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "The run directory used for training. Only used to derive the "
            "defaults of --ckpt and --out-dir. Defaults to runs/<model>."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=(
            "Checkpoint to evaluate. Defaults to "
            f"<run-dir>/{EVAL_CKPT_DIRNAME}/best.pt, which is the directory the "
            "original evaluation scripts read for every model -- note that "
            "clonemlp's training wrote to checkpoints_baseline/ instead, so for "
            "that model the default here is deliberately NOT where "
            "'cancer_sbi.cli.train --model clonemlp' puts its files. Pass the "
            "path you mean."
        ),
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help=(
            "Directory the rebuilt model tries to resume from before --ckpt is "
            "loaded over it. Defaults to the model's own training directory "
            "name, reproducing what the original scripts did by constructing "
            "the training class. Irrelevant to the result; it only decides "
            "which 'no checkpoint found' message you see."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Where figures and tables go. Defaults to <run-dir>/{DEFAULT_OUTPUT_DIRNAME}.",
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--num-posterior-samples",
        type=int,
        default=5000,
        help="Posterior samples per test case (5000 in every published figure).",
    )
    sampling.add_argument(
        "--prior-sd",
        type=float,
        default=None,
        help=(
            "Standard deviation of the reference prior, per arm. Defaults to "
            "the simulator's 0.2. The original scripts used sqrt(0.2)=0.447, "
            "which only ever affected the prior curve drawn in the KDE figures "
            "-- pass 0.4472135954999579 to redraw those exactly."
        ),
    )
    sampling.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N test cases. For a quick smoke test; the "
            "published figures used all of them."
        ),
    )

    extras = parser.add_argument_group("optional figure families")
    extras.add_argument(
        "--kde",
        action="store_true",
        help=(
            "Also draw the prior-versus-posterior KDE figures. Keeps every "
            "case's samples in memory (about 130 MB for 150 cases)."
        ),
    )
    extras.add_argument(
        "--pairplot",
        type=int,
        nargs="?",
        const=1,
        default=None,
        help=(
            "Also draw the sbi pairplot for this test-case index (the original "
            "ppc-plot.py used index 1)."
        ),
    )
    extras.add_argument(
        "--sbc",
        action="store_true",
        help=(
            "Also run simulation-based calibration and draw the rank, KS and "
            "c2st figures. Slow."
        ),
    )
    extras.add_argument(
        "--sbc-samples",
        type=int,
        default=1000,
        help="Posterior samples per case inside SBC (the notebook used 1000).",
    )
    extras.add_argument(
        "--show-backend",
        action="store_true",
        help=(
            "Do not force matplotlib's non-interactive Agg backend. The "
            "original scripts forced it because they ran on a headless cluster."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the evaluation command.

    Args:
        argv: Argument list; ``None`` means ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success.
    """
    args = build_parser().parse_args(argv)

    # Local imports: see the module docstring. Everything below needs torch.
    import numpy as np
    import torch

    from cancer_sbi.arms import ARM_LABELS
    from cancer_sbi.data.loaders import (
        build_clone_set_dataloaders,
        build_dominant_clone_dataloaders,
    )
    from cancer_sbi.data.splits import load_split
    from cancer_sbi.evaluation import diagnostics, figures, posterior as posterior_mod, style
    from cancer_sbi.training.trainer import Trainer, build_training_components, quirks_for

    data_root = require_path(args.data_root, "--data-root", DATA_ROOT_ENV)
    split_path = require_path(args.split, "--split", SPLIT_ENV)
    device = resolve_device(args.device)

    preset = get_preset(args.model)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(preset.name)
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / EVAL_CKPT_DIRNAME / "best.pt"
    resume_dir = Path(args.resume_dir) if args.resume_dir else run_dir / preset.train.ckpt_dir
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / DEFAULT_OUTPUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trap 11 made visible instead of silently resolved.
    if resume_dir.name != ckpt_path.parent.name:
        print(
            f"[note] {preset.name} trains into '{resume_dir.name}/' but this "
            f"command reads '{ckpt_path.parent.name}/best.pt', which is what the "
            f"original evaluation scripts did. Pass --ckpt to change it."
        )

    if not args.show_backend:
        # Base_NPE/z-score_violin.py:16 -- headless cluster, no display.
        style.use_agg_backend()
    style.use_paper_style()

    print(f"Using device: {device}")
    train_ids, test_ids = load_split(split_path)

    if preset.data.dataset == "clone_sets":
        train_loader, test_loader = build_clone_set_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            test_ids=test_ids,
            top_k=preset.data.top_k,
            batch_size=preset.data.batch_size,
            pin_memory=preset.data.pin_memory,
        )
    else:
        train_loader, test_loader = build_dominant_clone_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            test_ids=test_ids,
            batch_size=preset.data.batch_size,
            pin_memory=preset.data.pin_memory,
        )

    x_all, theta_all = posterior_mod.collect_test_tensors(
        test_loader, preset.data.dataset
    )
    print(f"x_all shape: {tuple(x_all.shape)}, theta_all shape: {tuple(theta_all.shape)}")

    # Rebuild the architecture exactly as training built it, then load the
    # weights over it. Constructing the Trainer also attempts a resume from
    # <resume-dir>/latest.pt, which is what the originals did by constructing
    # their InferenceModel class (Base_NPE/z-score_violin.py:99-103).
    components = build_training_components(
        preset, train_loader, device=device, log_progress=True
    )
    trainer = Trainer(
        density_estimator=components.density_estimator,
        train_loader=train_loader,
        val_loader=test_loader,
        optim_cfg=preset.optim,
        train_cfg=preset.train,
        embedding_net=components.embedding_net,
        dataset=preset.data.dataset,
        device=device,
        ckpt_dir=resume_dir,
        quirks=quirks_for(preset.name),
    )

    resumed = posterior_mod.load_checkpoint_for_eval(
        ckpt_path,
        trainer.density_estimator,
        trainer.optimizer,
        device=device,
        history_val_key=preset.train.history_val_key,
    )
    print(f"Best {preset.train.history_val_key}: {resumed.best_val_loss:.6f}")

    density_estimator = trainer.density_estimator
    density_estimator.eval()

    prior_sd = args.prior_sd if args.prior_sd is not None else preset.prior_sd
    prior = posterior_mod.build_prior(prior_sd=prior_sd, device=device)
    post = posterior_mod.build_posterior(density_estimator, prior)
    print(f"Prior: Normal(0, {prior_sd}) per arm, 44 arms")

    indices = (
        list(range(min(args.limit_cases, x_all.shape[0])))
        if args.limit_cases is not None
        else None
    )
    need_samples = args.kde or args.pairplot is not None

    evaluation = posterior_mod.summarise_test_set(
        post,
        x_all,
        theta_all,
        num_posterior_samples=args.num_posterior_samples,
        device=device,
        keep_samples=need_samples,
        indices=indices,
    )
    stacked = evaluation.stacked
    theta_used = theta_all if indices is None else theta_all[indices]
    print(f"z shape: {tuple(stacked.z.shape)}, post_mean shape: {tuple(stacked.post_mean.shape)}")

    # --- the core artefacts, in the order z-score_violin.py produced them ----
    pooled = diagnostics.pooled_z_summary(stacked.z)
    per_parameter = diagnostics.per_parameter_z_summary(stacked.z)

    figures.plot_global_z_histogram(stacked.z, out_dir)
    figures.plot_per_parameter_z_bars(per_parameter, out_dir, labels=ARM_LABELS)
    figures.write_zscore_summary(
        pooled,
        per_parameter,
        out_dir,
        labels=ARM_LABELS,
        # The originals computed the per-case L2 error and threw it away
        # (Base_NPE/z-score_violin.py:163). It is written into the summary here;
        # that adds a section, it does not change any existing number.
        l2=stacked.l2,
    )
    figures.plot_violin_zscores(stacked.z, out_dir, labels=ARM_LABELS)
    figures.plot_true_vs_postmean_scatter(
        theta_used, stacked.post_mean, out_dir, labels=ARM_LABELS
    )
    table = diagnostics.correlation_table(theta_used, stacked.post_mean, labels=ARM_LABELS)
    figures.write_correlation_excel(table, out_dir)
    figures.plot_l2_per_case(stacked.l2, out_dir)

    counts = diagnostics.calibration_counts(per_parameter.mean_z, per_parameter.std_z)
    print(
        f"Calibration: mean z={pooled.mean_z:.4f}, std z={pooled.std_z:.4f}, "
        f"{pooled.pct_within_2:.1f}% of |z| within 2 | "
        f"{counts.n_well_calibrated} arms well calibrated, "
        f"{counts.n_biased} biased, {counts.n_overconfident} overconfident | "
        f"mean L2={float(np.mean(stacked.l2)):.4f}"
    )

    # --- optional families ---------------------------------------------------
    if args.kde and evaluation.samples is not None:
        pooled_samples = np.concatenate(
            [s.numpy() for s in evaluation.samples], axis=0
        )
        n_test = len(evaluation.samples)
        figures.plot_kde_pooled_only(
            pooled_samples, n_test, out_dir, prior_std=prior_sd, labels=ARM_LABELS
        )
        selected = diagnostics.select_diverse_test_cases(theta_used, n_selected=5)
        figures.plot_kde_full(
            [evaluation.samples[i].numpy() for i in selected],
            pooled_samples,
            theta_used,
            selected,
            n_test,
            out_dir,
            prior_std=prior_sd,
            labels=ARM_LABELS,
        )

    if args.pairplot is not None and evaluation.samples is not None:
        case = args.pairplot
        if 0 <= case < len(evaluation.samples):
            figures.plot_posterior_pairplot(
                evaluation.samples[case],
                # Keep the batch dimension: the pairplot indexes theta as
                # theta[:, dims], exactly as Base_NPE/ppc-plot.py:124 did.
                theta_used[case : case + 1],
                out_dir,
                test_index=case,
                labels=ARM_LABELS,
            )
        else:
            print(f"[warn] --pairplot {case} is outside the evaluated cases; skipped.")

    if args.sbc:
        print(f"Running SBC with {args.sbc_samples} posterior samples per case...")
        x_sbc = x_all if indices is None else x_all[indices]
        ranks, dap_samples = diagnostics.run_sbc_ranks(
            theta_used,
            x_sbc,
            post,
            num_posterior_samples=args.sbc_samples,
            device=device,
        )
        stats = diagnostics.check_sbc_stats(
            ranks, theta_used, dap_samples, num_posterior_samples=args.sbc_samples
        )
        figures.plot_sbc_rank_hist_panels(ranks, args.sbc_samples, out_dir, labels=ARM_LABELS)
        figures.plot_sbc_rank_cdf(ranks, args.sbc_samples, out_dir, labels=ARM_LABELS)
        figures.plot_sbc_ks_pvalues(stats["ks_pvals"], out_dir, labels=ARM_LABELS)
        figures.plot_sbc_c2st_ranks(stats["c2st_ranks"], out_dir, labels=ARM_LABELS)
        figures.plot_sbc_c2st_dap(stats["c2st_dap"], out_dir, labels=ARM_LABELS)

    print(f"Wrote figures and tables to {out_dir}")
    # Keep torch's device bookkeeping honest on CUDA: the samples are large.
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
