#!/usr/bin/env python
"""
Stage 1 of the evaluation: draw the posterior ONCE per held-out tumour and save the raw samples.

Everything downstream (true R^2, Pearson r^2, expected coverage, posterior contraction, z-scores,
SBC ranks) is then a numpy recompute on the saved file instead of a fresh GPU job. That is the whole
point: today every diagnostic re-runs its own 5000-draw loop and keeps nothing.

PORTED VERSION. The legacy script (kept beside this one as sample_posteriors_legacy.py) rebuilt
each model by os.chdir()-ing into Base_NPE/, SetTransformer_NPE/ or Plain_NPE/ and importing the
trainer class that lives there, which tied the whole evaluation to those folders and to their
relative paths ("../Guassian_Normal/simulation_outputs", "train_test_split.pkl"). This version
builds the identical network through the verified `cancer_sbi` package instead, and takes every
path as an explicit argument:

  legacy                                   ported
  ---------------------------------------  --------------------------------------------------
  os.chdir(Base_NPE); import inference_model  cancer_sbi.config.get_preset("clonemlp")
  InferenceModelBaseline(train_loader,...)  build_training_components(preset, train_loader, ...)
                                            + build_optimizer(...)
  inference_model.load_checkpoint(tmp)      posterior.load_checkpoint_for_eval(ckpt, ...)
  utils.build_dataloader(...)               cancer_sbi.data.loaders.build_*_dataloaders(...)
  DATA_ROOT = "../Guassian_Normal/..."      --data-root / $CANCER_SBI_DATA_ROOT
  spec["split"] = "train_test_split.pkl"    --split      / $CANCER_SBI_SPLIT
  "checkpoints/best.pt" under the folder    --ckpt       (default $CANCER_SBI_RUNS/<model>/checkpoints/best.pt)

Everything that reaches the output file is unchanged: the same dataloaders (same root_dir,
batch_size, top_k, shuffle flags), the same split, the same checkpoint loaded with the same
RNG-key-popping tempfile dance, the same DirectPosterior over an independent Normal prior, the same
per-case 5000-draw loop in the same order, and the same .npz keys, dtypes and shapes.

Two deliberate differences from the legacy z-score scripts, both documented in the evaluation plan
and both already inherited by the legacy version of this file:
  1. prior sd is 0.2 (the simulator's actual value), not sqrt(0.2)=0.447. This is inert for
     sampling - a Normal's support is all of R^44 so DirectPosterior never rejects - but it is the
     correct value and it matters if log_prob is ever normalised.
  2. the run is seeded, so it can be repeated. The legacy training and evaluation are unseeded.

The seeding order is load-bearing and is preserved exactly: the model is constructed and the
checkpoint is loaded FIRST, and torch/numpy are seeded only afterwards. Every original constructor
ended in _try_resume(), which restores the RNG state saved by the last training epoch; seeding
before that would be overwritten and the draws would differ.

    python sample_posteriors.py --model clonemlp \
        --data-root ~/cancer/Guassian_Normal/simulation_outputs \
        --split     ~/cancer/Base_NPE/train_test_split.pkl \
        --ckpt      ~/cancer/Base_NPE/checkpoints/best.pt \
        --out-dir   ~/cancer/evaluation/out
    ... --limit 4      # smoke test
    ... --partition val  # sample the VALIDATION ids instead, into ..._val.npz

`--partition val` exists for one purpose: `recalibrate_posteriors.py` fits its per-arm affine
correction on a held-out set that is NOT the test set. The default is `test` and every byte of
that path - loaders, file name, meta - is unchanged.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- make `cancer_sbi` importable without depending on the working directory ---------------
# The package is a plain directory (~/cancer/cancer_sbi), not a pip install, so its parent has
# to be on sys.path. This file lives at src/cancer_sbi/evaluation/, so parents[2] is src/, the
# right answer; $CANCER_SBI_ROOT overrides it. This is the only path guess in the file and it
# does not depend on cwd, unlike the os.chdir() the legacy script did.
try:  # pragma: no cover - import side effect
    import cancer_sbi  # noqa: F401
except ImportError:
    _root = os.environ.get("CANCER_SBI_ROOT") or str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, _root)
    import cancer_sbi  # noqa: F401,E402

from cancer_sbi.cli import (  # noqa: E402
    DATA_ROOT_ENV,
    SPLIT_ENV,
    default_run_dir,
    require_path,
    resolve_device,
)
from cancer_sbi.config import PRESETS, config_to_dict, get_preset  # noqa: E402

N_ARMS = 44

#: Checkpoint directory every original evaluation script read, on every model. For clonemlp this
#: is NOT the directory training wrote to (checkpoints_baseline/) - trap 11. Preserved.
EVAL_CKPT_DIRNAME = "checkpoints"

#: The other directory a best.pt can live in. Only used to warn, exactly as the legacy script did.
ALT_CKPT_DIRNAME = "checkpoints_baseline"


def output_filename(model, limit=None, run_tag=None, partition="test"):
    """Name of the .npz this run writes: ``posteriors_<model>[_<tag>][_val].npz``.

    A full run keeps the plain name, which is the only name
    ``poster_metrics.py`` discovers (:312-323) - so per-run outputs belong in
    per-run ``--out-dir``s, not in suffixed filenames in a shared directory.

    The suffix exists for the two cases where a file must NOT be mistaken for a
    real run:
      * ``--limit N`` -> ``_limit<N>``. A smoke test used to overwrite the real
        posteriors of the same model, and the partial file is then silently
        skipped by the poster export guard (poster_metrics.py:378-380) rather
        than flagged.
      * ``--run-tag TAG`` -> ``_<TAG>``, for labelling a file by run (R0/R1/...)
        when one really must share a directory.

    An explicit ``--run-tag`` wins over the ``--limit`` suffix.

    ``partition="val"`` appends a further ``_val``, AFTER whichever suffix the
    two rules above chose. A validation-split file is not a run's result - it is
    the input a recalibration is fitted on - so it must never land on the name
    ``poster_metrics.posterior_path`` discovers, even when the run is tagged.
    ``partition="test"`` (the default) changes nothing.

    Args:
        model: Preset name, e.g. ``"clonemlp"``.
        limit: The ``--limit`` value, or None for a full run.
        run_tag: The ``--run-tag`` value, or None.
        partition: ``"test"`` (default) or ``"val"``.

    Returns:
        The file name (no directory).
    """
    if run_tag:
        suffix = f"_{run_tag}"
    elif limit is not None:
        suffix = f"_limit{int(limit)}"
    else:
        suffix = ""
    if partition == "val":
        suffix += "_val"
    elif partition != "test":
        raise ValueError(f"partition must be 'test' or 'val', not {partition!r}")
    return f"posteriors_{model}{suffix}.npz"


def test_sim_ids(dataset, n_expected):
    """The held-out simulation names, in loader order. Empty list if they cannot be recovered.

    Unchanged from the legacy script: cancer_sbi's dataset classes expose the same two attributes
    the original ones did - CNASimsDataset.items (each a dict with "sim_dir") and
    SimulationDataset.sim_dirs.
    """
    ids = []
    if hasattr(dataset, "items"):                 # CNASimsDataset: the post-filter list
        ids = [os.path.basename(it["sim_dir"]) for it in dataset.items]
    elif hasattr(dataset, "sim_dirs"):            # SimulationDataset: __getitem__ returns None for bad sims
        for i, d in enumerate(dataset.sim_dirs):
            try:
                if dataset[i] is not None:
                    ids.append(os.path.basename(d))
            except Exception:
                pass
    if len(ids) != n_expected:
        print(f"[warn] recovered {len(ids)} sim ids for {n_expected} test cases - storing indices instead")
        return []
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PRESETS))
    ap.add_argument("--data-root", type=Path, default=os.environ.get(DATA_ROOT_ENV),
                    help="folder holding sim1/, sim2/, ... "
                         f"Falls back to ${DATA_ROOT_ENV}.")
    ap.add_argument("--split", type=Path, default=os.environ.get(SPLIT_ENV),
                    help="pickle holding {'train_ids': [...], 'test_ids': [...]}. "
                         f"Falls back to ${SPLIT_ENV}.")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/cancer/evaluation/out"))
    ap.add_argument("--num-samples", type=int, default=5000,
                    help="posterior draws per test case (the legacy scripts use 5000)")
    ap.add_argument("--prior-sd", type=float, default=0.2)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint to evaluate. Default "
                         f"<$CANCER_SBI_RUNS|runs>/<model>/{EVAL_CKPT_DIRNAME}/best.pt - the "
                         "directory the published figures were made from.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test cases (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    override = ap.add_argument_group(
        "architecture overrides (old checkpoints only)",
        "A checkpoint written since 2026-09-24 carries the config it was trained "
        "with and the network is rebuilt from that. These flags describe a "
        "checkpoint that carries none; passing one that contradicts a checkpoint "
        "which does is an error, not a silent choice.")
    override.add_argument("--z-score-x", choices=["none", "structured", "independent"],
                          default=None, help="FlowConfig.z_score_x of the run being evaluated.")
    override.add_argument("--input-space", choices=["log2", "copy"], default=None,
                          help="CloneMLP encoder input space of the run being evaluated.")
    override.add_argument("--freq-renorm", action="store_true", default=None,
                          help="The run being evaluated used CloneAtt's frequency renormalisation.")
    override.add_argument("--require-all-trials", action="store_true", default=None,
                          help="The DominantClone run being evaluated was trained on the clone-set "
                               "models' sim set (every trial file present). The test loader is then "
                               "built with the same restriction.")
    ap.add_argument("--run-tag", default=None,
                    help="label appended to the output file: posteriors_<model>_<TAG>.npz. "
                         "Only needed when two runs of one model share an --out-dir; the "
                         "per-run convention is a per-run --out-dir instead, because "
                         "poster_metrics.py only discovers the untagged name.")
    ap.add_argument("--partition", choices=["test", "val"], default="test",
                    help="which held-out split to sample. 'test' (default) is the reported "
                         "one and is unchanged. 'val' samples the VALIDATION ids instead and "
                         "writes posteriors_<model>[_<tag>]_val.npz - the file "
                         "recalibrate_posteriors.py fits its affine correction on, so that the "
                         "correction is never fitted on the cases it is scored on.")
    args = ap.parse_args(argv)

    # Heavy imports after the arguments parse, as the rest of the package does, so that
    # --help works on a machine that cannot import torch.
    import numpy as np
    import torch

    from cancer_sbi.data.loaders import (
        build_clone_set_dataloaders,
        build_dominant_clone_dataloaders,
    )
    from cancer_sbi.data.splits import load_split
    from cancer_sbi.evaluation import posterior as posterior_mod
    from cancer_sbi.training.trainer import build_optimizer, build_training_components

    preset = get_preset(args.model)
    data_root = require_path(args.data_root, "--data-root", DATA_ROOT_ENV)
    split_path = require_path(args.split, "--split", SPLIT_ENV)
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = (
        Path(os.path.expanduser(str(args.ckpt))) if args.ckpt is not None
        else default_run_dir(preset.name) / EVAL_CKPT_DIRNAME / "best.pt"
    )

    device = resolve_device(args.device)
    print(f"model={args.model}  origin={preset.origin}  device={device}  seed={args.seed}", flush=True)

    # load_split returns a dict. Evaluation scores the TEST ids by default, so
    # `val_ids` is not passed to the builders and the middle element of the
    # 3-tuple they return is None -- unchanged. --partition val is the one
    # exception: it passes val_ids and takes that middle loader instead, so a
    # post-hoc recalibration can be fitted on held-out cases that are NOT the
    # ones it will be scored on.
    split = load_split(split_path)
    train_ids = split["train_ids"]
    test_ids = split["test_ids"]
    val_ids = split.get("val_ids")
    if args.partition == "val" and (val_ids is None or len(val_ids) == 0):
        raise SystemExit(
            f"--partition val needs a split with val_ids, and {split_path} has none.\n"
            "  Carve one first:  python -m cancer_sbi.cli.make_split --add-val --frac 0.1 "
            "--in <split.pkl> --out <train_val_test_split.pkl>")
    print(f"split: {len(train_ids)} train ids / {len(test_ids)} test ids"
          + (f" / {len(val_ids)} val ids" if val_ids is not None else ""), flush=True)
    print(f"partition: {args.partition}", flush=True)

    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # The architecture comes from the checkpoint, not from get_preset: an R1
    # checkpoint (z_score_x="structured") will not load into a preset-built flow
    # at all, and an R2/R4 one loads silently into the wrong encoder. Resolved
    # before the model is built, because it decides what is built -- and before
    # the loaders, because data.require_all_trials decides which sims the test
    # loader holds.
    eval_cfg = posterior_mod.resolve_eval_config(
        ckpt_path,
        args.model,
        z_score_x=args.z_score_x,
        input_space=args.input_space,
        freq_renorm=args.freq_renorm,
        require_all_trials=args.require_all_trials,
        device=device,
    )
    preset = eval_cfg.preset

    # val_ids is handed to the builders only for --partition val, so the default
    # path builds exactly the loaders it always did.
    builder_val_ids = val_ids if args.partition == "val" else None
    if preset.data.dataset == "clone_sets":
        train_loader, val_loader, test_loader = build_clone_set_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            test_ids=test_ids,
            top_k=preset.data.top_k,
            batch_size=preset.data.batch_size,
            pin_memory=preset.data.pin_memory,
            val_ids=builder_val_ids,
        )
    else:
        train_loader, val_loader, test_loader = build_dominant_clone_dataloaders(
            root_dir=str(data_root),
            train_ids=train_ids,
            test_ids=test_ids,
            batch_size=preset.data.batch_size,
            pin_memory=preset.data.pin_memory,
            require_all_trials=eval_cfg.require_all_trials,
            val_ids=builder_val_ids,
        )

    eval_loader = test_loader if args.partition == "test" else val_loader
    if eval_loader is None:
        raise SystemExit(
            f"--partition {args.partition}: the loader builder returned no validation loader "
            f"for {len(val_ids) if val_ids is not None else 0} val ids")

    x_all, theta_all = posterior_mod.collect_test_tensors(eval_loader, preset.data.dataset)
    n_total = int(x_all.shape[0])
    sim_ids = test_sim_ids(eval_loader.dataset, n_total)
    n = n_total if args.limit is None else min(args.limit, n_total)
    print(f"held-out cases: {n_total}" + ("" if args.limit is None else f" (using first {n})"), flush=True)
    print(f"X {tuple(x_all.shape)}  theta {tuple(theta_all.shape)}", flush=True)

    # The model has to be constructed before its weights can be loaded: the architecture is built
    # from a dummy batch of the train loader, exactly as each original __init__ did. The optimiser
    # is built too because the checkpoint carries its state and load_checkpoint restores it.
    components = build_training_components(preset, train_loader, device=device, log_progress=True)
    optimizer = build_optimizer(components.density_estimator, components.embedding_net, preset.optim)

    # CloneMLP trains into checkpoints_baseline/ but every published figure was evaluated from
    # checkpoints/. Keep that default for continuity, and say so when both exist.
    model_root = ckpt_path.parent.parent
    others = [
        str(c)
        for c in (model_root / EVAL_CKPT_DIRNAME / "best.pt", model_root / ALT_CKPT_DIRNAME / "best.pt")
        if c.exists() and os.path.abspath(c) != os.path.abspath(ckpt_path)
    ]
    if others:
        print(f"[warn] another checkpoint also exists and was NOT used: {others} "
              f"(pass --ckpt to choose it)", flush=True)
    resumed = posterior_mod.load_checkpoint_for_eval(
        ckpt_path,
        components.density_estimator,
        optimizer,
        device=device,
        history_val_key=preset.train.history_val_key,
    )
    best_val_loss = float(resumed.best_val_loss)
    print(f"checkpoint {ckpt_path}  epoch={resumed.epoch}  best_val_loss={best_val_loss:.6f}", flush=True)

    density_estimator = components.density_estimator
    density_estimator.eval()

    # seed AFTER construction: every original constructor ended in _try_resume(), which restores
    # the RNG state saved by the last training epoch and would otherwise overwrite this
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    prior = posterior_mod.build_prior(prior_sd=float(args.prior_sd), dim=N_ARMS, device=device)
    posterior = posterior_mod.build_posterior(density_estimator, prior)

    s = args.num_samples
    samples_out = np.empty((n, s, N_ARMS), dtype=np.float32)
    log_prob_true = np.full(n, np.nan, dtype=np.float32)
    log_prob_failed = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(n):
            x = x_all[i:i + 1].to(device)
            theta_true = theta_all[i].to(device)

            draws = posterior.sample((s,), x=x, show_progress_bars=False).detach()
            if draws.dim() == 3:                   # (S, 1, 44) -> (S, 44)
                draws = draws.squeeze(1)
            samples_out[i] = draws.cpu().numpy().astype(np.float32)

            try:
                lp = posterior.log_prob(theta_true.reshape(1, N_ARMS), x=x, norm_posterior=False)
                log_prob_true[i] = float(np.asarray(lp.detach().cpu()).ravel()[0])
            except Exception:
                log_prob_failed += 1

            if (i + 1) % 25 == 0 or i + 1 == n:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{n} cases   {rate:.2f} cases/s", flush=True)

    if log_prob_failed:
        print(f"[warn] log_prob unavailable for {log_prob_failed}/{n} cases (stored as NaN)", flush=True)

    theta_np = theta_all[:n].cpu().numpy().astype(np.float32)
    post_mean = samples_out.mean(axis=1)
    post_std = samples_out.std(axis=1, ddof=0)
    # SBC rank = how many of the S draws fall below the truth, per arm; uniform if calibrated
    # rank in {0..S}: S+1 possible outcomes, so the quantile position divides by S+1
    sbc_ranks = (samples_out < theta_np[:, None, :]).sum(axis=1).astype(np.int32)

    _as_dict = config_to_dict(preset)
    meta = {
        "model": args.model,
        "folder": preset.origin,
        "checkpoint": str(ckpt_path),
        "checkpoints_not_used": others,
        "checkpoint_epoch": int(resumed.epoch),
        "best_val_loss": best_val_loss,
        "n_cases": int(n),
        "n_cases_available": int(n_total),
        "num_samples": int(s),
        "prior_sd": float(args.prior_sd),
        "seed": int(args.seed),
        "limit": None if args.limit is None else int(args.limit),
        "run_tag": args.run_tag,
        "partition": args.partition,
        "device": device,
        "data_root": str(data_root),
        "split_file": str(split_path),
        "n_train_ids": int(len(train_ids)),
        "n_test_ids": int(len(test_ids)),
        "n_val_ids": None if val_ids is None else int(len(val_ids)),
        # The architecture these samples came from, so a .npz can be read back
        # years later without the checkpoint beside it.
        "flow_config": _as_dict["flow"],
        "encoder_config": _as_dict["encoder"],
        "config_from_checkpoint": bool(eval_cfg.from_checkpoint),
        "require_all_trials": bool(eval_cfg.require_all_trials),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "built_with": "cancer_sbi",
    }

    out_path = os.path.join(
        out_dir, output_filename(args.model, args.limit, args.run_tag, args.partition))
    np.savez_compressed(
        out_path,
        theta_true=theta_np,
        samples=samples_out,
        post_mean=post_mean.astype(np.float32),
        post_std=post_std.astype(np.float32),
        sbc_ranks=sbc_ranks,
        log_prob_true=log_prob_true,
        sim_ids=np.array(sim_ids[:n] if sim_ids else [f"idx{i}" for i in range(n)]),
        meta_json=json.dumps(meta),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nsaved -> {out_path}  ({size_mb:.0f} MB, {time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
