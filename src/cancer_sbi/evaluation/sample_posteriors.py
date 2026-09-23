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
from cancer_sbi.config import PRESETS, get_preset  # noqa: E402

N_ARMS = 44

#: Checkpoint directory every original evaluation script read, on every model. For clonemlp this
#: is NOT the directory training wrote to (checkpoints_baseline/) - trap 11. Preserved.
EVAL_CKPT_DIRNAME = "checkpoints"

#: The other directory a best.pt can live in. Only used to warn, exactly as the legacy script did.
ALT_CKPT_DIRNAME = "checkpoints_baseline"


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


def main():
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
    args = ap.parse_args()

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

    train_ids, test_ids = load_split(split_path)
    print(f"split: {len(train_ids)} train ids / {len(test_ids)} test ids", flush=True)

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

    x_all, theta_all = posterior_mod.collect_test_tensors(test_loader, preset.data.dataset)
    n_total = int(x_all.shape[0])
    sim_ids = test_sim_ids(test_loader.dataset, n_total)
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
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

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
        "device": device,
        "data_root": str(data_root),
        "split_file": str(split_path),
        "n_train_ids": int(len(train_ids)),
        "n_test_ids": int(len(test_ids)),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t0, 1),
        "built_with": "cancer_sbi",
    }

    out_path = os.path.join(out_dir, f"posteriors_{args.model}.npz")
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
