#!/usr/bin/env python
"""Stage 1c: a post-hoc per-arm affine correction of the posterior, fitted on the VALIDATION split.

Every run in the 2026-09-24 matrix is miscalibrated in the same two boring ways: the posterior mean
carries a small consistent bias on the arms it learned best, and the posterior width is wrong by a
roughly constant factor. Both are one-parameter-per-arm defects, and both can be read off a held-out
set and undone without retraining anything.

The correction, per arm j, in units of that case's own posterior sd:

    fitted on the VALIDATION file    z_ij = (theta_true_ij - post_mean_ij) / post_std_ij
                                     a_j  = mean_i(z_ij)          (the bias, in sd units)
                                     b_j  = std_i(z_ij, ddof=0)   (the width factor)

    applied to the TEST file         samples'_isj = post_mean_ij + a_j * post_std_ij
                                                   + b_j * (samples_isj - post_mean_ij)

i.e. per case, slide the whole cloud of draws by a_j of its own sd and scale its spread about the
(old) centre by b_j. b_j > 1 means the posterior was too narrow on arm j and is widened. `--method
shift` sets b_j = 1 and corrects the bias only.

This is honest only because a_j and b_j are fitted on cases the test file does not contain: fitting
them on the test file itself would be reporting a two-parameter-per-arm fit as a calibration. Hence
``sample_posteriors.py --partition val`` and the ``meta.checkpoint`` guard below - the two files must
come from the same model, or a_j and b_j describe a different network.

What is written, in the exact stage-1 layout so ``poster_metrics.py`` reads it unchanged:
  * ``samples``       corrected as above
  * ``post_mean``     mean of the corrected draws        (sample_posteriors.py:342)
  * ``post_std``      std, ddof=0, of the corrected draws (sample_posteriors.py:343)
  * ``sbc_ranks``     ``(samples < theta_true).sum(axis=1)`` over the corrected draws
  * ``log_prob_true`` NaN. The corrected samples come from a different density than the flow's, and
                      the flow cannot evaluate it; a KDE of 5000 draws in 44 dimensions is not a
                      density either. ``poster_metrics`` reduces this column with ``np.nanmean``
                      (poster_metrics.py:375) and so reports NaN rather than crashing. Said in meta.
  * ``theta_true``, ``sim_ids``  copied from the test file, untouched.

    python -m cancer_sbi.evaluation.recalibrate_posteriors \
        --val  .../posteriors_armtoken_AT0_val.npz \
        --test .../posteriors_armtoken_AT0.npz \
        --out-dir .../AT0rc/posteriors --model armtoken --run-tag AT0rc

Then, unchanged:
    POST=<out-dir> OUT=<results dir> RUN_TAG=AT0rc sbatch jobs/analyze.sh
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # pragma: no cover - import side effect
    import cancer_sbi  # noqa: F401
except ImportError:  # pragma: no cover
    _root = os.environ.get("CANCER_SBI_ROOT") or str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, _root)
    import cancer_sbi  # noqa: F401,E402

#: Added to post_std before dividing, exactly as poster_metrics.zscores does, so a degenerate
#: arm (every draw identical) gives a finite z instead of an inf.
EPS = 1e-8

LOG_PROB_NOTE = "not recomputed after recalibration"


def load_run(path):
    """Read one stage-1 ``.npz`` into a dict with ``meta`` already decoded."""
    d = dict(np.load(path, allow_pickle=True))
    d["meta"] = json.loads(str(d.pop("meta_json")))
    d["path"] = str(path)
    return d


def zscores(theta_true, post_mean, post_std):
    """(truth - posterior mean) / posterior sd, poster_metrics' convention."""
    return (np.asarray(theta_true, dtype=np.float64)
            - np.asarray(post_mean, dtype=np.float64)) / (
                np.asarray(post_std, dtype=np.float64) + EPS)


def fit_affine(val, method="affine"):
    """Per-arm ``(a, b)`` from the validation file.

    Args:
        val: A dict from :func:`load_run`.
        method: ``"affine"`` fits both; ``"shift"`` fits ``a`` and pins ``b`` to 1.

    Returns:
        ``(a, b, z)`` - two ``(n_arms,)`` float64 arrays and the validation z-scores.
    """
    if method not in ("affine", "shift"):
        raise ValueError(f"--method must be 'affine' or 'shift', not {method!r}")
    z = zscores(val["theta_true"], val["post_mean"], val["post_std"])
    a = z.mean(axis=0)
    b = z.std(axis=0, ddof=0) if method == "affine" else np.ones(z.shape[1], dtype=np.float64)
    # A validation arm on which every case has the same z (n=1, or a dead arm) would give b=0 and
    # collapse the test posterior to a point. Leave the width alone instead of destroying it.
    bad = ~np.isfinite(b) | (b <= 0)
    if bad.any():
        print(f"[warn] {int(bad.sum())} arm(s) have a non-positive validation z spread; "
              "their b is set to 1 (width left unchanged)", flush=True)
        b = np.where(bad, 1.0, b)
    a = np.where(np.isfinite(a), a, 0.0)
    return a, b, z


def apply_affine(samples, post_mean, post_std, a, b):
    """The correction itself. Broadcasting: ``(n, S, J)`` against ``(n, 1, J)`` and ``(J,)``."""
    mean = np.asarray(post_mean, dtype=np.float64)[:, None, :]
    sd = np.asarray(post_std, dtype=np.float64)[:, None, :]
    s = np.asarray(samples, dtype=np.float64)
    return (mean + a[None, None, :] * sd + b[None, None, :] * (s - mean)).astype(np.float32)


def check_same_model(val, test, force=False):
    """Raise unless the two files were sampled from the same checkpoint."""
    cv, ct = val["meta"].get("checkpoint"), test["meta"].get("checkpoint")
    if cv == ct:
        return
    msg = (f"checkpoint mismatch: --val was sampled from {cv!r} and --test from {ct!r}. "
           "A correction fitted on one model's validation posteriors does not describe another "
           "model's test posteriors. Pass --force only if you know why these differ.")
    if force:
        print(f"[warn] {msg}", flush=True)
        return
    raise ValueError(msg)


def summarise(name, v):
    v = np.asarray(v, dtype=np.float64)
    return (f"  {name:<3} min {np.nanmin(v): .4f}   median {np.nanmedian(v): .4f}   "
            f"max {np.nanmax(v): .4f}")


def output_filename(model, run_tag=None):
    """``posteriors_<model>[_<tag>].npz`` - the name poster_metrics.posterior_path builds."""
    return f"posteriors_{model}{f'_{run_tag}' if run_tag else ''}.npz"


def recalibrate(val_path, test_path, out_dir, model, run_tag=None, method="affine", force=False):
    """Fit on ``val_path``, apply to ``test_path``, write the corrected file, return its path."""
    val = load_run(val_path)
    test = load_run(test_path)
    check_same_model(val, test, force=force)

    nj_val = int(np.asarray(val["theta_true"]).shape[1])
    nj_test = int(np.asarray(test["theta_true"]).shape[1])
    if nj_val != nj_test:
        raise ValueError(f"arm count mismatch: --val has {nj_val} arms, --test has {nj_test}")

    print(f"val   {val['path']}  cases={val['theta_true'].shape[0]}  "
          f"partition={val['meta'].get('partition')}  run_tag={val['meta'].get('run_tag')}",
          flush=True)
    print(f"test  {test['path']}  cases={test['theta_true'].shape[0]}  "
          f"partition={test['meta'].get('partition')}  run_tag={test['meta'].get('run_tag')}",
          flush=True)
    if val["meta"].get("partition") == "test":
        print("[warn] --val was sampled with --partition test: the correction is then fitted on "
              "the very cases it is scored on, which is not a calibration.", flush=True)

    a, b, z_val = fit_affine(val, method=method)
    print(f"method={method}  n_val={z_val.shape[0]}  arms={z_val.shape[1]}", flush=True)
    print(summarise("a", a), flush=True)
    print(summarise("b", b), flush=True)

    # The same correction read back on the validation set: by construction (z - a)/b, so mean|z|
    # should fall unless the arm was already calibrated. This is the fit's own residual, not a
    # held-out check -- the held-out check is what analyze.sh then reports on the test file.
    z_val_after = (z_val - a[None, :]) / b[None, :]
    print(f"val mean|z|  before {np.abs(z_val).mean():.4f}  ->  after "
          f"{np.abs(z_val_after).mean():.4f}", flush=True)
    print(f"val mean z   before {z_val.mean():+.4f}  ->  after {z_val_after.mean():+.4f}  |  "
          f"std z before {z_val.std(ddof=0):.4f} -> after {z_val_after.std(ddof=0):.4f}",
          flush=True)

    samples = apply_affine(test["samples"], test["post_mean"], test["post_std"], a, b)
    theta = np.asarray(test["theta_true"], dtype=np.float32)
    post_mean = samples.mean(axis=1).astype(np.float32)
    post_std = samples.std(axis=1, ddof=0).astype(np.float32)
    # sample_posteriors.py:346, replicated exactly. The +0.5/(S+1) step is poster_metrics'.
    sbc_ranks = (samples < theta[:, None, :]).sum(axis=1).astype(np.int32)

    meta = dict(test["meta"])
    meta.update({
        "model": model,
        "run_tag": run_tag,
        "recalibration": {
            "method": method,
            "fitted_on": str(val_path),
            "n_val": int(z_val.shape[0]),
            "a": [float(x) for x in a],
            "b": [float(x) for x in b],
            "val_mean_abs_z_before": float(np.abs(z_val).mean()),
            "val_mean_abs_z_after": float(np.abs(z_val_after).mean()),
            "forced_checkpoint_mismatch": bool(force and val["meta"].get("checkpoint")
                                               != test["meta"].get("checkpoint")),
        },
        "log_prob_true": LOG_PROB_NOTE,
        "built_with": "cancer_sbi.evaluation.recalibrate_posteriors",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename(model, run_tag))
    np.savez_compressed(
        out_path,
        theta_true=theta,
        samples=samples,
        post_mean=post_mean,
        post_std=post_std,
        sbc_ranks=sbc_ranks,
        log_prob_true=np.full(theta.shape[0], np.nan, dtype=np.float32),
        sim_ids=np.asarray(test["sim_ids"]),
        meta_json=json.dumps(meta),
    )
    print(f"recalibrated  {out_path}  cases={samples.shape[0]}  draws={samples.shape[1]}  "
          f"({os.path.getsize(out_path) / 1e6:.0f} MB)", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True,
                    help="stage-1 .npz of the VALIDATION split (sample_posteriors --partition val) "
                         "- the correction is fitted on this")
    ap.add_argument("--test", required=True,
                    help="stage-1 .npz of the test split - the correction is applied to this")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True,
                    help="model name for the output file: posteriors_<model>[_<tag>].npz")
    ap.add_argument("--run-tag", default=None,
                    help="label for the recalibrated file, e.g. AT0rc. Pass the same --run-tag to "
                         "poster_metrics / analyze.sh")
    ap.add_argument("--method", choices=["affine", "shift"], default="affine",
                    help="affine (default) corrects bias and width; shift corrects bias only")
    ap.add_argument("--force", action="store_true",
                    help="proceed even if the two files name different checkpoints")
    args = ap.parse_args(argv)
    return recalibrate(args.val, args.test, args.out_dir, args.model, args.run_tag,
                       args.method, args.force)


if __name__ == "__main__":
    main()
