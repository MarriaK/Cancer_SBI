#!/usr/bin/env python
"""Combine several seeds of one run into a single "ensemble" posterior file.

Stage 1 (``sample_posteriors.py``) writes one ``posteriors_<model>[_<tag>].npz`` per
checkpoint. Training the same configuration under three seeds gives three such files,
each a different approximation of the same posterior. The honest way to report them
together is not to average the metrics but to evaluate the equal-weight MIXTURE

    q(theta | x) = (1/M) * sum_m q_m(theta | x)

which is sampled exactly by pooling the members' draws for that case. This script does
that pooling and rewrites every derived array from the pooled draws, in the layout
``poster_metrics.py`` and ``fig_shrinkage.py`` already read - so the ensemble is just
another ``--run-tag`` to them and no metric script changes.

    python -m cancer_sbi.evaluation.ensemble_posteriors \
        --inputs R12s0.npz R12s1.npz R12s2.npz \
        --out-dir ~/cancer/results/2026-09-24/R12ens/posteriors \
        --model cloneatt --run-tag R12ens
    ... --num-samples 5000     # subsample to 5000 draws, an equal share from each member

Then, unchanged:
    POST=<out-dir> OUT=<results dir> RUN_TAG=R12ens sbatch jobs/analyze.sh

What is recomputed, and what is copied:
  * ``samples``     pooled over members (axis 1), per case.
  * ``post_mean``   mean of the pooled draws          (sample_posteriors.py:342)
  * ``post_std``    std, ddof=0, of the pooled draws  (sample_posteriors.py:343)
  * ``sbc_ranks``   ``(samples < theta_true).sum(axis=1)`` over the pooled draws
                    (sample_posteriors.py:346). The +0.5/(S+1) convention lives in
                    ``poster_metrics.load_run`` and divides by ``meta["num_samples"]``,
                    so the pooled count MUST reach meta - it does, below.
  * ``log_prob_true`` the mixture density at the truth: logsumexp over members minus
                    log M. This is exact, not an approximation, because the members are
                    equally weighted; NaN if any member's value is NaN.
  * ``theta_true``, ``sim_ids``  copied; they must be identical across members (checked).
  * ``meta_json``   member A's meta, plus ``ensemble_of``/``ensemble_seeds``, with
                    ``num_samples`` updated and ``checkpoint``/``checkpoint_epoch``/
                    ``seed`` turned into lists.
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


def load_member(path):
    """Read one stage-1 ``.npz`` into a dict with ``meta`` already decoded."""
    d = dict(np.load(path, allow_pickle=True))
    d["meta"] = json.loads(str(d.pop("meta_json")))
    d["path"] = str(path)
    return d


def check_aligned(members):
    """Raise unless every member scores the same cases, in the same order.

    Pooling draws from two files that hold different tumours (or the same tumours in a
    different loader order) silently produces a posterior for nothing at all, and every
    downstream number would still look plausible. Name the mismatch instead.
    """
    a = members[0]
    for b in members[1:]:
        if a["theta_true"].shape != b["theta_true"].shape:
            raise ValueError(
                f"theta_true shape mismatch: {a['path']} has {a['theta_true'].shape}, "
                f"{b['path']} has {b['theta_true'].shape}")
        if not np.array_equal(a["theta_true"], b["theta_true"]):
            bad = np.argwhere(a["theta_true"] != b["theta_true"])
            i, j = int(bad[0][0]), int(bad[0][1])
            raise ValueError(
                f"theta_true mismatch between {a['path']} and {b['path']}: "
                f"{bad.shape[0]} differing entries, first at case {i} arm {j} "
                f"({a['theta_true'][i, j]!r} vs {b['theta_true'][i, j]!r}) - "
                "these files are not the same test set")
        ida = np.asarray(a["sim_ids"]).astype(str)
        idb = np.asarray(b["sim_ids"]).astype(str)
        if ida.shape != idb.shape or not np.array_equal(ida, idb):
            diff = [k for k in range(min(ida.size, idb.size)) if ida[k] != idb[k]]
            first = f" first at index {diff[0]} ({ida[diff[0]]} vs {idb[diff[0]]})" if diff else ""
            raise ValueError(
                f"sim_ids mismatch between {a['path']} and {b['path']}:{first} - "
                "these files are not the same test set")


def shares(num_samples, sizes):
    """How many draws to take from each member for a target of ``num_samples``.

    An equal share, with the remainder handed to the first members, and never more than
    a member actually has. Returns None when no subsampling is needed.
    """
    total = int(sum(sizes))
    if num_samples is None or num_samples >= total:
        return None
    if num_samples < len(sizes):
        raise ValueError(f"--num-samples {num_samples} is below the member count {len(sizes)}")
    base, rem = divmod(int(num_samples), len(sizes))
    take = [base + (1 if k < rem else 0) for k in range(len(sizes))]
    for k, (t, s) in enumerate(zip(take, sizes)):
        if t > s:
            raise ValueError(
                f"member {k} holds {s} draws, which is fewer than its "
                f"{t}-draw share of --num-samples {num_samples}")
    return take


def pool_samples(members, num_samples=None, seed=0):
    """Concatenate the members' draws per case, optionally to a ``num_samples`` target.

    Subsampling takes the first ``take[k]`` draws of member k, not a random subset: the
    member's own draws are already i.i.d. from its posterior and in no meaningful order,
    so a prefix is as good as a permutation and keeps the result reproducible.
    """
    sizes = [int(m["samples"].shape[1]) for m in members]
    take = shares(num_samples, sizes)
    parts = [m["samples"] if take is None else m["samples"][:, :take[k]]
             for k, m in enumerate(members)]
    return np.concatenate(parts, axis=1).astype(np.float32), take


def mixture_log_prob(members):
    """log of the equal-weight mixture density at the truth, from the members' values."""
    lp = np.stack([np.asarray(m["log_prob_true"], dtype=np.float64) for m in members], axis=0)
    out = np.full(lp.shape[1], np.nan, dtype=np.float64)
    ok = np.isfinite(lp).all(axis=0)
    if ok.any():
        col = lp[:, ok]
        mx = col.max(axis=0)
        out[ok] = mx + np.log(np.exp(col - mx).sum(axis=0)) - np.log(lp.shape[0])
    return out.astype(np.float32)


def build_meta(members, samples, take, args):
    """Member A's meta, edited to describe the mixture rather than one seed."""
    meta = dict(members[0]["meta"])
    metas = [m["meta"] for m in members]
    meta.update({
        "ensemble_of": [m["path"] for m in members],
        "ensemble_seeds": [m.get("seed") for m in metas],
        "ensemble_run_tags": [m.get("run_tag") for m in metas],
        "ensemble_members": len(members),
        "ensemble_member_num_samples": [int(m["samples"].shape[1]) for m in members],
        "ensemble_member_draws_used": (
            take if take is not None else [int(m["samples"].shape[1]) for m in members]),
        # Every metric that divides by num_samples (the +0.5/(S+1) rank convention in
        # poster_metrics.load_run) must see the POOLED count, not one member's.
        "num_samples": int(samples.shape[1]),
        "checkpoint": [m.get("checkpoint") for m in metas],
        "checkpoint_epoch": [m.get("checkpoint_epoch") for m in metas],
        "seed": [m.get("seed") for m in metas],
        "best_val_loss": [m.get("best_val_loss") for m in metas],
        "model": args.model,
        "run_tag": args.run_tag,
        "built_with": "cancer_sbi.evaluation.ensemble_posteriors",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return meta


def output_filename(model, run_tag=None):
    """``posteriors_<model>[_<tag>].npz`` - the name poster_metrics.posterior_path builds."""
    return f"posteriors_{model}{f'_{run_tag}' if run_tag else ''}.npz"


def ensemble(inputs, out_dir, model, run_tag=None, num_samples=None, args=None):
    """Write the mixture posterior of ``inputs`` and return its path."""
    members = [load_member(p) for p in inputs]
    if not members:
        raise ValueError("--inputs is empty")
    check_aligned(members)

    for m in members:
        mm = m["meta"]
        print(f"member  {m['path']}  cases={m['samples'].shape[0]}  "
              f"draws={m['samples'].shape[1]}  seed={mm.get('seed')}  "
              f"run_tag={mm.get('run_tag')}  epoch={mm.get('checkpoint_epoch')}", flush=True)

    samples, take = pool_samples(members, num_samples)
    theta = np.asarray(members[0]["theta_true"], dtype=np.float32)
    post_mean = samples.mean(axis=1).astype(np.float32)
    post_std = samples.std(axis=1, ddof=0).astype(np.float32)
    # sample_posteriors.py:346, replicated exactly. The +0.5/(S+1) step is poster_metrics'.
    sbc_ranks = (samples < theta[:, None, :]).sum(axis=1).astype(np.int32)

    ns = argparse.Namespace(model=model, run_tag=run_tag) if args is None else args
    meta = build_meta(members, samples, take, ns)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename(model, run_tag))
    np.savez_compressed(
        out_path,
        theta_true=theta,
        samples=samples,
        post_mean=post_mean,
        post_std=post_std,
        sbc_ranks=sbc_ranks,
        log_prob_true=mixture_log_prob(members),
        sim_ids=np.asarray(members[0]["sim_ids"]),
        meta_json=json.dumps(meta),
    )
    print(f"ensemble  {out_path}  cases={samples.shape[0]}  draws={samples.shape[1]}  "
          f"members={len(members)}  ({os.path.getsize(out_path) / 1e6:.0f} MB)", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="the stage-1 .npz files to pool (one per seed of the same run)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True,
                    help="model name for the output file: posteriors_<model>[_<tag>].npz")
    ap.add_argument("--run-tag", default=None,
                    help="label for the ensemble, e.g. R12ens. Pass the same --run-tag to "
                         "poster_metrics / analyze.sh")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="subsample the pooled draws to this many, an equal share from each "
                         "member. Default: keep all of them.")
    args = ap.parse_args(argv)
    ensemble(args.inputs, args.out_dir, args.model, args.run_tag, args.num_samples, args)


if __name__ == "__main__":
    main()
