#!/usr/bin/env python
"""
Sanity baseline for the NPE models: how much of theta can a ridge regression recover from
four hand-made statistics per chromosome arm?

It is deliberately restricted to what CloneMLP-NPE sees - the 100 most frequent unique arm
profiles of each tumour and their frequencies (same logic as utils.top_frequent_rows_tensor),
averaged over the 25 replicates - and it uses the project's own train/test split.

Per arm j, frequency-weighted over the top-100 profiles:
    share with 0 copies | share with 1 copy | share with >= 3 copies | mean copies
theta_j is then regressed on arm j's own four numbers (44 independent 4-feature ridge fits).

    ~/miniconda3/envs/cancer/bin/python summary_stat_baseline.py        # ~5 min first time, then cached

Writes summary_stat_baseline.csv (per-arm scores) next to this file.
"""
import argparse
import gzip
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ARMS = [f"chr{c}{a}" for c in range(1, 23) for a in "pq"]


def tumour_stats(path, top_k=100):
    a = np.asarray(pickle.load(gzip.open(path, "rb")), dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 44:
        return None
    u, c = np.unique(a, axis=0, return_counts=True)
    order = np.argsort(-c, kind="stable")[:top_k]
    copies = np.clip(2.0 ** (u[order] + 1.0), 0, 8)            # invert normalize(x) = log2(x) - 1
    w = (c[order] / c[order].sum())[:, None]
    return np.concatenate([((copies < 0.5) * w).sum(0), (((copies >= 0.5) & (copies < 1.5)) * w).sum(0),
                           ((copies > 2.5) * w).sum(0), (copies * w).sum(0)])


def build_features(root, n_trials=25):
    X, Y, ids = [], [], []
    sims = sorted(d for d in os.listdir(root) if d.startswith("sim"))
    for k, s in enumerate(sims):
        pp = os.path.join(root, s, "parameters.pkl")
        if not os.path.exists(pp) or os.path.getsize(pp) == 0:
            continue
        feats = []
        for t in range(1, n_trials + 1):
            p = os.path.join(root, s, str(t), "CNratios_all.pkl.gz")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                try:
                    f = tumour_stats(p)
                except Exception:
                    f = None
                if f is not None:
                    feats.append(f)
        if len(feats) < n_trials:                              # same rule as CNASimsDataset
            continue
        X.append(np.mean(feats, 0)); ids.append(s)
        Y.append(np.asarray(pickle.load(open(pp, "rb")), dtype=float).ravel()[2:])
        if k % 100 == 0:
            print(f"  scanned {k}/{len(sims)}  ({len(ids)} complete)", flush=True)
    return np.array(X), np.array(Y), np.array(ids)


def ridge(A, y, B, lam=1e-3):
    A1 = np.hstack([A, np.ones((len(A), 1))])
    R = lam * np.eye(A1.shape[1]); R[-1, -1] = 0
    return np.hstack([B, np.ones((len(B), 1))]) @ np.linalg.solve(A1.T @ A1 + R, A1.T @ y)


def true_r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(CODE, "Guassian_Normal", "simulation_outputs"))
    ap.add_argument("--split", default=os.path.join(CODE, "Base_NPE", "train_test_split.pkl"))
    ap.add_argument("--cache", default=os.path.join(HERE, "summary_stat_features.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "summary_stat_baseline.csv"))
    args = ap.parse_args()

    if os.path.exists(args.cache):
        d = np.load(args.cache, allow_pickle=True); X, Y, ids = d["X"], d["Y"], d["ids"]
        print(f"loaded cached features: {args.cache}")
    else:
        X, Y, ids = build_features(args.root)
        np.savez(args.cache, X=X, Y=Y, ids=ids)

    split = pickle.load(open(args.split, "rb"))
    tr, te = np.isin(ids, split["train_ids"]), np.isin(ids, split["test_ids"])
    shared = {tuple(np.round(v, 6)) for v in Y[tr]} & {tuple(np.round(v, 6)) for v in Y[te]}
    print(f"sims with all 25 replicates: {len(ids)} | train {tr.sum()} | test {te.sum()} | "
          f"theta vectors shared across the split: {len(shared)}")

    Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9)           # scaling uses the training set only
    rng = np.random.default_rng(0)
    rows = []
    for j, arm in enumerate(ARMS):
        own = [j, 44 + j, 88 + j, 132 + j]
        p = ridge(Xs[tr][:, own], Y[tr, j], Xs[te][:, own])
        p_shuf = ridge(Xs[tr][:, own], rng.permutation(Y[tr, j]), Xs[te][:, own])
        rows.append(dict(Chromosome_Arm=arm, R2=true_r2(Y[te, j], p), Pearson_r=np.corrcoef(p, Y[te, j])[0, 1],
                         R2_shuffled_labels=true_r2(Y[te, j], p_shuf)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"true R2 (1 - SSE/SST)   : mean {df.R2.mean():.3f}  min {df.R2.min():.3f}  max {df.R2.max():.3f}")
    print(f"squared Pearson r       : mean {(df.Pearson_r ** 2).mean():.3f}")
    print(f"shuffled-label control  : mean {df.R2_shuffled_labels.mean():.3f}   (should be ~0 or below)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
