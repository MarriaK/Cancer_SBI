"""Statistical tests for the prior predictive check (PPC, project plan step 4; Notion 3.5 section 5).

Works on plain numpy arrays / DataFrames of ALREADY-COMPUTED scalar summary statistics (one row per
tumour, one column per statistic); it does not compute the statistics itself.

Convention used throughout: ``A`` / ``sim`` = simulated tumours, ``B`` / ``real`` = real tumours.
In the classifier test the labels are real = 1, simulated = 0.

Rules and references
--------------------
* Marginal checks per scalar statistic (Gabry et al. 2019, JRSS-A, "Visualization in Bayesian
  workflow"): quantile position of the real-cohort MEDIAN inside the simulated distribution (flag
  outside [0.025, 0.975]); history-matching implausibility z = (mean_real - mean_sim) / sd_sim
  (flag |z| > 3; Vernon, Goldstein & Bower 2010, Bayesian Analysis); two-sample KS; Wasserstein-1.
* Joint two-sample tests on the vector of statistics standardised with the SIMULATED mean / sd:
  energy distance (Szekely & Rizzo 2013, J. Stat. Plan. Inference), MMD with an RBF kernel
  (Gretton et al. 2012, JMLR), classifier two-sample test C2ST (Lopez-Paz & Oquab 2017, ICLR; best
  of LR / RF, with the selection inside the permutation null) and
  a PCA "inside the simulation cloud" check (Lynch et al. 2022, eLife).
* Per-arm gain / loss frequency agreement (Spearman / Pearson; the fit statistic used by CINner).
* Between- vs within-theta variance split of the simulated replicates (one-way ANOVA).
* ``decision``: overall pass = marginal checks present, n_real > 0, no marginal flag AND C2ST
  accuracy < 0.6.

All randomness is seeded; NaN / inf rows are dropped (never imputed) and the drop counts reported.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance

Q_LO, Q_HI = 0.025, 0.975
Z_MAX = 3.0
C2ST_MAX_ACC = 0.6


# ----------------------------------------------------------------------------- helpers
def _as_2d(X: Any) -> np.ndarray:
    a = np.asarray(X, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return a


def _drop_bad_rows(X: Any) -> tuple[np.ndarray, int]:
    """Drop rows with any NaN / inf. Returns (clean array, number dropped)."""
    a = _as_2d(X)
    keep = np.all(np.isfinite(a), axis=1)
    return a[keep], int((~keep).sum())


def _perm_p(obs: float, null: np.ndarray) -> float:
    """Permutation p-value (1 + #{null >= obs}) / (1 + n_perm)."""
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + null.size))


def quantile_position(sim: Any, value: float) -> float:
    """Mid-rank quantile position of ``value`` in ``sim``: (#sim < v + 0.5 #sim == v) / n.

    NaNs in ``sim`` are dropped; returns NaN if nothing is left or ``value`` is NaN.
    """
    s = np.asarray(sim, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0 or not np.isfinite(value):
        return float("nan")
    return float((np.sum(s < value) + 0.5 * np.sum(s == value)) / s.size)


def standardise(A: Any, B: Any, columns: Sequence | None = None
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Standardise simulated ``A`` and real ``B`` with the SIMULATED column mean / sd (ddof=1).

    Columns are dropped if they are all-NaN / inf on EITHER side, or constant (sd == 0 or
    non-finite) in ``A``; then rows of either side with any NaN / inf in the kept columns are
    dropped (no imputation). Dropping a column that is all-NaN on the real side (e.g. ``ploidy_cw``
    for ASCAT cohorts) prevents it from silently removing every real row.

    ``A`` / ``B`` may be DataFrames (``columns`` = column names; default all of A's columns) or
    arrays (``columns`` = integer indices; default all). Returns (Z_sim, Z_real, info) with info =
    {columns_used, columns_dropped (all reasons), columns_dropped_sim_const (all-NaN or constant
    in A), columns_dropped_real_nan (all-NaN in B, not already dropped for A), n_sim_rows_dropped,
    n_real_rows_dropped, n_sim, n_real, sim_mean, sim_sd}.

    Raises ValueError if no column is left, or if no simulated or no real row is left after the
    row dropping.
    """
    if isinstance(A, pd.DataFrame):
        cols = list(A.columns) if columns is None else list(columns)
        a = A.loc[:, cols].to_numpy(dtype=float)
        b = (B.loc[:, cols] if isinstance(B, pd.DataFrame) else pd.DataFrame(_as_2d(B), columns=cols)
             ).to_numpy(dtype=float)
        names = cols
    else:
        a, b = _as_2d(A), _as_2d(B)
        idx = list(range(a.shape[1])) if columns is None else list(columns)
        a, b = a[:, idx], b[:, idx]
        names = idx
    a = np.where(np.isfinite(a), a, np.nan)
    b = np.where(np.isfinite(b), b, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0, ddof=1)
    good_sim = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    real_any = np.isfinite(b).any(axis=0) if b.shape[0] else np.zeros(b.shape[1], dtype=bool)
    good = good_sim & real_any
    used = [n for n, g in zip(names, good) if g]
    dropped = [n for n, g in zip(names, good) if not g]
    drop_sim = [n for n, g in zip(names, good_sim) if not g]
    drop_real = [n for n, gs, r in zip(names, good_sim, real_any) if gs and not r]
    if not used:
        raise ValueError(f"standardise: no usable column (dropped: {dropped})")
    za = (a[:, good] - mu[good]) / sd[good]
    zb = (b[:, good] - mu[good]) / sd[good]
    za, nda = _drop_bad_rows(za)
    zb, ndb = _drop_bad_rows(zb)
    if za.shape[0] == 0 or zb.shape[0] == 0:
        raise ValueError(f"standardise: no rows left after dropping NaN / inf rows (n_sim = {za.shape[0]}, "
                         f"{nda} dropped; n_real = {zb.shape[0]}, {ndb} dropped; columns used: {used})")
    info = {"columns_used": used, "columns_dropped": dropped,
            "columns_dropped_sim_const": drop_sim, "columns_dropped_real_nan": drop_real,
            "n_sim_rows_dropped": nda, "n_real_rows_dropped": ndb,
            "n_sim": int(za.shape[0]), "n_real": int(zb.shape[0]),
            "sim_mean": mu[good], "sim_sd": sd[good]}
    return za, zb, info


# ----------------------------------------------------------------------------- marginal
def marginal_checks(real: pd.DataFrame, sim: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Per-statistic marginal PPC checks (Gabry et al. 2019; Vernon, Goldstein & Bower 2010).

    For each column (NaN / inf dropped per column, per side) returns one row with: statistic,
    n_real, n_sim, real_median, sim_median, sim_q025, sim_q975, qpos_real_median (mid-rank quantile
    position of the real median in the simulated sample), z = (mean_real - mean_sim) / sd_sim
    (sd ddof=1; +-inf if sd_sim == 0 and the means differ, 0 if they agree), ks_stat, ks_p
    (scipy two-sample KS), wasserstein (W1), wasserstein_std (= W1 / sd_sim), flag_q
    (qpos < 0.025 or > 0.975) and flag_z (|z| > 3). A side with no data gives NaNs and no flag.
    """
    rows = []
    for c in columns:
        r = real[c].to_numpy(dtype=float)
        m = sim[c].to_numpy(dtype=float)
        r, m = r[np.isfinite(r)], m[np.isfinite(m)]
        row: dict[str, Any] = {"statistic": c, "n_real": int(r.size), "n_sim": int(m.size)}
        nan = float("nan")
        if r.size == 0 or m.size == 0:
            row.update(real_median=float(np.median(r)) if r.size else nan,
                       sim_median=float(np.median(m)) if m.size else nan,
                       sim_q025=float(np.quantile(m, Q_LO)) if m.size else nan,
                       sim_q975=float(np.quantile(m, Q_HI)) if m.size else nan,
                       qpos_real_median=nan, z=nan, ks_stat=nan, ks_p=nan,
                       wasserstein=nan, wasserstein_std=nan, flag_q=False, flag_z=False)
            rows.append(row)
            continue
        med = float(np.median(r))
        q = quantile_position(m, med)
        sd = float(m.std(ddof=1)) if m.size > 1 else 0.0
        diff = float(r.mean() - m.mean())
        if sd > 0:
            z = diff / sd
        else:
            z = 0.0 if diff == 0 else float(np.sign(diff) * np.inf)
        ks = ks_2samp(r, m)
        w1 = float(wasserstein_distance(r, m))
        row.update(real_median=med, sim_median=float(np.median(m)),
                   sim_q025=float(np.quantile(m, Q_LO)), sim_q975=float(np.quantile(m, Q_HI)),
                   qpos_real_median=q, z=float(z), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                   wasserstein=w1,
                   wasserstein_std=w1 / sd if sd > 0 else (0.0 if w1 == 0 else float("inf")),
                   flag_q=bool(q < Q_LO or q > Q_HI), flag_z=bool(abs(z) > Z_MAX))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- joint: distances
def _perm_masks(n_a: int, n_total: int, n_perm: int, seed: int) -> np.ndarray:
    """(n_perm, n_total) float matrix, 1 for the rows assigned to group A in each permutation."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_perm, n_total))
    for i in range(n_perm):
        W[i, rng.permutation(n_total)[:n_a]] = 1.0
    return W


def _group_means(M: np.ndarray, W: np.ndarray, n_a: int, n_b: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of M within A, within B and across A-B for each row-mask in W (vectorised)."""
    V = 1.0 - W
    MW = W @ M                                  # (P, N)
    s_aa = np.einsum("pn,pn->p", MW, W)
    s_ab = np.einsum("pn,pn->p", MW, V)
    tot = M.sum()
    s_bb = tot - s_aa - 2 * s_ab
    return s_aa / n_a ** 2, s_bb / n_b ** 2, s_ab / (n_a * n_b)


def energy_distance_test(A: Any, B: Any, n_perm: int = 200, seed: int = 0) -> dict:
    """Energy-distance two-sample test with a permutation p-value (Szekely & Rizzo 2013).

    Statistic (V-statistic form, Euclidean distance): E = 2 E|a-b| - E|a-a'| - E|b-b'|. The pooled
    distance matrix is computed once (cdist); permutations re-split its index set. Rows with NaN /
    inf are dropped. Returns {statistic, p_value, n_perm, n_a, n_b, n_a_dropped, n_b_dropped}.
    """
    a, da = _drop_bad_rows(A)
    b, db = _drop_bad_rows(B)
    out = {"n_a": int(a.shape[0]), "n_b": int(b.shape[0]), "n_a_dropped": da, "n_b_dropped": db,
           "n_perm": int(n_perm)}
    if a.shape[0] < 2 or b.shape[0] < 2:
        return {**out, "statistic": float("nan"), "p_value": float("nan")}
    X = np.vstack([a, b])
    D = cdist(X, X)
    na, nb = a.shape[0], b.shape[0]
    W0 = np.zeros((1, na + nb)); W0[0, :na] = 1.0
    aa, bb, ab = _group_means(D, W0, na, nb)
    obs = float(2 * ab[0] - aa[0] - bb[0])
    aa, bb, ab = _group_means(D, _perm_masks(na, na + nb, n_perm, seed), na, nb)
    null = 2 * ab - aa - bb
    return {**out, "statistic": obs, "p_value": _perm_p(obs, null)}


def mmd_test(A: Any, B: Any, n_perm: int = 200, seed: int = 0) -> dict:
    """MMD^2 two-sample test, RBF kernel, permutation p-value (Gretton et al. 2012, JMLR).

    k(x, y) = exp(-|x - y|^2 / (2 sigma^2)) with sigma = median pairwise Euclidean distance of the
    pooled sample (median heuristic). Biased (V-statistic) MMD^2 = mean K_aa + mean K_bb
    - 2 mean K_ab; the pooled kernel matrix is built once and permutations re-split its index set.
    Rows with NaN / inf are dropped. Returns {statistic, p_value, bandwidth, n_perm, n_a, n_b,
    n_a_dropped, n_b_dropped}.
    """
    a, da = _drop_bad_rows(A)
    b, db = _drop_bad_rows(B)
    out = {"n_a": int(a.shape[0]), "n_b": int(b.shape[0]), "n_a_dropped": da, "n_b_dropped": db,
           "n_perm": int(n_perm)}
    if a.shape[0] < 2 or b.shape[0] < 2:
        return {**out, "statistic": float("nan"), "p_value": float("nan"), "bandwidth": float("nan")}
    X = np.vstack([a, b])
    D = cdist(X, X)
    sigma = float(np.median(pdist(X)))
    if not sigma > 0:
        sigma = 1.0
    K = np.exp(-D ** 2 / (2 * sigma ** 2))
    na, nb = a.shape[0], b.shape[0]
    W0 = np.zeros((1, na + nb)); W0[0, :na] = 1.0
    aa, bb, ab = _group_means(K, W0, na, nb)
    obs = float(aa[0] + bb[0] - 2 * ab[0])
    aa, bb, ab = _group_means(K, _perm_masks(na, na + nb, n_perm, seed), na, nb)
    null = aa + bb - 2 * ab
    return {**out, "statistic": obs, "p_value": _perm_p(obs, null), "bandwidth": sigma}


# ----------------------------------------------------------------------------- joint: C2ST
def _c2st_perm_max(X: np.ndarray, y: np.ndarray, splitter, lr, rf_null, seed: int) -> float:
    """One permutation of the C2ST null: permute ``y`` (seeded), return max(CV acc LR, CV acc RF)."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    yp = np.random.default_rng(seed).permutation(y)
    a_lr = cross_val_score(clone(lr), X, yp, cv=splitter, scoring="accuracy").mean()
    a_rf = cross_val_score(clone(rf_null), X, yp, cv=splitter, scoring="accuracy").mean()
    return float(max(a_lr, a_rf))


def c2st(A: Any, B: Any, n_perm: int = 100, seed: int = 0, cv: int = 5,
         feature_names: Sequence[str] | None = None, n_jobs: int = -1,
         rf_trees_perm: int = 50) -> dict:
    """Classifier two-sample test (Lopez-Paz & Oquab 2017, ICLR): simulated ``A`` (label 0) vs
    real ``B`` (label 1).

    Rows with NaN / inf are dropped; the larger side is subsampled (seeded) to the size of the
    smaller one, so chance accuracy is 0.5. Two classifiers, ``cv``-fold stratified (shuffled,
    seeded) CV accuracy each: LogisticRegression (balanced class weight, on features standardised
    over the pooled balanced sample) and RandomForest (200 trees, balanced class weight, seeded).
    The observed statistic is ``accuracy`` = max(acc_lr, acc_rf).

    Permutation p-value with the classifier SELECTION inside the null: in each of ``n_perm``
    label permutations (seeded, one generator per permutation) BOTH classifiers are cross-validated
    with the same splitter and the larger of the two accuracies is the null draw;
    p = (1 + #null >= obs) / (1 + n_perm). Permutations run in parallel (joblib, ``n_jobs``). For
    speed the RF in the null uses ``rf_trees_perm`` trees (default 50, else identical to the
    observed RF); fewer trees make the null RF accuracies slightly noisier, i.e. the null max
    slightly larger and the p-value slightly conservative. Pass ``rf_trees_perm=200`` for the exact
    same-classifier null. The analytic binomial p-value is NOT reported: it ignores both the CV
    dependence and the best-of-two selection and rejected 0.17 of null datasets at nominal 0.05.

    Also reported: RF feature_importances_ and |LR coefficients| on standardised features (both
    fitted on all balanced rows).

    Returns {acc_lr, acc_rf, best, accuracy, p_value, n_perm, rf_trees_perm, null_mean, n_real,
    n_sim (per side after balancing), n_real_raw, n_sim_raw, n_real_dropped, n_sim_dropped,
    importance (DataFrame: feature, rf_importance, lr_abs_coef; sorted by rf_importance desc)}.
    """
    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    a, da = _drop_bad_rows(A)
    b, db = _drop_bad_rows(B)
    if feature_names is None:
        feature_names = list(A.columns) if isinstance(A, pd.DataFrame) else [f"x{i}" for i in range(a.shape[1])]
    rng = np.random.default_rng(seed)
    n = min(a.shape[0], b.shape[0])
    base = {"n_sim_raw": int(a.shape[0]), "n_real_raw": int(b.shape[0]),
            "n_sim_dropped": da, "n_real_dropped": db, "n_sim": int(n), "n_real": int(n),
            "n_perm": int(n_perm), "rf_trees_perm": int(rf_trees_perm)}
    if n < cv:
        nan = float("nan")
        return {**base, "acc_lr": nan, "acc_rf": nan, "best": None, "accuracy": nan,
                "p_value": nan, "null_mean": nan, "importance": pd.DataFrame(
                    {"feature": feature_names, "rf_importance": nan, "lr_abs_coef": nan})}
    a = a[rng.choice(a.shape[0], n, replace=False)] if a.shape[0] > n else a
    b = b[rng.choice(b.shape[0], n, replace=False)] if b.shape[0] > n else b
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=n_jobs)
    acc_lr = float(cross_val_score(lr, X, y, cv=splitter, scoring="accuracy").mean())
    acc_rf = float(cross_val_score(rf, X, y, cv=splitter, scoring="accuracy").mean())
    best = "rf" if acc_rf > acc_lr else "lr"
    acc = max(acc_lr, acc_rf)
    if n_perm > 0:
        rf_null = RandomForestClassifier(n_estimators=rf_trees_perm, class_weight="balanced",
                                         random_state=seed, n_jobs=1)
        seeds = np.random.SeedSequence(seed).generate_state(n_perm)
        null = np.asarray(Parallel(n_jobs=n_jobs)(
            delayed(_c2st_perm_max)(X, y, splitter, lr, rf_null, int(s)) for s in seeds))
        p = _perm_p(acc, null)
        null_mean = float(null.mean())
    else:
        p = null_mean = float("nan")
    rf.fit(X, y)
    lr.fit(X, y)
    imp = pd.DataFrame({"feature": list(feature_names),
                        "rf_importance": rf.feature_importances_,
                        "lr_abs_coef": np.abs(lr[-1].coef_.ravel())}
                       ).sort_values("rf_importance", ascending=False, ignore_index=True)
    return {**base, "acc_lr": acc_lr, "acc_rf": acc_rf, "best": best, "accuracy": acc,
            "p_value": p, "null_mean": null_mean, "importance": imp}


# ----------------------------------------------------------------------------- joint: PCA overlap
def pca_overlap(A_sim: Any, B_real: Any, n_components: Sequence[int] = (2, 5)) -> dict:
    """PCA "inside the simulation cloud" check (Lynch et al. 2022, eLife).

    Standardises with the simulated mean / sd (columns constant / all-NaN in the simulation are
    dropped, rows with NaN dropped), fits PCA with k components on the simulated rows, projects the
    real rows and computes the Mahalanobis distance in PC space, d^2 = sum_j score_j^2 / var_j.
    The threshold is the 95th percentile of the simulated (in-sample) distances; reported is the
    fraction of real rows above it (~0.05 when real and simulated agree). k is capped at
    min(n_features, n_sim). Returns {n_sim, n_real, n_sim_dropped, n_real_dropped, columns_used,
    by_k: {k: {threshold, outside_frac, explained_variance_ratio}}, outside_frac: value at the
    largest k}.
    """
    from sklearn.decomposition import PCA

    za, zb, info = standardise(A_sim, B_real)
    out: dict[str, Any] = {"n_sim": info["n_sim"], "n_real": info["n_real"],
                           "n_sim_dropped": info["n_sim_rows_dropped"],
                           "n_real_dropped": info["n_real_rows_dropped"],
                           "columns_used": info["columns_used"], "by_k": {}}
    if za.shape[0] < 3 or zb.shape[0] == 0 or za.shape[1] == 0:
        out["outside_frac"] = float("nan")
        return out
    kmax = min(za.shape[1], za.shape[0] - 1)
    last = float("nan")
    for k in n_components:
        kk = int(min(k, kmax))
        pca = PCA(n_components=kk).fit(za)
        var = pca.explained_variance_
        var = np.where(var > 0, var, np.finfo(float).tiny)
        d_sim = np.sqrt(((pca.transform(za) ** 2) / var).sum(axis=1))
        d_real = np.sqrt(((pca.transform(zb) ** 2) / var).sum(axis=1))
        thr = float(np.quantile(d_sim, 0.95))
        last = float(np.mean(d_real > thr))
        out["by_k"][int(k)] = {"k_used": kk, "threshold": thr, "outside_frac": last,
                               "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
    out["outside_frac"] = last
    return out


# ----------------------------------------------------------------------------- convenience
def joint_tests(real: pd.DataFrame, sim: pd.DataFrame, columns: Sequence[str],
                n_perm: int = 200, n_perm_c2st: int = 100, seed: int = 0, cv: int = 5) -> dict:
    """Standardise (``standardise``) then run energy, MMD, C2ST and PCA overlap.

    Returns {n_sim, n_real, standardise: info, energy, mmd, c2st, pca}; this is the ``joint``
    argument expected by ``decision``.
    """
    za, zb, info = standardise(sim, real, columns)
    names = info["columns_used"]
    return {"n_sim": info["n_sim"], "n_real": info["n_real"], "standardise": info,
            "energy": energy_distance_test(za, zb, n_perm=n_perm, seed=seed),
            "mmd": mmd_test(za, zb, n_perm=n_perm, seed=seed),
            "c2st": c2st(za, zb, n_perm=n_perm_c2st, seed=seed, cv=cv, feature_names=names),
            "pca": pca_overlap(za, zb)}


# ----------------------------------------------------------------------------- arm frequencies
def arm_frequency_agreement(real_freq: pd.DataFrame, sim_freq: pd.DataFrame,
                            columns: Sequence[str] = ("gain_freq", "loss_freq")) -> dict:
    """Agreement of per-arm gain / loss frequencies, real vs simulated (CINner's fit statistic).

    Inputs: DataFrames indexed by arm (or with an ``arm`` column) holding ``gain_freq`` and
    ``loss_freq``. Arms are aligned on the intersection of the indices; rows with NaN in a column
    are dropped for that column. Per column returns {n_arms, spearman, pearson, mean_abs_diff};
    correlations are NaN when either vector is constant. Also top-level n_arms_common.
    """
    def _idx(df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index("arm") if "arm" in df.columns else df
    r, s = _idx(real_freq), _idx(sim_freq)
    common = r.index.intersection(s.index)
    out: dict[str, Any] = {"n_arms_common": int(len(common))}
    for c in columns:
        x = r.loc[common, c].to_numpy(dtype=float)
        y = s.loc[common, c].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        res = {"n_arms": int(ok.sum()), "n_dropped": int((~ok).sum()),
               "mean_abs_diff": float(np.mean(np.abs(x - y))) if x.size else float("nan")}
        if x.size >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            res["spearman"] = float(spearmanr(x, y).statistic)
            res["pearson"] = float(pearsonr(x, y).statistic)
        else:
            res["spearman"] = res["pearson"] = float("nan")
        out[c] = res
    return out


# ----------------------------------------------------------------------------- variance split
def variance_split(sim_df: pd.DataFrame, columns: Iterable[str], group_col: str = "sim_id") -> pd.DataFrame:
    """One-way ANOVA decomposition of each simulated statistic across theta (``group_col``).

    Per column (NaN rows dropped for that column): total = between + within (population form,
    ddof=0), between = sum_g n_g (mean_g - grand)^2 / N, within = sum (x - mean_g)^2 / N. With equal
    group sizes ``between`` is exactly the variance of the theta means. Shares = component / total.
    ``between_share`` slightly overstates the true variance component because theta means still
    carry within-noise / n_g; ``icc_anova`` = (MSB - MSW) / (MSB + (n0 - 1) MSW) is the classical
    unbiased-ish estimate (n0 = ANOVA effective group size). High share = prior tuning on theta can
    move the statistic; low share = dominated by replicate noise.

    Returns one row per statistic: statistic, n_rows, n_groups, n_dropped, total_var, between_var,
    within_var, between_share, within_share, icc_anova.
    """
    rows = []
    for c in columns:
        d = sim_df[[group_col, c]].copy()
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        n_drop = int(d[c].isna().sum())
        d = d.dropna(subset=[c])
        N = len(d)
        nan = float("nan")
        if N == 0:
            rows.append({"statistic": c, "n_rows": 0, "n_groups": 0, "n_dropped": n_drop,
                         "total_var": nan, "between_var": nan, "within_var": nan,
                         "between_share": nan, "within_share": nan, "icc_anova": nan})
            continue
        x = d[c].to_numpy(dtype=float)
        g = d.groupby(group_col)[c]
        means = g.transform("mean").to_numpy(dtype=float)
        sizes = g.size().to_numpy(dtype=float)
        k = len(sizes)
        grand = x.mean()
        ss_b = float(((means - grand) ** 2).sum())
        ss_w = float(((x - means) ** 2).sum())
        tot = ss_b + ss_w
        if k > 1 and N > k:
            msb, msw = ss_b / (k - 1), ss_w / (N - k)
            n0 = (N - (sizes ** 2).sum() / N) / (k - 1)
            den = msb + (n0 - 1) * msw
            icc = float((msb - msw) / den) if den > 0 else nan
        else:
            icc = nan
        rows.append({"statistic": c, "n_rows": N, "n_groups": k, "n_dropped": n_drop,
                     "total_var": tot / N, "between_var": ss_b / N, "within_var": ss_w / N,
                     "between_share": ss_b / tot if tot > 0 else nan,
                     "within_share": ss_w / tot if tot > 0 else nan, "icc_anova": icc})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- decision
def decision(cohort: str, marginal: pd.DataFrame, joint: dict) -> dict:
    """Decision summary for one real cohort (plan Notion 3.5 section 5).

    flagged_statistics = statistics with flag_q or flag_z in ``marginal``; c2st_pass = C2ST
    accuracy (the better of LR / RF) < 0.6 (a NaN accuracy fails); overall_pass = at least one
    marginal row AND n_real > 0 AND no marginal flag AND c2st_pass. Energy / MMD p-values and the PCA outside fraction are reported, not gated on.
    ``joint`` is the dict from ``joint_tests`` (keys energy, mmd, c2st, pca, n_real, n_sim; missing
    keys give NaN).
    """
    flagged = marginal.loc[marginal["flag_q"].astype(bool) | marginal["flag_z"].astype(bool),
                           "statistic"].tolist() if len(marginal) else []
    nan = float("nan")
    c = joint.get("c2st", {}) or {}
    acc = float(c.get("accuracy", nan)) if c.get("accuracy") is not None else nan
    c2st_pass = bool(np.isfinite(acc) and acc < C2ST_MAX_ACC)
    n_real = joint.get("n_real", int(marginal["n_real"].max()) if len(marginal) else 0)
    n_sim = joint.get("n_sim", int(marginal["n_sim"].max()) if len(marginal) else 0)
    return {"cohort": cohort, "n_real": int(n_real), "n_sim": int(n_sim),
            "flagged_statistics": flagged, "c2st_accuracy": acc, "c2st_pass": c2st_pass,
            "energy_p": float((joint.get("energy") or {}).get("p_value", nan)),
            "mmd_p": float((joint.get("mmd") or {}).get("p_value", nan)),
            "pca_outside_frac": float((joint.get("pca") or {}).get("outside_frac", nan)),
            "overall_pass": bool(len(marginal) > 0 and int(n_real) > 0
                                 and len(flagged) == 0 and c2st_pass)}
