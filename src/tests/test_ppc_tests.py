"""Synthetic, fast tests for ppc.ppc_tests (PPC step 4 statistical tests)."""
import numpy as np
import pandas as pd
import pytest

from ppc.ppc_tests import (arm_frequency_agreement, c2st, decision, energy_distance_test,
                           joint_tests, marginal_checks, mmd_test, pca_overlap, quantile_position,
                           standardise, variance_split)

COLS = [f"s{i}" for i in range(5)]


def _frames(n_real: int = 300, n_sim: int = 1500, shift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    sim = pd.DataFrame(rng.normal(size=(n_sim, len(COLS))), columns=COLS)
    real = pd.DataFrame(rng.normal(size=(n_real, len(COLS))), columns=COLS)
    real["s0"] += shift
    return real, sim


@pytest.fixture(scope="module")
def same():
    real, sim = _frames()
    return real, sim, marginal_checks(real, sim, COLS), joint_tests(real, sim, COLS, n_perm=200, n_perm_c2st=20)


@pytest.fixture(scope="module")
def shifted():
    real, sim = _frames(shift=3.0, seed=1)
    return real, sim, marginal_checks(real, sim, COLS), joint_tests(real, sim, COLS, n_perm=200, n_perm_c2st=20)


def test_quantile_position_basic():
    sim = np.arange(1, 11, dtype=float)
    assert quantile_position(sim, 0.0) == 0.0
    assert quantile_position(sim, 11.0) == 1.0
    assert quantile_position(sim, 5.0) == 0.45
    assert np.isnan(quantile_position(np.array([np.nan]), 1.0))


def test_identical_marginal(same):
    _, _, m, _ = same
    assert list(m["statistic"]) == COLS
    assert (m["qpos_real_median"].sub(0.5).abs() < 0.1).all()
    assert (m["z"].abs() < 1).all()
    assert not m["flag_q"].any() and not m["flag_z"].any()
    assert (m["n_real"] == 300).all() and (m["n_sim"] == 1500).all()
    assert np.allclose(m["wasserstein_std"], m["wasserstein"] / 1.0, atol=0.05)


def test_identical_joint(same):
    _, _, _, j = same
    assert j["energy"]["p_value"] > 0.05
    assert j["mmd"]["p_value"] > 0.05
    assert abs(j["c2st"]["accuracy"] - 0.5) <= 0.08
    assert j["c2st"]["n_real"] == j["c2st"]["n_sim"] == 300
    assert abs(j["pca"]["outside_frac"] - 0.05) <= 0.04
    assert abs(j["pca"]["by_k"][2]["outside_frac"] - 0.05) <= 0.04


def test_shifted(shifted):
    _, _, m, j = shifted
    row = m.set_index("statistic").loc["s0"]
    assert row["flag_z"] and row["flag_q"]
    assert not m.set_index("statistic").drop("s0")[["flag_q", "flag_z"]].to_numpy().any()
    assert j["c2st"]["accuracy"] > 0.8
    assert j["c2st"]["p_value"] < 0.06
    assert j["energy"]["p_value"] < 0.05
    assert j["mmd"]["p_value"] < 0.05
    imp = j["c2st"]["importance"]
    assert imp.iloc[0]["feature"] == "s0"
    assert imp.set_index("feature")["lr_abs_coef"].idxmax() == "s0"
    assert j["pca"]["outside_frac"] > 0.2


def test_decision_logic(same, shifted):
    d_ok = decision("same", same[2], same[3])
    assert d_ok["overall_pass"] and d_ok["c2st_pass"] and d_ok["flagged_statistics"] == []
    assert d_ok["n_real"] == 300 and d_ok["n_sim"] == 1500
    d_bad = decision("shift", shifted[2], shifted[3])
    assert not d_bad["overall_pass"] and not d_bad["c2st_pass"] and d_bad["flagged_statistics"] == ["s0"]
    # marginal flag alone fails; c2st alone fails
    joint_ok = {"c2st": {"accuracy": 0.55}, "n_real": 1, "n_sim": 1}
    assert not decision("x", shifted[2], joint_ok)["overall_pass"]
    assert not decision("x", same[2], {"c2st": {"accuracy": 0.61}})["overall_pass"]
    assert decision("x", same[2], joint_ok)["overall_pass"]
    assert not decision("x", same[2], {})["overall_pass"]          # missing C2ST fails
    # F7: no marginal rows or no real rows -> never a pass
    assert not decision("x", same[2].iloc[:0], joint_ok)["overall_pass"]
    assert not decision("x", same[2], {**joint_ok, "n_real": 0})["overall_pass"]


def test_decision_ignores_subclonal_frac(same):
    """Pre-registered 2026-10-22: subclonal_frac(_nobp) flags are reported, never gating."""
    from ppc.ppc_tests import NON_GATING
    assert NON_GATING == {"subclonal_frac", "subclonal_frac_nobp"}
    m = same[2].copy()
    extra = m.iloc[[0, 0]].copy()
    extra["statistic"] = ["subclonal_frac", "subclonal_frac_nobp"]
    extra["flag_q"] = True
    extra["flag_z"] = True
    m = pd.concat([m, extra], ignore_index=True)
    joint_ok = {"c2st": {"accuracy": 0.55}, "n_real": 1, "n_sim": 1}
    d = decision("x", m, joint_ok)
    assert d["overall_pass"] and d["flagged_statistics"] == []
    assert d["flagged_non_gating"] == ["subclonal_frac", "subclonal_frac_nobp"]
    # a gating flag still fails; an empty non_gating set restores the old rule
    m.loc[0, "flag_z"] = True
    d2 = decision("x", m, joint_ok)
    assert not d2["overall_pass"] and d2["flagged_statistics"] == [m.loc[0, "statistic"]]
    assert not decision("x", m.iloc[1:], joint_ok, non_gating=())["overall_pass"]


def test_standardise_drops_all_nan_real_column():
    """F2: a column all-NaN on the REAL side is dropped (and reported), not every real row."""
    real, sim = _frames(n_real=50, n_sim=100, seed=8)
    real["cw"] = np.nan
    sim["cw"] = np.random.default_rng(1).normal(size=100)
    za, zb, info = standardise(sim, real, COLS + ["cw"])
    assert info["columns_dropped"] == ["cw"] and info["columns_dropped_real_nan"] == ["cw"]
    assert info["columns_dropped_sim_const"] == []
    assert info["n_real"] == 50 and info["n_real_rows_dropped"] == 0 and zb.shape == (50, 5)
    j = joint_tests(real, sim, COLS + ["cw"], n_perm=10, n_perm_c2st=5)
    assert j["n_real"] == 50 and j["c2st"]["n_real"] == 50


def test_standardise_raises_when_no_rows_left():
    real, sim = _frames(n_real=20, n_sim=40, seed=9)
    real.loc[:, "s0"] = np.nan
    real.loc[0, "s0"] = 1.0                     # not all-NaN, so the column is kept ...
    real.loc[0, "s1"] = np.nan                  # ... and every real row has a NaN
    with pytest.raises(ValueError, match="no rows left"):
        standardise(sim, real, COLS)
    sim2 = sim.copy(); sim2.loc[:, "s2"] = np.nan; sim2.loc[0, "s2"] = 0.0; sim2.loc[1, "s2"] = 1.0
    sim2.loc[[0, 1], "s3"] = np.nan             # every sim row has a NaN in s2 or s3
    with pytest.raises(ValueError, match="no rows left"):
        standardise(sim2, _frames(n_real=20, n_sim=40, seed=9)[0], COLS)
    with pytest.raises(ValueError, match="no usable column"):
        standardise(sim.assign(c=1.0), real.assign(c=1.0), ["c"])


def test_c2st_null_calibration_smoke():
    """F3: with the classifier selection inside the null, the max(LR, RF) C2ST is calibrated."""
    rej = 0
    for rep in range(40):
        rng = np.random.default_rng(1000 + rep)
        r = c2st(rng.normal(size=(60, 5)), rng.normal(size=(60, 5)), n_perm=19, seed=rep)
        assert "p_binomial" not in r and r["n_perm"] == 19
        rej += r["p_value"] <= 0.05
    assert rej / 40 <= 0.10, rej


def test_variance_split_known_design():
    rng = np.random.default_rng(3)
    n_theta, n_rep = 400, 25
    sid = np.repeat(np.arange(n_theta), n_rep)
    a = rng.normal(0, 1.0, n_theta)[sid] + rng.normal(0, np.sqrt(3.0), sid.size)   # share 0.25
    b = rng.normal(0, 2.0, n_theta)[sid] + rng.normal(0, 1.0, sid.size)            # share 0.80
    c = rng.normal(0, 1.0, sid.size)                                                # share 0
    df = pd.DataFrame({"sim_id": sid, "a": a, "b": b, "c": c})
    df.loc[0, "a"] = np.nan
    v = variance_split(df, ["a", "b", "c"]).set_index("statistic")
    for col, share in [("a", 0.25), ("b", 0.80), ("c", 0.0)]:
        assert abs(v.loc[col, "between_share"] - share) <= 0.05
        assert abs(v.loc[col, "icc_anova"] - share) <= 0.05
        assert v.loc[col, "between_share"] + v.loc[col, "within_share"] == pytest.approx(1.0)
    assert v.loc["a", "n_dropped"] == 1 and v.loc["a", "n_rows"] == sid.size - 1
    assert v.loc["b", "n_groups"] == n_theta


def test_c2st_imbalance():
    rng = np.random.default_rng(4)
    r = c2st(rng.normal(size=(1000, 4)), rng.normal(size=(100, 4)), n_perm=10, seed=0)
    assert r["n_sim"] == r["n_real"] == 100
    assert r["n_sim_raw"] == 1000 and r["n_real_raw"] == 100
    assert 0 < r["p_value"] <= 1 and np.isfinite(r["acc_lr"]) and np.isfinite(r["acc_rf"])


def test_arm_frequency_identical():
    rng = np.random.default_rng(5)
    arms = [f"{c}{a}" for c in range(1, 21) for a in "pq"][:39]
    f = pd.DataFrame({"gain_freq": rng.uniform(size=39), "loss_freq": rng.uniform(size=39)}, index=arms)
    out = arm_frequency_agreement(f, f.copy())
    assert out["n_arms_common"] == 39
    for c in ("gain_freq", "loss_freq"):
        assert out[c]["spearman"] == pytest.approx(1.0)
        assert out[c]["pearson"] == pytest.approx(1.0)
        assert out[c]["mean_abs_diff"] == 0.0
    g = f.copy(); g.iloc[0, 0] = np.nan
    out2 = arm_frequency_agreement(g.reset_index(names="arm"), f.iloc[::-1])
    assert out2["gain_freq"]["n_arms"] == 38 and out2["gain_freq"]["n_dropped"] == 1


def test_nan_handling_and_standardise():
    real, sim = _frames(n_real=200, n_sim=400, seed=6)
    sim["const"] = 1.0
    real["const"] = 2.0
    sim["allnan"] = np.nan
    real["allnan"] = 0.0
    real.loc[:4, "s1"] = np.nan          # 5 real rows
    sim.loc[:2, "s2"] = np.inf           # 3 sim rows
    cols = COLS + ["const", "allnan"]
    za, zb, info = standardise(sim, real, cols)
    assert info["columns_dropped"] == ["const", "allnan"] and info["columns_used"] == COLS
    assert info["n_real_rows_dropped"] == 5 and info["n_sim_rows_dropped"] == 3
    assert za.shape == (397, 5) and zb.shape == (195, 5)
    assert np.abs(za.mean(axis=0)).max() < 0.05
    m = marginal_checks(real, sim, cols).set_index("statistic")
    assert m.loc["s1", "n_real"] == 195 and m.loc["s2", "n_sim"] == 397
    assert np.isinf(m.loc["const", "z"]) and m.loc["const", "flag_z"]
    assert m.loc["allnan", "n_sim"] == 0 and not m.loc["allnan", "flag_z"]
    A = sim[COLS].to_numpy(); B = real[COLS].to_numpy()
    e = energy_distance_test(A, B, n_perm=20)
    assert e["n_a_dropped"] == 3 and e["n_b_dropped"] == 5 and e["n_a"] == 397
    mm = mmd_test(A, B, n_perm=20)
    assert mm["n_b_dropped"] == 5 and mm["bandwidth"] > 0
    p = pca_overlap(A, B)
    assert p["n_real_dropped"] == 5 and p["n_sim_dropped"] == 3
    c = c2st(A, B, n_perm=5)
    assert c["n_real_dropped"] == 5 and c["n_real"] == c["n_sim"] == 195


def test_seeded_reproducible():
    rng = np.random.default_rng(7)
    A, B = rng.normal(size=(100, 3)), rng.normal(size=(80, 3))
    assert energy_distance_test(A, B, seed=2)["p_value"] == energy_distance_test(A, B, seed=2)["p_value"]
    assert mmd_test(A, B, seed=2)["p_value"] == mmd_test(A, B, seed=2)["p_value"]
