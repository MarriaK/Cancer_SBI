"""Light tests for ppc.run_ppc (no real or simulated data touched; the end-to-end test uses only the
grid manifest and the structural mask, with synthetic profiles)."""
from pathlib import Path

import pandas as pd
import pytest

import numpy as np

from ppc import run_ppc
from ppc.run_ppc import (DEGENERATE_LABEL, PRIMARY_MIN_THETA, build_flag_matrix, build_parser,
                         degenerate_stats, flag_code, flag_matrix_md, resolve_primary)

PROJECT = Path("/Users/maryamkafi/Documents/PhD_Project/cancer")
GRID_ROOT = PROJECT / "data_sim/gene_calib_f10"


def test_flag_matrix_from_tiny_marginal_table():
    long = pd.DataFrame({
        "cohort": ["C1", "C1", "C2", "C2", "C3"],
        "statistic": ["ploidy", "fga", "ploidy", "fga", "ploidy"],
        "flag_q": [True, False, True, False, False],
        "flag_z": [False, True, True, False, False],
    })
    M = build_flag_matrix(long, ["C1", "C2", "C3"], ["ploidy", "fga", "subclonal_frac"])
    assert M.loc["C1"].tolist() == ["q", "z", ""]
    assert M.loc["C2"].tolist() == ["qz", "", ""]
    assert M.loc["C3"].tolist() == ["", "", ""]      # fga absent for C3 -> ''
    md = flag_matrix_md(M)
    assert md.splitlines()[2] == "| C1 | q | z | . | 2 |"
    assert md.splitlines()[4] == "| C3 | . | . | . | 0 |"
    assert flag_code(False, False) == ""


def test_parser_accepts_documented_flags():
    a = build_parser().parse_args(["--sim", "/x/root", "--label", "v2", "--out", "/y", "--n-perm", "50",
                                   "--figdir", "/z"])
    assert (a.sim, a.label, a.out, a.n_perm, a.figdir) == ("/x/root", "v2", "/y", 50, "/z")
    d = build_parser().parse_args(["--sim", "/x", "--label", "F10"])
    assert d.n_perm == 100 and d.out is None


def test_degenerate_statistics_counted_separately():
    long = pd.DataFrame({"cohort": ["C1", "C1"], "statistic": ["cn_frac_0", "fga"],
                         "flag_q": [False, True], "flag_z": [True, False]})
    M = build_flag_matrix(long, ["C1"], ["cn_frac_0", "fga"])
    md = flag_matrix_md(M, degenerate=["cn_frac_0"]).splitlines()
    assert DEGENERATE_LABEL in md[0] and "n degenerate flagged" in md[0]
    assert md[2] == "| C1 | z | q | 1 | 1 |"
    sim = pd.DataFrame({"cn_frac_0": np.zeros(10), "fga": np.linspace(0, 1, 10), "x": [np.nan] * 9 + [1.0]})
    assert degenerate_stats(sim, ["cn_frac_0", "fga", "x"]) == ["cn_frac_0", "x"]


def test_primary_auto_rule():
    assert PRIMARY_MIN_THETA == 100
    assert resolve_primary("auto", 100) == "one"
    assert resolve_primary("auto", 99) == "all"
    assert resolve_primary("auto", 21) == "all"
    assert resolve_primary("all", 500) == "all" and resolve_primary("one", 5) == "one"
    assert build_parser().parse_args(["--sim", "/x", "--label", "F10"]).primary == "auto"
    assert build_parser().parse_args(["--sim", "/x", "--label", "F10", "--primary", "one"]).primary == "one"


def _profiles(grid, n, rng):
    """n random integer profiles with a few arm-level changes each (baseline 2)."""
    X = np.full((n, grid.D), 2, dtype=np.int64)
    for r in range(n):
        for _ in range(rng.integers(2, 8)):
            a = grid.arms[rng.integers(len(grid.arms))]
            lo = rng.integers(a.start, a.end)
            X[r, lo:min(a.end, lo + rng.integers(1, a.n_blocks + 1))] = rng.integers(0, 6)
    return X


@pytest.mark.skipif(not ((GRID_ROOT / "manifest.json").exists() and Path(run_ppc.MASK_PATH).exists()),
                    reason="grid manifest or structural mask not available")
def test_primary_one_end_to_end_on_synthetic_cohorts(tmp_path, monkeypatch):
    from ppc.grid import load_grid
    from ppc.stats import apply_structural_mask
    grid = load_grid(str(GRID_ROOT / "manifest.json"))
    mask = np.load(run_ppc.MASK_PATH).astype(bool)
    rng = np.random.default_rng(1)
    n_theta, n_trial, n_real = 6, 3, 15

    def cw(X):
        return apply_structural_mask(X + rng.normal(0, 0.1, X.shape), mask)

    real = {}
    for lab, role, has_cw in (("TCGA_X", "target", False), ("PCAWG_Y", "target", True),
                              ("TCGA_C", "contrast", False)):
        X = _profiles(grid, n_real, rng)
        real[lab] = {"X": apply_structural_mask(X, mask), "X_cw": cw(X) if has_cw else None, "role": role,
                     "source": lab.split("_")[0]}
    XA = _profiles(grid, n_theta * n_trial, rng)
    XB = XA.copy()
    XB[:, 0] = 3
    meta = pd.DataFrame({"sim_id": np.repeat(np.arange(n_theta), n_trial),
                         "trial": np.tile(np.arange(1, n_trial + 1), n_theta),
                         "nWGD_largest": 0, "B_blocks_differ": (XB != XA).sum(axis=1)})
    sim = {"A": (apply_structural_mask(XA, mask), cw(XA)), "B": (apply_structural_mask(XB, mask), cw(XA))}
    monkeypatch.setattr(run_ppc, "load_real", lambda g, m: real)
    monkeypatch.setattr(run_ppc, "load_sim", lambda root, g, m: (sim, meta))
    for f in ("fig_ecdf", "fig_cn_states", "fig_arm_freq", "fig_block_freq", "fig_pca", "fig_variance_split"):
        monkeypatch.setattr(run_ppc, f, lambda *a, **k: None)

    run_ppc.main(["--sim", str(GRID_ROOT), "--label", "syn", "--out", str(tmp_path), "--figdir",
                  str(tmp_path), "--n-perm", "3", "--primary", "one"])
    for lab in ("TCGA_X", "PCAWG_Y"):
        m = pd.read_csv(tmp_path / f"marginal_{lab}_A.csv")
        assert set(m["n_sim"]) <= {n_theta} and m["n_sim"].max() == n_theta
    dec = pd.read_csv(tmp_path / "decisions.csv")
    assert (dec["primary_sample"] == "A_one").all() and (dec["n_sim"] == n_theta).all()
    assert (pd.read_csv(tmp_path / "joint_summary.csv")["primary_sample"] == "A_one").all()
    sec = pd.read_csv(tmp_path / "marginal_all_secondary.csv")
    assert sec["n_sim"].max() == n_theta * n_trial and set(sec["sim"]) == {"A_all"}
    vs = pd.read_csv(tmp_path / "variance_split.csv")
    assert (vs["n_groups"].dropna() == n_theta).all() and vs["n_rows"].max() == n_theta * n_trial
    assert (tmp_path / "stats_SIM_syn_A_all.csv").exists()
    note = (tmp_path / "ppc_results.md").read_text()
    assert "## Data used (primary sample, which decided: sim A-one)" in note
    assert "## Decisions (decided on sim A-one;" in note and "Secondary: vs sim A-all" in note
