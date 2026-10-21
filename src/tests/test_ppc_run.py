"""Light tests for ppc.run_ppc (no real or simulated data touched)."""
import pandas as pd

import numpy as np

from ppc.run_ppc import (DEGENERATE_LABEL, build_flag_matrix, build_parser, degenerate_stats, flag_code,
                         flag_matrix_md)


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
