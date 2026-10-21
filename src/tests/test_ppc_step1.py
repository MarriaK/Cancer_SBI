"""Light tests for ppc.step1_change_sizes (no real-data caches touched)."""
import numpy as np
import pandas as pd

from ppc.decompose import COUNT_COLUMNS
from ppc.step1_change_sizes import derive_per_tumour, focal_overlap_stats, quantile_position


def test_derive_per_tumour_three_rows():
    df = pd.DataFrame(0, index=range(3), columns=list(COUNT_COLUMNS))
    df.loc[0, ["whole_chrom_gain", "whole_chrom_loss"]] = [1, 2]
    df.loc[1, ["arm_gain", "arm_loss", "broad_gain", "broad_loss"]] = [3, 1, 1, 1]
    df.loc[2, ["focal_gain", "focal_loss", "focal_gain_orig_baseline", "focal_loss_orig_baseline"]] = [5, 4, 1, 2]
    df["ploidy"] = [2.0, 2.1, 1.9]
    df["fga"] = [0.1, 0.2, 0.3]
    df["n_breakpoints"] = [4, 5, 6]
    out = derive_per_tumour(df)
    assert out["n_chrom"].tolist() == [3, 0, 0]
    assert out["n_arm"].tolist() == [0, 4, 0]
    assert out["n_broad"].tolist() == [0, 2, 0]
    assert out["n_focal"].tolist() == [0, 0, 9]
    assert out["n_focal_strict"].tolist() == [0, 0, 6]
    assert out["n_breakpoints"].tolist() == [4, 5, 6]


def test_quantile_position_known_array():
    sim = np.arange(1, 11, dtype=float)          # 1..10
    assert quantile_position(sim, 0.0) == 0.0
    assert quantile_position(sim, 11.0) == 1.0
    assert quantile_position(sim, 5.0) == 0.45   # 4 below + 0.5 * 1 tie
    assert quantile_position(sim, 5.5) == 0.5
    assert quantile_position(np.full(4, 3.0), 3.0) == 0.5
    assert np.isnan(quantile_position(np.array([]), 1.0))


def test_focal_overlap_stats():
    ev = np.array([[0, 1, 0, 2, 4, 3, 1, 1, 0],     # chr idx 1, blocks 2-4
                   [1, 1, 0, 4, 6, 3, -1, 1, 0],    # overlaps the first at block 4
                   [0, 2, 0, 2, 4, 3, 1, 1, 0],     # other chromosome
                   [2, 1, 0, -1, -1, 10, 1, 0, 0]])  # arm event, ignored
    s = focal_overlap_stats(ev)
    assert s["true_focal_events_all_lineages"] == 3
    assert s["focal_overlapping_another"] == 2 and s["focal_overlap_pairs"] == 1
    assert s["distinct_focal_footprints"] == 3
