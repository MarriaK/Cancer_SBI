"""The 589-block genome grid shared by the simulator and the prior predictive check (PPC).

Everything is read from a simulator ``manifest.json`` (written by simulate_gene_selection.py), so
real data and simulated data are binned onto exactly the same blocks, chromosome order and
centromere positions. Coordinates are GRCh37/hg19 (SISTEM's chromosome table), 0-based
half-open block intervals [start_bp, end_bp).

Arm conventions (Taylor et al. 2018, Cancer Cell): the five acrocentric short arms
(13p, 14p, 15p, 21p, 22p) are never called, leaving 39 arms.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

ACROCENTRIC_P = ("chr13", "chr14", "chr15", "chr21", "chr22")


@dataclass(frozen=True)
class Arm:
    chrom: str
    arm: str            # 'p' or 'q'
    start: int          # first flat block index (inclusive)
    end: int            # last flat block index (exclusive)

    @property
    def name(self) -> str:
        return f"{self.chrom[3:]}{self.arm}"

    @property
    def n_blocks(self) -> int:
        return self.end - self.start


@dataclass
class Grid:
    chrom_order: list[str]
    nregions: dict[str, int]
    centromere_block: dict[str, int]
    chrom_lens_bp: dict[str, int]
    region_len: int
    D: int
    offsets: dict[str, int] = field(init=False)
    arms: list[Arm] = field(init=False)          # all 44 arms in chrom_order, p then q
    called_arms: list[Arm] = field(init=False)   # the 39 Taylor arms

    def __post_init__(self):
        self.offsets, off = {}, 0
        for c in self.chrom_order:
            self.offsets[c] = off
            off += self.nregions[c]
        assert off == self.D, "nregions do not sum to D"
        self.arms = []
        for c in self.chrom_order:
            o, cen, n = self.offsets[c], self.centromere_block[c], self.nregions[c]
            self.arms.append(Arm(c, "p", o, o + cen))
            self.arms.append(Arm(c, "q", o + cen, o + n))
        self.called_arms = [a for a in self.arms if not (a.arm == "p" and a.chrom in ACROCENTRIC_P)]

    def block_index(self, chrom: str, r: int) -> int:
        return self.offsets[chrom] + r

    def chrom_slice(self, chrom: str) -> slice:
        o = self.offsets[chrom]
        return slice(o, o + self.nregions[chrom])

    def block_bounds_bp(self, chrom: str, r: int) -> tuple[int, int]:
        """[start, end) in bp of block r of chrom, the last block truncated at the chromosome end."""
        return r * self.region_len, min((r + 1) * self.region_len, self.chrom_lens_bp[chrom])

    def chrom_of_block(self) -> np.ndarray:
        """Array of length D with the chromosome index (0..21) of every flat block."""
        out = np.empty(self.D, dtype=np.int16)
        for i, c in enumerate(self.chrom_order):
            out[self.chrom_slice(c)] = i
        return out


def load_grid(manifest_path: str) -> Grid:
    with open(manifest_path) as f:
        m = json.load(f)
    return Grid(
        chrom_order=list(m["chrom_order"]),
        nregions={c: int(v) for c, v in m["nregions"].items()},
        centromere_block={c: int(v) for c, v in m["centromere_block"].items()},
        chrom_lens_bp={c: int(v) for c, v in m["chrom_lens_bp"].items()},
        region_len=int(m["region_len"]),
        D=int(m["D"]),
    )


if __name__ == "__main__":
    import sys
    g = load_grid(sys.argv[1])
    print(f"D={g.D}, chromosomes={len(g.chrom_order)}, arms={len(g.arms)}, called arms={len(g.called_arms)}")
    for a in g.called_arms[:4] + g.called_arms[-3:]:
        print(f"  {a.name:4s} blocks {a.start}-{a.end - 1} (n={a.n_blocks})")
