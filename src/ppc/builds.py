"""Genome-build names and the chr1-length fingerprint used to tell hg19 from hg38.

Shared by :mod:`ppc.grid` and :mod:`ppc.liftover` (kept separate to avoid a circular import).
Canonical build names are ``'hg19'`` (GRCh37), ``'hg38'`` (GRCh38) and ``'custom'`` (toy or
test grids whose chromosome lengths match neither assembly).
"""
from __future__ import annotations

import re
from typing import Optional

HG19_CHR1_BP = 249250621
HG38_CHR1_BP = 248956422


class GenomeBuildMismatch(ValueError):
    """Two genome builds that must agree do not (data vs grid, or grid build vs its lengths)."""


def normalise_build(s: str) -> str:
    """'hg19' / 'GRCh37' -> 'hg19'; 'hg38' / 'GRCh38' -> 'hg38'; 'custom' -> 'custom'.

    Case-insensitive substring match. A label naming both assemblies is ambiguous (e.g. 'GRCh38
    (lifted over from hg19)') and raises ValueError, except SISTEM's manifest label
    'GRCh37/hg19 (SISTEM's table is misnamed hg38)', recognised by the phrase 'misnamed hg38',
    which is hg19. Anything else raises ValueError.
    """
    t = str(s).lower()
    is19 = "hg19" in t or "grch37" in t
    is38 = "hg38" in t or "grch38" in t
    if is19 and is38:
        if re.search(r"misnamed\s+hg38", t):
            return "hg19"
        raise ValueError(f"ambiguous genome build {s!r}: it names both hg19/GRCh37 and hg38/GRCh38")
    if is19:
        return "hg19"
    if is38:
        return "hg38"
    if t.strip() == "custom":
        return "custom"
    raise ValueError(f"unrecognised genome build {s!r} (expected hg19/GRCh37, hg38/GRCh38 or custom)")


def build_from_chr1_length(bp: int) -> Optional[str]:
    """'hg19' or 'hg38' if ``bp`` is that assembly's chr1 length, else None."""
    return {HG19_CHR1_BP: "hg19", HG38_CHR1_BP: "hg38"}.get(int(bp))
