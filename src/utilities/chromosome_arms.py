"""hg38 chromosome-arm lookup for mapping genomic bins to p/q arms.

Centromere split points are the p11.1/q11(.1) band boundary from UCSC's
hg38 cytoBandIdeo table (hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBandIdeo.txt.gz),
verified 2026-07 against CHISEL's own bin coordinates (chr1 max bin end =
248,956,422, matching hg38's chr1 length exactly; hg19's is 249,250,621).
"""

HG38_CENTROMERE = {
    "chr1": 123_400_000, "chr2": 93_900_000, "chr3": 90_900_000,
    "chr4": 50_000_000, "chr5": 48_800_000, "chr6": 59_800_000,
    "chr7": 60_100_000, "chr8": 45_200_000, "chr9": 43_000_000,
    "chr10": 39_800_000, "chr11": 53_400_000, "chr12": 35_500_000,
    "chr13": 17_700_000, "chr14": 17_200_000, "chr15": 19_000_000,
    "chr16": 36_800_000, "chr17": 25_100_000, "chr18": 18_500_000,
    "chr19": 26_200_000, "chr20": 28_100_000, "chr21": 12_000_000,
    "chr22": 15_000_000,
}


def bin_to_arm(chrom: str, start: int, end: int) -> str:
    """Map a genomic bin to its chromosome arm label, e.g. '1p', '17q'.

    Assigns by the bin's midpoint relative to the centromere split point,
    so a bin straddling the centromere goes to whichever side it mostly overlaps.
    """
    chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    centromere = HG38_CENTROMERE[chrom]
    midpoint = (start + end) / 2
    arm = "p" if midpoint < centromere else "q"
    return f"{chrom.replace('chr', '')}{arm}"
