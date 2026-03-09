#!/usr/bin/env python3
"""
07_annotate_hits.py

Annotate GWAS hits with nearest sorghum gene(s) using a GFF3 gene model file.

Inputs:
  --hits        : GWAS hit table (TSV or TSV.gz). Must contain either:
                    (chrom,pos)  OR  (chrom,window_start_pos,window_end_pos)
  --gff         : Sbicolor_454_v3.1.1.gene.gff3 (or .gz)
  --annotation  : Sbicolor_454_v3.1.1.P14.annotation_info.txt (or .gz) [optional but recommended]
  --out         : output TSV

Output columns added:
  - hit_pos (integer)
  - nearest_gene_id
  - nearest_gene_start / end / strand
  - nearest_gene_distance_bp
  - genes_within_flank (IDs within +/- flank bp, ';' separated)
  - nearest_gene_annotation (if annotation file provided)

Usage:
  python scripts/07_annotate_hits.py ^
    --hits 05_gwas_snp_A.tsv.gz ^
    --gff Sbicolor_454_v3.1.1.gene.gff3 ^
    --annotation Sbicolor_454_v3.1.1.P14.annotation_info.txt.gz ^
    --out 07_hits_annotated.tsv ^
    --flank 50000
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def open_textmaybe_gz(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_gff_attributes(attr: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for part in attr.split(";"):
        part = part.strip()
        if part == "" or "=" not in part:
            continue
        k, v = part.split("=", 1)
        d[k] = v
    return d


def load_genes_from_gff(gff_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict chrom -> dict with arrays:
      start, end, strand, gene_id
    """
    genes: Dict[str, List[Tuple[int, int, str, str]]] = {}
    with open_textmaybe_gz(gff_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype != "gene":
                continue
            try:
                s = int(start)
                e = int(end)
            except ValueError:
                continue
            ad = parse_gff_attributes(attrs)
            gid = ad.get("ID") or ad.get("Name") or ad.get("gene_id") or ""
            if gid == "":
                continue
            genes.setdefault(seqid, []).append((s, e, strand, gid))

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for chrom, lst in genes.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        starts = np.array([x[0] for x in lst_sorted], dtype=np.int64)
        ends = np.array([x[1] for x in lst_sorted], dtype=np.int64)
        strands = np.array([x[2] for x in lst_sorted], dtype="U2")
        gids = np.array([x[3] for x in lst_sorted], dtype="U80")
        out[chrom] = {"start": starts, "end": ends, "strand": strands, "gene_id": gids}
    return out


def load_annotation_map(ann_path: Path) -> Dict[str, str]:
    """
    Very generic parser:
      - takes first column as gene_id
      - joins remaining columns as a single annotation string
    """
    ann: Dict[str, str] = {}
    with open_textmaybe_gz(ann_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gid = parts[0].strip()
            if gid == "":
                continue
            desc = "\t".join(parts[1:]).strip()
            if gid not in ann:
                ann[gid] = desc
    return ann


def nearest_gene(chrom_genes: Dict[str, np.ndarray], pos: int) -> Tuple[str, int, int, str, int]:
    """
    Returns (gene_id, start, end, strand, distance_bp)
    distance=0 if pos is inside gene.
    """
    starts = chrom_genes["start"]
    ends = chrom_genes["end"]
    strands = chrom_genes["strand"]
    gids = chrom_genes["gene_id"]

    i = int(np.searchsorted(starts, pos))
    # check neighborhood
    lo = max(0, i - 20)
    hi = min(len(starts), i + 20)
    cand = range(lo, hi)

    best = ("", -1, -1, ".", 10**18)
    for j in cand:
        s = int(starts[j]); e = int(ends[j])
        if s <= pos <= e:
            d = 0
        elif pos < s:
            d = s - pos
        else:
            d = pos - e
        if d < best[4]:
            best = (str(gids[j]), s, e, str(strands[j]), int(d))
    return best


def genes_within_flank(chrom_genes: Dict[str, np.ndarray], pos: int, flank: int) -> List[str]:
    starts = chrom_genes["start"]
    ends = chrom_genes["end"]
    gids = chrom_genes["gene_id"]

    left = pos - flank
    right = pos + flank

    hi = int(np.searchsorted(starts, right, side="right"))
    ids: List[str] = []
    for j in range(0, hi):
        if int(ends[j]) >= left:
            ids.append(str(gids[j]))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True, help="GWAS hits TSV/TSV.gz")
    ap.add_argument("--gff", required=True, help="Gene GFF3 (can be .gz)")
    ap.add_argument("--annotation", default=None, help="Annotation info txt/tsv (can be .gz)")
    ap.add_argument("--flank", type=int, default=50000, help="+/- flank bp for gene list (default 50000)")
    ap.add_argument("--out", required=True, help="Output TSV")
    args = ap.parse_args()

    hits_path = Path(args.hits)
    gff_path = Path(args.gff)

    hits = pd.read_csv(hits_path, sep="\t", compression="infer")
    if "chrom" not in hits.columns:
        raise ValueError("Hits table must contain a 'chrom' column.")

    if "pos" in hits.columns:
        hits["hit_pos"] = hits["pos"].astype(int)
    elif ("window_start_pos" in hits.columns) and ("window_end_pos" in hits.columns):
        hits["hit_pos"] = ((hits["window_start_pos"].astype(int) + hits["window_end_pos"].astype(int)) // 2)
    else:
        raise ValueError("Hits must contain either 'pos' or ('window_start_pos' and 'window_end_pos').")

    print("[INFO] Loading genes from GFF...")
    genes = load_genes_from_gff(gff_path)
    print(f"[INFO] Loaded genes for {len(genes)} chromosomes/contigs")

    ann_map: Dict[str, str] = {}
    if args.annotation:
        print("[INFO] Loading annotation map...")
        ann_map = load_annotation_map(Path(args.annotation))
        print(f"[INFO] Loaded annotation entries: {len(ann_map):,}")

    nearest_ids = []
    nearest_starts = []
    nearest_ends = []
    nearest_strands = []
    nearest_dist = []
    within_flank = []
    nearest_anno = []

    for _, row in hits.iterrows():
        ch = str(row["chrom"])
        p = int(row["hit_pos"])
        if ch not in genes:
            nearest_ids.append("")
            nearest_starts.append(np.nan)
            nearest_ends.append(np.nan)
            nearest_strands.append("")
            nearest_dist.append(np.nan)
            within_flank.append("")
            nearest_anno.append("")
            continue

        gid, s, e, strand, d = nearest_gene(genes[ch], p)
        nearest_ids.append(gid)
        nearest_starts.append(s)
        nearest_ends.append(e)
        nearest_strands.append(strand)
        nearest_dist.append(d)

        ids = genes_within_flank(genes[ch], p, args.flank)
        within_flank.append(";".join(ids))

        if gid and (gid in ann_map):
            nearest_anno.append(ann_map[gid])
        else:
            nearest_anno.append("")

    hits["nearest_gene_id"] = nearest_ids
    hits["nearest_gene_start"] = nearest_starts
    hits["nearest_gene_end"] = nearest_ends
    hits["nearest_gene_strand"] = nearest_strands
    hits["nearest_gene_distance_bp"] = nearest_dist
    hits["genes_within_flank"] = within_flank
    hits["nearest_gene_annotation"] = nearest_anno

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
