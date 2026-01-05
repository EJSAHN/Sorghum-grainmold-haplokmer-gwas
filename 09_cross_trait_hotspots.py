#!/usr/bin/env python3
"""
09_cross_trait_hotspots.py

Summarize cross-trait / cross-method GWAS results into genomic "hotspots".

Input:
- Many GWAS result files (05_* and/or 06_*) with columns:
    SNP-GWAS:    chrom, pos, p
    Haplo-kmer:  chrom, window_start_pos, window_end_pos, p
  (this script will use midpoint position for haplo-kmer markers.)

Output:
- ONE TSV file with hotspot summary.

Selection options:
- --select topn --topn 300        (recommended: always produces content)
- --select bonferroni --alpha 0.05
- --select fdr --alpha 0.10
- --select p --p-thresh 1e-4

Hotspot definition:
- Markers are binned by chromosome and window bins of size --window (default 250 kb).
- A hotspot is one (chrom, bin).
- Lead marker = minimum p across all selected markers that fall in that hotspot.

Optional gene annotation:
- Provide --gff and --annotation to annotate lead markers with nearest gene (+/- flank).
- This version includes robust chromosome alias mapping between GWAS chrom labels
  (e.g., "10") and GFF chrom keys (e.g., "Chr10", "chr10", "Chr10", "Chr10").

Example:
  python scripts/09_cross_trait_hotspots.py ^
    --glob "05_gwas_snp_*.tsv.gz" --glob "06_gwas_haplokmer_*.tsv.gz" ^
    --select topn --topn 300 --window 250000 ^
    --gff Sbicolor_454_v3.1.1.gene.gff3.gz --annotation Sbicolor_454_v3.1.1.P14.annotation_info.txt.gz ^
    --out 09_hotspots_top300_w250kb.tsv

Notes:
- The output can be used directly for "pleiotropy/hotspot" tables.
"""

from __future__ import annotations

import argparse
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Utility: column detection
# -------------------------
def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# -------------------------
# GWAS loading (SNP + haplo-kmer)
# -------------------------
def load_gwas_minimal(path: str) -> pd.DataFrame:
    """
    Load a GWAS TSV/TSV.GZ and standardize to columns: chrom, pos, p, rsid.

    Supported position columns:
      - pos / position / bp / lead_pos
      - window_start_pos (+ window_end_pos) -> midpoint used
    """
    df = pd.read_csv(path, sep="\t", compression="infer")

    chrom_col = _find_col(df.columns.tolist(), ["chrom", "chr", "chromosome"])
    p_col = _find_col(df.columns.tolist(), ["p", "pval", "p_value", "p-value"])
    rsid_col = _find_col(df.columns.tolist(), ["rsid", "marker", "id", "marker_id"])

    pos_col = _find_col(df.columns.tolist(), ["pos", "position", "bp", "lead_pos"])
    wstart_col = _find_col(df.columns.tolist(), ["window_start_pos", "start_pos", "window_start", "start"])
    wend_col = _find_col(df.columns.tolist(), ["window_end_pos", "end_pos", "window_end", "end"])

    if chrom_col is None or p_col is None:
        raise ValueError(f"{path}: need chrom and p columns; got {df.columns.tolist()}")

    out = pd.DataFrame()
    out["chrom"] = df[chrom_col].astype(str)

    if pos_col is not None:
        out["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        if wstart_col is None:
            raise ValueError(
                f"{path}: need pos column OR window_start_pos/window_end_pos. Got {df.columns.tolist()}"
            )
        s = pd.to_numeric(df[wstart_col], errors="coerce")
        if wend_col is not None:
            e = pd.to_numeric(df[wend_col], errors="coerce")
            out["pos"] = (s + e) / 2.0
        else:
            out["pos"] = s

    out["p"] = pd.to_numeric(df[p_col], errors="coerce")

    if rsid_col is not None:
        out["rsid"] = df[rsid_col].astype(str)
    else:
        out["rsid"] = ""

    out = out.dropna(subset=["chrom", "pos", "p"])
    out = out[(out["p"] > 0) & (out["p"] <= 1)]
    out["pos"] = out["pos"].astype(int)
    out["chrom"] = out["chrom"].astype(str)
    return out


# -------------------------
# Multiple testing selection helpers
# -------------------------
def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = p.astype(float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


# -------------------------
# Filename -> labels
# -------------------------
def parse_label_from_filename(fn: str) -> Tuple[str, str]:
    """
    Derive (method, trait_label) from filenames like:
      05_gwas_snp_grainmold_A.tsv.gz
      06_gwas_haplokmer_grainmold_AminusC_k7.tsv.gz
      05_gwas_snp_anthracnose.tsv.gz
      06_gwas_haplokmer_anthracnose_k7.tsv.gz
    """
    stem = Path(fn).name
    stem = re.sub(r"\.tsv(\.gz)?$", "", stem)

    method = "UNKNOWN"
    if "_snp_" in stem:
        method = "SNP"
    elif "_haplokmer_" in stem:
        m = re.search(r"_k(\d+)$", stem)
        method = f"HKMER_k{m.group(1)}" if m else "HKMER"

    trait = stem
    trait = trait.replace("05_gwas_snp_", "")
    trait = trait.replace("06_gwas_haplokmer_", "")

    # normalize labels
    trait = trait.replace("grainmold_", "grainmold:")
    trait = trait.replace("anthracnose", "anthracnose")
    trait = trait.replace("_AminusC", ":A-C")
    trait = trait.replace("_MminusC", ":M-C")
    trait = trait.replace("_A", ":A")
    trait = trait.replace("_M", ":M")
    trait = trait.replace("_C", ":C")
    trait = re.sub(r"_k\d+$", "", trait)

    return method, trait


# -------------------------
# GFF + annotation parsing
# -------------------------
@dataclass
class Gene:
    chrom: str
    start: int
    end: int
    gene_id: str
    name: str


def open_textmaybe_gz(path: str):
    p = str(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "rt", encoding="utf-8", errors="replace")


def load_genes_from_gff(gff_path: str) -> Dict[str, List[Gene]]:
    """
    Load 'gene' features from GFF3 into dict: chrom -> sorted list of Gene(start).
    """
    genes: Dict[str, List[Gene]] = {}
    with open_textmaybe_gz(gff_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, phase, attrs = parts
            if feature != "gene":
                continue
            try:
                s = int(start)
                e = int(end)
            except Exception:
                continue

            gene_id = ""
            name = ""
            for field in attrs.split(";"):
                if field.startswith("ID="):
                    gene_id = field.replace("ID=", "").strip()
                elif field.startswith("Name="):
                    name = field.replace("Name=", "").strip()

            if not gene_id:
                continue

            genes.setdefault(str(chrom), []).append(Gene(str(chrom), s, e, gene_id, name))

    for c in genes:
        genes[c].sort(key=lambda g: g.start)
    return genes


def load_annotation_map(path: str) -> Dict[str, str]:
    """
    Best-effort mapping: gene_id -> description string.
    Uses the last column as a description-like field.
    """
    amap: Dict[str, str] = {}
    with open_textmaybe_gz(path) as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene_id = parts[0].strip()
            desc = parts[-1].strip()
            if gene_id and gene_id not in amap:
                amap[gene_id] = desc
    return amap


# -------------------------
# Chromosome alias resolution
# -------------------------
def resolve_chrom_key(genes_by_chrom: Dict[str, List[Gene]], chrom: str) -> Optional[str]:
    """
    Map GWAS chrom labels (e.g., '10') to GFF chrom keys (e.g., 'Chr10', 'chr10', 'Chr10', 'Chr10').
    Tries multiple common variants.
    """
    c = str(chrom).strip()
    if c in genes_by_chrom:
        return c

    # strip common prefixes
    c2 = re.sub(r"^(chr|chromosome)", "", c, flags=re.IGNORECASE).strip()
    if c2 in genes_by_chrom:
        return c2

    # numeric chromosome handling
    m = re.search(r"(\d+)", c2)
    if m:
        n = int(m.group(1))
        candidates = [
            str(n), f"{n:02d}",
            f"Chr{n}", f"Chr{n:02d}",
            f"chr{n}", f"chr{n:02d}",
        ]
        for cand in candidates:
            if cand in genes_by_chrom:
                return cand

    # last tries: prefix with Chr/chr
    for cand in [f"Chr{c2}", f"chr{c2}"]:
        if cand in genes_by_chrom:
            return cand

    return None


def nearest_gene(
    genes_by_chrom: Dict[str, List[Gene]],
    chrom: str,
    pos: int,
    flank: int,
) -> Tuple[str, str]:
    """
    Return (gene_id, gene_name) for nearest gene within +/- flank, else ("","").
    Chrom alias mapping included.
    """
    chrom_key = resolve_chrom_key(genes_by_chrom, chrom)
    if chrom_key is None:
        return "", ""

    genes = genes_by_chrom[chrom_key]

    left = pos - flank
    right = pos + flank

    best: Optional[Gene] = None
    best_dist: Optional[int] = None

    for g in genes:
        if g.end < left:
            continue
        if g.start > right:
            break

        if pos < g.start:
            dist = g.start - pos
        elif pos > g.end:
            dist = pos - g.end
        else:
            dist = 0

        if best is None or dist < best_dist:  # type: ignore[operator]
            best = g
            best_dist = dist

    if best is None:
        return "", ""
    return best.gene_id, best.name


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="Glob pattern(s) for GWAS TSV(.gz) files")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--window", type=int, default=250_000, help="Window size for hotspot binning (bp)")
    ap.add_argument("--select", choices=["topn", "bonferroni", "fdr", "p"], default="topn")
    ap.add_argument("--topn", type=int, default=300, help="Top N markers per file when --select topn")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for bonferroni or FDR")
    ap.add_argument("--p-thresh", type=float, default=1e-4, help="P-value threshold for --select p")
    ap.add_argument("--gff", default=None, help="GFF3(.gz) for annotation (optional)")
    ap.add_argument("--annotation", default=None, help="Annotation map txt(.gz) (optional)")
    ap.add_argument("--flank", type=int, default=50_000, help="Flank for nearest gene lookup (bp)")
    args = ap.parse_args()

    # Expand globs
    files: List[str] = []
    for pattern in args.glob:
        files.extend([str(p) for p in sorted(Path(".").glob(pattern))])
    files = sorted(set(files))
    if len(files) == 0:
        raise SystemExit("[ERROR] No input files matched your --glob patterns.")

    # Optional gene resources
    genes_by_chrom = None
    ann_map = None
    if args.gff and args.annotation:
        print("[INFO] Loading genes from GFF...")
        genes_by_chrom = load_genes_from_gff(args.gff)
        print(f"[INFO] Loaded genes for {len(genes_by_chrom):,} chromosomes/contigs")
        print("[INFO] Loading annotation map...")
        ann_map = load_annotation_map(args.annotation)
        print(f"[INFO] Loaded annotation entries: {len(ann_map):,}")

    all_rows = []
    for fp in files:
        method, trait = parse_label_from_filename(fp)
        df = load_gwas_minimal(fp)

        # select markers per file
        if args.select == "topn":
            sel = df.nsmallest(args.topn, "p").copy()
        elif args.select == "p":
            sel = df[df["p"] <= args.p_thresh].copy()
        elif args.select == "bonferroni":
            thr = args.alpha / len(df)
            sel = df[df["p"] <= thr].copy()
        elif args.select == "fdr":
            q = bh_fdr(df["p"].to_numpy())
            sel = df[q <= args.alpha].copy()
        else:
            sel = df.nsmallest(args.topn, "p").copy()

        if len(sel) == 0:
            continue

        sel["method"] = method
        sel["trait"] = trait
        sel["source_file"] = Path(fp).name
        all_rows.append(sel)

    if len(all_rows) == 0:
        raise SystemExit("[ERROR] No markers selected. Try --select topn or relax thresholds.")

    hits = pd.concat(all_rows, ignore_index=True)
    hits["chrom"] = hits["chrom"].astype(str)
    hits["pos"] = hits["pos"].astype(int)

    w = int(args.window)
    hits["bin"] = (hits["pos"] // w).astype(int)

    # group by hotspot
    grp = hits.groupby(["chrom", "bin"], as_index=False)

    # lead marker per hotspot (minimum p within group)
    # (pandas warning about groupby.apply is harmless; kept for broad compatibility)
    lead = grp.apply(lambda g: g.loc[g["p"].idxmin()]).reset_index(drop=True)

    # aggregate: how many markers and which traits/methods contribute
    def agg_unique(series: pd.Series) -> str:
        return ";".join(sorted(set(series.astype(str))))

    agg = grp.agg(
        n_markers=("p", "size"),
        best_p=("p", "min"),
        traits=("trait", agg_unique),
        methods=("method", agg_unique),
    ).reset_index()

    out = pd.merge(
        agg,
        lead[["chrom", "bin", "rsid", "pos", "p", "trait", "method", "source_file"]],
        on=["chrom", "bin"],
        how="left",
    )

    out = out.rename(
        columns={
            "pos": "lead_pos",
            "p": "lead_p",
            "rsid": "lead_marker",
            "trait": "lead_trait",
            "method": "lead_method",
            "source_file": "lead_source_file",
        }
    )

    out["window_start"] = out["bin"] * w
    out["window_end"] = out["window_start"] + w - 1
    out["n_traits"] = out["traits"].apply(lambda s: len(str(s).split(";")) if s else 0)
    out["n_methods"] = out["methods"].apply(lambda s: len(str(s).split(";")) if s else 0)

    # annotate lead marker with nearest gene
    out["nearest_gene_id"] = ""
    out["nearest_gene_name"] = ""
    out["nearest_gene_desc"] = ""

    if genes_by_chrom is not None and ann_map is not None:
        gids, gnames, gdescs = [], [], []
        for _, r in out.iterrows():
            gid, gname = nearest_gene(genes_by_chrom, r["chrom"], int(r["lead_pos"]), int(args.flank))
            gids.append(gid)
            gnames.append(gname)
            gdescs.append(ann_map.get(gid, "") if gid else "")
        out["nearest_gene_id"] = gids
        out["nearest_gene_name"] = gnames
        out["nearest_gene_desc"] = gdescs

    # sort by most significant hotspots first
    out = out.sort_values(["best_p", "chrom", "window_start"], ascending=[True, True, True]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote: {out_path}  (hotspots={len(out):,})")


if __name__ == "__main__":
    main()
