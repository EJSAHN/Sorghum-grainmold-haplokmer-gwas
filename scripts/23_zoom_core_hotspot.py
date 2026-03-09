#!/usr/bin/env python3
"""
23_zoom_core_hotspot.py

Zoom into the strongest CORE hotspot under a stringent selection rule
(e.g., p <= 1e-4) and visualize SNP vs HKMER association patterns across traits.

What it does
------------
1) Load GWAS result files (SNP and HKMER) using glob patterns.
2) Apply selection rule (default: p <= 1e-4) to define "significant hits".
3) Bin hits into hotspots using --window (default 250kb).
4) Find CORE hotspots: n_traits >= core_traits AND n_methods >= core_methods.
5) Choose the CORE hotspot with maximum strength:
   strength = sum(-log10(p)) across all selected hits within the hotspot.
6) Generate zoom plots (PNG 300dpi + PDF):
   - 2 rows: SNP, HKMER
   - 6 columns: traits (anthracnose, grainmold:A, A-C, C, M, M-C)
   - Scatter -log10(p) vs position within a +/- flank window.
   - Shade the hotspot bin and mark the bin center.

Outputs
-------
- <out_prefix>_core_hotspot_summary.tsv
- <out_prefix>_zoom_core_hotspot.png
- <out_prefix>_zoom_core_hotspot.pdf

Usage example (ONE LINE in CMD; avoid ^ to prevent "More?"):
python scripts/23_zoom_core_hotspot.py --glob "05_gwas_snp_*.tsv.gz" --glob "06_gwas_haplokmer_*.tsv.gz" --select p --p_thresh 1e-4 --window 250000 --core_traits 3 --core_methods 2 --flank 2000000 --out_prefix 23_zoom_p1e4

Optional gene label (requires GFF):
python scripts/23_zoom_core_hotspot.py --glob "05_gwas_snp_*.tsv.gz" --glob "06_gwas_haplokmer_*.tsv.gz" --select p --p_thresh 1e-4 --window 250000 --core_traits 3 --core_methods 2 --flank 2000000 --out_prefix 23_zoom_p1e4 --gff Sbicolor_454_v3.1.1.gene.gff3.gz
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_label_from_filename(fn: str) -> Tuple[str, str]:
    """
    Derive (method, trait_label) from filenames.
    Standardize contrast tokens to A-C and M-C.
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
    trait = trait.replace("grainmold_", "grainmold:")
    trait = trait.replace("anthracnose", "anthracnose")
    trait = re.sub(r"_k\d+$", "", trait)

    trait = trait.replace("AminusC", "A-C")
    trait = trait.replace("MminusC", "M-C")

    return method, trait


def load_gwas_minimal(path: str) -> pd.DataFrame:
    """
    Standardize to columns: chrom, pos, p.
    Supports haplo-kmer window format (midpoint).
    """
    df = pd.read_csv(path, sep="\t", compression="infer")

    chrom_col = _find_col(df.columns.tolist(), ["chrom", "chr", "chromosome"])
    p_col = _find_col(df.columns.tolist(), ["p", "pval", "p_value", "p-value"])

    pos_col = _find_col(df.columns.tolist(), ["pos", "position", "bp", "lead_pos"])
    wstart_col = _find_col(df.columns.tolist(), ["window_start_pos", "start_pos", "window_start", "start"])
    wend_col = _find_col(df.columns.tolist(), ["window_end_pos", "end_pos", "window_end", "end"])

    if chrom_col is None or p_col is None:
        raise ValueError(f"{path}: need chrom and p columns; got {df.columns.tolist()}")

    out = pd.DataFrame()
    out["chrom"] = df[chrom_col].astype(str)
    out["p"] = pd.to_numeric(df[p_col], errors="coerce")

    if pos_col is not None:
        out["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        if wstart_col is None:
            raise ValueError(f"{path}: need pos or window_start_pos; got {df.columns.tolist()}")
        s = pd.to_numeric(df[wstart_col], errors="coerce")
        if wend_col is not None:
            e = pd.to_numeric(df[wend_col], errors="coerce")
            out["pos"] = (s + e) / 2.0
        else:
            out["pos"] = s

    out = out.dropna(subset=["chrom", "pos", "p"])
    out = out[(out["p"] > 0) & (out["p"] <= 1)]
    out["pos"] = out["pos"].astype(int)
    out["chrom"] = out["chrom"].astype(str)
    return out.reset_index(drop=True)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = p.astype(float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


def select_hits(df: pd.DataFrame, select: str, topn: int, alpha: float, p_thresh: float) -> pd.DataFrame:
    if select == "topn":
        return df.nsmallest(topn, "p").copy()
    if select == "p":
        return df[df["p"] <= p_thresh].copy()
    if select == "bonferroni":
        thr = alpha / len(df)
        return df[df["p"] <= thr].copy()
    if select == "fdr":
        q = bh_fdr(df["p"].to_numpy())
        return df[q <= alpha].copy()
    raise ValueError(f"Unknown selection mode: {select}")


def build_contrib(sel: pd.DataFrame, trait: str, method: str, window: int) -> pd.DataFrame:
    d = sel.copy()
    d["trait"] = trait
    d["method"] = method
    d["bin"] = (d["pos"] // window).astype(int)
    d["hotspot_key"] = d["chrom"].astype(str) + ":" + d["bin"].astype(str)
    d["neglogp"] = -np.log10(np.clip(d["p"].to_numpy(float), np.nextafter(0, 1), 1.0))
    return d[["hotspot_key", "chrom", "bin", "pos", "p", "neglogp", "trait", "method"]]


def summarize_hotspots(contrib: pd.DataFrame, core_traits: int, core_methods: int) -> pd.DataFrame:
    """
    Return per-hotspot table with n_traits, n_methods, strength.
    """
    g = contrib.groupby("hotspot_key").agg(
        chrom=("chrom", "first"),
        bin=("bin", "first"),
        n_traits=("trait", pd.Series.nunique),
        n_methods=("method", pd.Series.nunique),
        strength=("neglogp", "sum"),
    ).reset_index(drop=False)

    g["is_core"] = (g["n_traits"] >= core_traits) & (g["n_methods"] >= core_methods)
    return g


# -------------------------
# Optional nearest gene label
# -------------------------
def open_textmaybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def resolve_chrom_key(genes_by_chrom: Dict[str, List], chrom: str) -> Optional[str]:
    c = str(chrom).strip()
    if c in genes_by_chrom:
        return c
    c2 = re.sub(r"^(chr|chromosome)", "", c, flags=re.IGNORECASE).strip()
    m = re.search(r"(\d+)", c2)
    if m:
        n = int(m.group(1))
        for cand in [str(n), f"{n:02d}", f"Chr{n}", f"Chr{n:02d}", f"chr{n}", f"chr{n:02d}"]:
            if cand in genes_by_chrom:
                return cand
    for cand in [f"Chr{c2}", f"chr{c2}"]:
        if cand in genes_by_chrom:
            return cand
    return None


def load_genes_from_gff(gff_path: str) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Load gene intervals: chrom -> list of (start,end,gene_id), sorted by start.
    """
    genes: Dict[str, List[Tuple[int, int, str]]] = {}
    with open_textmaybe_gz(gff_path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, *_rest, attrs = parts
            if feature != "gene":
                continue
            try:
                s = int(start); e = int(end)
            except Exception:
                continue
            gid = ""
            for field in attrs.split(";"):
                if field.startswith("ID="):
                    gid = field.replace("ID=", "").strip()
            if gid:
                genes.setdefault(str(chrom), []).append((s, e, gid))
    for c in genes:
        genes[c].sort(key=lambda x: x[0])
    return genes


def nearest_gene(genes_by_chrom: Dict[str, List[Tuple[int, int, str]]], chrom: str, pos: int, flank: int) -> str:
    ck = resolve_chrom_key(genes_by_chrom, chrom)
    if ck is None:
        return ""
    genes = genes_by_chrom[ck]
    left = pos - flank
    right = pos + flank
    best_gid = ""
    best_dist = None
    for s, e, gid in genes:
        if e < left:
            continue
        if s > right:
            break
        if pos < s:
            dist = s - pos
        elif pos > e:
            dist = pos - e
        else:
            dist = 0
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_gid = gid
    return best_gid


# -------------------------
# Plotting
# -------------------------
def make_zoom_plot(
    region_hits: Dict[Tuple[str, str], pd.DataFrame],
    chrom: str,
    start: int,
    end: int,
    bin_start: int,
    bin_end: int,
    out_prefix: str,
    gene_label: str = "",
):
    """
    Create 2x6 grid:
      rows: SNP, HKMER
      cols: 6 traits (sorted)
    """
    traits = sorted({t for (m, t) in region_hits.keys()})
    methods = ["SNP", "HKMER"]

    # fix trait order: anthracnose first, then grainmold:* in a stable order
    def trait_key(x: str):
        if x == "anthracnose":
            return (0, x)
        return (1, x)

    traits = sorted(traits, key=trait_key)

    fig, axes = plt.subplots(
        nrows=2, ncols=len(traits),
        figsize=(3.2 * len(traits), 7.0),
        sharex=False, sharey=False
    )

    center = (bin_start + bin_end) // 2

    for r, m in enumerate(methods):
        for c, t in enumerate(traits):
            ax = axes[r, c] if len(traits) > 1 else axes[r]
            df = region_hits.get((m, t), pd.DataFrame(columns=["pos", "p"]))
            if not df.empty:
                y = -np.log10(np.clip(df["p"].to_numpy(float), np.nextafter(0, 1), 1.0))
                ax.scatter(df["pos"], y, s=10, linewidths=0, alpha=0.8)
                ax.set_ylim(bottom=0)
            ax.axvspan(bin_start, bin_end, alpha=0.15)     # hotspot bin
            ax.axvline(center, linestyle="--", linewidth=1) # bin center
            ax.set_xlim(start, end)

            if r == 0:
                ax.set_title(t, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{m}\n-log10(p)", fontsize=11)

            if r == 1:
                ax.set_xlabel(f"pos on chr{chrom} (bp)", fontsize=10)

    title = f"Core hotspot zoom (chr{chrom}:{bin_start:,}-{bin_end:,})"
    if gene_label:
        title += f"\nNearest gene: {gene_label}"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(f"{out_prefix}_zoom_core_hotspot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_zoom_core_hotspot.pdf", bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="Glob patterns for GWAS result files")
    ap.add_argument("--select", choices=["topn", "p", "bonferroni", "fdr"], default="p")
    ap.add_argument("--topn", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--p_thresh", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=250_000)
    ap.add_argument("--core_traits", type=int, default=3)
    ap.add_argument("--core_methods", type=int, default=2)
    ap.add_argument("--flank", type=int, default=2_000_000, help="+/- flank around hotspot bin for zoom")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--gff", default=None, help="Optional: gene gff3(.gz) to label nearest gene")
    ap.add_argument("--gene_flank", type=int, default=200_000, help="Flank for nearest gene lookup")
    args = ap.parse_args()

    # expand
    files: List[str] = []
    for pat in args.glob:
        files.extend([str(p) for p in sorted(Path(".").glob(pat))])
    files = sorted(set(files))
    if not files:
        raise SystemExit("[ERROR] No files matched patterns.")

    # load + select contrib for hotspot discovery
    contrib_parts = []
    file_objs = []

    for fp in files:
        method, trait = parse_label_from_filename(fp)
        df = load_gwas_minimal(fp)
        sel = select_hits(df, args.select, args.topn, args.alpha, args.p_thresh)
        contrib_parts.append(build_contrib(sel, trait, method, args.window))
        file_objs.append({"file": fp, "method": method, "trait": trait})

    contrib = pd.concat(contrib_parts, ignore_index=True)
    hotspot_tbl = summarize_hotspots(contrib, args.core_traits, args.core_methods)

    core = hotspot_tbl[hotspot_tbl["is_core"]].copy()
    if core.empty:
        raise SystemExit("[ERROR] No core hotspots found under current settings. Try relaxing thresholds or core_traits.")

    # pick strongest core hotspot
    core = core.sort_values("strength", ascending=False).reset_index(drop=True)
    best = core.iloc[0]
    chrom = str(best["chrom"])
    bin_idx = int(best["bin"])
    bin_start = bin_idx * int(args.window)
    bin_end = bin_start + int(args.window) - 1

    # write summary
    summary_path = f"{args.out_prefix}_core_hotspot_summary.tsv"
    best_df = core.head(10).copy()
    best_df["bin_start"] = best_df["bin"].astype(int) * int(args.window)
    best_df["bin_end"] = best_df["bin_start"] + int(args.window) - 1
    best_df.to_csv(summary_path, sep="\t", index=False)
    print(f"[OK] Wrote: {summary_path}")
    print("[TOP core hotspots by strength]")
    print(best_df[["chrom", "bin_start", "bin_end", "n_traits", "n_methods", "strength"]].to_string(index=False))

    # optional gene label
    gene_label = ""
    if args.gff:
        genes = load_genes_from_gff(args.gff)
        lead_pos = int((bin_start + bin_end) // 2)
        gene_label = nearest_gene(genes, chrom, lead_pos, int(args.gene_flank))

    # build region plots
    start = max(0, bin_start - int(args.flank))
    end = bin_end + int(args.flank)

    region_hits: Dict[Tuple[str, str], pd.DataFrame] = {}

    # for each file, subset full df to region and store
    for obj in file_objs:
        fp = obj["file"]
        method = obj["method"]
        trait = obj["trait"]
        df = load_gwas_minimal(fp)
        dsub = df[(df["chrom"].astype(str) == chrom) & (df["pos"] >= start) & (df["pos"] <= end)].copy()
        region_hits[(method if method == "SNP" else "HKMER", trait)] = dsub

    make_zoom_plot(region_hits, chrom, start, end, bin_start, bin_end, args.out_prefix, gene_label=gene_label)
    print(f"[OK] Wrote figures: {args.out_prefix}_zoom_core_hotspot.(png/pdf)")


if __name__ == "__main__":
    main()
