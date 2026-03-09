#!/usr/bin/env python3
"""
24_core_hotspot_decomposition.py

Decompose the strongest CORE hotspot (>=core_traits traits AND >=core_methods methods)
under a selection rule (default: p<=1e-4) into per-trait / per-method contributions.

This is designed to turn the "core hotspot convergence" result into a reviewer-proof,
quantitative breakdown.

What it does
------------
1) Load GWAS result files (SNP + HKMER) from glob patterns.
2) Apply a selection rule (topn / p / bonferroni / fdr) per file.
3) Bin selected hits into hotspots using --window.
4) Find CORE hotspots and choose the one with maximum hotspot strength:
     strength = sum(-log10(p)) over all selected hits in the hotspot (across files)
5) For that best core hotspot, compute per-file contributions:
   - n_hits_in_bin
   - strength_in_bin = sum(-log10(p)) within the bin
   - min_p_in_bin
6) Write ONE TSV output and ONE stacked bar figure (PNG 300dpi + PDF).

Outputs
-------
- <out_prefix>_core_decomposition.tsv
- <out_prefix>_core_decomposition_stackedbar.(png/pdf)

Usage
-----
python scripts/24_core_hotspot_decomposition.py --glob "05_gwas_snp_*.tsv.gz" --glob "06_gwas_haplokmer_*.tsv.gz" --select p --p_thresh 1e-4 --window 250000 --core_traits 3 --core_methods 2 --out_prefix 24_core_p1e4

Recommended
-----------
Use the same settings as your strength-permutation run:
  --select p --p_thresh 1e-4 --window 250000 --core_traits 3 --core_methods 2
"""

from __future__ import annotations

import argparse
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


def load_gwas_minimal(path: str) -> pd.DataFrame:
    """
    Standardize GWAS file to columns: chrom, pos, p.
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


def make_contrib(sel: pd.DataFrame, trait: str, method: str, window: int) -> pd.DataFrame:
    d = sel.copy()
    d["trait"] = trait
    d["method"] = method
    d["bin"] = (d["pos"] // window).astype(int)
    d["hotspot_key"] = d["chrom"].astype(str) + ":" + d["bin"].astype(str)
    d["neglogp"] = -np.log10(np.clip(d["p"].to_numpy(float), np.nextafter(0, 1), 1.0))
    return d


def core_hotspot_table(contrib_all: pd.DataFrame, core_traits: int, core_methods: int) -> pd.DataFrame:
    g = contrib_all.groupby("hotspot_key").agg(
        chrom=("chrom", "first"),
        bin=("bin", "first"),
        n_traits=("trait", pd.Series.nunique),
        n_methods=("method", pd.Series.nunique),
        strength=("neglogp", "sum"),
    ).reset_index(drop=False)
    g["is_core"] = (g["n_traits"] >= core_traits) & (g["n_methods"] >= core_methods)
    return g


def stacked_bar_plot(df: pd.DataFrame, out_prefix: str, title: str):
    """
    Stacked bar of strength contributions per method for each trait.
    """
    # order traits
    def trait_key(x: str):
        if x == "anthracnose":
            return (0, x)
        return (1, x)

    traits = sorted(df["trait"].unique().tolist(), key=trait_key)

    methods = ["SNP", "HKMER"]
    pivot = df.pivot_table(index="trait", columns="method", values="strength_in_bin", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(traits)

    # Build stacked bars: SNP + HKMER
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    bottom = np.zeros(len(pivot), dtype=float)
    for m in methods:
        vals = pivot[m].to_numpy(float) if m in pivot.columns else np.zeros(len(pivot))
        ax.bar(pivot.index.tolist(), vals, bottom=bottom, label=m)
        bottom += vals

    ax.set_ylabel("Hotspot strength contribution (Σ -log10(p))")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_core_decomposition_stackedbar.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_core_decomposition_stackedbar.pdf", bbox_inches="tight")
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
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    # expand files
    files: List[str] = []
    for pat in args.glob:
        files.extend([str(p) for p in sorted(Path(".").glob(pat))])
    files = sorted(set(files))
    if not files:
        raise SystemExit("[ERROR] No files matched patterns.")

    # load + select contributions
    contrib_parts = []
    file_rows = []
    for fp in files:
        method, trait = parse_label_from_filename(fp)
        df = load_gwas_minimal(fp)
        sel = select_hits(df, args.select, args.topn, args.alpha, args.p_thresh)
        contrib = make_contrib(sel, trait, method if method == "SNP" else "HKMER", args.window)
        contrib_parts.append(contrib)
        file_rows.append((Path(fp).name, method, trait, len(df), len(sel)))

    contrib_all = pd.concat(contrib_parts, ignore_index=True) if contrib_parts else pd.DataFrame()
    if contrib_all.empty:
        raise SystemExit("[ERROR] No selected hits under current selection rule. Relax threshold or change --select.")

    # core hotspot discovery
    hot = core_hotspot_table(contrib_all, args.core_traits, args.core_methods)
    core = hot[hot["is_core"]].copy()
    if core.empty:
        raise SystemExit("[ERROR] No core hotspots found under current settings.")

    # choose strongest core
    core = core.sort_values("strength", ascending=False).reset_index(drop=True)
    best = core.iloc[0]
    chrom = str(best["chrom"])
    bin_idx = int(best["bin"])
    bin_start = bin_idx * int(args.window)
    bin_end = bin_start + int(args.window) - 1
    hotspot_key = str(best["hotspot_key"])

    # per-file decomposition within that hotspot bin
    in_bin = contrib_all[contrib_all["hotspot_key"] == hotspot_key].copy()
    if in_bin.empty:
        raise SystemExit("[ERROR] Internal: no records found for selected core hotspot_key.")

    # summarize per (trait, method)
    dec = in_bin.groupby(["trait", "method"], as_index=False).agg(
        n_hits_in_bin=("p", "size"),
        strength_in_bin=("neglogp", "sum"),
        min_p_in_bin=("p", "min"),
    )
    dec["chrom"] = chrom
    dec["bin_start"] = bin_start
    dec["bin_end"] = bin_end
    dec["hotspot_key"] = hotspot_key

    # also output a concise "who contributed what" table
    out_tsv = f"{args.out_prefix}_core_decomposition.tsv"
    dec.to_csv(out_tsv, sep="\t", index=False)

    # print summary to console
    print("[CORE hotspot chosen]")
    print(f"  chrom={chrom}  bin={bin_idx}  range={bin_start:,}-{bin_end:,}  n_traits={int(best['n_traits'])}  n_methods={int(best['n_methods'])}  strength={best['strength']:.6f}")
    print("[PER-file contributions]")
    print(dec.sort_values(["trait", "method"]).to_string(index=False))
    print(f"[OK] Wrote: {out_tsv}")

    # plot stacked bar
    title = f"Core hotspot decomposition (chr{chrom}:{bin_start:,}-{bin_end:,}; p<= {args.p_thresh:g})"
    stacked_bar_plot(dec, args.out_prefix, title)
    print(f"[OK] Wrote figures: {args.out_prefix}_core_decomposition_stackedbar.(png/pdf)")


if __name__ == "__main__":
    main()
