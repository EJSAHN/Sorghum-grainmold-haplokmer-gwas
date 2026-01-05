#!/usr/bin/env python3
"""
29_sensitivity_grid_chr5_core.py

Sensitivity grid for "core hotspot convergence" without re-running GWAS.

It uses your existing GWAS result TSVs:
- 05_gwas_snp_*.tsv.gz
- 06_gwas_haplokmer_*.tsv.gz

For each (window_bp, p_thresh) combination:
1) Filter each file by p <= p_thresh
2) Bin markers into hotspots (bin = pos // window)
3) Hotspot metrics:
   - n_traits (#unique trait labels)
   - n_methods (#unique methods: SNP vs HKMER_k7)
   - strength = sum(-log10(p)) across all selected hits in hotspot
4) Define core: n_traits >= core_traits AND n_methods >= core_methods
5) Pick strongest core hotspot and record if it overlaps chr5:60.25-60.50Mb

Outputs:
- <out_prefix>_grid.tsv
- <out_prefix>_heatmap_strength.(png/pdf)
- <out_prefix>_heatmap_present.(png/pdf)

Example:
python scripts/29_sensitivity_grid_chr5_core.py --glob "05_gwas_snp_*.tsv.gz" --glob "06_gwas_haplokmer_*.tsv.gz" --out_prefix 29_sens --p_list 1e-4,5e-5,1e-5 --window_list 250000,500000
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CHR5_START = 60250000
CHR5_END = 60499999


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


def load_gwas_minimal(path: str) -> pd.DataFrame:
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


def compute_hotspots(df_all: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df_all.copy()
    d["bin"] = (d["pos"] // window).astype(int)
    d["hotspot_key"] = d["chrom"].astype(str) + ":" + d["bin"].astype(str)
    d["neglogp"] = -np.log10(np.clip(d["p"].to_numpy(float), np.nextafter(0, 1), 1.0))

    g = d.groupby("hotspot_key").agg(
        chrom=("chrom", "first"),
        bin=("bin", "first"),
        n_traits=("trait", pd.Series.nunique),
        n_methods=("method", pd.Series.nunique),
        strength=("neglogp", "sum"),
        min_p=("p", "min"),
    ).reset_index(drop=False)

    g["bin_start"] = g["bin"].astype(int) * int(window)
    g["bin_end"] = g["bin_start"] + int(window) - 1
    return g


def overlap_chr5_core(chrom: str, start: int, end: int) -> bool:
    if str(chrom) != "5":
        return False
    # interval overlap
    return not (end < CHR5_START or start > CHR5_END)


def heatmap_plot(mat: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, out_prefix: str, suffix: str, vmin=None, vmax=None):
    fig = plt.figure(figsize=(1.6 + 1.1 * len(xlabels), 1.6 + 0.8 * len(ylabels)))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--p_list", default="1e-4,5e-5,1e-5")
    ap.add_argument("--window_list", default="250000,500000")
    ap.add_argument("--core_traits", type=int, default=3)
    ap.add_argument("--core_methods", type=int, default=2)
    args = ap.parse_args()

    # files
    files = []
    for pat in args.glob:
        files.extend([str(p) for p in sorted(Path(".").glob(pat))])
    files = sorted(set(files))
    if not files:
        raise SystemExit("[ERROR] No GWAS files found by --glob")

    # parse grids
    p_list = [float(x) for x in args.p_list.split(",")]
    window_list = [int(x) for x in args.window_list.split(",")]

    # preload gwas minimal
    objs = []
    for fp in files:
        method, trait = parse_label_from_filename(fp)
        df = load_gwas_minimal(fp)
        df["method"] = "SNP" if method == "SNP" else "HKMER"
        df["trait"] = trait
        objs.append((fp, df))

    rows = []
    # matrices for heatmaps: rows=windows, cols=p_thresh
    strength_mat = np.zeros((len(window_list), len(p_list)), dtype=float)
    present_mat = np.zeros((len(window_list), len(p_list)), dtype=float)

    for wi, w in enumerate(window_list):
        for pi, pthr in enumerate(p_list):
            # filter per file and concatenate
            parts = []
            for fp, df in objs:
                parts.append(df[df["p"] <= pthr].copy())
            all_hits = pd.concat(parts, ignore_index=True)
            if all_hits.empty:
                rows.append({
                    "window": w, "p_thresh": pthr,
                    "n_hotspots": 0, "n_core": 0,
                    "best_core_chrom": "", "best_core_start": "", "best_core_end": "",
                    "best_core_strength": 0.0,
                    "best_core_overlaps_chr5": 0,
                })
                strength_mat[wi, pi] = 0.0
                present_mat[wi, pi] = 0.0
                continue

            hot = compute_hotspots(all_hits, w)
            core = hot[(hot["n_traits"] >= args.core_traits) & (hot["n_methods"] >= args.core_methods)].copy()

            if core.empty:
                rows.append({
                    "window": w, "p_thresh": pthr,
                    "n_hotspots": int(len(hot)), "n_core": 0,
                    "best_core_chrom": "", "best_core_start": "", "best_core_end": "",
                    "best_core_strength": 0.0,
                    "best_core_overlaps_chr5": 0,
                })
                strength_mat[wi, pi] = 0.0
                present_mat[wi, pi] = 0.0
                continue

            core = core.sort_values("strength", ascending=False).reset_index(drop=True)
            best = core.iloc[0]
            overlaps = 1 if overlap_chr5_core(str(best["chrom"]), int(best["bin_start"]), int(best["bin_end"])) else 0

            rows.append({
                "window": w, "p_thresh": pthr,
                "n_hotspots": int(len(hot)), "n_core": int(len(core)),
                "best_core_chrom": str(best["chrom"]),
                "best_core_start": int(best["bin_start"]),
                "best_core_end": int(best["bin_end"]),
                "best_core_strength": float(best["strength"]),
                "best_core_overlaps_chr5": overlaps,
            })
            strength_mat[wi, pi] = float(best["strength"])
            present_mat[wi, pi] = float(overlaps)

    out_df = pd.DataFrame(rows).sort_values(["window", "p_thresh"])
    out_df.to_csv(f"{args.out_prefix}_grid.tsv", sep="\t", index=False)
    print(f"[OK] Wrote: {args.out_prefix}_grid.tsv")

    xlabels = [f"p≤{p:.0e}" for p in p_list]
    ylabels = [f"window={w//1000}kb" for w in window_list]

    heatmap_plot(
        strength_mat,
        xlabels, ylabels,
        "Best core hotspot strength (Σ−log10 p)\n(per window × p-threshold)",
        args.out_prefix,
        "heatmap_strength",
        vmin=0
    )
    heatmap_plot(
        present_mat,
        xlabels, ylabels,
        "Does best core hotspot overlap chr5:60.25–60.50Mb? (1=yes, 0=no)",
        args.out_prefix,
        "heatmap_present",
        vmin=0, vmax=1
    )
    print(f"[OK] Wrote figures: {args.out_prefix}_heatmap_strength.(png/pdf), {args.out_prefix}_heatmap_present.(png/pdf)")


if __name__ == "__main__":
    main()
