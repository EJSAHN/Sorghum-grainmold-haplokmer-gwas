#!/usr/bin/env python3
"""
08_compare_manhattan_qq.py

Make a manuscript-ready comparison figure for TWO GWAS result files
(e.g., SNP-GWAS vs haplotype-kmer GWAS) for the same trait.

Output: ONE image file (PNG/PDF/SVG) containing:
  - Manhattan (GWAS1)
  - Manhattan (GWAS2)
  - QQ plot (GWAS1)
  - QQ plot (GWAS2)

Supports BOTH formats:
1) SNP-GWAS output: chrom, pos, p
2) Haplo-kmer output: chrom, window_start_pos/window_end_pos, p
   - uses midpoint position for plotting.

Usage:
  python scripts/08_compare_manhattan_qq.py ^
    --gwas1 05_gwas_snp_grainmold_A.tsv.gz --label1 "SNP" ^
    --gwas2 06_gwas_haplokmer_grainmold_A_k7.tsv.gz --label2 "Haplo-kmer (k=7)" ^
    --out figs/Fig_grainmold_A_compare.png --title "Grain mold: A"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import chi2
except Exception:
    chi2 = None


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_gwas(path: str) -> pd.DataFrame:
    """
    Load a GWAS file and return standardized columns: chrom, pos, p.

    Supported position columns:
      - pos / position / bp / lead_pos
      - window_start_pos (+ window_end_pos) -> midpoint used
    """
    df = pd.read_csv(path, sep="\t", compression="infer")

    chrom_col = _find_col(df.columns.tolist(), ["chrom", "chr", "chromosome"])
    p_col = _find_col(df.columns.tolist(), ["p", "pval", "p_value", "p-value"])

    pos_col = _find_col(df.columns.tolist(), ["pos", "position", "bp", "lead_pos"])
    wstart_col = _find_col(df.columns.tolist(), ["window_start_pos", "start_pos", "window_start", "start"])
    wend_col = _find_col(df.columns.tolist(), ["window_end_pos", "end_pos", "window_end", "end"])

    if chrom_col is None or p_col is None:
        raise ValueError(f"GWAS file must contain chrom and p columns. Found: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["chrom"] = df[chrom_col].astype(str)

    if pos_col is not None:
        out["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        if wstart_col is None:
            raise ValueError(
                "GWAS file must contain a position column. "
                "Accepted: pos/position/bp/lead_pos OR window_start_pos(+window_end_pos). "
                f"Found: {df.columns.tolist()}"
            )
        s = pd.to_numeric(df[wstart_col], errors="coerce")
        if wend_col is not None:
            e = pd.to_numeric(df[wend_col], errors="coerce")
            out["pos"] = (s + e) / 2.0
        else:
            out["pos"] = s

    out["p"] = pd.to_numeric(df[p_col], errors="coerce")

    out = out.dropna(subset=["chrom", "pos", "p"])
    out = out[(out["p"] > 0) & (out["p"] <= 1)]
    out["pos"] = out["pos"].astype(float)
    return out


def _chrom_sort_key(x: str) -> Tuple[int, str]:
    s = str(x).strip()
    try:
        return (0, f"{int(s):03d}")
    except Exception:
        return (1, s)


def build_manhattan_coords(df: pd.DataFrame, gap: int = 1_000_000) -> Tuple[pd.DataFrame, List[Tuple[float, str]]]:
    """
    Add cumulative x coordinate for Manhattan plot.
    Returns: (df2 with 'x' and 'mlog10p'), ticks [(x_center, chrom_label)]
    """
    d = df.copy()
    d["chrom"] = d["chrom"].astype(str)
    chroms = sorted(d["chrom"].unique(), key=_chrom_sort_key)

    x_offset = 0.0
    ticks: List[Tuple[float, str]] = []
    d["x"] = np.nan

    for chrom in chroms:
        sub = d[d["chrom"] == chrom].sort_values("pos")
        if sub.empty:
            continue
        x_start = x_offset
        x_vals = sub["pos"].to_numpy(dtype=float) + x_offset
        d.loc[sub.index, "x"] = x_vals

        x_end = float(np.nanmax(x_vals))
        ticks.append(((x_start + x_end) / 2.0, chrom))
        x_offset = x_end + float(gap)

    # pandas Series.clip supports lower/upper; keep this here (d["p"] is a Series)
    d["mlog10p"] = -np.log10(d["p"].clip(lower=np.nextafter(0, 1)))
    d = d.dropna(subset=["x", "mlog10p"])
    return d, ticks


def genomic_inflation_lambda(p: np.ndarray) -> float:
    if chi2 is None:
        return float("nan")
    p = p[(p > 0) & (p <= 1)]
    if p.size == 0:
        return float("nan")
    chisq = chi2.isf(p, df=1)
    med = np.median(chisq)
    return float(med / 0.4549364)


def qq_data(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = p[(p > 0) & (p <= 1)]
    p = np.sort(p)
    n = p.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n)

    # IMPORTANT: numpy ndarray.clip does NOT accept lower/upper keywords.
    p_safe = np.clip(p, np.nextafter(0, 1), 1.0)
    obs = -np.log10(p_safe)
    return exp, obs


def plot_manhattan(ax, df: pd.DataFrame, ticks: List[Tuple[float, str]], title: str, bonf_alpha: float = 0.05):
    ax.scatter(df["x"], df["mlog10p"], s=4, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=8)
    ax.margins(x=0.01)

    m = len(df)
    if m > 0 and bonf_alpha > 0:
        bonf = bonf_alpha / m
        ax.axhline(-math.log10(bonf), linestyle="--", linewidth=1)
        ax.text(
            0.99, 0.95, f"Bonferroni 0.05/m\nm={m:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8
        )


def plot_qq(ax, p: np.ndarray, title: str):
    exp, obs = qq_data(p)

    ax.scatter(exp, obs, s=6, linewidths=0)

    maxv = 1.0
    if exp.size and obs.size:
        maxv = float(max(exp.max(), obs.max()))
        if not np.isfinite(maxv) or maxv <= 0:
            maxv = 1.0

    ax.plot([0, maxv], [0, maxv], linestyle="--", linewidth=1)

    lam = genomic_inflation_lambda(p)
    if np.isfinite(lam):
        ax.set_title(f"{title}  (lambda={lam:.3f})", fontsize=11)
    else:
        ax.set_title(title, fontsize=11)

    ax.set_xlabel(r"Expected $-\log_{10}(p)$")
    ax.set_ylabel(r"Observed $-\log_{10}(p)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwas1", required=True)
    ap.add_argument("--label1", required=True)
    ap.add_argument("--gwas2", required=True)
    ap.add_argument("--label2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--gap", type=int, default=1_000_000)
    ap.add_argument("--bonf-alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df1 = load_gwas(args.gwas1)
    df2 = load_gwas(args.gwas2)

    df1m, ticks1 = build_manhattan_coords(df1, gap=args.gap)
    df2m, ticks2 = build_manhattan_coords(df2, gap=args.gap)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.2], wspace=0.18, hspace=0.25)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    main_title = args.title.strip()
    if main_title:
        fig.suptitle(main_title, fontsize=14, y=0.99)

    plot_manhattan(ax11, df1m, ticks1, f"Manhattan: {args.label1}", bonf_alpha=args.bonf_alpha)
    plot_manhattan(ax12, df2m, ticks2, f"Manhattan: {args.label2}", bonf_alpha=args.bonf_alpha)

    plot_qq(ax21, df1m["p"].to_numpy(), f"QQ: {args.label1}")
    plot_qq(ax22, df2m["p"].to_numpy(), f"QQ: {args.label2}")

    ext = out_path.suffix.lower()
    if ext in {".png", ".pdf", ".svg"}:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
