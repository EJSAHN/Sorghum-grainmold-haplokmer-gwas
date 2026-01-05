#!/usr/bin/env python3
"""
27_chr5_ld_lead_vs_haplotype.py (FIXED: auto-detect genotype matrix orientation)

(B) Solid evidence upgrade:
- Compute LD (r^2) between:
  1) Lead SNP (default pos=60278659 on chr5)
  2) Representative haplotype in the core region:
     - take k nearest SNPs around the lead SNP (default k=7)
     - enumerate haplotypes across samples
     - choose haplotype with maximum r^2 vs lead SNP (binary presence)

Also produces an LD heatmap (r^2) among SNPs in the region.

Outputs:
- <out_prefix>_lead_ld.tsv
- <out_prefix>_ld_heatmap.(png/pdf)

Example:
python scripts/27_chr5_ld_lead_vs_haplotype.py --geno-npz 03_snp_matrix.filtered.npz --chrom 5 --start 60250000 --end 60499999 --lead_pos 60278659 --k 7 --max_snps 120 --out_prefix 27_chr5_ld
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_npz_genotypes(npz_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    G = None
    for k in ["G", "geno", "X", "genotypes"]:
        if k in keys:
            G = z[k]
            break
    if G is None:
        best = None
        best_size = -1
        for k in keys:
            arr = z[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                if arr.size > best_size:
                    best_size = arr.size
                    best = arr
        if best is None:
            raise ValueError(f"Could not find genotype matrix in {npz_path}. Keys={sorted(keys)}")
        G = best

    samples = None
    for k in ["samples", "sample_ids", "taxa", "ids"]:
        if k in keys:
            samples = [str(x) for x in z[k].tolist()]
            break
    if samples is None:
        raise ValueError(f"Could not find samples list in {npz_path}. Keys={sorted(keys)}")

    rsid = chrom = pos = None
    for k in ["rsid", "marker_id", "markers", "ids_marker"]:
        if k in keys:
            rsid = [str(x) for x in z[k].tolist()]
            break
    for k in ["chrom", "chr", "chroms"]:
        if k in keys:
            chrom = [str(x) for x in z[k].tolist()]
            break
    for k in ["pos", "position", "bp"]:
        if k in keys:
            pos = [int(x) for x in z[k].tolist()]
            break
    if rsid is None or chrom is None or pos is None:
        raise ValueError(f"Could not reconstruct rsid/chrom/pos from {npz_path}. Keys={sorted(keys)}")

    meta = pd.DataFrame({"rsid": rsid, "chrom": chrom, "pos": pos})
    return G.astype(float), meta, samples


def r2(x: np.ndarray, y: np.ndarray) -> float:
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 5:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) == 0 or np.std(yv) == 0:
        return float("nan")
    r = np.corrcoef(xv, yv)[0, 1]
    return float(r * r)


def encode_haplotypes(Gk: np.ndarray) -> List[str]:
    k, n = Gk.shape
    H = []
    for j in range(n):
        vals = []
        for i in range(k):
            v = Gk[i, j]
            if np.isnan(v):
                vals.append("N")
            else:
                vals.append(str(int(round(v))))
        H.append("-".join(vals))
    return H


def ld_heatmap(r2mat: np.ndarray, out_prefix: str, title: str):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(r2mat, interpolation="nearest", aspect="auto", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("r^2")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_ld_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_ld_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True)
    ap.add_argument("--chrom", default="5")
    ap.add_argument("--start", type=int, default=60250000)
    ap.add_argument("--end", type=int, default=60499999)
    ap.add_argument("--lead_pos", type=int, default=60278659)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--max_snps", type=int, default=120)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    G, meta, samples = load_npz_genotypes(args.geno_npz)
    meta["chrom"] = meta["chrom"].astype(str)

    n_samples = len(samples)
    n_markers = len(meta)

    # FIX ORIENTATION: want (m_markers x n_samples)
    if G.shape == (n_samples, n_markers):
        G = G.T
    elif G.shape == (n_markers, n_samples):
        pass
    else:
        raise SystemExit(
            f"[ERROR] Unexpected genotype matrix shape {G.shape}. "
            f"Expected ({n_markers},{n_samples}) or ({n_samples},{n_markers})."
        )

    # region SNPs
    region = meta[(meta["chrom"] == str(args.chrom)) & (meta["pos"] >= args.start) & (meta["pos"] <= args.end)].copy()
    if region.empty:
        raise SystemExit(f"[ERROR] No SNPs in region chr{args.chrom}:{args.start}-{args.end}")

    region = region.reset_index(drop=False).rename(columns={"index": "idx"}).sort_values("pos").reset_index(drop=True)

    # lead SNP closest by position
    lead_i = int((region["pos"] - args.lead_pos).abs().idxmin())
    lead_idx = int(region.loc[lead_i, "idx"])
    lead_pos_used = int(region.loc[lead_i, "pos"])
    lead_rsid = str(region.loc[lead_i, "rsid"])

    x_lead = G[lead_idx, :].astype(float)

    # k-SNP window around lead (nearest)
    half = args.k // 2
    left = max(0, lead_i - half)
    right = min(len(region) - 1, left + args.k - 1)
    left = max(0, right - args.k + 1)

    win = region.iloc[left : right + 1].copy()
    win_idx = win["idx"].to_numpy(dtype=int)

    Gk = G[win_idx, :]  # (k x n)
    hap_strings = encode_haplotypes(Gk)

    # compute r2 between lead SNP and each haplotype presence
    hap_counts = pd.Series(hap_strings).value_counts()
    candidates = [h for h, c in hap_counts.items() if c >= 10]

    best_h = ""
    best_r2 = float("nan")
    best_n = 0

    for h in candidates:
        y = np.array([1.0 if s == h else 0.0 for s in hap_strings], dtype=float)
        rr2 = r2(x_lead, y)
        if np.isnan(rr2):
            continue
        if np.isnan(best_r2) or rr2 > best_r2:
            best_r2 = rr2
            best_h = h
            best_n = int(hap_counts[h])

    # LD heatmap among SNPs in region (subsample if needed)
    m = len(region)
    if m > args.max_snps:
        take = np.linspace(0, m - 1, args.max_snps).round().astype(int)
        region_sub = region.iloc[take].copy()
    else:
        region_sub = region.copy()

    idx_sub = region_sub["idx"].to_numpy(dtype=int)
    Gsub = G[idx_sub, :]  # (m_sub x n)

    msub = Gsub.shape[0]
    r2mat = np.zeros((msub, msub), dtype=float)
    for i in range(msub):
        r2mat[i, i] = 1.0
        for j in range(i + 1, msub):
            rr2 = r2(Gsub[i, :], Gsub[j, :])
            if np.isnan(rr2):
                rr2 = 0.0
            r2mat[i, j] = rr2
            r2mat[j, i] = rr2

    # output summary TSV
    out = pd.DataFrame([{
        "chrom": str(args.chrom),
        "region_start": int(args.start),
        "region_end": int(args.end),
        "lead_pos_input": int(args.lead_pos),
        "lead_pos_used": int(lead_pos_used),
        "lead_rsid": lead_rsid,
        "hap_window_start": int(win["pos"].min()),
        "hap_window_end": int(win["pos"].max()),
        "k_snps": int(args.k),
        "best_haplotype": best_h,
        "best_haplotype_n": int(best_n),
        "r2_lead_vs_best_haplotype": float(best_r2) if not np.isnan(best_r2) else np.nan,
        "n_haplotype_candidates_ge10": int(len(candidates)),
        "n_region_snps": int(len(region)),
        "n_ld_heatmap_snps": int(len(region_sub)),
    }])

    out_path = f"{args.out_prefix}_lead_ld.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote: {out_path}")
    print(out.to_string(index=False))

    # plot heatmap
    title = f"LD (r^2) heatmap: chr{args.chrom}:{args.start:,}-{args.end:,} (n={len(region_sub)})"
    ld_heatmap(r2mat, args.out_prefix, title)
    print(f"[OK] Wrote: {args.out_prefix}_ld_heatmap.(png/pdf)")


if __name__ == "__main__":
    main()
