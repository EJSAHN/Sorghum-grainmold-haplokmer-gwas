#!/usr/bin/env python3
"""
31_placebo_windows_genomewide.py

Placebo (genome-wide random window) test for Chr5 core haplotype signal.

Goal
----
Show that the Chr5 core window's haplotype-cluster -> phenotype separation is
more extreme than random genomic windows of the same size.

Method
------
For each window:
1) Extract SNP genotypes from the window (from 03_snp_matrix.filtered.npz).
2) Build haplotype feature matrix (dosage 0/1/2; missing=3), using up to max_snps by MAF.
3) Auto cluster (Agglomerative Ward), choose k by silhouette in [k_min,k_max].
4) Compute Kruskal–Wallis p-value for phenotype across clusters.

We compute:
- p_chr5: KW p-value for the Chr5 core window
- placebo distribution: KW p-values for N random windows
- empirical_p = fraction(placebo_p <= p_chr5)

Outputs
-------
- <out_prefix>_placebo_windows.tsv
- <out_prefix>_summary.tsv
- <out_prefix>_hist_kw_p.(png/pdf)

Example
-------
python scripts/31_placebo_windows_genomewide.py ^
  --geno-npz 03_snp_matrix.filtered.npz ^
  --pheno-tsv 01_pheno_grain_mold.tsv --trait M ^
  --window_bp 250000 --N 50 --seed 1 ^
  --out_prefix 31_placebo_chr5_M

Notes
-----
- Uses PNG(300dpi) + PDF outputs.
- Avoids seaborn.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ------------------------
# NPZ loader (same idea as 26/27)
# ------------------------
def load_npz_genotypes(npz_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    # genotype matrix
    G = None
    for k in ["G", "geno", "X", "genotypes"]:
        if k in keys:
            G = z[k]
            break
    if G is None:
        # pick largest 2D
        best = None
        best_size = -1
        for k in keys:
            arr = z[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > best_size:
                best_size = arr.size
                best = arr
        if best is None:
            raise ValueError(f"Could not find genotype matrix in {npz_path}. Keys={sorted(keys)}")
        G = best

    # samples
    samples = None
    for k in ["samples", "sample_ids", "taxa", "ids"]:
        if k in keys:
            samples = [str(x) for x in z[k].tolist()]
            break
    if samples is None:
        raise ValueError(f"Could not find samples list in {npz_path}. Keys={sorted(keys)}")

    # meta arrays
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
        raise ValueError(f"Could not reconstruct marker meta from {npz_path}. Keys={sorted(keys)}")

    meta = pd.DataFrame({"rsid": rsid, "chrom": chrom, "pos": pos})
    return G.astype(float), meta, samples


# ------------------------
# Features / clustering
# ------------------------
def maf_from_dosage(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    if v.size == 0:
        return 0.0
    p = np.mean(v) / 2.0
    p = min(max(p, 0.0), 1.0)
    return float(min(p, 1.0 - p))


def build_feature_matrix(G_region: np.ndarray, max_snps: int) -> np.ndarray:
    """
    G_region: (m_snps x n_samples)
    returns X: (n_samples x k_features) int, missing encoded as 3
    """
    m, n = G_region.shape
    mafs = np.array([maf_from_dosage(G_region[i, :]) for i in range(m)], dtype=float)

    cand = np.where(mafs > 0.01)[0]
    if cand.size == 0:
        cand = np.arange(m)

    order = cand[np.argsort(-mafs[cand])]
    keep = order[: min(max_snps, order.size)]

    Gk = G_region[keep, :].copy()
    X = np.rint(Gk).astype(np.float64)
    X[np.isnan(X)] = 3.0
    return X.astype(int).T  # (n x k)


def auto_cluster(X: np.ndarray, k_min: int, k_max: int) -> Tuple[np.ndarray, int, float]:
    best_k = None
    best_score = -1.0
    best_labels = None

    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="euclidean")
        if score > best_score:
            best_score = float(score)
            best_k = k
            best_labels = labels

    if best_labels is None:
        model = AgglomerativeClustering(n_clusters=2, linkage="ward")
        best_labels = model.fit_predict(X)
        best_k = 2
        best_score = float("nan")

    return best_labels, int(best_k), float(best_score)


def kw_pvalue(trait_vec: np.ndarray, labels: np.ndarray) -> float:
    groups = []
    for c in sorted(set(labels.tolist())):
        g = trait_vec[labels == c]
        g = g[~np.isnan(g)]
        if g.size > 0:
            groups.append(g.astype(float))
    if len(groups) < 2:
        return float("nan")
    return float(kruskal(*groups).pvalue)


# ------------------------
# Window runner
# ------------------------
def run_window(
    G: np.ndarray,
    meta: pd.DataFrame,
    samples: List[str],
    trait_vec: np.ndarray,
    chrom: str,
    start: int,
    end: int,
    max_snps: int,
    k_min: int,
    k_max: int,
) -> Dict[str, object]:
    idx = meta.index[(meta["chrom"] == chrom) & (meta["pos"] >= start) & (meta["pos"] <= end)].to_numpy()
    if idx.size < 5:
        return {
            "chrom": chrom, "start": start, "end": end,
            "n_snps_total": int(idx.size),
            "k": 0, "silhouette": float("nan"), "kw_p": float("nan"),
            "status": "too_few_snps",
        }

    G_region = G[idx, :]  # (m x n)
    X = build_feature_matrix(G_region, max_snps=max_snps)
    labels, k, sil = auto_cluster(X, k_min=k_min, k_max=k_max)
    p = kw_pvalue(trait_vec, labels)

    return {
        "chrom": chrom, "start": start, "end": end,
        "n_snps_total": int(idx.size),
        "n_snps_used": int(X.shape[1]),
        "k": int(k),
        "silhouette": float(sil),
        "kw_p": float(p),
        "status": "ok",
    }


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True)
    ap.add_argument("--pheno-tsv", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--window_bp", type=int, default=250_000)
    ap.add_argument("--N", type=int, default=50, help="Number of placebo windows")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_snps", type=int, default=80, help="Max SNPs used for feature matrix")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--out_prefix", required=True)

    # Chr5 core (fixed per your result)
    ap.add_argument("--chr5_start", type=int, default=60250000)
    ap.add_argument("--chr5_end", type=int, default=60499999)
    ap.add_argument("--exclude_chr5_core", action="store_true", help="Exclude windows overlapping chr5 core from placebo sampling")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load geno
    G, meta, samples = load_npz_genotypes(args.geno_npz)
    meta["chrom"] = meta["chrom"].astype(str)
    n_samples = len(samples)
    n_markers = len(meta)

    # fix orientation -> (markers x samples)
    if G.shape == (n_samples, n_markers):
        G = G.T
    elif G.shape == (n_markers, n_samples):
        pass
    else:
        raise SystemExit(f"[ERROR] Unexpected G shape {G.shape} for samples={n_samples}, markers={n_markers}")

    # Load phenotype + align
    ph = pd.read_csv(args.pheno_tsv, sep="\t")
    if "sample_id" not in ph.columns or args.trait not in ph.columns:
        raise SystemExit("[ERROR] phenotype TSV must have sample_id and the trait column.")
    ph = ph[["sample_id", args.trait]].copy()
    ph.columns = ["sample_id", "trait_value"]
    ph["sample_id"] = ph["sample_id"].astype(str)
    ph["trait_value"] = pd.to_numeric(ph["trait_value"], errors="coerce")

    mp = dict(zip(ph["sample_id"].tolist(), ph["trait_value"].tolist()))
    trait_vec = np.array([mp.get(s, np.nan) for s in samples], dtype=float)

    # Universe chromosomes
    chroms = sorted(meta["chrom"].unique().tolist(), key=lambda x: (len(x), x))
    # per-chrom min/max
    chrom_ranges = {}
    for c in chroms:
        sub = meta.loc[meta["chrom"] == c, "pos"]
        if sub.empty:
            continue
        mn = int(sub.min())
        mx = int(sub.max())
        if mx - mn >= args.window_bp:
            chrom_ranges[c] = (mn, mx)

    if not chrom_ranges:
        raise SystemExit("[ERROR] No chromosomes have enough span for the chosen window.")

    # Compute p for chr5 core
    chr5_res = run_window(
        G, meta, samples, trait_vec,
        chrom="5",
        start=args.chr5_start,
        end=args.chr5_end,
        max_snps=args.max_snps,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    p_chr5 = chr5_res["kw_p"]

    print(f"[INFO] Chr5 core KW p = {p_chr5:.4g}  (k={chr5_res.get('k')}, n_snps_total={chr5_res.get('n_snps_total')})")

    # Sample placebo windows
    rows = []
    rows.append({"type": "chr5_core", **chr5_res})

    def overlaps_chr5_core(c: str, s: int, e: int) -> bool:
        if c != "5":
            return False
        return not (e < args.chr5_start or s > args.chr5_end)

    attempts = 0
    collected = 0
    max_attempts = args.N * 50

    chrom_list = list(chrom_ranges.keys())
    while collected < args.N and attempts < max_attempts:
        attempts += 1
        c = rng.choice(chrom_list)
        mn, mx = chrom_ranges[c]
        start = int(rng.integers(mn, mx - args.window_bp + 1))
        end = start + args.window_bp - 1

        if args.exclude_chr5_core and overlaps_chr5_core(c, start, end):
            continue

        res = run_window(
            G, meta, samples, trait_vec,
            chrom=str(c),
            start=start,
            end=end,
            max_snps=args.max_snps,
            k_min=args.k_min,
            k_max=args.k_max,
        )
        if res["status"] != "ok":
            continue

        rows.append({"type": "placebo", **res})
        collected += 1

        if collected % 10 == 0:
            print(f"  ... placebo {collected}/{args.N}")

    if collected < args.N:
        print(f"[WARN] Only collected {collected}/{args.N} placebo windows (attempts={attempts}). Consider lowering max_snps/k_max or increasing max_attempts.")

    out = pd.DataFrame(rows)
    out_path = f"{args.out_prefix}_placebo_windows.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote: {out_path}")

    # empirical p
    placebo = out[(out["type"] == "placebo") & (out["status"] == "ok")].copy()
    placebo_p = placebo["kw_p"].to_numpy(float)
    placebo_p = placebo_p[~np.isnan(placebo_p)]

    emp_p = float(np.mean(placebo_p <= float(p_chr5))) if placebo_p.size else float("nan")

    summary = pd.DataFrame([{
        "trait": args.trait,
        "window_bp": args.window_bp,
        "N_placebo": int(placebo_p.size),
        "chr5_kw_p": float(p_chr5),
        "empirical_p_placebo_le_chr5": emp_p,
        "seed": int(args.seed),
        "max_snps": int(args.max_snps),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
    }])
    sum_path = f"{args.out_prefix}_summary.tsv"
    summary.to_csv(sum_path, sep="\t", index=False)
    print(f"[OK] Wrote: {sum_path}")
    print(summary.to_string(index=False))

    # histogram
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.hist(placebo_p, bins=30)
    ax.axvline(float(p_chr5), linestyle="--", linewidth=2)
    ax.set_title(f"Placebo window test (trait={args.trait})\nempirical p = {emp_p:.3g}  (N={int(placebo_p.size)})")
    ax.set_xlabel("Kruskal–Wallis p (haplotype clusters)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_hist_kw_p.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_hist_kw_p.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote figures: {args.out_prefix}_hist_kw_p.(png/pdf)")


if __name__ == "__main__":
    main()
