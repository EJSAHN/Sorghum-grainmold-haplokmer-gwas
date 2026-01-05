#!/usr/bin/env python3
"""
30_cluster_bootstrap_stability.py

Bootstrap stability test for Chr5 core haplotype clustering -> phenotype separation.

For each bootstrap replicate:
- sample a fraction of lines without replacement
- cluster genotypes within chr5 core (same logic as script 26)
- compute Kruskal-Wallis p-value for trait across clusters
- record k chosen and silhouette score

Outputs:
- <out_prefix>_bootstrap.tsv
- <out_prefix>_p_hist.(png/pdf)
- <out_prefix>_k_hist.(png/pdf)

Example:
python scripts/30_cluster_bootstrap_stability.py --geno-npz 03_snp_matrix.filtered.npz --pheno-tsv 01_pheno_grain_mold.tsv --trait M --out_prefix 30_boot_M --B 200 --frac 0.8
"""

from __future__ import annotations

import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


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
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > best_size:
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


def maf_from_dosage(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    if v.size == 0:
        return 0.0
    p = np.mean(v) / 2.0
    p = min(max(p, 0.0), 1.0)
    return float(min(p, 1.0 - p))


def build_feature_matrix(G_region: np.ndarray, max_snps: int = 200) -> np.ndarray:
    """
    G_region: (m_snps x n_samples_sub)
    output X: (n_samples_sub x k) int, missing encoded as 3
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
    return X.astype(int).T


def auto_cluster(X: np.ndarray, k_min: int = 2, k_max: int = 6) -> Tuple[np.ndarray, int, float]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True)
    ap.add_argument("--pheno-tsv", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--chrom", default="5")
    ap.add_argument("--start", type=int, default=60250000)
    ap.add_argument("--end", type=int, default=60499999)
    ap.add_argument("--max_snps", type=int, default=200)
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=6)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

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

    # region marker indices
    region_idx = meta.index[(meta["chrom"] == str(args.chrom)) & (meta["pos"] >= args.start) & (meta["pos"] <= args.end)].to_numpy()
    if region_idx.size < 5:
        raise SystemExit(f"[ERROR] Too few SNPs in region chr{args.chrom}:{args.start}-{args.end}. Found={region_idx.size}")

    # phenotype
    ph = pd.read_csv(args.pheno_tsv, sep="\t")
    if "sample_id" not in ph.columns or args.trait not in ph.columns:
        raise SystemExit("[ERROR] phenotype TSV must have sample_id and the trait column.")
    ph = ph[["sample_id", args.trait]].copy()
    ph.columns = ["sample_id", "trait_value"]
    ph["sample_id"] = ph["sample_id"].astype(str)

    # align ph to genotype samples
    sample_to_trait = dict(zip(ph["sample_id"].tolist(), pd.to_numeric(ph["trait_value"], errors="coerce").tolist()))
    trait_vec = np.array([sample_to_trait.get(s, np.nan) for s in samples], dtype=float)

    # bootstrap
    B = int(args.B)
    n_sub = max(10, int(round(args.frac * n_samples)))

    rows = []
    for b in range(1, B + 1):
        idx = rng.choice(n_samples, size=n_sub, replace=False)
        idx = np.sort(idx)

        G_region = G[region_idx, :][:, idx]  # (m_region x n_sub)
        X = build_feature_matrix(G_region, max_snps=args.max_snps)
        labels, k, sil = auto_cluster(X, args.k_min, args.k_max)

        y = trait_vec[idx]
        # groups for KW
        groups = []
        for c in sorted(set(labels.tolist())):
            g = y[labels == c]
            g = g[~np.isnan(g)]
            if g.size > 0:
                groups.append(g.astype(float))

        kw_p = float("nan")
        if len(groups) >= 2:
            kw_p = float(kruskal(*groups).pvalue)

        rows.append({
            "iter": b,
            "n_sub": int(n_sub),
            "k": int(k),
            "silhouette": float(sil),
            "kw_p": kw_p,
            "kw_sig_0p05": int((not np.isnan(kw_p)) and (kw_p < 0.05)),
        })

        if b % 20 == 0:
            print(f"  ... {b}/{B}")

    out = pd.DataFrame(rows)
    out_path = f"{args.out_prefix}_bootstrap.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote: {out_path}")

    # plots
    # p histogram
    pvals = out["kw_p"].to_numpy(float)
    pvals = pvals[~np.isnan(pvals)]

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.hist(pvals, bins=30)
    ax.set_title(f"Bootstrap KW p-values (trait={args.trait})\nB={B}, frac={args.frac}")
    ax.set_xlabel("Kruskal–Wallis p")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_p_hist.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_p_hist.pdf", bbox_inches="tight")
    plt.close(fig)

    # k histogram
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.hist(out["k"].to_numpy(int), bins=range(out["k"].min(), out["k"].max() + 2))
    ax.set_title("Bootstrap selected k (Agglomerative silhouette)")
    ax.set_xlabel("k")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_k_hist.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{args.out_prefix}_k_hist.pdf", bbox_inches="tight")
    plt.close(fig)

    sig_rate = float(out["kw_sig_0p05"].mean())
    print(f"[INFO] Significant fraction (p<0.05): {sig_rate:.3f}")


if __name__ == "__main__":
    main()
