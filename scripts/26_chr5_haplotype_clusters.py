#!/usr/bin/env python3
"""
26_chr5_haplotype_clusters.py (FIXED: auto-detect genotype matrix orientation)

(A) Solid evidence upgrade:
- Extract SNP genotypes in the Chr5 core bin (default: 60.25-60.50 Mb).
- Build a haplotype-feature matrix across lines.
- Automatically cluster lines into haplotype groups (Agglomerative; silhouette selection).
- Boxplot phenotype by haplotype cluster + Kruskal-Wallis p-value.

Outputs:
- <out_prefix>_clusters.tsv
- <out_prefix>_stats.tsv
- <out_prefix>_boxplot.(png/pdf)

Example:
python scripts/26_chr5_haplotype_clusters.py --geno-npz 03_snp_matrix.filtered.npz --pheno-tsv 01_pheno_grain_mold.tsv --trait M --out_prefix 26_chr5_core_M
"""

from __future__ import annotations

import argparse
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering


# ------------------------
# NPZ loader (robust)
# ------------------------
def load_npz_genotypes(npz_path: str) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Load genotype matrix + marker meta + sample IDs from pipeline NPZ.
    Returns G as float ndarray, meta DataFrame(rsid,chrom,pos), samples list.

    NOTE: G can be either:
      - (m_markers x n_samples)   OR
      - (n_samples x m_markers)
    We'll fix orientation in main() once we know lengths.
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    # genotype matrix
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
            raise ValueError(f"Could not find genotype 2D array in {npz_path}. Keys={sorted(keys)}")
        G = best

    # samples
    samples = None
    for k in ["samples", "sample_ids", "taxa", "ids"]:
        if k in keys:
            samples = [str(x) for x in z[k].tolist()]
            break
    if samples is None:
        raise ValueError(f"Could not find samples list in {npz_path}. Keys={sorted(keys)}")

    # marker meta arrays
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
        raise ValueError(f"Could not reconstruct marker meta (rsid/chrom/pos) from {npz_path}. Keys={sorted(keys)}")

    meta = pd.DataFrame({"rsid": rsid, "chrom": chrom, "pos": pos})
    return G.astype(float), meta, samples


# ------------------------
# Haplotype feature matrix
# ------------------------
def maf_from_dosage(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    if v.size == 0:
        return 0.0
    p = np.mean(v) / 2.0
    p = min(max(p, 0.0), 1.0)
    return float(min(p, 1.0 - p))


def build_feature_matrix(G_region: np.ndarray, max_snps: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    G_region: (m_snps x n_samples)
    Returns:
      X: (n_samples x k) int features
      keep_idx: SNP indices kept within region
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
    X = X.astype(int).T
    return X, keep


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
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    G, meta, samples = load_npz_genotypes(args.geno_npz)
    meta["chrom"] = meta["chrom"].astype(str)

    n_samples = len(samples)
    n_markers = len(meta)

    # --- FIX ORIENTATION ---
    # Want G as (m_markers x n_samples)
    if G.shape == (n_samples, n_markers):
        G = G.T
    elif G.shape == (n_markers, n_samples):
        pass
    else:
        raise SystemExit(
            f"[ERROR] Unexpected genotype matrix shape {G.shape}. "
            f"Expected ({n_markers},{n_samples}) or ({n_samples},{n_markers})."
        )

    # region markers
    region_idx = meta.index[(meta["chrom"] == str(args.chrom)) & (meta["pos"] >= args.start) & (meta["pos"] <= args.end)].to_numpy()
    if region_idx.size < 5:
        raise SystemExit(f"[ERROR] Too few SNPs in region chr{args.chrom}:{args.start}-{args.end}. Found={region_idx.size}")

    G_region = G[region_idx, :]  # (m_region x n_samples)

    X, keep_local = build_feature_matrix(G_region, max_snps=args.max_snps)
    kept_region_idx = region_idx[keep_local]

    labels, k, sil = auto_cluster(X, args.k_min, args.k_max)

    # phenotype
    ph = pd.read_csv(args.pheno_tsv, sep="\t")
    if "sample_id" not in ph.columns:
        raise SystemExit("[ERROR] phenotype TSV must have 'sample_id' column.")
    if args.trait not in ph.columns:
        raise SystemExit(f"[ERROR] trait '{args.trait}' not found in {args.pheno_tsv}. Columns={ph.columns.tolist()}")

    ph = ph[["sample_id", args.trait]].copy()
    ph = ph.rename(columns={args.trait: "trait_value"})
    ph["sample_id"] = ph["sample_id"].astype(str)

    df = pd.DataFrame({"sample_id": samples, "cluster": labels})
    df = df.merge(ph, on="sample_id", how="left")

    # order clusters by median trait (low -> high)
    cluster_order = df.groupby("cluster")["trait_value"].median().sort_values().index.tolist()
    mapping = {c: i for i, c in enumerate(cluster_order)}
    df["cluster_ranked"] = df["cluster"].map(mapping).astype(int)

    # stats
    groups = [df.loc[df["cluster_ranked"] == c, "trait_value"].dropna().to_numpy(float) for c in sorted(df["cluster_ranked"].unique())]
    kw_p = float("nan")
    if sum(len(g) > 0 for g in groups) >= 2:
        kw_p = float(kruskal(*[g for g in groups if len(g) > 0]).pvalue)

    stats_rows = [{"test": "Kruskal-Wallis", "k": k, "silhouette": sil, "p_value": kw_p}]
    if len(cluster_order) == 2:
        g0, g1 = groups[0], groups[1]
        if len(g0) > 0 and len(g1) > 0:
            mw = mannwhitneyu(g0, g1, alternative="two-sided")
            stats_rows.append({"test": "Mann-Whitney U", "k": 2, "silhouette": sil, "p_value": float(mw.pvalue)})

    stats_df = pd.DataFrame(stats_rows)

    out_prefix = args.out_prefix
    df_out = df.sort_values(["cluster_ranked", "trait_value"], ascending=[True, True])
    df_out.to_csv(f"{out_prefix}_clusters.tsv", sep="\t", index=False)
    stats_df.to_csv(f"{out_prefix}_stats.tsv", sep="\t", index=False)

    # boxplot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    data = [df_out.loc[df_out["cluster_ranked"] == c, "trait_value"].dropna().to_numpy(float) for c in sorted(df_out["cluster_ranked"].unique())]
    ax.boxplot(data, labels=[f"H{c}" for c in sorted(df_out["cluster_ranked"].unique())], showfliers=False)
    ax.set_title(f"Chr{args.chrom}:{args.start:,}-{args.end:,} haplotype clusters vs phenotype\n{args.trait}  (k={len(data)}, KW p={kw_p:.3g})")
    ax.set_xlabel("Haplotype cluster (auto)")
    ax.set_ylabel(args.trait)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_boxplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_boxplot.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote: {out_prefix}_clusters.tsv")
    print(f"[OK] Wrote: {out_prefix}_stats.tsv")
    print(f"[OK] Wrote: {out_prefix}_boxplot.(png/pdf)")
    print(f"[INFO] Region SNPs total={region_idx.size:,}, used={len(kept_region_idx):,}, k_clusters={len(data)}, silhouette={sil:.3f}, KW p={kw_p:.3g}")


if __name__ == "__main__":
    main()
