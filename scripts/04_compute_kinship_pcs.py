#!/usr/bin/env python3
"""
04_compute_kinship_pcs.py

Compute a SNP-based kinship matrix (GRM) and principal components.

Input:
  - NPZ from step 03 (filtered SNP dosage matrix)

Output (single NPZ):
  - K: (n x n) kinship matrix (float64)
  - pcs: (n x n_pcs) top PCs from K (float64)
  - eigvals: eigenvalues of K (descending)
  - samples: sample ids

Usage:
  python scripts/04_compute_kinship_pcs.py --in-npz 03_snp_matrix.filtered.npz --out 04_covariates.npz --n-pcs 5

Notes:
- We build a standard genomic relationship matrix (GRM):
    Z_ij = (G_ij - 2p_j) / sqrt(2 p_j (1-p_j))
    K = (Z Z') / m
  where p_j is minor allele frequency.
- Missing genotypes are imputed as the mean dosage (2p_j).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True, help="Filtered SNP NPZ (step 03)")
    ap.add_argument("--out", required=True, help="Output NPZ")
    ap.add_argument("--n-pcs", type=int, default=5, help="Number of PCs to store (default 5)")
    args = ap.parse_args()

    in_path = Path(args.in_npz)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path, allow_pickle=True)
    G = data["G"].astype(np.float64)  # n x m
    samples = data["samples"]

    n, m = G.shape
    print(f"[INFO] n={n:,} samples, m={m:,} SNPs")

    # allele freq p from dosage (0/1/2): mean = 2p
    mean_dosage = np.nanmean(G, axis=0)  # length m
    p = mean_dosage / 2.0
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # impute missing as mean dosage
    G_imp = np.where(np.isnan(G), mean_dosage[None, :], G)

    denom = np.sqrt(2.0 * p * (1.0 - p))
    Z = (G_imp - mean_dosage[None, :]) / denom[None, :]

    # Kinship/GRM
    K = (Z @ Z.T) / m
    # Symmetrize
    K = (K + K.T) / 2.0

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(K)  # ascending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    n_pcs = min(args.n_pcs, n)
    pcs = eigvecs[:, :n_pcs]

    np.savez(out_path, K=K, pcs=pcs, eigvals=eigvals, samples=samples)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
