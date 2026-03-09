#!/usr/bin/env python3
"""
32_check_snp_count.py

Print genotype matrix shape and infer:
- n_samples
- n_markers (SNPs)
from 03_snp_matrix.filtered.npz
"""

import numpy as np


def main():
    z = np.load("03_snp_matrix.filtered.npz", allow_pickle=True)
    print("keys =", z.files)

    G = None
    for k in ["G", "geno", "X", "genotypes"]:
        if k in z.files:
            G = z[k]
            print(f"using matrix key = {k}")
            break

    if G is None:
        # fallback: largest 2D array
        mats = []
        for k in z.files:
            arr = z[k]
            if hasattr(arr, "ndim") and arr.ndim == 2:
                mats.append((k, arr))
        if not mats:
            raise SystemExit("[ERROR] No 2D matrices found in NPZ.")
        k, G = max(mats, key=lambda kv: kv[1].size)
        print(f"using fallback largest 2D key = {k}")

    print("G shape =", G.shape)

    # Heuristic: one dimension is n_samples (=306), other is n_markers (=61011)
    dims = sorted(G.shape)
    if 306 in G.shape:
        n_samples = 306
        n_markers = dims[1] if dims[0] == 306 else dims[0]
        print("n_samples =", n_samples)
        print("n_markers (SNPs) =", n_markers)
    else:
        print("[WARN] 306 not found in G shape; please verify sample count.")
        print("interpreting: n_samples =", dims[0], "n_markers =", dims[1])


if __name__ == "__main__":
    main()
