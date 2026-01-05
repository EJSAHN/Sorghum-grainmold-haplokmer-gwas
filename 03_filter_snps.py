#!/usr/bin/env python3
"""
03_filter_snps.py

Filter the SNP matrix NPZ by:
- minor allele frequency (MAF) >= --maf
- missing_rate <= --max-missing

Usage:
  python scripts/03_filter_snps.py --in-npz 02_snp_matrix.npz --out 03_snp_matrix.filtered.npz --maf 0.05 --max-missing 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True, help="Input NPZ from step 02")
    ap.add_argument("--out", required=True, help="Output NPZ (filtered)")
    ap.add_argument("--maf", type=float, default=0.05, help="Min MAF (default 0.05)")
    ap.add_argument("--max-missing", type=float, default=0.2, help="Max missing rate (default 0.2)")
    args = ap.parse_args()

    in_path = Path(args.in_npz)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path, allow_pickle=True)
    G = data["G"]
    maf = data["maf"].astype(float)
    miss = data["missing_rate"].astype(float)

    keep = np.isfinite(maf) & np.isfinite(miss) & (maf >= args.maf) & (miss <= args.max_missing)

    n0 = G.shape[1]
    n1 = int(keep.sum())
    print(f"[INFO] Variants before: {n0:,}")
    print(f"[INFO] Variants after : {n1:,}")
    print(f"[INFO] Removed       : {n0-n1:,}")

    # Subset arrays
    Gf = G[:, keep]
    out = {
        "G": Gf,
        "samples": data["samples"],
        "rsid": data["rsid"][keep],
        "chrom": data["chrom"][keep],
        "pos": data["pos"][keep],
        "allele1": data["allele1"][keep],
        "allele2": data["allele2"][keep],
        "minor": data["minor"][keep],
        "major": data["major"][keep],
        "maf": data["maf"][keep],
        "missing_rate": data["missing_rate"][keep],
    }

    np.savez(out_path, **out)
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
