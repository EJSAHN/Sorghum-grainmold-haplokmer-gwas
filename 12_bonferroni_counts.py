#!/usr/bin/env python3
"""
12_bonferroni_counts.py

Count Bonferroni-significant hits for all GWAS result files.

Outputs ONE TSV:
  - file
  - m (number of markers)
  - bonf_thr
  - bonf_hits
  - min_p

Default inputs:
  05_gwas_snp_*.tsv.gz
  06_gwas_haplokmer_*.tsv.gz

Usage:
  python scripts/12_bonferroni_counts.py --out 12_bonferroni_counts.tsv
  python scripts/12_bonferroni_counts.py --pattern "05_gwas_snp_*.tsv.gz" --pattern "06_gwas_haplokmer_*.tsv.gz" --out 12_bonferroni_counts.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        action="append",
        default=["05_gwas_snp_*.tsv.gz", "06_gwas_haplokmer_*.tsv.gz"],
        help="Glob pattern(s) for GWAS files. Can be repeated.",
    )
    ap.add_argument("--out", required=True, help="Output TSV filename")
    args = ap.parse_args()

    files = []
    for pat in args.pattern:
        files.extend(sorted([str(p) for p in Path(".").glob(pat)]))
    files = sorted(set(files))

    if not files:
        raise SystemExit("[ERROR] No GWAS files matched patterns in the current directory.")

    rows = []
    for f in files:
        df = pd.read_csv(f, sep="\t", compression="infer")
        if "p" not in df.columns:
            raise SystemExit(f"[ERROR] Missing 'p' column in: {f}")

        m = int(len(df))
        thr = 0.05 / m if m > 0 else float("nan")
        bonf_hits = int((df["p"] <= thr).sum()) if m > 0 else 0
        min_p = float(df["p"].min()) if m > 0 else float("nan")

        rows.append(
            {
                "file": Path(f).name,
                "m": m,
                "bonf_thr": thr,
                "bonf_hits": bonf_hits,
                "min_p": min_p,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["bonf_hits", "min_p"], ascending=[False, True])
    out_path = Path(args.out)
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"[OK] Wrote: {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
