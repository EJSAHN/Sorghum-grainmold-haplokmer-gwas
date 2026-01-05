#!/usr/bin/env python3
"""
02_hapmap_to_numeric.py

Convert a TASSEL HapMap file (.hmp or tab-delimited .txt) into a compact numeric matrix.

Output is a single NPZ file with:
- G: (n_samples x n_sites) float32 matrix of MINOR-allele dosage (0/1/2), NaN for missing
- samples: sample ids (normalized)
- rsid, chrom, pos, allele1, allele2, major, minor, maf, missing_rate

Why "minor-allele dosage"?
- It makes MAF filtering and downstream GWAS consistent.
- It avoids needing a reference allele (not always available in HapMap exports).

Usage:
  python scripts/02_hapmap_to_numeric.py --hapmap SAP.filtered_hmp_unnecessary_accession_gone.hmp --out 02_snp_matrix.npz

Tip:
- If you have extremely large HapMap (hundreds of thousands of SNPs), this script is still OK
  for SAP-scale (~260k rows). The sample count is small, so memory is dominated by the G matrix.

"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


PI_RE = re.compile(r"^\s*(?:PI)?\s*0*([0-9]+)\s*$", re.IGNORECASE)

IUPAC_HET = {
    "R": ("A", "G"),
    "Y": ("C", "T"),
    "S": ("G", "C"),
    "W": ("A", "T"),
    "K": ("G", "T"),
    "M": ("A", "C"),
}


def normalize_sample_id(x: str) -> str:
    s = str(x).strip()
    m = PI_RE.match(s)
    if m:
        return f"PI{m.group(1)}"
    # remove whitespace for safety
    return re.sub(r"\s+", "", s)


def open_textmaybe_gz(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_alleles_field(alleles: str) -> Tuple[str, str]:
    a = alleles.strip().upper()
    if "/" in a:
        parts = a.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
    # Fallback: try split by any non-letter
    parts = re.split(r"[^ACGT]+", a)
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
    # Last resort: take first two chars
    if len(a) >= 2:
        return a[0], a[1]
    return a, a


def call_to_counts(call: str, allele1: str, allele2: str) -> Optional[Tuple[int, int]]:
    """
    Return (allele1_count, allele2_count) as 0/1/2, or None if missing/unknown.
    """
    c = call.strip().upper()
    if c in ("", "N", "NA", ".", "-", "NN"):
        return None
    # sometimes exported as "C/T"
    if "/" in c:
        parts = [p for p in c.split("/") if p]
        if len(parts) == 2 and set(parts) == set([allele1, allele2]):
            return (1, 1)
        # if something else (e.g., indel), treat as missing
        return None

    if c == allele1:
        return (2, 0)
    if c == allele2:
        return (0, 2)

    if c in IUPAC_HET:
        a, b = IUPAC_HET[c]
        if set([a, b]) == set([allele1, allele2]):
            return (1, 1)

    # Unknown encoding -> missing
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hapmap", required=True, help="Input HapMap (.hmp / .txt / .gz)")
    ap.add_argument("--out", required=True, help="Output NPZ")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed (slower, smaller)")
    args = ap.parse_args()

    in_path = Path(args.hapmap)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read header
    with open_textmaybe_gz(in_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        if len(header) < 12:
            raise ValueError("HapMap header looks too short. Is it tab-delimited TASSEL HapMap?")
        # locate QCcode column to determine sample start
        qc_idx = None
        for i, h in enumerate(header):
            if h.strip().lower() == "qccode":
                qc_idx = i
                break
        sample_start = qc_idx + 1 if qc_idx is not None else 11
        sample_names_raw = header[sample_start:]
        if len(sample_names_raw) == 0:
            raise ValueError("No sample columns detected. Check the HapMap format.")
        samples = np.array([normalize_sample_id(s) for s in sample_names_raw], dtype="U50")

    # First pass: count variants
    n_variants = 0
    with open_textmaybe_gz(in_path) as f:
        _ = f.readline()  # skip header
        for line in f:
            if line.strip() == "":
                continue
            n_variants += 1

    n_samples = len(samples)
    print(f"[INFO] Samples: {n_samples:,}  Variants: {n_variants:,}")

    # Allocate arrays
    G = np.full((n_samples, n_variants), np.nan, dtype=np.float32)
    rsid = np.empty(n_variants, dtype="U80")
    chrom = np.empty(n_variants, dtype="U40")
    pos = np.empty(n_variants, dtype=np.int64)
    allele1_arr = np.empty(n_variants, dtype="U10")
    allele2_arr = np.empty(n_variants, dtype="U10")
    minor_arr = np.empty(n_variants, dtype="U10")
    major_arr = np.empty(n_variants, dtype="U10")
    maf_arr = np.full(n_variants, np.nan, dtype=np.float32)
    miss_arr = np.full(n_variants, np.nan, dtype=np.float32)

    # Second pass: fill
    v = 0
    with open_textmaybe_gz(in_path) as f:
        _ = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                continue
            fields = line.split("\t")
            if len(fields) < sample_start + n_samples:
                raise ValueError(f"Line has too few columns at variant index {v}: {len(fields)} columns")

            rsid[v] = fields[0]
            a1, a2 = parse_alleles_field(fields[1])
            allele1_arr[v] = a1
            allele2_arr[v] = a2
            chrom[v] = fields[2]
            try:
                pos[v] = int(fields[3])
            except ValueError:
                pos[v] = -1

            # parse calls
            a1_counts = np.full(n_samples, -1, dtype=np.int8)
            a2_counts = np.full(n_samples, -1, dtype=np.int8)
            calls = fields[sample_start : sample_start + n_samples]

            for i, c in enumerate(calls):
                cc = call_to_counts(c, a1, a2)
                if cc is None:
                    continue
                a1_counts[i] = cc[0]
                a2_counts[i] = cc[1]

            nonmiss = (a1_counts >= 0)
            n_nonmiss = int(nonmiss.sum())
            n_miss = n_samples - n_nonmiss
            miss_rate = n_miss / n_samples if n_samples > 0 else math.nan
            miss_arr[v] = miss_rate

            if n_nonmiss == 0:
                maf_arr[v] = math.nan
                minor_arr[v] = a2
                major_arr[v] = a1
                v += 1
                continue

            total_a1 = int(a1_counts[nonmiss].sum())
            total_a2 = int(a2_counts[nonmiss].sum())

            # determine minor allele
            if total_a1 < total_a2:
                minor = a1
                major = a2
                dosage = a1_counts.astype(np.float32)
            elif total_a2 < total_a1:
                minor = a2
                major = a1
                dosage = a2_counts.astype(np.float32)
            else:
                # tie -> pick allele2 as minor for deterministic behavior
                minor = a2
                major = a1
                dosage = a2_counts.astype(np.float32)

            dosage[~nonmiss] = np.nan
            G[:, v] = dosage

            minor_arr[v] = minor
            major_arr[v] = major

            minor_total = np.nansum(dosage)
            maf = float(minor_total / (2.0 * n_nonmiss)) if n_nonmiss > 0 else math.nan
            maf_arr[v] = maf

            v += 1
            if v % 10000 == 0:
                print(f"[INFO] Parsed {v:,}/{n_variants:,} variants...")

    assert v == n_variants

    # Save
    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(
        out_path,
        G=G,
        samples=samples,
        rsid=rsid,
        chrom=chrom,
        pos=pos,
        allele1=allele1_arr,
        allele2=allele2_arr,
        minor=minor_arr,
        major=major_arr,
        maf=maf_arr,
        missing_rate=miss_arr,
    )
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()
