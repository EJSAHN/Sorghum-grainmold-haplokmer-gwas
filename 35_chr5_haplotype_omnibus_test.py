#!/usr/bin/env python3
"""
35_chr5_haplotype_omnibus_test.py

Perform an omnibus (global) haplotype effect test for the top interval using
an OLS/ANCOVA model with PCs as covariates.

Workflow:
- identify the k nearest SNPs around the lead position within the specified region
- encode multi-SNP haplotypes per accession
- keep haplotypes with count >= min_ac (optionally pool rare haplotypes)
- test reduced model (PCs only) vs full model (PCs + haplotype factor)
  using an F test based on RSS reduction

Outputs:
- <out_prefix>_summary.tsv/.xlsx
- <out_prefix>_haplotype_counts.tsv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

PI_RE = re.compile(r"^\s*(?:PI)?\s*0*([0-9]+)\s*$", re.IGNORECASE)


def normalize_sample_id(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "nan":
        return None
    m = PI_RE.match(s)
    if m:
        return f"PI{m.group(1)}"
    return re.sub(r"\s+", "", s)


def load_geno(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    G = z["G"].astype(float)  # n x m
    samples = np.array([normalize_sample_id(x) for x in z["samples"]], dtype=object)
    chrom = np.array([str(x) for x in z["chrom"]])
    pos = z["pos"].astype(int)
    rsid = np.array([str(x) for x in z["rsid"]], dtype=object)
    return G, samples, chrom, pos, rsid


def load_cov(npz_path: str, n_pcs: int = 5):
    z = np.load(npz_path, allow_pickle=True)
    pcs = z["pcs"].astype(float)
    if pcs.ndim == 1:
        pcs = pcs.reshape(-1, 1)
    pcs = pcs[:, : min(n_pcs, pcs.shape[1])]
    samples = np.array([normalize_sample_id(x) for x in z["samples"]], dtype=object)
    return pcs, samples


def encode_haps(Gsub: np.ndarray) -> np.ndarray:
    # Gsub: n x k dosage with NaN allowed
    X = np.rint(Gsub.copy()).astype(float)
    X[np.isnan(X)] = 3.0
    X = X.astype(int)
    return np.array(["-".join(map(str, row.tolist())) for row in X], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True)
    ap.add_argument("--pheno-tsv", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--covar-npz", required=True)
    ap.add_argument("--chrom", default="5")
    ap.add_argument("--start", type=int, default=60250000)
    ap.add_argument("--end", type=int, default=60499999)
    ap.add_argument("--lead-pos", type=int, default=60278659)
    ap.add_argument("--k-snps", type=int, default=7)
    ap.add_argument("--min-ac", type=int, default=10)
    ap.add_argument("--pool-rare", action="store_true")
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    G, gsamp, chrom, pos, rsid = load_geno(args.geno_npz)
    pcs, csamp = load_cov(args.covar_npz, args.n_pcs)
    cov_map = {s: i for i, s in enumerate(csamp.tolist()) if s is not None}
    geno_map = {s: i for i, s in enumerate(gsamp.tolist()) if s is not None}

    reg_idx = np.where((chrom.astype(str) == str(args.chrom)) & (pos >= args.start) & (pos <= args.end))[0]
    if reg_idx.size == 0:
        raise SystemExit("[ERROR] No SNPs in specified region")
    lead_idx = reg_idx[np.argmin(np.abs(pos[reg_idx] - args.lead_pos))]
    reg_pos = pos[reg_idx]
    order = np.argsort(np.abs(reg_pos - pos[lead_idx]))
    take = reg_idx[order[: min(args.k_snps, len(order))]]
    take = take[np.argsort(pos[take])]

    ph = pd.read_csv(args.pheno_tsv, sep="\t")
    ph["sample_id"] = ph["sample_id"].apply(normalize_sample_id)
    ph[args.trait] = pd.to_numeric(ph[args.trait], errors="coerce")
    ph = ph.dropna(subset=["sample_id", args.trait]).drop_duplicates(subset=["sample_id"], keep="first")

    ids = []
    y = []
    Xpcs = []
    Gk = []
    for _, r in ph.iterrows():
        sid = r["sample_id"]
        if sid in geno_map and sid in cov_map:
            gi = geno_map[sid]
            ci = cov_map[sid]
            ids.append(sid)
            y.append(float(r[args.trait]))
            Xpcs.append(pcs[ci, :])
            Gk.append(G[gi, take])

    y = np.asarray(y, float)
    Xpcs = np.asarray(Xpcs, float)
    Gk = np.asarray(Gk, float)

    haps = encode_haps(Gk)
    counts = pd.Series(haps).value_counts().reset_index()
    counts.columns = ["haplotype", "count"]
    counts["kept"] = counts["count"] >= args.min_ac

    if args.pool_rare:
        keep = set(counts.loc[counts["count"] >= args.min_ac, "haplotype"])
        haps2 = np.array([h if h in keep else "RARE_POOL" for h in haps], dtype=object)
    else:
        keep = set(counts.loc[counts["count"] >= args.min_ac, "haplotype"])
        m = np.array([h in keep for h in haps], dtype=bool)
        y = y[m]
        Xpcs = Xpcs[m, :]
        haps2 = haps[m]

    counts2 = pd.Series(haps2).value_counts().reset_index()
    counts2.columns = ["haplotype", "count"]
    counts2.to_csv(f"{args.out_prefix}_haplotype_counts.tsv", sep="\t", index=False)

    # design matrices
    H = pd.get_dummies(pd.Series(haps2, dtype="category"), drop_first=True)
    X0 = np.column_stack([np.ones(len(y)), Xpcs])
    X1 = np.column_stack([X0, H.values.astype(float)])

    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    rss0 = float(np.sum((y - X0 @ b0) ** 2))
    rss1 = float(np.sum((y - X1 @ b1) ** 2))

    df_num = X1.shape[1] - X0.shape[1]
    df_den = len(y) - X1.shape[1]
    F = ((rss0 - rss1) / df_num) / (rss1 / df_den) if (df_num > 0 and df_den > 0 and rss1 > 0) else np.nan
    p = float(1 - stats.f.cdf(F, df_num, df_den)) if np.isfinite(F) else np.nan
    partial_r2 = (rss0 - rss1) / rss0 if rss0 > 0 else np.nan

    out = pd.DataFrame([
        {
            "trait": args.trait,
            "chrom": args.chrom,
            "region_start": args.start,
            "region_end": args.end,
            "lead_pos_input": args.lead_pos,
            "lead_pos_used": int(pos[lead_idx]),
            "lead_rsid": str(rsid[lead_idx]),
            "k_snps": int(len(take)),
            "window_start": int(pos[take].min()),
            "window_end": int(pos[take].max()),
            "n_samples_tested": int(len(y)),
            "n_haplotypes_total": int(pd.Series(haps2).nunique()),
            "pool_rare": bool(args.pool_rare),
            "min_ac": int(args.min_ac),
            "omnibus_F": F,
            "omnibus_p": p,
            "omnibus_partial_R2": partial_r2,
        }
    ])
    out.to_csv(f"{args.out_prefix}_summary.tsv", sep="\t", index=False)
    out.to_excel(f"{args.out_prefix}_summary.xlsx", index=False)
    print(f"[OK] Wrote: {args.out_prefix}_summary.tsv")
    print(f"[OK] Wrote: {args.out_prefix}_summary.xlsx")
    print(f"[OK] Wrote: {args.out_prefix}_haplotype_counts.tsv")
    print(out.to_string(index=False))
    print("\n[Haplotype counts]")
    print(counts2.to_string(index=False))


if __name__ == "__main__":
    main()
