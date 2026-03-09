#!/usr/bin/env python3
"""
36_chr5_k_sensitivity.py

Optional local sensitivity analysis for the Chr5 region across different k values.
For each k in --k-list, the script computes:
- omnibus haplotype effect test (using PCs as covariates)
- best haplotype LD (r^2) vs lead SNP
- number of haplotypes retained at min_ac

Outputs:
- <out_prefix>_summary.tsv
- <out_prefix>_summary.xlsx
"""
from __future__ import annotations

import argparse
import re
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
    return (
        z["G"].astype(float),
        np.array([normalize_sample_id(x) for x in z["samples"]], dtype=object),
        np.array([str(x) for x in z["chrom"]]),
        z["pos"].astype(int),
        np.array([str(x) for x in z["rsid"]], dtype=object),
    )


def load_cov(npz_path: str, n_pcs: int = 5):
    z = np.load(npz_path, allow_pickle=True)
    pcs = z["pcs"].astype(float)
    if pcs.ndim == 1:
        pcs = pcs.reshape(-1, 1)
    pcs = pcs[:, : min(n_pcs, pcs.shape[1])]
    samples = np.array([normalize_sample_id(x) for x in z["samples"]], dtype=object)
    return pcs, samples


def r2(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    xx = x[m].astype(float)
    yy = y[m].astype(float)
    if np.std(xx) == 0 or np.std(yy) == 0:
        return np.nan
    rr = np.corrcoef(xx, yy)[0, 1]
    return float(rr * rr)


def encode_haps(Gsub: np.ndarray) -> np.ndarray:
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
    ap.add_argument("--k-list", default="5,7,9")
    ap.add_argument("--min-ac", type=int, default=10)
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    k_list = [int(x) for x in str(args.k_list).split(",") if str(x).strip() != ""]

    G, gsamp, chrom, pos, rsid = load_geno(args.geno_npz)
    pcs, csamp = load_cov(args.covar_npz, args.n_pcs)
    geno_map = {s: i for i, s in enumerate(gsamp.tolist()) if s is not None}
    cov_map = {s: i for i, s in enumerate(csamp.tolist()) if s is not None}

    ph = pd.read_csv(args.pheno_tsv, sep="\t")
    ph["sample_id"] = ph["sample_id"].apply(normalize_sample_id)
    ph[args.trait] = pd.to_numeric(ph[args.trait], errors="coerce")
    ph = ph.dropna(subset=["sample_id", args.trait]).drop_duplicates(subset=["sample_id"], keep="first")

    reg_idx = np.where((chrom.astype(str) == str(args.chrom)) & (pos >= args.start) & (pos <= args.end))[0]
    if reg_idx.size == 0:
        raise SystemExit("[ERROR] No SNPs in specified region")
    lead_idx = reg_idx[np.argmin(np.abs(pos[reg_idx] - args.lead_pos))]

    rows = []
    for kval in k_list:
        reg_pos = pos[reg_idx]
        order = np.argsort(np.abs(reg_pos - pos[lead_idx]))
        take = reg_idx[order[: min(kval, len(order))]]
        take = take[np.argsort(pos[take])]

        ids = []
        y = []
        Xpcs = []
        Gk = []
        xlead = []
        for _, r in ph.iterrows():
            sid = r["sample_id"]
            if sid in geno_map and sid in cov_map:
                gi = geno_map[sid]
                ci = cov_map[sid]
                ids.append(sid)
                y.append(float(r[args.trait]))
                Xpcs.append(pcs[ci, :])
                Gk.append(G[gi, take])
                xlead.append(G[gi, lead_idx])
        y = np.asarray(y, float)
        Xpcs = np.asarray(Xpcs, float)
        Gk = np.asarray(Gk, float)
        xlead = np.asarray(xlead, float)

        haps = encode_haps(Gk)
        counts = pd.Series(haps).value_counts()
        keep = set(counts[counts >= args.min_ac].index.tolist())
        mkeep = np.array([h in keep for h in haps], dtype=bool)
        y2 = y[mkeep]
        pcs2 = Xpcs[mkeep, :]
        haps2 = haps[mkeep]
        xlead2 = xlead[mkeep]

        # omnibus
        H = pd.get_dummies(pd.Series(haps2, dtype="category"), drop_first=True)
        X0 = np.column_stack([np.ones(len(y2)), pcs2])
        X1 = np.column_stack([X0, H.values.astype(float)])
        b0 = np.linalg.lstsq(X0, y2, rcond=None)[0]
        b1 = np.linalg.lstsq(X1, y2, rcond=None)[0]
        rss0 = float(np.sum((y2 - X0 @ b0) ** 2))
        rss1 = float(np.sum((y2 - X1 @ b1) ** 2))
        df_num = X1.shape[1] - X0.shape[1]
        df_den = len(y2) - X1.shape[1]
        F = ((rss0 - rss1) / df_num) / (rss1 / df_den) if (df_num > 0 and df_den > 0 and rss1 > 0) else np.nan
        p_omni = float(1 - stats.f.cdf(F, df_num, df_den)) if np.isfinite(F) else np.nan

        # best haplotype r2 vs lead
        best_r2 = np.nan
        best_h = None
        best_n = 0
        counts2 = pd.Series(haps2).value_counts()
        for h, c in counts2.items():
            if c < args.min_ac:
                continue
            yhap = np.array([1.0 if s == h else 0.0 for s in haps2], dtype=float)
            rr2 = r2(xlead2, yhap)
            if np.isnan(best_r2) or (np.isfinite(rr2) and rr2 > best_r2):
                best_r2 = rr2
                best_h = h
                best_n = int(c)

        rows.append({
            "k_snps": kval,
            "n_samples_tested": int(len(y2)),
            "window_start": int(pos[take].min()),
            "window_end": int(pos[take].max()),
            "n_haplotypes_kept": int(len(counts2)),
            "omnibus_F": F,
            "omnibus_p": p_omni,
            "best_haplotype": best_h,
            "best_haplotype_n": best_n,
            "r2_lead_vs_best_haplotype": best_r2,
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.out_prefix}_summary.tsv", sep="\t", index=False)
    out.to_excel(f"{args.out_prefix}_summary.xlsx", index=False)
    print(f"[OK] Wrote: {args.out_prefix}_summary.tsv")
    print(f"[OK] Wrote: {args.out_prefix}_summary.xlsx")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
