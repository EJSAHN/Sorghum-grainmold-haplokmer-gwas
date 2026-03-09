#!/usr/bin/env python3
"""
34_chr5_effect_size_pve.py

Estimate effect-size summaries for the top Chr5 signal:
1) Lead SNP partial R^2 / p-value in an OLS model with PCs as covariates.
2) Haplotype-cluster effect size on the target trait using one-way ANOVA (eta^2)
   and Kruskal-Wallis epsilon^2 from a cluster assignment file.

Outputs:
- <out_prefix>_summary.tsv
- <out_prefix>_summary.xlsx
- <out_prefix>_cluster_means.tsv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

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


def partial_r2(y: np.ndarray, x: np.ndarray, C: np.ndarray) -> Tuple[float, float, float]:
    # full: y ~ C + x; reduced: y ~ C
    X0 = C
    X1 = np.column_stack([C, x])
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    rss0 = float(np.sum((y - X0 @ b0) ** 2))
    rss1 = float(np.sum((y - X1 @ b1) ** 2))
    p1 = X1.shape[1]
    df_num = 1
    df_den = max(1, len(y) - p1)
    F = ((rss0 - rss1) / df_num) / (rss1 / df_den) if rss1 > 0 else np.inf
    p = float(1 - stats.f.cdf(F, df_num, df_den)) if np.isfinite(F) else 0.0
    r2_partial = (rss0 - rss1) / rss0 if rss0 > 0 else np.nan
    return float(r2_partial), float(F), p


def anova_eta2(y: np.ndarray, g: np.ndarray) -> Tuple[float, float, float]:
    df = pd.DataFrame({"y": y, "g": g})
    groups = [v["y"].values for _, v in df.groupby("g")]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    F, p = stats.f_oneway(*groups)
    grand = df["y"].mean()
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups)
    ss_total = np.sum((df["y"] - grand) ** 2)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    return float(eta2), float(F), float(p)


def kw_epsilon2(y: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    df = pd.DataFrame({"y": y, "g": g})
    groups = [v["y"].values for _, v in df.groupby("g")]
    if len(groups) < 2:
        return np.nan, np.nan
    H, p = stats.kruskal(*groups)
    n = len(df)
    k = len(groups)
    eps2 = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
    return float(max(eps2, 0.0)), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True)
    ap.add_argument("--pheno-tsv", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--chrom", default="5")
    ap.add_argument("--start", type=int, default=60250000)
    ap.add_argument("--end", type=int, default=60499999)
    ap.add_argument("--lead-pos", type=int, default=60278659)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--covar-npz", default=None)
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    G, gsamp, chrom, pos, rsid = load_geno(args.geno_npz)
    pcs = None
    csamp = None
    if args.covar_npz:
        pcs, csamp = load_cov(args.covar_npz, args.n_pcs)

    ph = pd.read_csv(args.pheno_tsv, sep="\t")
    ph["sample_id"] = ph["sample_id"].apply(normalize_sample_id)
    ph[args.trait] = pd.to_numeric(ph[args.trait], errors="coerce")
    ph = ph.dropna(subset=["sample_id", args.trait]).drop_duplicates(subset=["sample_id"], keep="first")

    # lead SNP closest in region
    m = (chrom.astype(str) == str(args.chrom)) & (pos >= args.start) & (pos <= args.end)
    idx = np.where(m)[0]
    if idx.size == 0:
        raise SystemExit("[ERROR] No SNPs in specified region")
    lead_idx = idx[np.argmin(np.abs(pos[idx] - args.lead_pos))]

    geno_map = {s: i for i, s in enumerate(gsamp.tolist()) if s is not None}
    rows = []
    y_list = []
    x_list = []
    pc_list = []
    ids = []
    cov_map = {s: i for i, s in enumerate(csamp.tolist())} if csamp is not None else {}
    for _, r in ph.iterrows():
        sid = r["sample_id"]
        if sid in geno_map:
            gi = geno_map[sid]
            x = G[gi, lead_idx]
            if np.isnan(x):
                continue
            ids.append(sid)
            y_list.append(float(r[args.trait]))
            x_list.append(float(x))
            if pcs is not None and sid in cov_map:
                pc_list.append(pcs[cov_map[sid], :])
            elif pcs is not None:
                pc_list.append(np.full((pcs.shape[1],), np.nan))

    y = np.array(y_list, float)
    x = np.array(x_list, float)
    if pcs is not None:
        C = np.column_stack([np.ones(len(y)), np.asarray(pc_list, float)])
        ok = np.all(np.isfinite(C), axis=1)
        y = y[ok]
        x = x[ok]
        C = C[ok, :]
    else:
        C = np.ones((len(y), 1), float)

    r2, F, p = partial_r2(y, x, C)

    # clusters
    cl = pd.read_csv(args.clusters, sep="\t")
    cl["sample_id"] = cl["sample_id"].apply(normalize_sample_id)
    cluster_col = "cluster_ranked" if "cluster_ranked" in cl.columns else ("cluster" if "cluster" in cl.columns else None)
    if cluster_col is None:
        raise SystemExit("[ERROR] clusters TSV missing cluster column")
    merged = ph[["sample_id", args.trait]].merge(cl[["sample_id", cluster_col]], on="sample_id", how="inner")
    merged = merged.dropna(subset=[args.trait, cluster_col]).copy()
    eta2, F_anova, p_anova = anova_eta2(merged[args.trait].values.astype(float), merged[cluster_col].values)
    eps2, p_kw = kw_epsilon2(merged[args.trait].values.astype(float), merged[cluster_col].values)

    means = merged.groupby(cluster_col)[args.trait].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    means.to_csv(f"{args.out_prefix}_cluster_means.tsv", sep="\t", index=False)

    out = pd.DataFrame([
        {
            "trait": args.trait,
            "chrom": args.chrom,
            "region_start": args.start,
            "region_end": args.end,
            "lead_pos_input": args.lead_pos,
            "lead_pos_used": int(pos[lead_idx]),
            "lead_rsid": str(rsid[lead_idx]),
            "n_lead_model": int(len(y)),
            "lead_partial_R2": r2,
            "lead_F": F,
            "lead_p": p,
            "cluster_n": int(len(merged)),
            "cluster_eta2": eta2,
            "cluster_F": F_anova,
            "cluster_ANOVA_p": p_anova,
            "cluster_epsilon2": eps2,
            "cluster_KW_p": p_kw,
            "cluster_mean_range": float(means["mean"].max() - means["mean"].min()) if not means.empty else np.nan,
        }
    ])
    out.to_csv(f"{args.out_prefix}_summary.tsv", sep="\t", index=False)
    out.to_excel(f"{args.out_prefix}_summary.xlsx", index=False)
    print(f"[OK] Wrote: {args.out_prefix}_summary.tsv")
    print(f"[OK] Wrote: {args.out_prefix}_summary.xlsx")
    print(f"[OK] Wrote: {args.out_prefix}_cluster_means.tsv")
    print(out.to_string(index=False))
    print("\n[Cluster means]")
    print(means.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
