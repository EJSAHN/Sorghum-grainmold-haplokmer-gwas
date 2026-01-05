#!/usr/bin/env python3
"""
05_emmax_gwas.py

Fast GWAS scan using an EMMAX-style linear mixed model (LMM):

  y = Xb + g * beta + u + e
  u ~ N(0, sigma_g^2 K)
  e ~ N(0, sigma_e^2 I)

We estimate delta = sigma_e^2 / sigma_g^2 once under the null (no marker),
then perform a fast generalized least squares scan for all markers.

This is designed for SAP-scale data (n ~ 200-400). Works well for tens/hundreds of thousands of markers.

Inputs:
  --geno-npz : SNP matrix NPZ (step 03)
  --pheno-tsv: cleaned phenotype TSV (step 01)
  --trait    : column name in phenotype TSV
  --covar-npz: covariates NPZ (step 04) containing K and pcs

Output:
  --out : gzipped TSV with per-marker association results.

Example:
  python scripts/05_emmax_gwas.py ^
    --geno-npz 03_snp_matrix.filtered.npz ^
    --pheno-tsv 01_pheno_grain_mold.tsv ^
    --trait A ^
    --covar-npz 04_covariates.npz ^
    --out 05_gwas_snp_A.tsv.gz
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy import optimize, stats


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


def intersect_and_reorder(
    geno_samples: np.ndarray, pheno_df: pd.DataFrame, trait: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      sample_ids (intersection, ordered)
      geno_index (indices into geno_samples)
      y (phenotype vector, aligned)
    """
    df = pheno_df.copy()
    if "sample_id" not in df.columns:
        raise ValueError("Phenotype TSV must contain a 'sample_id' column.")

    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    df = df.dropna(subset=["sample_id"])

    if trait not in df.columns:
        raise ValueError(f"Trait '{trait}' not found in phenotype TSV columns: {list(df.columns)}")

    df[trait] = pd.to_numeric(df[trait], errors="coerce")
    df = df.dropna(subset=[trait])

    geno_map = {sid: i for i, sid in enumerate(geno_samples.tolist())}

    keep_rows = []
    geno_idx = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        if sid in geno_map:
            keep_rows.append(row)
            geno_idx.append(geno_map[sid])

    if len(keep_rows) < 20:
        raise ValueError(f"Too few overlapping samples between genotype and phenotype: {len(keep_rows)}")

    df2 = pd.DataFrame(keep_rows)
    sample_ids = df2["sample_id"].to_numpy(dtype="U50")
    y = df2[trait].to_numpy(dtype=float)

    return sample_ids, np.array(geno_idx, dtype=int), y


def reml_negloglik(log_delta: float, y_t: np.ndarray, X_t: np.ndarray, lam: np.ndarray) -> float:
    """
    REML negative log-likelihood (up to additive constant), parameterized by log(delta).
    K eigenvalues = lam (>=0)
    V = K + delta I
    In rotated space (U' y, U' X)
    """
    delta = 10.0 ** log_delta
    v = lam + delta  # length n
    iv = 1.0 / v

    # Xt' V^-1 Xt = X_t' diag(iv) X_t
    Xt_iv = X_t * iv[:, None]
    XtVinvX = X_t.T @ Xt_iv  # c x c

    # log|V| = sum log(v)
    logdetV = np.sum(np.log(v))

    # log|XtVinvX|
    try:
        sign, logdet_X = np.linalg.slogdet(XtVinvX)
        if sign <= 0:
            return np.inf
    except np.linalg.LinAlgError:
        return np.inf

    # y' P y
    yt_iv = y_t * iv
    XtVinvY = X_t.T @ yt_iv  # c
    try:
        beta = np.linalg.solve(XtVinvX, XtVinvY)
    except np.linalg.LinAlgError:
        return np.inf

    # yt' V^-1 yt - (XtVinvY)' (XtVinvX)^-1 (XtVinvY)
    yVinvY = float(y_t.T @ yt_iv)
    quad = float(XtVinvY.T @ beta)
    yPy = yVinvY - quad
    if yPy <= 0:
        return np.inf

    n, c = X_t.shape
    # profiled sigma_g^2_hat = yPy / (n-c)
    # reml nll ~ 0.5*( logdetV + logdet_X + (n-c)*log(yPy) )
    return 0.5 * (logdetV + logdet_X + (n - c) * np.log(yPy))


def estimate_delta_reml(y: np.ndarray, X: np.ndarray, K: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate delta = sigma_e^2 / sigma_g^2 under the null using REML.
    Returns:
      delta, eigvals, eigvecs  (eigvals descending, eigvecs columns aligned)
    """
    # Eigen decomposition (ascending)
    lam, U = np.linalg.eigh((K + K.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    # sort descending for convenience
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    U = U[:, idx]

    # rotate y and X
    y_t = U.T @ y
    X_t = U.T @ X

    # optimize log10(delta)
    obj = lambda logd: reml_negloglik(logd, y_t, X_t, lam)
    res = optimize.minimize_scalar(obj, bounds=(-5, 5), method="bounded")
    if not res.success:
        raise RuntimeError(f"Failed to optimize delta: {res}")
    log_delta = float(res.x)
    delta = 10.0 ** log_delta
    return delta, lam, U


def emmax_scan(
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    G: np.ndarray,
    block_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run EMMAX scan.

    Inputs:
      y: (n,)
      X: (n, c) with intercept already included
      K: (n, n) kinship
      G: (n, m) genotype dosage, NaN allowed (mean-imputed per marker)

    Returns:
      beta (m,), se (m,), t (m,), p (m,)
    """
    n, c = X.shape
    m = G.shape[1]

    delta, lam, U = estimate_delta_reml(y, X, K)
    print(f"[INFO] Estimated delta (sigma_e^2/sigma_g^2): {delta:.6g}")

    s = 1.0 / np.sqrt(lam + delta)  # length n

    # rotate + scale y and X
    y_t = (U.T @ y) * s
    X_t = (U.T @ X) * s[:, None]

    # Precompute (X'X)^-1 and residualized y
    XtX = X_t.T @ X_t
    XtX_inv = np.linalg.inv(XtX)
    beta_cov = XtX_inv @ (X_t.T @ y_t)
    y_res = y_t - X_t @ beta_cov
    y_norm = float(y_res.T @ y_res)

    df = n - c - 1
    if df <= 1:
        raise ValueError(f"Not enough degrees of freedom: n={n}, c={c}")

    betas = np.full(m, np.nan, dtype=np.float64)
    ses = np.full(m, np.nan, dtype=np.float64)
    ts = np.full(m, np.nan, dtype=np.float64)
    ps = np.full(m, np.nan, dtype=np.float64)

    for start in range(0, m, block_size):
        end = min(m, start + block_size)
        Gb = G[:, start:end].astype(np.float64, copy=True)

        # mean impute within block
        col_means = np.nanmean(Gb, axis=0)
        inds = np.where(np.isnan(Gb))
        if inds[0].size > 0:
            Gb[inds] = col_means[inds[1]]

        # rotate + scale: (U'G) * s
        G_t = (U.T @ Gb) * s[:, None]

        # residualize G w.r.t covariates
        H = X_t.T @ G_t               # c x b
        B = XtX_inv @ H               # c x b
        G_hat = X_t @ B               # n x b
        G_res = G_t - G_hat           # n x b

        num = y_res.T @ G_res         # (b,)
        den = np.sum(G_res * G_res, axis=0)  # (b,)

        # avoid division by zero
        ok = den > 1e-12
        beta = np.full(end - start, np.nan, dtype=np.float64)
        se = np.full(end - start, np.nan, dtype=np.float64)
        tval = np.full(end - start, np.nan, dtype=np.float64)
        pval = np.full(end - start, np.nan, dtype=np.float64)

        beta[ok] = num[ok] / den[ok]
        # rss = y_norm - (num^2 / den)
        rss = np.full(end - start, np.nan, dtype=np.float64)
        rss[ok] = y_norm - (num[ok] ** 2) / den[ok]
        rss = np.clip(rss, 1e-20, None)

        sigma2 = rss / df
        se[ok] = np.sqrt(sigma2[ok] / den[ok])
        tval[ok] = beta[ok] / se[ok]
        # F-test with 1 df
        F = (tval[ok] ** 2)
        pval[ok] = stats.f.sf(F, 1, df)

        betas[start:end] = beta
        ses[start:end] = se
        ts[start:end] = tval
        ps[start:end] = pval

        if start % (block_size * 10) == 0:
            print(f"[INFO] Scanned {start:,}-{end:,} / {m:,} markers")

    return betas, ses, ts, ps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True, help="Genotype NPZ (step 03 recommended)")
    ap.add_argument("--pheno-tsv", required=True, help="Phenotype TSV (step 01)")
    ap.add_argument("--trait", required=True, help="Trait column name in phenotype TSV")
    ap.add_argument("--covar-npz", required=True, help="Covariates NPZ (step 04)")
    ap.add_argument("--n-pcs", type=int, default=None, help="Number of PCs to include (default: all in covar-npz)")
    ap.add_argument("--block-size", type=int, default=5000, help="Markers per block (default 5000)")
    ap.add_argument("--out", required=True, help="Output TSV.gz")
    args = ap.parse_args()

    geno = np.load(Path(args.geno_npz), allow_pickle=True)
    G_all = geno["G"]
    geno_samples = geno["samples"]

    pheno = pd.read_csv(Path(args.pheno_tsv), sep="\t")
    # if user didn't create sample_id, try common alternatives
    if "sample_id" not in pheno.columns:
        if "Taxa" in pheno.columns:
            pheno = pheno.rename(columns={"Taxa": "sample_id"})
        else:
            raise ValueError("Phenotype file must have 'sample_id' column (or 'Taxa' which will be renamed).")

    sample_ids, geno_idx, y = intersect_and_reorder(geno_samples, pheno, args.trait)
    G = G_all[geno_idx, :]  # reorder rows

    cov = np.load(Path(args.covar_npz), allow_pickle=True)
    cov_samples = cov["samples"]
    # reorder K/pcs to match phenotype/genotype sample order
    cov_map = {sid: i for i, sid in enumerate(cov_samples.tolist())}
    cov_idx = []
    for sid in sample_ids.tolist():
        if sid not in cov_map:
            raise ValueError(f"Sample '{sid}' not found in covariates NPZ. Recompute step 04 on matching genotypes.")
        cov_idx.append(cov_map[sid])
    cov_idx = np.array(cov_idx, dtype=int)

    K = cov["K"][np.ix_(cov_idx, cov_idx)]
    pcs_all = cov["pcs"][cov_idx, :]

    if args.n_pcs is None:
        pcs = pcs_all
    else:
        pcs = pcs_all[:, : args.n_pcs]

    # Build covariate matrix X with intercept + PCs
    X = np.column_stack([np.ones(len(sample_ids)), pcs])

    betas, ses, ts, ps = emmax_scan(y=y, X=X, K=K, G=G, block_size=args.block_size)

    # Assemble output table
    out_df = pd.DataFrame({
        "rsid": geno["rsid"],
        "chrom": geno["chrom"],
        "pos": geno["pos"],
        "maf": geno["maf"],
        "missing_rate": geno["missing_rate"],
        "beta": betas,
        "se": ses,
        "t": ts,
        "p": ps,
    })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"[OK] Wrote: {out_path}")
    print("[INFO] Top hits:")
    print(out_df.sort_values("p").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
