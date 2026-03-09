#!/usr/bin/env python3
"""
33_trait_summary_heritability.py

Summarize accession-level phenotype distributions and estimate pseudo-heritability
for each trait using the same null-model REML delta used by 05_emmax_gwas.py.

Outputs:
- <out_prefix>_heritability.tsv
- <out_prefix>_heritability.xlsx

Traits processed:
- Grain mold: A, M, C, A-C, M-C
- Anthracnose: anthracnose

Pseudo-heritability is reported as:
    h2_pseudo = 1 / (1 + delta)
where delta = sigma_e^2 / sigma_g^2 estimated under the null LMM.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import optimize

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
    return G, samples


def load_covar(npz_path: str, n_pcs: int = 5):
    z = np.load(npz_path, allow_pickle=True)
    K = z["K"].astype(float)
    pcs = z["pcs"].astype(float)
    if pcs.ndim == 1:
        pcs = pcs.reshape(-1, 1)
    pcs = pcs[:, : min(n_pcs, pcs.shape[1])]
    samples = np.array([normalize_sample_id(x) for x in z["samples"]], dtype=object)
    return K, pcs, samples


def intersect_trait(geno_samples, cov_samples, pheno_df: pd.DataFrame, trait: str):
    if "sample_id" not in pheno_df.columns:
        raise ValueError("Phenotype TSV must contain 'sample_id'.")
    if trait not in pheno_df.columns:
        raise ValueError(f"Trait '{trait}' not found in phenotype TSV columns: {list(pheno_df.columns)}")

    df = pheno_df[["sample_id", trait]].copy()
    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    df[trait] = pd.to_numeric(df[trait], errors="coerce")
    df = df.dropna(subset=["sample_id", trait]).drop_duplicates(subset=["sample_id"], keep="first")

    cov_map = {s: i for i, s in enumerate(cov_samples.tolist()) if s is not None}
    geno_map = {s: i for i, s in enumerate(geno_samples.tolist()) if s is not None}

    keep_ids: List[str] = []
    geno_idx: List[int] = []
    cov_idx: List[int] = []
    y: List[float] = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        if sid in geno_map and sid in cov_map:
            keep_ids.append(sid)
            geno_idx.append(geno_map[sid])
            cov_idx.append(cov_map[sid])
            y.append(float(row[trait]))

    if len(keep_ids) < 20:
        raise ValueError(f"Too few overlapping samples for trait '{trait}': {len(keep_ids)}")
    return np.array(keep_ids), np.array(geno_idx, int), np.array(cov_idx, int), np.array(y, float)


def reml_negloglik(log_delta: float, y_t: np.ndarray, X_t: np.ndarray, lam: np.ndarray) -> float:
    delta = 10.0 ** log_delta
    v = lam + delta
    iv = 1.0 / v
    Xt_iv = X_t * iv[:, None]
    XtVinvX = X_t.T @ Xt_iv
    sign, logdet_X = np.linalg.slogdet(XtVinvX)
    if sign <= 0:
        return np.inf
    logdetV = np.sum(np.log(v))
    XtVinvY = X_t.T @ (y_t * iv)
    try:
        beta = np.linalg.solve(XtVinvX, XtVinvY)
    except np.linalg.LinAlgError:
        return np.inf
    yVinvY = float(y_t.T @ (y_t * iv))
    quad = float(XtVinvY.T @ beta)
    yPy = yVinvY - quad
    if yPy <= 0:
        return np.inf
    n, c = X_t.shape
    return 0.5 * (logdetV + logdet_X + (n - c) * np.log(yPy))


def estimate_delta_reml(y: np.ndarray, X: np.ndarray, K: np.ndarray) -> float:
    lam, U = np.linalg.eigh((K + K.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    U = U[:, idx]
    y_t = U.T @ y
    X_t = U.T @ X
    obj = lambda logd: reml_negloglik(logd, y_t, X_t, lam)
    res = optimize.minimize_scalar(obj, bounds=(-5, 5), method="bounded")
    if not res.success:
        raise RuntimeError(f"Failed to optimize delta: {res}")
    return 10.0 ** float(res.x)


def summarize_trait(name: str, pheno_tsv: str, geno_samples, cov_samples, K, pcs, n_pcs=5):
    ph = pd.read_csv(pheno_tsv, sep="\t")
    keep_ids, geno_idx, cov_idx, y = intersect_trait(geno_samples, cov_samples, ph, name)
    X = np.column_stack([np.ones(len(y)), pcs[cov_idx, : min(n_pcs, pcs.shape[1])]])
    Ksub = K[np.ix_(cov_idx, cov_idx)]
    delta = estimate_delta_reml(y, X, Ksub)
    pseudo_h2 = 1.0 / (1.0 + delta)
    return {
        "trait": name,
        "n": int(len(y)),
        "mean": float(np.mean(y)),
        "sd": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "delta": float(delta),
        "pseudo_h2": float(pseudo_h2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True)
    ap.add_argument("--covar-npz", required=True)
    ap.add_argument("--pheno-gm", required=True)
    ap.add_argument("--pheno-anth", required=True)
    ap.add_argument("--n-pcs", type=int, default=5)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    G, geno_samples = load_geno(args.geno_npz)
    K, pcs, cov_samples = load_covar(args.covar_npz, args.n_pcs)

    gm_traits = ["A", "M", "C", "A-C", "M-C"]
    rows = []
    for t in gm_traits:
        rows.append(summarize_trait(t, args.pheno_gm, geno_samples, cov_samples, K, pcs, args.n_pcs))
    rows.append(summarize_trait("anthracnose", args.pheno_anth, geno_samples, cov_samples, K, pcs, args.n_pcs))

    out = pd.DataFrame(rows)
    out_path = Path(f"{args.out_prefix}_heritability.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    out.to_excel(Path(f"{args.out_prefix}_heritability.xlsx"), index=False)
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Wrote: {args.out_prefix}_heritability.xlsx")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
