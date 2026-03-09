#!/usr/bin/env python3
"""
06_haplotype_kmer_gwas.py

"k-mer GWAS" analogue using **k-SNP haplotypes**:

- For each sliding window of k consecutive SNPs (within a chromosome),
  encode each sample's haplotype as a base-3 integer (0/1/2 dosage per SNP).
- Each distinct haplotype allele becomes a presence/absence marker (one-vs-rest).
- Test each haplotype marker with an EMMAX-style LMM scan (delta estimated once under null).

Why this is useful:
- Works even when you *don't have raw FASTQ reads* (only a HapMap genotype matrix).
- Captures local multi-SNP haplotypes (often more stable than single SNP hits).
- Conceptually bridges "k-mer" thinking with GWAS, especially for sparse marker sets.

Inputs:
  --geno-npz : filtered SNP dosage matrix (step 03)
  --pheno-tsv: phenotype TSV (step 01)
  --trait    : trait column
  --covar-npz: kinship+PCs (step 04)

Output:
  gzipped TSV with haplotype-marker hits.

Example:
  python scripts/06_haplotype_kmer_gwas.py ^
    --geno-npz 03_snp_matrix.filtered.npz ^
    --pheno-tsv 01_pheno_grain_mold.tsv ^
    --trait A ^
    --covar-npz 04_covariates.npz ^
    --k-snps 7 ^
    --step 1 ^
    --min-ac 10 ^
    --block-size 5000 ^
    --out 06_gwas_haplokmer_A_k7.tsv.gz

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
    df = pheno_df.copy()
    if "sample_id" not in df.columns:
        if "Taxa" in df.columns:
            df = df.rename(columns={"Taxa": "sample_id"})
        else:
            raise ValueError("Phenotype TSV must contain a 'sample_id' column.")

    df["sample_id"] = df["sample_id"].apply(normalize_sample_id)
    df = df.dropna(subset=["sample_id"])
    if trait not in df.columns:
        raise ValueError(f"Trait '{trait}' not found in phenotype TSV.")
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
        raise ValueError(f"Too few overlapping samples: {len(keep_rows)}")

    df2 = pd.DataFrame(keep_rows)
    sample_ids = df2["sample_id"].to_numpy(dtype="U50")
    y = df2[trait].to_numpy(dtype=float)

    return sample_ids, np.array(geno_idx, dtype=int), y


def reml_negloglik(log_delta: float, y_t: np.ndarray, X_t: np.ndarray, lam: np.ndarray) -> float:
    delta = 10.0 ** log_delta
    v = lam + delta
    iv = 1.0 / v

    Xt_iv = X_t * iv[:, None]
    XtVinvX = X_t.T @ Xt_iv

    logdetV = np.sum(np.log(v))
    try:
        sign, logdet_X = np.linalg.slogdet(XtVinvX)
        if sign <= 0:
            return np.inf
    except np.linalg.LinAlgError:
        return np.inf

    yt_iv = y_t * iv
    XtVinvY = X_t.T @ yt_iv
    try:
        beta = np.linalg.solve(XtVinvX, XtVinvY)
    except np.linalg.LinAlgError:
        return np.inf

    yVinvY = float(y_t.T @ yt_iv)
    quad = float(XtVinvY.T @ beta)
    yPy = yVinvY - quad
    if yPy <= 0:
        return np.inf

    n, c = X_t.shape
    return 0.5 * (logdetV + logdet_X + (n - c) * np.log(yPy))


def estimate_delta_reml(y: np.ndarray, X: np.ndarray, K: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
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
    delta = 10.0 ** float(res.x)
    return delta, lam, U


class EMMAXContext:
    """
    Precomputed context for scanning many markers with the same (y, X, K).
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, K: np.ndarray):
        self.y = y.astype(np.float64)
        self.X = X.astype(np.float64)
        self.K = K.astype(np.float64)

        n, c = self.X.shape
        self.n = n
        self.c = c

        self.delta, self.lam, self.U = estimate_delta_reml(self.y, self.X, self.K)
        print(f"[INFO] Estimated delta (sigma_e^2/sigma_g^2): {self.delta:.6g}")

        self.s = 1.0 / np.sqrt(self.lam + self.delta)

        # rotate + scale y and X
        self.y_t = (self.U.T @ self.y) * self.s
        self.X_t = (self.U.T @ self.X) * self.s[:, None]

        XtX = self.X_t.T @ self.X_t
        self.XtX_inv = np.linalg.inv(XtX)

        beta_cov = self.XtX_inv @ (self.X_t.T @ self.y_t)
        self.y_res = self.y_t - self.X_t @ beta_cov
        self.y_norm = float(self.y_res.T @ self.y_res)

        self.df = self.n - self.c - 1
        if self.df <= 1:
            raise ValueError(f"Not enough degrees of freedom: n={self.n}, c={self.c}")

    def scan_block(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scan a block of markers.

        M: (n x b) marker matrix, can have NaN (mean imputed internally).
        Returns beta, se, t, p arrays of length b.
        """
        n, b = M.shape
        assert n == self.n

        Mb = M.astype(np.float64, copy=True)
        col_means = np.nanmean(Mb, axis=0)
        inds = np.where(np.isnan(Mb))
        if inds[0].size > 0:
            Mb[inds] = col_means[inds[1]]

        # rotate + scale
        M_t = (self.U.T @ Mb) * self.s[:, None]

        # residualize vs X
        H = self.X_t.T @ M_t
        B = self.XtX_inv @ H
        M_hat = self.X_t @ B
        M_res = M_t - M_hat

        num = self.y_res.T @ M_res
        den = np.sum(M_res * M_res, axis=0)

        beta = np.full(b, np.nan, dtype=np.float64)
        se = np.full(b, np.nan, dtype=np.float64)
        tval = np.full(b, np.nan, dtype=np.float64)
        pval = np.full(b, np.nan, dtype=np.float64)

        ok = den > 1e-12
        beta[ok] = num[ok] / den[ok]
        rss = np.full(b, np.nan, dtype=np.float64)
        rss[ok] = self.y_norm - (num[ok] ** 2) / den[ok]
        rss = np.clip(rss, 1e-20, None)

        sigma2 = rss / self.df
        se[ok] = np.sqrt(sigma2[ok] / den[ok])
        tval[ok] = beta[ok] / se[ok]
        F = tval[ok] ** 2
        pval[ok] = stats.f.sf(F, 1, self.df)

        return beta, se, tval, pval


def chrom_sort_key(ch: str) -> Tuple[int, str]:
    """
    Try numeric sort for chromosomes like '1','2',...,'10'.
    """
    s = str(ch)
    try:
        return (int(s), "")
    except ValueError:
        return (10**9, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geno-npz", required=True, help="Filtered SNP NPZ (step 03 recommended)")
    ap.add_argument("--pheno-tsv", required=True, help="Phenotype TSV (step 01)")
    ap.add_argument("--trait", required=True, help="Trait column name")
    ap.add_argument("--covar-npz", required=True, help="Covariates NPZ (step 04)")
    ap.add_argument("--n-pcs", type=int, default=None, help="Number of PCs to include (default: all in covar-npz)")
    ap.add_argument("--k-snps", type=int, default=7, help="Haplotype window size in SNPs (default 7)")
    ap.add_argument("--step", type=int, default=1, help="Step size between windows (default 1)")
    ap.add_argument("--min-ac", type=int, default=10, help="Minimum allele count per haplotype allele (default 10)")
    ap.add_argument("--max-missing-hap", type=float, default=0.3, help="Skip windows with > this missing haplotypes (default 0.3)")
    ap.add_argument("--block-size", type=int, default=5000, help="Markers per scan block (default 5000)")
    ap.add_argument("--out", required=True, help="Output TSV.gz")
    args = ap.parse_args()

    geno = np.load(Path(args.geno_npz), allow_pickle=True)
    G_all = geno["G"]
    geno_samples = geno["samples"]
    chrom_all = geno["chrom"].astype(str)
    pos_all = geno["pos"].astype(int)
    rsid_all = geno["rsid"].astype(str)

    pheno = pd.read_csv(Path(args.pheno_tsv), sep="\t")
    sample_ids, geno_idx, y = intersect_and_reorder(geno_samples, pheno, args.trait)
    G = G_all[geno_idx, :]  # reorder rows
    n = G.shape[0]

    cov = np.load(Path(args.covar_npz), allow_pickle=True)
    cov_samples = cov["samples"]
    cov_map = {sid: i for i, sid in enumerate(cov_samples.tolist())}
    cov_idx = np.array([cov_map[sid] for sid in sample_ids.tolist()], dtype=int)

    K = cov["K"][np.ix_(cov_idx, cov_idx)]
    pcs_all = cov["pcs"][cov_idx, :]
    pcs = pcs_all if args.n_pcs is None else pcs_all[:, : args.n_pcs]

    X = np.column_stack([np.ones(n), pcs])

    ctx = EMMAXContext(y=y, X=X, K=K)

    # Sort SNPs by (chrom, pos) and reorder columns
    order = np.lexsort((pos_all, np.array([chrom_sort_key(c)[0] for c in chrom_all])))
    chrom = chrom_all[order]
    pos = pos_all[order]
    rsid = rsid_all[order]
    Gs = G[:, order]

    # process per chromosome
    unique_chroms = sorted(set(chrom.tolist()), key=chrom_sort_key)

    k = args.k_snps
    step = args.step
    pow3 = (3 ** np.arange(k)).astype(np.int64)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_path, "wt", encoding="utf-8") as out:
        out.write("\t".join([
            "marker_id", "chrom", "window_start_pos", "window_end_pos",
            "rsid_start", "rsid_end", "k_snps",
            "hap_code", "allele_count", "allele_freq",
            "beta", "se", "t", "p"
        ]) + "\n")

        block_markers: List[np.ndarray] = []
        block_meta: List[Tuple] = []

        def flush_block():
            nonlocal block_markers, block_meta
            if len(block_markers) == 0:
                return
            M = np.column_stack(block_markers)  # n x b
            beta, se, tval, pval = ctx.scan_block(M)

            for j, meta in enumerate(block_meta):
                marker_id, ch, w_start, w_end, r_start, r_end, hap_code, ac, af = meta
                out.write("\t".join(map(str, [
                    marker_id, ch, w_start, w_end, r_start, r_end, k,
                    hap_code, ac, f"{af:.6g}",
                    f"{beta[j]:.6g}", f"{se[j]:.6g}", f"{tval[j]:.6g}", f"{pval[j]:.6g}"
                ])) + "\n")

            block_markers = []
            block_meta = []

        total_markers = 0
        total_windows = 0

        for ch in unique_chroms:
            idx = np.where(chrom == ch)[0]
            if idx.size < k:
                continue

            # consecutive SNP indices for this chromosome
            for s0 in range(0, idx.size - k + 1, step):
                win = idx[s0 : s0 + k]
                W = Gs[:, win]  # n x k
                miss = np.isnan(W).any(axis=1)
                miss_rate = miss.mean()
                if miss_rate > args.max_missing_hap:
                    continue

                nonmiss = ~miss
                n_nonmiss = int(nonmiss.sum())
                if n_nonmiss < max(2 * args.min_ac, 20):
                    continue

                # base-3 encoding for non-missing
                W_int = W[nonmiss, :].astype(np.int64)
                codes_nm = W_int @ pow3  # (n_nonmiss,)
                # count haplotypes
                uniq, counts = np.unique(codes_nm, return_counts=True)

                # Only consider alleles with enough counts and not fixed
                for hap_code, ac in zip(uniq.tolist(), counts.tolist()):
                    if ac < args.min_ac:
                        continue
                    if ac > (n_nonmiss - args.min_ac):
                        continue

                    marker = np.full(n, np.nan, dtype=np.float64)
                    marker[nonmiss] = (codes_nm == hap_code).astype(np.float64)

                    w_start = int(pos[win[0]])
                    w_end = int(pos[win[-1]])
                    r_start = rsid[win[0]]
                    r_end = rsid[win[-1]]
                    af = ac / n_nonmiss

                    marker_id = f"{ch}:{w_start}-{w_end}|{r_start}-{r_end}|H{hap_code}"

                    block_markers.append(marker)
                    block_meta.append((marker_id, ch, w_start, w_end, r_start, r_end, hap_code, ac, af))
                    total_markers += 1

                    if len(block_markers) >= args.block_size:
                        flush_block()

                total_windows += 1
                if total_windows % 1000 == 0:
                    print(f"[INFO] {ch}: windows processed={total_windows:,}, markers tested={total_markers:,}")

        flush_block()

    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] Total windows scanned: {total_windows:,}")
    print(f"[INFO] Total haplotype markers tested: {total_markers:,}")


if __name__ == "__main__":
    main()
