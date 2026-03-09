#!/usr/bin/env python3
"""
15_permutation_core_hotspots.py (STRENGTH UPGRADE)

Permutation test for hotspot overlap AND hotspot strength under a rank-preserving null.

Null model:
- For each GWAS file, shuffle p-values across markers (chrom/pos fixed).
- Apply the SAME selection rule (topn / p / bonferroni / fdr).
- Recompute hotspot metrics.

Hotspot definition:
- bin = floor(pos / window)
- hotspot_key = chrom:bin

Metrics (count-based):
- n_hotspots
- n_core (hotspots with n_traits>=core_traits AND n_methods>=core_methods)
- max_n_traits, max_n_methods
- n_traits_ge3, n_traits_ge4, n_traits_ge5

NEW metrics (strength-based):
- max_strength: maximum hotspot strength across all hotspots
- core_strength_sum: sum of strengths over core hotspots
- core_strength_max: max strength among core hotspots
- strength_sum_total: sum of strengths over all hotspots (selected hits)

Hotspot strength:
- strength = sum_{all selected hits in the hotspot} (-log10(p))

Outputs:
- <prefix>_perm_metrics.tsv
- <prefix>_perm_summary.tsv (empirical p_ge / p_le / p_two)
- Figures (PNG 300dpi + PDF):
    <prefix>_hist_n_core.(png/pdf)
    <prefix>_hist_n_hotspots.(png/pdf)
    <prefix>_hist_max_n_traits.(png/pdf)
    <prefix>_hist_core_strength_sum.(png/pdf)
    <prefix>_hist_max_strength.(png/pdf)

Example:
python scripts/15_permutation_core_hotspots.py --glob "05_gwas_snp_*.tsv.gz" --glob "06_gwas_haplokmer_*.tsv.gz" --select p --p_thresh 1e-4 --window 250000 --core_traits 3 --core_methods 2 --nperm 500 --seed 1 --out_prefix 15_perm_strength_p1e4
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def parse_label_from_filename(fn: str) -> Tuple[str, str]:
    stem = Path(fn).name
    stem = re.sub(r"\.tsv(\.gz)?$", "", stem)

    method = "UNKNOWN"
    if "_snp_" in stem:
        method = "SNP"
    elif "_haplokmer_" in stem:
        m = re.search(r"_k(\d+)$", stem)
        method = f"HKMER_k{m.group(1)}" if m else "HKMER"

    trait = stem
    trait = trait.replace("05_gwas_snp_", "")
    trait = trait.replace("06_gwas_haplokmer_", "")
    trait = trait.replace("grainmold_", "grainmold:")
    trait = trait.replace("anthracnose", "anthracnose")
    trait = re.sub(r"_k\d+$", "", trait)

    # normalize contrast tokens
    trait = trait.replace("AminusC", "A-C")
    trait = trait.replace("MminusC", "M-C")

    return method, trait


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = p.astype(float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


def load_gwas_minimal(path: str) -> pd.DataFrame:
    """
    Standardize GWAS file to columns: chrom, pos, p.
    Supports haplo-kmer (window_start_pos/window_end_pos -> midpoint).
    """
    df = pd.read_csv(path, sep="\t", compression="infer")

    chrom_col = _find_col(df.columns.tolist(), ["chrom", "chr", "chromosome"])
    p_col = _find_col(df.columns.tolist(), ["p", "pval", "p_value", "p-value"])

    pos_col = _find_col(df.columns.tolist(), ["pos", "position", "bp", "lead_pos"])
    wstart_col = _find_col(df.columns.tolist(), ["window_start_pos", "start_pos", "window_start", "start"])
    wend_col = _find_col(df.columns.tolist(), ["window_end_pos", "end_pos", "window_end", "end"])

    if chrom_col is None or p_col is None:
        raise ValueError(f"{path}: need chrom and p columns; got {df.columns.tolist()}")

    out = pd.DataFrame()
    out["chrom"] = df[chrom_col].astype(str)
    out["p"] = pd.to_numeric(df[p_col], errors="coerce")

    if pos_col is not None:
        out["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        if wstart_col is None:
            raise ValueError(f"{path}: need pos or window_start_pos; got {df.columns.tolist()}")
        s = pd.to_numeric(df[wstart_col], errors="coerce")
        if wend_col is not None:
            e = pd.to_numeric(df[wend_col], errors="coerce")
            out["pos"] = (s + e) / 2.0
        else:
            out["pos"] = s

    out = out.dropna(subset=["chrom", "pos", "p"])
    out = out[(out["p"] > 0) & (out["p"] <= 1)]
    out["pos"] = out["pos"].astype(int)
    out["chrom"] = out["chrom"].astype(str)
    return out.reset_index(drop=True)


def select_hits(df: pd.DataFrame, select: str, topn: int, alpha: float, p_thresh: float) -> pd.DataFrame:
    if select == "topn":
        return df.nsmallest(topn, "p").copy()
    if select == "p":
        return df[df["p"] <= p_thresh].copy()
    if select == "bonferroni":
        thr = alpha / len(df)
        return df[df["p"] <= thr].copy()
    if select == "fdr":
        q = bh_fdr(df["p"].to_numpy())
        return df[q <= alpha].copy()
    raise ValueError(f"Unknown selection mode: {select}")


def build_contrib(sel: pd.DataFrame, trait: str, method: str, window: int) -> pd.DataFrame:
    d = sel.copy()
    d["trait"] = trait
    d["method"] = method
    d["bin"] = (d["pos"] // window).astype(int)
    d["hotspot_key"] = d["chrom"].astype(str) + ":" + d["bin"].astype(str)
    # strength component
    d["neglogp"] = -np.log10(np.clip(d["p"].to_numpy(float), np.nextafter(0, 1), 1.0))
    return d[["hotspot_key", "chrom", "bin", "pos", "p", "neglogp", "trait", "method"]]


def summarize_hotspots(contrib: pd.DataFrame, core_traits: int, core_methods: int) -> Dict[str, float]:
    if contrib.empty:
        return {
            "n_hotspots": 0.0,
            "n_core": 0.0,
            "max_n_traits": 0.0,
            "max_n_methods": 0.0,
            "n_traits_ge3": 0.0,
            "n_traits_ge4": 0.0,
            "n_traits_ge5": 0.0,
            "max_strength": 0.0,
            "core_strength_sum": 0.0,
            "core_strength_max": 0.0,
            "strength_sum_total": 0.0,
        }

    g = contrib.groupby("hotspot_key").agg(
        n_traits=("trait", pd.Series.nunique),
        n_methods=("method", pd.Series.nunique),
        strength=("neglogp", "sum"),
    )

    n_hotspots = float(len(g))
    is_core = (g["n_traits"] >= core_traits) & (g["n_methods"] >= core_methods)

    return {
        "n_hotspots": n_hotspots,
        "n_core": float(is_core.sum()),
        "max_n_traits": float(g["n_traits"].max()),
        "max_n_methods": float(g["n_methods"].max()),
        "n_traits_ge3": float((g["n_traits"] >= 3).sum()),
        "n_traits_ge4": float((g["n_traits"] >= 4).sum()),
        "n_traits_ge5": float((g["n_traits"] >= 5).sum()),
        # strength metrics
        "max_strength": float(g["strength"].max()),
        "core_strength_sum": float(g.loc[is_core, "strength"].sum()) if is_core.any() else 0.0,
        "core_strength_max": float(g.loc[is_core, "strength"].max()) if is_core.any() else 0.0,
        "strength_sum_total": float(g["strength"].sum()),
    }


def hist_with_obs(values: np.ndarray, obs: float, title: str, xlabel: str, out_prefix: str, suffix: str):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(values, bins=30)
    ax.axvline(obs, linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_{suffix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="Glob pattern(s) for GWAS TSV(.gz)")
    ap.add_argument("--out_prefix", required=True, help="Output prefix")
    ap.add_argument("--window", type=int, default=250_000, help="Hotspot bin size (bp)")
    ap.add_argument("--select", choices=["topn", "p", "bonferroni", "fdr"], default="topn")
    ap.add_argument("--topn", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--p_thresh", type=float, default=1e-4)
    ap.add_argument("--core_traits", type=int, default=3)
    ap.add_argument("--core_methods", type=int, default=2)
    ap.add_argument("--nperm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    files: List[str] = []
    for pattern in args.glob:
        files.extend([str(p) for p in sorted(Path(".").glob(pattern))])
    files = sorted(set(files))
    if not files:
        raise SystemExit("[ERROR] No input files matched --glob patterns.")

    print(f"[INFO] Loading {len(files)} GWAS files and selecting hits...")
    gwas_list = []
    trait_set = set()
    method_set = set()

    for fp in files:
        method, trait = parse_label_from_filename(fp)
        df = load_gwas_minimal(fp)
        sel = select_hits(df, args.select, args.topn, args.alpha, args.p_thresh)
        print(f"  - {Path(fp).name}: markers={len(df):,} selected={len(sel):,} method={method} trait={trait}")
        gwas_list.append({"file": fp, "df": df, "method": method, "trait": trait})
        trait_set.add(trait)
        method_set.add(method)

    print(f"[INFO] Traits ({len(trait_set)}): {', '.join(sorted(trait_set))}")
    print(f"[INFO] Methods ({len(method_set)}): {', '.join(sorted(method_set))}")

    # OBS
    print("[INFO] Building OBS hotspots...")
    obs_parts = []
    for obj in gwas_list:
        sel = select_hits(obj["df"], args.select, args.topn, args.alpha, args.p_thresh)
        obs_parts.append(build_contrib(sel, obj["trait"], obj["method"], args.window))
    obs_hits = pd.concat(obs_parts, ignore_index=True) if obs_parts else pd.DataFrame()
    obs_metrics = summarize_hotspots(obs_hits, args.core_traits, args.core_methods)
    print(f"[OBS] {obs_metrics}")

    # PERM
    print(f"[INFO] Running permutations: nperm={args.nperm} (rank-preserving p shuffle per file)")
    perm_rows = [{"perm_id": -1, **obs_metrics}]

    for b in range(1, args.nperm + 1):
        parts = []
        for obj in gwas_list:
            df = obj["df"].copy()
            p = df["p"].to_numpy()
            p_shuf = p.copy()
            rng.shuffle(p_shuf)
            df["p"] = p_shuf
            sel = select_hits(df, args.select, args.topn, args.alpha, args.p_thresh)
            parts.append(build_contrib(sel, obj["trait"], obj["method"], args.window))

        hits = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        met = summarize_hotspots(hits, args.core_traits, args.core_methods)
        perm_rows.append({"perm_id": b, **met})

        if b % 50 == 0:
            print(f"  ... {b}/{args.nperm} done")

    perm_df = pd.DataFrame(perm_rows)
    metrics_path = f"{args.out_prefix}_perm_metrics.tsv"
    perm_df.to_csv(metrics_path, sep="\t", index=False)
    print(f"[OK] Wrote: {metrics_path}")

    null = perm_df[perm_df["perm_id"] >= 1].copy()

    summary_rows = []
    metric_list = [
        "n_core", "n_hotspots", "max_n_traits", "max_n_methods",
        "n_traits_ge3", "n_traits_ge4", "n_traits_ge5",
        "max_strength", "core_strength_sum", "core_strength_max", "strength_sum_total",
    ]

    for metric in metric_list:
        obs = float(obs_metrics[metric])
        arr = null[metric].to_numpy(dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        p_ge = float((arr >= obs).mean())
        p_le = float((arr <= obs).mean())
        p_two = float(min(1.0, 2.0 * min(p_ge, p_le)))
        summary_rows.append({
            "metric": metric,
            "observed": obs,
            "perm_mean": mean,
            "perm_sd": sd,
            "perm_p_ge": p_ge,
            "perm_p_le": p_le,
            "perm_p_two": p_two,
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = f"{args.out_prefix}_perm_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[OK] Wrote: {summary_path}")
    print("[SUMMARY]")
    print(summary.to_string(index=False))

    # figures
    outp = args.out_prefix
    hist_with_obs(null["n_core"].to_numpy(float), float(obs_metrics["n_core"]), "Permutation null: n_core", "n_core", outp, "hist_n_core")
    hist_with_obs(null["n_hotspots"].to_numpy(float), float(obs_metrics["n_hotspots"]), "Permutation null: n_hotspots", "n_hotspots", outp, "hist_n_hotspots")
    hist_with_obs(null["max_n_traits"].to_numpy(float), float(obs_metrics["max_n_traits"]), "Permutation null: max_n_traits", "max_n_traits", outp, "hist_max_n_traits")
    hist_with_obs(null["core_strength_sum"].to_numpy(float), float(obs_metrics["core_strength_sum"]), "Permutation null: core_strength_sum", "core_strength_sum", outp, "hist_core_strength_sum")
    hist_with_obs(null["max_strength"].to_numpy(float), float(obs_metrics["max_strength"]), "Permutation null: max_strength", "max_strength", outp, "hist_max_strength")

    print(f"[OK] Wrote figures: {outp}_hist_*. (png/pdf)")


if __name__ == "__main__":
    main()
