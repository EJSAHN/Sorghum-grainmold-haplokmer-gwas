#!/usr/bin/env python3
"""
63_make_table2_near_miss_candidates.py

Build Table 2 = "Near-miss candidates" under the same strict marker inclusion (p <= 1e-4)
but excluding the final strict core bin (chr5:60.25–60.50 Mb).

Design goal:
- Keep the rigor (p <= 1e-4, 250 kb bins, strength = sum(-log10 p))
- Restore "interesting breadth" by listing top candidate bins that fail *one* core criterion.

Selection logic (ordered):
  Tier 1: >=2 traits AND both methods (SNP + HKMER)   [closest to core]
  Tier 2: >=3 traits (any method)
  Tier 3: >=2 traits (any method)

We fill up to --topn rows by tier order, excluding the chr5 final core bin.

Outputs:
- Table2_NearMiss.tsv
- Table2_NearMiss.csv
- Table2_NearMiss.xlsx

Run:
  python scripts/63_make_table2_near_miss_candidates.py --topn 20
"""

from __future__ import annotations
import argparse
import glob
import gzip
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---- defaults matching your pipeline ----
P_THRESH_DEFAULT = 1e-4
WINDOW_BP_DEFAULT = 250_000

# final strict core bin you found
FINAL_CORE_CHR = "5"
FINAL_CORE_START = 60250000
FINAL_CORE_END = 60499999

# optional annotation resources
GFF_DEFAULT = Path("Sbicolor_454_v3.1.1.gene.gff3.gz")
ANNOT_DEFAULT = Path("Sbicolor_454_v3.1.1.P14.annotation_info.txt.gz")


def _find_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def load_gwas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer")
    cols = df.columns.tolist()

    chrom_col = _find_col(cols, ["chrom", "chr", "chromosome"])
    p_col = _find_col(cols, ["p", "pval", "p_value", "p-value"])
    pos_col = _find_col(cols, ["pos", "position", "bp"])
    ws_col = _find_col(cols, ["window_start_pos", "window_start", "start_pos", "start"])
    we_col = _find_col(cols, ["window_end_pos", "window_end", "end_pos", "end"])

    if chrom_col is None or p_col is None:
        raise ValueError(f"{path}: missing chrom/p columns")

    out = pd.DataFrame()
    out["chrom"] = df[chrom_col].astype(str)
    out["p"] = pd.to_numeric(df[p_col], errors="coerce")

    if pos_col is not None:
        out["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        if ws_col is None:
            raise ValueError(f"{path}: missing pos or window_start_pos")
        s = pd.to_numeric(df[ws_col], errors="coerce")
        if we_col is not None:
            e = pd.to_numeric(df[we_col], errors="coerce")
            out["pos"] = (s + e) / 2.0
        else:
            out["pos"] = s

    out = out.dropna(subset=["chrom", "pos", "p"])
    out = out[(out["p"] > 0) & (out["p"] <= 1)]
    out["pos"] = out["pos"].astype(int)

    fn = Path(path).name
    method = "SNP" if "_snp_" in fn else ("HKMER" if "_haplokmer_" in fn else "UNK")

    trait = fn
    trait = trait.replace("05_gwas_snp_", "").replace("06_gwas_haplokmer_", "")
    trait = trait.replace(".tsv.gz", "").replace(".tsv", "")
    trait = trait.replace("_AminusC", ":A-C").replace("_MminusC", ":M-C")
    trait = trait.replace("_A", ":A").replace("_M", ":M").replace("_C", ":C")
    trait = trait.replace("grainmold_", "grainmold").replace("anthracnose", "anthracnose")
    trait = re.sub(r"_k\d+$", "", trait)

    out["method"] = method
    out["trait"] = trait
    out["source_file"] = fn
    return out


def open_textmaybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def load_genes(gff: Path):
    genes: Dict[str, List[Tuple[int, int, str, str]]] = {}
    if not gff.exists():
        return genes
    with open_textmaybe_gz(gff) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, *_rest, attrs = parts
            if feature != "gene":
                continue
            try:
                s = int(start); e = int(end)
            except:
                continue
            gid = ""
            name = ""
            for field in attrs.split(";"):
                if field.startswith("ID="):
                    gid = field.replace("ID=", "").strip()
                elif field.startswith("Name="):
                    name = field.replace("Name=", "").strip()
            if gid:
                genes.setdefault(str(chrom), []).append((s, e, gid, name))
    for c in genes:
        genes[c].sort(key=lambda x: x[0])
    return genes


def load_annotation_map(path: Path):
    amap: Dict[str, str] = {}
    if not path.exists():
        return amap
    with open_textmaybe_gz(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gid = parts[0].strip()
            desc = parts[-1].strip()
            if gid:
                amap[gid] = desc
    return amap


def resolve_chrom_key(genes_by_chrom: Dict[str, list], chrom: str) -> Optional[str]:
    c = str(chrom).strip()
    if c in genes_by_chrom:
        return c
    m = re.search(r"(\d+)", c)
    if not m:
        return None
    n = int(m.group(1))
    candidates = [c, str(n), f"{n:02d}", f"Chr{n}", f"Chr{n:02d}", f"chr{n}", f"chr{n:02d}"]
    for k in candidates:
        if k in genes_by_chrom:
            return k
    return None


def nearest_gene(genes_by_chrom, ann_map, chrom: str, pos: int, flank: int = 200_000):
    ck = resolve_chrom_key(genes_by_chrom, chrom)
    if ck is None:
        return ("", "", "")
    genes = genes_by_chrom[ck]
    left = pos - flank
    right = pos + flank
    best = None
    best_dist = None
    for (s, e, gid, name) in genes:
        if e < left:
            continue
        if s > right:
            break
        if pos < s:
            dist = s - pos
        elif pos > e:
            dist = pos - e
        else:
            dist = 0
        if best is None or dist < best_dist:
            best = (gid, name)
            best_dist = dist
    if best is None:
        return ("", "", "")
    gid, name = best
    desc = ann_map.get(gid, "") if gid else ""
    return (gid, name, desc)


def uniq_join(x: pd.Series) -> str:
    return ";".join(sorted(set(x.astype(str))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=P_THRESH_DEFAULT, help="marker inclusion threshold (default 1e-4)")
    ap.add_argument("--window", type=int, default=WINDOW_BP_DEFAULT, help="bin size in bp (default 250000)")
    ap.add_argument("--topn", type=int, default=20, help="rows in Table 2 (default 20)")
    ap.add_argument("--gff", default=str(GFF_DEFAULT), help="GFF3 genes (optional)")
    ap.add_argument("--annotation", default=str(ANNOT_DEFAULT), help="annotation_info (optional)")
    args = ap.parse_args()

    files = sorted(glob.glob("05_gwas_snp_*.tsv.gz") + glob.glob("06_gwas_haplokmer_*.tsv.gz"))
    if not files:
        raise SystemExit("[ERROR] No GWAS files found (05_gwas_snp_*.tsv.gz / 06_gwas_haplokmer_*.tsv.gz).")

    parts = []
    for fp in files:
        df = load_gwas(fp)
        df = df[df["p"] <= args.p].copy()
        if len(df):
            parts.append(df)

    if not parts:
        raise SystemExit("[ERROR] No markers pass p-threshold in any GWAS file.")

    hits = pd.concat(parts, ignore_index=True)
    hits["bin"] = (hits["pos"] // args.window).astype(int)
    hits["bin_start"] = hits["bin"] * args.window
    hits["bin_end"] = hits["bin_start"] + args.window - 1
    hits["strength"] = -np.log10(np.clip(hits["p"].to_numpy(float), np.nextafter(0, 1), 1.0))

    agg = hits.groupby(["chrom", "bin", "bin_start", "bin_end"], as_index=False).agg(
        strength_sum=("strength", "sum"),
        best_p=("p", "min"),
        n_hits=("p", "size"),
        traits=("trait", uniq_join),
        methods=("method", uniq_join),
    )
    agg["n_traits"] = agg["traits"].map(lambda s: 0 if not s else len(s.split(";")))
    agg["n_methods"] = agg["methods"].map(lambda s: 0 if not s else len(s.split(";")))

    # exclude final chr5 core bin
    agg = agg[~((agg["chrom"].astype(str) == FINAL_CORE_CHR) &
                (agg["bin_start"].astype(int) == FINAL_CORE_START) &
                (agg["bin_end"].astype(int) == FINAL_CORE_END))].copy()

    # lead marker per bin
    lead = hits.sort_values("p").drop_duplicates(subset=["chrom", "bin"], keep="first")
    agg = agg.merge(lead[["chrom","bin","pos","p","trait","method","source_file"]], on=["chrom","bin"], how="left")
    agg = agg.rename(columns={"pos":"lead_pos","p":"lead_p","trait":"lead_trait","method":"lead_method","source_file":"lead_source_file"})

    # ---- tiered selection ----
    tier1 = agg[(agg["n_traits"] >= 2) & (agg["n_methods"] >= 2)].copy()
    tier1["tier"] = "Tier1: >=2 traits + SNP&HKMER"

    tier2 = agg[(agg["n_traits"] >= 3)].copy()
    tier2 = tier2[~tier2.set_index(["chrom","bin"]).index.isin(tier1.set_index(["chrom","bin"]).index)]
    tier2["tier"] = "Tier2: >=3 traits (any method)"

    tier3 = agg[(agg["n_traits"] >= 2)].copy()
    tier3 = tier3[~tier3.set_index(["chrom","bin"]).index.isin(tier1.set_index(["chrom","bin"]).index)]
    tier3 = tier3[~tier3.set_index(["chrom","bin"]).index.isin(tier2.set_index(["chrom","bin"]).index)]
    tier3["tier"] = "Tier3: >=2 traits (any method)"

    cand = pd.concat([tier1, tier2, tier3], ignore_index=True)
    cand = cand.sort_values(["strength_sum","best_p"], ascending=[False, True]).reset_index(drop=True)
    cand = cand.head(args.topn).copy()

    # annotation (optional)
    genes = load_genes(Path(args.gff))
    ann = load_annotation_map(Path(args.annotation))
    gids=[]; gnames=[]; gdescs=[]
    for _, r in cand.iterrows():
        gid, gname, gdesc = nearest_gene(genes, ann, str(r["chrom"]), int(r["lead_pos"]), flank=200_000)
        gids.append(gid); gnames.append(gname); gdescs.append(gdesc)
    cand["nearest_gene_id"] = gids
    cand["nearest_gene_name"] = gnames
    cand["nearest_gene_desc"] = gdescs

    # Table2 pretty
    out = pd.DataFrame({
        "Tier": cand["tier"],
        "Chr": cand["chrom"].astype(str),
        "Interval (Mb)": (cand["bin_start"]/1e6).map(lambda x: f"{x:.2f}") + "–" + (cand["bin_end"]/1e6).map(lambda x: f"{x:.2f}"),
        "Hotspot strength (Σ−log10 p)": cand["strength_sum"].map(lambda x: f"{x:.2f}"),
        "Peak p-value": cand["best_p"].map(lambda x: f"{x:.2e}"),
        "# hits": cand["n_hits"].astype(int),
        "# traits": cand["n_traits"].astype(int),
        "# methods": cand["n_methods"].astype(int),
        "Traits": cand["traits"].astype(str).str.replace(";", ", "),
        "Methods": cand["methods"].astype(str).str.replace("HKMER", "Haplo-kmer").str.replace(";", " + "),
        "Lead position (bp)": cand["lead_pos"].astype(int),
        "Lead trait": cand["lead_trait"],
        "Lead method": cand["lead_method"].astype(str).str.replace("HKMER", "Haplo-kmer"),
        "Nearest gene": cand["nearest_gene_id"].replace("", "-"),
        "Gene note": cand["nearest_gene_desc"].replace("", "-"),
    })

    out.to_csv("Table2_NearMiss.tsv", sep="\t", index=False)
    out.to_csv("Table2_NearMiss.csv", index=False)
    try:
        out.to_excel("Table2_NearMiss.xlsx", index=False)
    except Exception as e:
        print("[WARN] Excel write failed:", e)

    print("[OK] Wrote: Table2_NearMiss.tsv / .csv / .xlsx")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
