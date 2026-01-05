#!/usr/bin/env python3
"""
62_make_table1_strict_core_p1e4.py

Build the *final* Table 1 under the strict convergence rule:
- marker inclusion: p <= 1e-4
- binning: 250 kb
- core: >=3 traits AND both methods (SNP + Haplo-kmer)
- ranking: hotspot strength = sum(-log10(p)) within bin (across all files)

Outputs:
- Table1_Final_StrictCore.tsv
- Table1_Final_StrictCore.csv
- Table1_Final_StrictCore.xlsx

Notes:
- This version is designed to include the chr5 core locus (60.25–60.50 Mb) by construction.
- If you want "Top 20", it will return up to 20 rows; if only 1 core exists, you'll get 1 row (that’s fine).
"""

from __future__ import annotations
import argparse
import glob
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd
import gzip

# ---- parameters (defaults match your pipeline) ----
P_THRESH = 1e-4
WINDOW_BP = 250_000
CORE_TRAITS = 3
CORE_METHODS = 2  # SNP + HKMER
TOPN_OUT = 20

CHR5 = "5"
CORE_START = 60250000
CORE_END   = 60499999

# annotation resources (optional)
GFF = Path("Sbicolor_454_v3.1.1.gene.gff3.gz")
ANNOT = Path("Sbicolor_454_v3.1.1.P14.annotation_info.txt.gz")

# outputs
OUT_TSV = Path("Table1_Final_StrictCore.tsv")
OUT_CSV = Path("Table1_Final_StrictCore.csv")
OUT_XLSX = Path("Table1_Final_StrictCore.xlsx")


def load_gwas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer")
    cols = df.columns.tolist()
    low = {c.lower(): c for c in cols}

    chrom_col = low.get("chrom")
    p_col = low.get("p")
    if chrom_col is None or p_col is None:
        raise ValueError(f"{path}: missing chrom/p")

    # SNP or HKMER position logic
    if "pos" in low:
        pos_col = low["pos"]
        df["pos"] = pd.to_numeric(df[pos_col], errors="coerce")
    else:
        # haplokmer window
        ws = low.get("window_start_pos") or low.get("window_start") or low.get("start_pos") or low.get("start")
        we = low.get("window_end_pos") or low.get("window_end") or low.get("end_pos") or low.get("end")
        if ws is None:
            raise ValueError(f"{path}: missing pos/window_start_pos")
        s = pd.to_numeric(df[ws], errors="coerce")
        if we is not None:
            e = pd.to_numeric(df[we], errors="coerce")
            df["pos"] = ((s + e) / 2.0)
        else:
            df["pos"] = s

    df["chrom"] = df[chrom_col].astype(str)
    df["p"] = pd.to_numeric(df[p_col], errors="coerce")
    df = df.dropna(subset=["chrom", "pos", "p"])
    df = df[(df["p"] > 0) & (df["p"] <= 1)]
    df["pos"] = df["pos"].astype(int)

    # label method/trait from filename
    fn = Path(path).name
    method = "SNP" if "_snp_" in fn else ("HKMER" if "_haplokmer_" in fn else "UNK")
    trait = fn
    trait = trait.replace("05_gwas_snp_", "").replace("06_gwas_haplokmer_", "")
    trait = trait.replace(".tsv.gz", "").replace(".tsv", "")
    trait = trait.replace("_AminusC", ":A-C").replace("_MminusC", ":M-C")
    trait = trait.replace("_A", ":A").replace("_M", ":M").replace("_C", ":C")
    trait = trait.replace("grainmold_", "grainmold").replace("anthracnose", "anthracnose")
    trait = re.sub(r"_k\d+$", "", trait)

    df["method"] = method
    df["trait"] = trait
    df["source_file"] = fn
    return df


# ---- gene annotation (nearest gene within flank) ----
def open_textmaybe_gz(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def load_genes(gff: Path):
    genes = {}
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
            if not gid:
                continue
            genes.setdefault(str(chrom), []).append((s, e, gid, name))
    for c in genes:
        genes[c].sort(key=lambda x: x[0])
    return genes


def load_annotation_map(path: Path):
    amap = {}
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


def resolve_chrom_key(genes_by_chrom, chrom: str):
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


def main():
    files = sorted(glob.glob("05_gwas_snp_*.tsv.gz") + glob.glob("06_gwas_haplokmer_*.tsv.gz"))
    if not files:
        raise SystemExit("[ERROR] No GWAS files found (05_gwas_snp_*.tsv.gz / 06_gwas_haplokmer_*.tsv.gz).")

    # load + filter p
    parts = []
    for fp in files:
        df = load_gwas(fp)
        df = df[df["p"] <= P_THRESH].copy()
        if len(df):
            parts.append(df)

    if not parts:
        raise SystemExit("[ERROR] No markers pass p<=1e-4 in any GWAS file.")

    hits = pd.concat(parts, ignore_index=True)
    hits["bin"] = (hits["pos"] // WINDOW_BP).astype(int)
    hits["bin_start"] = hits["bin"] * WINDOW_BP
    hits["bin_end"] = hits["bin_start"] + WINDOW_BP - 1

    # strength per marker = -log10(p)
    hits["strength"] = -np.log10(np.clip(hits["p"].to_numpy(float), np.nextafter(0, 1), 1.0))

    # summarize per bin
    def uniq_join(x): return ";".join(sorted(set(x.astype(str))))

    agg = hits.groupby(["chrom", "bin", "bin_start", "bin_end"], as_index=False).agg(
        strength_sum=("strength", "sum"),
        best_p=("p", "min"),
        n_hits=("p", "size"),
        traits=("trait", uniq_join),
        methods=("method", uniq_join),
    )
    agg["n_traits"] = agg["traits"].map(lambda s: 0 if not s else len(s.split(";")))
    agg["n_methods"] = agg["methods"].map(lambda s: 0 if not s else len(s.split(";")))

    # core filter
    core = agg[(agg["n_traits"] >= CORE_TRAITS) & (agg["n_methods"] >= CORE_METHODS)].copy()
    if core.empty:
        raise SystemExit("[ERROR] No strict core bins found under p<=1e-4 (>=3 traits & both methods).")

    # lead marker per bin: the hit with min p
    hits_sorted = hits.sort_values("p").drop_duplicates(subset=["chrom", "bin"], keep="first")
    core = core.merge(hits_sorted[["chrom","bin","pos","p","source_file","method","trait"]], on=["chrom","bin"], how="left")
    core = core.rename(columns={"pos":"lead_pos","p":"lead_p","method":"lead_method","trait":"lead_trait","source_file":"lead_source_file"})

    # rank by strength
    core = core.sort_values(["strength_sum","best_p"], ascending=[False, True]).reset_index(drop=True)

    # annotation (optional)
    genes = load_genes(GFF)
    ann = load_annotation_map(ANNOT)
    gids=[]; gnames=[]; gdescs=[]
    for _, r in core.iterrows():
        gid, gname, gdesc = nearest_gene(genes, ann, str(r["chrom"]), int(r["lead_pos"]), flank=200_000)
        gids.append(gid); gnames.append(gname); gdescs.append(gdesc)
    core["nearest_gene_id"]=gids
    core["nearest_gene_name"]=gnames
    core["nearest_gene_desc"]=gdescs

    # Flag chr5 canonical core
    core["is_chr5_core"] = (core["chrom"].astype(str) == CHR5) & (core["bin_start"].astype(int) == CORE_START) & (core["bin_end"].astype(int) == CORE_END)

    # Output (top 20, but keep all cores in file too if you want later)
    out = core.head(TOPN_OUT).copy()

    # Pretty columns for Table 1
    out_tab = pd.DataFrame({
        "Chr": out["chrom"].astype(str).map(lambda x: f"{x}★" if False else x),
        "Interval (Mb)": (out["bin_start"]/1e6).map(lambda x: f"{x:.2f}") + "–" + (out["bin_end"]/1e6).map(lambda x: f"{x:.2f}"),
        "Hotspot strength (Σ−log10 p)": out["strength_sum"].map(lambda x: f"{x:.2f}"),
        "Peak p-value": out["best_p"].map(lambda x: f"{x:.2e}"),
        "# hits": out["n_hits"].astype(int),
        "# traits": out["n_traits"].astype(int),
        "Traits": out["traits"],
        "Methods": out["methods"].str.replace("HKMER", "Haplo-kmer").str.replace(";", " + "),
        "Lead position (bp)": out["lead_pos"].astype(int),
        "Lead trait": out["lead_trait"],
        "Lead method": out["lead_method"].str.replace("HKMER", "Haplo-kmer"),
        "Nearest gene": out["nearest_gene_id"].replace("", "-"),
        "Gene note": out["nearest_gene_desc"].replace("", "-"),
    })

    # put chr5 core first if present
    # (reorder without changing content)
    if out["is_chr5_core"].any():
        idx = out["is_chr5_core"].to_numpy().nonzero()[0][0]
        # move that row to top
        chr5_row = out_tab.iloc[[idx]]
        rest = out_tab.drop(out_tab.index[idx])
        out_tab = pd.concat([chr5_row, rest], ignore_index=True)
        # add star to Chr
        out_tab.loc[0, "Chr"] = str(out_tab.loc[0, "Chr"]) + "★"

    out_tab.to_csv(OUT_TSV, sep="\t", index=False)
    out_tab.to_csv(OUT_CSV, index=False)
    try:
        out_tab.to_excel(OUT_XLSX, index=False)
    except Exception as e:
        print("[WARN] Excel write failed:", e)

    print(f"[OK] Wrote: {OUT_TSV}, {OUT_CSV}, {OUT_XLSX}")
    print(out_tab.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
