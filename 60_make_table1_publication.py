#!/usr/bin/env python3
"""
60_make_table1_publication.py

Build a publication-ready Table 1 from the "core hotspots top20" table and
properly attach Sorghum P14 defline descriptions (NOT just 'defLine'/'pdef' tags).

Outputs:
- Table1_Publication_Ready.tsv
- Table1_Publication_Ready.csv
- Table1_Publication_Ready.xlsx

Run (from project root):
  python scripts/60_make_table1_publication.py
"""

from __future__ import annotations
import gzip
import re
from pathlib import Path
import pandas as pd

BASE = Path(".").resolve()

# --- inputs (edit only if your filenames differ) ---
IN_TABLE_CANDIDATES = [
    BASE / "Table1_core_hotspots_top20_flank200kb_withDEF.tsv",
    BASE / "Table1_core_hotspots_top20_flank200kb_withDesc.tsv",
    BASE / "Table1_core_hotspots_top20_flank200kb.tsv",
]
DEFLINE_GZ = BASE / "Sbicolor_454_v3.1.1.P14.defline.txt.gz"
LOCUS_MAP = BASE / "Sbicolor_454_v3.1.1.locus_transcript_name_map.txt"

OUT_TSV = BASE / "Table1_Publication_Ready.tsv"
OUT_CSV = BASE / "Table1_Publication_Ready.csv"
OUT_XLSX = BASE / "Table1_Publication_Ready.xlsx"

LOCUS_RE = re.compile(r"(Sobic\.\d+G\d+)")


def pick_input_table() -> Path:
    for p in IN_TABLE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No Table1 core-top20 TSV found among candidates.")


def load_locus_to_transcript_map(path: Path) -> dict[str, str]:
    """
    Map locus (Sobic.###G#####) -> a representative transcript (Sobic.###G#####.1).
    Chooses the first transcript encountered per locus.
    """
    mp: dict[str, str] = {}
    if not path.exists():
        return mp
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            locus = parts[0].strip()
            transcript = parts[2].strip()
            if locus and transcript and locus not in mp:
                mp[locus] = transcript
    return mp


def load_defline(path_gz: Path) -> dict[str, str]:
    """
    defline file format (as previewed):
      transcript<TAB>tag<TAB>description
    Example:
      Sobic.001G000200.1    defLine  similar to ...
      Sobic.001G000200.1    pdef     (1 of 1) ...
    We prefer defLine; if missing, use pdef.
    Returns: transcript -> chosen description string
    """
    if not path_gz.exists():
        raise FileNotFoundError(f"Missing defline gz: {path_gz}")

    tmp: dict[str, dict[str, str]] = {}
    with gzip.open(path_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            tx = parts[0].strip()
            tag = parts[1].strip()
            desc = parts[2].strip()
            if not tx:
                continue
            tmp.setdefault(tx, {})
            # keep first occurrence per tag
            if tag not in tmp[tx]:
                tmp[tx][tag] = desc

    out: dict[str, str] = {}
    for tx, d in tmp.items():
        if "defLine" in d:
            out[tx] = d["defLine"]
        elif "pdef" in d:
            out[tx] = d["pdef"]
        else:
            # fallback: any tag
            out[tx] = next(iter(d.values()))
    return out


def clean_traits(s: str) -> str:
    s = (s or "")
    s = s.replace("grainmold:", "")
    s = s.replace("anthracnose", "Anthracnose")
    s = s.replace(";", ", ")
    s = s.replace("AminusC", "A–C").replace("MminusC", "M–C")
    return s


def clean_methods(s: str) -> str:
    s = (s or "")
    s = s.replace("HKMER_k7", "Haplo-kmer")
    s = s.replace("HKMER", "Haplo-kmer")
    s = s.replace(";", " + ")
    return s


def main():
    in_table = pick_input_table()
    print(f"[INFO] Using input table: {in_table.name}")

    df = pd.read_csv(in_table, sep="\t", keep_default_na=False)

    # Ensure gene locus exists
    if "nearest_gene_id" not in df.columns:
        raise ValueError("Input Table1 TSV must contain 'nearest_gene_id' column.")

    # Extract locus from nearest_gene_id (e.g., Sobic.010G250900.v3.2 -> Sobic.010G250900)
    df["nearest_locus"] = df["nearest_gene_id"].apply(
        lambda x: (LOCUS_RE.search(x).group(1) if LOCUS_RE.search(x) else "")
    )

    locus2tx = load_locus_to_transcript_map(LOCUS_MAP)
    defline = load_defline(DEFLINE_GZ)

    # Representative transcript
    df["nearest_transcript"] = df["nearest_locus"].map(lambda l: locus2tx.get(l, ""))

    # Attach defline description via transcript
    df["nearest_gene_defline_desc"] = df["nearest_transcript"].map(lambda tx: defline.get(tx, ""))

    # If transcript missing, try any transcript with same locus prefix
    # (light fallback: match defline keys starting with locus + ".")
    defline_keys = list(defline.keys())
    def fallback_desc(locus: str) -> str:
        if not locus:
            return ""
        prefix = locus + "."
        for k in defline_keys:
            if k.startswith(prefix):
                return defline[k]
        return ""

    df.loc[df["nearest_gene_defline_desc"] == "", "nearest_gene_defline_desc"] = df["nearest_locus"].apply(fallback_desc)

    # Publication columns
    cols = []
    for c in ["chrom", "window_start", "window_end", "best_p", "n_traits", "n_methods", "traits", "methods", "lead_marker",
              "nearest_gene_id", "nearest_transcript", "nearest_gene_defline_desc"]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()

    # Formatting
    if "best_p" in out.columns:
        out["best_p"] = pd.to_numeric(out["best_p"], errors="coerce")
        out["best_p"] = out["best_p"].map(lambda x: f"{x:.2e}" if pd.notna(x) else "")

    if "traits" in out.columns:
        out["traits"] = out["traits"].map(clean_traits)
    if "methods" in out.columns:
        out["methods"] = out["methods"].map(clean_methods)

    out = out.rename(columns={
        "chrom": "Chr",
        "window_start": "Start (bp)",
        "window_end": "End (bp)",
        "best_p": "Peak p-value",
        "n_traits": "# Traits",
        "n_methods": "# Methods",
        "traits": "Associated traits",
        "methods": "Supported methods",
        "lead_marker": "Lead marker",
        "nearest_gene_id": "Nearest gene (locus)",
        "nearest_transcript": "Nearest transcript",
        "nearest_gene_defline_desc": "Gene annotation (defline)",
    })

    out.to_csv(OUT_TSV, sep="\t", index=False)
    out.to_csv(OUT_CSV, index=False)

    try:
        out.to_excel(OUT_XLSX, index=False)
    except Exception as e:
        print(f"[WARN] Could not write xlsx (openpyxl missing?). CSV/TSV still written. Error: {e}")

    filled = (out["Gene annotation (defline)"] != "").sum() if "Gene annotation (defline)" in out.columns else 0
    print(f"[OK] Wrote: {OUT_TSV.name}, {OUT_CSV.name}, {OUT_XLSX.name}")
    print(f"[INFO] Defline descriptions filled: {filled}/{len(out)}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
