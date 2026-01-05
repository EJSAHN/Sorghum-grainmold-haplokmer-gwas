#!/usr/bin/env python3
"""
61_make_table1_main_slim.py

Make a Word-friendly (narrow) main Table 1 from Table1_Publication_Ready.xlsx.

Outputs:
- Table1_Main_Slim.tsv  (best for Word: Insert > Table > Convert Text to Table)
- Table1_Main_Slim.csv  (for Excel)

Rules:
- Convert Start/End bp -> Mb range (e.g., 60.25–60.50)
- Shorten lead marker (use middle token when haplo marker contains '|')
- Shorten gene annotation (keep informative tail; limit length)
- Mark the chr5 core bin (60.25–60.50 Mb) with a star ★ in Chr column
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

BASE = Path(".").resolve()
IN_XLSX = BASE / "Table1_Publication_Ready.xlsx"

OUT_TSV = BASE / "Table1_Main_Slim.tsv"
OUT_CSV = BASE / "Table1_Main_Slim.csv"

CORE_CHR = 5
CORE_START = 60250000
CORE_END = 60499999

LOCUS_RE = re.compile(r"(Sobic\.\d+G\d+)")


def mb_range(start_bp: int, end_bp: int) -> str:
    return f"{start_bp/1e6:.2f}\u2013{end_bp/1e6:.2f}"  # en-dash


def shorten_lead(marker: str) -> str:
    s = str(marker or "").strip()
    if "|" in s:
        parts = s.split("|")
        # for haplo markers: use the middle segment like S5_...-S5_...
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    return s


def strip_gene_version(gene: str) -> str:
    s = str(gene or "").strip()
    m = LOCUS_RE.search(s)
    return m.group(1) if m else s


def shorten_defline(desc: str, maxlen: int = 85) -> str:
    s = str(desc or "").strip()
    if not s:
        return "-"
    # remove leading "defLine " if present
    s = re.sub(r"^\s*defLine\s+", "", s, flags=re.IGNORECASE).strip()
    # if pdef-like "(1 of X) ... - SOMETHING", keep right side after first " - "
    if " - " in s:
        s = s.split(" - ", 1)[1].strip()
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    # truncate
    if len(s) > maxlen:
        s = s[: maxlen - 1].rstrip() + "…"
    return s


def main():
    if not IN_XLSX.exists():
        raise FileNotFoundError(f"Missing input: {IN_XLSX}")

    df = pd.read_excel(IN_XLSX)

    # Expected columns (from your file)
    required = ["Chr", "Start (bp)", "End (bp)", "Peak p-value",
                "# Traits", "Associated traits", "Supported methods",
                "Lead marker", "Nearest gene (locus)", "Gene annotation (defline)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {IN_XLSX.name}. Found: {df.columns.tolist()}")

    out = pd.DataFrame()
    out["Chr"] = df["Chr"].astype(int).astype(str)
    out["Interval (Mb)"] = [mb_range(int(a), int(b)) for a, b in zip(df["Start (bp)"], df["End (bp)"])]
    out["Peak p"] = pd.to_numeric(df["Peak p-value"], errors="coerce").map(lambda x: f"{x:.2e}" if pd.notna(x) else "")
    out["#Traits"] = df["# Traits"].astype(int)
    out["Traits"] = df["Associated traits"].astype(str)
    out["Methods"] = df["Supported methods"].astype(str)
    out["Lead marker"] = df["Lead marker"].map(shorten_lead)
    out["Nearest gene"] = df["Nearest gene (locus)"].map(strip_gene_version)
    out["Annotation (short)"] = df["Gene annotation (defline)"].map(shorten_defline)

    # Mark the chr5 core interval with ★
    mask_core = (df["Chr"].astype(int) == CORE_CHR) & (df["Start (bp)"].astype(int) == CORE_START) & (df["End (bp)"].astype(int) == CORE_END)
    out.loc[mask_core, "Chr"] = out.loc[mask_core, "Chr"] + "★"

    out.to_csv(OUT_TSV, sep="\t", index=False)
    out.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote: {OUT_TSV.name}")
    print(f"[OK] Wrote: {OUT_CSV.name}")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
