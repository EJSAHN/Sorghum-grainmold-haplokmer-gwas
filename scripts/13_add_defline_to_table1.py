#!/usr/bin/env python3
"""
13_add_defline_to_table1.py (FIXED / ROBUST)

Add functional descriptions from:
  Sbicolor_454_v3.1.1.P14.defline.txt.gz

Key fixes:
- defline file has 3 columns: <transcript> <tag(defLine/pdef)> <text>
  We must use column 3 as the description text.
- Table1 uses locus IDs like 'Sobic.010G250900.v3.2' (locus level),
  while defline uses transcript IDs like 'Sobic.010G250900.1' (transcript level).
  We map locus -> transcript using:
    Sbicolor_454_v3.1.1.locus_transcript_name_map.txt

Input:
  Table1_core_hotspots_top20_flank200kb_withDesc.tsv

Output:
  Table1_core_hotspots_top20_flank200kb_withDEF.tsv
"""

from __future__ import annotations

import re
import gzip
import pandas as pd


# Extract locus like Sobic.010G250900 from anything (including .v3.2)
LOCUS_RE = re.compile(r"(Sobic\.\d+G\d+)")


def extract_locus_id(s: str) -> str:
    m = LOCUS_RE.search(s or "")
    return m.group(1) if m else ""


def load_locus_to_transcript_map(path: str) -> dict:
    """
    Parse Sbicolor_454_v3.1.1.locus_transcript_name_map.txt

    Columns:
      new-locusName  old-locusName  new-transcriptName  old-transcriptName

    We want: locus -> one representative transcript (new-transcriptName).
    """
    m = {}
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            locus = parts[0].strip()
            transcript = parts[2].strip() if len(parts) >= 3 else ""
            if locus and transcript and locus not in m:
                m[locus] = transcript
    return m


def load_deflines(def_gz: str) -> tuple[dict, dict]:
    """
    Parse defline.gz into two dicts:
      - defline_text[transcript] = "similar to ... "
      - pdef_text[transcript]    = "(1 of 1) ..."

    File format (tab-delimited):
      transcript   tag(defLine/pdef)   text...
    """
    defline_text = {}
    pdef_text = {}

    with gzip.open(def_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            tid = parts[0].strip()
            tag = parts[1].strip()
            text = "\t".join(parts[2:]).strip()  # keep any extra tabs as part of text

            if not tid or not tag:
                continue

            if tag == "defLine":
                if tid not in defline_text:
                    defline_text[tid] = text
            elif tag == "pdef":
                if tid not in pdef_text:
                    pdef_text[tid] = text

    return defline_text, pdef_text


def main():
    in_tsv = "Table1_core_hotspots_top20_flank200kb_withDesc.tsv"
    map_txt = "Sbicolor_454_v3.1.1.locus_transcript_name_map.txt"
    def_gz = "Sbicolor_454_v3.1.1.P14.defline.txt.gz"
    out_tsv = "Table1_core_hotspots_top20_flank200kb_withDEF.tsv"

    t = pd.read_csv(in_tsv, sep="\t", keep_default_na=False)

    locus2tx = load_locus_to_transcript_map(map_txt)
    defline_text, pdef_text = load_deflines(def_gz)

    # Add locus and representative transcript for each nearest gene
    t["nearest_locus"] = t["nearest_gene_id"].map(lambda x: extract_locus_id(x))
    t["nearest_transcript"] = t["nearest_locus"].map(lambda x: locus2tx.get(x, ""))

    # Prefer defLine; fallback to pdef
    def get_defline(tx: str) -> str:
        if not tx:
            return ""
        if tx in defline_text and defline_text[tx]:
            return defline_text[tx]
        if tx in pdef_text and pdef_text[tx]:
            return pdef_text[tx]
        return ""

    def get_pdef(tx: str) -> str:
        if not tx:
            return ""
        return pdef_text.get(tx, "")

    t["nearest_gene_defline"] = t["nearest_transcript"].map(get_defline)
    t["nearest_gene_pdef"] = t["nearest_transcript"].map(get_pdef)

    t.to_csv(out_tsv, sep="\t", index=False)

    nonempty = int((t["nearest_gene_defline"].astype(str).str.len() > 0).sum())
    print(f"[OK] Wrote: {out_tsv}")
    print(f"[INFO] Deflines filled: {nonempty}/{len(t)}")
    print(t[["nearest_gene_id", "nearest_transcript", "nearest_gene_defline"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
