#!/usr/bin/env python3
"""
01_clean_phenotypes.py

Clean phenotype tables into a standardized TSV:

Output columns:
- sample_id  (normalized, e.g., PI152651)
- trait columns (numeric)

Supported modes:
  - grain_mold: expects columns like Taxa, IBS, A, M, C, ...
  - tag2019_anthracnose: parses the messy supplementary Excel where the first data row is the header.

Usage examples (Windows CMD):
  python scripts/01_clean_phenotypes.py --input Grain_mold_score.xlsx --out 01_pheno_grain_mold.tsv --mode grain_mold
  python scripts/01_clean_phenotypes.py --input 122_2019_3285_MOESM1_ESM.xlsx --out 01_pheno_anthracnose.tsv --mode tag2019_anthracnose
  python scripts/01_clean_phenotypes.py --input 122_2019_3285_MOESM1_ESM.xlsx --out 01_pheno_anthracnose.tsv --mode tag2019_anthracnose --sheet 0
  python scripts/01_clean_phenotypes.py --input 122_2019_3285_MOESM1_ESM.xlsx --out 01_pheno_anthracnose.tsv --mode tag2019_anthracnose --sheet Sheet1

This script is intentionally conservative: it drops rows that cannot be reliably mapped to a sample_id.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


PI_RE = re.compile(r"^\s*(?:PI)?\s*0*([0-9]+)\s*$", re.IGNORECASE)


def normalize_sample_id(x: object) -> Optional[str]:
    """
    Normalize SAP IDs like:
      'PI152651' -> 'PI152651'
      'PI 152651' -> 'PI152651'
      ' 152651 ' -> 'PI152651'
    Returns None if cannot parse.
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "nan":
        return None
    m = PI_RE.match(s)
    if not m:
        # If user already has a non-PI id, keep it but strip spaces
        # (You can customize this if your panel uses other naming.)
        s2 = re.sub(r"\s+", "", s)
        return s2 if s2 else None
    digits = m.group(1)
    return f"PI{digits}"


def read_grain_mold_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "Taxa" not in df.columns:
        raise ValueError("grain_mold mode expects a 'Taxa' column.")
    df = df.copy()
    df["sample_id"] = df["Taxa"].apply(normalize_sample_id)
    df = df.dropna(subset=["sample_id"])
    df = df.drop(columns=[c for c in ["Taxa"] if c in df.columns])

    # Keep only numeric phenotype columns + sample_id
    keep_cols = ["sample_id"] + [c for c in df.columns if c != "sample_id"]
    df = df[keep_cols]

    # Coerce everything except sample_id to numeric if possible
    for c in df.columns:
        if c == "sample_id":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _resolve_sheet_arg(sheet: Optional[str]) -> object:
    """
    Resolve --sheet arg:
      - None or "" -> 0 (first sheet)
      - "0" -> 0
      - "Sheet1" -> "Sheet1"
    """
    if sheet is None:
        return 0
    s = str(sheet).strip()
    if s == "":
        return 0
    return int(s) if s.isdigit() else s


def read_tag2019_anthracnose_excel(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Parse the TAG 2019 supplementary phenotype table where the first row contains headers like:
      SAP | Acc. Name | PI | subpop | AVG score | #/ave

    Strategy:
      - Read with header=None
      - Ensure we read a SINGLE sheet (pandas returns dict if sheet_name=None)
      - Find the header row that contains 'PI' and 'AVG score' (case-insensitive)
      - Use that row as header
      - Data starts next row
    """
    sheet_to_use = _resolve_sheet_arg(sheet)
    raw = pd.read_excel(path, sheet_name=sheet_to_use, header=None, engine="openpyxl")

    # Safety: if a dict still appears for any reason, pick the first sheet deterministically.
    if isinstance(raw, dict):
        raw = raw[list(raw.keys())[0]]

    if not hasattr(raw, "iloc"):
        raise TypeError(f"Expected DataFrame from read_excel, got: {type(raw)}")

    # Find header row
    header_idx = None
    for i in range(min(80, len(raw))):
        row = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        # Very permissive: find any row that includes 'pi' and something resembling avg score
        if ("pi" in row) and (("avg score" in row) or ("avg" in row) or ("score" in row)):
            header_idx = i
            break
    if header_idx is None:
        # Fallback: try the first non-empty-ish row
        header_idx = 0

    header = raw.iloc[header_idx].tolist()
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = header

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Identify PI column name robustly (allow variants)
    pi_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"pi", "pi#", "pi no", "pi number", "pi number.", "pi number "}:
            pi_col = c
            break
    if pi_col is None:
        # secondary heuristic: column name contains 'pi'
        for c in df.columns:
            if "pi" == str(c).strip().lower().replace(".", ""):
                pi_col = c
                break
    if pi_col is None:
        raise ValueError(
            "Could not find a 'PI' column in the TAG2019 sheet after parsing headers. "
            "Try specifying the correct sheet with --sheet."
        )

    df = df.copy()
    df["sample_id"] = df[pi_col].apply(normalize_sample_id)
    df = df.dropna(subset=["sample_id"])

    # Coerce numeric columns (best-effort)
    for c in df.columns:
        if c in [pi_col, "sample_id"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # Keep sample_id + all other columns except the raw PI column
    out_cols = ["sample_id"] + [c for c in df.columns if c not in [pi_col, "sample_id"]]
    df = df[out_cols]

    # Rename a couple of common columns to stable names
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "avg score":
            rename_map[c] = "anthracnose_avg_score"
        elif cl in {"subpop", "sub-pop", "sub population", "subpopulation"}:
            rename_map[c] = "subpop"
    df = df.rename(columns=rename_map)

    # Coerce the anthracnose score to numeric if present
    if "anthracnose_avg_score" in df.columns:
        df["anthracnose_avg_score"] = pd.to_numeric(df["anthracnose_avg_score"], errors="coerce")

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Excel file (.xlsx)")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--mode", required=True, choices=["grain_mold", "tag2019_anthracnose"], help="Parsing mode")
    ap.add_argument("--sheet", default=None, help="Optional Excel sheet name or index (default: first sheet)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "grain_mold":
        df = read_grain_mold_excel(in_path)
    elif args.mode == "tag2019_anthracnose":
        df = read_tag2019_anthracnose_excel(in_path, sheet=args.sheet)
    else:
        raise RuntimeError("Unsupported mode")

    # Drop duplicate sample_id rows by averaging numeric traits (rare but can happen)
    num_cols = [c for c in df.columns if c != "sample_id" and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) > 0:
        df = df.groupby("sample_id", as_index=False).agg(
            {**{c: "mean" for c in num_cols}, **{c: "first" for c in df.columns if c not in num_cols and c != "sample_id"}}
        )
    else:
        df = df.drop_duplicates(subset=["sample_id"], keep="first")

    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Wrote {len(df):,} rows to: {out_path}")


if __name__ == "__main__":
    main()
