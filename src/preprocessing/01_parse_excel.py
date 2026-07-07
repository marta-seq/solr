"""
01_parse_excel.py
Reads the master Excel file from data/data_curated/,
backs it up if not already backed up today,
cleans and normalizes the data,
and outputs methods.csv and datasets.csv to data/processed/.

Usage:
    python src/preprocessing/01_parse_excel.py
"""

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CURATED_DIR   = ROOT / "data" / "data_curated"
BACKUP_DIR    = ROOT / "data" / "data_curated_backup"
PROCESSED_DIR = ROOT / "data" / "processed"

xlsx_files = list(CURATED_DIR.glob("*.xlsx"))
if not xlsx_files:
    raise FileNotFoundError(f"No .xlsx file found in {CURATED_DIR}")
SOURCE_FILE = xlsx_files[0]

# ── Backup ───────────────────────────────────────────────────────────────────
def backup_if_needed(source: Path, backup_dir: Path) -> None:
    today = date.today().strftime("%Y_%m_%d")
    backup_name = f"{source.stem}_{today}{source.suffix}"
    backup_path = backup_dir / backup_name
    if not backup_path.exists():
        shutil.copy2(source, backup_path)
        print(f"Backed up to {backup_path}")
    else:
        print(f"Backup already exists for today: {backup_path}")

# ── DOI normalisation ────────────────────────────────────────────────────────
def normalize_doi(doi) -> str:
    if pd.isna(doi):
        return ""
    doi = str(doi).strip()
    if doi in ("", "NA", "na", "nan"):
        return ""
    if doi.startswith("https://doi.org/"):
        return doi
    if doi.startswith("doi.org/"):
        return "https://" + doi
    if doi.startswith("10."):
        return "https://doi.org/" + doi
    if doi.startswith("http://doi.org/"):
        return doi.replace("http://", "https://")
    return doi

# ── ID validation ─────────────────────────────────────────────────────────────
def is_valid_id(val) -> bool:
    val = str(val).strip()
    if val in ("", "nan", "NA", "na", "None", "none"):
        return False
    if val.startswith("#"):
        return False
    if val.lower().startswith("placeholder"):
        return False
    if not val[0].isalpha():
        return False
    return True

# ── Find header row ───────────────────────────────────────────────────────────
def find_header_row(df: pd.DataFrame, marker: str) -> int:
    for i, row in df.iterrows():
        if any(str(c).strip() == marker for c in row):
            return i
    raise ValueError(f"Could not find header row containing '{marker}'")

# ── Drop junk columns ─────────────────────────────────────────────────────────
JUNK_COLS = {"nan", "", "in future add metadata on markers", "notes"}

def drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if str(c).strip().lower() not in JUNK_COLS]
    return df[keep]

# ── Parse papers sheet ───────────────────────────────────────────────────────
def parse_papers(xl: pd.ExcelFile) -> pd.DataFrame:
    df = xl.parse("papers", header=None)

    header_row = find_header_row(df, "P_ENTRY_ID")
    df.columns = df.iloc[header_row].astype(str).str.strip()
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    df = drop_junk_columns(df)
    df = df.rename(columns={"P_ENTRY_ID": "entry_id"})

    mask = df["entry_id"].apply(is_valid_id)
    df = df[mask].reset_index(drop=True)

    # Normalize DOI
    for col in df.columns:
        if str(col).upper() == "DOI":
            df[col] = df[col].apply(normalize_doi)
            break

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    df["is_placeholder"] = df["category"].astype(str).str.contains(
        "Placeholder", case=False, na=False
    )

    return df

# ── Parse dataset sheets ──────────────────────────────────────────────────────
def parse_dataset_sheet(xl: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = xl.parse(sheet_name, header=None)

    header_row = find_header_row(df, "P_ENTRY_ID")
    df.columns = df.iloc[header_row].astype(str).str.strip()
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    df = drop_junk_columns(df)
    df = df.rename(columns={"P_ENTRY_ID": "entry_id"})

    mask = df["entry_id"].apply(is_valid_id)
    df = df[mask].reset_index(drop=True)

    # Normalize DOI columns
    for col in df.columns:
        if "DOI" in str(col).upper():
            df[col] = df[col].apply(normalize_doi)

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Reading {SOURCE_FILE.name} ...")
    backup_if_needed(SOURCE_FILE, BACKUP_DIR)

    xl = pd.ExcelFile(SOURCE_FILE)
    print(f"Sheets found: {xl.sheet_names}")

    # Papers
    papers = parse_papers(xl)
    print(f"  Papers: {len(papers)} entries ({papers['is_placeholder'].sum()} placeholders)")

    # Datasets
    ds_sp = parse_dataset_sheet(xl, "Data_SP")
    print(f"  Data_SP: {len(ds_sp)} entries")

    ds_st = parse_dataset_sheet(xl, "Data_ST")
    print(f"  Data_ST: {len(ds_st)} entries")

    datasets = pd.concat([ds_sp, ds_st], ignore_index=True, sort=False)
    print(f"  Datasets combined: {len(datasets)} entries")

    # Output filenames derived from source filename
    stem   = SOURCE_FILE.stem
    suffix = "_".join(stem.split("_")[2:])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    papers_out   = PROCESSED_DIR / f"methods_{suffix}.csv"
    datasets_out = PROCESSED_DIR / f"datasets_{suffix}.csv"

    papers.to_csv(papers_out, index=False)
    datasets.to_csv(datasets_out, index=False)

    print(f"\nOutputs written:")
    print(f"  {papers_out}")
    print(f"  {datasets_out}")

if __name__ == "__main__":
    main()