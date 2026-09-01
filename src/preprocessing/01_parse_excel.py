"""
01_parse_excel.py
Reads the master Excel file from data/data_curated/,
backs it up if not already backed up today,
cleans and normalizes the data,
and outputs methods.csv and datasets.csv to data/processed/.

Schema (as of the 2026-07 method_pub/AP_pub/data rename):
    - "method_pub": computational-method papers (was "papers"). Header row 1.
    - "AP_pub":     application papers (new sheet, split out of "papers").
                    Header row 2 (row 1 is a merged-cell section-title row).
    - "data":       all dataset entries, replacing the old Data_SP/Data_ST/
                    Data_multi three-sheet split. Header row 2.
    method_pub and AP_pub are combined into one methods output, tagged with
    a "paper_type" column ("method" / "application") so downstream code can
    tell them apart without re-deriving it from `category` every time.

Header row position is NOT assumed fixed - every sheet is scanned for the
row containing the "P_ENTRY_ID" marker, same approach as before. This is
deliberate: method_pub and AP_pub/data currently have their header on
different rows, and that's exactly the kind of drift that broke this
script last time the sheets were renamed.

Usage:
    python src/preprocessing/01_parse_excel.py
"""

import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from category_maps import (
    CATEGORY_MAP_METHOD_PUB,
    CATEGORY_MAP_AP_PUB,
    PIPELINE_CATEGORY_MAP,
    SPATIAL_DATA_CATEGORY_MAP_METHOD_PUB,
    SPATIAL_DATA_CATEGORY_MAP_DATA,
    REVIEW_STATUS_MAP,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CURATED_DIR   = ROOT / "data" / "data_curated"
BACKUP_DIR    = ROOT / "data" / "data_curated_backup"
PROCESSED_DIR = ROOT / "data" / "processed"

# Exclude "autoreview" outputs from merge_candidates.py - those are pending
# manual review/merge, not the finalized curated file this script should run
# against (see the handoff cycle: run_pipeline -> staging -> merge_candidates
# -> manual review of the autoreview file -> rename (drop "autoreview") ->
# re-run 01/02/03). Also excludes Excel's own "~$filename.xlsx" lock/temp
# files (created while the real file is open, or left behind by an unclean
# close) - these are NOT data and glob("*.xlsx") happily matches them too.
#
# Deliberately NOT picking "most recently modified" when more than one
# candidate remains: right after a fresh `git pull`/clone, every file in the
# working tree gets its mtime reset to checkout time, so mtime-based
# tie-breaking becomes arbitrary rather than actually reflecting recency
# (caught on gaia: a stale "~$..." lock file that should have been excluded
# ended up looking "newest"). If there's genuine ambiguity, fail loudly and
# say exactly what to do about it, rather than silently guessing.
_candidates = [f for f in CURATED_DIR.glob("*.xlsx")
               if "autoreview" not in f.name.lower() and not f.name.startswith("~$")]
if not _candidates:
    raise FileNotFoundError(
        f"No usable .xlsx file found in {CURATED_DIR}. "
        f"(Files containing 'autoreview', or starting with '~$' (Excel lock files), are skipped.)"
    )
if len(_candidates) > 1:
    names = ", ".join(f.name for f in _candidates)
    raise FileNotFoundError(
        f"Found {len(_candidates)} candidate curated files in {CURATED_DIR}, can't tell which "
        f"is current: {names}. Move the stale one(s) into {BACKUP_DIR} and leave exactly one "
        f"non-autoreview .xlsx in {CURATED_DIR} before re-running."
    )
SOURCE_FILE = _candidates[0]

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

# ── Category/status normalization (see category_maps.py) ────────────────────
def _normalize_or_warn(value, mapping: dict, field_name: str, sheet_name: str, entry_id: str) -> str:
    """Whole-cell lookup against one of the maps in category_maps.py.

    A value in the map that maps to None means "deliberately left unset for
    now" (a real decision, not a gap) and becomes "". A value NOT in the map
    at all means the sheet has a raw value nobody has looked at yet - that's
    printed loudly rather than guessed at or silently dropped, same
    philosophy as the suspicious-DOI check above.
    """
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    if not raw or raw.lower() in ("na", "nan", "none"):
        return ""
    if raw in mapping:
        mapped = mapping[raw]
        return "" if mapped is None else mapped
    print(f"  WARNING [{sheet_name}]: unmapped {field_name} value {raw!r} "
          f"(entry_id: {entry_id}) - not in category_maps.py, passing through "
          f"unchanged. Add it to the mapping once you know what it should be.")
    return raw

def _looks_like_doi_or_url(val: str) -> bool:
    """Loose sanity check, NOT enforcement - just used to print a heads-up
    when a DOI-ish column holds something that clearly isn't a DOI/URL (e.g.
    a loose note like 'To delete' or 'check this'), so it's visible in the
    console instead of silently propagating into Crossref lookups later."""
    if not val:
        return True  # empty is fine, nothing to flag
    return val.startswith("http") or val.startswith("10.") or val.startswith("doi.org/")

# ── ID validation ─────────────────────────────────────────────────────────────
import re as _re
# Allows dots too - some dataset IDs are deliberately sub-numbered
# (e.g. "D_MULTI_12.1", "D_MULTI_12.2" for a multi-part dataset).
_ID_SHAPE = _re.compile(r"^[A-Za-z][A-Za-z0-9_.]*$")

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
    # Real IDs (M_PR_1, D_SP_IMC_1, AP_14, M_AUTO_5, ...) are short alnum/
    # underscore tokens with no spaces. This rejects free-text scratch notes
    # that sometimes end up typed straight into the ID column (caught on
    # real data: two rows in method_pub had full sentences as their
    # "P_ENTRY_ID", which would otherwise have been ingested as bogus entries).
    if not _ID_SHAPE.match(val) or len(val) > 40:
        return False
    return True

# ── Find header row ───────────────────────────────────────────────────────────
def find_header_row(df: pd.DataFrame, marker: str) -> int:
    for i, row in df.iterrows():
        if any(str(c).strip() == marker for c in row):
            return i
    raise ValueError(f"Could not find header row containing '{marker}'")

# ── Drop junk columns ─────────────────────────────────────────────────────────
# NOTE: "notes"/"Notes" is deliberately in here - free-text curation notes
# (including Marta's own in-progress questions/TODOs left in cells) are kept
# OUT of the processed CSVs/frontend on purpose, same as before the schema
# rename. They still live in the source Excel for manual review.
JUNK_COLS = {"nan", "", "in future add metadata on markers", "notes"}

def drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if str(c).strip().lower() not in JUNK_COLS]
    return df[keep]

# Canonical renames for headers that gained trailing text/whitespace in the
# new schema but should still line up with the column names 02_fetch_metadata.py
# and 03_export_json.py already expect, so those two scripts don't also need
# touching every time a header's wording drifts slightly.
CANONICAL_RENAMES = {
    "Resolution and also add dimensions": "Resolution",
    "N markers (proteins/genes …)": "N markers",
    # method_pub only. Was silently carried through under its raw header
    # (unused downstream); giving it a stable name so 03_export_json.py can
    # expose it as part of the source_type filter (peer-reviewed/bioRxiv/
    # arXiv/etc, added 2026-09-01).
    "arxiv/bioarxiv/peer reviewed": "source_type_manual",
}

def apply_canonical_renames(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: CANONICAL_RENAMES[str(c).strip()]
                  for c in df.columns if str(c).strip() in CANONICAL_RENAMES}
    return df.rename(columns=rename_map)

# ── Parse a "papers"-shaped sheet (method_pub or AP_pub) ─────────────────────
def parse_pub_sheet(xl: pd.ExcelFile, sheet_name: str, paper_type: str) -> pd.DataFrame:
    df = xl.parse(sheet_name, header=None)

    header_row = find_header_row(df, "P_ENTRY_ID")
    df.columns = df.iloc[header_row].astype(str).str.strip()
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    df = drop_junk_columns(df)
    df = apply_canonical_renames(df)
    df = df.rename(columns={"P_ENTRY_ID": "entry_id"})

    mask = df["entry_id"].apply(is_valid_id)
    df = df[mask].reset_index(drop=True)

    # Normalize DOI, with a lightweight sanity check
    for col in df.columns:
        if str(col).upper() == "DOI":
            suspicious = df[col].apply(
                lambda v: bool(str(v).strip()) and not _looks_like_doi_or_url(str(v).strip())
            )
            if suspicious.any():
                bad_ids = df.loc[suspicious, "entry_id"].tolist()
                print(f"  WARNING [{sheet_name}]: {suspicious.sum()} row(s) have a DOI value "
                      f"that doesn't look like a DOI/URL (entry_id(s): {bad_ids}) - "
                      f"likely a loose note left in the DOI cell, worth a manual check.")
            df[col] = df[col].apply(normalize_doi)
            break

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # is_placeholder must be computed from the RAW category text, before
    # normalization below rewrites "Placeholder - method" -> "Placeholder"
    # (still matches, but keeping this first is clearer/order-independent).
    df["is_placeholder"] = df["category"].astype(str).str.contains(
        "Placeholder", case=False, na=False
    )
    df["paper_type"] = paper_type

    # Normalize category / pipeline_category / spatial_data_category /
    # REVIEW_STATUS against the whole-cell maps in category_maps.py (see
    # that file for the reasoning behind each mapping). method_pub and
    # AP_pub use different category vocabularies; pipeline_category and
    # spatial_data_category only exist on method_pub.
    category_map = CATEGORY_MAP_METHOD_PUB if paper_type == "method" else CATEGORY_MAP_AP_PUB
    if "category" in df.columns:
        df["category"] = df.apply(
            lambda r: _normalize_or_warn(r["category"], category_map, "category", sheet_name, r["entry_id"]),
            axis=1,
        )
    if "pipeline_category" in df.columns:
        df["pipeline_category"] = df.apply(
            lambda r: _normalize_or_warn(r["pipeline_category"], PIPELINE_CATEGORY_MAP, "pipeline_category", sheet_name, r["entry_id"]),
            axis=1,
        )
    if "spatial_data_category" in df.columns:
        df["spatial_data_category"] = df.apply(
            lambda r: _normalize_or_warn(r["spatial_data_category"], SPATIAL_DATA_CATEGORY_MAP_METHOD_PUB, "spatial_data_category", sheet_name, r["entry_id"]),
            axis=1,
        )
    if "REVIEW_STATUS" in df.columns:
        df["REVIEW_STATUS"] = df.apply(
            lambda r: _normalize_or_warn(r["REVIEW_STATUS"], REVIEW_STATUS_MAP, "REVIEW_STATUS", sheet_name, r["entry_id"]),
            axis=1,
        )

    return df

# ── Parse the "data" sheet ────────────────────────────────────────────────────
def parse_dataset_sheet(xl: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    df = xl.parse(sheet_name, header=None)

    header_row = find_header_row(df, "P_ENTRY_ID")
    df.columns = df.iloc[header_row].astype(str).str.strip()
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    df = drop_junk_columns(df)
    df = apply_canonical_renames(df)
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

    if "spatial_data_category" in df.columns:
        df["spatial_data_category"] = df.apply(
            lambda r: _normalize_or_warn(r["spatial_data_category"], SPATIAL_DATA_CATEGORY_MAP_DATA, "spatial_data_category", sheet_name, r["entry_id"]),
            axis=1,
        )
    if "REVIEW_STATUS" in df.columns:
        df["REVIEW_STATUS"] = df.apply(
            lambda r: _normalize_or_warn(r["REVIEW_STATUS"], REVIEW_STATUS_MAP, "REVIEW_STATUS", sheet_name, r["entry_id"]),
            axis=1,
        )

    return df

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Reading {SOURCE_FILE.name} ...")
    backup_if_needed(SOURCE_FILE, BACKUP_DIR)

    xl = pd.ExcelFile(SOURCE_FILE)
    print(f"Sheets found: {xl.sheet_names}")

    # Papers: method_pub + AP_pub combined
    methods_pub = parse_pub_sheet(xl, "method_pub", paper_type="method")
    print(f"  method_pub: {len(methods_pub)} entries ({methods_pub['is_placeholder'].sum()} placeholders)")

    ap_pub = parse_pub_sheet(xl, "AP_pub", paper_type="application")
    print(f"  AP_pub: {len(ap_pub)} entries ({ap_pub['is_placeholder'].sum()} placeholders)")

    papers = pd.concat([methods_pub, ap_pub], ignore_index=True, sort=False)
    print(f"  Papers combined: {len(papers)} entries")

    # Datasets: single "data" sheet
    datasets = parse_dataset_sheet(xl, "data")
    print(f"  data: {len(datasets)} entries")

    # Output filenames derived from source filename
    stem   = SOURCE_FILE.stem
    suffix = "_".join(stem.split("_")[2:]) or stem

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
