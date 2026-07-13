"""
merge_candidates.py
Reads data/agent_review/staging.xlsx (everything the agents have proposed
so far) and the most recent master Excel in data/data_curated/, applies
candidates into a NEW file - datasets_curated_autoreview_<date>.xlsx -
and NEVER overwrites your original manually-curated file. Archives
staging.xlsx afterward so the next pipeline run starts clean.

Rules:
    - A row with REVIEW_STATUS == "manual" is NEVER touched, full stop -
      belt-and-suspenders on top of triage already excluding these from
      being processed in the first place.
    - If two candidates touch the same cell before you've merged, last-
      written-wins (staging.xlsx is append-only/chronological, so
      processing it top-to-bottom naturally gives this - no special
      tracking needed).
    - needs_review (low-confidence) entries are merged in alongside auto
      entries, same file - REVIEW_STATUS already lets you filter/sort in
      Excel, so a separate file wasn't worth the extra thing to track.
    - Data_multi's real columns don't match data_fetch_agent.py's field
      names at all (its DOI columns are 'DOI '/'DOI', not 'data_DOI'/
      'paper_DOI') - rather than guess and risk writing into the wrong
      column of a confusingly-structured sheet, anything targeting
      Data_multi goes into a separate NEEDS_MANUAL_PLACEMENT sheet instead.

Usage:
    python -m src.agents.merge_candidates
"""

import shutil
from datetime import date
from pathlib import Path

import openpyxl

from .common import config

HEADER_ROW = 2  # row 1 has merged-cell section titles, real headers are row 2

ID_COLUMN_BY_SHEET = {
    "papers": "PLACEHOLDER_ENTRY_ID",
    "Data_SP": "P_ENTRY_ID",
    "Data_ST": "P_ENTRY_ID",
    "Data_multi": "P_ENTRY_ID",
}

# see module docstring - Data_multi's real column names don't match what
# data_fetch_agent.py writes at all, so it's not safe to auto-merge into it
SHEETS_TOO_AMBIGUOUS_TO_AUTO_MERGE = {"Data_multi"}

# staging.xlsx's own bookkeeping columns - never written as a cell value,
# handled separately (as AUTO_* audit columns) instead
_BOOKKEEPING_COLUMNS = (
    "entry_id", "action", "target_sheet", "curation_agent", "curation_model",
    "curation_date", "confidence", "source_paper_entry_id", "notes",
)


def _latest_master_file() -> Path:
    matches = sorted(config.CURATED_DIR.glob("datasets_curated_*.xlsx"))
    matches = [m for m in matches if "autoreview" not in m.name]
    if not matches:
        raise FileNotFoundError(f"No datasets_curated_*.xlsx found in {config.CURATED_DIR}")
    return matches[-1]


def _normalize_header(h) -> str:
    return str(h).strip() if h is not None else ""


def _header_map(ws) -> dict:
    """{normalized_header_name: column_index} for the real header row."""
    result = {}
    for cell in ws[HEADER_ROW]:
        name = _normalize_header(cell.value)
        if name:
            result[name] = cell.column
    return result


def _find_row_by_id(ws, id_col_idx: int, entry_id: str):
    for row in range(HEADER_ROW + 1, ws.max_row + 1):
        val = ws.cell(row=row, column=id_col_idx).value
        if val is not None and str(val).strip() == str(entry_id).strip():
            return row
    return None


def _get_or_add_column(ws, header_map: dict, field_name: str) -> int:
    normalized = _normalize_header(field_name)
    if normalized in header_map:
        return header_map[normalized]
    new_col = ws.max_column + 1
    ws.cell(row=HEADER_ROW, column=new_col, value=field_name)
    header_map[normalized] = new_col
    return new_col


def _load_staging_records(staging_path: Path) -> list:
    wb = openpyxl.load_workbook(staging_path)
    records = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        for row in ws.iter_rows(min_row=2, values_only=True):
            rec = dict(zip(header, row))
            if rec.get("entry_id"):
                records.append(rec)
    return records


def merge():
    master_path = _latest_master_file()
    print(f"[merge] Using master file: {master_path.name}")

    staging_path = config.STAGING_DIR / "staging.xlsx"
    if not staging_path.exists():
        print("[merge] No staging.xlsx found - nothing to merge.")
        return

    records = _load_staging_records(staging_path)
    if not records:
        print("[merge] staging.xlsx has no candidates - nothing to merge.")
        return

    master_wb = openpyxl.load_workbook(master_path)

    applied = 0
    manual_skipped = 0
    needs_manual_placement = []

    for rec in records:
        action = rec.get("action")
        target_sheet = rec.get("target_sheet")
        entry_id = rec.get("entry_id")
        if not action or not target_sheet or not entry_id:
            continue

        if target_sheet in SHEETS_TOO_AMBIGUOUS_TO_AUTO_MERGE:
            needs_manual_placement.append(rec)
            continue

        if target_sheet not in master_wb.sheetnames:
            print(f"[merge] WARNING: sheet '{target_sheet}' not in master file, skipping {entry_id}")
            continue

        ws = master_wb[target_sheet]
        id_col_name = ID_COLUMN_BY_SHEET.get(target_sheet, "entry_id")
        header_map = _header_map(ws)
        id_col_idx = header_map.get(id_col_name)
        if id_col_idx is None:
            print(f"[merge] WARNING: ID column '{id_col_name}' not found in '{target_sheet}', skipping {entry_id}")
            continue

        existing_row = _find_row_by_id(ws, id_col_idx, entry_id)

        # never touch a manually-reviewed row - belt and suspenders on top
        # of triage already excluding these from being processed at all
        if existing_row is not None:
            review_col = header_map.get("REVIEW_STATUS")
            if review_col:
                current_status = ws.cell(row=existing_row, column=review_col).value
                if str(current_status).strip().lower() == "manual":
                    manual_skipped += 1
                    continue

        candidate_fields = {
            k: v for k, v in rec.items()
            if k not in _BOOKKEEPING_COLUMNS and v not in (None, "")
        }

        if existing_row is None:
            # create_entry (or an update_field that arrived before any
            # create_entry for the same id, which shouldn't normally happen)
            new_row_num = ws.max_row + 1
            id_col_idx = _get_or_add_column(ws, header_map, id_col_name)
            ws.cell(row=new_row_num, column=id_col_idx, value=entry_id)
            for field_name, value in candidate_fields.items():
                col_idx = _get_or_add_column(ws, header_map, field_name)
                ws.cell(row=new_row_num, column=col_idx, value=value)
            for audit_field in ("curation_agent", "curation_model", "curation_date", "confidence", "notes"):
                if rec.get(audit_field) not in (None, ""):
                    col_idx = _get_or_add_column(ws, header_map, f"AUTO_{audit_field.upper()}")
                    ws.cell(row=new_row_num, column=col_idx, value=rec.get(audit_field))
            applied += 1
        else:
            # update_field on an existing row (or create_entry for an id
            # that's somehow already there - treat as an update, don't
            # duplicate the row)
            for field_name, value in candidate_fields.items():
                col_idx = _get_or_add_column(ws, header_map, field_name)
                ws.cell(row=existing_row, column=col_idx, value=value)
            for audit_field in ("curation_agent", "curation_model", "curation_date", "confidence", "notes"):
                if rec.get(audit_field) not in (None, ""):
                    col_idx = _get_or_add_column(ws, header_map, f"AUTO_{audit_field.upper()}")
                    ws.cell(row=existing_row, column=col_idx, value=rec.get(audit_field))
            applied += 1

    if needs_manual_placement:
        ws_manual = master_wb.create_sheet("NEEDS_MANUAL_PLACEMENT")
        all_keys = sorted({k for rec in needs_manual_placement for k in rec.keys()})
        ws_manual.append(all_keys)
        for rec in needs_manual_placement:
            ws_manual.append([rec.get(k, "") for k in all_keys])

    today = date.today().isoformat()
    output_path = config.CURATED_DIR / f"datasets_curated_autoreview_{today}.xlsx"
    master_wb.save(output_path)
    print(f"[merge] Wrote {output_path.name}:")
    print(f"[merge]   {applied} candidates applied")
    print(f"[merge]   {manual_skipped} skipped (manually-reviewed rows protected)")
    print(f"[merge]   {len(needs_manual_placement)} routed to NEEDS_MANUAL_PLACEMENT sheet")

    archived_path = config.STAGING_DIR / f"staging_merged_{today}.xlsx"
    shutil.move(str(staging_path), str(archived_path))
    print(f"[merge] Archived staging.xlsx -> {archived_path.name} (next run starts clean)")


if __name__ == "__main__":
    merge()
