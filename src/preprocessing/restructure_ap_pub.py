"""
restructure_ap_pub.py - one-off tidy of the AP_pub sheet of a curated workbook (Marta's review of 2026-10-11, done 2026-10-13).

Reads a workbook, writes a NEW one (never overwrites the input), only the AP_pub sheet changes:
  1. Column order: every human-facing field sits side by side on the left (... type of spatial data, spatial_data_category,
     spatial_data_method, animal, tissue, disease, tissue/disease [old combined column, values kept], NOTES); the agent/audit columns
     (AUTO_*, *_REVIEW_*, APPLICATION_METADATA_*) stay to the right. Before, `tissue`/`disease` had been appended in the middle of the
     agent columns by merge_candidates.
  2. spatial_data_category / spatial_data_method derived from `type of spatial data` (+ title/abstract when that cell names no platform),
     via agents/common/spatial_labels.py. The raw `type of spatial data` is kept untouched.
  3. REVIEW_STATUS blank + row touched by an agent -> auto-N (N = highest AUTO_CURATION_DATE_LEVEL filled, else 1). Rows no agent ever
     touched stay blank (blank = untouched, see triage.py).
  4. Date(last_change_to_entry) blank -> newest agent date on the row (YYYYMMDD int), else Date(added_to_dataset). Existing values that are
     newer are kept.
  5. name blank -> lowercase first-author surname + year, collision letters (dataset_naming.fill_blank_names). Existing names are untouched.
Nothing non-blank is ever overwritten.

Usage:  python restructure_ap_pub.py [input.xlsx] [output.xlsx]
"""
import re
import sys
from copy import copy
from pathlib import Path

import openpyxl

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from agents.common import config, dataset_naming, spatial_labels  # noqa: E402

HUMAN_ORDER_AFTER = "type of spatial data"
NEW_COLS = ["spatial_data_category", "spatial_data_method"]
# columns that must follow `type of spatial data` directly, in this order (those that exist)
TAIL_HUMAN = ["animal", "tissue", "disease", "tissue/disease", "NOTES"]
_AGENT_RE = re.compile(r"^(AUTO_|METHOD_COMPARISON_REVIEW_|DATASET_REVIEW_|APPLICATION_METADATA_)")
_LEVEL_RE = re.compile(r"^AUTO_CURATION_DATE_LEVEL(\d+)$")
_DATE_RE = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})")


def _blank(v) -> bool:
    return v is None or str(v).strip() == "" or str(v).strip().lower() in ("nan", "none")


def _as_date_int(v):
    m = _DATE_RE.search("" if v is None else str(v))
    return int("".join(m.groups())) if m else None


def restructure(src: Path, dst: Path) -> dict:
    wb = openpyxl.load_workbook(src)
    ws = wb["AP_pub"]
    header_row = next(r for r in range(1, 6) if any(c.value == "P_ENTRY_ID" for c in ws[r]))
    max_col = ws.max_column
    old_headers = [ws.cell(header_row, c).value for c in range(1, max_col + 1)]
    old_styles = {c: ws.cell(header_row, c)._style for c in range(1, max_col + 1)}
    first_data = header_row + 1
    rows = [[ws.cell(r, c).value for c in range(1, max_col + 1)] for r in range(first_data, ws.max_row + 1)]
    rows = [r for r in rows if not all(_blank(v) for v in r)]
    idx = {h: i for i, h in enumerate(old_headers) if h}

    for need in ("REVIEW_STATUS", "Date(last_change_to_entry)", "name", "type of spatial data", "tissue", "disease"):
        if need not in idx:
            raise SystemExit(f"AP_pub has no '{need}' column")

    stats = {"rows": len(rows)}

    # --- values -----------------------------------------------------------------------------------------------------------
    def get(r, h):
        return r[idx[h]] if h in idx else None

    # 2. spatial split
    cat_vals, meth_vals = [], []
    for r in rows:
        ctx = f"{get(r, 'title') or ''} {get(r, 'abstract') or ''}"
        cat, meth = spatial_labels.classify(get(r, "type of spatial data"), ctx)
        cat_vals.append(cat)
        meth_vals.append(meth)
    stats["category_filled"] = sum(bool(c) for c in cat_vals)
    stats["method_filled"] = sum(bool(m) for m in meth_vals)

    # 3. REVIEW_STATUS
    level_cols = [i for h, i in idx.items() if _LEVEL_RE.match(str(h))]
    date_cols = [i for h, i in idx.items() if h and str(h).endswith("_DATE") or (h and _LEVEL_RE.match(str(h)))]
    n_status = 0
    for r in rows:
        if _blank(r[idx["REVIEW_STATUS"]]):
            levels = [int(_LEVEL_RE.match(str(h)).group(1)) for h, i in idx.items()
                      if _LEVEL_RE.match(str(h)) and not _blank(r[i])]
            if levels or any(not _blank(r[i]) for i in date_cols):
                r[idx["REVIEW_STATUS"]] = config.auto_level_status(max(levels) if levels else 1)
                n_status += 1
    stats["review_status_filled"] = n_status

    # 4. last change
    n_date = n_fallback = 0
    for r in rows:
        agent_dates = [d for d in (_as_date_int(r[i]) for i in date_cols) if d]
        existing = _as_date_int(r[idx["Date(last_change_to_entry)"]])
        cand = max(agent_dates) if agent_dates else None
        if cand is None and existing is None:
            cand = _as_date_int(get(r, "Date(added_to_dataset)"))
            n_fallback += cand is not None
        best = max(x for x in (existing, cand) if x) if (existing or cand) else None
        if best and best != existing:
            r[idx["Date(last_change_to_entry)"]] = best
            n_date += 1
    stats["last_change_filled_or_raised"] = n_date
    stats["last_change_from_added_date"] = n_fallback

    # 5. names
    dicts = [{"name": None if _blank(get(r, "name")) else get(r, "name"),
              "authors": get(r, "authors"), "year": get(r, "year")} for r in rows]
    stats["names_filled"] = dataset_naming.fill_blank_names(dicts)
    for r, d in zip(rows, dicts):
        if _blank(get(r, "name")) and d.get("name"):
            r[idx["name"]] = d["name"]

    # --- new column order -------------------------------------------------------------------------------------------------
    present = [h for h in old_headers if h]
    anchored = set(NEW_COLS) | {h for h in TAIL_HUMAN}
    human_front, agent_cols = [], []
    for h in present:
        if h in anchored:
            continue
        (agent_cols if _AGENT_RE.match(h) else human_front).append(h)
    if HUMAN_ORDER_AFTER not in human_front:
        raise SystemExit(f"'{HUMAN_ORDER_AFTER}' not in AP_pub header")
    at = human_front.index(HUMAN_ORDER_AFTER) + 1
    human = human_front[:at] + NEW_COLS + [h for h in TAIL_HUMAN if h in present] + human_front[at:]
    # Blank-header columns hold stray legacy cells (misaligned audit cells, a curator note) - keep them, same order, as the gap
    # between the human block and the agent block.
    spacer_src = [i for i, h in enumerate(old_headers) if not h]
    new_headers = human + [None] * len(spacer_src) + agent_cols

    new_vals = {"spatial_data_category": cat_vals, "spatial_data_method": meth_vals}
    base_style = old_styles[idx["type of spatial data"] + 1]

    # --- rewrite the sheet ------------------------------------------------------------------------------------------------
    for r in range(header_row, ws.max_row + 1):
        for c in range(1, max(max_col, len(new_headers)) + 1):
            ws.cell(r, c).value = None
    for c, h in enumerate(new_headers, start=1):
        cell = ws.cell(header_row, c)
        cell.value = h
        cell._style = copy(old_styles[idx[h] + 1] if h in idx else base_style)
    spacer_iter = iter(spacer_src)
    src_of = []                                              # per new column: index into the old row (None = computed column)
    for h in new_headers:
        src_of.append(next(spacer_iter) if h is None else idx.get(h))
    for n, r in enumerate(rows):
        for c, h in enumerate(new_headers, start=1):
            if h in new_vals:
                ws.cell(first_data + n, c).value = new_vals[h][n]
            elif src_of[c - 1] is not None:
                ws.cell(first_data + n, c).value = r[src_of[c - 1]]

    wb.save(dst)
    stats["human_block"] = human
    return stats


if __name__ == "__main__":
    cur = config.CURATED_DIR
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(cur.glob("datasets_curated_autoreview_*.xlsx"))[-1]
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else cur / "datasets_curated_autoreview_2026-10-13.xlsx"
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to overwrite the input file")
    out = restructure(src, dst)
    print(f"{src.name} -> {dst.name}")
    for k, v in out.items():
        print(f"  {k}: {v}")
