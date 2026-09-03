"""
triage.py (the mailroom)
Takes the whole pool of papers and decides who's a seed paper for the
methods desk: excludes anything not tagged as a computational method, and
excludes anything already manually reviewed. What's left is depth-0 work.

This is deliberately separate from run_pipeline.py so "what counts as
in-scope" is one readable function, not buried inside the orchestration loop.
"""

from ...common import config, staging


def _is_true(val) -> bool:
    return str(val).strip().lower() == "true"


def _is_method_category(category: str) -> bool:
    if not category or str(category).strip().lower() in ("nan", ""):
        return False
    category_lower = str(category).lower()
    return any(kw in category_lower for kw in config.METHOD_CATEGORY_KEYWORDS)


def _is_already_reviewed(review_status: str) -> bool:
    """True if this row has already gone through review - manual OR auto.
    Only truly untouched rows (empty REVIEW_STATUS) should be re-processed."""
    status = str(review_status).strip().lower()
    return status in (config.REVIEW_STATUS_MANUAL, config.REVIEW_STATUS_AUTO)


def _has_doi(doi: str) -> bool:
    return bool(doi) and str(doi).strip().lower() not in ("", "nan", "na")


def _get_already_attempted_paper_ids() -> set:
    """Papers already attempted this session (staged something, whether it
    succeeded, was skipped, or failed) - re-attempting them right now would
    just waste LLM budget re-discovering the same outcome, since nothing
    changes about a paper between runs until you actually merge staging.xlsx
    into the master CSV (REVIEW_STATUS only updates at that point). Without
    this, re-running the pipeline before merging restarts from the SAME
    seed papers instead of continuing to the next unprocessed ones."""
    attempted = set()
    for rec in staging.load_all_candidates_for_run():
        if rec.get("curation_agent") in ("compared_methods_agent", "data_fetch_agent"):
            source_id = rec.get("source_paper_entry_id")
            if source_id:
                attempted.add(source_id)
    return attempted


def build_seed_queue(methods_df) -> list:
    """
    Returns a list of {"entry_id": ..., "doi": ..., "depth": 0} dicts -
    the starting queue for the compared-methods track.

    Excluded from the pool:
      - placeholder rows (is_placeholder == True)
      - paper_type == "application" (i.e. AP_pub rows) - this track is
        method_pub-only, full stop. Checked via the paper_type column
        (set deterministically by 01_parse_excel.py from which sheet a row
        came from) rather than relying only on the category-keyword check
        below, because some AP_pub rows have compound category values like
        "Application; computational analysis - method" or "Technical
        Methods, Application" that satisfy the keyword match despite
        genuinely being application papers (caught live: AP_30/AP_31/
        AP_39/AP_43/AP_52 all have such compound categories, and were
        slipping into this queue before this check existed - see
        compared_methods_agent.py's docstring for what that caused
        downstream). Rows missing paper_type entirely (older data) are NOT
        excluded by this check - the category-keyword check below still
        applies to those as before.
      - not tagged as a computational method (category doesn't match
        config.METHOD_CATEGORY_KEYWORDS)
      - already manually reviewed (REVIEW_STATUS == "manual") - don't
        re-touch curated work
      - no DOI to fetch text with in the first place
      - already attempted this session (staged something in staging.xlsx,
        even if the outcome was "skipped" or "failed") - re-run of the
        pipeline continues to NEW papers instead of repeating the same
        ones, until you actually merge staging.xlsx into the master CSV
    """
    already_attempted = _get_already_attempted_paper_ids()
    queue = []
    for _, row in methods_df.iterrows():
        if _is_true(row.get("is_placeholder")):
            continue
        if str(row.get("paper_type", "")).strip().lower() == "application":
            continue
        if not _is_method_category(row.get("category")):
            continue
        if _is_already_reviewed(row.get("REVIEW_STATUS")):
            continue
        if not _has_doi(row.get("DOI")):
            continue
        if row["entry_id"] in already_attempted:
            continue
        queue.append({"entry_id": row["entry_id"], "doi": row["DOI"], "depth": 0})
    return queue



# Common fields every modality should have; modality-specific fields only
# apply to their own kind (a proteomics entry shouldn't be flagged for a
# missing "N genes" - it doesn't have genes to begin with).
_COMMON_METADATA_FIELDS = ["organism", "tissue", "disease", "spatial_data_method", "N samples"]
_MODALITY_SPECIFIC_FIELDS = {
    "spatial_proteomics": ["N markers", "Marker"],
    "spatial_transcriptomics": ["N genes", "Genes"],
    # spatial_multi entries can reasonably need both - handled via fallback below
}


def _fields_for_modality(modality: str, available_columns) -> list:
    fields = list(_COMMON_METADATA_FIELDS)
    modality = str(modality).strip().lower()
    if "proteomic" in modality:
        fields += _MODALITY_SPECIFIC_FIELDS["spatial_proteomics"]
    elif "transcriptomic" in modality:
        fields += _MODALITY_SPECIFIC_FIELDS["spatial_transcriptomics"]
    else:
        # unknown/multi modality - only check fields that actually exist as
        # columns, rather than guessing which sub-type of fields apply
        fields += _MODALITY_SPECIFIC_FIELDS["spatial_proteomics"]
        fields += _MODALITY_SPECIFIC_FIELDS["spatial_transcriptomics"]
    return [f for f in fields if f in available_columns]


def build_data_pool(datasets_df) -> list:
    """
    The parallel pool for the data-curation subdepartment's intern agent:
    every dataset entry with at least one empty metadata field, regardless
    of how the entry got there (hand-added or agent-created). Not recursive,
    not depth-tracked - just a flat worklist.

    Field checklist is modality-aware: a spatial_proteomics entry is only
    checked against proteomics-relevant fields (markers, not genes) and
    vice versa for spatial_transcriptomics.
    """
    pool = []
    columns = datasets_df.columns
    for _, row in datasets_df.iterrows():
        if _is_already_reviewed(row.get("REVIEW_STATUS")):
            continue
        fields = _fields_for_modality(row.get("spatial_data_category", ""), columns)
        if not fields:
            continue
        missing = [f for f in fields if _is_empty(row.get(f))]
        if missing:
            pool.append({"entry_id": row["entry_id"], "missing_fields": missing})
    return pool


def _is_empty(val) -> bool:
    return val is None or str(val).strip().lower() in ("", "nan", "na", "?", "none")
