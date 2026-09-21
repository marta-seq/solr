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
    """True if this row has already gone through review - manual OR auto
    (any level - "auto-1", "auto-2", etc, per the 2026-09-18 leveled-
    REVIEW_STATUS redesign, see config.py). Only truly untouched rows
    (blank, or "needs_review" - which specifically means "retry me") should
    be re-processed. Behavior-preserving vs. the old flat "auto" check -
    just recognizes the new "auto-N" format too."""
    status = str(review_status).strip().lower()
    return status == config.REVIEW_STATUS_MANUAL or config.parse_auto_level(status) is not None


_DESK_DONE_STATUSES = ("auto", "manual")


def _desk_field_done(row, column: str) -> bool:
    """True if `column` (one of the per-field review-provenance columns
    added 2026-09-20, e.g. METHOD_COMPARISON_REVIEW_STATUS) shows this
    specific desk has already completed a pass on this row - "auto" (an
    agent attempted it, whether or not it found anything) or "manual" (a
    human populated/confirmed it). Missing column or "NA"/blank both mean
    "not attempted yet"."""
    if column not in row.index:
        return False
    return str(row.get(column)).strip().lower() in _DESK_DONE_STATUSES


def _still_needs_a_desk_pass(row) -> bool:
    """True if EITHER the methods desk or the data desk still has work to do
    on this row, per its own per-field status - NOT the whole-row
    REVIEW_STATUS. Replaces relying on whole-row REVIEW_STATUS=auto-N to mean
    "done, never touch again", which had two real bugs (found 2026-09-20):
    (1) a paper where the LLM was called and found zero comparisons got no
    REVIEW_STATUS stamp at all, so it kept being re-queued forever; (2) a
    citation-chased entry, once merged at REVIEW_STATUS=auto-1, was
    permanently excluded from ever having its OWN comparisons/datasets
    checked in a later run, even though only the whole-row status had been
    touched, not these specific fields. Both agents independently re-check
    their own field before actually doing work (see
    compared_methods_agent.py's/data_fetch_agent.py's own skip gates), so
    this only needs to decide whether the row belongs in the queue at all."""
    return (not _desk_field_done(row, "METHOD_COMPARISON_REVIEW_STATUS")
            or not _desk_field_done(row, "DATASET_REVIEW_STATUS"))


def _has_doi(doi: str) -> bool:
    return bool(doi) and str(doi).strip().lower() not in ("", "nan", "na")


def _get_already_attempted_paper_ids(agents=("compared_methods_agent", "data_fetch_agent")) -> set:
    """Papers already attempted THIS SESSION by any of `agents` (staged
    something, whether it succeeded, was skipped, or failed) - re-attempting
    them right now would just waste LLM budget re-discovering the same
    outcome, since nothing changes about a paper between runs until you
    actually merge staging.xlsx into the master CSV. Without this, re-running
    the pipeline before merging restarts from the SAME papers instead of
    continuing to the next unprocessed ones."""
    attempted = set()
    for rec in staging.load_all_candidates_for_run():
        if rec.get("curation_agent") in agents:
            source_id = rec.get("source_paper_entry_id")
            if source_id:
                attempted.add(source_id)
    return attempted


def build_category_audit_pool(methods_df) -> list:
    """Candidate pool for category_audit_agent.py - every method-category
    row that isn't a placeholder, isn't manually reviewed, and hasn't
    already been attempted this session.

    KNOWN LIMITATION, not yet resolved (flagged 2026-09-20, needs Marta's
    call): there is no persistent per-row marker for "already audited,
    verdict was clean" - unlike the methods/data desks, which got a
    dedicated per-field review column each (METHOD_COMPARISON_REVIEW_STATUS/
    DATASET_REVIEW_STATUS). Adding a THIRD such column wasn't something
    Marta explicitly signed off on when scoping that schema change ("just
    the 2 fields"), so this deliberately doesn't invent one unilaterally.
    Practical effect: a row that audits clean in one run will be re-audited
    (another LLM call) in every SEPARATE future run, since only the
    session-scoped staging check prevents re-attempts WITHIN one run. A row
    that gets flagged is fine either way - REVIEW_STATUS=needs_review
    already keeps it out of both this pool and the seed queue afterward.
    Revisit if repeated-audit cost turns out to matter in practice; the fix
    would be a third per-field column (e.g. CATEGORY_AUDIT_STATUS),
    mirroring the existing two."""
    already_attempted = _get_already_attempted_paper_ids(agents=("category_audit_agent",))
    pool = []
    for _, row in methods_df.iterrows():
        if _is_true(row.get("is_placeholder")):
            continue
        if str(row.get("paper_type", "")).strip().lower() == "application":
            continue
        if not _is_method_category(row.get("category")):
            continue
        if str(row.get("REVIEW_STATUS", "")).strip().lower() == config.REVIEW_STATUS_MANUAL:
            continue
        if row["entry_id"] in already_attempted:
            continue
        entry_id = row["entry_id"]
        sheet = "AP_pub" if str(entry_id).upper().startswith("AP") else "method_pub"
        pool.append({"entry_id": entry_id, "sheet": sheet})
    return pool


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
      - BOTH desks have already completed a pass, per the per-field
        METHOD_COMPARISON_REVIEW_STATUS/DATASET_REVIEW_STATUS columns (added
        2026-09-20) - see _still_needs_a_desk_pass()'s own docstring for why
        this replaced a whole-row-REVIEW_STATUS-based check. A row with
        REVIEW_STATUS="auto-1" (e.g. merged in from the literature scanner)
        is NOT excluded by this alone - if neither desk has touched its
        comparison/dataset list yet, it still belongs in the queue.
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
        if str(row.get("REVIEW_STATUS", "")).strip().lower() == config.REVIEW_STATUS_MANUAL:
            continue
        if not _still_needs_a_desk_pass(row):
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
