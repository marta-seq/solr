"""
compared_methods_agent.py (methods desk)
For one seed paper: reads its methods section, asks the LLM which methods it
compares itself against and how each is cited, resolves each citation to a
DOI, matches against the existing DB or creates a new M_AUTO entry, and
stages everything. Returns a stats dict including the newly-created entries
so run_pipeline.py can queue them at depth+1 (subject to MAX_HOPS).

Nothing here writes to the master Excel - everything goes through
common.staging.append_candidate().

Every staging call here targets sheet="method_pub" specifically (never
AP_pub) - this track's seed queue (mailroom.triage.build_seed_queue) is
already filtered to method-category rows, and every entry this agent
creates is itself a "computational analysis - method" M_AUTO_x row, so
there's no ambiguity to resolve at write time the way data_fetch_agent.py
has to (that one processes both method and application papers).
"""

import re

from ..common import config, staging
from ..common.llm_client import call_llm_json, LLMError, LLMExhaustedError
from ..common.paper_fetcher import fetch_paper, get_agent_text, was_truncated, is_probably_real_content
from ..common.reference_list_parser import parse_reference_list
from ..common.reference_resolver import resolve_citation, is_confident

SYSTEM_PROMPT = """You are a careful research assistant extracting factual information from a \
scientific paper's methods section. You will be given the methods section text (or, if that \
wasn't available, the full paper text). Identify every OTHER computational method or tool that \
this paper compares its own method against (baseline comparisons, benchmark methods) - do NOT \
include methods that are merely used as a preprocessing step or a component of the paper's own \
pipeline, only genuine comparison/benchmark methods.

STRICT RULES:
- method_name MUST be an actual named algorithm, software tool, or pipeline (e.g. "Cellpose", \
"StarDist", "Mesmer") - a proper noun for a specific piece of software/method.
- method_name must NEVER be a generic author-year citation (e.g. "Zeisel et al." or "Smith et \
al., 2020" is NOT a method name). If a citation doesn't name a specific tool/algorithm - even if \
it appears in a comparison-like sentence - skip it entirely. Only extract it if the paper itself \
gives the compared method a proper name.
- Do NOT include datasets, biological findings, or general background citations, even if phrased \
similarly to a comparison.

Return ONLY a JSON array, nothing else, no markdown fences. Each element:
{
  "method_name": "the name of the compared method/tool, as written in the paper - MUST be an \
actual tool/algorithm proper name per the strict rules above, never an author-year citation",
  "citation_marker": "ONLY digits, nothing else - e.g. '12' or '41'. Do NOT include the word \
'ref', brackets, parentheses, periods, or any other character - digits only. If multiple numbers \
are cited together (e.g. text shows '[12,13]'), return only the FIRST number. Leave this as a \
genuinely empty string only if there is truly no number anywhere near the mention (the paper \
uses author-year citations like 'Smith et al., 2020' instead of numbers).",
  "citation_text": "ONLY fill this in if citation_marker is a genuinely empty string - the \
author-year citation as written near the mention (e.g. 'Smith et al., 2020'). Leave this empty \
if you filled citation_marker - never fill both."
}
If no comparison methods are mentioned, return an empty JSON array: []
"""


def _build_user_prompt(paper_entry_id: str, methods_text: str) -> str:
    return f"Methods section (paper {paper_entry_id}):\n\n{methods_text}"


def _fallback_marker_from_text(citation_text: str) -> str:
    """Defensive fallback: even with an explicit prompt asking for digits-only
    markers, the LLM sometimes still puts marker-like text into citation_text
    instead (caught via testing on real output: 'ref. 41(' landed in
    citation_text with citation_marker left empty, so the reference-list
    lookup never got a chance to run and it fell straight to an unreliable
    Crossref search on garbled text). If citation_text looks like a bare
    reference marker (e.g. 'ref. 41', 'ref 41(', '[41]'), pull the number out
    so the reference-list lookup can still be attempted. Deliberately
    lenient about trailing junk (stray brackets/parens/periods) since OCR/
    extraction artifacts are common here - only the leading shape matters."""
    m = re.match(r"^\s*\[?\(?\s*ref\.?\s*(\d{1,3})\b", citation_text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^\s*\[(\d{1,3})\]", citation_text)
    return m.group(1) if m else ""


def _append_to_comparison_list(existing: str, new_id: str) -> str:
    """Mirrors the existing 'M_PR_5, M_PR_2, M_PH_1' comma-separated format
    already used in Method_comparison_P_ENTRY_ID."""
    existing = "" if existing is None else str(existing).strip()
    if existing in ("", "nan", "NA"):
        return new_id
    ids = [x.strip() for x in existing.split(",") if x.strip()]
    if new_id not in ids:
        ids.append(new_id)
    return ", ".join(ids)


def _empty_stats(skip_reason: str, new_queue_items=None, llm_exhausted: bool = False) -> dict:
    """Consistent return shape at every early-exit point. skip_reason is
    ALWAYS a non-empty string here (the LLM was never called) - callers
    should show it directly rather than leaving people to infer whether
    0 results means 'skipped' or 'called and found nothing'.
    llm_exhausted is True ONLY when the whole provider/model chain failed
    (LLMExhaustedError) - run_pipeline.py checks this to stop a run early
    instead of repeating the same doomed wait on every remaining paper."""
    return {
        "new_queue_items": new_queue_items or [],
        "total_encountered": 0,
        "total_resolved": 0,
        "skip_reason": skip_reason,
        "llm_exhausted": llm_exhausted,
    }


def process_paper(db, paper_entry: dict, fetched: dict = None, reference_map: dict = None) -> dict:
    """
    paper_entry: {"entry_id": ..., "doi": ..., "depth": ...}
    fetched/reference_map: pass these in if the orchestrator already fetched
    this paper (e.g. for a shared pass with data_fetch_agent), to avoid
    re-fetching/re-parsing the same paper twice.

    Returns {"new_queue_items": [...], "total_encountered": N,
    "total_resolved": M, "skip_reason": str or None}. skip_reason is None
    ONLY when the LLM was genuinely called - always check this field rather
    than inferring from a 0 count, since 0 encountered can otherwise mean
    either "skipped, never called" or "called, found nothing".
    new_queue_items is {"entry_id": ..., "doi": ..., "depth": depth+1} dicts
    for newly-created method entries, ready to feed back into the seed
    queue - caller is responsible for checking MAX_HOPS/MAX_PAPERS_PER_RUN
    before actually queueing them.
    """
    entry_id = paper_entry["entry_id"]
    doi = paper_entry["doi"]
    depth = paper_entry["depth"]

    if fetched is None:
        fetched = fetch_paper(doi)
    methods_text, text_source = get_agent_text(fetched, "methods")

    if not methods_text:
        staging.append_candidate(
            action="update_field", sheet="method_pub", entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"No usable text could be fetched for this DOI (fetch source: {fetched['source']}).",
        )
        return _empty_stats(f"no text could be fetched (fetch source: {fetched['source']})")

    if not is_probably_real_content(methods_text):
        staging.append_candidate(
            action="update_field", sheet="method_pub", entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"Extracted text (source: {text_source}) failed the content quality check - "
                  f"looks like website navigation/boilerplate rather than real article text. "
                  f"Skipped the LLM call to avoid wasting budget on unreliable input.",
        )
        return _empty_stats(f"text (source: {text_source}) failed the content-quality check "
                             f"(looked like navigation/boilerplate, not real article text)")

    if config.REQUIRE_ISOLATED_SECTION and text_source != "section:methods":
        staging.append_candidate(
            action="update_field", sheet="method_pub", entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"Could not cleanly isolate the methods section (got '{text_source}' instead) - "
                  f"skipped the LLM call rather than spend a query on unfocused/low-confidence text.",
        )
        return _empty_stats(f"could not cleanly isolate the methods section "
                             f"(got '{text_source}' instead of 'section:methods') - "
                             f"REQUIRE_ISOLATED_SECTION blocked the LLM call")



    truncated_note = ""
    if was_truncated(fetched, "methods"):
        truncated_note = " (input text was truncated - findings may be incomplete)"

    if reference_map is None:
        references_text, _ = get_agent_text(fetched, "references")
        reference_map = parse_reference_list(references_text) if references_text else {}

    try:
        extracted, model_used = call_llm_json(SYSTEM_PROMPT, _build_user_prompt(entry_id, methods_text))
    except LLMError as e:
        staging.append_candidate(
            action="update_field", sheet="method_pub", entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"LLM extraction failed: {e}",
        )
        return _empty_stats(f"LLM WAS called but every provider/model failed: {e}",
                             llm_exhausted=isinstance(e, LLMExhaustedError))

    if not isinstance(extracted, list):
        extracted = []

    new_queue_items = []
    comparison_ids_added = []

    for item in extracted:
        method_name = (item or {}).get("method_name", "").strip()
        citation_marker = (item or {}).get("citation_marker", "").strip()
        citation_text = (item or {}).get("citation_text", "").strip()
        if not method_name or not (citation_marker or citation_text):
            continue

        if not citation_marker and citation_text:
            fallback_marker = _fallback_marker_from_text(citation_text)
            if fallback_marker:
                citation_marker = fallback_marker

        resolved = resolve_citation(citation_marker, citation_text, reference_map,
                                     expected_name_hint=method_name)
        confident = is_confident(resolved)
        resolved_doi = resolved["resolved_doi"]

        if resolved_doi and resolved_doi in db.doi_index:
            # already in the DB (or already staged earlier this run) - just link it
            matched_id = db.doi_index.lookup(resolved_doi)
            comparison_ids_added.append(matched_id)
            continue

        if not resolved_doi:
            # couldn't resolve a DOI at all - stage a note, nothing to link/create
            staging.append_candidate(
                action="update_field", sheet="method_pub", entry_id=entry_id,
                fields={},
                source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
                curation_model=model_used, confidence=0.0,
                notes=f"Could not resolve DOI for compared method '{method_name}' "
                      f"(marker: '{citation_marker}', citation: {resolved['input_text'][:200]})",
            )
            continue

        # genuinely new - allocate an ID and stage a new M_AUTO entry
        # Deliberately NOT trying to guess the real category prefix (M_PR/M_SE/etc.) -
        # that requires understanding your category scheme, which you're planning to
        # redesign anyway. A distinct, obviously-generic prefix makes it clear during
        # review that this entry needs YOU to assign its real category, rather than
        # silently guessing wrong (caught via testing: SSAM/Baysor, both segmentation
        # methods, got created as M_PR_x when they should've been M_SE_x or matched
        # to their existing entries).
        new_id = db.id_allocator.next_id("M_AUTO")
        review_status = config.REVIEW_STATUS_AUTO if confident else config.REVIEW_STATUS_NEEDS_REVIEW

        staging.append_candidate(
            action="create_entry", sheet="method_pub", entry_id=new_id,
            fields={
                "DOI": resolved_doi,
                "category": "computational analysis - method",
                "name": method_name,
                "REVIEW_STATUS": review_status,
            },
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model=model_used, confidence=resolved["confidence"],
            notes=f"Discovered as a comparison method in {entry_id}. "
                  f"Matched Crossref title: '{resolved['matched_title']}'{truncated_note}",
        )

        # so later papers this same run see it and don't duplicate-create it
        db.doi_index.add(resolved_doi, new_id)
        comparison_ids_added.append(new_id)

        new_queue_items.append({"entry_id": new_id, "doi": resolved_doi, "depth": depth + 1})

    if comparison_ids_added:
        existing_value = ""
        row = db.methods.loc[db.methods["entry_id"] == entry_id, "Method_comparison_P_ENTRY_ID"]
        if not row.empty:
            existing_value = row.iloc[0]
        updated_value = existing_value
        for new_id in comparison_ids_added:
            updated_value = _append_to_comparison_list(updated_value, new_id)

        staging.append_candidate(
            action="update_field", sheet="method_pub", entry_id=entry_id,
            fields={"Method_comparison_P_ENTRY_ID": updated_value},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model=model_used, confidence=None,
            notes="Appended newly-found comparison method IDs.",
        )

    return {
        "new_queue_items": new_queue_items,
        "total_encountered": len(extracted),
        "total_resolved": len(comparison_ids_added),
        "skip_reason": None,
    }
