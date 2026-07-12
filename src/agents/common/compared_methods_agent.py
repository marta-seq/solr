"""
compared_methods_agent.py (methods desk)
For one seed paper: reads its methods section, asks the LLM which methods it
compares itself against and how each is cited, resolves each citation to a
DOI, matches against the existing DB or creates a new M_PR entry, and stages
everything. Returns the newly-created entries so run_pipeline.py can queue
them at depth+1 (subject to MAX_HOPS).

Nothing here writes to the master Excel - everything goes through
common.staging.append_candidate().
"""

from ..common import config, staging
from ..common.llm_client import call_llm_json, LLMError
from ..common.paper_fetcher import fetch_paper, get_agent_text, was_truncated
from ..common.reference_list_parser import parse_reference_list
from ..common.reference_resolver import resolve_citation, is_confident

SYSTEM_PROMPT = """You are a careful research assistant extracting factual information from a \
scientific paper's methods section. You will be given the methods section text (or, if that \
wasn't available, the full paper text). Identify every OTHER computational method or tool that \
this paper compares its own method against (baseline comparisons, benchmark methods) - do NOT \
include methods that are merely used as a preprocessing step or a component of the paper's own \
pipeline, only genuine comparison/benchmark methods.

Return ONLY a JSON array, nothing else, no markdown fences. Each element:
{
  "method_name": "the name of the compared method/tool, as written in the paper",
  "citation_marker": "the reference NUMBER as it appears at that mention, e.g. '12' or '[12]' - \
this paper likely uses numbered citations, so look for a bracketed or superscript number right \
after the method name. Leave this as an empty string if there is genuinely no number (the paper \
uses author-year citations like 'Smith et al., 2020' instead).",
  "citation_text": "ONLY fill this in if citation_marker is empty - the author-year citation as \
written near the mention (e.g. 'Smith et al., 2020'). Leave empty if you filled citation_marker."
}
If no comparison methods are mentioned, return an empty JSON array: []
"""


def _build_user_prompt(paper_entry_id: str, methods_text: str) -> str:
    return f"Methods section (paper {paper_entry_id}):\n\n{methods_text}"


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


def process_paper(db, paper_entry: dict, fetched: dict = None, reference_map: dict = None) -> list:
    """
    paper_entry: {"entry_id": ..., "doi": ..., "depth": ...}
    fetched/reference_map: pass these in if the orchestrator already fetched
    this paper (e.g. for a shared pass with data_fetch_agent), to avoid
    re-fetching/re-parsing the same paper twice.

    Returns: list of {"entry_id": ..., "doi": ..., "depth": depth+1} for
    newly-created method entries, ready to feed back into the seed queue -
    caller is responsible for checking MAX_HOPS/MAX_PAPERS_PER_RUN before
    actually queueing them.
    """
    entry_id = paper_entry["entry_id"]
    doi = paper_entry["doi"]
    depth = paper_entry["depth"]
    new_queue_items = []

    if fetched is None:
        fetched = fetch_paper(doi)
    methods_text, text_source = get_agent_text(fetched, "methods")

    if not methods_text:
        staging.append_candidate(
            action="update_field", sheet="papers", entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"No usable text could be fetched for this DOI (fetch source: {fetched['source']}).",
        )
        return new_queue_items

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
            action="update_field", sheet="papers", entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"LLM extraction failed: {e}",
        )
        return new_queue_items

    if not isinstance(extracted, list):
        extracted = []

    comparison_ids_added = []

    for item in extracted:
        method_name = (item or {}).get("method_name", "").strip()
        citation_marker = (item or {}).get("citation_marker", "").strip()
        citation_text = (item or {}).get("citation_text", "").strip()
        if not method_name or not (citation_marker or citation_text):
            continue

        resolved = resolve_citation(citation_marker, citation_text, reference_map)
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
                action="update_field", sheet="papers", entry_id=entry_id,
                fields={},
                source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
                curation_model=model_used, confidence=0.0,
                notes=f"Could not resolve DOI for compared method '{method_name}' "
                      f"(marker: '{citation_marker}', citation: {resolved['input_text'][:200]})",
            )
            continue

        # genuinely new - allocate an ID and stage a new M_PR entry
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
            action="create_entry", sheet="papers", entry_id=new_id,
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
            action="update_field", sheet="papers", entry_id=entry_id,
            fields={"Method_comparison_P_ENTRY_ID": updated_value},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model=model_used, confidence=None,
            notes="Appended newly-found comparison method IDs.",
        )

    return new_queue_items
