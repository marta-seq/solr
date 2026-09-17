"""
scan.py
Orchestrator for the literature-search source (the third way a paper enters
SOLR, alongside manual entry and the citation-chasing desk - see
LITERATURE_SEARCH_CONTEXT.md). Queries PubMed and/or bioRxiv/medRxiv, runs
every result through one funnel - seen_ledger dedup -> exact-DOI dedup
against the live DB -> keyword prefilter -> LLM relevance pass - and stages
genuinely new, relevant candidates into staging.xlsx exactly like
compared_methods_agent.py/data_fetch_agent.py already do. No parallel
infrastructure: same staging.append_candidate(), same DoiIndex, same
merge_candidates.py routing downstream.

NOT wired into run_pipeline.py yet - run standalone:
    python -m src.agents.living_ingestion.literature_search.scan pubmed \
        --query "spatial proteomics" --mindate 2026/08/01 --maxdate 2026/09/01
    python -m src.agents.living_ingestion.literature_search.scan biorxiv \
        --start 2026-08-01 --end 2026-09-01 [--medrxiv]

ID-prefix note: reuses the existing citation-chasing agent's conventions -
"M_AUTO" for new method entries (matches compared_methods_agent.py),
"AP" for new application entries (matches data_fetch_agent.py's stub
creation) - rather than inventing a new prefix. Provenance is already
recorded properly in staging.xlsx's curation_agent column
("literature_search_scanner"), not meant to be inferred from the ID itself.
Flagged as a judgment call worth revisiting if it turns out to be confusing
in the reviewed Excel - trivial to change since nothing here has been run
for real yet.

Known gap, not built here: bioRxiv's `published` field lets us prefer the
published-version DOI at INGEST time (see _canonical_doi below), but a
preprint staged/rejected BEFORE bioRxiv resolves its published match won't
get automatically re-keyed later when that match appears - that needs a
separate rescan pass over the ledger's existing biorxiv/medrxiv entries,
not yet built. Low risk in the meantime: DoiIndex would still catch the
would-be duplicate once the published version is separately found via
PubMed, since dedup checks are DOI-based, not ledger-based, for the
already-in-DB case.
"""

import argparse

from ...common import config, staging
from ...common.db_loader import Database
from ...common.llm_client import LLMError
from . import biorxiv_client, pubmed_client, seen_ledger
from .relevance import (
    keyword_prefilter, llm_relevance_pass, category_pass_is_confident,
    publication_type_reject_reason, is_review,
)

CURATION_AGENT_NAME = "literature_search_scanner"


def _canonical_doi(record: dict) -> str:
    """Prefer the published-version DOI when bioRxiv/medRxiv has already
    matched one - see module docstring. Plain pass-through for PubMed
    records (no 'published_doi' key)."""
    return record.get("published_doi") or record.get("doi", "")


def _category_field(verdict: dict, review: bool) -> tuple:
    """The coarse `category` field (distinct from the fine-grained
    pipeline_category taxonomy) - matches canonical values category_maps.py
    already defines, never invents new ones. Returns (category_string,
    caveat_note_or_empty).

    `review` (from PubMed's own PublicationType, see relevance.is_review) is
    an ADDITIONAL tag, not a replacement classification - method vs.
    application (and pipeline_category for methods) is decided exactly the
    same whether or not this is a review; review-ness only changes which
    category string gets used, per Marta's 2026-09-17 clarification (a
    review of computational methods must still be tagged/categorized as
    such, not just dumped in a generic "review" bucket blind to that).

    compared_methods_agent.py hardcodes "computational analysis - method"
    for every new (non-review) method entry; mirrored here for consistency.
    For application papers, distinguishes technical (lab technique/platform/
    protocol papers - IMC/CODEX description, protocol improvement) from
    ordinary biological application papers, using the SAME LLM call's
    "technical_application" flag - not a separate pass.

    Known taxonomy gap: category_maps.py has no standalone canonical value
    for "technical application, and also a review" (only a 3-way combo,
    "Application review; Technical review; General omics review", which
    doesn't cleanly decompose to just this pair) - collapses to "Application
    review" in that specific case rather than inventing a new string, with a
    caveat note so it's visible at merge/review time."""
    if verdict["paper_type"] == "method":
        return ("computational analysis - review" if review else "computational analysis - method"), ""
    if review:
        caveat = ("collapsed technical_application+review into 'Application review' - "
                   "no standalone 'technical review' category exists yet") if verdict["technical_application"] else ""
        return "Application review", caveat
    return ("Technical Methods; Application" if verdict["technical_application"] else "Application"), ""


def _stage_candidate(db: Database, record: dict, doi: str, verdict: dict) -> str:
    sheet = "method_pub" if verdict["paper_type"] == "method" else "AP_pub"
    prefix = "M_AUTO" if sheet == "method_pub" else "AP"
    entry_id = db.id_allocator.next_id(prefix)

    category, category_caveat = _category_field(verdict, is_review(record.get("publication_types", [])))
    fields = {
        "DOI": doi,
        "title": record.get("title", ""),
        "year": record.get("year"),
        "journal": record.get("journal", ""),
        "category": category,
        "REVIEW_STATUS": config.REVIEW_STATUS_SCRAPED,
    }
    if record.get("doi") and record.get("doi") != doi:
        # Only true for a preprint whose published DOI we staged under
        # instead - keep the original so nothing about how this was found
        # is lost.
        fields["preprint_doi"] = record["doi"]

    notes = verdict["reason"]
    if category_caveat:
        notes += f" | NOTE: {category_caveat}"
    if category_pass_is_confident(verdict):
        # category_confidence/categories came from the SAME LLM call as
        # relevant/paper_type above (merged 2026-09-17, see relevance.py's
        # module docstring) - below CONFIDENCE_FLOOR or empty, just leave
        # pipeline_category blank rather than stage a low-confidence guess.
        fields["pipeline_category"] = ";".join(verdict["categories"])
        notes += (f" | category tag ({verdict['category_confidence']:.2f} "
                  f"confidence): {verdict['categories']}")

    staging.append_candidate(
        action="create_entry",
        sheet=sheet,
        entry_id=entry_id,
        fields=fields,
        source_paper_entry_id="",
        curation_agent=CURATION_AGENT_NAME,
        curation_model=verdict["model_used"],
        confidence=None,
        notes=notes,
    )
    db.doi_index.add(doi, entry_id)
    return entry_id


def _evaluate(record: dict, source_label: str, db: Database, ledger: seen_ledger.SeenLedger) -> str:
    """Runs one record through the full funnel. Returns what happened:
    'seen' / 'already_in_db' / 'rejected' / 'staged' / 'llm_error'."""
    doi = _canonical_doi(record)
    native_id = f"pmid:{record['pmid']}" if record.get("pmid") else ""
    if not doi and not native_id:
        return "skipped_no_identity"
    key = seen_ledger.make_key(doi=doi, native_id=native_id)

    if ledger.is_seen(key):
        return "seen"

    if doi and doi in db.doi_index:
        ledger.mark(key, seen_ledger.STATUS_ALREADY_IN_DB, source=source_label, title=record.get("title", ""))
        return "already_in_db"

    reject_reason = publication_type_reject_reason(record.get("publication_types", []))
    if reject_reason:
        ledger.mark(key, seen_ledger.STATUS_REJECTED, source=source_label,
                     title=record.get("title", ""), notes=reject_reason)
        return "rejected"

    matched_kw = keyword_prefilter(record.get("title", ""), record.get("abstract", ""))
    if not matched_kw:
        ledger.mark(key, seen_ledger.STATUS_REJECTED, source=source_label,
                     title=record.get("title", ""), notes="failed keyword prefilter")
        return "rejected"

    try:
        verdict = llm_relevance_pass(record.get("title", ""), record.get("abstract", ""))
    except LLMError as e:
        # Don't mark the ledger - leave it unseen so a later run (once the
        # LLM chain recovers) retries this one instead of silently losing it.
        print(f"[scan] LLM relevance check failed for {record.get('title', '')[:60]!r}: {e}")
        return "llm_error"

    if not verdict["relevant"] or verdict["paper_type"] not in ("method", "application"):
        ledger.mark(key, seen_ledger.STATUS_REJECTED, source=source_label,
                     title=record.get("title", ""), notes=verdict["reason"])
        return "rejected"

    entry_id = _stage_candidate(db, record, doi, verdict)
    ledger.mark(key, seen_ledger.STATUS_STAGED, source=source_label,
                 title=record.get("title", ""), notes=f"staged as {entry_id}")
    return "staged"


def _run_funnel(records: list, source_label: str, db: Database, ledger: seen_ledger.SeenLedger,
                 max_new: int = None) -> dict:
    counts = {}
    staged = 0
    for record in records:
        outcome = _evaluate(record, source_label, db, ledger)
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "staged":
            staged += 1
            if max_new is not None and staged >= max_new:
                print(f"[scan] Reached max_new={max_new}, stopping early "
                      f"({len(records)} candidates fetched total).")
                break
    return counts


def scan_pubmed(query: str, mindate: str, maxdate: str, max_new: int = None) -> dict:
    db = Database()
    ledger = seen_ledger.SeenLedger()
    print(f"[scan] PubMed: querying {mindate}-{maxdate} for {query!r}...")
    records = pubmed_client.search_and_fetch(query, mindate, maxdate)
    print(f"[scan] PubMed: fetched {len(records)} candidate records.")
    return _run_funnel(records, "pubmed", db, ledger, max_new=max_new)


def scan_biorxiv(start_date: str, end_date: str, server: str = "biorxiv", max_new: int = None) -> dict:
    db = Database()
    ledger = seen_ledger.SeenLedger()
    print(f"[scan] {server}: fetching {start_date}..{end_date}...")
    raw = biorxiv_client.fetch_window(server, start_date, end_date)
    records = [biorxiv_client.normalize(r) for r in raw]
    print(f"[scan] {server}: fetched {len(records)} candidate records.")
    return _run_funnel(records, server, db, ledger, max_new=max_new)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)

    p_pubmed = sub.add_parser("pubmed")
    p_pubmed.add_argument("--query", required=True)
    p_pubmed.add_argument("--mindate", required=True, help="YYYY/MM/DD")
    p_pubmed.add_argument("--maxdate", required=True, help="YYYY/MM/DD")
    p_pubmed.add_argument("--max-new", type=int, default=None)

    p_biorxiv = sub.add_parser("biorxiv")
    p_biorxiv.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_biorxiv.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_biorxiv.add_argument("--medrxiv", action="store_true", help="scan medRxiv instead of bioRxiv")
    p_biorxiv.add_argument("--max-new", type=int, default=None)

    args = parser.parse_args()

    if args.source == "pubmed":
        counts = scan_pubmed(args.query, args.mindate, args.maxdate, max_new=args.max_new)
    else:
        counts = scan_biorxiv(args.start, args.end, server="medrxiv" if args.medrxiv else "biorxiv",
                                max_new=args.max_new)

    print(f"[scan] Done. {counts}")


if __name__ == "__main__":
    _main()
