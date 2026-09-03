"""
run_pipeline.py
The department manager. Ties everything together:

    1. Load the DB (methods_metadata.csv + datasets.csv), build the shared
       DoiIndex/IdAllocator.
    2. Build the paper-recursion queue (mailroom.triage.build_seed_queue) and
       the independent data-metadata pool (mailroom.triage.build_data_pool).
    3. Work through the paper queue: for each paper, fetch its text ONCE and
       share it between the methods desk and the data desk (both need the
       same paper), respecting MAX_HOPS and MAX_PAPERS_PER_RUN as hard caps.
    4. Separately, work through the data pool with the intern agent - this
       track is independent and doesn't touch the paper queue at all.

Everything is staged via common.staging - nothing here ever touches the
master Excel directly. Run this, then open data/agent_review/staging.xlsx
to review before merging anything back in.

Usage:
    python -m src.agents.run_pipeline
"""

import argparse
import sys
import time

# Bump this string every time this file changes. Printed as the very first
# line of every run, specifically so stale-file/bytecode-cache issues (which
# have happened more than once) are immediately visible in the log itself,
# rather than needing a separate manual grep check to confirm which version
# is actually running.
CODE_VERSION = "2026-08-08-r18-stop-early-on-llm-exhaustion"

from .common import config, db_loader
from .living_ingestion.mailroom import triage
from .agent_curation.methods_desk import compared_methods_agent
from .agent_curation.data_desk import data_fetch_agent, intern_agent
from .common.paper_fetcher import fetch_paper, get_agent_text
from .common.reference_list_parser import parse_reference_list


def _log(msg: str) -> None:
    print(f"[run_pipeline] {msg}", flush=True)


def run_paper_queue(db, budget: int) -> tuple:
    """
    Works through the compared-methods/data-fetch queue starting from
    build_seed_queue(), recursing into newly-discovered method papers up to
    MAX_HOPS, and stopping when `budget` (a share of MAX_PAPERS_PER_RUN) runs
    out - see main() for how the total budget is split across phases.

    Returns (summary_dict, papers_actually_used).
    """
    queue = triage.build_seed_queue(db.methods)
    _log(f"Seed queue: {len(queue)} papers at depth 0")

    processed = 0
    skipped_depth = 0
    summary = {"processed": [], "new_method_entries": [], "data_entries_linked": [], "llm_exhausted": False}

    while queue and processed < budget:
        paper_entry = queue.pop(0)

        if paper_entry["depth"] > config.MAX_HOPS:
            skipped_depth += 1
            continue

        entry_id = paper_entry["entry_id"]
        _log(f"[{processed + 1}/{budget}] {entry_id} (depth {paper_entry['depth']})")

        # Fetch once, share between both desks - they both need this paper's text
        fetched = fetch_paper(paper_entry["doi"])
        if fetched["source"] == "none":
            _log(f"  FAILED: no text could be fetched from any source for {entry_id} - "
                 f"both desks will skip, staged as needs_review")
        references_text, _ = get_agent_text(fetched, "references")
        reference_map = parse_reference_list(references_text) if references_text else {}

        if fetched["source"] != "none":
            _, methods_text_source = get_agent_text(fetched, "methods")
            if methods_text_source == "section:methods":
                _log(f"  methods section: ISOLATED correctly - proceeding to check with the LLM")
            elif methods_text_source == "full_text_fallback":
                _log(f"  methods section: NOT found - would need to send unfocused full-paper "
                     f"text instead, which REQUIRE_ISOLATED_SECTION blocks by default")
            elif methods_text_source == "abstract_only":
                _log(f"  methods section: NOT found, and no full text either - only an abstract "
                     f"is available, which REQUIRE_ISOLATED_SECTION blocks by default")
            else:
                _log(f"  methods section: no usable text at all")

            _, data_avail_text_source = get_agent_text(fetched, "data_availability")
            if data_avail_text_source == "section:data_availability":
                _log(f"  data-availability section: ISOLATED correctly - proceeding to check with the LLM")
            elif data_avail_text_source == "full_text_fallback":
                _log(f"  data-availability section: NOT found - would need to send unfocused "
                     f"full-paper text instead, which REQUIRE_ISOLATED_SECTION blocks by default")
            elif data_avail_text_source == "abstract_only":
                _log(f"  data-availability section: NOT found, and no full text either - only "
                     f"an abstract is available, which REQUIRE_ISOLATED_SECTION blocks by default")
            else:
                _log(f"  data-availability section: no usable text at all")

        methods_ok = True
        try:
            methods_result = compared_methods_agent.process_paper(
                db, paper_entry, fetched=fetched, reference_map=reference_map
            )
        except Exception as e:
            _log(f"  methods desk CRASHED on {entry_id}: {e}")
            methods_result = {"new_queue_items": [], "total_encountered": 0, "total_resolved": 0,
                               "skip_reason": f"crashed: {e}"}
            methods_ok = False

        data_ok = True
        try:
            data_result = data_fetch_agent.process_paper(
                db, paper_entry, fetched=fetched, reference_map=reference_map
            )
        except Exception as e:
            _log(f"  data desk CRASHED on {entry_id}: {e}")
            data_result = {"linked_ids": [], "total_encountered": 0, "total_resolved": 0,
                            "skip_reason": f"crashed: {e}"}
            data_ok = False

        new_items = methods_result["new_queue_items"]
        linked = data_result["linked_ids"]

        if fetched["source"] != "none":
            status = "OK" if (methods_ok and data_ok) else "PARTIAL FAILURE"
            _log(f"  {status}: fetched via {fetched['source']}")

            if methods_result["skip_reason"] is not None:
                _log(f"    methods desk: SKIPPED - LLM was NOT called ({methods_result['skip_reason']})")
            else:
                _log(f"    methods desk: LLM WAS called - encountered "
                     f"{methods_result['total_encountered']} compared method(s) -> "
                     f"{methods_result['total_resolved']} resolved (matched or created), "
                     f"{len(new_items)} genuinely NEW entries")

            if data_result["skip_reason"] is not None:
                _log(f"    data desk: SKIPPED - LLM was NOT called ({data_result['skip_reason']})")
            else:
                _log(f"    data desk: LLM WAS called - encountered "
                     f"{data_result['total_encountered']} dataset mention(s) -> "
                     f"{data_result['total_resolved']} resolved (matched or created)")


        summary["processed"].append(entry_id)
        summary["new_method_entries"].extend(new_items)
        summary["data_entries_linked"].extend(linked)

        for item in new_items:
            if item["depth"] <= config.MAX_HOPS:
                queue.append(item)

        processed += 1

        # methods_result/data_result set llm_exhausted=True ONLY when
        # LLMExhaustedError was raised - i.e. every single provider/model in
        # the whole fallback chain failed for this call, not just one model
        # having a bad moment. Every remaining paper would hit the exact
        # same dead chain and burn the exact same multi-minute wait for a
        # predictably identical result, so stop the run here rather than
        # grinding through the rest of the queue for nothing. Re-run once
        # you expect the outage/quota to have cleared (e.g. OpenRouter's
        # free-tier daily cap resets the next day).
        if methods_result.get("llm_exhausted") or data_result.get("llm_exhausted"):
            _log(f"  LLM fallback chain is FULLY EXHAUSTED (every provider/model failed) - "
                 f"stopping this run early instead of repeating the same doomed wait on each "
                 f"of the {len(queue)} remaining paper(s). Re-run later once the outage/quota "
                 f"has cleared.")
            summary["llm_exhausted"] = True
            break

    if queue:
        _log(f"Stopped paper queue at its budget ({budget}) with {len(queue)} "
             f"papers still queued - re-run to continue.")
    if skipped_depth:
        _log(f"Skipped {skipped_depth} papers that exceeded MAX_HOPS ({config.MAX_HOPS}).")

    return summary, processed


def run_data_pool(db, budget: int) -> tuple:
    """Independent track: sweeps every Data_* entry with a metadata gap and
    tries to fill it from the associated paper's abstract. Not affected by
    MAX_HOPS (there's no recursion here), but shares the same overall
    MAX_PAPERS_PER_RUN budget as the paper queue - see main().

    Returns (summary_dict, entries_actually_used).
    """
    pool = triage.build_data_pool(db.datasets)
    _log(f"Data pool: {len(pool)} entries with metadata gaps")

    summary = {"filled": [], "skipped": [], "llm_exhausted": False}
    used = 0

    for item in pool:
        if used >= budget:
            _log(f"Stopped data pool at its remaining budget ({budget}) - "
                 f"{len(pool) - used} entries remain.")
            break

        _log(f"[{used + 1}/{budget}] {item['entry_id']} "
             f"(missing: {', '.join(item['missing_fields'])})")

        try:
            result = intern_agent.process_entry(db, item)
        except Exception as e:
            _log(f"  CRASHED on {item['entry_id']}: {e}")
            result = {"filled": [], "skipped_reason": str(e)}

        if result["filled"]:
            _log(f"  OK: filled {result['filled']}")
            summary["filled"].append((item["entry_id"], result["filled"]))
        else:
            _log(f"  SKIPPED: {result['skipped_reason']}")
            summary["skipped"].append((item["entry_id"], result["skipped_reason"]))

        used += 1

        # Same reasoning as run_paper_queue's identical check - a fully
        # exhausted LLM chain would fail the exact same way on every
        # remaining data-pool entry too, so stop here instead of grinding
        # through the rest for a predictably identical result.
        if result.get("llm_exhausted"):
            _log(f"  LLM fallback chain is FULLY EXHAUSTED (every provider/model failed) - "
                 f"stopping the data pool early instead of repeating the same doomed wait on "
                 f"each of the {len(pool) - used} remaining entries. Re-run later once the "
                 f"outage/quota has cleared.")
            summary["llm_exhausted"] = True
            break

    return summary, used


def main():
    parser = argparse.ArgumentParser(description="Run the SOLR Phase 2 agent pipeline.")
    parser.add_argument(
        "--max-papers", type=int, default=None,
        help="Override config.MAX_PAPERS_PER_RUN for this run only, without editing config.py "
             "(e.g. --max-papers 15 for a quick test)."
    )
    args = parser.parse_args()

    _log(f"=== CODE_VERSION: {CODE_VERSION} === (if this doesn't match what you expect, "
         f"you're running stale files - re-sync src/agents/ before trusting anything below)")
    _log("Loading database...")
    db = db_loader.Database()
    _log(f"Loaded {len(db.methods)} papers, {len(db.datasets)} dataset entries, "
         f"{len(db.doi_index)} known DOIs.")

    start = time.time()
    total_budget = args.max_papers if args.max_papers is not None else config.MAX_PAPERS_PER_RUN
    _log(f"Total shared budget for this run: {total_budget} "
         f"(paper queue and data pool draw from the same pool, in that order)"
         f"{' [overridden via --max-papers]' if args.max_papers is not None else ''}.")

    paper_summary, papers_used = run_paper_queue(db, budget=total_budget)
    _log(f"Paper queue done: {len(paper_summary['processed'])} papers processed, "
         f"{len(paper_summary['new_method_entries'])} new method entries, "
         f"{len(paper_summary['data_entries_linked'])} dataset entries linked/created.")

    if paper_summary.get("llm_exhausted"):
        _log("Skipping the data pool this run too - it would hit the exact same exhausted "
             "LLM chain and fail identically on every entry there as well.")
        data_summary, entries_used = {"filled": [], "skipped": [], "llm_exhausted": True}, 0
    else:
        remaining_budget = total_budget - papers_used
        data_summary, entries_used = run_data_pool(db, budget=remaining_budget)
    _log(f"Data pool done: {len(data_summary['filled'])} entries filled, "
         f"{len(data_summary['skipped'])} skipped.")
    if paper_summary.get("llm_exhausted") or data_summary.get("llm_exhausted"):
        _log("NOTE: this run ended early because the LLM fallback chain was fully exhausted, "
             "not because it ran out of papers/budget - there's likely still queued work left. "
             "Re-run once you expect the outage/quota to have cleared.")

    elapsed = time.time() - start
    _log(f"Total time: {elapsed:.1f}s "
         f"({papers_used + entries_used}/{total_budget} of shared budget used).")
    _log(f"Review staged candidates in data/agent_review/staging.xlsx before merging.")


if __name__ == "__main__":
    sys.exit(main())
