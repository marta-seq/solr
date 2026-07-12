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
master Excel directly. Run this, then open data/agent_review/staging_<date>.xlsx
to review before merging anything back in.

Usage:
    python -m src.agents.run_pipeline
"""

import sys
import time

from .common import config, db_loader
from .mailroom import triage
from .methods_desk import compared_methods_agent
from .data_desk import data_fetch_agent, intern_agent
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
    summary = {"processed": [], "new_method_entries": [], "data_entries_linked": []}

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
            _log(f"  methods input: {methods_text_source or 'NONE'} "
                 f"(section:methods = isolated correctly; full_text_fallback = heading "
                 f"not found, sent from start of paper - may miss late-paper comparisons; "
                 f"abstract_only = weakest signal)")

        methods_ok = True
        try:
            new_items = compared_methods_agent.process_paper(
                db, paper_entry, fetched=fetched, reference_map=reference_map
            )
        except Exception as e:
            _log(f"  methods desk CRASHED on {entry_id}: {e}")
            new_items = []
            methods_ok = False

        data_ok = True
        try:
            linked = data_fetch_agent.process_paper(
                db, paper_entry, fetched=fetched, reference_map=reference_map
            )
        except Exception as e:
            _log(f"  data desk CRASHED on {entry_id}: {e}")
            linked = []
            data_ok = False

        if fetched["source"] != "none":
            status = "OK" if (methods_ok and data_ok) else "PARTIAL FAILURE"
            _log(f"  {status}: fetched via {fetched['source']} | "
                 f"methods desk created {len(new_items)} NEW method entr{'y' if len(new_items)==1 else 'ies'} "
                 f"(matches to EXISTING papers are linked directly and not counted here) | "
                 f"data desk created/linked {len(linked)} dataset entr{'y' if len(linked)==1 else 'ies'} "
                 f"(check Method_comparison_P_ENTRY_ID / DataID on {entry_id} in staging.xlsx for the full "
                 f"picture including existing-entry links; 0 everywhere can also mean an LLM error already "
                 f"staged as needs_review)")

        summary["processed"].append(entry_id)
        summary["new_method_entries"].extend(new_items)
        summary["data_entries_linked"].extend(linked)

        for item in new_items:
            if item["depth"] <= config.MAX_HOPS:
                queue.append(item)

        processed += 1

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

    summary = {"filled": [], "skipped": []}
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

    return summary, used


def main():
    _log("Loading database...")
    db = db_loader.Database()
    _log(f"Loaded {len(db.methods)} papers, {len(db.datasets)} dataset entries, "
         f"{len(db.doi_index)} known DOIs.")

    start = time.time()
    total_budget = config.MAX_PAPERS_PER_RUN
    _log(f"Total shared budget for this run: {total_budget} "
         f"(paper queue and data pool draw from the same pool, in that order).")

    paper_summary, papers_used = run_paper_queue(db, budget=total_budget)
    _log(f"Paper queue done: {len(paper_summary['processed'])} papers processed, "
         f"{len(paper_summary['new_method_entries'])} new method entries, "
         f"{len(paper_summary['data_entries_linked'])} dataset entries linked/created.")

    remaining_budget = total_budget - papers_used
    data_summary, entries_used = run_data_pool(db, budget=remaining_budget)
    _log(f"Data pool done: {len(data_summary['filled'])} entries filled, "
         f"{len(data_summary['skipped'])} skipped.")

    elapsed = time.time() - start
    _log(f"Total time: {elapsed:.1f}s "
         f"({papers_used + entries_used}/{total_budget} of shared budget used).")
    _log(f"Review staged candidates in data/agent_review/staging_<date>.xlsx before merging.")


if __name__ == "__main__":
    sys.exit(main())
