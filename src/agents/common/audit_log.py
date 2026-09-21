"""
audit_log.py
Persistent, append-only record of what the LLM actually said and what the
system did with it - one JSON line per LLM extraction call, in
data/agent_review/llm_audit_log.jsonl. Added 2026-09-21, per Marta's ask
after finding staging.xlsx only shows the FINAL outcome (which ID got
linked/created), not the evidence for why - e.g. which specific extracted
item resolved to which existing entry, or why an item got skipped instead of
staged (no identifier, low confidence, etc.).

Two things worth knowing about this before relying on it:
1. This is a debug/audit trail, not a candidate-review file like
   staging.xlsx - it's never read by merge_candidates.py or any other part
   of the pipeline, purely for humans investigating "what actually happened
   on this paper" after the fact.
2. Also captures the real practical benefit Marta asked about: since the raw
   extracted items are saved here, resolution logic can be re-run against
   them later (e.g. after a bugfix) WITHOUT spending another LLM call - the
   expensive part (the LLM call itself) never needs repeating just because
   the downstream matching/resolution code improved.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _log_path() -> Path:
    return config.STAGING_DIR / "llm_audit_log.jsonl"


def log_extraction(agent: str, source_paper_entry_id: str, model_used: str,
                    raw_extracted: list, item_outcomes: list) -> None:
    """
    agent: "compared_methods_agent" | "data_fetch_agent"
    source_paper_entry_id: the seed paper this extraction ran against
    model_used: the actual model that served the call (e.g. "gemini-3.5-flash")
    raw_extracted: the LLM's own parsed JSON output, unmodified - exactly
        what it said, before any resolution/matching happened
    item_outcomes: one dict per item in raw_extracted (same order), each
        describing what the system did with it - e.g.
        {"outcome": "linked_existing", "entry_id": "D_SP_IMC_5", "matched_by": "accession"}
        {"outcome": "created_new", "entry_id": "M_AUTO_374", "confidence": 0.3}
        {"outcome": "skipped", "reason": "no accession/DOI/access_link on this mention"}
        {"outcome": "skipped", "reason": "could not resolve a DOI for this citation"}

    Appended, never overwritten - one line per call, safe to append from a
    long-running pipeline that may be interrupted mid-run (same reasoning as
    staging.py's own save-after-every-candidate design).
    """
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "source_paper_entry_id": source_paper_entry_id,
        "model_used": model_used,
        "raw_extracted": raw_extracted,
        "item_outcomes": item_outcomes,
    }
    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
