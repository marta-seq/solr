"""
seen_ledger.py
Persistent "have we already evaluated this paper" record for the
literature-search scanner - the actual dedup guarantee, NOT the date window
a scan runs against. A date window only scopes how big one query is; it
can't stop the same paper turning up again across overlapping windows
(indexing lag near a window boundary), across different keyword queries in
the same run, or across the monthly preprint-republish recheck. Every one of
those re-encounters must resolve to "already evaluated, skip" without
spending another LLM relevance call - that's what this file is for.

Keyed by normalized DOI when available (see common/doi_utils.normalize_doi),
falling back to a source-native ID (pmid:<id> for PubMed records that
genuinely have no DOI - rare but real) since that's the only stable identity
those records have.

One JSON file, saved after every single mark() - same "never batch, always
flush" philosophy as staging.py, for the same reason: a scan can die mid-run
(LLM rate limit, network blip) and the ledger must reflect everything
actually evaluated so far, not lose it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from ...common import config
from ...common.doi_utils import normalize_doi

STATUS_STAGED = "staged"                    # became a new candidate in staging.xlsx
STATUS_ALREADY_IN_DB = "already_in_db"      # matched an existing entry_id via DoiIndex
STATUS_REJECTED = "rejected_not_relevant"    # failed the keyword or LLM relevance check


def _ledger_path() -> Path:
    return config.STAGING_DIR / "literature_search_seen.json"


def make_key(doi: str = "", native_id: str = "") -> str:
    """DOI wins whenever present - it's the identity the rest of the pipeline
    (DoiIndex, staging.xlsx) already keys on. native_id (e.g. "pmid:42746148")
    is only a fallback for the rare record with no DOI at all."""
    doi = normalize_doi(doi)
    if doi:
        return doi
    if native_id:
        return native_id
    raise ValueError("make_key() needs at least one of doi/native_id")


class SeenLedger:
    def __init__(self):
        self._path = _ledger_path()
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)

    def is_seen(self, key: str) -> bool:
        return key in self._data

    def status(self, key: str):
        rec = self._data.get(key)
        return rec["status"] if rec else None

    def mark(self, key: str, status: str, source: str, title: str = "", notes: str = ""):
        assert status in (STATUS_STAGED, STATUS_ALREADY_IN_DB, STATUS_REJECTED)
        self._data[key] = {
            "status": status,
            "source": source,
            "title": title,
            "notes": notes,
            "seen_date": datetime.now(timezone.utc).isoformat(),
        }
        self._save()  # <-- flushed immediately, same reasoning as staging.py

    def __len__(self):
        return len(self._data)

    def counts(self) -> dict:
        """Quick summary for end-of-scan logging."""
        out = {STATUS_STAGED: 0, STATUS_ALREADY_IN_DB: 0, STATUS_REJECTED: 0}
        for rec in self._data.values():
            out[rec["status"]] = out.get(rec["status"], 0) + 1
        return out
