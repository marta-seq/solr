"""
biorxiv_client.py
Wraps api.biorxiv.org's /details endpoint (covers both bioRxiv and medRxiv -
same API, different `server` value). Verified against the real API
(2026-09-17).

Two things worth knowing that shaped this module:
1. This endpoint has NO keyword or category query parameter - it only slices
   by date window + cursor pagination (30 records/page). Every record in the
   window comes back regardless of topic; ALL topic filtering (keyword
   prefilter, LLM relevance) has to happen downstream in relevance.py, not
   here. For a full window this means paginating through everything bioRxiv/
   medRxiv published in that window - fine for a monthly incremental scan,
   but worth sanity-checking real per-day volume before kicking off the full
   2014-present historical backfill in one go (see CLAUDE.md's backfill
   horizon note) - it may be worth chunking that backfill into smaller date
   windows run over several sessions rather than one long call.
2. Each record's own `published` field IS the preprint-published dedup
   mechanism, already maintained by bioRxiv itself: "NA" until a journal
   match is found, then the published version's DOI. Confirmed live: ~66%
   of a 15-month-old window already shows a match. This means the "monthly
   recheck for preprint->published pairing" the design doc called for is
   just re-fetching the same window later and diffing `published` against
   what the seen_ledger recorded last time - no separate matching heuristic
   needed.
"""

import time

import requests

from ...common import config

BASE_URL = "https://api.biorxiv.org/details"
PAGE_SIZE = 30  # fixed by the API, not configurable

_MIN_REQUEST_INTERVAL_S = 0.5  # no documented rate limit, but be polite


def is_published(record: dict) -> str:
    """Returns the published-version DOI if bioRxiv has matched one, else ''."""
    pub = record.get("published", "NA")
    return pub if pub and pub != "NA" else ""


def fetch_window(server: str, start_date: str, end_date: str, max_pages: int = None) -> list:
    """server: 'biorxiv' or 'medrxiv'. start_date/end_date: 'YYYY-MM-DD'.
    Paginates through every page in the window and returns the combined list.
    max_pages caps how many pages to fetch (for testing / a bounded first
    pass) - None means fetch the whole window."""
    assert server in ("biorxiv", "medrxiv")
    records = []
    cursor = 0
    page = 0
    while True:
        time.sleep(_MIN_REQUEST_INTERVAL_S)
        resp = requests.get(f"{BASE_URL}/{server}/{start_date}/{end_date}/{cursor}",
                             timeout=config.FETCH_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        page_records = data.get("collection", [])
        records.extend(page_records)
        page += 1

        msg = data["messages"][0]
        total = int(msg.get("total", 0))
        cursor += PAGE_SIZE
        if cursor >= total or not page_records:
            break
        if max_pages is not None and page >= max_pages:
            break
    return records


def normalize(record: dict) -> dict:
    """Reshapes a raw bioRxiv/medRxiv record into the same field names
    pubmed_client.fetch_details returns, so relevance.py/scan.py can treat
    both sources uniformly."""
    return {
        "pmid": "",
        "doi": record.get("doi", ""),
        "title": record.get("title", ""),
        "journal": "bioRxiv" if record.get("server") == "bioRxiv" else "medRxiv",
        "year": (record.get("date") or "")[:4] or None,
        "abstract": record.get("abstract", ""),
        "authors": [a.strip() for a in (record.get("authors") or "").split(";") if a.strip()],
        "source": (record.get("server") or "").lower() or "biorxiv",
        "published_doi": is_published(record),
    }
