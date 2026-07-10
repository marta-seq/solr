"""
reference_resolver.py
Shared by both desk agents: given a citation as written (a reference-list
entry, or a shorter "Author et al., Journal, Year" fragment), tries to
resolve it to a DOI via Crossref's bibliographic search, then fuzzy-matches
the returned title against the citation text to judge how confident we
should be.

This is necessarily best-effort - citation text is messy (OCR artifacts from
PDF extraction, inconsistent formatting, truncated author lists). Callers
should treat low-confidence results as "flag for manual review", not "trust
and create an entry".
"""

import re
from difflib import SequenceMatcher

import requests

from . import config
from .doi_utils import normalize_doi
from .reference_list_parser import lookup_reference, extract_embedded_doi

CROSSREF_SEARCH_URL = "https://api.crossref.org/works"
HEADERS = {"User-Agent": f"solr-living-review/1.0 (mailto:{config.CONTACT_EMAIL})"}

# Below this, don't trust the match - return it but flag low confidence
CONFIDENCE_FLOOR = 0.55


def _clean_citation_text(text: str) -> str:
    """Strips numbering/brackets ('[12] ', '12. ') and collapses whitespace -
    Crossref's bibliographic search works better on the "meat" of the
    citation than on raw reference-list formatting."""
    text = re.sub(r"^\s*\[?\d{1,3}\]?[.)]?\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _title_similarity(candidate_title: str, citation_text: str) -> float:
    """0-1 similarity between a candidate's title and the citation text.
    Uses containment-style scoring since citation_text is a whole reference
    (author, journal, year) while candidate_title is just the title - a
    direct ratio would unfairly penalize exact title matches buried in a
    longer citation string."""
    candidate_title = candidate_title.lower().strip()
    citation_lower = citation_text.lower()
    if not candidate_title:
        return 0.0
    if candidate_title in citation_lower:
        return 1.0
    return SequenceMatcher(None, candidate_title, citation_lower).ratio()


def resolve_citation(citation_marker: str, citation_text: str, reference_map: dict) -> dict:
    """
    Main entry point for both desk agents. Tries, in order:
        1. Look up citation_marker (e.g. "12") in the paper's own parsed
           reference list -> if that entry already prints a DOI, use it
           directly at full confidence - no LLM recall, no fuzzy matching.
        2. If the reference list entry exists but has no embedded DOI,
           resolve THAT exact text via Crossref search (still better than
           the LLM's recollection, since it's the paper's real wording).
        3. If there's no reference list (author-year style papers, or no
           references section could be isolated) or the marker wasn't
           found, fall back to resolving the LLM's citation_text guess.
    """
    ref_text = lookup_reference(citation_marker, reference_map) if reference_map else ""

    if ref_text:
        embedded_doi = extract_embedded_doi(ref_text)
        if embedded_doi:
            return {
                "input_text": ref_text,
                "resolved_doi": embedded_doi,
                "matched_title": "",
                "confidence": 1.0,
                "source": "reference_list_embedded_doi",
            }
        result = resolve_reference(ref_text)
        result["source"] = "reference_list_text_via_" + result["source"]
        return result

    # no usable reference-list entry - fall back to the LLM's recalled text
    result = resolve_reference(citation_text)
    result["source"] = "llm_recalled_text_via_" + result["source"]
    return result


def resolve_reference(citation_text: str, timeout: int = None) -> dict:
    """
    Returns:
        {
            "input_text": original citation_text,
            "resolved_doi": normalized DOI string, or "" if nothing usable found,
            "matched_title": the Crossref title that was matched, or "",
            "confidence": 0-1 float,
            "source": "crossref_bibliographic_search" | "none",
        }
    """
    timeout = timeout or config.FETCH_TIMEOUT_S
    cleaned = _clean_citation_text(citation_text)

    result = {
        "input_text": citation_text,
        "resolved_doi": "",
        "matched_title": "",
        "confidence": 0.0,
        "source": "none",
    }

    if not cleaned or len(cleaned) < 8:
        return result

    try:
        r = requests.get(
            CROSSREF_SEARCH_URL,
            params={"query.bibliographic": cleaned, "rows": 3},
            headers=HEADERS, timeout=timeout,
        )
        if r.status_code != 200:
            return result
        items = r.json().get("message", {}).get("items", [])
    except (requests.RequestException, ValueError):
        return result

    if not items:
        return result

    best = None
    best_score = 0.0
    for item in items:
        titles = item.get("title", [])
        if not titles:
            continue
        score = _title_similarity(titles[0], cleaned)
        if score > best_score:
            best_score = score
            best = item

    if best is None:
        return result

    result["resolved_doi"] = normalize_doi(best.get("DOI", ""))
    result["matched_title"] = best.get("title", [""])[0]
    result["confidence"] = round(best_score, 3)
    result["source"] = "crossref_bibliographic_search"
    return result


def is_confident(resolved: dict) -> bool:
    return bool(resolved.get("resolved_doi")) and resolved.get("confidence", 0) >= CONFIDENCE_FLOOR
