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
from .doi_utils import normalize_doi, bare_doi
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


def _fetch_doi_title(doi: str, timeout: int = None) -> str:
    """Fetches the REAL title Crossref has on file for a DOI - used to sanity-
    check an embedded DOI actually belongs to the reference text it was
    extracted from, rather than blindly trusting it. Returns '' on any
    failure (caller should treat that as 'could not verify', not 'confirmed
    wrong' - Crossref doesn't have every DOI, especially non-journal ones)."""
    timeout = timeout or config.FETCH_TIMEOUT_S
    try:
        r = requests.get(
            f"{CROSSREF_SEARCH_URL}/{bare_doi(doi)}",
            headers=HEADERS, timeout=timeout,
        )
        if r.status_code != 200:
            return ""
        titles = r.json().get("message", {}).get("title", [])
        return titles[0] if titles else ""
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return ""


def resolve_citation(citation_marker: str, citation_text: str, reference_map: dict,
                      expected_name_hint: str = None) -> dict:
    """
    Main entry point for both desk agents. Tries, in order:
        1. Look up citation_marker (e.g. "12") in the paper's own parsed
           reference list -> if that entry already prints a DOI, verify it
           by fetching the DOI's real title from Crossref and checking it
           actually appears in the reference text before trusting it fully.
        2. If the reference list entry exists but has no embedded DOI,
           resolve THAT exact text via Crossref search (still better than
           the LLM's recollection, since it's the paper's real wording).
        3. If there's no reference list (author-year style papers, or no
           references section could be isolated) or the marker wasn't
           found, fall back to resolving the LLM's citation_text guess.

    expected_name_hint: pass the method/tool name the LLM identified (e.g.
    "StarDist") when you have one, to catch a DIFFERENT failure mode than
    the internal DOI-vs-its-own-reference-text check above: the reference-
    list marker-to-entry MAPPING itself grabbing the wrong numbered entry
    entirely (caught via testing on real output - a "StarDist" mention
    resolved to a confidently-wrong 3D-printer paper whose own DOI/title
    were perfectly self-consistent, just for the WRONG reference). If the
    hint is given and doesn't appear anywhere in what we resolved to,
    confidence is capped regardless of how the internal checks went.
    """
    ref_text = lookup_reference(citation_marker, reference_map) if reference_map else ""

    if ref_text:
        embedded_doi = extract_embedded_doi(ref_text)
        if embedded_doi:
            real_title = _fetch_doi_title(embedded_doi)
            if real_title and _title_similarity(real_title, ref_text) >= 0.3:
                confidence = 1.0
            elif real_title:
                confidence = 0.3
            else:
                confidence = 0.7
            confidence = _apply_name_hint_check(expected_name_hint, ref_text, real_title, confidence)
            return {
                "input_text": ref_text,
                "resolved_doi": embedded_doi,
                "matched_title": real_title,
                "confidence": confidence,
                "source": "reference_list_embedded_doi",
            }
        result = resolve_reference(ref_text)
        result["confidence"] = _apply_name_hint_check(
            expected_name_hint, ref_text, result["matched_title"], result["confidence"])
        result["source"] = "reference_list_text_via_" + result["source"]
        return result

    # no usable reference-list entry - fall back to the LLM's recalled text
    result = resolve_reference(citation_text)
    result["confidence"] = _apply_name_hint_check(
        expected_name_hint, citation_text, result["matched_title"], result["confidence"])
    result["source"] = "llm_recalled_text_via_" + result["source"]
    return result


def _apply_name_hint_check(expected_name_hint: str, ref_text: str, matched_title: str,
                            current_confidence: float) -> float:
    """If a method-name hint was given and it doesn't appear anywhere in
    either the reference text or the matched title, cap confidence - this is
    an independent sanity check on top of whatever internal consistency
    checks already ran, since it catches the resolved reference being
    entirely the wrong one, not just internally self-inconsistent."""
    if not expected_name_hint:
        return current_confidence
    haystack = f"{ref_text} {matched_title}".lower()
    if expected_name_hint.strip().lower() not in haystack:
        return min(current_confidence, 0.35)
    return current_confidence


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
            params={"query.bibliographic": cleaned, "rows": 5},
            headers=HEADERS, timeout=timeout,
        )
        if r.status_code != 200:
            return result
        items = r.json().get("message", {}).get("items", [])
    except (requests.RequestException, ValueError):
        return result

    # Crossref registers DOIs for more than just full papers - individual
    # figures, tables, and supplementary files within an article get their
    # own "component" DOIs, and can score a good title match without being
    # the paper we actually want (caught via testing: a figure's DOI got
    # matched instead of the article's own DOI). Only accept types that are
    # genuinely a paper/preprint - everything else is discarded even if its
    # title similarity score would have won.
    ACCEPTED_TYPES = {
        "journal-article", "posted-content", "proceedings-article",
        "book-chapter", "monograph", "reference-entry", "report",
    }
    items = [item for item in items if item.get("type") in ACCEPTED_TYPES]

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
