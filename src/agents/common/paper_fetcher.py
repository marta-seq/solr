"""
paper_fetcher.py
Given a DOI, tries (in order) to get the fullest text available:
    1. Unpaywall          -> open-access PDF/HTML location, if one exists
    2. Europe PMC         -> full text for PMC-indexed papers
    3. bioRxiv/medRxiv    -> full text JSON for preprints
    4. Crossref           -> title + abstract only (last-resort fallback)

Every result is cached to disk keyed by normalized DOI, so re-running the
pipeline never re-fetches a paper it already has. Returns a dict:

    {
        "doi": "https://doi.org/...",
        "source": "unpaywall_pdf" | "europepmc" | "biorxiv" | "crossref_abstract_only",
        "is_full_text": bool,
        "text": "<all extracted text or just the abstract>",
        "sections": {"abstract": "...", "methods": "...", "data_availability": "..."},
        "fetched_at": "2026-07-10T..."
    }

If nothing at all could be retrieved, returns {"doi": ..., "source": "none",
"is_full_text": False, "text": "", "sections": {}, "fetched_at": ...}
"""

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config
from .doi_utils import normalize_doi, bare_doi

HEADERS = {"User-Agent": f"solr-living-review/1.0 (mailto:{config.CONTACT_EMAIL})"}

# Section header patterns used to naively split full text into rough sections.
# Real papers vary a lot in heading wording - this is best-effort, not exact.
SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "methods": r"\b(methods|materials and methods|methodology|experimental procedures)\b",
    "results": r"\bresults\b",
    "data_availability": r"\bdata availability\b",
    "references": r"\b(references|bibliography)\b",
}


def _cache_path(doi: str) -> Path:
    h = hashlib.sha256(normalize_doi(doi).encode()).hexdigest()[:20]
    return config.CACHE_DIR / f"{h}.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _empty_result(doi, source="none"):
    return {
        "doi": normalize_doi(doi),
        "source": source,
        "is_full_text": False,
        "text": "",
        "sections": {},
        "fetched_at": _now(),
    }


def _split_into_sections(text: str) -> dict:
    """Best-effort split of full text into named sections using header matches.
    Falls back to empty dict if nothing matches - callers should treat missing
    keys as 'not found' and fall back to full text."""
    if not text:
        return {}

    # find all header positions
    matches = []
    for name, pattern in SECTION_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # only consider it a real heading if it's short standalone-ish text
            # (crude heuristic: preceded by newline or start of doc)
            start = m.start()
            preceding = text[max(0, start - 2):start]
            if preceding in ("", "\n", "\n\n") or start < 5:
                matches.append((start, name))

    if not matches:
        return {}

    matches.sort()
    sections = {}
    for i, (start, name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections.setdefault(name, "")
        sections[name] += text[start:end].strip() + "\n"
    return sections


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed - run: pip install pdfplumber --break-system-packages")
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _try_unpaywall(doi: str):
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{bare_doi(doi)}",
            params={"email": config.CONTACT_EMAIL},
            headers=HEADERS, timeout=config.FETCH_TIMEOUT_S,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        loc = data.get("best_oa_location") or {}
        pdf_url = loc.get("url_for_pdf") or loc.get("url")
        if not pdf_url:
            return None
        pr = requests.get(pdf_url, headers=HEADERS, timeout=config.FETCH_TIMEOUT_S)
        if pr.status_code != 200 or not pr.content:
            return None
        if pdf_url.lower().endswith(".pdf") or "pdf" in pr.headers.get("Content-Type", ""):
            text = _extract_pdf_text(pr.content)
        else:
            # HTML landing page - very rough text strip
            text = re.sub(r"<[^>]+>", " ", pr.text)
        if text and len(text) > 500:
            return text
        return None
    except Exception:
        return None


def _try_europepmc(doi: str):
    try:
        bd = bare_doi(doi)
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": f"DOI:{bd}", "format": "json"},
            headers=HEADERS, timeout=config.FETCH_TIMEOUT_S,
        )
        results = r.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        pmcid = results[0].get("pmcid")
        if not pmcid:
            return None
        r2 = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            headers=HEADERS, timeout=config.FETCH_TIMEOUT_S,
        )
        if r2.status_code != 200:
            return None
        text = re.sub(r"<[^>]+>", " ", r2.text)
        return text if len(text) > 500 else None
    except Exception:
        return None


def _try_biorxiv(doi: str):
    try:
        bd = bare_doi(doi)
        r = requests.get(
            f"https://api.biorxiv.org/details/biorxiv/{bd}/na/json",
            timeout=config.FETCH_TIMEOUT_S,
        )
        collection = r.json().get("collection", [])
        if not collection:
            return None
        # bioRxiv's public API only reliably gives abstract, not full body text
        abstract = collection[0].get("abstract", "")
        return abstract if abstract else None
    except Exception:
        return None


def _try_crossref_abstract(doi: str):
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{bare_doi(doi)}",
            headers=HEADERS, timeout=config.FETCH_TIMEOUT_S,
        )
        if r.status_code != 200:
            return None
        msg = r.json().get("message", {})
        abstract = msg.get("abstract", "")
        abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
        return abstract if abstract else None
    except Exception:
        return None


def get_agent_text(paper: dict, section_name: str) -> tuple:
    """
    The one function every agent calls to get its input text - handles the
    fallback chain and the char cap in one place, so compared_methods_agent
    and data_fetch_agent don't each reinvent this.

    Fallback order:
        1. The named section (e.g. "methods", "data_availability"), if the
           heading regex found it.
        2. The full fetched text, if we have it but the heading wasn't found
           (common - the heuristic misses non-standard headings).
        3. Whatever text we have at all (even just an abstract).

    Returns (text, source_label) where source_label is one of:
        "section:<name>" | "full_text_fallback" | "abstract_only" | ""
    Always truncated to config.MAX_SECTION_CHARS - callers should treat a
    truncated result as lower-confidence (the agent may be missing context
    that ran past the cap) and can check with was_truncated().
    """
    sections = paper.get("sections") or {}
    text = paper.get("text") or ""

    if section_name in sections and sections[section_name].strip():
        chunk = sections[section_name]
        source = f"section:{section_name}"
    elif paper.get("is_full_text") and text.strip():
        chunk = text
        source = "full_text_fallback"
    elif text.strip():
        chunk = text
        source = "abstract_only"
    else:
        return "", ""

    return chunk[:config.MAX_SECTION_CHARS], source


def was_truncated(paper: dict, section_name: str) -> bool:
    """Cheap check so an agent can lower its own confidence/note it when the
    text it saw was cut off before the section actually ended."""
    text, _ = get_agent_text(paper, section_name)
    sections = paper.get("sections") or {}
    full = sections.get(section_name) or paper.get("text") or ""
    return len(full) > len(text)
    """Main entry point. Cached on disk per normalized DOI."""
    doi = normalize_doi(doi)
    if not doi:
        return _empty_result(doi)

    cache_file = _cache_path(doi)
    if cache_file.exists() and not force_refetch:
        return json.loads(cache_file.read_text())

    result = None

    text = _try_unpaywall(doi)
    if text:
        result = {"source": "unpaywall_pdf", "is_full_text": True, "text": text}

    if result is None:
        text = _try_europepmc(doi)
        if text:
            result = {"source": "europepmc", "is_full_text": True, "text": text}

    if result is None:
        text = _try_biorxiv(doi)
        if text:
            result = {"source": "biorxiv_abstract", "is_full_text": False, "text": text}

    if result is None:
        text = _try_crossref_abstract(doi)
        if text:
            result = {"source": "crossref_abstract_only", "is_full_text": False, "text": text}

    if result is None:
        result = {"source": "none", "is_full_text": False, "text": ""}

    result["doi"] = doi
    result["fetched_at"] = _now()
    result["sections"] = _split_into_sections(result["text"]) if result["is_full_text"] else {}

    cache_file.write_text(json.dumps(result, indent=2))
    return result
