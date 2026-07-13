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

IMPORTANT: HTML/XML from Unpaywall and Europe PMC is properly cleaned with
BeautifulSoup (nav/header/footer/script/style removed, Europe PMC's real
JATS <sec> tags parsed directly for section boundaries) rather than a blind
tag-strip regex. The blind-strip approach left large amounts of website
chrome (navigation menus, author-contribution-role lists, figure-download
widgets) mixed into the extracted text, which then confused section-heading
detection - caught via testing: a journal's nav-menu entry literally titled
"Data Availability" (a policy page link, not the paper's own data statement)
got captured as if it were the real section.
"""

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from . import config
from .doi_utils import normalize_doi, bare_doi

HEADERS = {"User-Agent": f"solr-living-review/1.0 (mailto:{config.CONTACT_EMAIL})"}

# Section header patterns used to naively split full text into rough sections
# when a source doesn't give us real structure (e.g. Unpaywall HTML/PDF).
# Real papers vary a lot in heading wording - this is best-effort, not exact.
SECTION_PATTERNS = {
    "abstract": r"\babstract\b",
    "methods": r"\b(methods|materials and methods|methodology|experimental procedures)\b",
    "results": r"\bresults\b",
    "data_availability": r"\bdata availability\b",
    "references": r"\b(references|bibliography)\b",
}

# Allows a heading to be preceded by section numbering/whitespace (e.g. "\n2. ",
# "\n\nIII. ", "\n3.1 ") rather than requiring the heading word be immediately
# preceded by a bare newline - real papers almost always number their
# sections, and the old strict check missed nearly all of them.
_HEADING_PRECEDING_RE = re.compile(r"^[\s\d.()ivxlc:-]{0,20}$", re.IGNORECASE)

# Chrome/boilerplate class-or-id name fragments seen across publisher sites -
# elements matching these get removed before extracting text.
_CHROME_HINTS = (
    "nav", "menu", "sidebar", "footer", "header", "cookie", "banner",
    "social", "share-", "breadcrumb", "advert", "subscribe", "paywall",
    "download-links", "figure-viewer", "citation-widget", "metrics-widget",
)

# Phrases that strongly indicate captured text is website navigation/footer
# boilerplate rather than actual article content - used as a pre-LLM sanity
# gate, since sending this to the LLM just wastes a call on garbage input.
_CHROME_MARKERS = [
    "reset zoom", "show in context", "advanced search",
    "article metrics are unavailable", "loading metrics", "editorial board",
    "guidelines for reviewers", "accepted manuscripts", "press and media",
    "publication fees", "peer review process", "editor center", "download:",
    "journal information", "corrections, expressions of concern",
]


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


_HEADING_KEYWORDS = {
    "abstract": ["abstract"],
    "methods": ["method", "materials and methods", "methodology", "experimental procedures"],
    "results": ["results"],
    "data_availability": ["data availability"],
    "references": ["references", "bibliography"],
}


def _classify_heading_text(text: str):
    """Matches a short heading string (not a whole document) against known
    section keywords. Returns the section key, or None if it's not a
    recognized heading at all - used to tell a real 'Methods' heading apart
    from unrelated short text."""
    t = text.strip().lower()
    if not t or len(t) > 60:  # a real heading is short - long text is body content
        return None
    for key, keywords in _HEADING_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return key
    return None


def _strip_chrome(soup: BeautifulSoup) -> None:
    """Removes navigation/header/footer/script/style/chrome elements in place."""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "form"]):
        tag.decompose()
    for tag in soup.find_all(True):
        classes = " ".join(tag.get("class", [])).lower()
        tag_id = (tag.get("id") or "").lower()
        if any(hint in classes or hint in tag_id for hint in _CHROME_HINTS):
            tag.decompose()




def _collect_text_between(start_tag, stop_tag) -> str:
    """Collects the actual leaf text between two tags in document order,
    using BeautifulSoup's .next_elements (a flat, document-order iterator
    over every subsequent node) so nested tags don't get double-counted -
    a naive descendant walk would re-collect text from both a <div> and
    its child <p> if done carelessly."""
    from bs4 import NavigableString
    collected = []
    for el in start_tag.next_elements:
        if stop_tag is not None and el is stop_tag:
            break
        if isinstance(el, NavigableString):
            s = str(el).strip()
            if s:
                collected.append(s)
    return " ".join(collected).strip()


def _parse_html_sections(html: str) -> tuple:
    """
    Tries to isolate real sections from HTML using its actual structure,
    rather than flattening to text first and regex-guessing where headings
    WERE (which is backwards - the semantic structure is usually still
    right there in the HTML, it's only lost once flattened):

        Tier 1: real <h1>-<h6> heading tags, classified by their own short
                text against known section keywords.
        Tier 2: only if tier 1 found nothing - some publishers style
                headings as plain <p>/<div>/<strong>/<b> instead of real
                heading tags, purely for visual formatting. Looks for such
                elements whose ENTIRE text is short (<=4 words) and matches
                a section keyword.

    Falls back to (flattened_text, {}) if neither tier finds anything -
    caller uses the existing regex-on-flattened-text approach in that case,
    worse but better than nothing.

    Returns (flattened_text, sections_dict).
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return re.sub(r"<[^>]+>", " ", html), {}

    _strip_chrome(soup)
    article = soup.find("article")
    scope = article if article else soup

    matched = []
    for h in scope.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        key = _classify_heading_text(h.get_text())
        if key:
            matched.append((h, key))

    if not matched:
        for tag in scope.find_all(["p", "div", "strong", "b", "span"]):
            own_text = tag.get_text(strip=True)
            if own_text and len(own_text.split()) <= 4:
                key = _classify_heading_text(own_text)
                if key:
                    matched.append((tag, key))

    flattened = scope.get_text(separator="\n", strip=True)
    flattened = re.sub(r"\n{3,}", "\n\n", flattened)

    if not matched:
        return flattened, {}

    sections = {}
    for i, (tag, key) in enumerate(matched):
        next_tag = matched[i + 1][0] if i + 1 < len(matched) else None
        text = _collect_text_between(tag, next_tag)
        if text:
            sections[key] = (sections.get(key, "") + "\n" + text).strip()

    return flattened, sections


def _parse_europepmc_xml(xml_text: str) -> tuple:
    """Europe PMC's fullTextXML is real JATS XML with structured <sec> tags -
    parse it directly instead of stripping tags and regex-guessing at
    headings on flattened text. Returns (flattened_text, sections_dict).
    Falls back to (stripped_text, {}) if parsing fails for any reason."""
    try:
        soup = BeautifulSoup(xml_text, "lxml-xml")
    except Exception:
        try:
            soup = BeautifulSoup(xml_text, "html.parser")
        except Exception:
            return re.sub(r"<[^>]+>", " ", xml_text), {}

    sections = {}

    abstract_tag = soup.find("abstract")
    if abstract_tag:
        sections["abstract"] = abstract_tag.get_text(separator=" ", strip=True)

    for sec in soup.find_all("sec"):
        sec_type = (sec.get("sec-type") or "").lower()
        title_tag = sec.find("title")
        title_text = title_tag.get_text(strip=True).lower() if title_tag else ""

        key = None
        if "method" in sec_type or "method" in title_text:
            key = "methods"
        elif ("data" in sec_type and "avail" in sec_type) or "data availability" in title_text:
            key = "data_availability"
        elif "result" in sec_type or "results" in title_text:
            key = "results"

        if key:
            text = sec.get_text(separator=" ", strip=True)
            sections[key] = (sections.get(key, "") + "\n" + text).strip()

    ref_list = soup.find("ref-list")
    if ref_list:
        refs = [f"{i}. " + ref.get_text(separator=" ", strip=True)
                for i, ref in enumerate(ref_list.find_all("ref"), 1)]
        if refs:
            sections["references"] = "\n".join(refs)

    flattened = soup.get_text(separator="\n", strip=True)
    flattened = re.sub(r"\n{3,}", "\n\n", flattened)
    return flattened, sections


def _split_into_sections(text: str) -> dict:
    """Best-effort split of full text into named sections using header matches.
    Used when a source doesn't give us real structure (Unpaywall HTML/PDF) -
    Europe PMC gets its sections from _parse_europepmc_xml instead, which is
    far more reliable since it reads the document's actual XML structure."""
    if not text:
        return {}

    matches = []
    for name, pattern in SECTION_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            # a real heading is preceded only by whitespace/numbering on ITS
            # OWN LINE - not preceded by real words (mentioned mid-sentence,
            # or part of an author-contribution-role list like "Formal
            # analysis,\n\nMethodology,\n\nSoftware," which also matches
            # "methodology" as a bare word but isn't a real heading).
            start = m.start()
            line_start = text.rfind("\n", 0, start)
            line_prefix = text[line_start + 1:start] if line_start != -1 else text[:start]
            if start < 5 or _HEADING_PRECEDING_RE.match(line_prefix):
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


def is_probably_real_content(text: str, min_length: int = 200) -> bool:
    """Cheap pre-LLM sanity gate: does this text actually look like article
    content, or did we capture website navigation/footer boilerplate
    instead? Check BEFORE spending an LLM call on it - a real problem this
    caught: a journal's nav-menu text (policy page links like "Ethical
    Publishing Practice", "Editor Center") got extracted as if it were the
    paper's own data-availability statement. Callers should skip the LLM
    call entirely and stage a needs_review note when this returns False.

    Two independent checks, either one can fail it:
      1. Keyword markers strongly associated with journal-website chrome.
      2. Structural shape: navigation menus are made of many short, terse
         lines (menu items), unlike real prose paragraphs. A keyword list
         alone missed a real case (only matched 1 of the listed phrases,
         not enough to trip a count-based threshold) - this structural
         check catches it regardless of which exact words are used."""
    if not text or len(text.strip()) < min_length:
        return False

    lower = text.lower()
    if sum(1 for marker in _CHROME_MARKERS if marker in lower) >= 2:
        return False

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    short_line_ratio = sum(1 for l in lines if len(l) < 40) / len(lines)
    avg_words_per_line = sum(len(l.split()) for l in lines) / len(lines)
    if short_line_ratio > 0.6 and avg_words_per_line < 8:
        return False

    return True


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
    """Returns text (str) for PDF hits, or (text, sections) tuple for HTML
    hits where structure could be parsed - fetch_paper() checks the type."""
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
            return text if text and len(text) > 500 else None
        else:
            text, sections = _parse_html_sections(pr.text)
            return (text, sections) if text and len(text) > 500 else None
    except Exception:
        return None


def _try_europepmc(doi: str):
    """Returns (text, structured_sections) or None."""
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
        text, sections = _parse_europepmc_xml(r2.text)
        return (text, sections) if text and len(text) > 500 else None
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
        1. The named section (e.g. "methods", "data_availability"), if we
           have real structure (Europe PMC) or the heading regex found it.
        2. The full fetched text, if we have it but the section wasn't found
           (common - the heuristic misses non-standard headings).
        3. Whatever text we have at all (even just an abstract).

    Returns (text, source_label) where source_label is one of:
        "section:<name>" | "full_text_fallback" | "abstract_only" | ""
    Always truncated to config.MAX_SECTION_CHARS - callers should treat a
    truncated result as lower-confidence (the agent may be missing context
    that ran past the cap) and can check with was_truncated().

    NOTE: this does NOT check content quality - call is_probably_real_content()
    on the result before spending an LLM call on it.
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


def fetch_paper(doi: str, force_refetch: bool = False) -> dict:
    """Main entry point. Cached on disk per normalized DOI."""
    doi = normalize_doi(doi)
    if not doi:
        return _empty_result(doi)

    cache_file = _cache_path(doi)
    if cache_file.exists() and not force_refetch:
        return json.loads(cache_file.read_text())

    result = None
    structured_sections = None  # only set when a source gives real structure (Europe PMC)

    print(f"[paper_fetcher] {doi}: trying Unpaywall...", flush=True)
    unpaywall_result = _try_unpaywall(doi)
    if unpaywall_result:
        if isinstance(unpaywall_result, tuple):
            text, structured_sections = unpaywall_result
            print(f"[paper_fetcher] {doi}: got full text via Unpaywall (HTML, "
                  f"structured sections found: {list(structured_sections.keys()) or 'none'})", flush=True)
        else:
            text = unpaywall_result
            print(f"[paper_fetcher] {doi}: got full text via Unpaywall (PDF)", flush=True)
        result = {"source": "unpaywall_pdf", "is_full_text": True, "text": text}
    else:
        print(f"[paper_fetcher] {doi}: Unpaywall had nothing usable, trying Europe PMC...", flush=True)

    if result is None:
        europepmc_result = _try_europepmc(doi)
        if europepmc_result:
            text, structured_sections = europepmc_result
            print(f"[paper_fetcher] {doi}: got full text via Europe PMC "
                  f"(structured sections found: {list(structured_sections.keys()) or 'none'})", flush=True)
            result = {"source": "europepmc", "is_full_text": True, "text": text}
        else:
            print(f"[paper_fetcher] {doi}: Europe PMC had nothing usable, trying bioRxiv...", flush=True)

    if result is None:
        text = _try_biorxiv(doi)
        if text:
            print(f"[paper_fetcher] {doi}: got abstract via bioRxiv", flush=True)
            result = {"source": "biorxiv_abstract", "is_full_text": False, "text": text}
        else:
            print(f"[paper_fetcher] {doi}: bioRxiv had nothing usable, trying Crossref abstract...", flush=True)

    if result is None:
        text = _try_crossref_abstract(doi)
        if text:
            print(f"[paper_fetcher] {doi}: got abstract via Crossref", flush=True)
            result = {"source": "crossref_abstract_only", "is_full_text": False, "text": text}
        else:
            print(f"[paper_fetcher] {doi}: no text retrievable from any source", flush=True)

    if result is None:
        result = {"source": "none", "is_full_text": False, "text": ""}

    result["doi"] = doi
    result["fetched_at"] = _now()
    if structured_sections:
        result["sections"] = structured_sections
    else:
        result["sections"] = _split_into_sections(result["text"]) if result["is_full_text"] else {}

    cache_file.write_text(json.dumps(result, indent=2))
    return result
