"""
pubmed_client.py
Search + fetch against NCBI's E-utilities (esearch/efetch) - plain `requests`
calls, no MCP dependency, so this works unattended on gaia exactly like
paper_fetcher.py's other API calls. Verified against the real API (2026-09-17):
esearch returns a PMID list; efetch's <PubmedData><ArticleIdList> is the
authoritative source for a record's DOI (more reliable than the <ELocationID>
inside <Article>, which some record types omit).

Rate limiting: NCBI allows 3 req/s without an API key, 10 req/s with one
(set NCBI_API_KEY in .env - optional, falls back to unauthenticated). Always
sends `email` (from config.CONTACT_EMAIL) per NCBI's usage policy.
"""

import time
import xml.etree.ElementTree as ET

import requests

from ...common import config

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

_EFETCH_BATCH_SIZE = 200  # NCBI's documented comfortable batch size for efetch
_MIN_REQUEST_INTERVAL_S = 0.34  # ~3 req/s (no API key case; see _sleep_for_rate_limit)

_last_request_time = [0.0]


def _sleep_for_rate_limit():
    elapsed = time.time() - _last_request_time[0]
    if elapsed < _MIN_REQUEST_INTERVAL_S:
        time.sleep(_MIN_REQUEST_INTERVAL_S - elapsed)
    _last_request_time[0] = time.time()


def _common_params() -> dict:
    import os
    params = {"email": config.CONTACT_EMAIL}
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def search(query: str, mindate: str, maxdate: str, retmax: int = 9999) -> list:
    """mindate/maxdate: 'YYYY/MM/DD'. Returns a list of PMID strings.
    Uses datetype=pdat (publication date) - the relevant date for "what's
    new this month", not the (later, variable) date PubMed indexed it."""
    _sleep_for_rate_limit()
    resp = requests.get(ESEARCH_URL, params={
        **_common_params(),
        "db": "pubmed",
        "term": query,
        "datetype": "pdat",
        "mindate": mindate,
        "maxdate": maxdate,
        "retmax": retmax,
        "retmode": "json",
    }, timeout=config.FETCH_TIMEOUT_S)
    resp.raise_for_status()
    result = resp.json()["esearchresult"]
    count = int(result.get("count", 0))
    if count > retmax:
        print(f"[pubmed_client] WARNING: query matched {count} results but "
              f"retmax={retmax} - {count - retmax} results silently dropped. "
              f"Narrow the query or raise retmax. Query: {query!r}")
    return result.get("idlist", [])


def _text_or_none(el):
    return el.text if el is not None else None


def _parse_article(article_el) -> dict:
    pmid = _text_or_none(article_el.find(".//MedlineCitation/PMID"))

    doi = ""
    for aid in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text or ""
            break

    title = _text_or_none(article_el.find(".//Article/ArticleTitle")) or ""
    journal = _text_or_none(article_el.find(".//Article/Journal/Title")) or ""

    year = _text_or_none(article_el.find(".//Article/Journal/JournalIssue/PubDate/Year"))
    if not year:
        # Some records only have a free-text MedlineDate ("2020 Jan-Feb") -
        # the year is always the leading 4 digits when this path is hit.
        medline_date = _text_or_none(article_el.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
        year = (medline_date or "")[:4] or None

    abstract_parts = [
        (el.text or "") for el in article_el.findall(".//Article/Abstract/AbstractText")
    ]
    abstract = " ".join(p.strip() for p in abstract_parts if p.strip())

    # PubMed's own PublicationType tags (e.g. "Journal Article", "Review",
    # "Editorial", "Letter", "Comment", "Published Erratum", "Retraction of
    # Publication") - real, free metadata, used by relevance.py to
    # hard-reject corrections/retractions/editorials before spending an LLM
    # call, and to reliably tag reviews instead of asking the LLM to guess
    # "review-ness" from title/abstract alone.
    publication_types = [
        (el.text or "").strip()
        for el in article_el.findall(".//Article/PublicationTypeList/PublicationType")
        if (el.text or "").strip()
    ]

    authors = []
    for author_el in article_el.findall(".//Article/AuthorList/Author"):
        last = _text_or_none(author_el.find("LastName"))
        fore = _text_or_none(author_el.find("ForeName"))
        if last:
            authors.append(f"{fore} {last}".strip() if fore else last)

    return {
        "pmid": pmid,
        "doi": doi,
        "title": title,
        "journal": journal,
        "year": year,
        "abstract": abstract,
        "authors": authors,
        "source": "pubmed",
        "publication_types": publication_types,
    }


def fetch_details(pmids: list) -> list:
    """Batched efetch (200 IDs/request, NCBI's documented comfortable size) +
    XML parse. Returns one dict per PMID (see _parse_article for fields)."""
    records = []
    for i in range(0, len(pmids), _EFETCH_BATCH_SIZE):
        batch = pmids[i:i + _EFETCH_BATCH_SIZE]
        _sleep_for_rate_limit()
        resp = requests.get(EFETCH_URL, params={
            **_common_params(),
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }, timeout=config.FETCH_TIMEOUT_S)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for article_el in root.findall(".//PubmedArticle"):
            records.append(_parse_article(article_el))
    return records


def search_and_fetch(query: str, mindate: str, maxdate: str, retmax: int = 9999) -> list:
    pmids = search(query, mindate, maxdate, retmax=retmax)
    if not pmids:
        return []
    return fetch_details(pmids)
