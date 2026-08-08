"""
02_fetch_metadata.py
Reads methods CSV from data/processed/,
fetches metadata for each DOI (title, authors, year, journal,
citations, abstract, publication_type),
and writes methods_metadata_YYYY_MM_DD.csv to data/processed/.

Sources:
    - Crossref  → title, authors, year, journal, citations, type
    - PubMed    → abstract (published papers)
    - bioRxiv   → abstract (preprints)

Usage:
    python src/preprocessing/02_fetch_metadata.py
"""

import time
import re
from pathlib import Path
import os
import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# Find the most recent methods CSV. Excludes "methods_metadata_*.csv" - that's
# THIS script's own output, and without the exclusion it matches the same
# "methods_*.csv" glob as its input. Alphabetically "methods_metadata..."
# sorts after "methods_2026...", so an old metadata output could get picked
# up as if it were fresh input and re-processed (caught live: produced
# "methods_metadata_metadata_2026_07_07.csv" from a stale prior run, 0
# fetched because everything in it was already enriched).
methods_files = sorted(
    p for p in PROCESSED_DIR.glob("methods_*.csv") if "metadata" not in p.stem
)
if not methods_files:
    raise FileNotFoundError(f"No methods_*.csv (excluding methods_metadata_*.csv) found in {PROCESSED_DIR}")
METHODS_FILE = methods_files[-1]

# Derive date suffix from filename e.g. methods_2026_07_07.csv -> 2026_07_07
suffix = "_".join(METHODS_FILE.stem.split("_")[1:])
OUTPUT_FILE = PROCESSED_DIR / f"methods_metadata_{suffix}.csv"

# ── Crossref ──────────────────────────────────────────────────────────────────
CROSSREF_URL = "https://api.crossref.org/works/{doi}"
HEADERS = {"User-Agent": f"solr-living-review/1.0 (mailto:{os.environ.get('SOLR_EMAIL', 'anonymous')})"}

def fetch_crossref(doi: str) -> dict:
    url = CROSSREF_URL.format(doi=doi.replace("https://doi.org/", ""))
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json().get("message", {})

        # Authors
        authors_raw = data.get("author", [])
        authors = []
        for a in authors_raw:
            given = a.get("given", "")
            family = a.get("family", "")
            authors.append(f"{given} {family}".strip())
        first_author = authors[0] if authors else ""
        authors_str  = "; ".join(authors)

        # Year
        year = ""
        for date_field in ["published", "published-print", "published-online", "created"]:
            date_parts = data.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])
                break

        # Journal
        container = data.get("container-title", [])
        journal = container[0] if container else ""

        # Publication type
        pub_type = data.get("type", "")
        if pub_type == "journal-article":
            # Check if bioRxiv/medRxiv preprint
            if any(p in journal.lower() for p in ["biorxiv", "medrxiv"]):
                publication_type = "preprint"
            else:
                publication_type = "peer-reviewed"
        elif pub_type == "posted-content":
            publication_type = "preprint"
        else:
            publication_type = pub_type  # keep raw for other types

        return {
            "title":            data.get("title", [""])[0],
            "first_author":     first_author,
            "authors":          authors_str,
            "year":             year,
            "journal":          journal,
            "citations":        data.get("is-referenced-by-count", ""),
            "publication_type": publication_type,
        }
    except Exception as e:
        print(f"    Crossref error for {doi}: {e}")
        return {}

# ── PubMed abstract ───────────────────────────────────────────────────────────
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def fetch_pubmed_abstract(doi: str) -> str:
    try:
        # Search for PMID by DOI
        r = requests.get(PUBMED_SEARCH, params={
            "db": "pubmed", "term": doi, "retmode": "json"
        }, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""

        # Fetch abstract
        r2 = requests.get(PUBMED_FETCH, params={
            "db": "pubmed", "id": ids[0], "rettype": "abstract", "retmode": "text"
        }, timeout=10)
        text = r2.text

        # Extract abstract section
        match = re.search(r"AB\s+-\s+(.+?)(?=\n[A-Z]{2}\s+-|\Z)", text, re.DOTALL)
        if match:
            return " ".join(match.group(1).split())
        return ""
    except Exception as e:
        print(f"    PubMed error for {doi}: {e}")
        return ""

# ── bioRxiv abstract ──────────────────────────────────────────────────────────
BIORXIV_URL = "https://api.biorxiv.org/details/biorxiv/{doi}/na/json"

def fetch_biorxiv_abstract(doi: str) -> str:
    try:
        bare = doi.replace("https://doi.org/", "")
        r = requests.get(BIORXIV_URL.format(doi=bare), timeout=10)
        collection = r.json().get("collection", [])
        if collection:
            return collection[0].get("abstract", "")
        return ""
    except Exception as e:
        print(f"    bioRxiv error for {doi}: {e}")
        return ""

# ── Fetch abstract ────────────────────────────────────────────────────────────
def fetch_abstract(doi: str, publication_type: str) -> str:
    if publication_type == "preprint":
        abstract = fetch_biorxiv_abstract(doi)
        if abstract:
            return abstract
    # Fall back to PubMed for everything else
    return fetch_pubmed_abstract(doi)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Reading {METHODS_FILE.name} ...")
    df = pd.read_csv(METHODS_FILE, dtype=str)

    # Add metadata columns if not present
    meta_cols = ["title", "first_author", "authors", "year",
                 "journal", "citations", "abstract", "publication_type"]
    for col in meta_cols:
        if col not in df.columns:
            df[col] = ""

    doi_col = "DOI"
    total   = len(df)
    skipped = 0
    fetched = 0
    failed  = 0

    for i, row in df.iterrows():
        doi = str(row.get(doi_col, "")).strip()

        # Skip placeholders and empty DOIs
        if not doi or doi in ("", "nan", "NA"):
            skipped += 1
            continue

        # Skip if already enriched
        if str(row.get("title", "")).strip() not in ("", "nan"):
            skipped += 1
            continue

        print(f"  [{i+1}/{total}] {doi}")

        meta = fetch_crossref(doi)
        if not meta:
            print(f"    No Crossref data found")
            failed += 1
            continue

        # Fetch abstract separately
        meta["abstract"] = fetch_abstract(doi, meta.get("publication_type", ""))

        for col, val in meta.items():
            df.at[i, col] = val

        fetched += 1
        time.sleep(0.2)  # be polite to APIs

    print(f"\nDone: {fetched} fetched, {skipped} skipped, {failed} failed")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Output written: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()