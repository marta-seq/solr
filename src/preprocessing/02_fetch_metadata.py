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
import shutil
from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
BACKUP_DIR    = ROOT / "data" / "processed_backup"

# Find the most recent methods CSV. Excludes "methods_metadata_*.csv" - that's
# THIS script's own output, and without the exclusion it matches the same
# "methods_*.csv" glob as its input. Alphabetically "methods_metadata..."
# sorts after "methods_2026...", so an old metadata output could get picked
# up as if it were fresh input and re-processed (caught live: produced
# "methods_metadata_metadata_2026_07_07.csv" from a stale prior run, 0
# fetched because everything in it was already enriched).
#
# Sort key normalizes "-" to "_" before comparing: filenames' date suffix
# isn't consistently hyphens or underscores (depends on how the source
# curated .xlsx happened to be named that day - both exist in this repo's
# history), and plain string sort puts "-" before "_" in ASCII, so e.g.
# "methods_2026-08-29.csv" would otherwise sort BEFORE "methods_2026_07_07.csv"
# and get picked as the "oldest", silently processing the wrong file
# (caught 2026-09-01: both of those files existed here at once).
methods_files = sorted(
    (p for p in PROCESSED_DIR.glob("methods_*.csv") if "metadata" not in p.stem),
    key=lambda p: p.stem.replace("-", "_"),
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
        if "arxiv" in doi.lower():
            # Crossref sometimes returns arXiv DOIs as "journal-article" if the
            # preprint was later published elsewhere - the DOI itself is the
            # reliable signal, not Crossref's type field.
            publication_type = "preprint"
        elif pub_type == "journal-article":
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
            # str() explicitly: Crossref returns this as an int, and this
            # column is read/written as pandas' strict StringDtype (not
            # legacy object dtype) - an int assignment raises TypeError
            # ("Invalid value '884' for dtype 'str'") instead of silently
            # coercing, first hit 2026-09-15 once AP_pub rows (which have a
            # citations-bearing DOI far more often than a title-based skip
            # check let through before) started actually reaching this code.
            "citations":        str(data.get("is-referenced-by-count", "")),
            "publication_type": publication_type,
        }
    except Exception as e:
        print(f"    Crossref error for {doi}: {e}")
        return {}

# ── PubMed abstract ───────────────────────────────────────────────────────────
PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def fetch_pubmed_abstract(doi: str) -> dict:
    """Returns {"abstract": str, "keywords": list} - both pulled from the
    SAME efetch call (MEDLINE text format tags each field: AB=abstract,
    OT=author-supplied keyword), added 2026-09-18 so keywords don't need a
    second API round-trip. Keywords come from MEDLINE's OT ("Other Term")
    field specifically - the genuine author-supplied keyword list, NOT the MH
    ("MeSH Heading") field, which is NLM's own broad, indexer-assigned
    controlled-vocabulary subject tags (e.g. "Humans", "Animals") - real PubMed
    data, not model-invented, but not what a reader means by "the paper's
    keywords" either (found 2026-09-20, after Marta flagged MH-derived output
    as "not the keywords from the paper" during review). Not every record has
    OT lines (depends on whether the journal submitted author keywords) - if
    none are present, keywords stays an empty list (caller writes "NA", not a
    MeSH fallback, per Marta's explicit ask). No OT/MH equivalent exists for
    bioRxiv/medRxiv preprints (not MEDLINE-indexed until published) -
    keywords stays empty for those, same coverage gap as abstract already has
    for unpublished preprints."""
    try:
        # Search for PMID by DOI. "[doi]" field restriction added 2026-09-20 -
        # without it, PubMed's automatic term-mapping can silently fuzzy-match
        # an unrelated record when the exact DOI isn't indexed (found via the
        # M_SE_29/RNA2seg contamination incident, 2026-09-19), returning a
        # confidently wrong abstract/keywords with no signal anything went
        # wrong. With "[doi]", an unindexed DOI correctly yields zero results
        # instead of a wrong match.
        r = requests.get(PUBMED_SEARCH, params={
            "db": "pubmed", "term": f"{doi}[doi]", "retmode": "json"
        }, timeout=10)
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"abstract": "", "keywords": []}

        # Fetch abstract + author keywords. rettype="medline" (NOT "abstract" -
        # found live 2026-09-18: "abstract" returns a human-readable citation
        # display with NO tagged fields at all for some records, so the
        # AB/OT regexes below silently never matched; "medline" reliably
        # returns proper PMID-/TI-/AB-/OT- tagged text). This alone likely
        # improves the previously-documented low abstract hit rate too, not
        # just keywords - both were being extracted from the wrong format.
        r2 = requests.get(PUBMED_FETCH, params={
            "db": "pubmed", "id": ids[0], "rettype": "medline", "retmode": "text"
        }, timeout=10)
        text = r2.text

        abstract = ""
        match = re.search(r"AB\s+-\s+(.+?)(?=\n[A-Z]{2}\s+-|\Z)", text, re.DOTALL)
        if match:
            abstract = " ".join(match.group(1).split())

        # OT lines: "OT  - Keyword phrase" - the author-supplied keyword list.
        keywords = []
        for line in text.splitlines():
            m = re.match(r"OT\s+-\s+(.+)", line)
            if m:
                keywords.append(m.group(1).strip())

        return {"abstract": abstract, "keywords": keywords}
    except Exception as e:
        print(f"    PubMed error for {doi}: {e}")
        return {"abstract": "", "keywords": []}

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

# ── Fetch abstract + keywords ─────────────────────────────────────────────────
def fetch_abstract_and_keywords(doi: str, publication_type: str) -> dict:
    """Returns {"abstract": str, "keywords": list}. bioRxiv/medRxiv preprints
    have no keyword concept (not MeSH-indexed pre-publication) - only
    abstract comes back for those. Everything else tries PubMed, which can
    supply both."""
    if publication_type == "preprint":
        abstract = fetch_biorxiv_abstract(doi)
        if abstract:
            return {"abstract": abstract, "keywords": []}
    # Fall back to PubMed for everything else
    return fetch_pubmed_abstract(doi)

# ── Main ──────────────────────────────────────────────────────────────────────
# Resumable/checkpointed (added 2026-09-01): this run can be interrupted
# (e.g. a wall-clock-limited execution environment, or Ctrl-C) partway
# through 371 rows of real network calls, and previously this only wrote
# output once at the very end - an interruption anywhere in the loop lost
# 100% of that run's progress. Now: (1) if OUTPUT_FILE already exists from a
# prior partial run, resume from IT instead of the bare input file, so
# already-fetched rows are skipped via the existing "already enriched"
# check; (2) save a checkpoint every CHECKPOINT_EVERY rows, not just at the
# end, so an interruption only loses work since the last checkpoint.
CHECKPOINT_EVERY = 15

# OUTPUT_FILE holds real fetched work (Crossref/PubMed/bioRxiv metadata,
# citation counts) that costs API calls and wall-clock time to rebuild, not
# just a regenerable intermediate like the raw 01 output - unlike
# data/data_curated/, it had no backup-before-overwrite safety net even
# though it's overwritten by every checkpoint and at the end of every run.
# Mirrors the existing data_curated -> data_curated_backup/ pattern: one
# dated copy taken before this run touches the file at all, so a crash mid-
# checkpoint or a bad run can be recovered from without a full 371-row
# re-fetch. Not git-tracked, same as data_curated_backup/ and processed/.
def backup_existing_output():
    if not OUTPUT_FILE.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"{OUTPUT_FILE.stem}_{stamp}{OUTPUT_FILE.suffix}"
    shutil.copy2(OUTPUT_FILE, backup)
    print(f"Backed up existing {OUTPUT_FILE.name} -> {backup.relative_to(ROOT)}")

def main():
    backup_existing_output()

    if OUTPUT_FILE.exists():
        print(f"Resuming from existing {OUTPUT_FILE.name} (partial run found)")
        df = pd.read_csv(OUTPUT_FILE, dtype=str)
    else:
        print(f"Reading {METHODS_FILE.name} ...")
        df = pd.read_csv(METHODS_FILE, dtype=str)

    # Add metadata columns if not present
    meta_cols = ["title", "first_author", "authors", "year",
                 "journal", "citations", "abstract", "publication_type", "keywords"]
    for col in meta_cols:
        if col not in df.columns:
            df[col] = ""

    doi_col = "DOI"
    total   = len(df)
    skipped = 0
    fetched = 0
    failed  = 0

    def _is_blank(v) -> bool:
        return str(v).strip() in ("", "nan", "NA", "None")

    for i, row in df.iterrows():
        doi = str(row.get(doi_col, "")).strip()

        # Skip placeholders and empty DOIs
        if not doi or doi in ("", "nan", "NA"):
            skipped += 1
            continue

        # Checked independently, not as one all-or-nothing bundle (changed
        # 2026-09-18, per Marta's ask: "if entries are missing go search for
        # them" - fetch only what's actually absent, not blindly everything
        # whenever ANY one field is missing). Two independent pieces:
        #   1. Crossref bundle (title/authors/year/journal/citations/
        #      publication_type) - these all come from ONE Crossref call, so
        #      they're still fetched together, gated on publication_type
        #      (reliable "already fetched" marker - see the 2026-09-15 note
        #      below, still applies). Title is NOT used as that marker:
        #      AP_pub rows carry their own manually-curated title from
        #      01_parse_excel.py regardless of whether this script has ever
        #      run on them, so a title-based check silently skipped every
        #      AP_pub row forever (152/154 have a real DOI, 0/154 ever got
        #      abstract/year/journal/citations - found 2026-09-15).
        #   2. Abstract - fetched via a completely separate method
        #      (fetch_abstract, PubMed/bioRxiv DOI lookup, not Crossref), so
        #      it's checked and retried independently of the Crossref bundle.
        #      This matters now that upstream sources (the literature-search
        #      scanner) can sometimes supply a real abstract directly - once
        #      that flows through to the master Excel, this skip check means
        #      it won't get needlessly overwritten by a re-fetch via the
        #      lower-hit-rate DOI-search method.
        needs_crossref = _is_blank(row.get("publication_type"))
        needs_abstract = _is_blank(row.get("abstract"))
        # keywords added 2026-09-18 - a genuinely new field nothing has ever
        # fetched before, so EVERY existing row needs at least one pass to
        # attempt it, same "fetch only what's missing" principle as abstract.
        needs_keywords = _is_blank(row.get("keywords"))
        if not needs_crossref and not needs_abstract and not needs_keywords:
            skipped += 1
            continue

        print(f"  [{i+1}/{total}] {doi}"
              f"{' (crossref)' if needs_crossref else ''}"
              f"{' (abstract)' if needs_abstract else ''}"
              f"{' (keywords)' if needs_keywords else ''}")

        publication_type = row.get("publication_type", "")
        if needs_crossref:
            meta = fetch_crossref(doi)
            if not meta:
                print(f"    No Crossref data found")
                failed += 1
                continue
            for col, val in meta.items():
                df.at[i, col] = val
            publication_type = meta.get("publication_type", "")

        if needs_abstract or needs_keywords:
            # Known trade-off: unlike the Crossref bundle (gated on
            # publication_type, permanently skipped once set), there's no
            # "we tried and it's genuinely unavailable" marker here - a
            # paper whose abstract/keywords truly aren't fetchable via this
            # method will get retried on every future run, not just once.
            # Accepted for now rather than adding a sentinel column; revisit
            # if this becomes a real time cost as the corpus grows.
            result = fetch_abstract_and_keywords(doi, publication_type)
            if result["abstract"]:
                df.at[i, "abstract"] = result["abstract"]
            if needs_keywords and publication_type != "preprint":
                # NA on a genuine attempt-and-fail, per Marta's ask 2026-09-20 -
                # distinguishes "tried, no OT keywords on this record" from
                # "never attempted". Preprints are skipped here (not marked
                # NA) since they structurally have no MEDLINE record to try
                # against - not a failed fetch, just not applicable.
                df.at[i, "keywords"] = "; ".join(result["keywords"]) if result["keywords"] else "NA"

        fetched += 1
        if fetched % CHECKPOINT_EVERY == 0:
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"    [checkpoint: {fetched} fetched so far, saved to {OUTPUT_FILE.name}]")
        time.sleep(0.2)  # be polite to APIs

    print(f"\nDone: {fetched} fetched, {skipped} skipped, {failed} failed")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Output written: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()