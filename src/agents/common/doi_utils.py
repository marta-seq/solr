"""
doi_utils.py
DOI normalization (identical logic to 01_parse_excel.py's normalize_doi,
duplicated here so agents/ has no import dependency on preprocessing/)
plus helpers to match a discovered DOI against the existing database.
"""

import re
import pandas as pd


def _strip_trailing_punctuation(doi: str) -> str:
    """Removes sentence-ending punctuation accidentally captured as part of
    a DOI - e.g. '...23807-4.' extracted from '...see https://doi.org/
    10.1038/s41467-021-23807-4.' (the period ends the SENTENCE, not the DOI).
    Caught via testing on real output: EVERY embedded-DOI extraction from a
    reference list had a trailing period, silently breaking matches against
    existing entries that (correctly) don't have one - two different-looking
    strings for what should be the identical DOI. Also balances a trailing
    ')' if there's no matching '(' within the DOI - some DOI suffixes
    legitimately contain balanced parentheses."""
    doi = re.sub(r"[.,;:]+$", "", doi)
    if doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    return doi


def normalize_doi(doi) -> str:
    """Canonical form: https://doi.org/10.xxxx/... or '' if empty/invalid.
    Mirrors src/preprocessing/01_parse_excel.py exactly - keep these in sync.
    Strips trailing sentence punctuation in every branch, not just the
    regex-fallback one - a DOI that already looks like a full URL still
    needs this, since a PREVIOUSLY mis-extracted trailing period would
    otherwise just pass straight through unchanged on every re-normalization."""
    if doi is None or (isinstance(doi, float) and pd.isna(doi)):
        return ""
    doi = str(doi).strip()
    if doi in ("", "NA", "na", "nan", "None", "none"):
        return ""

    if doi.startswith("https://doi.org/"):
        result = doi
    elif doi.startswith("http://doi.org/"):
        result = doi.replace("http://", "https://")
    elif doi.startswith("doi.org/"):
        result = "https://" + doi
    elif doi.startswith("10."):
        result = "https://doi.org/" + doi
    else:
        # last resort: strip any leading junk and hope it's a bare DOI
        m = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", doi)
        if not m:
            return ""
        result = "https://doi.org/" + m.group(0)

    return _strip_trailing_punctuation(result)


def bare_doi(doi: str) -> str:
    """Strip the https://doi.org/ prefix, for display or fuzzy comparisons."""
    doi = normalize_doi(doi)
    return doi.replace("https://doi.org/", "") if doi else ""


class DoiIndex:
    """
    In-memory lookup: normalized DOI -> entry_id, built from the master DB
    (methods.csv / datasets.csv) PLUS any candidates created earlier in the
    same pipeline run (so we don't create duplicate entries for the same
    paper twice within one run, before the staging file has been merged back).
    """

    def __init__(self):
        self._doi_to_id = {}

    def load_from_dataframe(self, df: pd.DataFrame, doi_col: str, id_col: str = "entry_id"):
        for _, row in df.iterrows():
            doi = normalize_doi(row.get(doi_col, ""))
            entry_id = row.get(id_col, "")
            # pandas NaN is a truthy float in plain Python - `if entry_id` alone
            # would wrongly index a real DOI against the literal string "nan"
            # when entry_id is genuinely missing (caught via testing).
            if pd.isna(entry_id):
                continue
            entry_id = str(entry_id).strip()
            if doi and entry_id and entry_id.lower() not in ("", "nan", "na"):
                self._doi_to_id[doi] = entry_id

    def add(self, doi: str, entry_id: str):
        doi = normalize_doi(doi)
        if doi:
            self._doi_to_id[doi] = entry_id

    def lookup(self, doi: str):
        """Returns entry_id if this DOI is already known, else None."""
        return self._doi_to_id.get(normalize_doi(doi))

    def __contains__(self, doi: str) -> bool:
        return normalize_doi(doi) in self._doi_to_id

    def __len__(self):
        return len(self._doi_to_id)
