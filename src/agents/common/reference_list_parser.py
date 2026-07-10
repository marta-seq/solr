"""
reference_list_parser.py
Parses a paper's references/bibliography section (already isolated by
paper_fetcher's section split) into {marker: full_reference_text}, so a
numbered in-text citation like "[12]" can be looked up to get the EXACT
reference as printed - not an LLM's best-effort recollection of it.

Only works for numbered reference styles ([12], 12., etc.) - author-year
style (Smith et al., 2020) has no number to look up, so callers need a
fallback path for that case regardless.
"""

import re

from .doi_utils import normalize_doi

# Matches a reference-list entry marker at the start of a line: "[12] ",
# "12. ", "12) " - the number is captured, brackets/punctuation are not.
_REF_MARKER_PATTERN = re.compile(r"(?:^|\n)\s*\[?(\d{1,3})\]?[.)]\s+")

# Matches an in-text citation marker as the LLM might report it: "12",
# "[12]", "12,15" (only the first number is used - multi-ref markers should
# be split by the caller before calling lookup_reference on each).
_MARKER_NUMBER_PATTERN = re.compile(r"\d{1,3}")


def parse_reference_list(references_text: str) -> dict:
    """Returns {"12": "Author, A. et al. Title. Journal. 2020.", ...}.
    Empty dict if no numbered pattern was found (author-year style, or no
    references section was available at all)."""
    if not references_text:
        return {}

    matches = list(_REF_MARKER_PATTERN.finditer(references_text))
    if not matches:
        return {}

    result = {}
    for i, m in enumerate(matches):
        marker = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(references_text)
        entry_text = references_text[start:end].strip()
        if entry_text:
            result[marker] = entry_text
    return result


def lookup_reference(marker: str, reference_map: dict) -> str:
    """marker may arrive as '12', '[12]', etc. - pulls out the number and
    looks it up. Returns '' if not found."""
    if not marker or not reference_map:
        return ""
    m = _MARKER_NUMBER_PATTERN.search(str(marker))
    if not m:
        return ""
    return reference_map.get(m.group(0), "")


def extract_embedded_doi(reference_text: str) -> str:
    """If the reference entry already prints a DOI, pull it out directly -
    normalize_doi's regex fallback finds a DOI anywhere in a longer string,
    so this works on a full reference paragraph, not just a bare DOI."""
    return normalize_doi(reference_text)
