"""
relevance.py
Two-stage relevance filter for scraped candidates, per the 2026-09-04 design
discussion: a cheap keyword prefilter first (catches the obvious no's for
free, no LLM spend), then an LLM pass on title+abstract only for whatever
survives, to classify it into method/application (or reject) and catch
papers that use different vocabulary than the keyword list expects.

Deliberately excludes plain "immunofluorescence"/IF as a keyword - it's a
generic technique name used across most of cell biology and would flood the
prefilter with irrelevant matches, defeating the point of prefiltering
before spending an LLM call. Anchored instead on genuinely spatial-
proteomics-specific platform names and phrases.
"""

import re

from ...common.llm_client import call_llm_json

SP_KEYWORDS = [
    "imaging mass cytometry", "IMC",
    "multiplexed ion beam imaging", "MIBI", "MIBI-TOF",
    "CODEX", "PhenoCycler",
    "CyCIF",
    "Akoya Biosciences", "Akoya",
    "Vectra", "Opal multiplex",
    "spatial proteomics",
    "multiplexed imaging", "multiplex imaging",
    "highly multiplexed tissue imaging",
]

# Short acronyms need word-boundary matching (IMC, MIBI) so they don't match
# as a substring inside an unrelated word. Multi-word phrases don't have that
# risk, so a plain case-insensitive substring check is enough for those.
_ACRONYMS = {"IMC", "MIBI", "MIBI-TOF", "CODEX", "CyCIF"}


def keyword_prefilter(title: str, abstract: str) -> list:
    """Returns the list of matched keywords (empty list = no match = reject
    before spending an LLM call). Checks title+abstract combined."""
    text = f"{title or ''} {abstract or ''}"
    matched = []
    for kw in SP_KEYWORDS:
        if kw in _ACRONYMS:
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                matched.append(kw)
        elif kw.lower() in text.lower():
            matched.append(kw)
    return matched


_SYSTEM_PROMPT = """You are screening candidate papers for a curated review of \
spatial proteomics computational methods and datasets (protein markers via \
imaging-based multiplexed tissue technologies like IMC, MIBI, CODEX, CyCIF, \
Akoya PhenoCycler, Vectra/Opal - NOT spatial transcriptomics / gene panels).

Given a paper's title and abstract, decide:
1. Is this genuinely relevant to spatial proteomics (a new computational \
method/tool applied to spatial proteomics data, OR a paper that generates/uses \
a spatial proteomics dataset)? Mentioning immunofluorescence or imaging in \
passing does NOT count - the paper's actual subject must be spatial \
proteomics specifically.
2. If relevant, classify it as "method" (introduces or benchmarks a \
computational method/tool) or "application" (uses existing methods to study \
a biological question, e.g. a disease cohort study).

Respond with ONLY a JSON object: {"relevant": true/false, "paper_type": \
"method" or "application" or null, "reason": "one short sentence"}"""


def llm_relevance_pass(title: str, abstract: str) -> dict:
    """Returns {"relevant": bool, "paper_type": "method"|"application"|None,
    "reason": str, "model_used": str}. Raises LLMError/LLMExhaustedError on
    total failure - same as every other agent's LLM call, let the caller
    decide whether to skip this one paper or stop the whole scan."""
    user_prompt = f"Title: {title}\n\nAbstract: {abstract or '(no abstract available)'}"
    parsed, model_used = call_llm_json(_SYSTEM_PROMPT, user_prompt)
    return {
        "relevant": bool(parsed.get("relevant")),
        "paper_type": parsed.get("paper_type"),
        "reason": parsed.get("reason", ""),
        "model_used": model_used,
    }
