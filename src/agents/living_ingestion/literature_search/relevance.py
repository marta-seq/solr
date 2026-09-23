"""
relevance.py
Two-stage relevance filter for scraped candidates, per the 2026-09-04 design
discussion: a cheap keyword prefilter first (catches the obvious no's for
free, no LLM spend), then ONE LLM pass on title+abstract only for whatever
survives, to classify it into method/application (or reject) AND, in the
same call, tag pipeline_category. Merged into a single call 2026-09-17
(was briefly two separate calls; Marta asked to fold it back into one to
stay light on LLM spend without giving up quality - the paper-type decision
and the category decision are asked together, one JSON response covers
both).

SP_KEYWORDS is loaded from sp_keywords.txt (same directory) rather than
hardcoded here, so the keyword list can be reviewed/edited without touching
code, and so it can double as the future PubMed query source (see that
file's own header comment).
"""

import re
from pathlib import Path

from ...common.llm_client import call_llm_json
from ...common.reference_resolver import CONFIDENCE_FLOOR
from ....preprocessing.category_maps import PIPELINE_CATEGORY_TAXONOMY


DEFAULT_KEYWORDS_PATH = Path(__file__).parent / "sp_keywords.txt"


def load_keywords(path=None) -> list:
    """Loads a keyword list from a file in sp_keywords.txt's format (one
    keyword/phrase per line, '#'-comments and blank lines ignored). Defaults
    to DEFAULT_KEYWORDS_PATH (the built-in SP keyword list) when no path is
    given. Public/parameterized (not just an SP_KEYWORDS-loading internal) so
    scan.py's --keywords-file can point this at a different file entirely -
    per Marta's 2026-09-17 ask, keeping the door open for SOLR to search a
    different domain later without a code change, even though the project's
    current scope is spatial-proteomics-only (see CLAUDE.md)."""
    with open(path or DEFAULT_KEYWORDS_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


SP_KEYWORDS = load_keywords()


def build_pubmed_query(keywords: list = None) -> str:
    """OR-joins a keyword list (defaults to SP_KEYWORDS) into a PubMed query
    string, quoting any keyword containing a space (PubMed's search treats an
    unquoted multi-word term as an AND of the individual words, not the exact
    phrase). Built fresh every call - editing the underlying file changes
    this query with no code change needed, per Marta's 2026-09-17 ask
    (previously this had to be hand-retyped into the CLI --query argument on
    every run, silently drifting out of sync with the file)."""
    keywords = keywords if keywords is not None else SP_KEYWORDS
    return " OR ".join(f'"{kw}"' if " " in kw else kw for kw in keywords)


# PubMed PublicationType values (see pubmed_client.py's _parse_article) that
# should never be staged regardless of what the title/abstract say - checked
# BEFORE the keyword prefilter and LLM call, zero cost. Always empty for
# bioRxiv/medRxiv records (no PublicationType concept there). "Editorial"
# included per Marta's 2026-09-17 call - editorials are structurally
# commentary (introducing a themed issue, reacting to a same-issue paper,
# field-debate opinion), essentially never carrying new data/method/dataset
# content. "Letter" deliberately NOT included - some journals (e.g. Nature)
# publish genuine short-format research as "Letter", too risky to blanket
# -reject; left to the normal keyword+LLM funnel instead.
HARD_REJECT_PUBLICATION_TYPES = {
    "Editorial",
    "Comment",
    "Published Erratum",
    "Retraction of Publication",
    "Retracted Publication",
    "Corrected and Republished Article",
}


def publication_type_reject_reason(publication_types: list) -> str:
    """Returns a human-readable reject reason if this record's PubMed
    PublicationType list hits HARD_REJECT_PUBLICATION_TYPES, else "" (don't
    reject). Checked in scan.py before the keyword prefilter."""
    hit = next((t for t in publication_types if t in HARD_REJECT_PUBLICATION_TYPES), None)
    return f"publication type: {hit}" if hit else ""


def is_review(publication_types: list) -> bool:
    """True if PubMed tagged this record "Review". Used to tag the `category`
    field (see scan.py's _category_field) - deliberately does NOT change the
    method/application/pipeline_category decision below, which still runs
    exactly the same for a review as for an original article. Always False
    for bioRxiv/medRxiv (no PublicationType concept there)."""
    return "Review" in publication_types


def keyword_prefilter(title: str, abstract: str, keywords: list = None) -> list:
    """Returns the list of matched keywords (empty list = no match = reject
    before spending an LLM call). Checks title+abstract combined. `keywords`
    defaults to SP_KEYWORDS - pass a different list (e.g. from scan.py's
    --keywords-file) to prefilter against something other than the built-in
    SP vocabulary.

    Matching rule (see sp_keywords.txt's header): a keyword with no space in
    it (an acronym/single token, e.g. IMC, MIBI-TOF) is matched with word
    boundaries so it can't false-positive as a substring inside an unrelated
    word; a keyword with a space (a phrase, e.g. "spatial proteomics") is a
    plain case-insensitive substring match. Derived automatically from
    whether the keyword contains a space - no per-keyword flag needed."""
    keywords = keywords if keywords is not None else SP_KEYWORDS
    text = f"{title or ''} {abstract or ''}"
    matched = []
    for kw in keywords:
        if " " not in kw:
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                matched.append(kw)
        elif kw.lower() in text.lower():
            matched.append(kw)
    return matched


_SYSTEM_PROMPT = f"""You are screening candidate papers for a curated review of \
spatial proteomics computational methods and datasets (protein markers via \
imaging-based multiplexed tissue technologies like IMC, MIBI, CODEX, CyCIF, \
Akoya PhenoCycler, Vectra/Opal - NOT spatial transcriptomics / gene panels).

Given a paper's title and abstract, decide:

1. Is this genuinely relevant to spatial proteomics (a new computational \
method/tool applied to spatial proteomics data, OR a paper that generates/uses \
a spatial proteomics dataset)? Mentioning immunofluorescence or imaging in \
passing does NOT count - the paper's actual subject must be spatial \
proteomics specifically.

2. If relevant, classify it as "method" ONLY if it introduces or benchmarks \
a new COMPUTATIONAL/SOFTWARE tool - an algorithm, a statistical model, a \
pipeline/package that processes data on a computer. If it instead introduces \
or improves a WET-LAB technique - new chemistry, a labeling/amplification \
method, a staining or antibody panel protocol, a tissue-preparation or \
imaging-hardware technique - that is NOT "method", even though it is also a \
genuinely new technique. Classify it as "application" instead (see step 3 - \
it will get the TECHNICAL application tag there). "Introduces something new" \
is not sufficient for "method" by itself - the new thing must be \
computational/software, not wet-lab/chemical/hardware.

3. ONLY if you classified it as "application" in step 2: is this a \
TECHNICAL application? Set "technical_application": true ONLY if the paper's \
MAIN CONTRIBUTION is the technique/protocol/platform itself - it introduces, \
validates, or improves a wet-lab method (e.g. introducing IMC or CODEX as a \
platform, a new amplification/labeling chemistry, a new antibody panel or \
staining protocol, an improved tissue-preparation workflow). Set it to false \
- a BIOLOGICAL application - if the paper's main contribution is a \
biological/clinical finding, even if it describes the platform/protocol in \
detail to justify its use (e.g. "using IMC to characterize the immune \
microenvironment in pancreatic cancer" is BIOLOGICAL/false, even though it \
names and describes IMC - the paper is ABOUT pancreatic cancer biology, not \
about IMC itself). The test is: does the abstract's main claim describe a \
NEW or IMPROVED technique, or does it describe a biological/clinical \
finding obtained USING an existing technique? Merely naming, describing, or \
using a named platform/technology is NOT sufficient for true - disease \
cohort studies, tissue atlases, and biomarker/mechanism studies are almost \
always BIOLOGICAL (false), even when the platform is central to the method \
section. Only meaningful when paper_type is "application" - set it to false \
(unused) for "method" or when not relevant.

4. ONLY if you classified it as "method" in step 2: which pipeline \
category/categories does it belong to? Choose ONLY from this fixed list - do \
not invent, rename, or reword any label. Copy the exact string(s) character \
for character:
{chr(10).join(f'- {c}' for c in PIPELINE_CATEGORY_TAXONOMY)}
A paper usually fits exactly one category; some genuinely fit more than one \
(pick every category that clearly applies, don't pad the list). If nothing \
on the list clearly applies, or you are not confident, return an empty list \
rather than guessing. Leave this empty for "application" papers - the \
taxonomy above doesn't apply to them.

Two specific disambiguation rules, added after real mis-tagging was found in \
review (2026-09-22):
- If the paper is a REVIEW/SURVEY whose main contribution is summarizing or \
comparing many existing methods across a field, rather than presenting ONE \
specific new method the paper itself implements, return an EMPTY category \
list even though it may discuss several categories below in passing - none \
of them describe what this specific paper itself does.
- "Cell type Deconvolution" applies ONLY to spot-based/multi-cell-resolution \
technologies (e.g. Visium) where each measurement mixes multiple cells and \
must be computationally split into per-cell-type proportions. Do NOT apply \
it to single-cell/subcellular-resolution platforms (IMC, MIBI, CODEX, CyCIF, \
Akoya PhenoCycler) where each measurement is already one cell - a paper \
predicting or classifying cell types/states on those platforms is \
"Phenotyping", not deconvolution. Separately: predicting/recovering missing \
marker measurements from the SAME modality (e.g. low-plex protein imaging -> \
higher-plex protein prediction) is "Data alignment / integration / \
imputation", not "Virtual staining" - reserve "Virtual staining" for \
generating a stain/channel from a DIFFERENT modality entirely (e.g. \
predicting protein signal from a plain H&E/brightfield image with no \
protein channels at all).

Respond with ONLY a JSON object: {{"relevant": true/false, "paper_type": \
"method" or "application" or null, "technical_application": true/false (only \
meaningful for paper_type "application", see step 3), "categories": \
["<exact label>", ...] (only for paper_type "method", else []), \
"category_confidence": <0.0-1.0, your confidence that every label in \
"categories" is correct - irrelevant/unused if "categories" is empty>, \
"reason": "one short sentence"}}"""


def llm_relevance_pass(title: str, abstract: str) -> dict:
    """Single LLM call deciding relevance, method/application type, (for
    application papers only) technical-vs-biological application, and (for
    method papers only) pipeline_category tags. Returns {"relevant": bool,
    "paper_type": "method"|"application"|None, "technical_application": bool
    (only meaningful for "application" - a paper describing/improving a
    wet-lab technique/platform/protocol itself, e.g. introducing IMC/CODEX
    or optimizing a staining protocol, as opposed to using an established
    platform to study a biological question), "categories": [...] (already
    validated against PIPELINE_CATEGORY_TAXONOMY - anything the LLM returns
    that isn't an exact match is dropped, not corrected/guessed),
    "category_confidence": float, "reason": str, "model_used": str}. Raises
    LLMError/LLMExhaustedError on total failure - same as every other
    agent's LLM call, let the caller decide whether to skip this one paper
    or stop the whole scan."""
    user_prompt = f"Title: {title}\n\nAbstract: {abstract or '(no abstract available)'}"
    # skip_openrouter=True per Marta's 2026-09-17 ask: this scanner can fetch
    # and evaluate a high volume of candidates in one run, which would burn
    # through OpenRouter's low, account-wide daily free cap and starve every
    # other agent sharing that same account for the rest of the day. Scoped
    # to just this call - other agents still use OpenRouter normally.
    parsed, model_used = call_llm_json(_SYSTEM_PROMPT, user_prompt, skip_openrouter=True)
    raw_categories = parsed.get("categories") or []
    valid_categories = [c for c in raw_categories if c in PIPELINE_CATEGORY_TAXONOMY]
    return {
        "relevant": bool(parsed.get("relevant")),
        "paper_type": parsed.get("paper_type"),
        "technical_application": bool(parsed.get("technical_application")),
        "categories": valid_categories,
        "category_confidence": float(parsed.get("category_confidence") or 0.0),
        "reason": parsed.get("reason", ""),
        "model_used": model_used,
    }


def category_pass_is_confident(verdict: dict) -> bool:
    """Gate per Marta's explicit ask (2026-09-17): below CONFIDENCE_FLOOR
    (the same 0.55 threshold reference_resolver.py already uses elsewhere in
    this codebase, reused rather than inventing a new number), don't tag at
    all - leave pipeline_category blank on the staged candidate rather than
    stage a low-confidence guess. An empty validated categories list also
    counts as "not confident" even if the model reported high confidence for
    nothing. The paper still stages fine either way - this only gates
    whether the category field gets attached."""
    return bool(verdict["categories"]) and verdict["category_confidence"] >= CONFIDENCE_FLOOR
