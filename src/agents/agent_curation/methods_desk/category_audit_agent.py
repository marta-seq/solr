"""
category_audit_agent.py (methods desk)
Cheap first-pass gate, run BEFORE the methods/data desks: for a paper
currently tagged as a computational method, checks (title+abstract only, no
full-text fetch needed) whether that's actually correct, or whether the
paper's real contribution is a wet-lab/chemistry/hardware technique wrongly
filed as a computational method. Built 2026-09-20 after a live example
surfaced during manual review: M_AUTO_73 (PASTA, a tyramide-oligonucleotide
amplification method) sitting in the DB tagged "computational analysis -
method" despite being a wet-lab technique.

The wet-lab-vs-computational distinction is adapted from
src/agents/living_ingestion/literature_search/relevance.py's existing,
already-tuned prompt language - reused rather than reinvented. NOT foolproof:
that exact PASTA case showed a local 14B model can miss an explicit, on-point
version of this prompt with high self-reported confidence - see
_SYSTEM_PROMPT's own note on why this audit is routed off the cheap/local
tier by default.

Deliberately does NOT auto-reclassify or move a flagged row to a different
sheet/category - that would be a bigger, harder-to-reverse structural change
than anything else any agent in this codebase does (every other agent only
ever creates/updates fields within the row's own existing sheet, or creates a
brand-new row - never moves an existing one between sheets). Instead stages a
REVIEW_STATUS=needs_review flag with a clear note, for Marta to confirm/
reject at merge-review time - consistent with the "stage everything, never
auto-apply a judgment call" philosophy already used everywhere else in this
codebase. Revisit if this proves too conservative in practice.

Nothing here writes to the master Excel - everything goes through
common.staging.append_candidate().
"""

from ...common import config, staging
from ...common.llm_client import call_llm_json, LLMError, LLMExhaustedError
from ...common.reference_resolver import CONFIDENCE_FLOOR

# NOT skip_openrouter'd like the desk agents/literature scanner - deliberate.
# This audit is low-volume (the existing corpus, once, not a firehose) and
# high-stakes-per-call (a wrong "correctly categorized" verdict just leaves
# bad data sitting silently; a wrong "flagged" verdict wastes Marta's review
# time) - the PASTA case showed the cheap local-model tier can be confidently
# wrong on exactly this judgment. Worth spending OpenRouter's shared quota on
# this specific task rather than routing it through the same zero-cost chain
# used for high-volume extraction.
_SYSTEM_PROMPT = """You are auditing an existing curated database entry that has already been \
tagged as a "computational analysis - method" paper - i.e. a paper introducing or benchmarking \
a COMPUTATIONAL/SOFTWARE tool (an algorithm, statistical model, or software pipeline that \
processes data on a computer).

Given the paper's title and abstract, decide TWO things:

1. Is the "computational analysis - method" tag CORRECT? It is INCORRECT if the paper's main \
contribution is actually a WET-LAB technique - new chemistry, a labeling/amplification method, \
a staining or antibody panel protocol, a tissue-preparation or imaging-hardware technique - even \
if the paper is a genuinely new and important technique. "Introduces something new" is not \
sufficient by itself - the new thing must be computational/software, not wet-lab/chemical/\
hardware, for the tag to be correct.

2. Which spatial omics modality/modalities does this paper's data/method actually concern? \
Choose one or more from EXACTLY these labels (copy character for character, do not invent \
others): "spatial_transcriptomics", "spatial_proteomics", "spatial_metabolomics". A paper \
usually fits exactly one; some genuinely fit more than one (multi-omics). If you cannot tell \
from the title/abstract, return an empty list rather than guessing.

Respond with ONLY a JSON object: {"correctly_categorized": true/false, "confidence": <0.0-1.0>, \
"reason": "one short sentence explaining your verdict", "spatial_modality": ["<exact label>", ...]}"""

_VALID_MODALITIES = {"spatial_transcriptomics", "spatial_proteomics", "spatial_metabolomics"}


def _build_user_prompt(entry_id: str, title: str, abstract: str) -> str:
    return f"Entry {entry_id}\nTitle: {title}\n\nAbstract: {abstract or '(no abstract available)'}"


def _empty_result(skip_reason: str, llm_exhausted: bool = False) -> dict:
    return {"audited": False, "flagged": False, "confidence": 0.0, "reason": "",
            "spatial_modality": [], "is_st_only": False,
            "model_used": "none", "skip_reason": skip_reason, "llm_exhausted": llm_exhausted}


def audit_entry(entry_id: str, title: str, abstract: str) -> dict:
    """
    Returns {"audited": bool, "flagged": bool, "confidence": float,
    "reason": str, "model_used": str, "skip_reason": str or None,
    "llm_exhausted": bool}. skip_reason is None only when the LLM was
    genuinely called - always check this field, same convention as every
    other agent's process_paper(). "flagged" is only True when the model
    says NOT correctly categorized AND is confident enough
    (>= CONFIDENCE_FLOOR) - a low-confidence "incorrect" verdict isn't
    trusted any more than a low-confidence "correct" one. llm_exhausted is
    True ONLY when the whole provider/model chain failed
    (LLMExhaustedError), same meaning/consumer as the desk agents' identical
    field - run_pipeline.py stops the run early on this rather than
    repeating the same doomed wait on every remaining paper.
    """
    if not title and not abstract:
        return _empty_result("no title or abstract available to audit")

    try:
        parsed, model_used = call_llm_json(_SYSTEM_PROMPT, _build_user_prompt(entry_id, title, abstract))
    except LLMError as e:
        return _empty_result(f"LLM WAS called but every provider/model failed: {e}",
                              llm_exhausted=isinstance(e, LLMExhaustedError))

    correctly_categorized = bool(parsed.get("correctly_categorized", True))
    confidence = float(parsed.get("confidence") or 0.0)

    raw_modality = parsed.get("spatial_modality") or []
    spatial_modality = [m for m in raw_modality if m in _VALID_MODALITIES]
    # ST-only = exactly {spatial_transcriptomics}, nothing else - matches the
    # same rule already confirmed for the Methods Graph rendering filter
    # (see CLAUDE.md's 2026-09-19 design decision) - reused here rather than
    # inventing a different threshold. Added 2026-09-21, per Marta's ask, to
    # skip the methods/data desks on ST-only papers for this first version -
    # deliberately NOT proteomics-only or metabolomics-only, and NOT
    # multi-omics combos including ST, which still get processed.
    is_st_only = spatial_modality == ["spatial_transcriptomics"]

    return {
        "audited": True,
        "flagged": (not correctly_categorized) and confidence >= CONFIDENCE_FLOOR,
        "confidence": confidence,
        "reason": parsed.get("reason", ""),
        "spatial_modality": spatial_modality,
        "is_st_only": is_st_only,
        "model_used": model_used,
        "skip_reason": None,
        "llm_exhausted": False,
    }


def process_entry(db, entry_id: str, sheet: str) -> dict:
    """Looks up title/abstract from db.methods, runs the audit, and stages a
    needs_review flag only if the category looks wrong - a clean
    "correctly categorized" verdict isn't new information worth writing to
    the row, so nothing is staged for it (keeps staging.xlsx focused on
    things Marta actually needs to look at)."""
    row = db.methods.loc[db.methods["entry_id"] == entry_id]
    if row.empty:
        return _empty_result("entry not found in DB")

    title = str(row.iloc[0].get("title", "") or "")
    abstract = str(row.iloc[0].get("abstract", "") or "")

    result = audit_entry(entry_id, title, abstract)

    if result["flagged"]:
        staging.append_candidate(
            action="update_field", sheet=sheet, entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="category_audit_agent",
            curation_model=result["model_used"], confidence=result["confidence"],
            notes=f"Category audit flagged this entry as possibly NOT a computational "
                  f"method: {result['reason']}",
        )

    return result
