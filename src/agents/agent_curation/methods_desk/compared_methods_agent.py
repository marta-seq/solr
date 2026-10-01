"""
compared_methods_agent.py (methods desk)
For one seed paper: reads its methods section, asks the LLM which methods it
compares itself against and how each is cited, resolves each citation to a
DOI, matches against the existing DB or creates a new M_AUTO entry, and
stages everything. Returns a stats dict including the newly-created entries
so run_pipeline.py can queue them at depth+1 (subject to MAX_HOPS).

Nothing here writes to the master Excel - everything goes through
common.staging.append_candidate().

Newly-created comparison-method entries always target sheet="method_pub"
(a genuinely new comparison method is always a method, never an
application). But updates written back onto the SEED paper itself
(REVIEW_STATUS flags, appended comparison IDs) route by that paper's own
entry_id prefix via _sheet_for_paper_entry, same as data_fetch_agent.py -
NOT hardcoded to method_pub. This matters because the seed queue is
supposed to be method_pub-only, but some AP_pub rows have compound
category values (e.g. "Application; computational analysis - method")
that satisfy triage's keyword-based method-category check and slip in
anyway (caught live: AP_30/AP_31/AP_39/AP_43/AP_52 all have such compound
categories). Before this fix, an update written for one of those actually
landed in method_pub - since "AP_30" doesn't exist there, merge_candidates
created a brand-new, mostly-empty PHANTOM row with that same ID inside
method_pub, leaving the real AP_30 (with its real DOI/category) untouched
in AP_pub. Routing by prefix here is defense-in-depth on top of the
matching fix in mailroom/triage.py (which now excludes paper_type ==
"application" from the seed queue outright, rather than relying only on
the free-text category-keyword match).
"""

import re
from datetime import datetime, timezone

from ...common import config, staging, audit_log
from ...common.llm_client import call_llm_json, LLMError, LLMExhaustedError
from ...common.paper_fetcher import fetch_paper, get_agent_text, was_truncated, is_probably_real_content
from ...common.reference_list_parser import parse_reference_list
from ...common.reference_resolver import resolve_citation, is_confident
from ...common.db_loader import _normalize_method_name
from . import category_audit_agent

# Well-known, generic (non-spatial-omics-specific) technique names - checked
# BEFORE citation resolution, added 2026-09-28 after real duplicates/
# mis-tagging were found live: SVM, XGBoost, and a LightGBM-adjacent paper
# were each independently created and mis-tagged spatial_proteomics, because
# their real (classic/foundational) papers have no fetchable abstract on
# Crossref - the independent modality classification below silently had
# nothing to classify and fell back to inheriting the SEED paper's modality,
# reproducing the exact bug that fallback was meant to avoid. A citation to
# any of these well-known generic names now routes straight to the same
# placeholder/auto-complete treatment as `is_baseline_group` - fast, free,
# and doesn't depend on an abstract being fetchable at all. Deliberately a
# denylist of specific, unambiguous, genuinely-generic names, not a broad
# pattern - reuses Marta's own existing M_PH_* placeholder conventions
# (Watershed, UNET, kmeans, Mask-R-CNN, ResNet, etc. were already being
# placeholder'd by hand) plus the classical-ML/GNN baseline names she listed
# in notes.md. Expect to keep growing this list as new cases surface.
_GENERIC_TECHNIQUE_NAMES = {
    _normalize_method_name(n) for n in [
        "SVM", "Support Vector Machine", "Support Vector Machines",
        "Random Forest", "RF",
        "Logistic Regression",
        "Decision Tree",
        "Naive Bayes",
        "Gradient Boosting", "Gradient Boosting Classifier",
        "AdaBoost", "AdaBoost Classifier",
        "XGBoost",
        "LightGBM", "Light GBM",
        "MLP", "Multi-layer Perceptron", "Multilayer Perceptron",
        "kNN", "k-Nearest Neighbors",
        "K-means", "kmeans",
        "PCA", "Principal Component Analysis",
        "UMAP", "t-SNE",
        "GCN", "Graph Convolutional Network", "Graph Convolutional Networks",
        "GAT", "Graph Attention Network", "Graph Attention Networks",
        "GIN", "Graph Isomorphism Network",
        "GraphSAGE", "SAGEConv", "GraphConv",
        "U-Net", "UNET", "ResNet", "VGG", "AlexNet",
        "Pix2Pix", "CycleGAN", "GAN",
        "Mask R-CNN", "Mask-R-CNN", "MaskRCNN",
        "Watershed", "Noise2Void",
    ]
}


def _is_generic_technique(method_name: str) -> bool:
    return _normalize_method_name(method_name) in _GENERIC_TECHNIQUE_NAMES


def _today_stamp() -> int:
    """Same YYYYMMDD-integer format scan.py already uses for
    Date(added_to_dataset) - reused here since this agent never stamped it
    at all before (found 2026-09-27: neither desk agent stamped a creation
    date on new entries, unlike the scanner and manual entries)."""
    return int(datetime.now(timezone.utc).strftime("%Y%m%d"))

# Per-field review-provenance column added 2026-09-20 (see CLAUDE.md's
# "Per-field review-provenance columns" design decision) - tracks whether
# THIS paper's comparison list specifically has ever been attempted,
# independent of the whole-row REVIEW_STATUS. Values: "NA" (never attempted),
# "auto" (an agent attempted it, whether or not it found anything), "manual"
# (a human populated/confirmed it). Checking this INSTEAD of relying on
# triage.py's whole-row REVIEW_STATUS check fixes two real bugs found
# 2026-09-20: (1) a paper where the LLM was called and found ZERO comparisons
# previously got no REVIEW_STATUS stamp at all, so it was re-fetched and
# re-processed on every future run forever; (2) a citation-chased M_AUTO_*
# entry, once merged at REVIEW_STATUS="auto-1", was permanently excluded from
# ever having its OWN comparisons checked in a later run (the "auto-N always
# excluded" rule in triage.py), even though only the whole-row status had
# been touched, not this specific field.
_COMPARISON_STATUS_COL = "METHOD_COMPARISON_REVIEW_STATUS"


def _comparison_status(methods_df, entry_id: str) -> str:
    if _COMPARISON_STATUS_COL not in methods_df.columns:
        return ""
    row = methods_df.loc[methods_df["entry_id"] == entry_id, _COMPARISON_STATUS_COL]
    return str(row.iloc[0]).strip().lower() if not row.empty else ""

SYSTEM_PROMPT = """You are a careful research assistant extracting factual information from a \
scientific paper's methods section. You will be given the methods section text (or, if that \
wasn't available, the full paper text). Identify every OTHER computational method or tool that \
this paper compares its own method against (baseline comparisons, benchmark methods) - do NOT \
include methods that are merely used as a preprocessing step or a component of the paper's own \
pipeline, only genuine comparison/benchmark methods.

STRICT RULES:
- method_name MUST be an actual named algorithm, software tool, or pipeline (e.g. "Cellpose", \
"StarDist", "Mesmer") - a proper noun for a specific piece of software/method.
- method_name must NEVER be a generic author-year citation (e.g. "Zeisel et al." or "Smith et \
al., 2020" is NOT a method name). If a citation doesn't name a specific tool/algorithm - even if \
it appears in a comparison-like sentence - skip it entirely. Only extract it if the paper itself \
gives the compared method a proper name.
- Do NOT include datasets, biological findings, or general background citations, even if phrased \
similarly to a comparison.

Two real failure modes to avoid, found during review of real extractions (2026-09-27) - both \
directions matter equally, read carefully:
1. FALSE POSITIVE - do NOT extract a tool the paper APPLIES as a step in its OWN analysis, even \
if it's a well-known named tool with its own citation. Real examples that were wrongly extracted \
as "comparisons": a paper using SHAP for model explainability, or NicheNet for cell-cell \
communication scoring, as part of its OWN pipeline - these are components the paper's method is \
BUILT FROM, not competing methods it benchmarks ITSELF against. Ask: "does this paper claim its \
own method outperforms, matches, or differs from this other method on the same task" - not just \
"is this tool mentioned or used somewhere in the paper."
2. FALSE NEGATIVE - do NOT miss an explicit list of named baselines just because they're all named \
in one sentence. Real example that was missed entirely (left blank) despite being this explicit: \
"We compared TopKAT to existing approaches ... including SPOT, FunSpace, and SPF." Extract EVERY \
named method in a list like that as a separate item, not just the first one.

Generic baseline FAMILIES: sometimes a paper benchmarks against a broad set of standard, \
off-the-shelf models rather than one specific named published method with its own paper - e.g. \
several classical ML classifiers (random forest, SVM, logistic regression, AdaBoost, MLP, decision \
tree, naive Bayes, gradient boosting) or several generic graph-neural-network variants (GCN, GAT, \
GIN, GraphSAGE, GraphConv). These are real and worth recording, but do NOT create one item per \
variant and do NOT try to find individual citations for them - instead return ONE item for the \
whole family, with "is_baseline_group": true and "method_name" set to a short descriptive label \
built from the actual names mentioned (e.g. "Classical ML baselines (RF, SVM, Logistic Regression, \
AdaBoost)" or "Graph neural network baselines (GCN, GAT, GIN)"). Leave citation_marker and \
citation_text empty for these - no per-baseline citation lookup is needed.

Return ONLY a JSON array, nothing else, no markdown fences. Each element:
{
  "method_name": "the name of the compared method/tool, as written in the paper - MUST be an \
actual tool/algorithm proper name per the strict rules above, never an author-year citation. For \
a generic baseline family (is_baseline_group=true), a short descriptive label instead (see above)",
  "is_baseline_group": true/false (true ONLY for a generic baseline family as described above - \
false for every genuinely named, individually-citable method; should be false for the vast \
majority of items),
  "citation_marker": "ONLY digits, nothing else - e.g. '12' or '41'. Do NOT include the word \
'ref', brackets, parentheses, periods, or any other character - digits only. If multiple numbers \
are cited together (e.g. text shows '[12,13]'), return only the FIRST number. Leave this as a \
genuinely empty string only if there is truly no number anywhere near the mention (the paper \
uses author-year citations like 'Smith et al., 2020' instead of numbers), or if is_baseline_group \
is true.",
  "citation_text": "ONLY fill this in if citation_marker is a genuinely empty string AND \
is_baseline_group is false - the author-year citation as written near the mention (e.g. 'Smith \
et al., 2020'). Leave this empty if you filled citation_marker, or if is_baseline_group is true - \
never fill both."
}
If no comparison methods are mentioned, return an empty JSON array: []
"""


def _build_user_prompt(paper_entry_id: str, methods_text: str) -> str:
    return f"Methods section (paper {paper_entry_id}):\n\n{methods_text}"


def _describe_existing(methods_df, entry_id: str) -> str:
    """Human-readable label for an EXISTING entry being linked to (not
    created) - added 2026-09-21 so staging.xlsx's notes and the audit log
    show what a linked ID actually IS, not just the bare ID, without Marta
    having to manually cross-reference the master Excel during review. Only
    affects review-time text - the actual Method_comparison_P_ENTRY_ID field
    written to the master stays ID-only, same as every existing cross-
    reference in that column."""
    row = methods_df.loc[methods_df["entry_id"] == entry_id]
    if row.empty:
        return entry_id
    title = str(row.iloc[0].get("title", "") or "").strip()
    return f"{entry_id} ({title})" if title else entry_id


def _fallback_marker_from_text(citation_text: str) -> str:
    """Defensive fallback: even with an explicit prompt asking for digits-only
    markers, the LLM sometimes still puts marker-like text into citation_text
    instead (caught via testing on real output: 'ref. 41(' landed in
    citation_text with citation_marker left empty, so the reference-list
    lookup never got a chance to run and it fell straight to an unreliable
    Crossref search on garbled text). If citation_text looks like a bare
    reference marker (e.g. 'ref. 41', 'ref 41(', '[41]'), pull the number out
    so the reference-list lookup can still be attempted. Deliberately
    lenient about trailing junk (stray brackets/parens/periods) since OCR/
    extraction artifacts are common here - only the leading shape matters."""
    m = re.match(r"^\s*\[?\(?\s*ref\.?\s*(\d{1,3})\b", citation_text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^\s*\[(\d{1,3})\]", citation_text)
    return m.group(1) if m else ""


def _sheet_for_paper_entry(entry_id: str) -> str:
    """Which sheet the SEED paper itself lives in - AP_pub if its ID starts
    with 'AP', method_pub otherwise. Mirrors data_fetch_agent.py's helper of
    the same name. Only for writes targeting the seed paper's own row -
    genuinely NEW comparison-method entries always go to method_pub
    regardless of this (see module docstring)."""
    return "AP_pub" if str(entry_id).upper().startswith("AP") else "method_pub"


def _append_to_comparison_list(existing: str, new_id: str) -> str:
    """Mirrors the existing 'M_PR_5, M_PR_2, M_PH_1' comma-separated format
    already used in Method_comparison_P_ENTRY_ID."""
    existing = "" if existing is None else str(existing).strip()
    if existing in ("", "nan", "NA"):
        return new_id
    ids = [x.strip() for x in existing.split(",") if x.strip()]
    if new_id not in ids:
        ids.append(new_id)
    return ", ".join(ids)


def _empty_stats(skip_reason: str, new_queue_items=None, llm_exhausted: bool = False) -> dict:
    """Consistent return shape at every early-exit point. skip_reason is
    ALWAYS a non-empty string here (the LLM was never called) - callers
    should show it directly rather than leaving people to infer whether
    0 results means 'skipped' or 'called and found nothing'.
    llm_exhausted is True ONLY when the whole provider/model chain failed
    (LLMExhaustedError) - run_pipeline.py checks this to stop a run early
    instead of repeating the same doomed wait on every remaining paper."""
    return {
        "new_queue_items": new_queue_items or [],
        "total_encountered": 0,
        "total_resolved": 0,
        "skip_reason": skip_reason,
        "llm_exhausted": llm_exhausted,
    }


def process_paper(db, paper_entry: dict, fetched: dict = None, reference_map: dict = None,
                   seed_spatial_modality: list = None) -> dict:
    """
    paper_entry: {"entry_id": ..., "doi": ..., "depth": ...}
    fetched/reference_map: pass these in if the orchestrator already fetched
    this paper (e.g. for a shared pass with data_fetch_agent), to avoid
    re-fetching/re-parsing the same paper twice.
    seed_spatial_modality: the SEED paper's own spatial_modality (from
    category_audit_agent, e.g. ["spatial_proteomics"]) - propagated onto any
    newly-created comparison-method entry's spatial_data_category, added
    2026-09-21 per Marta's ask so new entries aren't left with this field
    blank (which would otherwise force a full LLM audit call to determine it
    later). This is an INHERITED default, not independently verified for
    the comparison method itself - flagged as such in that entry's notes,
    since a paper can in principle cite a cross-modality comparison.

    Returns {"new_queue_items": [...], "total_encountered": N,
    "total_resolved": M, "skip_reason": str or None}. skip_reason is None
    ONLY when the LLM was genuinely called - always check this field rather
    than inferring from a 0 count, since 0 encountered can otherwise mean
    either "skipped, never called" or "called, found nothing".
    new_queue_items is {"entry_id": ..., "doi": ..., "depth": depth+1} dicts
    for newly-created method entries, ready to feed back into the seed
    queue - caller is responsible for checking MAX_HOPS/MAX_PAPERS_PER_RUN
    before actually queueing them.
    """
    entry_id = paper_entry["entry_id"]
    doi = paper_entry["doi"]
    depth = paper_entry["depth"]

    existing_status = _comparison_status(db.methods, entry_id)
    if existing_status in ("auto", "manual"):
        return _empty_stats(f"comparison list already {existing_status} "
                             f"({_COMPARISON_STATUS_COL}) - not re-attempting")

    if fetched is None:
        fetched = fetch_paper(doi)
    methods_text, text_source = get_agent_text(fetched, "methods")

    if not methods_text:
        staging.append_candidate(
            action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"No usable text could be fetched for this DOI (fetch source: {fetched['source']}).",
        )
        return _empty_stats(f"no text could be fetched (fetch source: {fetched['source']})")

    if not is_probably_real_content(methods_text):
        staging.append_candidate(
            action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"Extracted text (source: {text_source}) failed the content quality check - "
                  f"looks like website navigation/boilerplate rather than real article text. "
                  f"Skipped the LLM call to avoid wasting budget on unreliable input.",
        )
        return _empty_stats(f"text (source: {text_source}) failed the content-quality check "
                             f"(looked like navigation/boilerplate, not real article text)")

    if config.REQUIRE_ISOLATED_SECTION and text_source != "section:methods":
        staging.append_candidate(
            action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"Could not cleanly isolate the methods section (got '{text_source}' instead) - "
                  f"skipped the LLM call rather than spend a query on unfocused/low-confidence text.",
        )
        return _empty_stats(f"could not cleanly isolate the methods section "
                             f"(got '{text_source}' instead of 'section:methods') - "
                             f"REQUIRE_ISOLATED_SECTION blocked the LLM call")



    truncated_note = ""
    if was_truncated(fetched, "methods"):
        truncated_note = " (input text was truncated - findings may be incomplete)"

    if reference_map is None:
        references_text, _ = get_agent_text(fetched, "references")
        reference_map = parse_reference_list(references_text) if references_text else {}

    try:
        # only_provider=["gemini", "groq", "cloudflare"] (Gemini added
        # 2026-09-21, superseding the earlier skip_openrouter=True choice;
        # Groq + Cloudflare added same day after Gemini's real quota turned
        # out genuinely exhausted - both confirmed genuinely free, no card,
        # not trials): real testing that day (the PENGUIN paper, denser text
        # than the earlier MAPS validation) showed qwen2.5:14b can ignore the
        # "return ONLY a JSON array" instruction on harder input, and
        # qwen2.5:32b is too slow to be viable (300s+/call) - neither
        # reliable enough to be primary. Gemini was reliable and fast (~13s)
        # on every real test. Still deliberately bounded, no OpenRouter,
        # no OpenAI/Grok/DeepSeek/Cerebras (all confirmed to require payment
        # now, not genuinely free - see CLAUDE.md) - see
        # _build_provider_chain()'s docstring: once every listed provider is
        # exhausted, LLMExhaustedError fires so run_pipeline.py stops the run
        # rather than silently degrading further or looping.
        extracted, model_used = call_llm_json(
            SYSTEM_PROMPT, _build_user_prompt(entry_id, methods_text),
            only_provider=["gemini", "groq", "cloudflare"]
        )
    except LLMError as e:
        staging.append_candidate(
            action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
            fields={"REVIEW_STATUS": config.REVIEW_STATUS_NEEDS_REVIEW},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model="none", confidence=0.0,
            notes=f"LLM extraction failed: {e}",
        )
        return _empty_stats(f"LLM WAS called but every provider/model failed: {e}",
                             llm_exhausted=isinstance(e, LLMExhaustedError))

    if not isinstance(extracted, list):
        extracted = []

    new_queue_items = []
    comparison_ids_added = []
    item_outcomes = []

    for item in extracted:
        item = item or {}
        method_name = (item.get("method_name") or "").strip()
        is_baseline_group = bool(item.get("is_baseline_group"))
        citation_marker = (item.get("citation_marker") or "").strip()
        citation_text = (item.get("citation_text") or "").strip()

        if not method_name:
            item_outcomes.append({"outcome": "skipped", "reason": "no method_name in this item"})
            continue

        generic_hit = is_baseline_group or _is_generic_technique(method_name)
        if generic_hit:
            # Generic baseline family OR a well-known generic technique name
            # (see _GENERIC_TECHNIQUE_NAMES above) - added 2026-09-27/28 after
            # real batches were cluttered with one row per generic baseline
            # variant (RF/SVM/GCN/GAT/...), AND after real mis-tagging was
            # found where these got created as full entries anyway (when
            # individually citable) and mis-tagged spatial_proteomics because
            # their classic papers have no fetchable abstract for independent
            # classification to work with. No DOI resolution is attempted
            # even if a citation WAS given - created directly as a
            # "Placeholder - method" entry (the SAME literal category value
            # category_maps.py already treats as is_placeholder=True, so it's
            # excluded from the frontend graph/table and from every triage
            # pool exactly like any other placeholder stub, no new mechanism
            # needed) with COMPLETE="NA" since there's nothing further to
            # annotate on a generic/family placeholder.
            new_id = db.id_allocator.next_id("M_AUTO")
            reason = ("Generic baseline family grouped from a single comparison mention"
                      if is_baseline_group else
                      "Well-known generic technique (not spatial-omics-specific)")
            staging.append_candidate(
                action="create_entry", sheet="method_pub", entry_id=new_id,
                fields={
                    "category": "Placeholder - method",
                    "name": method_name,
                    "REVIEW_STATUS": config.REVIEW_STATUS_AUTO,
                    "addition_method": config.ADDITION_METHOD_CITATION_CHASE,
                    "COMPLETE": "NA",
                    "Date(added_to_dataset)": _today_stamp(),
                    "Internal notes": f"{reason} in {entry_id} - not a specific citable spatial-"
                                       f"omics method, no DOI resolution attempted.",
                },
                source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
                curation_model=model_used, confidence=None,
                notes=f"{reason} in {entry_id}: '{method_name}'.",
            )
            db.name_index.add(method_name, new_id)
            comparison_ids_added.append(new_id)
            item_outcomes.append({"outcome": "created_new", "entry_id": new_id,
                                   "confidence": None, "review_status": config.REVIEW_STATUS_AUTO})
            continue

        if not (citation_marker or citation_text):
            item_outcomes.append({"outcome": "skipped",
                                   "reason": "no citation info in this item"})
            continue

        if not citation_marker and citation_text:
            fallback_marker = _fallback_marker_from_text(citation_text)
            if fallback_marker:
                citation_marker = fallback_marker

        resolved = resolve_citation(citation_marker, citation_text, reference_map,
                                     expected_name_hint=method_name)
        confident = is_confident(resolved)
        resolved_doi = resolved["resolved_doi"]

        if resolved_doi and resolved_doi in db.doi_index:
            # already in the DB (or already staged earlier this run) - just link it
            matched_id = db.doi_index.lookup(resolved_doi)
            comparison_ids_added.append(matched_id)
            item_outcomes.append({"outcome": "linked_existing", "entry_id": matched_id,
                                   "description": _describe_existing(db.methods, matched_id),
                                   "matched_by": "doi", "resolved_doi": resolved_doi})
            continue

        # Name-based dedup, added 2026-09-27: DOI-based matching above only
        # catches a duplicate when both mentions share the exact same DOI -
        # a well-known tool is often cited casually by name with no
        # resolvable citation, or under a slightly different combined name
        # ("Mesmer" vs the existing entry's "Mesmer/DeepCell"), so a
        # DOI-only check let real duplicates through (confirmed live:
        # Mesmer, CellProfiler, PhenoGraph, Mask R-CNN each created twice).
        # Checked AFTER the DOI check (DOI stays the primary, more reliable
        # signal) and BEFORE the "couldn't resolve a DOI" rejection below,
        # so a name match can still short-circuit even when no DOI was
        # resolved at all. See db_loader.NameIndex for the matching rules.
        name_match_id = db.name_index.lookup(method_name)
        if name_match_id:
            comparison_ids_added.append(name_match_id)
            item_outcomes.append({"outcome": "linked_existing", "entry_id": name_match_id,
                                   "description": _describe_existing(db.methods, name_match_id),
                                   "matched_by": "name"})
            continue

        if not resolved_doi:
            # couldn't resolve a DOI at all - stage a note, nothing to link/create
            staging.append_candidate(
                action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
                fields={},
                source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
                curation_model=model_used, confidence=0.0,
                notes=f"Could not resolve DOI for compared method '{method_name}' "
                      f"(marker: '{citation_marker}', citation: {resolved['input_text'][:200]})",
            )
            item_outcomes.append({"outcome": "skipped",
                                   "reason": f"could not resolve a DOI for '{method_name}'"})
            continue

        # genuinely new - allocate an ID and stage a new M_AUTO entry
        # Deliberately NOT trying to guess the real category prefix (M_PR/M_SE/etc.) -
        # that requires understanding your category scheme, which you're planning to
        # redesign anyway. A distinct, obviously-generic prefix makes it clear during
        # review that this entry needs YOU to assign its real category, rather than
        # silently guessing wrong (caught via testing: SSAM/Baysor, both segmentation
        # methods, got created as M_PR_x when they should've been M_SE_x or matched
        # to their existing entries).
        new_id = db.id_allocator.next_id("M_AUTO")
        review_status = config.REVIEW_STATUS_AUTO if confident else config.REVIEW_STATUS_NEEDS_REVIEW

        new_fields = {
            "DOI": resolved_doi,
            "category": "computational analysis - method",
            "name": method_name,
            "REVIEW_STATUS": review_status,
            "addition_method": config.ADDITION_METHOD_CITATION_CHASE,
            "Date(added_to_dataset)": _today_stamp(),
        }
        modality_note = ""

        # Independent modality classification, added 2026-09-27 - previously
        # this new entry just inherited the SEED paper's spatial_data_category
        # unconditionally, which is wrong whenever the compared method is
        # itself a generic, non-spatial-specific tool (e.g. CellProfiler,
        # PhenoGraph) cited by a spatial paper: Marta's real complaint was
        # that citation-chasing "can grab papers that are only
        # transcriptomics" or otherwise out-of-scope tools and clutter the
        # agent desk with them. Reuses category_audit_agent's classification
        # via a cheap, disk-cached abstract-only fetch of the NEW entry's own
        # DOI - never blocks entry creation if this fails (network error, no
        # abstract available, audit LLM call fails), just falls back to the
        # old inherited-modality behavior below.
        own_modality = None
        try:
            own_fetched = fetch_paper(resolved_doi)
            own_abstract = (own_fetched.get("sections") or {}).get("abstract") or own_fetched.get("text", "")
            if own_abstract:
                own_audit = category_audit_agent.audit_entry(new_id, resolved["matched_title"], own_abstract)
                if own_audit.get("audited"):
                    own_modality = own_audit["spatial_modality"]
        except Exception as e:
            own_modality = None
            modality_note = f" (own-paper modality classification failed: {e})"

        if own_modality is not None:
            if not own_modality:
                # Confidently classified and NOT spatial-omics at all -
                # auto-complete rather than clutter the review queue with a
                # full-curation candidate for a generic, out-of-scope tool.
                new_fields["spatial_data_category"] = "not_spatial_omics"
                new_fields["COMPLETE"] = "NA"
                modality_note = (" Independently classified as NOT spatial-omics-specific "
                                  "- auto-completed (COMPLETE=NA).")
            elif own_modality == ["spatial_transcriptomics"]:
                new_fields["spatial_data_category"] = "spatial_transcriptomics"
                new_fields["COMPLETE"] = "skipped for now, purely ST"
                modality_note = (" Independently classified as spatial-transcriptomics-only "
                                  "- out of scope for now (COMPLETE=ST skip).")
            else:
                new_fields["spatial_data_category"] = ";".join(own_modality)
                modality_note = (f" Independently classified spatial_data_category: "
                                  f"{';'.join(own_modality)}.")
        else:
            # own_modality is None - independent classification unavailable
            # (most often: no fetchable abstract for this DOI at all) and
            # neither the baseline-group nor generic-technique-name checks
            # above caught it either. Changed 2026-09-28: this branch used to
            # blindly inherit the SEED paper's own modality - found live to
            # be actively wrong (SVM, XGBoost, Pix2Pix, and a LightGBM-
            # adjacent paper were all mis-tagged spatial_proteomics this way,
            # because classic/foundational papers frequently have no
            # fetchable Crossref abstract, so this branch fired almost every
            # time in practice - not the rare fallback it was designed to
            # be). An honest "unknown, needs review" is safer than a
            # confident wrong guess - leave spatial_data_category unset and
            # force review rather than silently propagating the seed's
            # modality onto a method that was never actually checked.
            new_fields["REVIEW_STATUS"] = config.REVIEW_STATUS_NEEDS_REVIEW
            seed_note = (f" ({entry_id}'s own modality is "
                         f"{';'.join(seed_spatial_modality)}, NOT applied here since it was "
                         f"never independently verified for this method)" if seed_spatial_modality
                         else "")
            modality_note = (modality_note +
                              f" Could not independently verify spatial_data_category (no "
                              f"fetchable abstract available) - left unset rather than "
                              f"guessing{seed_note}.")

        staging.append_candidate(
            action="create_entry", sheet="method_pub", entry_id=new_id,
            fields=new_fields,
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model=model_used, confidence=resolved["confidence"],
            notes=f"Discovered as a comparison method in {entry_id}. "
                  f"Matched Crossref title: '{resolved['matched_title']}'{truncated_note}{modality_note}",
        )

        # so later papers this same run see it and don't duplicate-create it
        db.doi_index.add(resolved_doi, new_id)
        db.name_index.add(method_name, new_id)
        comparison_ids_added.append(new_id)
        item_outcomes.append({"outcome": "created_new", "entry_id": new_id,
                               "confidence": resolved["confidence"], "review_status": review_status})

        new_queue_items.append({"entry_id": new_id, "doi": resolved_doi, "depth": depth + 1})

    audit_log.log_extraction("compared_methods_agent", entry_id, model_used, extracted, item_outcomes)

    # Always stamp the per-field review status once the LLM has genuinely
    # been called, whether or not it found anything - a "zero comparisons
    # found" result is a completed, successful pass, not a skip (see
    # _COMPARISON_STATUS_COL's module-level comment for the bug this fixes).
    review_fields = {
        _COMPARISON_STATUS_COL: "auto",
        "METHOD_COMPARISON_REVIEW_AGENT": "compared_methods_agent",
        "METHOD_COMPARISON_REVIEW_MODEL": model_used,
        "METHOD_COMPARISON_REVIEW_DATE": datetime.now(timezone.utc).isoformat(),
    }

    if comparison_ids_added:
        existing_value = ""
        row = db.methods.loc[db.methods["entry_id"] == entry_id, "Method_comparison_P_ENTRY_ID"]
        if not row.empty:
            existing_value = row.iloc[0]
        updated_value = existing_value
        for new_id in comparison_ids_added:
            updated_value = _append_to_comparison_list(updated_value, new_id)

        # Human-readable per-ID breakdown, not just "IDs were appended" - so
        # the notes column itself (not just the audit log) shows what each
        # linked ID actually is without cross-referencing the master Excel.
        added_descriptions = [
            (o.get("description") or o["entry_id"]) for o in item_outcomes
            if o.get("outcome") in ("linked_existing", "created_new") and o.get("entry_id") in comparison_ids_added
        ]
        staging.append_candidate(
            action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
            fields={"Method_comparison_P_ENTRY_ID": updated_value, **review_fields},
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model=model_used, confidence=None,
            notes="Appended comparison method IDs: " + "; ".join(added_descriptions),
        )
    else:
        staging.append_candidate(
            action="update_field", sheet=_sheet_for_paper_entry(entry_id), entry_id=entry_id,
            fields=review_fields,
            source_paper_entry_id=entry_id, curation_agent="compared_methods_agent",
            curation_model=model_used, confidence=None,
            notes=f"Methods desk pass completed - {len(extracted)} comparison mention(s) "
                  f"encountered, none resolved to a linkable/creatable entry."
                  if extracted else
                  "Methods desk pass completed - no comparison methods found in this paper.",
        )

    return {
        "new_queue_items": new_queue_items,
        "total_encountered": len(extracted),
        "total_resolved": len(comparison_ids_added),
        "skip_reason": None,
    }
