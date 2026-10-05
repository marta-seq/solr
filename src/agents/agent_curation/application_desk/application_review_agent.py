"""
application_review_agent.py (application desk)
For one AP_pub paper: extracts paper-level metadata (animal/tissue/disease/
type of spatial data) from its abstract, and - unless the paper is itself a
review/survey (see _is_review_category) - checks whether it describes data
it GENUINELY GENERATED ITSELF (not merely presented/discussed/cited
elsewhere), linking or creating the corresponding dataset entry. Explicitly
does NOT extract comparison methods - AP_pub papers never get that pass at
all (methods_desk is method_pub-only, see triage.build_seed_queue()'s own
exclusion).

Built 2026-10-04 after discussing scope directly with Marta. Two things she
was explicit about, both encoded below rather than left to LLM judgement
alone:
1. "we only want to get the data that is generated in the data [paper]! not
   just presented in the paper" - the dataset prompt's instructions mirror
   data_fetch_agent.py's existing `is_in_house` standard ("new, in-house-
   generated data from this paper's own lab" - NOT "reported for the first
   time" if actually produced by a collaborator) rather than inventing new
   wording, since that distinction is already proven there.
2. "If the paper is a review we can for example pass already that" - see
   _is_review_category(): a cheap category-text check (no LLM call needed)
   skips the dataset-detection question entirely for review/survey/
   perspective papers. Metadata extraction still runs regardless - a
   review's topic area is still useful, harmless information, and carries
   none of the "did it generate data" risk.

**Split into two INDEPENDENT calls, 2026-10-05** (originally one combined
call - see git history/SESSION_LOG.md if curious): metadata extraction
(animal/tissue/disease/technique) and dataset-detection are gated by their
own separate per-field status columns (APPLICATION_METADATA_STATUS /
DATASET_REVIEW_STATUS, mirroring compared_methods_agent.py's/
data_fetch_agent.py's existing METHOD_COMPARISON_REVIEW_STATUS/
DATASET_REVIEW_STATUS pattern) and use DIFFERENT model tiers:
- Metadata: `gemini_lite_first` - cheap, low-risk (worst case a field stays
  blank), and Marta's explicit ask was to actually test flash-lite here.
- Dataset detection: `gemini_flash_only` - a real 75-paper batch on
  gemini_lite_first showed a genuine false-positive problem specifically on
  THIS question (42/52 items flagged "self-generated" - implausibly high;
  confirmed live on AP_29, an "in situ sequencing" dataset claimed as
  self-generated even though the paper's own declared workflow only
  combines histology+MSI+Visium). Splitting into two calls (rather than one
  combined call on a single model) also means metadata keeps progressing on
  whichever quota is still available even when the other is exhausted -
  the two tasks no longer block each other.

Also handles **co-registered multi-modal datasets** (added 2026-10-05, per
Marta's explicit ask): when the SAME physical samples/slides were profiled
with multiple techniques (e.g. the AP_29 case above - MSI + Visium on the
same tissue sections, genuinely part of one multimodal workflow), these get
grouped under ONE shared base id (`D_MULTI_<YYYYMMDD>_<seq>`) with `.1`/
`.2`/`.3` suffixes identifying each technique-subset, tagged `spatial_multi`
- instead of creating unrelated single-technique entries. See the
`co_registered_group` field in SYSTEM_PROMPT_DATASET_ONLY and the grouping
pre-pass in process_paper().

Deliberately distinct from data_fetch_agent.py's dataset handling for method
papers in one structural way: method_pub's "DataID (data_used_in_the_paper)"
column means "every dataset this paper USED" (self-generated OR reused),
but AP_pub's "Associated data - produced by the paper " column means ONLY
data this paper itself produced - so the extraction schema here never even
asks about reused data at all (no relationship field, no origin-citation
resolution, no stub-paper creation) - simpler than data_fetch_agent.py's
schema by construction, not just by post-filtering.

Reuses data_fetch_agent.py's modality-routing/ID-prefix/access-link-dedup/
is_simulated-singleton helpers directly (same dataset-creation logic, just
gated to this paper's own self-generated mentions) rather than duplicating
them. Reuses intern_agent.py's backfill logic directly for the case where a
dataset row ALREADY exists linking back to this paper (e.g. created earlier
by a method paper's citation-chase, crediting this AP_pub paper as the
data's origin, long before this AP_pub paper itself was ever processed) -
in that case the right move is filling ITS blank fields, not creating a
second row.

Nothing here writes to the master Excel - everything goes through
common.staging.append_candidate().
"""

from datetime import datetime, timezone

from ...common import config, staging, audit_log
from ...common.doi_utils import normalize_doi
from ...common.llm_client import call_llm_json, LLMError, LLMExhaustedError
from ...common.paper_fetcher import fetch_paper, get_agent_text, is_probably_real_content
from ...living_ingestion.mailroom import triage
from ..data_desk import intern_agent
from ..data_desk import data_fetch_agent as dfa  # reuse its dataset-creation helpers directly


def _today_stamp() -> int:
    return int(datetime.now(timezone.utc).strftime("%Y%m%d"))


# Two independent per-field status columns (2026-10-05 split - see module
# docstring). DATASET_REVIEW_STATUS already existed (added 2026-09-20,
# never populated until this agent); APPLICATION_METADATA_STATUS is new,
# mirroring the exact same column-set convention.
_DATASET_STATUS_COL = "DATASET_REVIEW_STATUS"
_METADATA_STATUS_COL = "APPLICATION_METADATA_STATUS"
_ASSOC_DATA_COL = "Associated data - produced by the paper "

_DONE_VALUES = ("auto", "manual")


def _col_status(methods_df, entry_id: str, col: str) -> str:
    if col not in methods_df.columns:
        return ""
    row = methods_df.loc[methods_df["entry_id"] == entry_id, col]
    return str(row.iloc[0]).strip().lower() if not row.empty else ""


def _is_review_category(category: str) -> bool:
    """True if this AP_pub paper's own category marks it as a review/survey/
    perspective rather than an original-research application paper - real
    values confirmed live in the category column 2026-10-04: "Application
    review", "Technical review", "General omics review", "computational
    analysis - review", "perspectives/review" - all contain "review" as a
    substring, same lightweight keyword-match style already used elsewhere
    in this codebase (e.g. triage._is_method_category). Such papers
    essentially never generate new data themselves (they discuss/summarize
    others' work) - gated here so the dataset-detection question is never
    even asked for them, rather than relying on the LLM to always get this
    right on its own."""
    return "review" in str(category or "").lower()


def _clean(val) -> str:
    val = "" if val is None else str(val).strip()
    return "" if val.lower() in ("nan", "na", "none") else val


def _blank(val) -> bool:
    return not _clean(val)


SYSTEM_PROMPT_METADATA_ONLY = """You are a careful research assistant reading an application paper in \
spatial omics (a paper that APPLIES existing computational methods/techniques to answer a biological \
question). From the abstract given, extract:
1. "animal": the organism studied (e.g. "human", "mouse"), or empty string if not stated.
2. "tissue": the tissue/organ studied (e.g. "breast", "brain cortex"), or empty string if not stated.
3. "disease": the disease/condition studied, or "healthy" if explicitly described as healthy controls, \
or empty string if not stated.
4. "type_of_spatial_data": the specific spatial technique(s) used (e.g. "IMC", "CODEX", "Visium", \
"MERFISH"), or empty string if not stated.
Return ONLY a JSON object, nothing else, no markdown fences:
{"animal": "", "tissue": "", "disease": "", "type_of_spatial_data": ""}
"""

SYSTEM_PROMPT_DATASET_ONLY = """You are a careful research assistant reading an application paper in \
spatial omics (a paper that APPLIES existing computational methods/techniques to answer a biological \
question, rather than introducing a new method itself). From the data-availability text given, extract \
every dataset this paper's OWN methods explicitly describe ACTUALLY GENERATING/PRODUCING itself - e.g. a \
new imaging run, a new patient cohort, new tissue samples collected specifically for this study.

This is a STRICT bar: do NOT include a dataset merely because it is shown, analyzed, discussed, compared \
against, or cited as an example - only include it if the paper's own methods explicitly describe \
PRODUCING it. Use the same standard as "true in-house data from this paper's own lab" - NOT "reported \
here for the first time" if it was actually produced by a different/collaborating group whose data this \
paper is simply using. If genuinely unsure whether a dataset was produced by this paper or merely reused \
from elsewhere, do NOT include it - leave it out rather than guess. Return an empty array if this paper \
did not itself generate any data at all - this is the common case, since most application papers reuse \
existing published datasets rather than producing their own.

IMPORTANT real failure case to avoid (found 2026-10-05): a paper whose own declared workflow combines \
ONLY histology + mass spectrometry imaging + Visium (stated explicitly as its method) also had an \
unrelated "in situ sequencing" dataset wrongly extracted as self-generated - that technique was never \
part of the paper's own declared workflow at all, so it must have been a reused/comparison dataset \
mentioned elsewhere in the text, not something this paper produced. Before including ANY dataset, check \
it uses a technique/platform the paper's own methods explicitly say THEY used to generate data - a \
technique mentioned elsewhere in the paper (introduction, comparison, discussion) that isn't part of the \
paper's own stated data-generation workflow is a strong signal it's reused, not self-generated.

Return ONLY a JSON object, nothing else, no markdown fences:
{"datasets": []}

Each element of "datasets":
{
  "accession": "repository accession number (e.g. GEO GSExxxxx, ArrayExpress E-MTAB-xxxx), or empty \
string if none given",
  "data_doi": "a DOI printed for the dataset ITSELF (common for Zenodo/Dryad/figshare), or empty string",
  "access_link": "a URL where the data can be accessed, or empty string",
  "technique": "the spatial technique used for this dataset if stated (e.g. 'IMC', 'CODEX', 'MERFISH', \
'Visium'), or empty string if not stated",
  "modality": "spatial_proteomics" | "spatial_transcriptomics" | "spatial_multi" | "non_spatial" | \
"unknown",
  "organism": "", "tissue": "", "disease": "", "n_images": 0, "n_patients": 0,
  "is_simulated": false,
  "co_registered_group": "if this dataset is the SAME physical samples/tissue slides as another item \
in this array, just measured with a DIFFERENT technique (e.g. the same sections profiled with both MSI \
and Visium as part of one multimodal workflow) - give both items the SAME short label here (e.g. 'A') \
so they can be linked as one multi-modal dataset. Leave empty string if this dataset stands alone (not \
co-registered with anything else in this array)."
}

Do NOT include a dataset that is purely dissociated single-cell data with no spatial component (plain \
scRNA-seq/scATAC-seq, no tissue coordinates/imaging) - set "modality" to "non_spatial" for it instead of \
omitting it silently, so it's clear one was seen and deliberately excluded.
"""


def _build_user_prompt(entry_id: str, text: str) -> str:
    return f"Application paper {entry_id}:\n\n{text}"


def _empty_stats(skip_reason: str, llm_exhausted: bool = False) -> dict:
    return {
        "metadata_filled": [], "linked_ids": [], "total_encountered": 0, "total_resolved": 0,
        "skip_reason": skip_reason, "llm_exhausted": llm_exhausted,
    }


def _find_linked_dataset(datasets_df, entry_id: str, doi: str):
    """Reverse lookup: does a dataset row ALREADY exist crediting THIS paper
    as its origin (paper_ENTRY_ID or paper_DOI match)? Real case this
    matters for: a method paper's citation-chase can create a dataset row
    (data_fetch_agent.py's "reused_external" resolution) crediting an
    application paper as the data's origin DOI, long before that
    application paper is ever itself processed by this agent - without this
    check, this agent would create a SECOND, duplicate dataset row for the
    exact same data the first time it ran. Checked by paper_ENTRY_ID first
    (exact, cheap), falling back to normalized paper_DOI (covers the case
    where the earlier-created row only ever recorded the DOI, not the
    entry_id, of its origin paper)."""
    by_id = datasets_df[datasets_df["paper_ENTRY_ID"].astype(str).str.strip() == str(entry_id).strip()]
    if not by_id.empty:
        return by_id.iloc[0]["entry_id"]
    if doi:
        target = normalize_doi(doi)
        col = datasets_df["paper_DOI"].astype(str).map(normalize_doi)
        by_doi = datasets_df[col == target]
        if not by_doi.empty:
            return by_doi.iloc[0]["entry_id"]
    return None


def _missing_fields_for(row, columns) -> list:
    fields = triage._fields_for_modality(row.get("spatial_data_category", ""), columns)
    return [f for f in fields if triage._is_empty(row.get(f))]


def _run_metadata_task(db, entry_id: str, row0, abstract: str) -> tuple:
    """Independent sub-task 1/2: fills animal/tissue/disease/technique from
    the abstract alone, gemini_lite_first. Returns (metadata_filled,
    llm_exhausted). Stages its own update_field call and stamps
    APPLICATION_METADATA_STATUS - does NOT touch DATASET_REVIEW_STATUS at
    all, so this can succeed/fail completely independently of the dataset
    sub-task (see module docstring)."""
    if not abstract:
        return [], False  # nothing to extract from - leave unstamped for a future run with a better fetch

    try:
        parsed, model_used = call_llm_json(
            SYSTEM_PROMPT_METADATA_ONLY, _build_user_prompt(entry_id, abstract),
            only_provider="gemini_lite_first"
        )
    except LLMError as e:
        staging.append_candidate(
            action="update_field", sheet="AP_pub", entry_id=entry_id, fields={},
            source_paper_entry_id=entry_id, curation_agent="application_review_agent",
            curation_model="none", confidence=0.0, notes=f"Metadata LLM extraction failed: {e}",
        )
        return [], isinstance(e, LLMExhaustedError)

    if not isinstance(parsed, dict):
        parsed = {}

    metadata_fields = {}
    metadata_filled = []
    for col, key in (("animal", "animal"), ("tissue", "tissue"), ("disease", "disease"),
                      ("type of spatial data", "type_of_spatial_data")):
        if _blank(row0.get(col)):
            val = _clean(parsed.get(key, ""))
            if val:
                metadata_fields[col] = val
                metadata_filled.append(col)

    metadata_fields.update({
        _METADATA_STATUS_COL: "auto",
        "APPLICATION_METADATA_AGENT": "application_review_agent",
        "APPLICATION_METADATA_MODEL": model_used,
        "APPLICATION_METADATA_DATE": datetime.now(timezone.utc).isoformat(),
    })
    staging.append_candidate(
        action="update_field", sheet="AP_pub", entry_id=entry_id, fields=metadata_fields,
        source_paper_entry_id=entry_id, curation_agent="application_review_agent",
        curation_model=model_used, confidence=None,
        notes=(f"Filled metadata fields: {', '.join(metadata_filled)}." if metadata_filled
               else "Metadata pass completed - nothing new to fill."),
    )
    return metadata_filled, False


def _stamp_dataset_status(entry_id: str, model_used: str, notes: str) -> None:
    staging.append_candidate(
        action="update_field", sheet="AP_pub", entry_id=entry_id,
        fields={
            _DATASET_STATUS_COL: "auto",
            "DATASET_REVIEW_AGENT": "application_review_agent",
            "DATASET_REVIEW_MODEL": model_used,
            "DATASET_REVIEW_DATE": datetime.now(timezone.utc).isoformat(),
        },
        source_paper_entry_id=entry_id, curation_agent="application_review_agent",
        curation_model=model_used, confidence=None, notes=notes,
    )


def _run_dataset_task(db, entry_id: str, doi: str, row0, is_review: bool, fetched: dict) -> tuple:
    """Independent sub-task 2/2: the self-generated-data question,
    gemini_flash_only (see module docstring for why this is NOT on
    flash-lite). Returns (linked_ids, total_encountered, total_resolved,
    llm_exhausted). Stages its own update_field call(s) and stamps
    DATASET_REVIEW_STATUS - does NOT touch APPLICATION_METADATA_STATUS."""
    if is_review:
        _stamp_dataset_status(entry_id, "none",
                               "Category marked as review/survey - dataset detection skipped by design.")
        return [], 0, 0, False

    data_text, data_source = get_agent_text(fetched, "data_availability")
    data_text_usable = bool(data_text) and is_probably_real_content(data_text) and (
        not config.REQUIRE_ISOLATED_SECTION or data_source == "section:data_availability"
    )
    if not data_text_usable:
        return [], 0, 0, False  # leave DATASET_REVIEW_STATUS unstamped - a future, better fetch might work

    try:
        parsed, model_used = call_llm_json(
            SYSTEM_PROMPT_DATASET_ONLY, _build_user_prompt(entry_id, data_text),
            only_provider="gemini_flash_only"
        )
    except LLMError as e:
        staging.append_candidate(
            action="update_field", sheet="AP_pub", entry_id=entry_id, fields={},
            source_paper_entry_id=entry_id, curation_agent="application_review_agent",
            curation_model="none", confidence=0.0, notes=f"Dataset-detection LLM extraction failed: {e}",
        )
        return [], 0, 0, isinstance(e, LLMExhaustedError)

    if not isinstance(parsed, dict):
        parsed = {}
    extracted = parsed.get("datasets") or []
    if not isinstance(extracted, list):
        extracted = []

    linked_ids = []
    item_outcomes = []

    # Pre-pass: find co_registered_group labels shared by 2+ items - these
    # are the SAME physical samples/slides profiled with different
    # techniques (e.g. MSI + Visium on the same tissue sections), per
    # Marta's explicit ask 2026-10-05. A label used by only 1 item has
    # nothing to actually group with, so it's ignored (treated as standalone
    # below). Solely for pairing - no IDs allocated yet here, that happens
    # lazily below so an existing-entry match still short-circuits normally
    # without wasting an ID.
    group_counts = {}
    for item in extracted:
        label = ((item or {}).get("co_registered_group") or "").strip()
        if label:
            group_counts[label] = group_counts.get(label, 0) + 1
    real_groups = {label for label, count in group_counts.items() if count >= 2}
    group_base_ids = {}      # label -> shared base id, allocated on first member created
    group_next_suffix = {}   # label -> next .N suffix to assign

    for item in extracted:
        item = item or {}
        accession = (item.get("accession") or "").strip()
        data_doi = normalize_doi(item.get("data_doi", ""))
        access_link = (item.get("access_link") or "").strip()
        technique = (item.get("technique") or "").strip()
        modality = (item.get("modality") or "").strip()
        organism = (item.get("organism") or "").strip()
        tissue = (item.get("tissue") or "").strip()
        disease = (item.get("disease") or "").strip()
        n_images = item.get("n_images") or ""
        n_patients = item.get("n_patients") or ""
        is_simulated = bool(item.get("is_simulated"))
        group_label = (item.get("co_registered_group") or "").strip()
        in_real_group = group_label in real_groups

        if modality == "non_spatial":
            item_outcomes.append({"outcome": "skipped",
                                   "reason": "modality is non_spatial (dissociated/single-cell, out of scope)"})
            continue

        # Simulated data routes to the shared singleton row, same as
        # data_fetch_agent.py - see that module's own comment for why.
        if is_simulated:
            sim_id = {"spatial_proteomics": "D_SP_simulation",
                      "spatial_transcriptomics": "D_ST_simulation"}.get(modality)
            if sim_id and sim_id in set(db.datasets["entry_id"]):
                linked_ids.append(sim_id)
                item_outcomes.append({"outcome": "linked_existing", "entry_id": sim_id,
                                       "matched_by": "simulated_singleton"})
                continue

        if not accession and not data_doi and not access_link:
            item_outcomes.append({"outcome": "skipped",
                                   "reason": "no accession/data_doi/access_link on this mention"})
            continue

        # Existing-entry matching, same priority order as data_fetch_agent.py:
        # data_doi -> accession -> access_link.
        existing_id = db.doi_index.lookup(data_doi) if data_doi else None
        matched_by = "data_doi" if existing_id else None
        if existing_id is None and accession:
            existing_id = dfa._find_by_accession(db.datasets, accession)
            matched_by = "accession" if existing_id else None
        if existing_id is None and access_link:
            existing_id = dfa._find_by_access_link(db.datasets, access_link)
            matched_by = "access_link" if existing_id else None

        if existing_id:
            linked_ids.append(existing_id)
            item_outcomes.append({"outcome": "linked_existing", "entry_id": existing_id,
                                   "description": dfa._describe_existing_dataset(db.datasets, existing_id),
                                   "matched_by": matched_by})
            continue

        # genuinely new - this paper is its own origin (no citation to
        # resolve, no stub source-paper needed, unlike data_fetch_agent.py's
        # reused_external path - this agent only ever sees self-generated
        # mentions by construction of its prompt, see module docstring).
        group_note = ""
        if in_real_group:
            # Same physical samples/slides, different technique - share ONE
            # base id (date + sequence, per Marta's explicit ask 2026-10-05)
            # across every member of this group, distinguished by a .N
            # suffix, and force modality to spatial_multi regardless of this
            # item's own single-technique modality (it's genuinely a
            # multi-modal dataset even though each item only names one of
            # the techniques involved).
            if group_label not in group_base_ids:
                group_base_ids[group_label] = db.id_allocator.next_id(f"D_MULTI_{_today_stamp()}")
                group_next_suffix[group_label] = 1
            base_id = group_base_ids[group_label]
            suffix = group_next_suffix[group_label]
            group_next_suffix[group_label] += 1
            new_id = f"{base_id}.{suffix}"
            modality = "spatial_multi"
            group_note = (f" Part of a co-registered multi-modal dataset (same samples, different "
                          f"techniques) - grouped under {base_id} with sibling subset(s).")
        else:
            new_id = db.id_allocator.next_id(
                f"{dfa._id_prefix_for_modality(modality)}_{dfa._technique_slug(technique)}")

        staging.append_candidate(
            action="create_entry", sheet="data", entry_id=new_id,
            fields={
                "data_DOI": data_doi,
                "data_accession_number": accession,
                "data_acess_link": access_link,
                "paper_DOI": doi,
                "paper_ENTRY_ID": entry_id,
                "spatial_data_category": modality,
                "spatial_data_method": technique,
                "REVIEW_STATUS": config.REVIEW_STATUS_AUTO,
                "addition_method": config.ADDITION_METHOD_CITATION_CHASE,
                "organism": organism, "tissue": tissue, "disease": disease,
                "N images": n_images, "N patients": n_patients,
                "Date(added_to_dataset)": _today_stamp(),
            },
            source_paper_entry_id=entry_id, curation_agent="application_review_agent",
            curation_model=model_used, confidence=None,
            notes=f"Extracted as data genuinely produced by {entry_id} itself "
                  f"(application-desk pass, not a reused/external mention).{group_note}",
        )
        if data_doi:
            db.doi_index.add(data_doi, new_id)
        linked_ids.append(new_id)
        item_outcomes.append({"outcome": "created_new", "entry_id": new_id})

    # --- reverse check: does a dataset row ALREADY exist crediting this
    # paper as its origin, even though we found nothing new above (or in
    # addition to it)? Backfills rather than duplicates - see
    # _find_linked_dataset()'s own docstring for the real case this covers. ---
    already_linked_id = _find_linked_dataset(db.datasets, entry_id, doi)
    if already_linked_id and already_linked_id not in linked_ids:
        linked_ids.append(already_linked_id)
        data_row = db.datasets.loc[db.datasets["entry_id"] == already_linked_id]
        if not data_row.empty:
            missing = _missing_fields_for(data_row.iloc[0], db.datasets.columns)
            if missing:
                try:
                    intern_agent.process_entry(db, {"entry_id": already_linked_id, "missing_fields": missing})
                except Exception:
                    pass  # best-effort backfill - a failure here shouldn't block the rest of this pass

    assoc_fields = {}
    if linked_ids and _blank(row0.get(_ASSOC_DATA_COL)):
        assoc_fields[_ASSOC_DATA_COL] = "; ".join(dict.fromkeys(linked_ids))  # de-duplicated, order-preserving

    audit_log.log_extraction("application_review_agent", entry_id, model_used, extracted, item_outcomes)

    notes_parts = []
    if linked_ids:
        notes_parts.append(f"Linked/created dataset(s): {', '.join(linked_ids)}.")
    if not notes_parts:
        notes_parts.append("Dataset-detection pass completed - nothing new found to fill or link.")

    staging.append_candidate(
        action="update_field", sheet="AP_pub", entry_id=entry_id,
        fields={
            **assoc_fields,
            _DATASET_STATUS_COL: "auto",
            "DATASET_REVIEW_AGENT": "application_review_agent",
            "DATASET_REVIEW_MODEL": model_used,
            "DATASET_REVIEW_DATE": datetime.now(timezone.utc).isoformat(),
        },
        source_paper_entry_id=entry_id, curation_agent="application_review_agent",
        curation_model=model_used, confidence=None,
        notes=" ".join(notes_parts),
    )

    total_resolved = len([o for o in item_outcomes if o["outcome"] != "skipped"])
    return linked_ids, len(extracted), total_resolved, False


def process_paper(db, paper_entry: dict, fetched: dict = None,
                   attempt_metadata: bool = True, attempt_dataset: bool = True) -> dict:
    """
    paper_entry: {"entry_id": ..., "doi": ...} (an AP_pub row; not recursive,
    nothing here is ever queued for further processing).

    Runs TWO independent sub-tasks (see module docstring for why they're
    split): metadata extraction (gemini_lite_first) and dataset detection
    (gemini_flash_only). Either, both, or neither may run on a given call,
    depending on each sub-task's OWN per-field status column AND the
    attempt_metadata/attempt_dataset flags - a paper whose metadata is
    already done but dataset detection isn't (e.g. because flash's quota
    was exhausted on an earlier run) will only re-attempt the dataset half
    here, not redo the metadata.

    attempt_metadata/attempt_dataset (added 2026-10-05): let the CALLER
    (run_pipeline.py) disable one sub-task for the rest of a run once IT
    specifically has been found exhausted, without stopping the other -
    critical given the two sub-tasks use different provider chains with
    independent quotas (gemini_lite_first vs gemini_flash_only): without
    this, the first flash exhaustion on paper N's dataset check would
    otherwise halt metadata progress on papers N+1..end too, defeating the
    whole point of splitting them.

    Returns {"metadata_filled": [...], "linked_ids": [...],
    "total_encountered": N, "total_resolved": M, "skip_reason": str or
    None, "metadata_exhausted": bool, "dataset_exhausted": bool,
    "llm_exhausted": bool (true if EITHER sub-task exhausted - kept for
    back-compat with callers checking only this field)}. skip_reason is
    None if EITHER sub-task was genuinely attempted (even if the other was
    skipped/already done/disabled) - only non-None when NEITHER ran at all.
    """
    entry_id = paper_entry["entry_id"]
    doi = paper_entry["doi"]

    metadata_done = _col_status(db.methods, entry_id, _METADATA_STATUS_COL) in _DONE_VALUES
    dataset_done = _col_status(db.methods, entry_id, _DATASET_STATUS_COL) in _DONE_VALUES
    if (metadata_done or not attempt_metadata) and (dataset_done or not attempt_dataset):
        return {**_empty_stats(f"both sub-tasks already done or disabled this run "
                                f"({_METADATA_STATUS_COL}, {_DATASET_STATUS_COL})"),
                "metadata_exhausted": False, "dataset_exhausted": False}

    row = db.methods.loc[db.methods["entry_id"] == entry_id]
    if row.empty:
        return {**_empty_stats("entry not found in DB"),
                "metadata_exhausted": False, "dataset_exhausted": False}
    row0 = row.iloc[0]
    is_review = _is_review_category(row0.get("category", ""))
    abstract = _clean(row0.get("abstract", ""))

    attempted_anything = False
    metadata_exhausted = False
    dataset_exhausted = False
    metadata_filled = []
    linked_ids = []
    total_encountered = 0
    total_resolved = 0

    if not metadata_done and attempt_metadata:
        metadata_filled, metadata_exhausted = _run_metadata_task(db, entry_id, row0, abstract)
        if abstract:
            attempted_anything = True

    if not dataset_done and attempt_dataset:
        if fetched is None:
            fetched = fetch_paper(doi)
        linked_ids, total_encountered, total_resolved, dataset_exhausted = _run_dataset_task(
            db, entry_id, doi, row0, is_review, fetched
        )
        if is_review or total_encountered or linked_ids:
            attempted_anything = True

    if not attempted_anything:
        return {**_empty_stats("no usable text available for either enabled sub-task this run "
                                "(left unstamped for a future run with a better fetch)"),
                "metadata_exhausted": metadata_exhausted, "dataset_exhausted": dataset_exhausted}

    return {
        "metadata_filled": metadata_filled,
        "linked_ids": linked_ids,
        "total_encountered": total_encountered,
        "total_resolved": total_resolved,
        "skip_reason": None,
        "metadata_exhausted": metadata_exhausted,
        "dataset_exhausted": dataset_exhausted,
        "llm_exhausted": metadata_exhausted or dataset_exhausted,
    }
