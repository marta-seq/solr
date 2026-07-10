"""
intern_agent.py (data desk)
Works off mailroom.triage.build_data_pool()'s worklist: for one dataset
entry with missing metadata fields, finds the associated paper's abstract
and asks the LLM to fill in ONLY the fields it can support with what the
abstract actually says - no guessing. Not recursive, not queued anywhere -
this is the independent "data pool" track (see the pipeline diagram).

Nothing here writes to the master Excel - everything goes through
common.staging.append_candidate().
"""

from ..common import config, staging
from ..common.llm_client import call_llm_json, LLMError
from ..common.paper_fetcher import fetch_paper, get_agent_text

FIELD_DESCRIPTIONS = {
    "organism": "the organism the sample is from (e.g. 'human', 'mouse')",
    "tissue": "the tissue or organ sampled (e.g. 'breast', 'brain cortex')",
    "disease": "the disease/condition studied, or 'healthy' if explicitly described as healthy controls",
    "spatial_data_method": "the specific spatial technique used (e.g. 'IMC', 'Xenium', 'MERFISH')",
    "N samples": "the number of samples/sections profiled, as a plain number",
    "N markers": "the number of protein markers used (proteomics only), as a plain number",
    "Marker": "the list of protein marker names, semicolon-separated",
    "N genes": "the number of genes profiled (transcriptomics only), as a plain number",
    "Genes": "the list of gene names, semicolon-separated, ONLY if a small targeted panel is named",
}


def _build_system_prompt(fields_needed: list) -> str:
    field_lines = "\n".join(
        f'  "{f}": "{FIELD_DESCRIPTIONS.get(f, f)}"' for f in fields_needed
    )
    return f"""You are a careful research assistant filling in dataset metadata from a paper's \
abstract. You will be given the abstract and a list of fields to fill in. For EACH field, only \
fill it in if the abstract explicitly supports it - if the abstract doesn't clearly state a \
field, leave it as an empty string. Do not guess or infer beyond what the text actually says.

Return ONLY a JSON object, nothing else, no markdown fences, with exactly these keys:
{{
{field_lines}
}}
"""


def _get_abstract(db, paper_entry_id: str, paper_doi: str) -> tuple:
    """Returns (abstract_text, source_label). Prefers the already-fetched
    'abstract' column in methods_metadata.csv (free - no new API call) over
    fetching fresh, since 02_fetch_metadata.py already did this for every
    real paper in the sheet. Falls back to a fresh fetch for anything not
    yet in that sheet (e.g. a stub Application paper created this same run).

    paper_ENTRY_ID is often blank on a dataset row even when paper_DOI is
    filled in and that DOI genuinely matches an existing paper elsewhere in
    methods.csv (caught via testing) - so if paper_entry_id isn't given
    directly, resolve it via the DOI index before giving up on the free
    precomputed abstract."""
    resolved_entry_id = paper_entry_id
    if not resolved_entry_id and paper_doi:
        looked_up = db.doi_index.lookup(paper_doi)
        if looked_up:
            resolved_entry_id = looked_up

    if resolved_entry_id:
        row = db.methods.loc[db.methods["entry_id"] == resolved_entry_id]
        if not row.empty:
            abstract = row.iloc[0].get("abstract", "")
            if abstract and str(abstract).strip().lower() not in ("", "nan"):
                return str(abstract).strip(), "precomputed_metadata_csv"

    if paper_doi:
        fetched = fetch_paper(paper_doi)
        abstract, _ = get_agent_text(fetched, "abstract")
        if abstract:
            return abstract, f"fresh_fetch:{fetched['source']}"
        if fetched.get("text"):
            return fetched["text"], f"fresh_fetch_full_text:{fetched['source']}"

    return "", "none"


def _sheet_for_entry(entry_id: str) -> str:
    """Infers which real sheet this dataset entry lives in from its ID prefix -
    needed since staging.append_candidate requires a target sheet name and
    build_data_pool()'s worklist doesn't carry that through."""
    if entry_id.startswith("D_SP"):
        return "Data_SP"
    if entry_id.startswith("D_ST"):
        return "Data_ST"
    return "Data_multi"


def _clean(val) -> str:
    val = str(val).strip() if val is not None else ""
    return "" if val.lower() in ("nan", "na", "none") else val


def process_entry(db, data_item: dict) -> dict:
    """
    data_item: {"entry_id": ..., "missing_fields": [...]} from
    mailroom.triage.build_data_pool().

    Returns {"filled": [...], "skipped_reason": str or None} for logging/testing.
    """
    entry_id = data_item["entry_id"]
    missing_fields = data_item["missing_fields"]
    sheet = _sheet_for_entry(entry_id)

    row = db.datasets.loc[db.datasets["entry_id"] == entry_id]
    if row.empty:
        return {"filled": [], "skipped_reason": "entry_id not found in datasets"}
    row = row.iloc[0]

    paper_entry_id = _clean(row.get("paper_ENTRY_ID", ""))
    paper_doi = _clean(row.get("paper_DOI", ""))

    if not paper_entry_id and not paper_doi:
        staging.append_candidate(
            action="update_field", sheet=sheet, entry_id=entry_id, fields={},
            source_paper_entry_id="", curation_agent="intern_agent",
            curation_model="none", confidence=0.0,
            notes="No paper_ENTRY_ID or paper_DOI on this dataset row - "
                  "can't find an abstract to curate from.",
        )
        return {"filled": [], "skipped_reason": "no paper link"}

    abstract, abstract_source = _get_abstract(db, paper_entry_id, paper_doi)
    if not abstract:
        staging.append_candidate(
            action="update_field", sheet=sheet, entry_id=entry_id, fields={},
            source_paper_entry_id=paper_entry_id, curation_agent="intern_agent",
            curation_model="none", confidence=0.0,
            notes="No abstract could be found for the associated paper.",
        )
        return {"filled": [], "skipped_reason": "no abstract"}

    try:
        result, model_used = call_llm_json(
            _build_system_prompt(missing_fields), f"Abstract:\n\n{abstract}"
        )
    except LLMError as e:
        staging.append_candidate(
            action="update_field", sheet=sheet, entry_id=entry_id, fields={},
            source_paper_entry_id=paper_entry_id, curation_agent="intern_agent",
            curation_model="none", confidence=0.0, notes=f"LLM extraction failed: {e}",
        )
        return {"filled": [], "skipped_reason": "llm error"}

    if not isinstance(result, dict):
        result = {}

    filled_fields = {
        field: result[field].strip()
        for field in missing_fields
        if isinstance(result.get(field), str) and result[field].strip()
    }

    if not filled_fields:
        return {"filled": [], "skipped_reason": "llm found nothing supported by abstract"}

    filled_fields["REVIEW_STATUS"] = config.REVIEW_STATUS_AUTO

    staging.append_candidate(
        action="update_field", sheet=sheet, entry_id=entry_id,
        fields=filled_fields, source_paper_entry_id=paper_entry_id,
        curation_agent="intern_agent", curation_model=model_used, confidence=None,
        notes=f"Filled from abstract (source: {abstract_source}).",
    )

    return {"filled": [k for k in filled_fields if k != "REVIEW_STATUS"], "skipped_reason": None}
