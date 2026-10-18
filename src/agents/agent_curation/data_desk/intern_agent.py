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

from ...common import config, staging
from ...common.llm_client import call_llm_json, LLMError, LLMExhaustedError
from ...common.paper_fetcher import fetch_paper, get_agent_text, is_probably_real_content

# Prompt + input changed 2026-10-11 (Marta): the details (panel size, markers, sample counts) live in the data-availability / methods /
# sample-preparation parts, not the abstract, so the agent now reads abstract + data-availability + methods when a real full text exists
# (capped), falling back to the abstract alone otherwise. Benchmarked in benchmarks/methods_desk test5 (context/LOCAL_LLM_BENCHMARK.md).
MAX_FULLTEXT_CHARS = 40000
_NO_FULLTEXT_SOURCES = ("biorxiv_fulltext", "none", None)    # never default to bioRxiv when the journal text is closed

FIELD_DESCRIPTIONS = {
    "organism": "the species, one lowercase word or phrase: 'human', 'mouse', 'non-human primate'... Never a sample ID or a "
                "sentence. If the sample is clearly from patients/human tissue say 'human'.",
    "tissue": "the organ or tissue sampled, lowercase, e.g. 'breast', 'pancreas', 'lung', 'colorectal', 'kidney', 'skin', "
              "'tonsil', 'lymph node', 'bone marrow', 'brain'. Several organs: separate with '; '. This is the ORGAN, never the "
              "disease or the preparation ('tumor', 'FFPE tissue' are NOT tissues - name the organ the tumor is in).",
    "disease": "the condition studied. If it is a cancer write 'cancer' followed, after '; ', by the specific type when the text "
               "gives one (e.g. 'cancer; triple-negative breast cancer'). Otherwise prefer one of: 'healthy', 'diabetes type 1', "
               "'inflammatory bowel disease', 'tuberculosis', 'pulmonary arterial hypertension', 'COVID-19', 'wound healing', or "
               "the disease name exactly as the text gives it. 'healthy' ONLY if the text explicitly describes healthy/normal/"
               "control tissue. If a cohort mixes healthy and diseased samples, list both ('healthy; cancer').",
    "spatial_data_method": "the spatial-proteomics platform in its usual short name: 'IMC', 'CODEX', 'PhenoCycler', 'MIBI-TOF', "
                           "'IF', 'mIF', 't-CyCIF', '4i', 'MALDI-MSI', 'LC-MS'... Use the platform's own name, not a generic word "
                           "like 'microscopy', unless the text gives nothing more specific. Several platforms: '; '. For a "
                           "transcriptomics dataset, the platform name (e.g. 'Xenium', 'MERFISH').",
    "N samples": "the number of samples/sections profiled, as a plain number",
    "N markers": "a plain integer: the number of protein markers/antibodies/channels in the panel, ONLY if the text states it "
                 "(e.g. '35 markers', '40-plex', 'a 52-antibody panel'). Do NOT count names yourself from a partial list.",
    "N patients": "the number of patients/donors/individuals the data comes from, a plain integer, ONLY if the text states it.",
    "N images": "the number of images / tissue sections / regions of interest (ROIs) / fields of view, a plain integer, ONLY if the "
                "text states it. If it gives only a number of samples or slides, use that number.",
    "Marker": "protein marker names that the text explicitly lists, separated by '; ', in their usual short form "
              "('CD3; CD8; PD-1; Pan-Cytokeratin'). Leave empty if the text only says how many, or lists just a few examples "
              "of a larger panel.",
    "N genes": "the number of genes profiled (transcriptomics only), as a plain number",
    "Genes": "the list of gene names, semicolon-separated, ONLY if a small targeted panel is named",
}


def _build_system_prompt(fields_needed: list) -> str:
    field_lines = "\n".join(
        f'  "{f}": "{FIELD_DESCRIPTIONS.get(f, f)}"' for f in fields_needed
    )
    return f"""You are a careful research assistant filling in the metadata of a spatial-omics dataset from the text of the \
paper that describes it (its abstract and, when given, its data-availability and methods sections - panel sizes, marker lists and sample \
numbers are usually in the sample-preparation / staining / data-acquisition parts). For EACH field below, fill it in only if the text explicitly supports \
it; if the text does not clearly state a field, return an empty string "". Never guess, never infer from general knowledge of the technique, never invent \
numbers. An empty field is a correct answer; a wrong value is not.

Write values in the compact style of a curated database (short, lowercase organ names, platform acronyms, plain integers), as defined \
for each field. Return ONLY a JSON object, nothing else, no markdown fences, with exactly these keys:
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


def _get_input_text(db, paper_entry_id: str, paper_doi: str) -> tuple:
    """Returns (user_prompt_text, source_label): abstract + data-availability + methods sections when the paper has a real, non-bioRxiv
    full text (capped at MAX_FULLTEXT_CHARS), otherwise the abstract alone exactly as before ("" if neither exists)."""
    abstract, abstract_source = _get_abstract(db, paper_entry_id, paper_doi)
    doi = paper_doi
    if not doi and paper_entry_id:
        row = db.methods.loc[db.methods["entry_id"] == paper_entry_id]
        doi = _clean(row.iloc[0].get("DOI", "")) if not row.empty else ""
    if doi:
        fetched = fetch_paper(doi)
        if fetched.get("source") not in _NO_FULLTEXT_SOURCES:
            dt, ds = get_agent_text(fetched, "data_availability")
            mt, ms = get_agent_text(fetched, "methods")
            has_d = ds == "section:data_availability" and is_probably_real_content(dt)
            has_m = ms == "section:methods" and is_probably_real_content(mt)
            if has_d or has_m:
                text = ((f"ABSTRACT:\n{abstract}\n\n" if abstract else "") +
                        (f"DATA AVAILABILITY SECTION:\n{dt}\n\n" if has_d else "") +
                        (f"METHODS SECTION:\n{mt}" if has_m else ""))[:MAX_FULLTEXT_CHARS]
                return f"Paper text (abstract, data-availability and methods sections):\n\n{text}", f"fulltext:{fetched['source']}"
    return (f"Abstract:\n\n{abstract}" if abstract else ""), abstract_source


def _sheet_for_entry(entry_id: str) -> str:
    """As of the method_pub/AP_pub/data schema there's only one dataset
    sheet - kept as a function (rather than inlining the literal at both
    call sites) since staging.append_candidate still requires a target
    sheet name and build_data_pool()'s worklist doesn't carry that through."""
    return "data"


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

    user_text, abstract_source = _get_input_text(db, paper_entry_id, paper_doi)
    if not user_text:
        staging.append_candidate(
            action="update_field", sheet=sheet, entry_id=entry_id, fields={},
            source_paper_entry_id=paper_entry_id, curation_agent="intern_agent",
            curation_model="none", confidence=0.0,
            notes="No abstract or usable full text could be found for the associated paper.",
        )
        return {"filled": [], "skipped_reason": "no abstract"}

    try:
        result, model_used = call_llm_json(
            _build_system_prompt(missing_fields), user_text,
            **config.llm_kwargs("dataset_annotation")    # local mode: this stage's model; api mode: default chain, as before
        )
    except LLMError as e:
        staging.append_candidate(
            action="update_field", sheet=sheet, entry_id=entry_id, fields={},
            source_paper_entry_id=paper_entry_id, curation_agent="intern_agent",
            curation_model="none", confidence=0.0, notes=f"LLM extraction failed: {e}",
        )
        return {"filled": [], "skipped_reason": "llm error",
                "llm_exhausted": isinstance(e, LLMExhaustedError)}

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
        notes=f"Filled from {'full text' if str(abstract_source).startswith('fulltext') else 'abstract'} (source: {abstract_source}).",
    )

    return {"filled": [k for k in filled_fields if k != "REVIEW_STATUS"], "skipped_reason": None}
