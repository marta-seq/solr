"""
03_export_json.py
Converts processed CSVs to JSON files for the frontend.
Reads the most recent methods_metadata and datasets CSVs,
and writes to docs/data/.

Usage:
    python src/preprocessing/03_export_json.py
"""

import json
import re
from pathlib import Path

import pandas as pd

import tissue_disease_maps as tdm
import platform_maps as pfm
import category_maps as cm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR       = ROOT / "docs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Find most recent files ────────────────────────────────────────────────────
def find_latest(pattern: str) -> Path:
    # Sort key normalizes "-" to "_" - the date suffix isn't consistently
    # hyphens or underscores (depends on the source curated .xlsx filename
    # that day), and plain string sort puts "-" before "_" in ASCII, which
    # would silently pick an OLDER file as "latest" whenever both formats
    # are present (caught 2026-09-01: methods_2026-08-29.csv and
    # methods_2026_07_07.csv existed at the same time).
    files = sorted(PROCESSED_DIR.glob(pattern), key=lambda p: p.stem.replace("-", "_"))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {PROCESSED_DIR}")
    return files[-1]

# ── Clean a value for JSON ────────────────────────────────────────────────────
def clean(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "na", "none"):
        return ""
    # Columns with any blank cell (e.g. "year") get read as float64 by
    # pandas even though every real value is a whole number, so it lands
    # here as "2022.0" - strip the trailing ".0" so it renders as "2022".
    if re.fullmatch(r"-?\d+\.0", s):
        s = s[:-2]
    return s

# ── Source type (bioRxiv / arXiv / medRxiv / peer-reviewed / preprint) ──────
# Single consolidated field - the old separate `source_type_manual` column
# (method_pub-only "arxiv/bioarxiv/peer reviewed") was merged into
# `publication_type` directly in the master Excel 2026-09-18 (manual value
# migrated in, column dropped) since having two columns for the same fact
# was redundant and let them drift out of sync (e.g. M_SE_25: manual said
# "peer-reviewed", DOI was 10.48550/arXiv.1901.03353 - manual was just
# wrong there). Priority order, most to least authoritative:
#   1. DOI prefix "10.1101/" or "10.64898/" (bioRxiv/medRxiv's registered
#      Crossref prefixes) or journal text containing "biorxiv"/"medrxiv" -
#      deterministic, derived from the DOI itself so it can't drift stale.
#      Crossref returns posted-content DOIs (how current bioRxiv papers
#      register) with no container-title at all, so the journal-text check
#      alone doesn't catch everything - verified 2026-09-15, every
#      then-mislabeled row had an empty `journal` and a 10.1101/... DOI.
#   2. DOI text containing "arxiv" - same reasoning; must come before the
#      generic publication_type check below, since Crossref/manual curation
#      can both mislabel an arXiv DOI as "peer-reviewed" or "preprint".
#   3. publication_type - auto-fetched via Crossref in 02_fetch_metadata.py,
#      or a manually-curated value now living in the same column. Covers
#      AP_pub too (which never had the old manual column).
# Returns "" (unknown) only when none of the above have anything.
BIORXIV_DOI_PREFIXES = ("10.1101/", "10.64898/")

def compute_source_type(row) -> str:
    doi     = clean(row.get("DOI")).lower()
    journal = clean(row.get("journal")).lower()
    if "medrxiv" in journal:
        return "medRxiv"
    if "biorxiv" in journal or any(p in doi for p in BIORXIV_DOI_PREFIXES):
        return "bioRxiv"
    if "arxiv" in doi:
        return "arXiv"
    pub_type = clean(row.get("publication_type"))
    if pub_type in ("peer-reviewed", "preprint"):
        return pub_type
    return ""

# ── Parse semicolon-separated ID lists ───────────────────────────────────────
def parse_id_list(val) -> list:
    s = clean(val)
    if not s:
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]

# ── Marker name canonicalization (spatial_proteomics datasets only) ─────────
# Raw `Marker` text is free-form across papers/curators - the same antibody
# target often ends up spelled differently (case, hyphens/spaces, or a Greek
# letter vs its Latin-alphabet stand-in for the exact same symbol, e.g.
# "a-SMA" / "aSMA" / "αSMA"). Canonicalizing lets the datasets-tab marker
# filter treat these as one entry instead of fragmenting into near-duplicates.
#
# Deliberately does NOT fold in differences that are a naming-convention
# choice rather than a rendering difference - e.g. "CD8" vs "CD8A" (informal
# name vs the specific gene symbol) is left as two separate entries, even
# though nothing in the real corpus ever uses "CD8B" (the only other real
# subunit) - reviewed against the actual data 2026-09-02, see CLAUDE.md.
_GREEK_TO_LATIN = {
    "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E",
    "Κ": "K", "Λ": "L", "Μ": "M", "Ν": "N", "Π": "P",
    "Ρ": "R", "Σ": "S", "Τ": "T", "Φ": "F", "Χ": "CH",
    "Ψ": "PS", "Ω": "O",
}

def _marker_canonical_key(tok: str) -> str:
    key = re.sub(r"[-_\s]", "", tok.upper())
    return "".join(_GREEK_TO_LATIN.get(ch, ch) for ch in key)

def _build_marker_display_map(df: pd.DataFrame) -> dict:
    """canonical key -> the most common raw spelling for it in the corpus
    (ties broken alphabetically), so the filter shows a real spelling
    someone actually used rather than an all-caps canonical key."""
    counts = {}
    for _, row in df.iterrows():
        if "proteomics" not in clean(row.get("spatial_data_category")).lower():
            continue
        val = clean(row.get("Marker"))
        if not val:
            continue
        for tok in re.split(r"[;,]", val):
            tok = tok.strip()
            if not tok:
                continue
            key = _marker_canonical_key(tok)
            counts.setdefault(key, {}).setdefault(tok, 0)
            counts[key][tok] += 1
    return {
        key: sorted(spellings.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for key, spellings in counts.items()
    }

def parse_marker_list(val, spatial_category, display_map: dict) -> list:
    if "proteomics" not in clean(spatial_category).lower():
        return []
    s = clean(val)
    if not s:
        return []
    seen = []
    for tok in re.split(r"[;,]", s):
        tok = tok.strip()
        if not tok:
            continue
        name = display_map.get(_marker_canonical_key(tok), tok)
        if name not in seen:
            seen.append(name)
    return seen

# ── Tissue/disease canonicalization (tissue_disease_maps.py) ────────────────
def _normalize_list_or_warn(raw: str, mapping: dict, field_name: str, entry_id: str) -> list:
    """List version of 01_parse_excel.py's _normalize_or_warn - a mapped
    value of None means deliberately left unset (needs_review), not a
    guess. An unmapped raw value passes through unchanged with a loud
    warning instead of being silently guessed at."""
    if raw in mapping:
        mapped = mapping[raw]
        return [] if mapped is None else [x.strip() for x in re.split(r"[;,]", mapped) if x.strip()]
    print(f"  WARNING [datasets]: unmapped {field_name} value {raw!r} "
          f"(entry_id: {entry_id}) - not in the {field_name} map, passing "
          f"through unchanged. Add it to the mapping once reviewed.")
    return [raw]

def parse_tissue_list(val, entry_id: str) -> list:
    s = clean(val)
    if not s:
        return []
    return _normalize_list_or_warn(s, tdm.TISSUE_MAP, "tissue", entry_id)

def parse_platform_list(val, entry_id: str) -> list:
    s = clean(val)
    if not s:
        return []
    return _normalize_list_or_warn(s, pfm.PLATFORM_MAP, "platform", entry_id)

def parse_ap_spatial_list(val, entry_id: str) -> list:
    s = clean(val)
    if not s:
        return []
    return _normalize_list_or_warn(s, cm.SPATIAL_DATA_CATEGORY_MAP_AP_PUB, "spatial_data_type_ap", entry_id)

def parse_disease_lists(disease_val, tissue_val, entry_id: str) -> tuple:
    """Returns (disease_list, disease_specifics_list). Also applies
    CROSS_COLUMN_FIXES for the rare row where disease info leaked into the
    `tissue` cell while `disease` itself was left blank - see
    tissue_disease_maps.py docstring."""
    s = clean(disease_val)
    disease_vals, specifics_vals = [], []
    if s:
        disease_vals = _normalize_list_or_warn(s, tdm.DISEASE_MAP, "disease", entry_id)
        specifics_raw = tdm.DISEASE_SPECIFICS_MAP.get(s)
        if specifics_raw:
            specifics_vals = [x.strip() for x in re.split(r"[;,]", specifics_raw) if x.strip()]

    fix = tdm.CROSS_COLUMN_FIXES.get(clean(tissue_val))
    if fix:
        for d in re.split(r"[;,]", fix.get("disease", "")):
            d = d.strip()
            if d and d not in disease_vals:
                disease_vals.append(d)
        for sp in re.split(r"[;,]", fix.get("disease_specifics", "")):
            sp = sp.strip()
            if sp and sp not in specifics_vals:
                specifics_vals.append(sp)
    return disease_vals, specifics_vals

# ── Export methods ────────────────────────────────────────────────────────────
def export_methods(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        records.append({
            "id":                clean(row.get("entry_id")),
            "doi":               clean(row.get("DOI")),
            "name":              clean(row.get("name")),
            "source_type":       compute_source_type(row),
            "category":          clean(row.get("category")),
            # "method" or "application" - added by 01_parse_excel.py when it
            # combines method_pub + AP_pub, so the frontend doesn't need to
            # re-derive this from free-text category matching.
            "paper_type":        clean(row.get("paper_type")),
            "pipeline_category": clean(row.get("pipeline_category")),
            "spatial_data_category": clean(row.get("spatial_data_category")),
            # Since the 2026-09-01 category cleanup, `category` and
            # `pipeline_category` are canonical and ';'-separated when a
            # paper genuinely has more than one tag (most have exactly one).
            # These list versions are additive - existing frontend code that
            # reads the scalar fields above keeps working unchanged; new code
            # (e.g. a future per-category pie on the graph) should use these.
            "categories":            parse_id_list(row.get("category")),
            "pipeline_categories":   parse_id_list(row.get("pipeline_category")),
            "review_status":     clean(row.get("REVIEW_STATUS")),
            # New 2026-09-18, split out of review_status - see config.py's
            # REVIEW_STATUS section for the reasoning (how a row entered the
            # DB vs. whether/how much it's been automatically/manually
            # reviewed are now two independent things).
            "addition_method":   clean(row.get("addition_method")),
            "is_placeholder":    str(row.get("is_placeholder", "")).lower() == "true",
            "date_added":        clean(row.get("Date(added_to_dataset)")),
            # metadata from script 02
            "title":             clean(row.get("title")),
            "authors":           clean(row.get("authors")),
            "year":              clean(row.get("year")),
            "journal":           clean(row.get("journal")),
            "citations":         clean(row.get("citations")),
            "abstract":          clean(row.get("abstract")),
            "publication_type":  clean(row.get("publication_type")),
            # New 2026-09-18: PubMed MeSH headings, fetched from the same
            # efetch call as abstract (see 02_fetch_metadata.py). Only
            # available for papers PubMed has already curator-indexed -
            # typically has a lag of weeks-to-months after publication, so
            # coverage is lower than abstract's, not a bug.
            "keywords":          parse_id_list(row.get("keywords")),
            # relationships (method_pub only - empty for AP_pub rows)
            "data_ids":          parse_id_list(row.get("DataID (data_used_in_the_paper)")),
            "comparison_ids":    parse_id_list(row.get("Method_comparison_P_ENTRY_ID")),
            # AP_pub-only fields (empty for method_pub rows). Note "title"
            # above already picks up AP_pub's own manually-curated title
            # column via the same row.get("title") - no separate handling
            # needed there.
            "associated_data":       clean(row.get("Associated data")),
            "associated_data_ids":   parse_id_list(row.get("Associated data")),
            "spatial_data_type_ap":  clean(row.get("type of spatial data")),
            # Canonicalized list version (see category_maps.py's
            # SPATIAL_DATA_CATEGORY_MAP_AP_PUB) - AP_pub never populates the
            # shared spatial_data_category field (0/154 rows), so this is
            # the only source of SP/ST/other modality info for AP_pub.
            "spatial_modality_ap":   parse_ap_spatial_list(row.get("type of spatial data"), row.get("entry_id")),
            "animal":                clean(row.get("animal")),
            "tissue_disease":        clean(row.get("tissue/disease")),
        })
    return records

# ── Export datasets ───────────────────────────────────────────────────────────
def export_datasets(df: pd.DataFrame) -> list:
    marker_display_map = _build_marker_display_map(df)
    records = []
    for _, row in df.iterrows():
        disease_list, disease_specifics_list = parse_disease_lists(
            row.get("disease"), row.get("tissue"), row.get("entry_id"))
        records.append({
            "id":                   clean(row.get("entry_id")),
            "data_doi":             clean(row.get("data_DOI")),
            "accession_number":     clean(row.get("data_accession_number")),
            "access_link":          clean(row.get("data_acess_link")),
            "paper_doi":            clean(row.get("paper_DOI")),
            "paper_entry_id":       clean(row.get("paper_ENTRY_ID")),
            "internal_name":        clean(row.get("paper_internal_name")),
            "year":                 clean(row.get("year")),
            "review_status":        clean(row.get("REVIEW_STATUS")),
            "addition_method":      clean(row.get("addition_method")),
            "spatial_data_category": clean(row.get("spatial_data_category")),
            "spatial_data_method":  clean(row.get("spatial_data_method")),
            # Canonicalized list version (see platform_maps.py), additive
            # like tissue_list/disease_list above - raw scalar keeps working
            # unchanged. A dataset genuinely has >1 platform only for the
            # rare comparison row (e.g. "CODEX; CyCIF; Vectra; ...").
            "platform_list":        parse_platform_list(row.get("spatial_data_method"), row.get("entry_id")),
            "organism":             clean(row.get("organism")),
            "tissue":               clean(row.get("tissue")),
            "disease":              clean(row.get("disease")),
            # Canonicalized via tissue_disease_maps.py, additive like
            # markers_list above - raw scalars keep working unchanged.
            # disease_list/disease_specifics_list are both derived from the
            # same raw `disease` cell (there's no separate raw specifics
            # column) - see that module's docstring for the design and the
            # items still flagged for review.
            "tissue_list":          parse_tissue_list(row.get("tissue"), row.get("entry_id")),
            "disease_list":         disease_list,
            "disease_specifics_list": disease_specifics_list,
            # Was "n_samples": clean(row.get("N samples")) - "N samples" isn't
            # a real column (checked 2026-09-16), so it was silently always
            # empty. The actual raw column is "N images"; renamed the output
            # key to match so it's not misleading. See also `size_type`
            # (demo/cohort/segmentation dataset), which qualifies what the
            # image count means but isn't surfaced separately yet.
            "n_images":             clean(row.get("N images")),
            "n_patients":           clean(row.get("N patients")),
            "clinical_data":        clean(row.get("clinical data")),
            # SP fields
            "n_markers":            clean(row.get("N markers")),
            "markers":              clean(row.get("Marker")),
            # Canonicalized list version, additive like categories/
            # pipeline_categories above - existing code reading "markers"
            # (the raw scalar) keeps working unchanged. Empty for non-SP
            # rows (ST gene panels are out of scope - see CLAUDE.md).
            "markers_list":         parse_marker_list(
                                        row.get("Marker"),
                                        row.get("spatial_data_category"),
                                        marker_display_map),
            # ST fields
            "n_genes":              clean(row.get("N genes")),
            "genes":                clean(row.get("Genes")),
            "resolution":           clean(row.get("Resolution")),
            "notes":                clean(row.get("Notes")),
        })
    return records

# ── Compute statistics ────────────────────────────────────────────────────────
def compute_stats(methods: list, datasets: list) -> dict:
    total_papers     = len(methods)
    placeholders     = sum(1 for m in methods if m["is_placeholder"])
    # "auto_confirmed" (the 2026-09-01 status meaning auto-generated then
    # manually confirmed) was retired 2026-09-18 - those rows were migrated
    # to plain "manual" (addition_method now separately records that they
    # originated automatically). Kept in this check anyway, harmlessly, in
    # case an un-migrated older export ever gets read here.
    curated          = sum(1 for m in methods if m["review_status"] in ("manual", "auto_confirmed"))
    # Was substring-matching the free-text `category` field ("computational"
    # / starts with "application"); switched to `paper_type`, which is set
    # directly from which sheet the row came from and no longer depends on
    # category wording that changed in the 2026-09-01 cleanup.
    comp_methods     = sum(1 for m in methods if m["paper_type"] == "method")
    applications     = sum(1 for m in methods if m["paper_type"] == "application")

    # Pipeline category counts. A method with more than one canonical
    # category (';'-separated) now contributes to each of its categories'
    # counts, rather than creating one combined "A; B" bucket.
    pipeline_counts = {}
    for m in methods:
        pcs = m["pipeline_categories"] or ["Other"]
        for pc in pcs:
            pipeline_counts[pc] = pipeline_counts.get(pc, 0) + 1

    # Dataset stats
    total_datasets = len(datasets)
    sp_datasets    = sum(1 for d in datasets if "proteomics" in d["spatial_data_category"].lower())
    st_datasets    = sum(1 for d in datasets if "transcriptomics" in d["spatial_data_category"].lower())

    # Organisms
    organism_counts = {}
    for d in datasets:
        org = d["organism"] or "unknown"
        organism_counts[org] = organism_counts.get(org, 0) + 1

    # Disease
    disease_counts = {}
    for d in datasets:
        dis = d["disease"] or "unknown"
        disease_counts[dis] = disease_counts.get(dis, 0) + 1

    # Marker counts (canonicalized, spatial_proteomics only - see
    # markers_list above), for building a filter sorted by real usage
    # rather than alphabetically.
    marker_counts = {}
    for d in datasets:
        for m in d["markers_list"]:
            marker_counts[m] = marker_counts.get(m, 0) + 1

    # Canonicalized tissue/disease counts (see tissue_disease_maps.py) -
    # additive alongside the raw-string organism_counts/disease_counts
    # above, same reasoning as marker_counts: a clean small vocabulary is
    # what a real filter UI needs, not every distinct free-text spelling.
    tissue_counts = {}
    for d in datasets:
        for t in d["tissue_list"]:
            tissue_counts[t] = tissue_counts.get(t, 0) + 1

    platform_counts = {}
    for d in datasets:
        for p in d["platform_list"]:
            platform_counts[p] = platform_counts.get(p, 0) + 1

    disease_clean_counts = {}
    for d in datasets:
        for dis in d["disease_list"]:
            disease_clean_counts[dis] = disease_clean_counts.get(dis, 0) + 1

    disease_specifics_counts = {}
    for d in datasets:
        for sp in d["disease_specifics_list"]:
            disease_specifics_counts[sp] = disease_specifics_counts.get(sp, 0) + 1

    return {
        "total_papers":      total_papers,
        "placeholders":      placeholders,
        "curated":           curated,
        "comp_methods":      comp_methods,
        "applications":      applications,
        "pipeline_counts":   pipeline_counts,
        "total_datasets":    total_datasets,
        "sp_datasets":       sp_datasets,
        "st_datasets":       st_datasets,
        "organism_counts":   organism_counts,
        "disease_counts":    disease_counts,
        "marker_counts":     marker_counts,
        "tissue_counts":     tissue_counts,
        "platform_counts":   platform_counts,
        "disease_clean_counts":     disease_clean_counts,
        "disease_specifics_counts": disease_specifics_counts,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Find latest files — prefer metadata version, fall back to plain methods
    try:
        methods_file = find_latest("methods_metadata_*.csv")
    except FileNotFoundError:
        methods_file = find_latest("methods_*.csv")
    datasets_file = find_latest("datasets_*.csv")

    print(f"Methods:  {methods_file.name}")
    print(f"Datasets: {datasets_file.name}")

    methods_df  = pd.read_csv(methods_file,  dtype=str)
    datasets_df = pd.read_csv(datasets_file, dtype=str)

    methods  = export_methods(methods_df)
    datasets = export_datasets(datasets_df)
    stats    = compute_stats(methods, datasets)

    # Write JSON files. Explicit UTF-8 - write_text() otherwise defaults to
    # the platform locale encoding (cp1252 on Windows), which breaks on any
    # non-ASCII character (unicode hyphens in titles, Greek letters in
    # markers_list, etc.) - never surfaced before because this has only
    # ever been run on gaia/Linux, where the locale default is UTF-8.
    # newline="\n" pins LF regardless of platform - write_text() otherwise
    # applies platform newline translation (CRLF on Windows), which would
    # make every line look changed in git despite identical content.
    (OUT_DIR / "methods.json").write_text(
        json.dumps(methods,  indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    (OUT_DIR / "datasets.json").write_text(
        json.dumps(datasets, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats,    indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    print(f"\nOutputs written to {OUT_DIR}:")
    print(f"  methods.json  ({len(methods)} entries)")
    print(f"  datasets.json ({len(datasets)} entries)")
    print(f"  stats.json")
    print(f"\nStats preview:")
    print(f"  Total papers:   {stats['total_papers']}")
    print(f"  Curated:        {stats['curated']}")
    print(f"  Total datasets: {stats['total_datasets']}")
    print(f"  SP datasets:    {stats['sp_datasets']}")
    print(f"  ST datasets:    {stats['st_datasets']}")

if __name__ == "__main__":
    main()