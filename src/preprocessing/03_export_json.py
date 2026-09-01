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
    return "" if s.lower() in ("nan", "na", "none") else s

# ── Parse semicolon-separated ID lists ───────────────────────────────────────
def parse_id_list(val) -> list:
    s = clean(val)
    if not s:
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]

# ── Export methods ────────────────────────────────────────────────────────────
def export_methods(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
        records.append({
            "id":                clean(row.get("entry_id")),
            "doi":               clean(row.get("DOI")),
            "name":              clean(row.get("name")),
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
            "is_placeholder":    str(row.get("is_placeholder", "")).lower() == "true",
            "date_added":        clean(row.get("Date(added_to_dataset)")),
            # metadata from script 02
            "title":             clean(row.get("title")),
            "first_author":      clean(row.get("first_author")),
            "authors":           clean(row.get("authors")),
            "year":              clean(row.get("year")),
            "journal":           clean(row.get("journal")),
            "citations":         clean(row.get("citations")),
            "abstract":          clean(row.get("abstract")),
            "publication_type":  clean(row.get("publication_type")),
            # relationships (method_pub only - empty for AP_pub rows)
            "data_ids":          parse_id_list(row.get("DataID (data_used_in_the_paper)")),
            "comparison_ids":    parse_id_list(row.get("Method_comparison_P_ENTRY_ID")),
            # AP_pub-only fields (empty for method_pub rows). Note "title"
            # above already picks up AP_pub's own manually-curated title
            # column via the same row.get("title") - no separate handling
            # needed there.
            "associated_data":       clean(row.get("Associated data")),
            "spatial_data_type_ap":  clean(row.get("type of spatial data")),
            "animal":                clean(row.get("animal")),
            "tissue_disease":        clean(row.get("tissue/disease")),
        })
    return records

# ── Export datasets ───────────────────────────────────────────────────────────
def export_datasets(df: pd.DataFrame) -> list:
    records = []
    for _, row in df.iterrows():
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
            "spatial_data_category": clean(row.get("spatial_data_category")),
            "spatial_data_method":  clean(row.get("spatial_data_method")),
            "organism":             clean(row.get("organism")),
            "tissue":               clean(row.get("tissue")),
            "disease":              clean(row.get("disease")),
            "n_samples":            clean(row.get("N samples")),
            "n_patients":           clean(row.get("N patients")),
            "clinical_data":        clean(row.get("clinical data")),
            # SP fields
            "n_markers":            clean(row.get("N markers")),
            "markers":              clean(row.get("Marker")),
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
    # "auto_confirmed" (new 2026-09-01 status) means auto-generated then
    # manually confirmed - counts as curated same as "manual".
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

    # Write JSON files
    (OUT_DIR / "methods.json").write_text(
        json.dumps(methods,  indent=2, ensure_ascii=False))
    (OUT_DIR / "datasets.json").write_text(
        json.dumps(datasets, indent=2, ensure_ascii=False))
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats,    indent=2, ensure_ascii=False))

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