"""
category_maps.py
Canonical taxonomies and raw-value -> canonical mappings for the four messy
free-text fields in the curated Excel: `pipeline_category` (method_pub),
`category` (method_pub + AP_pub), `spatial_data_category` (method_pub + data),
and `REVIEW_STATUS` (all sheets).

Built from every distinct raw value actually present in
datasets_curated_2026-08-29.xlsx (267 method_pub / 167 AP_pub / 201 data
rows), reviewed and decided together with Marta on 2026-09-01. Each mapping
below is a literal whole-cell lookup (not fuzzy/substring matching) so it's
auditable and so a genuinely new raw value added later fails loudly instead
of being silently guessed at - see `_normalize_or_warn` in 01_parse_excel.py.

Multi-label cells: a raw value that already contains multiple ';'-separated
tags (e.g. "Cell segmentation - Imaging & Transcript based") maps to a
canonical string that is also ';'-joined. Most papers end up with exactly
one canonical category; some genuinely have more than one - that's expected
and is what lets the frontend eventually render a per-category pie rather
than a single color (not built yet, this just makes the data ready for it).

A mapped value of None means: deliberately left unset for now (Marta's
call, not a guess) - downstream code should treat this the same as a blank
cell, not as an error.
"""

# ── Canonical pipeline_category taxonomy (method_pub) ────────────────────────
# The 13 pre-existing categories (already used as keys in docs/js/state.js
# CATS) plus categories the real data clearly needed. If you rename anything
# here, also update CATS in docs/js/state.js so the new name gets a real
# color/position instead of falling into the gray "unassigned" default.
PIPELINE_CATEGORY_TAXONOMY = [
    "Preprocessing",
    "Cell segmentation - Imaging based",
    "Cell segmentation - Transcript based",
    "Cell segmentation - unspecified",
    "Phenotyping",
    "Niche/Neighborhood/domain analysis",
    "Cell-Cell-Communication",
    "Spatial Variable Genes",
    "Cell type Deconvolution",
    "Clustering",
    "Survival prediction",
    "Data alignment / integration / imputation",
    "Integration of modalities",
    "Label separation/pattern extraction - ML",
    "Foundation model",
    "Computer vision (H&E)",
    "Virtual staining (proteomics)",
    "Virtual staining (transcriptomics)",
    "Subcellular localization",
    "General Framework",
    "spatiotemporal dynamics",
    "Immune infiltration scoring",
    "Analysis/workflow optimization",
    "Other",
]

# raw pipeline_category cell (method_pub) -> canonical cell, or None to leave
# unset for now. Decisions from the 2026-09-01 review:
#   - "Cell type classification/annotation" folded into Phenotyping (same
#     category, not worth a separate bucket).
#   - "factor analysis" variants folded into the general ML bucket.
#   - "DL" alone -> Label separation/pattern extraction - ML (Marta's own
#     best guess, low confidence, revisit if it recurs).
#   - "Quality control" folded into Preprocessing (single instance, not
#     enough volume yet for its own stage).
#   - Virtual staining split by modality (checked the actual rows: the bare
#     "virtual staining" entry is MISO, spatial_data_category = spatial
#     transcriptomics, so it's the transcriptomics bucket).
#   - Two rows that contained a full sentence / a paper title instead of an
#     actual category -> left as None, not guessed.
#   - "M_CCC_59.1 M_AUTO_14" is a duplicate-ID artifact (CellPhoneDB entry
#     merge bug noted in the project handoff), not a category value at all -
#     deliberately NOT mapped here; it flows through unchanged until the
#     ID-consolidation tool merges the two entries.
PIPELINE_CATEGORY_MAP = {
    "Niche/Neighborhood/domain analysis": "Niche/Neighborhood/domain analysis",
    "Cell segmentation - Imaging based": "Cell segmentation - Imaging based",
    "Cell-Cell-Communication": "Cell-Cell-Communication",
    "Cell segmentation - Transcript based": "Cell segmentation - Transcript based",
    "Preprocessing": "Preprocessing",
    "Spatial Variable Genes": "Spatial Variable Genes",
    "General Framework": "General Framework",
    "Cell type Deconvolution": "Cell type Deconvolution",
    "Cell segmentation": "Cell segmentation - unspecified",
    "data alignment / gene imputation/ integration methods": "Data alignment / integration / imputation",
    "Label separation/pattern extraction - ML ; foundation model": "Label separation/pattern extraction - ML; Foundation model",
    "Phenotyping": "Phenotyping",
    "survival prediction": "Survival prediction",
    "clustering": "Clustering",
    "Foundational model": "Foundation model",
    "Label separation/pattern extraction - ML": "Label separation/pattern extraction - ML",
    "Integration of modalities; Other": "Integration of modalities; Other",
    "Computer Vision H&E": "Computer vision (H&E)",
    "Cell type classification/ annotation": "Phenotyping",
    "virtual protein staining": "Virtual staining (proteomics)",
    "Subcellular localization": "Subcellular localization",
    "segmentation general": "Cell segmentation - unspecified",
    "General": "General Framework",
    "Cell segmentation - Imaging & Transcript based": "Cell segmentation - Imaging based; Cell segmentation - Transcript based",
    "Cell segmentation - Transcript based; cell type annotation": "Cell segmentation - Transcript based; Phenotyping",
    "Label separation/pattern extraction - ML; Application": "Label separation/pattern extraction - ML",
    "Niche/Neighborhood/domain analysis;": "Niche/Neighborhood/domain analysis",
    "Niche/Neighborhood/domain analysis ? dL?": "Niche/Neighborhood/domain analysis",
    "Other; spatiotemporal dynamics": "Other; spatiotemporal dynamics",
    "General Framework (not spatial": "General Framework",
    "DL": "Label separation/pattern extraction - ML",
    "Label separation/pattern extraction - ML ; foundation model; cell type classification; Cell segmentation - Imaging based; Cell type classification": "Label separation/pattern extraction - ML; Foundation model; Phenotyping; Cell segmentation - Imaging based",
    "Label separation/pattern extraction - factor analysis ; spatiotemporal dynamics": "Label separation/pattern extraction - ML; spatiotemporal dynamics",
    "Label separation/pattern extraction - factor analysis": "Label separation/pattern extraction - ML",
    "Quality control": "Preprocessing",
    "spatially resolved H&E annotation": "Computer vision (H&E)",
    "spatiotemporal dynamics; other": "spatiotemporal dynamics; Other",
    "General framework - MCP": "General Framework",
    "virtual staining": "Virtual staining (transcriptomics)",
    "xenium ell type annotation xenium": "Phenotyping",
    "optimize xenium analysis": "Analysis/workflow optimization",
    "subcellular": "Subcellular localization",
    "Spatial Variable Genes?": "Spatial Variable Genes",
    "immune infiltration scoring": "Immune infiltration scoring",
    "trajectory-centric framework that reconstructs continuous TME dynamics by integrating agent-based mathematical modeling and simulation with state space analysis": None,
    "Machine learning-based spatial characterization of tumor-immune microenvironment in the EORTC 10994/BIG 1-00 early breast cancer trial": None,
    "cell segmentation; Niche/Neighborhood/domain analysis; phenotyping?????": "Cell segmentation - unspecified; Niche/Neighborhood/domain analysis; Phenotyping",
    "Computer Vision": "Computer vision (H&E)",
    "virtual staining transcriptomics": "Virtual staining (transcriptomics)",
}

# ── category (paper contribution type) ───────────────────────────────────────
# method_pub and AP_pub use overlapping but not identical vocabularies, so
# they get separate maps. Double-tagging (e.g. "Application; computational
# analysis - method") is intentional per Marta - some AP_pub papers really
# are both.
#
# IMPORTANT (corrected 2026-09-01, per Marta): this field keeps its
# established wording as-is - "computational analysis - method" stays
# "computational analysis - method", NOT shortened to "Method". An earlier
# draft of this map renamed these to a shorter vocabulary without asking;
# that was wrong and is reverted below. This map now only fixes things that
# are objectively broken - typos, inconsistent delimiters/casing, or a
# literal "?" - never a wording change that wasn't explicitly agreed.
CATEGORY_MAP_METHOD_PUB = {
    "computational analysis - method": "computational analysis - method",
    "Placeholder - method": "Placeholder - method",
    "computational analysis - review": "computational analysis - review",
    "computational analysis - review; computational analysis - Benchmarking": "computational analysis - review; computational analysis - Benchmarking",
    "computational analysis - method; computational analysis - Benchmarking": "computational analysis - method; computational analysis - Benchmarking",
    "computational analysis - method; Technical Methods": "computational analysis - method; Technical Methods",
    "computational analysis - Benchmarking": "computational analysis - Benchmarking",
    "?": None,  # needs a real value - flagged, not guessed
    "computational analysis - Benchmarking; computational analysis - method": "computational analysis - Benchmarking; computational analysis - method",
}

# Rows still marked "Dataset" / "may be methods??" are unresolved on purpose
# (Marta: "for now just the category they have now or None, once we
# understand better we move them") - "Dataset" passes through unchanged
# rather than being force-fit into this taxonomy; "may be methods??" isn't a
# usable category string as-is, so it's left None instead of kept literally.
CATEGORY_MAP_AP_PUB = {
    "Application": "Application",
    "Technical Methods, Application": "Technical Methods; Application",
    "Technical Methods; Application": "Technical Methods; Application",
    "Application review, Tecnhical review; General omics review": "Application review; Technical review; General omics review",
    "Application review": "Application review",
    "Application; computational analysis - method": "Application; computational analysis - method",
    "Application; computational analysis - review; computational analysis - Benchmarking": "Application; computational analysis - review; computational analysis - Benchmarking",
    "application review, computational analysis - review": "Application review; computational analysis - review",  # casing + delimiter only
    "application review": "Application review",  # casing only
    "Technical Methods, Computational methods": "Technical Methods; Computational methods",  # delimiter only
    "Dataset": "Dataset",       # passthrough - unresolved, revisit later
    "may be methods??": None,   # unresolved - revisit later
}

# ── spatial_data_category ─────────────────────────────────────────────────────
# method_pub and data sheets use slightly different raw spellings but the
# same underlying meaning, so they share one canonical vocabulary. The three
# entries Marta asked to leave untouched ("microscopy", "check", "protocol
# of MRNA + IMC") pass through as literal strings - not reinterpreted.
SPATIAL_DATA_CATEGORY_MAP_METHOD_PUB = {
    "spatial transcriptomics": "spatial_transcriptomics",
    "not spatial omics": "not_spatial_omics",
    "spatial proteomics": "spatial_proteomics",
    "histopathological": "histopathology",
    # Deliberately NOT mapped: this is the CellPhoneDB duplicate-ID row
    # (P_ENTRY_ID == "M_CCC_59.1 M_AUTO_14"), a merge artifact, not a real
    # category value - left to flow through untouched until the two
    # entries are consolidated.
    "M_CCC_59.1 M_AUTO_14": "M_CCC_59.1 M_AUTO_14",
}

SPATIAL_DATA_CATEGORY_MAP_DATA = {
    "spatial_proteomics": "spatial_proteomics",
    "spatial_transcriptomics": "spatial_transcriptomics",
    "spatial transcriptomics": "spatial_transcriptomics",
    "spatial proteomics": "spatial_proteomics",
    "microscopy": "microscopy",  # passthrough - unresolved, revisit later
    "check": "check",            # passthrough - unresolved, revisit later
    "spatial_proteomics & spatial transcriptomics": "multi_omics (proteomics+transcriptomics)",
    "protocol of MRNA + IMC": "protocol of MRNA + IMC",  # passthrough - unresolved, revisit later
}

# ── REVIEW_STATUS ─────────────────────────────────────────────────────────────
# Same meaning across all three sheets. Resolves the open "REVIEW_STATUS
# design" item from the project handoff: "manual & auto" / "auto&manual" /
# "auto & manual" are all the same thing typed three different ways - they
# become a real 4th status value, auto_confirmed (auto-generated, later
# manually confirmed), rather than a separate REVIEWED_BY/DATE column.
REVIEW_STATUS_MAP = {
    "manual": "manual",
    "needs_review": "needs_review",
    "auto": "auto",
    "manual & auto": "auto_confirmed",
    "auto&manual": "auto_confirmed",
    "auto & manual": "auto_confirmed",
}
