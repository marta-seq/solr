"""
tissue_disease_maps.py
Canonical taxonomies and raw-value -> canonical mappings for the datasets
sheet's `tissue` and `disease` free-text columns, plus a new `disease_specifics`
field derived from `disease` (no separate raw column for it - it's split out
of the same free-text cell).

Built from every distinct raw value actually present in
datasets_2026-08-29.csv (160 rows, 73 distinct (tissue, disease) pairs),
reviewed and decided together with Marta on 2026-09-03. Same conventions as
category_maps.py:
  - Literal whole-cell lookup (not fuzzy/substring matching) - auditable,
    and a genuinely new raw value added later should fail loudly rather
    than being silently guessed at, once this is wired into 01_parse_excel.py.
  - All three fields are multi-label: a mapped value is a ';'-joined
    canonical string (parsed into a real list at export time via the
    existing parse_id_list pattern), same as category/pipeline_category.
  - A mapped value of None means: deliberately left unset (Marta's call,
    not a guess) - treat the same as a blank cell.

Design decisions from the 2026-09-03 conversation:
  - `disease` is kept a SMALL, CLEAN, CONSISTENT vocabulary (cancer, healthy,
    diabetes type 1, inflammatory bowel disease, tuberculosis, pulmonary
    arterial hypertension, COVID-19, wound healing) - every specific cancer
    type (melanoma, glioblastoma, TNBC, NSCLC, HCC, PDAC, sarcoma, chordoma,
    CTCL, cutaneous SCC, ...) buckets under disease="cancer", with the
    specific name/subtype living in disease_specifics instead. Rationale:
    a top-level filter needs to stay small to be useful, and nothing is lost
    since disease_specifics still supports exact search on the specific type.
  - A row can genuinely have >1 disease value (e.g. a healthy/cancer mixed
    cohort) - this is expected, not an error.
  - `tissue` brain subregions (hippocampus, CA1 region, somatosensory
    cortex, ...) are deliberately NOT collapsed into a bare "brain" value -
    unlike disease, there's no tissue-equivalent of disease_specifics to
    catch that detail if it were collapsed, and these are genuinely
    different regions, not messy duplicates of the same thing. Instead each
    becomes a multi-value tissue cell: "brain; hippocampus" - a coarse
    "show all brain datasets" filter still works (substring/contains-brain
    matching over the list), without losing the specific region.

Flagged for Marta's review (not guessed at - see inline comments below):
  - Rows where `disease` is blank but `tissue` isn't (mostly brain
    subregions) are NOT defaulted to "healthy" - genuinely unknown/
    unstated, not the same claim as "confirmed healthy control".
  - 'normal / disease' (bone marrow) - too vague to assign a specific
    disease; left as needs_review.
  - 'several' / 'several; disease-free controls; multiple types of
    carcinomas, sarcomas, and central nervous system lesions' (tissue) -
    genuinely unspecified across many samples; left as needs_review rather
    than guessed.
  - 'benign nevi and 67 melanomas' - "benign nevi" isn't cleanly cancer or
    healthy; bucketed as disease="cancer" (since the melanoma cases
    dominate the dataset's purpose) with both terms kept in specifics -
    flag if that's wrong.
  - 'biopsy-induced physical wounding' - given its own disease bucket
    "wound healing" rather than forced into cancer/healthy - confirm the
    bucket name.
  - Two rows (see CROSS_COLUMN_FIXES below) had disease info leak into the
    `tissue` cell while `disease` itself was left blank: 'gliomas' and the
    10xGenomics URL row (inferred disease=cancer/ovarian cancer from
    "...OvarianCancer" in the URL text itself, since the cell had no other
    information) - confirm both guesses are correct.
"""

# ── Canonical disease taxonomy ────────────────────────────────────────────
DISEASE_TAXONOMY = [
    "cancer",
    "healthy",
    "diabetes type 1",
    "inflammatory bowel disease",
    "tuberculosis",
    "pulmonary arterial hypertension",
    "COVID-19",
    "wound healing",
]

# raw `tissue` cell -> canonical ';'-joined tissue string, or None to leave
# unset. Full enumeration of every distinct raw value actually in the data
# (same convention as category_maps.py) - including trivial identity
# entries for already-clean values - so an unmapped value reliably means
# "genuinely new, not yet reviewed" and can fail loudly at export time
# instead of being silently guessed at.
TISSUE_MAP = {
    "Lung":                                  "lung",  # unify casing with the separate "lung" row below
    "bone":                                  "bone",
    "bone marrow":                           "bone marrow",
    "brain":                                 "brain",
    "breast":                                "breast",
    "breast; renal cell; head and neck; colorectal; lung":
                                              "breast; renal cell; head and neck; colorectal; lung",
    "colon":                                 "colon",
    "colorectal":                            "colorectal",
    "decidua":                               "decidua",
    "embryo":                                "embryo",
    "heart":                                 "heart",
    "ileum":                                 "ileum",
    "liver":                                 "liver",
    "lung":                                  "lung",
    "pancreas":                              "pancreas",
    "skin":                                  "skin",
    "spleen":                                "spleen",
    "VISp":                                  "visual cortex (VISp)",
    "brain CA1 region":                      "brain; CA1 region",
    "brain CNS":                             "brain",
    "brain hippocampus":                     "brain; hippocampus",
    "brain mouse VISp? 3D":                  "brain; visual cortex (VISp)",
    "brain preoptic hypothalamus":           "brain; preoptic hypothalamus",
    "brain somatosensory cortex":            "brain; somatosensory cortex",
    "brain subventricular zone and olfactory bulb ; NIH/3T3 cells":
                                              "brain; subventricular zone; olfactory bulb",
    "colon, liver":                          "colon; liver",
    "dorsolateral prefrontal cortex":        "brain; dorsolateral prefrontal cortex",
    "endometrio":                            "endometrium",
    # mislabeled - these are diseases, not tissues (see DISEASE_MAP for the
    # corresponding disease-side fix on these same rows)
    "gliomas":                               "brain",
    "melanoma":                              "skin",
    "https://support.10xgenomics.com/spatial-gene-expression/datasets/1.2.0/Parent_Visium_Human_OvarianCancer":
                                              "ovary",  # inferred from URL text - confirm
    "intestine;  eight sections from nine individuals. The eight regions "
    "(in order of trajectory from the stomach) were as follows: the "
    "duodenum, proximal jejunum, mid-jejunum and ileum from the small "
    "intestine, and the ascending, transverse, descending and sigmoid "
    "regions of the large intestine":
                                              "small intestine; large intestine; duodenum; jejunum; ileum",
    "lymphoid tissues: three tonsils, a spleen, and a LN":
                                              "tonsil; spleen; lymph node",
    "primary visual cortex (VISp)":          "visual cortex (VISp)",
    "whole brain":                           "brain",
    # genuinely too vague to assign specific tissue(s) - needs_review, not guessed
    "several":                               None,
    "several; disease-free controls; multiple types of carcinomas, "
    "sarcomas, and central nervous system lesions":
                                              None,
}

# raw `disease` cell -> canonical ';'-joined disease string (small
# vocabulary - see DISEASE_TAXONOMY). None = needs_review, not guessed.
DISEASE_MAP = {
    "COVID-19; acute lung injury":           "COVID-19",
    "Cutaneous T cell lymphomas (CTCL). pembrolizumab clinical trial "
    "responders vs non responders":          "cancer",
    "NCLC (LUAD + LUSC)":                    "cancer",
    "NSCLC":                                 "cancer",
    "Non–Small Cell Lung Cancer":       "cancer",
    "Squamous cell carcinoma; Adenocarcinoma; Matched Metastatic Lymph Node": "cancer",
    "TNBC ( Triple Negative Breast Cancer)": "cancer",
    "Type-I diabetes, progression":          "diabetes type 1",
    "adenocarcinoma (LUAD)":                 "cancer",
    "benign nevi and 67 melanomas":          "cancer",  # see flagged note above
    "biopsy-induced physical wounding":      "wound healing",  # see flagged note above
    "breast cancer (BC), renal cell carcinoma (RCC), squamous cell "
    "carcinoma of head and neck (SCCHN), colorectal carcinoma (CRC), and "
    "non-small cell lung cancer (NSCLC)":    "cancer",
    "cancer":                                "cancer",
    "cancer - advanced":                     "cancer",
    "cancer - early stage":                  "cancer",
    "cancer - high grade":                   "cancer",
    "cancer - transition to invasive":       "cancer",
    "cancer Infiltrating ductal carcinoma, Ductal carcinoma in situ;  "
    "Invasive lobular carcinoma":            "cancer",
    "cancer, Pancreatic ductal adenocarcinoma (PDAC); liver metastases": "cancer",
    "cancer, TNBC":                          "cancer",
    "cancer, TNBC. Black vs white women":    "cancer",
    "cancer, TNBC. phase 2 TONIC trial, with samples spanning primary "
    "tumors, pretreatment metastases and on-treatment metastases during "
    "nivolumab therapy":                     "cancer",
    "cancer; pancreatic ductal adenocarcinoma": "cancer",
    "cancer; primary breast tumors and matched lymph node metastases": "cancer",
    "chordoma":                              "cancer",
    "cutaneous squamous cell carcinoma":     "cancer",
    "diabetes type 1":                       "diabetes type 1",
    "disease-free controls; multiple types of carcinomas, sarcomas, and "
    "central nervous system lesions":        "healthy; cancer",
    "glioblastoma":                          "cancer",
    "healthy":                               "healthy",
    "healthy, aging":                        "healthy",
    "healthy/cancer Adult human healthy lung section ; Adult human lung "
    "with Invasive AdenoCarcinoma":          "healthy; cancer",
    "hepatocellular carcinoma (HCC)":        "cancer",
    "hepatocellular carcinoma (HCC)\xa0(\xa0checkpoint immunotherapy )": "cancer",
    "inflammatory bowel disease (IBD)":      "inflammatory bowel disease",
    "melanoma":                              "cancer",
    "melanoma - stage IV w/ treatment":      "cancer",
    "melanoma receiving anti-programmed cell death-1 (anti-PD-1) therapy": "cancer",
    "non-small cell lung cancer":            "cancer",
    "non-small cell lung cancer (NSCLC)":    "cancer",
    "normal / disease":                      None,  # too vague - needs_review
    "pulmonary arterial hypertension (PAH)": "pulmonary arterial hypertension",
    "sarcoma; undifferentiated pleomorphic sarcoma; myxofibrosarcoma": "cancer",
    "triple-negative breast cancer (TNBC). With treatment": "cancer",
    "tuberculosis":                          "tuberculosis",
    "type 1 diabetes":                       "diabetes type 1",
}

# raw `disease` cell -> canonical ';'-joined disease_specifics string (loose
# vocabulary - specific subtype names, treatment/trial/cohort/stage notes,
# anything worth keeping that doesn't belong in the small `disease` bucket
# above). Not listed = no specifics beyond the plain disease bucket.
DISEASE_SPECIFICS_MAP = {
    "COVID-19; acute lung injury":           "acute lung injury",
    "Cutaneous T cell lymphomas (CTCL). pembrolizumab clinical trial "
    "responders vs non responders":          "CTCL; pembrolizumab clinical trial; responders vs non-responders",
    "NCLC (LUAD + LUSC)":                    "NSCLC; LUAD; LUSC",
    "NSCLC":                                 "NSCLC",
    "Non–Small Cell Lung Cancer":       "NSCLC",
    "Squamous cell carcinoma; Adenocarcinoma; Matched Metastatic Lymph Node":
                                              "squamous cell carcinoma; adenocarcinoma; metastatic lymph node",
    "TNBC ( Triple Negative Breast Cancer)": "TNBC",
    "Type-I diabetes, progression":          "progression cohort",
    "adenocarcinoma (LUAD)":                 "LUAD",
    "benign nevi and 67 melanomas":          "benign nevi; melanoma",
    "breast cancer (BC), renal cell carcinoma (RCC), squamous cell "
    "carcinoma of head and neck (SCCHN), colorectal carcinoma (CRC), and "
    "non-small cell lung cancer (NSCLC)":    "BC; RCC; SCCHN; CRC; NSCLC",
    "cancer - advanced":                     "advanced stage",
    "cancer - early stage":                  "early stage",
    "cancer - high grade":                   "high grade",
    "cancer - transition to invasive":       "transition to invasive",
    "cancer Infiltrating ductal carcinoma, Ductal carcinoma in situ;  "
    "Invasive lobular carcinoma":            "infiltrating ductal carcinoma; ductal carcinoma in situ; invasive lobular carcinoma",
    "cancer, Pancreatic ductal adenocarcinoma (PDAC); liver metastases": "PDAC; liver metastases",
    "cancer, TNBC":                          "TNBC",
    "cancer, TNBC. Black vs white women":    "TNBC; Black vs white women cohort",
    "cancer, TNBC. phase 2 TONIC trial, with samples spanning primary "
    "tumors, pretreatment metastases and on-treatment metastases during "
    "nivolumab therapy":                     "TNBC; TONIC trial (phase 2); nivolumab therapy; primary tumor; pretreatment metastases; on-treatment metastases",
    "cancer; pancreatic ductal adenocarcinoma": "PDAC",
    "cancer; primary breast tumors and matched lymph node metastases": "primary breast tumor; matched lymph node metastases",
    "chordoma":                              "chordoma",
    "cutaneous squamous cell carcinoma":     "cutaneous squamous cell carcinoma",
    "disease-free controls; multiple types of carcinomas, sarcomas, and "
    "central nervous system lesions":        "disease-free controls; carcinoma; sarcoma; CNS lesion",
    "glioblastoma":                          "glioblastoma",
    "healthy, aging":                        "aging cohort",
    "healthy/cancer Adult human healthy lung section ; Adult human lung "
    "with Invasive AdenoCarcinoma":          "invasive adenocarcinoma",
    "hepatocellular carcinoma (HCC)":        "HCC",
    "hepatocellular carcinoma (HCC)\xa0(\xa0checkpoint immunotherapy )": "HCC; checkpoint immunotherapy",
    "melanoma":                              "melanoma",
    "melanoma - stage IV w/ treatment":      "melanoma; stage IV; with treatment",
    "melanoma receiving anti-programmed cell death-1 (anti-PD-1) therapy": "melanoma; anti-PD-1 therapy",
    "non-small cell lung cancer":            "NSCLC",
    "non-small cell lung cancer (NSCLC)":    "NSCLC",
    "sarcoma; undifferentiated pleomorphic sarcoma; myxofibrosarcoma":
                                              "sarcoma; undifferentiated pleomorphic sarcoma; myxofibrosarcoma",
    "triple-negative breast cancer (TNBC). With treatment": "TNBC; with treatment",
    "biopsy-induced physical wounding":      "biopsy-induced physical wounding model",
}

# One-off cross-column fixes: a raw value that leaked disease information
# into the `tissue` cell while the `disease` cell itself was left blank -
# can't be expressed as a normal per-column TISSUE_MAP/DISEASE_MAP lookup,
# since the disease-column lookup would run on an empty string, never on
# this value. Keyed by the raw `tissue` value; when it matches, apply the
# given disease/disease_specifics IN ADDITION to (not instead of) the
# normal TISSUE_MAP fix for that same raw tissue value.
CROSS_COLUMN_FIXES = {
    "gliomas": {"disease": "cancer", "disease_specifics": "glioma"},
    "https://support.10xgenomics.com/spatial-gene-expression/datasets/1.2.0/Parent_Visium_Human_OvarianCancer":
        {"disease": "cancer", "disease_specifics": "ovarian cancer"},
}
