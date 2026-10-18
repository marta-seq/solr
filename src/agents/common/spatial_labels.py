"""
spatial_labels.py - split the free-text AP_pub `type of spatial data` into
  * spatial_data_category: spatial_proteomics / spatial_transcriptomics / spatial_metabolomics (';'-joined)
  * spatial_data_method:   the concrete platform(s) - IMC, CODEX, GeoMx DSP, Visium ... ('; '-joined)
(same two-level split the `data` sheet already has: spatial_data_category + spatial_data_method).

Keyword based, first-match-per-platform, deliberately simple and auditable. `classify(raw, context)`: platforms are looked for in the raw
cell first; only when the raw cell names no platform at all (generic "spatial proteomics", "ST", "SP") the context (title + abstract) is
searched. The category comes from the platforms found, plus generic words in the raw cell.
"""
import re

SP, ST, SM = "spatial_proteomics", "spatial_transcriptomics", "spatial_metabolomics"
_CAT_ORDER = (SP, ST, SM)

# (regex, canonical method, category) - order = output order for a multi-platform cell.
# category None = the technique does not decide the modality (MSI can be proteomics, metabolomics or both; Marta 2026-10-18):
# for those the category comes only from what the text says is measured, see _MEASURED_CAT.
_PLATFORMS = [
    (r"CS:\bIMC\b", "IMC", SP),
    (r"imaging mass cytometry|image mass cytometry|hyperion|SABER-IMC", "IMC", SP),
    (r"\bMIBI|multiplexed ion[- ]beam", "MIBI-TOF", SP),
    (r"codex|phenocycler|phenoimager", "CODEX", SP),
    (r"cy\s?cif|\b4i\b", "CyCIF", SP),
    (r"\bMILAN\b", "MILAN", SP),
    (r"\bIBEX\b", "IBEX", SP),
    (r"macsima|\bMICS\b", "MACSima", SP),
    (r"\bCOMET\b|lunaphore", "COMET", SP),
    (r"chipcytometry", "ChipCytometry", SP),
    (r"\bMELC\b", "MELC", SP),
    (r"cell dive", "Cell DIVE", SP),
    (r"\bOrion\b", "Orion", SP),
    (r"seqIF|sequential immunofluorescence", "seqIF", SP),
    (r"multiplex(ed)? (immuno)?fluorescen|\bq?mIF\b|opal|vectra|polaris|high-plex (spatial )?immunofluorescence|"
     r"multispectral imaging|fluorescent immunohistochemistry|fluorescent IHC", "mIF", SP),
    (r"multiplex(ed)? (IHC|immunohistochemistry)|\bmIHC\b|\bmfIHC\b|\bsIHC\b|multicolor immunohistochemistry", "mIHC", SP),
    (r"\bIHC\b|^immunohistochemistry$", "IHC", SP),
    (r"laser[- ]capture micro-?dissection|laser micro-?dissection|\bLCM\b|\bLMD\b|deep visual proteomics|\bDVP\b|\bPSPro\b", "LCM-MS", SP),
    (r"geomx|digital spatial profil|\bDSP\b", "GeoMx DSP", SP),
    (r"maldi|\bMSI\b|mass spectrometry imaging|imaging mass spectrometry|\bdesi\b|nano-desi|LA-ICP|\bSIMS\b", "MSI", None),
    (r"visium", "Visium", ST),
    (r"xenium", "Xenium", ST),
    (r"cosmx", "CosMx", ST),
    (r"merfish|merscope", "MERFISH", ST),
    (r"seqfish|seq-smfish", "seqFISH", ST),
    (r"smfish", "smFISH", ST),
    (r"rnascope", "RNAscope", ST),
    (r"starmap", "STARmap", ST),
    (r"slide-?seq", "Slide-seq", ST),
    (r"slide-tags", "Slide-tags", ST),
    (r"stereo-?seq", "Stereo-seq", ST),
    (r"seq-scope", "Seq-Scope", ST),
    (r"in situ sequencing|\bISS\b|barista|exseq", "ISS", ST),
    (r"\bHCR\b|hybridization chain reaction", "HCR", ST),
    (r"select-seq|sci-space|fullscope", "other ST", ST),
]
# "CS:" prefix = case-sensitive (short acronyms that are ordinary words/other things when lowercased)
_PLATFORMS = [(re.compile(p[3:]) if p.startswith("CS:") else re.compile(p, re.I), m, c) for p, m, c in _PLATFORMS]

_GENERIC_CAT = [
    (re.compile(r"proteom|\bSP\b|multiplex|immuno|mass cytometry", re.I), SP),
    (re.compile(r"transcriptom|\bST\b|RNA[- ]seq|in situ hybrid", re.I), ST),
    (re.compile(r"metabol|lipid", re.I), SM),
]

# What a technique-agnostic platform (MSI) measured - looked for in the raw cell, then in title/abstract. Never inferred from the method.
_MEASURED_CAT = [
    (re.compile(r"proteom|peptide|\bproteins?\b|tryptic|on-tissue digest", re.I), SP),
    (re.compile(r"metabol|lipid|small molecule|glycan|\bdrugs?\b", re.I), SM),
]


def _platforms_in(text: str):
    found = []
    for rx, method, cat in _PLATFORMS:
        if rx.search(text) and (method, cat) not in found:
            found.append((method, cat))
    return found


def classify(raw, context: str = "") -> tuple:
    """-> (spatial_data_category, spatial_data_method), both '' when nothing can be said."""
    raw = "" if raw is None else str(raw).strip()
    if raw.lower() in ("", "na", "nan", "none"):
        raw = ""
    found = _platforms_in(raw)
    if not found and context:
        found = _platforms_in(context)
    cats = {c for _, c in found if c}
    for rx, cat in _GENERIC_CAT:
        if raw and rx.search(raw):
            # a generic word in the raw cell only adds a category when no platform already decided it
            if not found:
                cats.add(cat)
    # GeoMx can be protein (IPA) or RNA (WTA) - add ST when the raw cell says so
    if any(m == "GeoMx DSP" for m, _ in found) and re.search(r"transcript|WTA|RNA", raw, re.I):
        cats.add(ST)
    # generic ST/proteomics words alongside a platform ("IMC, spatial transcriptomics") still count
    if found and raw:
        for rx, cat in _GENERIC_CAT[1:2]:
            if rx.search(raw):
                cats.add(cat)
    # technique-agnostic platforms (MSI): category only from the measured analyte named in the raw cell, else the context
    if any(c is None for _, c in found) and not (cats & {SP, SM}):
        for text in (raw, context):
            hit = {cat for rx, cat in _MEASURED_CAT if text and rx.search(text)}
            if hit:
                cats |= hit
                break
    category = ";".join(c for c in _CAT_ORDER if c in cats)
    method = "; ".join(m for m, _ in found)
    return category, method
