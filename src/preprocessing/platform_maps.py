"""
platform_maps.py
Canonical raw-value -> canonical mapping for the datasets sheet's
`spatial_data_method` free-text column (the acquisition platform: IMC,
CODEX, Visium, etc.).

Built from every distinct raw value actually present in
datasets_2026-08-29.csv (160 rows, 49 distinct values), 2026-09-15. Same
conventions as tissue_disease_maps.py:
  - Literal whole-cell lookup (not fuzzy/substring matching) - a genuinely
    new raw value fails loudly at export time rather than being silently
    guessed at.
  - Multi-label: a mapped value is a ';'-joined canonical string, parsed
    into a real list at export time. Most rows have exactly one platform;
    a few genuine multi-platform comparison rows have several.
  - A mapped value of None means: deliberately left unset (needs Marta's
    call, not a guess) - treated the same as a blank cell.

Unlike disease/tissue, most of this map is mechanical (casing/vendor-prefix
dedup: "10x Visium" / "10X Visium" / "VISIUM" / "10x Genomics Visium" all ->
"Visium"; "CosMX SMI" / "NanoString CosMx" -> "CosMx"; "imaging mass
cytometry" -> "IMC"), not a judgment call - the platform names themselves
are an established, external vocabulary. "MIBI" is folded into "MIBI-TOF"
as the same instrument/technique referred to by its short name in some
papers - flag if that merge is wrong.

Flagged for Marta's review (not guessed at):
  - "10x Visium resolution??" (2 rows) - the "??" is in the raw data itself,
    i.e. she was already unsure what this meant when curating it.
  - "IMC + mRNA" (1 row) - genuinely a combined IMC + spatial transcriptomics
    row, but which specific transcriptomics method isn't stated.
  - "spatial ATAC�RNA-seq & Spatial CUT&Tag�RNA-seq" (1 row) - raw
    cell has mangled encoding characters (likely an em-dash or ampersand
    that didn't survive a prior export) on top of describing a composite
    multi-omic method with no single platform name.
"""

# raw `spatial_data_method` cell -> canonical ';'-joined platform string, or
# None to leave unset. Full enumeration of every distinct raw value actually
# in the data (same convention as tissue_disease_maps.py).
PLATFORM_MAP = {
    "IMC":                                      "IMC",
    "MIBI-TOF":                                 "MIBI-TOF",
    "MIBI":                                     "MIBI-TOF",  # short name for the same instrument/technique - confirm
    "CODEX":                                    "CODEX",
    "MERFISH":                                  "MERFISH",
    "MERSCOPE":                                 "MERSCOPE",
    "IF":                                       "IF",
    "IHC":                                      "IHC",
    "mIF":                                      "mIF",
    "mfIHC":                                    "mfIHC",
    "t-CyCIF":                                  "t-CyCIF",
    "orion":                                    "Orion",
    "osmFISH":                                  "osmFISH",
    "smFISH":                                   "smFISH",
    "seqFISH":                                  "seqFISH",
    "seqFISH+":                                 "seqFISH+",
    "VeraFISH":                                 "VeraFISH",
    "ISS":                                      "ISS",
    "Seq-Scope":                                "Seq-Scope",
    "GeoMx DSP":                                "GeoMx DSP",
    "Nanostring GeoMx Digital Spatial Profiling (DSP)": "GeoMx DSP",
    "Xenium":                                   "Xenium",
    "10x Genomics Xenium":                      "Xenium",
    "Visium":                                   "Visium",
    "VISIUM":                                   "Visium",
    "10x Visium":                               "Visium",
    "10X Visium":                               "Visium",
    "10x Genomics Visium":                      "Visium",
    "Visium (brain)":                           "Visium",  # tissue detail lives in the tissue field already
    "Visium (kidney)":                          "Visium",
    "Slide-seq":                                "Slide-seq",
    "Slide-seqV2":                              "Slide-seqV2",
    "Slide-seq v.2":                            "Slide-seqV2",
    "Stereo-seq":                               "Stereo-seq",
    "Stero-seq":                                "Stereo-seq",  # typo fix
    "STARmap":                                  "STARmap",
    "STARMap":                                  "STARmap",  # casing fix
    "STARmap PLUS":                             "STARmap PLUS",
    "CosMx":                                    "CosMx",
    "NanoString CosMx":                         "CosMx",
    "CosMX SMI":                                "CosMx",
    "imaging mass cytometry":                   "IMC",
    "CODEX; CyCIF; Vectra; MIBI-TOF; MxIF; IMC": "CODEX; CyCIF; Vectra; MIBI-TOF; MxIF; IMC",
    "Fluorescent; H&E; light microscopy":       "Fluorescent microscopy; H&E; light microscopy",
    # genuinely ambiguous / flagged by Marta herself - needs_review, not guessed
    "10x Visium resolution??":                  None,
    "IMC + mRNA":                               None,
    # en-dashes (U+2013), not hyphens - built via chr() rather than a typed
    # literal to guarantee an exact match against the raw CSV cell.
    f"spatial ATAC{chr(0x2013)}RNA-seq & Spatial CUT&Tag{chr(0x2013)}RNA-seq": None,

    # Added 2026-09-23: new raw values surfaced by Phase 3 agent-created
    # dataset rows (compared_methods_agent/data_fetch_agent free-text
    # extraction), found via 03_export_json.py's unmapped-value warnings -
    # Marta noticed these fragmenting the Datasets tab's platform filter
    # (e.g. "IMC" and "Imaging Mass Cytometry (IMC)" showing as separate
    # options for the same real platform).
    # Casing/naming variants of already-canonical platforms - collapsed:
    "Imaging Mass Cytometry":                   "IMC",
    "Imaging Mass Cytometry (IMC)":             "IMC",
    "cycif":                                    "t-CyCIF",  # only existing CyCIF-family canonical value - confirm this is the right merge, not a distinct non-tissue-based CyCIF variant
    "multiplex immunofluorescence":             "mIF",
    "VectraPolaris":                            "Vectra/Polaris",
    "Vectra/Polaris":                           "Vectra/Polaris",
    "Vectra Polaris":                           "Vectra/Polaris",
    "Phenocycler":                              "CODEX",  # Akoya's rebrand of CODEX - same instrument
    # Generic modality description, not a specific platform - same
    # treatment as "10x Visium resolution??" above. This is the literal
    # value Marta flagged as "spatial transcriptomics is appearing, and it
    # shouldn't" - it isn't an acquisition platform, just a vague label.
    "spatial transcriptomics":                  None,
    # compound value mixing a real platform with the same generic noise
    # term above - keep the real platform, drop the noise.
    "spatial transcriptomics; multiplexed ion beam imaging": "MIBI-TOF",
    # Genuinely new platforms not seen in the original 2026-08-29 snapshot -
    # already-clean names, identity-mapped (same pattern as the original
    # map's already-clean single values).
    "scMEP":                                    "scMEP",
    "MALDI-MSI":                                 "MALDI-MSI",
    "LOPIT":                                    "LOPIT",
    "SIMS":                                     "SIMS",
    "MELC":                                     "MELC",
    "4i":                                       "4i",
    "LSFM":                                     "LSFM",
    "SPOT":                                     "SPOT",
    "EASI-FISH":                                "EASI-FISH",
    "Slide-DNA-seq":                            "Slide-DNA-seq",
    "Slide-TCR-seq":                            "Slide-TCR-seq",
    "Spatial-CITE-seq":                         "Spatial-CITE-seq",
    "spatial CUT&Tag-RNA-seq":                  "Spatial CUT&Tag-RNA-seq",
    "Sequential Immunofluorescence":            "Sequential Immunofluorescence",  # Lunaphore-style seqIF - confirm this shouldn't just collapse into mIF
    # Genuinely unclear acronyms / not actually an acquisition platform -
    # left as needs_review rather than guessed, same as the "??"-flagged
    # rows above.
    "IST":                                      None,  # acronym meaning not confirmed
    "IBT":                                      None,  # acronym meaning not confirmed
    "SpaSim":                                   None,  # this looks like a spatial-data SIMULATION tool, not a real acquisition platform - confirm before mapping
}
