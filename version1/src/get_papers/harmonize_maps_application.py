# This file contains all keyword mapping dictionaries for the Living Review project.

import re

keyword_map_methods = {

    # the ones techniques
    '10x_Genomics_Visium': ['visium'],
    'Xenium_In_Situ': ['xenium'],
    'CosMx_Spatial_Molecular_Imaging': ['cosmx'],
    'multiplexed_CODetection_by_indEXing(CODEX)': ['codex'],
    'GeoMx_Digital_Spatial_Profiling': ['geomx', 'Digital Spatial Profiling',
                                        "NanoString profiling", "DSP", "digital spatial RNA profiling"],
    'Imaging_Mass_Cytometry(IMC)': ['imaging mass cytometry', 'imc'],

    # Adding these last to not interfere with the others concepts
    'Immunohistochemistry(IHC)': ['immunohistochemistry', 'IHC'],
    'Immunofluorescence(IF)': ['immunofluorescence', 'IF', 'mIF'],
    'Histology_Images': ['histology', 'HE', 'histopathology', 'histological sections',
                         "Histology", "H&E", "histopathological", "FFPE", ],

    "MALDI-MSI": ["(MALDI-MSI)", "MALDI-MSI", "MALDI-IMS"],
    "Mass_Spectrometry_Imaging": ["imaging mass spectrometry", "Mass Spectrometry Imaging", "MSI"],

    "Multiplexed_Ion_Beam_Imaging(MIBI)": ["MIBI", "MIBI-TOF"],

    "Single-cell_RNA-seq(scRNA-seq)": ["scRNA-seq", "single cell RNA sequencing", "single cell RNA seq",
                                       "single-cell RNA"],
    "Single-nucleus_RNA-seq(snRNA-seq)": ["snRNA-seq", "single-nucleus RNA sequencing", "single nuclei RNA sequencing",
                                          "single-nucleus RNA-seq"],
    "Bulk_RNAseq": ["bulk RNA-seq", "bulk RNA sequencing", "bulk transcriptomic data", "Bulk RNA-seq"],

    'data_related_terms': ['data', 'dataset', 'datasets'],
    'analysis_related_terms': ['analysis', "analyses"],

    "general_spatial_transcriptomics": ["Spatial transcriptomics", "single-nuclear spatial-RNA expression"],
    "general_spatial_metabolomics": ["Spatial metabolomics", "spatial metabolome"],
    "general_spatial_proteomics": ["Spatial proteomics"],
    "TCR_related": ["TCR", "spatial immunoreceptor profiling", "BCR profiling"],

    # other irrelevant
    "other_imaging": ["positron emission tomography", "PET", "Magnetic Resonance Imaging", "MRI", "DTI", "FTIR",
                      "Magnetic Resonance Spectroscopy", "Transmission electron microscopy",
                      "Transmission electron microscopy",
                      "microscopy", "Echo-Planar Imaging", "tomography",
                      "stained", "staining", "radiography", "NODDI", "tomo-seq",
                      ],

    "other_methods": ["affinity proteomics", "size-exclusion chromatography", "PAGE", "LC-MS/MS", "PCR",
                      "ELISA", "whole-genome sequencing", "sequencing", "western", "NMR",
                      "TCR sequencing", "spectral flow cytometry", "2D DIGE", "electrophoresis",
                      "genomics", "Raman", "microdissection", "chromatin accessibility", "global chromatin mapping"
                                                                                         "Real-Time Polymerase Chain Reaction",
                      "Polymerase Chain Reaction", "flow cytometry", "CRISPR", "docking", "optogenetics",
                      "wire myography", "LIBS", "mass spectrometry", "mass cytometry", "LC-MS",
                      "immunoblotting", "immunolabeling", "Mass spectrometry", "UPLC-Orbitrap Fusion MS",
                      "liquid chromatography-mass spectrometry", "MALDI-MS", "EDX", "siRNA transfection",
                      "UPLC-Q-TOF/MS", "ToF-SIMS", "HPLC", "GC-MS", "UHPLC-Q exactive orbitrap-HRMS",
                      "electrospray ionization", "nLC-ESI-MS/MS", "ultrasound", "echocardiography",
                      "UPLC-DAD-MS", "MALDI-TOF MS", "MALDI", "LC-QQQ-MS",
                      "ATAC-seq", "snATAC-seq", "scATAC-seq", "ATAC", "chromatin conformation capture",
                      "chromatin immunoprecipitation",
                      "CITE-seq", "ChIP-seq", "SNARE-seq2", "qRT-RNA",
                      "cell-free RNA", "SHARE-seq"
                      ],

    "general_transcriptomics": ["transcriptome", "transcriptome profiling", "transcriptomic", "bulk transcriptome",
                                "gene expression", "epigenetic", "RNASeq", "bulk RNA",
                                ],
    "general_proteomics": ["proteome", "bulk proteomic", "phosphoproteomic", "proteomic", "peptidomic"],
    "general_metabolomics": ["metabolomic", "bulk metabolomic", "untargeted metabolomic", "lipidomic",
                             ],
    "general_omics": ["spatial omics", "multiomics", "multi-omics", "spatial epigenome", "glycomics",
                      "spatial metallomics",
                      ],

    "irrelevant_terms": ['in vitro', 'ex vivo', 'in vivo', 'mouse', 'Survey', 'cell line', 'organelle',
                         'human', 'chloroplast', 'clones', "field observations", "Geographic", "organoid",
                         "patient", "zebrafish", "samples", "clinical", "Body Mass Index",
                         "cryosections", "TurboID", "neurons", "RGB images", "Demography", "Geography", "Socioeconomic",
                         "HRGRE", "salinity", "turbidity", "nutrients", "phytoplankton community composition",
                         "diatom species abundance", "topsoil metal concentrations",
                         "personal exposimeter measurements",
                         "Incidence rates", "rates", "experimental seedling sowing", "swabs", "statistical", "tools",
                         "networks", "Plague", "autopsy", "survey", "atlas", "GEO", "keratinocytes", "co-culture",
                         "transwell", "MHC", "animal", "co-immunoprecipitation", "microarray", "TCGA", "arrays",
                         "lymph", "extracellular", "DNA methylation", "mice", "assay", "lymphocyte", "tests",
                         "xenograft", "models", "specimens", "biopsies", "cells", "studies", "experiments", "ratio",
                         "PD/L1", "populations", "insect", "sera", "dilution", "lineage-tracing", "inhibition",
                         "kidney", "rat", "electrophysiology", "velocity", "multi-ome", "Exosomes", "murine",
                         "brain", "tumours", "tumor", "cerebrospinal", "profiles", "rabies", "carcinoma", "tissue",
                         "metaplasia", "hippocampus", "panel", "cultures", "Morris water maze test", "TMB",
                         "networking",
                         "TEM", "hormone", "CUT&RUN", "hair", "GTEX", "EDS", "biopsy", "dysplasia", "ubiquitinome",
                         "measurements", "Staphylococcus aureus", "reporter tagging", "regions", "cryosection",
                         "optical clearing", "tracing", "array", "Golgi", "biology", "experiment", "blood", "biopsy",
                         "picrosirius", "trichrome", "assessment", "mucosa", "hormone", "model", "markers", "viral",
                         "BioID", "morphometry", "single-cell", "immunolocalization", "culture", "CGGA", "MRS",
                         "oxygen", "cancer", "genetics", "features", "DESI", "SEM", "skin", "statistic", "interfering",
                         "PBMC", "serial", "paraffin", "dietary", "variation", "recognition", "deletion", "RECIST",
                         "validation", "radiation", "replication", "testing", "changes", "fluorescence", "detection",
                         "histologic types", "ST", "mutational", "cytokine", "alveolar", "bulk-seq", "method",
                         "network", "resonance", "ultrasounds", "immunoprecipitation", "labeling",
                         "pbacBarcode", "ATOH1", "consumption", "epitope", "exomes", "recordings",
                         "antibodies", "dermoscopy", "epigenome", "metabolite", "digital pathology",
                         "microspectroscopy", "methylation profiling", "chromatin", "spatial profiling",
                         "LOH", "digital spatial profiler",

                         "spatial expression profiling",

                         "proteome", "mRNA", "10x Genomics", "Neuroimaging", "multiplexed imaging", "imaging",
                         "protein", "DNA methylome", "metabolites", "genomic", "microRNA", "transcript",
                         "methylome",

                         ],
    "RNA_in_situ_hybridization": ["RNA in situ hybridization", "RNA in situ",
                                  "in situ hybridization", ],
    "FISH": ["RNA FISH", "RISH"],

    "RNAScope": ["RNAScope"],

}

keyword_map_methods_general = {
    'spatial_transcriptomics': ['general_spatial_transcriptomics', '10x_Genomics_Visium',
                                'CosMx_Spatial_Molecular_Imaging',
                                'RNA_in_situ_hybridization', 'Xenium_In_Situ', 'MERFISH',
                                'FISH', 'asmFISH', 'smFISH', 'ISS', 'Slide-seq',
                                'RNAscope', 'MERSCOPE', 'quantitative FISH',
                                'BaseScope',
                                ],

    'histology': ['Histology_Images'],

    'transcriptomics': ['Single-cell_RNA-seq(scRNA-seq)', 'general_transcriptomics',
                        'Single-nucleus_RNA-seq(snRNA-seq)', 'Bulk_RNAseq', 'TCR_related',
                        ],

    'spatial_proteomics': ['Immunohistochemistry(IHC)', 'Immunofluorescence(IF)',
                           'general_spatial_proteomics', 'Imaging_Mass_Cytometry(IMC)',
                           'Multiplexed_Ion_Beam_Imaging(MIBI)', 'multiplexed_CODetection_by_indEXing(CODEX)',

                           ],

    'proteomics': ['general_proteomics', ],

    'spatial metabolomics': ['Mass_Spectrometry_Imaging', 'general_spatial_metabolomics',
                             'MALDI-MSI'],

    'metabolomics': ['general_metabolomics'],
    'general_omics': ['general_omics'],
    'other_methods': ['other_methods', 'other_imaging', ],
    'spatial multiomics': ['GeoMx_Digital_Spatial_Profiling', ],

}

TISSUE_MAP = {
    'Nervous System (Brain/CNS)': [
        re.compile(
            r'\b(?:cerebellum|hippocampus|cortex|spinal\s*cords?|white\s*matter|locus\s*coeruleus|choroid\s*plexus|olfactory\s*bulb|hypothalamus|dorsal\s*root\s*ganglia|DRG|posterior\s*fossa)\b',
            re.IGNORECASE),  # Added specific brain parts, DRG, hypothalamus, posterior fossa
        re.compile(
            r'\b(?:brain(?:s)?|neural|neurological|cerebral|glioma(?:s)?|glioblastoma|neuron(?:s)?|pituitary|ependymoma)\b',
            re.IGNORECASE),  # Added ependymoma
        re.compile(
            r'\b(?:parkinson|alzheimer|schizophrenia|stroke|subarachnoid\s*hemorrhage|SAH|epilepsy|intracerebral\s*hemorrhage|ICH|neuroblastoma|medulloblastoma|multiple\s*sclerosis|MS|amyotrophic\s*lateral\s*sclerosis|ALS|spinal\s*muscular\s*atrophy|SMA)\b',
            re.IGNORECASE),  # Re-added MS, ALS, SMA
        re.compile(r'\b(?:CNS|central\s*nervous\s*system|nerve(?:s)?|neuropath(?:y|ies)|temporal\s*lobe)\b',
                   re.IGNORECASE),
        re.compile(r'\b(?:hippocampal|Hypothalamic)\b', re.IGNORECASE),
        # Added specific brain parts, DRG, hypothalamus, posterior fossa

    ],
    'Oral/Pharyngeal/Head & Neck': [
        re.compile(
            r'\b(?:oral|gingiva|buccal\s*mucosa|pharyngeal|mouth|laryngeal|vocal\s*cord|lip|epipharynx|periodontitis|tooth|eyelid)\b',
            re.IGNORECASE),  # Re-added eyelid
        re.compile(r'\b(?:nasopharyngeal|rhinosinusitis|sinusitis|adenoid\s*cystic\s*carcinoma)\b', re.IGNORECASE),
        re.compile(r'\b(?:HNSCC|head\s*and\s*neck|craniofacial|palatogenesis|mandibular)\b', re.IGNORECASE),
    ],
    'Gastrointestinal_colorectal': [  # New broader GI category
        re.compile(
            r'\b(?:colon|rectum|colorectal|colitis|crohn(?:s)?\s*disease|ileum|j-pouch|ileal|colectomy|anastomosis|IBD)\b',
            re.IGNORECASE),  # Re-added ileum, j-pouch, ileal, colectomy, anastomosis, IBD
    ],
    'Gastrointestinal_other': [  # New broader GI category
        re.compile(r'\b(?:gastric|stomach|intestine|esophageal|gastroesophageal|celiac|hemorrhoids|intestin)\b',
                   re.IGNORECASE),
    ],
    'Breast': [
        re.compile(r'\bbreast\b', re.IGNORECASE),
    ],
    'Liver': [
        re.compile(r'\b(?:liver|hepatic|hepato|cholangiocarcinoma|hepatocellular|hepatitis)\b', re.IGNORECASE),
        # Added hepatocellular
        re.compile(r'\b(?:HCC|ICC)\b', re.IGNORECASE),
    ],
    'Kidney': [
        re.compile(r'\bkidney(?:s)?\b', re.IGNORECASE),
        re.compile(r'\brenal\b', re.IGNORECASE),
        re.compile(r'\b(?:wilms\s*tumor|nephritis|nephropathy|glomerular)\b', re.IGNORECASE),
        # Re-added nephropathy, glomerular
    ],
    'Skin': [
        re.compile(r'\bskin\b', re.IGNORECASE),
        re.compile(
            r'\b(?:epidermal|cutaneous|melanoma|acne|dermatitis|psoriasis|mycosis\s*fungoides|pyoderma\s*gangrenosum|merkel|squamous)\b',
            re.IGNORECASE),  # Re-added pyoderma gangrenosum
    ],
    'Lung': [
        re.compile(r'\blung(?:s)?\b', re.IGNORECASE),
        re.compile(r'\bpulmonary\b', re.IGNORECASE),
        re.compile(r'\bbronchial\b', re.IGNORECASE),
        re.compile(r'\b(?:airway(?:s)?|ARDS|acute\s*respiratory\s*distress\s*syndrome|asthma|asthmatic)\b',
                   re.IGNORECASE),  # Already includes asthma/asthmatic
    ],
    'Blood/Immune System': [
        re.compile(r'\bblood\b', re.IGNORECASE),
        re.compile(r'\bimmune\b', re.IGNORECASE),
        re.compile(r'\blymph(?:ocyte)?(?:s)?\b', re.IGNORECASE),
        re.compile(r'\b(?:PBMC|peripheral blood mononuclear cells)\b', re.IGNORECASE),
        re.compile(r'\bserum\b', re.IGNORECASE),
        re.compile(r'\bplasma\b', re.IGNORECASE),
        re.compile(r'\b(?:T-?cells?|B-?cells?)\b', re.IGNORECASE),
        re.compile(r'\bleukemia\b', re.IGNORECASE),
        re.compile(r'\b(?:lymphoma|hodgkin|chronic\s*granulomatous\s*disease|CGD|castleman\s*disease)\b',
                   re.IGNORECASE),  # Added CGD, Castleman
    ],
    'Bone/Joint': [
        re.compile(
            r'\b(?:bone|joint|skeletal|cartilage|arthritis|osteoarthritis|ossification|knee|chordoma|synovium|hamstring\s*tendon|tendon|limb)\b',
            re.IGNORECASE),  # Already includes osteoarthritis
        re.compile(r'\b(?:HO|osteosarcoma)\b', re.IGNORECASE),
    ],
    'Pancreas': [
        re.compile(r'\bpancreatic\b', re.IGNORECASE),
        re.compile(r'\bislets?\b', re.IGNORECASE),
        re.compile(r'\b(?:type\s*1\s*diabetes|T1D|type\s*2\s*diabetes|T2D|diabetes(?:\s*mellitus)?)\b', re.IGNORECASE),

    ],
    'Heart': [
        re.compile(r'\bheart\b', re.IGNORECASE),
        re.compile(r'\bcardi(?:ac)?\b', re.IGNORECASE),
        re.compile(r'\b(?:cardiomyopathy|myocarditis|myocardial\s*infarction|MI|ventricular)\b', re.IGNORECASE),
        # ADDED myocardial infarction, MI, ventricular
    ],
    'Prostate': [
        re.compile(r'\bprostate\b', re.IGNORECASE),
        re.compile(r'\b(?:benign\s*prostatic\s*hyperplasia|BPH)\b', re.IGNORECASE),

    ],
    'Reproductive': [
        re.compile(
            r'\b(?:uterus|ovar(?:y|ian)?|testis|gonad|genital|penile|cervical|endometrioid|endometrial|vulvar|implantation)\b',
            re.IGNORECASE),
        re.compile(r'\b(?:endometriotic\s*lesions?)\b', re.IGNORECASE)
    ],
    'Thyroid': [
        re.compile(r'\bthyroid\b', re.IGNORECASE),
    ],
    'Eye': [
        re.compile(r'\b(?:eye|cornea|retina|optic\s*nerve|retinal|ocular)\b', re.IGNORECASE),
    ],
    'Vascular': [  # New category for blood vessels
        re.compile(
            r'\b(?:aortic|aneurysm|atherosclerotic\s*plaques?|atherosclerosis|arteriovenous\s*malformations|eAVMs|artery(?:ies)?|vein(?:s)?|vessel(?:s)?|vascular)\b',
            re.IGNORECASE),  # ADDED 'atherosclerosis'
    ],
    'Placenta': [  # New category for Placenta
        re.compile(r'\bplacenta\b', re.IGNORECASE),
    ],

    'Adipose Tissue': [
        re.compile(r'\b(?:adipose\s*tissue|fat)\b', re.IGNORECASE),
    ],

    'Embryonic/Developmental Stage': [  # New category for developmental stages
        re.compile(r'\b(?:embryo(?:nic)?|fetal|carnegie\s*stage|development|palatogenesis)\b', re.IGNORECASE),
        # Added development
    ],
    'Other Tissue/Organ': [  # Catch-all for other distinct organs/tissues not fitting specific categories
        re.compile(r'\b(?:spleen|gland|muscle|adrenal)\b', re.IGNORECASE),
    ],
    'Vegetal/otherNA': [
        re.compile(r'\b(?:plant|alga(?:e)?|peach|barley|rice|germinat|roots|lotus|tea)\b', re.IGNORECASE),
        # Added plant terms
        re.compile(r'\b(?:zebrafish|axolot|axolotl|toxoplasma|fish|drosophila|honeybees|elegans)\b', re.IGNORECASE),
        # pig|dog|cat|monkey|primate|rabbit|chicken|cow|sheep|bovine|porcine|canine|feline   these ones may have the tissue.
    ],
}

# 1.3. Disease Harmonization Map (this will populate your 'disease' column)
DISEASE_MAP = {
    'Alzheimer': [
        re.compile(r'\balzheimer(?:s)?\s*disease\b', re.IGNORECASE),
        re.compile(r'\bAD\b', re.IGNORECASE),
        re.compile(r'\balzheimer(?:s)?\b', re.IGNORECASE),  # More general to catch just "Alzheimer"
    ],
    'Parkinson': [
        re.compile(r'\bparkinson(?:s)?\s*disease\b', re.IGNORECASE),
        re.compile(r'\bparkinson(?:s)?\b', re.IGNORECASE),

    ],

    'Schizophrenia': [
        re.compile(r'\bschizophrenia(?:s)?\s*disease\b', re.IGNORECASE),
        re.compile(r'\bschizophrenia(?:s)?\b', re.IGNORECASE),
    ],

    'Cancer (General)': [  # More general cancer terms, placed after specific cancers
        re.compile(r'\bcancer\b', re.IGNORECASE),
        re.compile(r'\bcarcinoma\b', re.IGNORECASE),
        re.compile(r'\btumor(?:s)?\b', re.IGNORECASE),
        re.compile(r'\bglioblastoma\b', re.IGNORECASE),
        re.compile(r'\bGBM\b', re.IGNORECASE),
        re.compile(r'\bsarcoma\b', re.IGNORECASE),
        re.compile(r'\blymphoma\b', re.IGNORECASE),
        re.compile(r'\bleukemia\b', re.IGNORECASE),
        re.compile(r'\bmelanoma\b', re.IGNORECASE),
        re.compile(r'\bosteosarcoma\b', re.IGNORECASE),
        re.compile(r'\badenocarcinoma\b', re.IGNORECASE),
        re.compile(r'\bmetastasis\b', re.IGNORECASE),
        re.compile(r'\bmetastas\b', re.IGNORECASE),
        re.compile(r'\b(adenomas|adenoma|glioma|glioblastoma)\b', re.IGNORECASE),

        re.compile(r'\b(?:oncology|malignanc(?:y|ies))\b', re.IGNORECASE),  # Other cancer-related terms
    ],

    'Normal/Healthy': [
        re.compile(r'\bnormal\b', re.IGNORECASE),
        re.compile(r'\bhealthy\b', re.IGNORECASE),
        re.compile(r'\bcontrol\b', re.IGNORECASE),  # Often implies healthy control
    ],

    'Infectious Disease': [
        re.compile(r'\b(?:COVID-19|SARS-CoV-2|malaria|schistosomiasis|infection(?:s)?|viral|bacterial|fungal)\b',
                   re.IGNORECASE),
        # Add more specific infectious diseases
    ],

    'Inflammatory Disease': [
        re.compile(r'\binflammation\b', re.IGNORECASE),
        re.compile(r'\barthritis\b', re.IGNORECASE),
        re.compile(r'\bpsoriasis\b', re.IGNORECASE),
        re.compile(r'\b(?:colitis|Crohn\'s|ulcerative)\b', re.IGNORECASE),  # IBD
        # Add more specific inflammatory conditions
    ],

    'Autoimmune Disease': [
        re.compile(r'\b(?:autoimmune|lupus|MS|multiple\s*sclerosis)\b', re.IGNORECASE)
    ],

    'Neurological Disorder (General)': [  # Placed after Alzheimer's for specificity
        re.compile(r'\bALS\b', re.IGNORECASE),  # Amyotrophic Lateral Sclerosis
        re.compile(r'\bPD\b', re.IGNORECASE),
        re.compile(r'\bHuntington(?:s)?\s*disease\b', re.IGNORECASE),
        re.compile(r'\bneurodegen(?:erative)?\b', re.IGNORECASE),
        # Add more specific neurological disorders
    ],
    'Cardiovascular Disease': [
        re.compile(r'\bheart\b', re.IGNORECASE),
        re.compile(r'\bcardiac\b', re.IGNORECASE),
        re.compile(r'\bvascular\b', re.IGNORECASE),
        re.compile(r'\bischemia\b', re.IGNORECASE),
        re.compile(r'\b(?:myocardial\s*infarction|MI|stroke)\b', re.IGNORECASE)
        # Add more specific cardiovascular conditions
    ],
    'Metabolic Disorder': [
        re.compile(r'\b(?:metabolic|metabolism)\b', re.IGNORECASE),
        re.compile(r'\b(?:diabetes|T1D|T2D|obesity!diabetic)\b', re.IGNORECASE),
        # Add more specific metabolic disorders
    ],
    'Developmental Disorder': [
        re.compile(r'\b(?:developmental|autism|ASD)\b', re.IGNORECASE)
    ],
    'Wound Healing/Scarring': [
        re.compile(r'\b(?:wound\s*healing|scarring|keloid)\b', re.IGNORECASE)
    ],
    'Skin disoders': [
        re.compile(r'\b(?:acne|dermatitis)\b', re.IGNORECASE)
    ],

    #     'Aging': [
    #         re.compile(r'\b(?:aging|)\b', re.IGNORECASE)
    #     ],
    # Add more broad disease categories here

    'Vegetal/otherNA': [
        re.compile(r'\b(?:plant|alga(?:e)?|peach|barley|rice|germinat|roots|lotus|tea)\b', re.IGNORECASE),
        # Added plant terms
        re.compile(r'\b(?:zebrafish|axolot|axolotl|toxoplasma|fish|drosophila|honeybees|elegans)\b', re.IGNORECASE),
        # pig|dog|cat|monkey|primate|rabbit|chicken|cow|sheep|bovine|porcine|canine|feline   these ones may have the tissue.
    ],

}
ANIMAL_MAP = {
    'Human': [
        re.compile(r'\bhuman(?:s)?\b', re.IGNORECASE),
        re.compile(r'\bpatient(?:s)?\b', re.IGNORECASE),
        # Add more human-specific terms if needed
    ],
    'Rodent': [
        re.compile(r'\bmouse(?:s)?\b', re.IGNORECASE),
        re.compile(r'\bmice\b', re.IGNORECASE),
        re.compile(r'\brat(?:s)?\b', re.IGNORECASE),
        re.compile(r'\bmurine\b', re.IGNORECASE),
        re.compile(r'\brodent(?:s)?\b', re.IGNORECASE),
        re.compile(r'\bxenograft(?:s)?\b', re.IGNORECASE),  # Xenografts are typically in rodents
        # Add more rodent-specific terms if needed
    ],

    'Vegetal': [
        re.compile(r'\b(?:plant|alga(?:e)?|peach|barley|rice|germinat|roots|tea)\b', re.IGNORECASE),
        # Added plant terms
    ],
    'Other Animal': [
        re.compile(r'\b(?:zebrafish|axolot|toxoplasma|fish|drosophila|honeybees|elegans)\b', re.IGNORECASE),

        re.compile(r'\b(?:pig|dog|cat|monkey|primate|rabbit|chicken|cow|sheep|bovine|porcine|canine|feline)\b',
                   re.IGNORECASE),
    ],
    # Add other broad animal categories if necessary (e.g., 'Insect', 'Plant' if applicable)
}