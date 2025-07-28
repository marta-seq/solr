# src/llm_annotation/llm_annotation_lama.py
import pandas as pd
import json
import os
import argparse
import logging
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import tqdm for progress bars
from tqdm import tqdm

# Import utility functions
from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi
import ollama  # Import the ollama client

# --- Configuration (Default values, can be overridden by CLI args) ---
DEFAULT_INPUT_PAPERS_FILE = "data/intermediate/papers_with_extracted_sections.jsonl"  # Input from merge_extracted_sections
DEFAULT_OUTPUT_LLM_ANNOTATED_JSONL = "data/intermediate/llm_annotated_papers.jsonl"
DEFAULT_LOGGING_DIR = "data/logs/llm_annotation_logs/"

# --- Ollama Model Configuration ---
# IMPORTANT: Set this to the name of the model you have pulled and are running with Ollama
# e.g., "llama3", "mistral", "mixtral", "codellama"
OLLAMA_MODEL_NAME = "llama3"  # <--- CONFIGURE YOUR OLLAMA MODEL NAME HERE

# --- Logging Setup ---
logger = logging.getLogger(__name__)


# --- Helper Functions for LLM Prompt Generation ---

def truncate_section(text: Optional[str], max_chars: int) -> Optional[str]:
    """Truncates a string to max_chars, ensuring it's not None or empty after stripping."""
    if text is None:
        return None
    text = str(text).strip()  # Ensure it's a string and strip whitespace
    if not text:  # Check if empty after stripping
        return None
    return text[:max_chars]


def generate_llm_prompt(article_data: Dict[str, Any], llm_response_schema: Dict[str, Any]) -> str:
    """
    Generates a structured prompt for the LLM to extract detailed information
    from a scientific paper's metadata and extracted sections, with enhanced
    classification clarity.
    """
    title = article_data.get('title', 'N/A')
    abstract = article_data.get('abstract', 'N/A')

    # Define a maximum character limit for sections to manage token usage
    MAX_SECTION_CHARS_ABSTRACT = 2000
    MAX_SECTION_CHARS_SMALL = 1500
    MAX_SECTION_CHARS_LARGE = 4000

    # Apply truncation and cleaning *before* using the sections for checks or prompt construction
    code_availability_section = truncate_section(article_data.get('extracted_code_availability_section'),
                                                 MAX_SECTION_CHARS_SMALL)
    data_availability_section = truncate_section(article_data.get('extracted_data_availability_section'),
                                                 MAX_SECTION_CHARS_SMALL)
    results_discussion_section = truncate_section(article_data.get('extracted_results_discussion_section'),
                                                  MAX_SECTION_CHARS_LARGE)
    conclusion_section = truncate_section(article_data.get('extracted_conclusion_section'), MAX_SECTION_CHARS_SMALL)
    introduction_section = truncate_section(article_data.get('extracted_introduction_section'), MAX_SECTION_CHARS_LARGE)
    methods_section = truncate_section(article_data.get('extracted_methods_section'), MAX_SECTION_CHARS_LARGE)

    # Ensure abstract is also truncated if it's excessively long for the LLM
    abstract_truncated = truncate_section(abstract, MAX_SECTION_CHARS_ABSTRACT)

    # Flag to indicate if sections were available after cleaning
    sections_available = any(s is not None and s.strip() != "" for s in [
        code_availability_section, data_availability_section,
        results_discussion_section, conclusion_section,
        introduction_section, methods_section
    ])
    llm_notes_on_sections_warning = ""
    if not sections_available:
        llm_notes_on_sections_warning = "Warning: Full text sections were not available or were empty after processing. Annotation is based solely on Title and Abstract, which may lead to incomplete or incorrect results."

    # Build the core content for the LLM
    text_content = f"Title: {title}\n"
    text_content += f"Abstract: {abstract_truncated}\n\n"

    # Only add sections if they exist and are not empty after truncation/cleaning
    if introduction_section:
        text_content += f"--- Introduction Section ---\n{introduction_section}\n\n"
    if methods_section:
        text_content += f"--- Methods Section ---\n{methods_section}\n\n"
    if results_discussion_section:
        text_content += f"--- Results and Discussion Section ---\n{results_discussion_section}\n\n"
    if conclusion_section:
        text_content += f"--- Conclusion Section ---\n{conclusion_section}\n\n"
    if data_availability_section:
        text_content += f"--- Data Availability Section ---\n{data_availability_section}\n\n"
    if code_availability_section:
        text_content += f"--- Code Availability Section ---\n{code_availability_section}\n\n"

    # Add the note about missing sections to the prompt itself, so LLM is aware
    if llm_notes_on_sections_warning:
        text_content += f"\n\n**Important Note to LLM:** {llm_notes_on_sections_warning}\n"

    # --- REVISED PROMPT STRUCTURE ---
    # The prompt is now very direct about the output format.
    # The system message in call_llm_api will handle the persona.
    # The schema is explicitly provided.
    prompt = f"""
    Analyze the following scientific paper content and extract specific, structured information.

    ---
    **Paper Content for Analysis:**

    {text_content}

    ---

    **Instructions for JSON Output:**
    * **STRICTLY adhere to the JSON schema provided below.**
    * **Your response MUST be ONLY the JSON object, with no other text, preambles, or conversational filler.**
    * If a field is not applicable or information is not found, use `null` for single values or an empty array `[]` for list fields, unless the schema explicitly requires an item (e.g., `data_modalities` in `data_used_details`).
    * For `enum` fields, select values strictly from the provided lists. If "Other" is chosen, explain briefly in `llm_notes`.

    * **Detailed Instructions for 'classification' fields:**
        * `is_computational_methods_paper`: `True` if the paper's *core contribution* is developing/improving a computational method. `False` if it primarily *applies* methods, is a review, or describes a lab protocol.
        * `reasoning_for_classification`: 1-2 sentences explaining `is_computational_methods_paper` decision.
        * `primary_application_domain`: Main biological domain/data type. Choose from enum.
        * `assigned_categories`: Select ALL applicable categories from the enum. If `is_computational_methods_paper` is `True`, choose a "Computational Methods for..." category. If lab protocol, choose "Technical Protocol/Method (Lab-based)". If applying existing tools, choose "Application Paper". If summarizing, choose "Review".

    * **CRUCIAL CONDITIONAL LOGIC:**
        * If `is_computational_methods_paper` is **`false`**, then the following fields **MUST be empty arrays `[]` or `null`**:
            * `package_algorithm_name` (should be `null`)
            * `algorithms_bases_used` (should be `[]`)
            * `pipeline_analysis_steps` (should be `[]`)
            * `tested_data_modalities` (should be `[]`)
            * `tested_assay_types_platforms` (should be `[]`)
            * `compared_algorithms_packages` (should be `[]`)
            * `code_availability_details` (should have `status: "Not Applicable"`, `link: null`, `license: null`, `version: null`)
            * `data_used_details` (should be `[]`)
        * This ensures fields relevant only to computational methods are not populated for other paper types.

    **JSON Schema (Your output must strictly conform to this):**
    {json.dumps(llm_response_schema, indent=2)}

    """
    return prompt


def call_llm_api(article_data: Dict[str, Any], llm_response_schema: Dict[str, Any], retries_per_model: int = 3,
                 delay_between_retries: int = 1) -> str:
    """Calls the local Ollama LLM to generate content."""
    article_doi = article_data.get('doi', 'N/A')
    model_to_use = OLLAMA_MODEL_NAME

    if model_to_use is None:
        logger.error(f"Ollama model name is not set. Skipping annotation for DOI {article_doi}.")
        return ""

    prompt = generate_llm_prompt(article_data, llm_response_schema)

    for attempt in range(1, retries_per_model + 1):
        try:
            logger.info(f"Attempt {attempt}/{retries_per_model} for DOI {article_doi} using '{model_to_use}'")

            response = ollama.chat(
                model=model_to_use,
                messages=[
                    # SYSTEM MESSAGE: Keep it purely about the persona, NO output instructions here.
                    {"role": "system",
                     "content": "You are an expert bioinformatician and machine learning researcher specializing in spatial omics data analysis."},
                    # USER MESSAGE: Contains all content and STRICT output instructions.
                    {"role": "user", "content": prompt}
                ],
                format='json'  # Crucial Ollama-specific parameter for JSON output
            )

            llm_output_text = response['message']['content']

            try:
                llm_output = json.loads(llm_output_text)
                return llm_output_text

            except json.JSONDecodeError:
                # Robust cleaning: try to find the JSON block if wrapped in markdown
                cleaned_response_text = llm_output_text.strip()
                if cleaned_response_text.startswith("```json"):
                    cleaned_response_text = cleaned_response_text[len("```json"):].strip()
                if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-len("```")].strip()
                elif cleaned_response_text.startswith("```"):  # In case it's just ``` followed by text
                    cleaned_response_text = cleaned_response_text[len("```"):].strip()

                llm_output = json.loads(cleaned_response_text)
                logger.warning(
                    f"DOI {doi}: Ollama response required cleaning. Cleaned response (first 200 chars): {cleaned_response_text[:200]}...")
                return cleaned_response_text

        except json.JSONDecodeError as e:
            logger.error(
                f"JSON decoding error for DOI {doi} (Attempt {attempt}/{retries_per_model}): {e}. Raw response (first 200 chars): {llm_output_text[:200] if 'llm_output_text' in locals() else 'N/A'}...",
                exc_info=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
            else:
                logger.critical(f"DOI {doi}: Max retries reached for JSON parsing error.")
                return ""

        except ollama.ResponseError as e:
            logger.error(
                f"Ollama response error for DOI {doi}: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}",
                exc_info=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
            else:
                logger.critical(f"DOI {doi}: Max retries reached for Ollama response error.")
                return ""
        except Exception as e:
            llm_status = 'llm_error_api'  # Ensure status is set for general errors
            llm_error_message = f"General API call failed for DOI {doi}: {e}"
            logger.error(f"DOI {doi}: {llm_error_message}", exc_info=True)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
            else:
                logger.critical(f"DOI {doi}: Max retries reached for general API call error.")
                return ""

    logger.error(
        f"Failed to get a valid response from Ollama for DOI {article_doi} after all attempts. Skipping annotation.")
    return ""


def run_llm_annotation(input_papers_file: str, output_llm_annotated_jsonl: str, force_reannotate_all: bool = False):
    """
    Main function to orchestrate loading data, performing LLM annotation, and saving results.
    Args:
        input_papers_file: Path to the input JSONL file of papers with extracted sections.
        output_llm_annotated_jsonl: Path to the output JSONL file for LLM annotations.
        force_reannotate_all: If True, re-annotates all papers, ignoring existing 'middle_llm_annotated' status.
    """
    # Setup logging
    log_dir_path = DEFAULT_LOGGING_DIR
    setup_logging(log_dir=log_dir_path, log_prefix="llm_annotation_log")

    logger.info("--- Starting LLM Annotation Process ---")
    logger.info(f"Input papers file: {input_papers_file}")
    logger.info(f"Output LLM annotated file: {output_llm_annotated_jsonl}")
    logger.info(f"Force reannotate all: {force_reannotate_all}")
    logger.info(f"Using Ollama model: {OLLAMA_MODEL_NAME}")

    # Define the JSON schema for the LLM output (same as in generate_llm_prompt)
    # This is passed to the API call for structured response
    llm_response_schema = {
        "type": "object",
        "properties": {
            "main_goal_of_paper": {
                "type": "string",
                "description": "A concise summary (1-3 sentences) of the primary problem the paper addresses and its proposed solution or contribution. Focus on the core aim."
            },
            "package_algorithm_name": {
                "type": ["string", "null"],
                "description": "The primary name of the developed package, algorithm, or tool (e.g., 'SpatialDE', 'CellRanger', 'SSAM', 'ComSeg'). If multiple are developed, prioritize the most central one. Can be null if the paper does not introduce a named method or tool."
            },
            "contribution_type": {
                "type": "array",
                "description": "Classification of the primary type(s) of contribution this paper offers. Select all applicable. If 'Other', explain in 'llm_notes'.",
                "items": {
                    "type": "string",
                    "enum": [
                        "Framework", "Tool", "Pipeline", "Algorithm", "Software",
                        "Database", "Benchmark Study", "Methodology", "Library", "Protocol",
                        "Review", "Application", "Other"
                    ]
                }
            },
            "classification": {
                "type": "object",
                "description": "Categorization of the paper based on its primary focus regarding computational methods and spatial omics.",
                "properties": {
                    "is_computational_methods_paper": {
                        "type": "boolean",
                        "description": "True if the paper's core contribution is the *development or significant improvement* of a computational method (e.g., new algorithm, software, analytical pipeline). False if it primarily *applies* existing methods, is a review, or describes a lab protocol."
                    },
                    "reasoning_for_classification": {
                        "type": "string",
                        "description": "Brief reasoning (1-2 sentences) for the 'is_computational_methods_paper' decision. Explain *why* it is or isn't a computational methods paper."
                    },
                    "primary_application_domain": {
                        "type": "string",
                        "enum": [
                            "Spatial Omics",
                            "Other Biomedical Imaging",
                            "General Bioinformatics/Computational Biology",
                            "Other Biological Domain",
                            "Not Applicable (e.g., Review, Protocol)"
                        ],
                        "description": "The primary biological domain or data type the paper's method (if any) is applied to or developed for. 'Spatial Omics' includes technologies like Visium, MERFISH, CosMx, Xenium, IMC, MIBI, Slide-seq, Stereo-seq, GeoMx DSP. 'Other Biomedical Imaging' includes general microscopy, MRI, CT scans, etc., not focused on spatially resolved molecular data. 'General Bioinformatics/Computational Biology' refers to methods not tied to specific imaging."
                    },
                    "assigned_categories": {
                        "type": "array",
                        "description": "Select ALL categories from the provided list that accurately describe the paper's main type(s). Prioritize the most specific and accurate category. If `is_computational_methods_paper` is True, choose either 'Computational Methods for Spatial Omics' or 'Computational Methods for Other Imaging/Biology' based on `primary_application_domain`. If it's a lab-based protocol, choose 'Technical Protocol/Method (Lab-based)'.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "Computational Methods for Spatial Omics",
                                "Computational Methods for Other Imaging/Biology",
                                "Computational Methods (General Bioinformatics)",
                                "Application Paper (Spatial Omics)",
                                "Application Paper (Other Imaging/Biology)",
                                "Application Paper (General Biological)",
                                "Review (Computational Methods)",
                                "Review (Application)",
                                "Review (General Scientific)",
                                "Benchmarking Study (Computational)",
                                "Technical Protocol/Method (Lab-based)",
                                "Database/Resource Paper",
                                "Other"
                            ]
                        }
                    }
                },
                "required": ["is_computational_methods_paper", "reasoning_for_classification",
                             "primary_application_domain", "assigned_categories"]
            },
            "algorithms_bases_used": {
                "type": "array",
                "description": "If a new computational method is introduced, list the foundational algorithms or mathematical concepts it explicitly builds upon or heavily utilizes. Each item should have 'name' and optionally 'description_how_used'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of a foundational algorithm/mathematical concept (e.g., 'Principal Component Analysis (PCA)', 'UMAP', 'k-means clustering', 'Support Vector Machine (SVM)', 'Graph Neural Network', 'Non-negative Matrix Factorization', 'Deep Learning', 'Convolutional Neural Networks')."
                        },
                        "description_how_used": {
                            "type": ["string", "null"],
                            "description": "Brief description (1-2 sentences) of how this base is applied, adapted, or integrated into the new method."
                        }
                    },
                    "required": ["name"]
                }
            },
            "pipeline_analysis_steps": {
                "type": "array",
                "description": "If this paper introduces or significantly contributes to a computational method, list the main computational analysis steps it addresses within a spatial transcriptomics/bioinformatics pipeline. Select from predefined categories or suggest new ones in 'llm_notes' if none fit perfectly.",
                "items": {
                    "type": "string",
                    "enum": [
                        "Preprocessing", "Normalization", "Dimensionality Reduction", "Cell Segmentation",
                        "Segmentation-free Cell Identification", "Cell Type Deconvolution",
                        "Cell Phenotyping", "Spatial Variable Gene Identification",
                        "Spatial Domain Identification", "Sample Integration/Batch Correction",
                        "Differential Expression Analysis (Spatial)", "Data Visualization",
                        "Image Processing", "Clustering", "Network Analysis", "Modeling",
                        "Benchmarking", "Trajectory Inference", "Cell-Cell Interaction Analysis",
                        "De-noising", "Data Imputation", "Other"
                    ]
                }
            },
            "tested_data_modalities": {
                "type": "array",
                "description": "List of broad data modalities that the method is designed for or explicitly tested with (e.g., 'Transcriptomics', 'Proteomics', 'Metabolomics', 'Genomics', 'Epigenomics', 'Imaging').",
                "items": {
                    "type": "string",
                    "enum": [
                        "Transcriptomics", "Proteomics", "Metabolomics", "Genomics",
                        "Epigenomics", "Imaging", "Histology", "Clinical Data", "Morphology", "Other"
                    ]
                }
            },
            "tested_assay_types_platforms": {
                "type": "array",
                "description": "List of specific spatial omics assay types or platforms the method is designed for or explicitly tested with. This is more granular than data modality (e.g., 'Visium', 'MERFISH', 'CosMx', 'Xenium', 'IMC', 'MIBI', 'MALDI-MSI', 'Slide-seq', 'Stereo-seq', 'NanoString GeoMx DSP', 'seqFISH', 'seqFISH+'). If 'Other', provide the name from the text.",
                "items": {"type": "string"}
            },
            "compared_algorithms_packages": {
                "type": "array",
                "description": "List of other algorithms or packages that this paper explicitly compares its performance against. Each item should have 'name' and optionally 'doi_pmid_of_paper' and 'comparison_details'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the compared algorithm/package (e.g., 'Seurat', 'Scanpy', 'SpaGCN', 'STAGATE')."
                        },
                        "doi_pmid_of_paper": {
                            "type": ["string", "null"],
                            "description": "DOI or PMID of the paper introducing this comparison target, if available (e.g., '10.xxxx/yyyy' or 'PMID:12345678'). Prioritize DOI."
                        },
                        "comparison_details": {
                            "type": ["string", "null"],
                            "description": "Brief description (1-2 sentences) of the comparison context or key findings regarding this comparison (e.g., 'compared for accuracy on simulated data', 'benchmarked against existing tools for scalability', 'showed improved runtime on large datasets')."
                        }
                    },
                    "required": ["name"]
                }
            },
            "code_availability_details": {
                "type": "object",
                "description": "Details about the availability of the code for the method.",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["Available", "Not Available", "Partially Available", "Upon Request", "Unclear",
                                 "Not Applicable"],
                        "description": "Overall status of code availability. Choose 'Available' only if a direct link is provided and appears functional. 'Not Applicable' if the paper is not a computational methods paper."
                    },
                    "link": {
                        "type": ["string", "null"],
                        "description": "The direct URL(s) of the code repository or download link(s), if provided (e.g., GitHub, Zenodo, GitLab). If multiple, provide a comma-separated string."
                    },
                    "license": {
                        "type": ["string", "null"],
                        "description": "Software license if mentioned (e.g., 'MIT', 'GPL', 'Apache 2.0', 'BSD')."
                    },
                    "version": {
                        "type": ["string", "null"],
                        "description": "Software version if mentioned (e.g., 'v1.0', '0.5.2', 'v2.1.3')."
                    }
                },
                "required": ["status"]
            },
            "data_used_details": {
                "type": "array",
                "description": "List of specific datasets used for demonstration, validation, or benchmarking. Extract as much detail as possible, focusing on publicly available data. If no specific datasets are mentioned, return an empty array.",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Official or described name of the dataset (e.g., 'MERFISH mouse brain dataset', 'TCGA-BRCA cohort', 'Human Lung Atlas'). If no specific name, describe (e.g., 'Simulated spatial transcriptomics data', 'Internal patient samples')."
                        },
                        "accession_numbers": {
                            "type": "array",
                            "description": "List of associated accession numbers, if available (e.g., ['GSE12345', 'SRP000000', 'EGAS0000100xxxx']). Look for common prefixes (GSE, GEO, SRA, E-MTAB, PRJNA, PXD, ENA, EGA, ProteomeXchange, etc.). If none, return an empty array.",
                            "items": {"type": "string"}
                        },
                        "source_database_or_platform": {
                            "type": ["string", "null"],
                            "description": "The public database/repository where the dataset is hosted, or the platform it originated from (e.g., 'GEO', 'SRA', 'ArrayExpress', 'dbGaP', 'TCGA', 'Zenodo', 'Figshare', '10x Genomics Xenium')."
                        },
                        "sample_info": {
                            "type": "object",
                            "description": "Detailed information about the biological samples in this dataset. Provide 'null' for fields not found.",
                            "properties": {
                                "species": {"type": ["string", "null"],
                                            "description": "e.g., 'Homo sapiens' (Human), 'Mus musculus' (Mouse), 'Drosophila melanogaster', 'Danio rerio' (Zebrafish)."},
                                "tissue": {"type": ["string", "null"],
                                           "description": "e.g., 'Brain', 'Lung', 'Tumor', 'Kidney cortex', 'Heart tissue', 'Spleen', 'Whole embryo'."},
                                "cell_type": {"type": ["string", "null"],
                                              "description": "Specific cell types if mentioned (e.g., 'Neurons', 'Macrophages', 'T-cells', 'Fibroblasts', 'Cardiomyocytes', 'Glia')."},
                                "disease_state": {"type": ["string", "null"],
                                                  "description": "e.g., 'Healthy', 'Lung Adenocarcinoma', 'Alzheimer\'s Disease', 'Type 2 Diabetes', 'Colorectal Cancer', 'Inflammation'."},
                                "assay_type": {"type": ["string", "null"],
                                               "description": "The experimental assay used for this dataset (e.g., 'Visium', 'MERFISH', 'scRNA-seq', 'bulk RNA-seq', 'IMC', 'seqFISH')."},
                                "other_conditions": {"type": ["string", "null"],
                                                     "description": "Any other relevant experimental conditions or characteristics (e.g., 'drug treatment A', 'time series day 7', 'untreated control', 'simulated')."}
                            },
                            "required": []
                        },
                        "data_modalities": {
                            "type": "array",
                            "description": "List of specific data modalities present in this dataset (e.g., 'spatial transcriptomics', 'single-cell RNA-seq', 'bulk RNA-seq', 'proteomics', 'imaging', 'methylation', 'ATAC-seq'). Always provide at least one modality.",
                            "items": {"type": "string"}
                        },
                        "dataset_size_or_scale": {
                            "type": ["string", "null"],
                            "description": "Brief description of the dataset's size or scale (e.g., '10X Genomics Xenium - 2 samples, ~100k cells/sample', 'Visium - 5 sections, 30k spots each', 'single-cell RNA-seq - 50k cells', '10 human patients')."
                        }
                    },
                    "required": ["dataset_name", "data_modalities"]
                }
            },
            "llm_notes": {
                "type": ["string", "null"],
                "description": "Any additional relevant notes, clarifications, or ambiguities encountered by the LLM during analysis. This includes warnings if full text was not available or if specific fields were difficult to extract. Also use this to explain 'Other' categories if chosen."
            }
        },
        "required": ["main_goal_of_paper", "contribution_type", "classification"]
    }

    # 1. Load papers with extracted sections
    df_papers = load_jsonl_to_dataframe(input_papers_file)
    if df_papers.empty:
        logger.error(f"Input papers file '{input_papers_file}' is empty or does not exist. Exiting.")
        ensure_dir(os.path.dirname(output_llm_annotated_jsonl))
        save_jsonl_records([], output_llm_annotated_jsonl, append=False)  # Create empty output file
        return
    df_papers['doi'] = df_papers['doi'].apply(clean_doi)
    # Ensure unique DOIs for processing, keep first in case of duplicates
    df_papers.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_papers)} unique papers from {input_papers_file}.")

    # 2. Load existing annotated data to determine what to skip (for incremental runs)
    df_existing_output = load_jsonl_to_dataframe(output_llm_annotated_jsonl)
    existing_annotated_dois = set()
    if not df_existing_output.empty:
        if 'doi' in df_existing_output.columns:
            df_existing_output['doi'] = df_existing_output['doi'].apply(clean_doi)
        if 'llm_annotation_status' in df_existing_output.columns:
            # Only consider 'middle_llm_annotated' as successfully processed for skipping
            existing_annotated_dois = set(
                df_existing_output[
                    df_existing_output['llm_annotation_status'] == 'middle_llm_annotated'
                    ]['doi'].tolist()
            )
        logger.info(
            f"Found {len(existing_annotated_dois)} DOIs already 'middle_llm_annotated' in '{output_llm_annotated_jsonl}'.")

    # Filter papers to process: NOW CONSIDERING ALL PAPERS FOR LLM ANNOTATION
    papers_to_process_df = df_papers.copy()

    initial_llm_process_count = len(papers_to_process_df)
    logger.info(f"Initial count of ALL papers identified for LLM processing: {initial_llm_process_count}")

    if not force_reannotate_all and not papers_to_process_df.empty:
        papers_to_process_df = papers_to_process_df[~papers_to_process_df['doi'].isin(existing_annotated_dois)].copy()
        logger.info(
            f"Filtered to {len(papers_to_process_df)} papers not yet 'middle_llm_annotated' (or forced re-annotation is off).")

    if papers_to_process_df.empty:
        logger.info("No new papers found to process with LLM. Exiting.")
        # Ensure output file exists, even if empty, for Snakemake's sake
        ensure_dir(os.path.dirname(output_llm_annotated_jsonl))
        if not os.path.exists(output_llm_annotated_jsonl) or os.path.getsize(output_llm_annotated_jsonl) == 0:
            save_jsonl_records([], output_llm_annotated_jsonl, append=False)
        return

    papers_to_process_list = papers_to_process_df.to_dict(orient='records')
    logger.info(f"Starting LLM annotation for {len(papers_to_process_list)} papers.")

    # Ensure output directory exists for the LLM annotations
    ensure_dir(os.path.dirname(output_llm_annotated_jsonl))

    # --- LLM API Call Loop ---
    for i, article_data in enumerate(tqdm(papers_to_process_list, desc="LLM Annotating Papers")):
        doi = article_data.get('doi')
        if not doi:
            logger.warning(f"Skipping paper due to missing DOI: {article_data.get('title', 'N/A')}")
            article_data['llm_annotation_status'] = 'skipped_no_doi'
            article_data['llm_annotation_timestamp'] = datetime.now().isoformat()
            article_data['llm_model_used'] = OLLAMA_MODEL_NAME
            save_jsonl_records([article_data], output_llm_annotated_jsonl, append=True)
            continue

        llm_output_text = None
        llm_output = None
        llm_status = 'llm_error_general'
        llm_error_message = None

        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                # Pass the schema to the prompt generator
                prompt_text = generate_llm_prompt(article_data, llm_response_schema)

                response = ollama.chat(
                    model=OLLAMA_MODEL_NAME,
                    messages=[
                        # SYSTEM MESSAGE: Keep it purely about the persona, NO output instructions here.
                        {"role": "system",
                         "content": "You are an expert bioinformatician and machine learning researcher specializing in spatial omics data analysis. Your ONLY task is to output a JSON object according to the provided schema. Do NOT include any other text, preambles, or conversational filler."},
                        # USER MESSAGE: Contains all content and STRICT output instructions.
                        {"role": "user", "content": prompt_text}
                    ],
                    format='json'  # Crucial Ollama-specific parameter for JSON output
                )

                llm_output_text = response['message']['content']

                try:
                    llm_output = json.loads(llm_output_text)
                    llm_status = 'middle_llm_annotated'
                    logger.info(f"Successfully annotated DOI: {doi}")
                    break

                except json.JSONDecodeError:
                    # Robust cleaning: try to find the JSON block if wrapped in markdown
                    cleaned_response_text = llm_output_text.strip()
                    if cleaned_response_text.startswith("```json"):
                        cleaned_response_text = cleaned_response_text[len("```json"):].strip()
                    if cleaned_response_text.endswith("```"):
                        cleaned_response_text = cleaned_response_text[:-len("```")].strip()
                    elif cleaned_response_text.startswith("```"):
                        cleaned_response_text = cleaned_response_text[len("```"):].strip()

                    llm_output = json.loads(cleaned_response_text)
                    llm_status = 'middle_llm_annotated'
                    logger.warning(
                        f"DOI {doi}: Ollama response required cleaning. Cleaned response (first 200 chars): {cleaned_response_text[:200]}...")
                    break

            except json.JSONDecodeError as e:
                llm_status = 'llm_error_json_parse'
                llm_error_message = f"LLM returned malformed JSON for DOI {doi}: {e}. Raw response (first 200 chars): {llm_output_text[:200] if llm_output_text else 'N/A'}..."
                logger.error(f"DOI {doi}: {llm_error_message}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
                else:
                    logger.critical(f"DOI {doi}: Max retries reached for JSON parsing error.")

            except ollama.ResponseError as e:
                llm_status = 'llm_error_ollama_response'
                llm_error_message = f"Ollama response error for DOI {doi}: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}"
                logger.error(f"DOI {doi}: {llm_error_message}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
                else:
                    logger.critical(f"DOI {doi}: Max retries reached for Ollama response error.")
            except Exception as e:
                llm_status = 'llm_error_api'
                llm_error_message = f"General API call failed for DOI {doi}: {e}"
                logger.error(f"DOI {doi}: {llm_error_message}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
                else:
                    logger.critical(f"DOI {doi}: Max retries reached for general API call error.")

        # Merge LLM output into the article data
        if llm_output:
            for key, value in llm_output.items():
                article_data[f'llm_annot_{key}'] = value

        article_data['llm_annotation_status'] = llm_status
        article_data['llm_annotation_timestamp'] = datetime.now().isoformat()
        article_data['llm_model_used'] = OLLAMA_MODEL_NAME
        if llm_error_message:
            article_data['llm_annotation_error'] = llm_error_message
        else:
            article_data['llm_annotation_error'] = None

        # Append the annotated paper to the output file immediately
        save_jsonl_records([article_data], output_llm_annotated_jsonl, append=True)

    logger.info("--- LLM Annotation Process Complete. ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs LLM-based annotation of scientific papers using Ollama."
    )
    parser.add_argument(
        "--input_papers_file",
        type=str,
        default=DEFAULT_INPUT_PAPERS_FILE,
        help=f"Path to the input JSONL file of papers with extracted sections (default: {DEFAULT_INPUT_PAPERS_FILE})."
    )
    parser.add_argument(
        "--output_llm_annotated_jsonl",
        type=str,
        default=DEFAULT_OUTPUT_LLM_ANNOTATED_JSONL,
        help=f"Path to the output JSONL file for LLM annotated papers (default: {DEFAULT_OUTPUT_LLM_ANNOTATED_JSONL})."
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="If set, re-annotates all papers, ignoring existing 'middle_llm_annotated' status."
    )

    args = parser.parse_args()

    run_llm_annotation(
        input_papers_file=args.input_papers_file,
        output_llm_annotated_jsonl=args.output_llm_annotated_jsonl,
        force_reannotate_all=args.force_reannotate
    )