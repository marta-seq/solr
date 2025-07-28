import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Set, Union # Import Union
import logging
from tqdm import tqdm
import re # For DOI cleaning
import ollama # Import the ollama client


# --- Configuration ---
# Input files
ORIGINAL_METHODS_FILE = "../../data/methods_papers.jsonl" # Your file with basic metadata
EXTRACTED_SECTIONS_LOG_FILE = "../../data/extracted_sections_log.jsonl" # Your file with extracted sections and logs
SLEEP_TIME_BETWEEN_LLM_REQUESTS = 1
# Output file for LLM-annotated data
OUTPUT_LLM_ANNOTATED_JSONL = "../../data/llm_annotated_METHODS_llama.jsonl"

LOG_DIR = "../../data/llm_annotation_logs"

# --- Logging Setup ---
def ensure_dir(path: str):
    """Ensures a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

ensure_dir(LOG_DIR) # Make sure log directory exists
log_filename = datetime.now().strftime(f"{LOG_DIR}/llm_annotation_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplicate logs if re-running in same session
if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- DOI Cleaning Helper ---
def clean_doi(doi: str) -> str:
    """Cleans a DOI string to a standard format."""
    if pd.isna(doi) or not isinstance(doi, str):
        return ""
    doi = doi.lower().strip()
    # Remove common URL prefixes
    doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
    # Remove trailing slashes or other non-DOI characters
    doi = doi.split(';')[0].split(',')[0].split(' ')[0] # take first part if multiple are present
    return doi

# --- LLM Configuration & Dynamic Switching ---
# New global variable for the Ollama model name you want to use
OLLAMA_MODEL_NAME = "llama3" # Or "mistral", "mixtral", etc. - the one you're downloading

def configure_llm():
    """Configures the Ollama client and sets the initial LLM model."""
    global LLM_MODEL_INSTANCE # Renaming this variable to be model-agnostic, or just use OLLAMA_MODEL_NAME directly in call_llm_api
    global OLLAMA_MODEL_NAME # Use this to store your chosen Ollama model

    logger.info(f"Attempting to connect to Ollama and use model: {OLLAMA_MODEL_NAME}")
    try:
        # Ollama doesn't need an API key or specific configuration here,
        # it connects to the local server by default.
        # We'll just check if the model is available.
        # This will raise an exception if the model isn't downloaded or server isn't running.
        ollama.list() # Simple call to check if Ollama server is responsive
        # You can add a more specific check if a model is downloaded:
        # available_local_models = [m['name'] for m in ollama.list()['models']]
        # if f"{OLLAMA_MODEL_NAME}:latest" not in available_local_models and OLLAMA_MODEL_NAME not in available_local_models:
        #    logger.error(f"Model '{OLLAMA_MODEL_NAME}' not found locally. Please run 'ollama run {OLLAMA_MODEL_NAME}' first.")
        #    LLM_MODEL_INSTANCE = None # Indicate failure to configure
        #    return

        # For Ollama, LLM_MODEL_INSTANCE doesn't need to be a specific object like Gemini's.
        # We just need the model name.
        LLM_MODEL_INSTANCE = OLLAMA_MODEL_NAME # Store the string name
        logger.info(f"Successfully configured for Ollama model: {OLLAMA_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Error connecting to Ollama or checking models: {e}. Ensure Ollama server is running and model '{OLLAMA_MODEL_NAME}' is downloaded.", exc_info=True)
        LLM_MODEL_INSTANCE = None

def call_llm_api(article_data: Dict[str, Any], retries_per_model: int = 3, delay_between_retries: int = 1) -> str:
    """Calls the local Ollama LLM to generate content."""
    article_doi = article_data.get('doi', 'N/A')
    model_to_use = OLLAMA_MODEL_NAME # Use the global OLLAMA_MODEL_NAME directly

    if model_to_use is None: # Check if configure_llm failed
        logger.error(f"Ollama model not configured. Skipping annotation for DOI {article_doi}.")
        return ""

    prompt = generate_llm_prompt(article_data)

    for attempt in range(1, retries_per_model + 1):
        try:
            logger.info(f"Attempt {attempt}/{retries_per_model} for DOI {article_doi} using '{model_to_use}'")

            # Ollama API call
            # Use ollama.chat for conversational models, or ollama.generate for simpler text completion.
            # Given your structured output, ollama.chat with a system role might be more robust.
            response = ollama.chat(
                model=model_to_use,
                messages=[
                    # A system message can help guide the model's output format
                    {"role": "system", "content": "You are an expert bioinformatician and machine learning researcher "
                                                  "specializing in spatial omics data analysis."
                                                  " Your task is to extract structured information into a JSON object."
                                                  " Provide ONLY the JSON object, with no conversational text."},
                    {"role": "user", "content": prompt}
                ],
                # Ollama doesn't have a direct equivalent for response_mime_type like Gemini
                # You'll rely on the prompt to force JSON output
                format='json' # This is a crucial Ollama-specific parameter for JSON output
            )

            # Extract the content from the Ollama response
            response_text = response['message']['content']

            if response_text and response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                return response_text
            else:
                # Attempt to clean the response if it's wrapped in markdown
                cleaned_response_text = response_text.strip()
                if cleaned_response_text.startswith("```json"):
                    cleaned_response_text = cleaned_response_text[len("```json"):].strip()
                if cleaned_response_text.endswith("```"):
                    cleaned_response_text = cleaned_response_text[:-len("```")].strip()
                elif cleaned_response_text.startswith("```"): # In case it's just ``` followed by text
                     cleaned_response_text = cleaned_response_text[len("```"):].strip()

                try:
                    json.loads(cleaned_response_text)
                    return cleaned_response_text
                except json.JSONDecodeError:
                    logger.warning(f"Ollama did not return direct JSON and cleaning failed for DOI {article_doi}. Raw response (first 200 chars): {response_text[:200]}...")
                    # If cleaning fails, it's a critical issue for structured output
                    raise ValueError("Ollama response is not valid JSON after cleaning.")

        except json.JSONDecodeError as e:
            # Re-raise JSONDecodeError, it's now handled outside this try block but can occur from cleaning
            logger.error(f"JSON decoding error for DOI {article_doi} (Attempt {attempt}/{retries_per_model}): {e}")
            if attempt < retries_per_model:
                time.sleep(delay_between_retries) # Still good to have a small delay for stability
            else:
                logger.error(f"Max retries ({retries_per_model}) reached for DOI {article_doi} due to JSON decoding issues.")
                return "" # Failed after retries

        except Exception as e:
            # Catch all other exceptions (e.g., connection errors to Ollama server, model not found)
            logger.error(
                f"Ollama API call failed (Attempt {attempt}/{retries_per_model}) for DOI {article_doi}: {e}")
            if attempt < retries_per_model:
                time.sleep(delay_between_retries)
            else:
                logger.error(
                    f"Max retries ({retries_per_model}) reached for DOI {article_doi} due to persistent error.")
                return ""

    logger.error(f"Failed to get a valid response from Ollama for DOI {article_doi} after all attempts. Skipping annotation.")
    return ""

# --- Helper to truncate and clean section text ---
def truncate_section(section_text: Union[str, float, None], max_chars: int) -> Union[str, None]:
    """
    Truncates a section to max_chars, handles None/NaN/empty strings.
    Returns None if the section is effectively empty after cleaning/truncation.
    """
    if section_text is None or pd.isna(section_text):
        return None
    section_text = str(section_text).strip() # Ensure it's a string and strip whitespace
    if not section_text: # Handle empty strings after stripping
        return None
    return section_text[:max_chars] if len(section_text) > max_chars else section_text


def generate_llm_prompt(article_data: Dict[str, Any]) -> str:
    """
    Generates a structured prompt for the LLM to extract detailed information
    from a scientific paper's metadata and extracted sections, with enhanced
    classification clarity.
    """
    title = article_data.get('doi_pool_title', article_data.get('weak_annot_title', 'N/A'))
    abstract = article_data.get('doi_pool_abstract', 'N/A')

    # Define a maximum character limit for sections to manage token usage
    # Adjust these values based on your Llama model's actual context window
    # and the typical length of your sections. Consider that prompt instructions
    # and schema also consume tokens.
    MAX_SECTION_CHARS_ABSTRACT = 2000  # Abstract is critical, give it more space
    MAX_SECTION_CHARS_SMALL = 1500  # For shorter sections like availability, conclusion
    MAX_SECTION_CHARS_LARGE = 4000  # For longer sections like methods, intro, results

    # Apply truncation and cleaning *before* using the sections for checks or prompt construction
    # Use 'None' as a default if the key is missing to ensure consistency with `is not None` checks
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
    text_content += f"Abstract: {abstract_truncated}\n\n"  # Use the truncated abstract

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

    # Define the desired JSON schema for the LLM output with updated lists
    json_schema = {
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
                                "Computational Methods (General Bioinformatics)",  # Added for clarity
                                "Application Paper (Spatial Omics)",
                                "Application Paper (Other Imaging/Biology)",
                                "Application Paper (General Biological)",  # Added for clarity
                                "Review (Computational Methods)",
                                "Review (Application)",
                                "Review (General Scientific)",
                                "Benchmarking Study (Computational)",
                                "Technical Protocol/Method (Lab-based)",  # For wet-lab protocols etc.
                                "Database/Resource Paper",  # For papers presenting a new database or data resource
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
                        "Benchmarking", "Trajectory Inference", "Cell-Cell Interaction Analysis",  # Added more
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
                        "enum": ["Available", "Not Available", "Partially Available", "Upon Request", "Unclear"],
                        "description": "Overall status of code availability. Choose 'Available' only if a direct link is provided and appears functional."
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
                            "required": []  # Making all sub-fields optional initially
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

    prompt = f"""
    You are an expert bioinformatician and machine learning researcher specializing in spatial omics data analysis.
    Your task is to meticulously analyze the provided scientific paper content and extract specific, structured information into a JSON object.

    ---
    **Paper Content for Analysis:**

    Title: {title}

    Abstract:
    {abstract_truncated}
    """
    # Conditionally add sections to the prompt only if they exist and are not empty
    if introduction_section:
        prompt += f"""
    --- Introduction Section ---
    {introduction_section}
    """
    if methods_section:
        prompt += f"""
    --- Methods Section ---
    {methods_section}
    """
    if results_discussion_section:
        prompt += f"""
    --- Results and Discussion Section ---
    {results_discussion_section}
    """
    if conclusion_section:
        prompt += f"""
    --- Conclusion Section ---
    {conclusion_section}
    """
    if data_availability_section:
        prompt += f"""
    --- Data Availability Section ---
    {data_availability_section}
    """
    if code_availability_section:
        prompt += f"""
    --- Code Availability Section ---
    {code_availability_section}
    """

    # Add the note about missing sections to the prompt itself, so LLM is aware
    if llm_notes_on_sections_warning:
        prompt += f"\n\n**Important Note to LLM:** {llm_notes_on_sections_warning}\n"

    prompt += f"""
    ---

    **Instructions for Output:**
    * Provide your output as a **single JSON object**, strictly adhering to the JSON schema provided below.
    * **Do not include any preamble, postamble, or conversational text outside the JSON object.**
    * If a field is not applicable or information is not found in the provided text, use `null` for single values (string, boolean, object) or an empty array `[]` for list fields, unless the schema explicitly states otherwise (e.g., `data_modalities` in `data_used_details` must have at least one item).
    * For `enum` fields, you **must** select one or more values strictly from the provided lists. If none fit perfectly, select "Other" and provide a brief explanation in the `llm_notes` field.

    * **Detailed Instructions for 'classification' fields:**
        * `is_computational_methods_paper`: Set to `True` if the paper's *primary contribution* is the development or a significant methodological improvement of a computational method (e.g., a new algorithm, software, or an analytical pipeline). Set to `False` if the paper primarily *applies* existing computational methods, is a review of existing literature, or describes a laboratory-based (wet-lab) protocol.
        * `reasoning_for_classification`: Briefly explain your decision for `is_computational_methods_paper` (1-2 sentences).
        * `primary_application_domain`: Identify the main biological domain or data type that the paper's computational method (if applicable) or overall research is developed for or applied to.
            * Choose **'Spatial Omics'** if the paper primarily deals with spatially resolved molecular data generated by technologies like Visium, MERFISH, CosMx, Xenium, IMC, MIBI, Slide-seq, Stereo-seq, NanoString GeoMx DSP, seqFISH, etc. This is your primary target.
            * Choose **'Other Biomedical Imaging'** if it involves computational methods for general imaging techniques (e.g., microscopy, MRI, CT scans) that are *not* specifically focused on spatially resolved *molecular* data.
            * Choose **'General Bioinformatics/Computational Biology'** if the methods are broadly applicable to biological data (e.g., bulk RNA-seq, single-cell RNA-seq analysis, genomics, proteomics) and are *not* tied to imaging, or if the imaging aspect is secondary.
            * Choose **'Other Biological Domain'** for papers focused on other specific biological areas (e.g., pure clinical studies, ecology, evolutionary biology) where computational methods are applied but don't fit the above.
            * Choose **'Not Applicable (e.g., Review, Protocol)'** if the paper type is a review, a wet-lab protocol, or a database description, where a "primary application domain" for a *method* isn't relevant.
        * `assigned_categories`: Select **ALL** applicable categories.
            * If `is_computational_methods_paper` is `True`, you **must** select one of the "Computational Methods for..." categories that aligns with `primary_application_domain`.
            * If it's a lab-based protocol (e.g., describing a new staining procedure for spatial omics), select "Technical Protocol/Method (Lab-based)".
            * If it primarily applies existing computational tools to a biological question, select an "Application Paper" category.
            * If it summarizes existing research, select a "Review" category.

    **JSON Schema:**
    {json.dumps(json_schema, indent=2)}

    """
    return prompt

# --- Data Loading Helper ---
def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a JSONL file into a pandas DataFrame."""
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_json(file_path, lines=True)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        else:
            logger.warning(f"File not found or empty: {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# --- Main Annotation Logic ---
def run_llm_annotation(force_reannotate_all: bool = False):
    """
    Main function to orchestrate loading data, performing LLM annotation, and saving results.
    Args:
        force_reannotate_all: If True, re-annotates all papers, ignoring existing 'middle_llm_annotated' status.
    """
    logger.info("--- Starting LLM Annotation Process ---")

    # 1. Load original methods papers (metadata only)
    df_methods = load_jsonl_to_dataframe(ORIGINAL_METHODS_FILE)
    if df_methods.empty:
        logger.error(f"Original methods file '{ORIGINAL_METHODS_FILE}' is empty or does not exist. Exiting.")
        return
    df_methods['doi'] = df_methods['doi'].apply(clean_doi)
    df_methods.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_methods)} unique papers from {ORIGINAL_METHODS_FILE}.")


    # 2. Load extracted sections log and select best attempts
    df_extracted_log = load_jsonl_to_dataframe(EXTRACTED_SECTIONS_LOG_FILE)
    df_best_sections = pd.DataFrame() # Initialize empty
    if not df_extracted_log.empty:
        df_extracted_log['doi'] = df_extracted_log['doi'].apply(clean_doi)

        # Ensure 'extracted_at_timestamp' is datetime and has a default for older entries
        if 'extracted_at_timestamp' in df_extracted_log.columns:
            df_extracted_log['extracted_at_timestamp'] = pd.to_datetime(
                df_extracted_log['extracted_at_timestamp'], errors='coerce'
            ).fillna(pd.Timestamp('1970-01-01')) # Fill NaT with a very old timestamp
        else:
            df_extracted_log['extracted_at_timestamp'] = pd.Timestamp('1970-01-01') # Add if missing

        # Define status rank for selection
        status_rank_map = {
            'extracted from pdf (sections found)': 4,
            'extracted from html (sections found)': 3,
            'pdf processed but no relevant sections found': 2,
            'html processed (no sections found by heuristics)': 1,
            # Lower ranks for access statuses or errors, so higher ranks (successful extraction) are prioritized
            'oa_html_landing_page': 0.5,
            'oa_pdf_link': 0.5,
            'oa_pdf_fallback': 0.5,
            'oa but no direct location info from unpaywall': 0,
            'closed access or not found in unpaywall': 0,
            'unpaywall/network error': -1,
            'html fetch error': -1,
            'pdf download error': -1,
            'general error': -1,
            'initial: not processed': -2,
            'serialization error': -3
        }
        df_extracted_log['status_rank'] = df_extracted_log['section_extraction_status'].str.lower().map(status_rank_map).fillna(-4)

        # Group by DOI, then select the row with the max status_rank.
        # If ties in status_rank, take the most recent extraction.
        idx = df_extracted_log.loc[df_extracted_log.groupby('doi')['status_rank'].idxmax()].index
        df_best_attempts_status = df_extracted_log.loc[idx]

        idx_latest_timestamp = df_best_attempts_status.loc[df_best_attempts_status.groupby('doi')['extracted_at_timestamp'].idxmax()].index
        df_best_sections = df_extracted_log.loc[idx_latest_timestamp]

        # Select only the section columns and DOI for merging
        section_cols = [col for col in df_best_sections.columns if col.startswith('extracted_')] + ['doi']
        df_best_sections = df_best_sections[section_cols].copy()

        logger.info(f"Selected best extraction attempts for {len(df_best_sections)} DOIs from section log.")
    else:
        logger.warning(f"Extracted sections log file '{EXTRACTED_SECTIONS_LOG_FILE}' is empty or does not exist. No sections will be merged.")

    # 3. Merge original papers with their best extracted sections
    df_to_process = pd.merge(df_methods, df_best_sections, on='doi', how='left', suffixes=('_original', '_extracted'))
    logger.info(f"Merged original papers with extracted sections. Total papers for processing: {len(df_to_process)}.")

    # Identify papers that explicitly should be processed by LLM (e.g., 'method' type)
    if 'paper_type' not in df_to_process.columns:
        logger.warning("No 'paper_type' column found in input data. Processing all papers as if they are 'method' papers.")
        papers_to_process_df = df_to_process.copy()
    else:
        papers_to_process_df = df_to_process[df_to_process['paper_type'] == 'method'].copy()
    initial_method_papers_count = len(papers_to_process_df)
    logger.info(f"Initial count of 'method' papers from merged data: {initial_method_papers_count}")


    # 4. Load existing annotated data to determine what to skip
    df_existing_output = load_jsonl_to_dataframe(OUTPUT_LLM_ANNOTATED_JSONL)
    existing_annotated_dois = set()
    if not df_existing_output.empty:
        if 'doi' in df_existing_output.columns:
             df_existing_output['doi'] = df_existing_output['doi'].apply(clean_doi)
        if 'llm_annotation_status' in df_existing_output.columns: # Use the correct column for status
            existing_annotated_dois = set(
                df_existing_output[
                    df_existing_output['llm_annotation_status'] == 'middle_llm_annotated'
                ]['doi'].tolist()
            )
        logger.info(f"Found {len(existing_annotated_dois)} DOIs already 'middle_llm_annotated' in '{OUTPUT_LLM_ANNOTATED_JSONL}'.")

    if not force_reannotate_all and not papers_to_process_df.empty:
        papers_to_process_df = papers_to_process_df[~papers_to_process_df['doi'].isin(existing_annotated_dois)]
        logger.info(f"Filtered to {len(papers_to_process_df)} method papers not yet 'middle_llm_annotated'.")

    if papers_to_process_df.empty:
        logger.info("No new 'method' papers found to process with LLM. Exiting.")
        return

    papers_to_process_list = papers_to_process_df.to_dict(orient='records')

    # 5. Configure LLM
    configure_llm()
    if LLM_MODEL_INSTANCE is None:
        logger.critical("LLM not configured. Cannot proceed with annotation.")
        return

    logger.info(f"Starting LLM annotation for {len(papers_to_process_list)} papers...")

    # Open the output file in append mode.
    with open(OUTPUT_LLM_ANNOTATED_JSONL, 'a', encoding='utf-8') as outfile:
        for i, paper_data in tqdm(enumerate(papers_to_process_list),
                                 total=len(papers_to_process_list),
                                 desc="Annotating DOIs"):
            doi = paper_data.get('doi', 'N/A')
            logger.info(f"Processing DOI: {doi}")

            # Create a copy of the original paper_data (with merged sections)
            # This ensures we retain all original and merged fields for the output record
            output_record = paper_data.copy()
            output_record['llm_annotation_status'] = 'llm_not_processed' # Use 'llm_annotation_status' to distinguish
            output_record['llm_annotation_error'] = None
            output_record['llm_raw_response'] = None
            output_record['llm_model_used'] = None
            output_record['llm_annotation_date'] = None

            try:
                llm_response_text = call_llm_api(paper_data)

                if llm_response_text:
                    try:
                        llm_parsed_data = json.loads(llm_response_text)

                        for key, value in llm_parsed_data.items():
                            output_record[f'llm_annot_{key}'] = value

                        output_record['llm_annotation_status'] = 'middle_llm_annotated'
                        # output_record['llm_model_used'] = LLM_MODEL_INSTANCE.model_name
                        output_record['llm_model_used'] = OLLAMA_MODEL_NAME # This will now store the string name, e.g., "llama3"
                        output_record['llm_annotation_date'] = datetime.now().isoformat()

                        logger.info(f"Successfully annotated DOI {doi} using {OLLAMA_MODEL_NAME}")

                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to decode JSON from LLM response for DOI {doi}: {e}. Raw response (first 500 chars): {llm_response_text[:500]}..."
                        logger.error(error_msg)
                        output_record['llm_annotation_error'] = f"JSONDecodeError: {e}"
                        output_record['llm_raw_response'] = llm_response_text
                        output_record['llm_annotation_status'] = 'llm_processing_failed_json_decode'
                    except Exception as e:
                        error_msg = f"Unexpected error processing LLM response for DOI {doi}: {e}"
                        logger.exception(error_msg)
                        output_record['llm_annotation_error'] = f"Processing Error: {e}"
                        output_record['llm_raw_response'] = llm_response_text
                        output_record['llm_annotation_status'] = 'llm_processing_failed_internal_error'
                else:
                    logger.warning(f"LLM returned empty response for DOI {doi}. Skipping annotation for this paper.")
                    output_record['llm_annotation_error'] = "Empty LLM response"
                    output_record['llm_annotation_status'] = 'llm_processing_failed_empty_response'

            except Exception as e:
                logger.critical(f"Critical error during LLM call for DOI {doi}: {e}", exc_info=True)
                output_record['llm_annotation_error'] = f"Critical LLM call error: {e}"
                output_record['llm_annotation_status'] = 'llm_api_call_failed'

            try:
                # Ensure all values are JSON serializable. Convert sets to lists if any snuck in,
                # and convert Timestamp objects to ISO format strings.
                def convert_non_json_serializable(obj):
                    if isinstance(obj, set):
                        return list(obj)
                    if isinstance(obj, pd.Timestamp): # Added for Timestamp conversion
                        return obj.isoformat()
                    if isinstance(obj, datetime): # Also handle standard datetime objects
                        return obj.isoformat()
                    if isinstance(obj, dict):
                        return {k: convert_non_json_serializable(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [convert_non_json_serializable(elem) for elem in obj]
                    return obj

                serializable_output_record = convert_non_json_serializable(output_record)

                outfile.write(json.dumps(serializable_output_record, ensure_ascii=False) + '\n')
                outfile.flush()
                logger.debug(f"Record for DOI {doi} saved to {OUTPUT_LLM_ANNOTATED_JSONL}")
            except TypeError as e:
                logger.error(f"Failed to serialize final record for DOI {doi}: {e}. Data (first 500 chars): {str(output_record)[:500]}...")
            except Exception as e:
                logger.error(f"Unexpected error while writing record for DOI {doi}: {e}")

            # time.sleep(SLEEP_TIME_BETWEEN_LLM_REQUESTS)

    logger.info("--- LLM Annotation Process Complete ---")
    logger.info(f"Annotated data saved to: {OUTPUT_LLM_ANNOTATED_JSONL}")

    df_final_output = load_jsonl_to_dataframe(OUTPUT_LLM_ANNOTATED_JSONL)
    successfully_annotated = df_final_output[df_final_output['llm_annotation_status'] == 'middle_llm_annotated']
    logger.info(f"Total papers in final output: {len(df_final_output)}")
    logger.info(f"Successfully LLM-annotated papers: {len(successfully_annotated)}")

def analyze_llm_classifications():
    logger.info("--- Starting LLM Classification Analysis ---")

    df_annotated = load_jsonl_to_dataframe(OUTPUT_LLM_ANNOTATED_JSONL)
    if df_annotated.empty:
        logger.warning(f"No annotated data found in '{OUTPUT_LLM_ANNOTATED_JSONL}'. Exiting analysis.")
        return

    # Filter for successfully annotated papers
    df_successfully_annotated = df_annotated[
        df_annotated['llm_annotation_status'] == 'middle_llm_annotated'
    ].copy()

    if df_successfully_annotated.empty:
        logger.info("No successfully LLM-annotated papers to analyze. Exiting.")
        return

    logger.info(f"Analyzing {len(df_successfully_annotated)} successfully annotated papers.")

    # Access the classification details
    # The 'llm_annot_classification' column will contain dictionaries (parsed JSON)
    # We need to be careful if it's missing or malformed for some rows.

    # 1. Papers not classified as computational methods
    non_computational_papers = df_successfully_annotated[
        df_successfully_annotated['llm_annot_classification'].apply(
            lambda x: x.get('is_computational_methods_paper') == False if isinstance(x, dict) else False
        )
    ].copy()

    logger.info(f"Number of papers flagged as NON-COMPUTATIONAL METHODS: {len(non_computational_papers)}")

    # 2. Extract relevant columns for these papers
    if not non_computational_papers.empty:
        # Define the columns you want in the output CSV
        output_cols = [
            'doi',
            'title', # Assuming 'title' is also in your comprehensive JSONL
            'llm_annot_main_goal_of_paper',
            'llm_annot_classification', # Keep the full dict for inspection
            'llm_annot_llm_notes',
            # Add other relevant original or LLM-annotated columns as needed
        ]

        # Ensure all output_cols exist before selecting
        cols_to_select = [col for col in output_cols if col in non_computational_papers.columns]
        df_non_computational_output = non_computational_papers[cols_to_select]

        # Extract primary_application_domain and assigned_categories for display/analysis
        df_non_computational_output['primary_domain'] = df_non_computational_output['llm_annot_classification'].apply(
            lambda x: x.get('primary_application_domain') if isinstance(x, dict) else None
        )
        df_non_computational_output['assigned_categories'] = df_non_computational_output['llm_annot_classification'].apply(
            lambda x: ', '.join(x.get('assigned_categories', [])) if isinstance(x, dict) and x.get('assigned_categories') else None
        )
        df_non_computational_output['classification_reasoning'] = df_non_computational_output['llm_annot_classification'].apply(
            lambda x: x.get('reasoning_for_classification') if isinstance(x, dict) else None
        )


        # Drop the raw 'llm_annot_classification' column if you only want the flattened ones
        df_non_computational_output = df_non_computational_output.drop(columns=['llm_annot_classification'])


        output_csv_path = 'non_computational_papers_summary.csv'
        df_non_computational_output.to_csv(output_csv_path, index=False)
        logger.info(f"Details of non-computational papers saved to: {output_csv_path}")

        # You can also print a summary to the console
        logger.info("\n--- Summary of Non-Computational Papers (DOI, Title, Primary Domain, Categories) ---")
        for idx, row in df_non_computational_output.head(10).iterrows(): # Print first 10 for quick check
            logger.info(f"DOI: {row.get('doi', 'N/A')}")
            logger.info(f"Title: {row.get('title', 'N/A')}")
            logger.info(f"Primary Domain: {row.get('primary_domain', 'N/A')}")
            logger.info(f"Categories: {row.get('assigned_categories', 'N/A')}")
            logger.info("-" * 50)
        if len(df_non_computational_output) > 10:
            logger.info(f"...and {len(df_non_computational_output) - 10} more. See '{output_csv_path}' for full list.")

    else:
        logger.info("No papers classified as non-computational methods found.")

    logger.info("--- LLM Classification Analysis Complete ---")
# Example of how to run this script:
if __name__ == "__main__":
    run_llm_annotation(force_reannotate_all=False) # Set to True to re-run all, False to skip already annotated
    analyze_llm_classifications()