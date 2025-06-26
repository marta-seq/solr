import google.generativeai as genai
import json
import time
import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple
import google.api_core.exceptions as exceptions
# --- Script Description ---
"""
This script performs "weak annotation" and relevance classification on scientific
articles stored in a 'doi_pool.jsonl' file. It leverages a Large Language Model (LLM)
(specifically Google's Gemini Pro via Google AI Studio) to extract nuanced information
and determine paper relevance.

Key functionalities include:
1.  **Loading Existing Data:** Reads articles from 'doi_pool.jsonl' (which should
    already contain basic metadata like title, abstract, authors, etc., populated
    by a previous scraping step). It also tracks which DOIs have already been
    processed for weak annotation from 'weak_annotation_pool.jsonl'.
2.  **Relevance Classification:** For each article, an LLM is queried to determine
    if the paper is relevant to "spatial omics methodology". This status updates
    the 'relevance_score' field in 'doi_pool.jsonl'.
3.  **Detailed Weak Annotation:** The LLM extracts detailed information for all
    relevant papers, or for "method," "method comparison," and "review" papers.
    This includes:
    * Refined paper types (method, application, review, method_comparison)
    * Biological context (disease, organism, tissue) for application papers
    * Specific data modalities used (HE, Xenium, IMC, etc.)
    * Type of review (methods review, disease-specific review, etc.)
    * Methodology type (pipeline, algorithm, workflow), covered analysis steps,
        package name, and code availability (with link if mentioned).
4.  **Selective Processing:** By default, it only processes DOIs not already present
    in 'weak_annotation_pool.jsonl' or those marked as 'irrelevant' (which are skipped).
    An optional flag allows re-processing all articles.
5.  **Output:** All extracted weak annotations and relevance status are saved to
    'weak_annotation_pool.jsonl'. The 'relevance_score' in the main 'doi_pool.jsonl'
    is updated in place.

Requires a Google Gemini API key, ideally set as an environment variable (GOOGLE_API_KEY).
Leverages Google AI Studio's free tier, subject to rate limits.
"""

# --- Configuration ---
# You need to set this environment variable for Gemini.
# On Linux/macOS: export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
# On Windows (PowerShell): $env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
# On Windows (CMD): set GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# Base directory for data files (assuming it's relative to where this script is run)
BASE_DIR = "../../data"
LOGGING_DIR = os.path.join(BASE_DIR, "logging_weak_annotator")
DOI_POOL_FILE = os.path.join(BASE_DIR, "doi_pool.jsonl")
WEAK_ANNOTATION_POOL_FILE = os.path.join(BASE_DIR, "weak_annotation_pool.jsonl")

# Set a reasonable sleep time to avoid hitting LLM rate limits
# Adjust based on your usage and Gemini's free tier limits
SLEEP_TIME_BETWEEN_LLM_REQUESTS = 5  # seconds

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    """Sets up logging to console and a file with date/time in filename."""
    ensure_dir(LOGGING_DIR)  # Ensure logging directory exists

    # Create file handler with a unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = os.path.join(LOGGING_DIR, f"weak_annotation_log_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Remove any existing handlers to prevent duplicate output if setup_logging is called multiple times
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized for Weak Annotator.")
    logger.info(f"Log file: {log_file_name}")


def ensure_dir(path: str):
    """Ensures a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


# --- LLM Configuration & Dynamic Switching ---
GEMINI_API_KEY = None
LLM_MODEL_INSTANCE = None  # Renamed for clarity to distinguish from model names
EXHAUSTED_MODELS_TODAY = set()  # Stores model names that have hit ResourceExhausted for current run

# Ordered preference for models (try these first)
PREFERRED_MODEL_KEYWORDS_ORDER = [
    "gemini-2.5-flash",  # Typically higher free tier RPD
    "gemini-2.0-flash",  # Also potentially good RPD
    "gemini-1.5-flash-latest",  # Your previous model (50 RPD limit hit)
    "gemini-1.5-pro-latest",  # More capable, but often lower RPD free tier
    "gemini-pro"  # Older, kept as a very last resort
]
ALL_AVAILABLE_GEMINI_MODELS = []  # Populated by configure_llm with full model names


def switch_to_next_available_model() -> bool:
    """
    Attempts to switch the global LLM_MODEL_INSTANCE to the next available model
    from PREFERRED_MODEL_KEYWORDS_ORDER that hasn't been exhausted yet.
    Returns True if a switch was successful, False otherwise.
    """
    global LLM_MODEL_INSTANCE, EXHAUSTED_MODELS_TODAY, ALL_AVAILABLE_GEMINI_MODELS

    current_model_name = LLM_MODEL_INSTANCE.model_name if LLM_MODEL_INSTANCE else "None"
    logger.info(f"Attempting to switch from current model: {current_model_name}")

    for preferred_keyword in PREFERRED_MODEL_KEYWORDS_ORDER:
        for m_info in ALL_AVAILABLE_GEMINI_MODELS:
            full_model_name = m_info.name
            if preferred_keyword in full_model_name and full_model_name not in EXHAUSTED_MODELS_TODAY:
                try:
                    # Attempt to load this model
                    LLM_MODEL_INSTANCE = genai.GenerativeModel(full_model_name)
                    logger.info(f"Successfully switched LLM model to: {LLM_MODEL_INSTANCE.model_name}")
                    return True  # Successfully found and set a new model
                except Exception as e:
                    logger.warning(
                        f"Could not load or use model '{full_model_name}' (was listed as available): {e}. Adding to exhausted list for this session.")
                    EXHAUSTED_MODELS_TODAY.add(full_model_name)  # Treat as exhausted if we can't load/use it
                    # Continue to next model in ALL_AVAILABLE_GEMINI_MODELS for this preferred_keyword
        # If we finished iterating through ALL_AVAILABLE_GEMINI_MODELS for a preferred_keyword
        # and didn't find one that wasn't exhausted, move to the next preferred_keyword.

    LLM_MODEL_INSTANCE = None  # No suitable model found
    logger.critical(
        "No more suitable LLM models available. All preferred models are exhausted or inaccessible for the current session.")
    return False


def configure_llm():
    """
    Configures the Gemini API and sets the initial LLM model.
    Populates ALL_AVAILABLE_GEMINI_MODELS and attempts to set LLM_MODEL_INSTANCE.
    """
    global GEMINI_API_KEY, LLM_MODEL_INSTANCE, ALL_AVAILABLE_GEMINI_MODELS

    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set. Gemini API calls will fail.")
        return

    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Get all models that support 'generateContent'
        ALL_AVAILABLE_GEMINI_MODELS = [m for m in genai.list_models() if
                                       'generateContent' in m.supported_generation_methods]
        logger.info("Available Gemini models supporting 'generateContent':")
        if not ALL_AVAILABLE_GEMINI_MODELS:
            logger.error("No Gemini models found that support 'generateContent'. LLM annotation will be skipped.")
            return

        for m_info in ALL_AVAILABLE_GEMINI_MODELS:
            logger.info(f"  - {m_info.name}")

        # Attempt to set the initial model using the switching logic
        if not switch_to_next_available_model():
            logger.error("Failed to configure any initial LLM model from preferred list.")

    except Exception as e:
        logger.error(f"Error during initial LLM configuration or model listing: {e}", exc_info=True)
        LLM_MODEL_INSTANCE = None

def call_llm_api(article_data: Dict[str, Any], retries_per_model: int = 3, delay_between_retries: int = 5) -> str:
    """
    Calls the LLM API to generate content. Handles model switching on quota exhaustion and other persistent errors.
    Args:
        article_data: Dictionary containing paper title, abstract, etc.
        retries_per_model: Number of retries for non-quota errors before trying next model (if any).
        delay_between_retries: Sleep time between retries.
    Returns:
        The LLM's response text, or an empty string if all models fail.
    """
    global LLM_MODEL_INSTANCE, EXHAUSTED_MODELS_TODAY

    article_doi = article_data.get('doi', 'N/A')

    # Loop to try different models if the current one gets exhausted or errors
    while LLM_MODEL_INSTANCE:
        prompt = generate_llm_prompt(article_data)  # Generate prompt using the article data

        for attempt in range(1, retries_per_model + 1):
            try:
                logger.info(
                    f"Attempt {attempt}/{retries_per_model} for DOI {article_doi} using '{LLM_MODEL_INSTANCE.model_name}'")
                response = LLM_MODEL_INSTANCE.generate_content(prompt)
                return response.text  # Success! Return the response

            except exceptions.ResourceExhausted as e:
                # Quota hit for the current model. Mark it and try to switch.
                logger.error(f"Quota exhausted for model '{LLM_MODEL_INSTANCE.model_name}' on DOI {article_doi}: {e}")
                EXHAUSTED_MODELS_TODAY.add(LLM_MODEL_INSTANCE.model_name)

                if not switch_to_next_available_model():
                    logger.critical(f"All available LLM models exhausted. Stopping further LLM attempts for this run.")
                    return ""  # No more models to try

                # If switched successfully, break from inner retry loop and
                # the outer while loop will naturally re-attempt with the new model
                break  # Break from the inner 'retries_per_model' loop to try the new model immediately

            except Exception as e:
                # Catch all other errors (like 404 Deprecated, API errors, etc.)
                logger.error(
                    f"LLM API call failed with '{LLM_MODEL_INSTANCE.model_name}' (Attempt {attempt}/{retries_per_model}) for DOI {article_doi}: {e}")
                if attempt < retries_per_model:
                    time.sleep(delay_between_retries)  # Short delay for transient errors
                else:
                    # Max retries reached for *this specific model* due to a persistent error.
                    # Treat it like an exhausted model for this session and try to switch.
                    logger.error(
                        f"Max retries ({retries_per_model}) reached for model '{LLM_MODEL_INSTANCE.model_name}' due to persistent error. Adding to exhausted list and attempting to switch.")
                    EXHAUSTED_MODELS_TODAY.add(LLM_MODEL_INSTANCE.model_name) # Mark as unusable for this run
                    if not switch_to_next_available_model():
                        logger.critical(f"All available LLM models exhausted. Stopping further LLM attempts for this run.")
                        return ""  # No more models to try

                    break # Break from inner loop to retry with new model

    logger.error(f"No active LLM model available for DOI {article_doi}. Skipping annotation.")
    return ""

def generate_llm_prompt(article_data: Dict[str, Any]) -> str:
    """
    Generates a structured prompt for the LLM to categorize and annotate an article.
    """
    title = article_data.get('title', 'N/A')
    abstract = article_data.get('abstract', 'N/A')
    mesh_terms = ", ".join(article_data.get('mesh_terms', []))
    author_keywords = ", ".join(article_data.get('author_keywords', []))
    authors = ", ".join(article_data.get('authors', []))
    journal = article_data.get('journal', 'N/A')
    year = article_data.get('year', 'N/A')

    prompt = f"""
    Analyze the following scientific paper and provide the requested information in a JSON format.
    Ensure the JSON is well-formed and can be directly parsed. Do not include any text outside the JSON block.

    Paper Title: {title}
    Abstract: {abstract}
    Authors: {authors}
    Journal: {journal}
    Year: {year}
    MeSH Terms: {mesh_terms}
    Author Keywords: {author_keywords}

    ---
    JSON Output Structure:
    {{
        // Definition of paper types:
        // "method": Primarily introduces a novel computational method, algorithm, software, or pipeline for spatial omics data analysis, or significantly improves existing ones. Focuses on the technical development and validation of the tool.
        // "application": Primarily uses existing spatial omics methods/tools to study a specific biological question or system, presenting new biological insights rather than methodological innovations.
        // "review": A comprehensive overview of existing literature, trends, or methods in the field.
        // "method_comparison": The primary focus is to compare the performance or characteristics of multiple existing spatial omics methods or tools. This paper type prioritizes comparison over novel method development or biological application.
        // "other": Papers that do not fit the above categories.
        "paper_type": "method" | "application" | "review" | "method_comparison" | "other", // Primary classification

        // LLM's assessment of relevance to spatial omics *methodology*.
        // "highly_relevant": Directly develops or significantly advances core spatial omics computational methods or provides a deep comparative analysis of them.
        // "relevant": Applies or extends spatial omics methods in a meaningful way, or is a foundational paper.
        // "low_relevance": Tangentially related, or an application paper that uses standard tools without new insights on methodology.
        // "irrelevant": Not related to spatial omics methodology or completely out of scope for this project.
        "relevance_score_llm": "highly_relevant" | "relevant" | "low_relevance" | "irrelevant",

        // These fields are applicable to ALL paper types, describing the biological context and data used.
        "biological_context": string, // E.g., 'Pancreatic cancer', 'Mouse brain development', 'Alzheimer\'s disease', 'Healthy human kidney', 'B. subtilis cells'. Return "N/A" if not applicable or not specified.
        "data_modalities_used": list of strings, // Specific spatial or relevant omics data modalities employed in the study (e.g., 'HE-stained images', 'Xenium', 'IMC', 'MIBI', 'IF', 'Visium', 'Slide-seq', 'MERFISH', 'DLPFC', 'spatial transcriptomics', 'spatial proteomics', 'snRNA-seq', 'smFISH', 'ExSeq', 'sequencing', 'light-sheet microscopy', 'histology images', 'multiplexed imaging', 'electron microscopy'). List all applicable modalities that were *used in the study*.

        "weak_annotations": {{ // This block contains more specific annotations, often for method, method_comparison, or review papers.
            "is_method_paper": boolean, // True if this paper primarily develops or introduces a new computational methodology, algorithm, or software (should align with paper_type="method").
            "is_method_comparison_paper": boolean, // True if the paper's main focus is to compare multiple spatial omics methods (should align with paper_type="method_comparison").

            // For "review" papers only:
            "review_type": "methods_review" | "technology_review" | "disease_specific_review" | "application_review" | "general_review" | "N/A", // Type of review if paper_type is "review".
            // Definitions for review_type:
            // "methods_review": Focuses on surveying and evaluating computational methods for spatial omics.
            // "technology_review": Focuses on surveying and evaluating spatial omics technologies (e.g., Xenium, Visium, MIBI).
            // "disease_specific_review": Focuses on spatial omics applications or methods in a particular disease (e.g., spatial omics in cancer, neurodegeneration).
            // "application_review": Focuses on surveying the applications of spatial omics in general biological contexts.
            // "general_review": A broad overview of spatial omics without a specific narrow focus.

            // Methodology details (if applicable, primarily for method/method_comparison papers, or deep reviews):
            // The primary type of methodological contribution if it's a method or method_comparison paper.
            // "pipeline": A multi-step, integrated set of computational tools for complex analysis.
            // "algorithm": A specific, novel computational procedure/mathematical model to solve a problem (e.g., for clustering, dimension reduction, data integration, segmentation).
            // "workflow": A sequence of recommended steps, often using existing tools, for a specific analysis goal, with a focus on best practices.
            // "framework": A foundational structure, conceptual model, or common platform for developing specific methods or tools.
            // "tool": A standalone software, package, or script designed for a specific task.
            // "other_methodology": A method that doesn't fit the above categories.
            "methodology_type": "pipeline" | "algorithm" | "workflow" | "framework" | "tool" | "other_methodology" | "N/A",

            // List common spatial omics analysis steps that the paper's method/tool primarily covers or enables.
            // Only include if explicitly a focus of the paper or a core functionality of the method.
            "covered_analysis_steps": list of strings, // E.g., ['Preprocessing', 'Normalization', 'Denoising', 'Batch Correction', 'Cell Segmentation', 'Cell-Cell Interaction Analysis', 'Spatial Domain Identification', 'Spatial Gene Expression Analysis', 'Trajectory Inference', 'Deconvolution', 'Clustering', 'Dimension Reduction', 'Visualization', 'Integration with other omics'].

            "pipeline_package_name": string, // The official name of the software/pipeline/package (e.g., "Cellpose", "Baysor", "SpatialDE", "squidpy"). Return "N/A" if not applicable or not explicitly named.

            "code_availability": {{
                "status": "GitHub link provided" | "Upon request" | "Not mentioned" | "Available on CRAN/Bioconductor" | "Proprietary" | "Other", // Status of code availability
                "link_in_abstract_or_title": string // The exact URL (e.g., GitHub, Zenodo, project website) if explicitly mentioned in the provided text (abstract/title/keywords). Return "N/A" if not found.
            }},

            "main_goals": string, // Briefly describe the main goals/contributions of the paper (max 3 sentences).

            "data_links_mentioned": list of strings // Any external data links (DOIs, URLs to datasets) explicitly mentioned in the provided text (abstract/title/keywords). If none, return empty list.
        }}
    }}

    If any boolean is not clearly indicated by the text, default to false. For strings, if not applicable or not found, use "N/A". For lists, return an empty list if no items are found.
    Output only the JSON.
    """
    return prompt


def parse_llm_response(response_text: str, doi: str) -> Dict[str, Any]:
    """
    Parses the JSON response from the LLM.
    Includes robust error handling for malformed JSON.
    """
    try:
        json_match = re.search(r'```json\n({.*})\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text.strip()
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse LLM JSON response for DOI {doi}: {e}. Response snippet: {response_text[:500]}...",
            exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM response for DOI {doi}: {e}", exc_info=True)
        return {}


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads all entries from a .jsonl file."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:  # Added encoding
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} at line {line_num}: {line.strip()}")
    return data


def write_jsonl_file(file_path: str, data: List[Dict[str, Any]]):
    """Writes a list of dictionaries to a .jsonl file (overwrites if exists)."""
    with open(file_path, 'w', encoding='utf-8') as f:  # Added encoding
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    logger.info(f"Successfully wrote {len(data)} entries to {file_path}.")


def process_articles_for_weak_annotation(
        force_reannotate_all: bool = False
):
    """
    Processes articles from the DOI pool for weak annotation using an LLM.
    Skips irrelevant papers and already annotated papers by default.
    Updates 'relevance_score' in the main doi_pool.jsonl.
    Each successful weak annotation is immediately written to weak_annotation_pool.jsonl.
    """
    logger.info(f"\n--- Starting weak annotation process (Force re-annotate: {force_reannotate_all}) ---")

    # Load all articles from the main DOI pool
    doi_pool_articles = read_jsonl_file(DOI_POOL_FILE)
    logger.info(f"Loaded {len(doi_pool_articles)} articles from the main DOI pool.")

    # Load DOIs already in the weak annotation pool
    weak_annotated_dois_set = set(entry.get('doi', '').lower() for entry in read_jsonl_file(WEAK_ANNOTATION_POOL_FILE))
    logger.info(f"Loaded {len(weak_annotated_dois_set)} DOIs from weak annotation pool.")

    articles_to_annotate = []
    updated_doi_pool_list = list(doi_pool_articles)

    for i, article in enumerate(updated_doi_pool_list):
        doi = article.get('doi', '').lower()

        if not doi:
            logger.debug(f"Skipping article with no DOI in main pool: {article.get('title', 'N/A')}")
            continue

        if not force_reannotate_all and doi in weak_annotated_dois_set:
            logger.debug(f"Skipping already weakly annotated DOI: {doi}")
            continue

        if article.get('relevance_score') == 'irrelevant':
            logger.info(f"Skipping DOI {doi} as it's pre-marked as 'irrelevant' in {DOI_POOL_FILE}.")
            continue

        articles_to_annotate.append((i, article))

    logger.info(f"Found {len(articles_to_annotate)} articles to process for weak annotation via LLM.")

    if not articles_to_annotate:
        logger.info("No new articles require weak annotation.")
        return

    doi_pool_was_updated = False

    for original_index, article_data in articles_to_annotate:
        doi = article_data.get('doi', '')
        pmid = article_data.get('pmid')
        title = article_data.get('title', 'N/A')

        logger.info(f"Absolute path to WEAK_ANNOTATION_POOL_FILE: {os.path.abspath(WEAK_ANNOTATION_POOL_FILE)}")

        logger.info(f"Processing ({original_index + 1}/{len(doi_pool_articles)} total in main pool): {doi} - {title}")

        if not article_data.get('abstract'):
            logger.warning(f"Abstract missing for {doi}. LLM annotation quality may be reduced.")

        # --- IMPORTANT: Call call_llm_api with the entire article_data dictionary ---
        llm_response_text = call_llm_api(article_data)

        if llm_response_text:
            parsed_llm_output = parse_llm_response(llm_response_text, doi)

            if parsed_llm_output:
                weak_annotation_entry = {
                    "doi": doi,
                    "pmid": pmid,
                    "title": title,
                    "paper_type": parsed_llm_output.get("paper_type", "other"),
                    "relevance_score_llm": parsed_llm_output.get("relevance_score_llm", "low_relevance"),
                    "biological_context": parsed_llm_output.get("biological_context", "N/A"),
                    "data_modalities_used": parsed_llm_output.get("data_modalities_used", []),
                    "weak_annotations": parsed_llm_output.get("weak_annotations", {}),
                    "annotation_status": "weak_llm_annotated",
                    "llm_query_date": datetime.now().isoformat(),
                    "llm_model": LLM_MODEL_INSTANCE.model_name if LLM_MODEL_INSTANCE else "N/A"
                    # This ensures the model name is saved
                }

                try:
                    with open(WEAK_ANNOTATION_POOL_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(weak_annotation_entry) + "\n")
                    logger.info(f"Successfully wrote weak annotation for DOI {doi} to {WEAK_ANNOTATION_POOL_FILE}.")
                    weak_annotated_dois_set.add(doi.lower())  # Mark as processed for current session
                except Exception as e:
                    logger.error(
                        f"Error writing weak annotation for DOI {doi} to file {WEAK_ANNOTATION_POOL_FILE}: {e}",
                        exc_info=True)

                current_relevance_score_in_pool = updated_doi_pool_list[original_index].get('relevance_score')
                llm_assigned_relevance = weak_annotation_entry['relevance_score_llm']

                relevance_order = {'irrelevant': 0, 'low_relevance': 1, 'uncertain': 1.5, 'relevant': 2,
                                   'highly_relevant': 3, 'null': 1.5}
                current_score_val = relevance_order.get(str(current_relevance_score_in_pool).lower(), 1.5)
                llm_score_val = relevance_order.get(str(llm_assigned_relevance).lower(), 1.5)

                if llm_score_val > current_score_val or current_relevance_score_in_pool in [None, 'null',
                                                                                            'uncertain'] or llm_assigned_relevance == 'irrelevant':
                    updated_doi_pool_list[original_index]['relevance_score'] = llm_assigned_relevance
                    logger.info(f"Updated relevance_score for DOI {doi} in main pool to: {llm_assigned_relevance}")
                    doi_pool_was_updated = True

                if llm_assigned_relevance == 'irrelevant':
                    logger.info(
                        f"DOI {doi} (Title: {title}) marked as 'irrelevant' by LLM. No further detailed weak annotation for this paper.")

            else:
                logger.error(f"Could not parse LLM response for DOI {doi}. Skipping annotation for this entry.")
        else:
            logger.error(f"LLM API call returned empty response for DOI {doi}. Skipping annotation.")
            # If LLM_MODEL_INSTANCE becomes None within call_llm_api (all models exhausted),
            # this means we should stop processing subsequent articles for LLM calls.
            if LLM_MODEL_INSTANCE is None:
                logger.critical("No more LLM models available. Halting further LLM processing for this run.")
                break  # Exit the main article processing loop if all models are exhausted.

        time.sleep(SLEEP_TIME_BETWEEN_LLM_REQUESTS)  # General delay between successful/unsuccessful calls

    if doi_pool_was_updated:
        logger.info(f"Rewriting {DOI_POOL_FILE} with updated relevance scores.")
        write_jsonl_file(DOI_POOL_FILE, updated_doi_pool_list)
    else:
        logger.info(f"No relevance scores updated in {DOI_POOL_FILE}.")

    logger.info("Weak annotation process complete.")
    total_weak_annotated = len(read_jsonl_file(WEAK_ANNOTATION_POOL_FILE))
    logger.info(f"Total unique DOIs weakly annotated: {total_weak_annotated}")


# --- Main execution block ---
if __name__ == "__main__":
    setup_logging()
    ensure_dir(BASE_DIR)
    configure_llm()  # Configure LLM at the start

    # --- IMPORTANT ---
    # Before running this script:
    # 1. Ensure your 'data/doi_pool.jsonl' exists and is populated with articles
    # 2. Set your GOOGLE_API_KEY environment variable.

    # Example of how to run:
    # process_articles_for_weak_annotation(force_reannotate_all=False)
    # To re-annotate ALL papers (including already annotated ones):
    # process_articles_for_weak_annotation(force_reannotate_all=True)

    # To ensure a fresh run for testing, you might delete existing pools:
    # if os.path.exists(WEAK_ANNOTATION_POOL_FILE):
    #    os.remove(WEAK_ANNOTATION_POOL_FILE)
    #    logger.info(f"Removed existing {WEAK_ANNOTATION_POOL_FILE} for a fresh start.")
    # if os.path.exists(DOI_POOL_FILE):
    #     logger.warning(f"Consider running the main scraper script first to populate {DOI_POOL_FILE}.")
    #     # For a full fresh start, you might also remove doi_pool.jsonl and rerun scraper:
    #     # os.remove(DOI_POOL_FILE)

    process_articles_for_weak_annotation(force_reannotate_all=False)  # Default behavior

    # AIzaSyArIq - lqVSfPgNTf - oSTnqJ1mU8X_tMYmg