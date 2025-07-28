# src/llm_annotation/llm_broad_classifier.py
import pandas as pd
import json
import os
import argparse
import logging
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

from tqdm import tqdm

# Add project root to path for utility imports (assuming script runs from project root or similar)
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi
import ollama

logger = logging.getLogger(__name__)


# --- Helper Functions for LLM Prompt Generation ---

def truncate_section(text: Optional[str], max_chars: int) -> Optional[str]:
    """Truncates a string to max_chars, ensuring it's not None or empty after stripping."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    return text[:max_chars]


def generate_llm_prompt(article_data: Dict[str, Any], llm_response_schema: Dict[str, Any]) -> str:
    """
    Generates a structured prompt for the LLM to extract broad classification information.
    """
    title = article_data.get('title', 'N/A')
    abstract = article_data.get('abstract', 'N/A')

    MAX_SECTION_CHARS_ABSTRACT = 2000
    MAX_SECTION_CHARS_SMALL = 1500
    MAX_SECTION_CHARS_LARGE = 4000

    sections_content = []
    if article_data.get('extracted_introduction_section'):
        sections_content.append(
            f"--- Introduction Section ---\n{truncate_section(article_data['extracted_introduction_section'], MAX_SECTION_CHARS_LARGE)}")
    if article_data.get('extracted_methods_section'):
        sections_content.append(
            f"--- Methods Section ---\n{truncate_section(article_data['extracted_methods_section'], MAX_SECTION_CHARS_LARGE)}")
    if article_data.get('extracted_results_discussion_section'):
        sections_content.append(
            f"--- Results and Discussion Section ---\n{truncate_section(article_data['extracted_results_discussion_section'], MAX_SECTION_CHARS_LARGE)}")
    if article_data.get('extracted_conclusion_section'):
        sections_content.append(
            f"--- Conclusion Section ---\n{truncate_section(article_data['extracted_conclusion_section'], MAX_SECTION_CHARS_SMALL)}")
    if article_data.get('extracted_data_availability_section'):
        sections_content.append(
            f"--- Data Availability Section ---\n{truncate_section(article_data['extracted_data_availability_section'], MAX_SECTION_CHARS_SMALL)}")
    if article_data.get('extracted_code_availability_section'):
        sections_content.append(
            f"--- Code Availability Section ---\n{truncate_section(article_data['extracted_code_availability_section'], MAX_SECTION_CHARS_SMALL)}")

    sections_text = "\n\n".join(sections_content)

    llm_notes_on_sections_warning = ""
    if not sections_text.strip():
        llm_notes_on_sections_warning = "Note: Full text sections were not available or were empty. Annotation is based solely on Title and Abstract."

    text_content = f"Title: {title}\n"
    text_content += f"Abstract: {truncate_section(abstract, MAX_SECTION_CHARS_ABSTRACT)}\n\n"
    if sections_text:
        text_content += f"--- Additional Sections ---\n{sections_text}\n\n"

    if llm_notes_on_sections_warning:
        text_content += f"\n\n**Important Note:** {llm_notes_on_sections_warning}\n"

    # --- REVISED PROMPT STRUCTURE FOR CLARITY ---
    prompt = f"""
    Analyze the following scientific paper content. Your task is to extract specific information
    into a JSON object. Your response MUST be ONLY the JSON object. Do NOT include any other text, preambles, or conversational filler.
    The JSON object you output must conform EXACTLY to the structure defined by the JSON schema provided below.

    ---
    **Paper Content for Analysis:**

    {text_content}

    ---

    **JSON Schema (This defines the structure of your output. Fill in the VALUES according to this structure):**
    {json.dumps(llm_response_schema, indent=2)}

    """
    return prompt


def run_llm_annotation(input_papers_file: str, output_llm_annotated_jsonl: str, llm_schema: Dict[str, Any],
                       force_reannotate_all: bool = False):
    """
    Main function to orchestrate loading data, performing LLM annotation, and saving results.
    Args:
        input_papers_file: Path to the input JSONL file of papers with extracted sections.
        output_llm_annotated_jsonl: Path to the output JSONL file for LLM annotations.
        llm_schema: The JSON schema to use for this LLM call.
        force_reannotate_all: If True, re-annotates all papers, ignoring existing 'middle_llm_annotated' status.
    """
    log_dir_path = "data/logs/llm_annotation_logs/"
    setup_logging(log_dir=log_dir_path, log_prefix="llm_broad_classifier_log")

    logger.info("--- Starting LLM Broad Classification Process ---")
    logger.info(f"Input papers file: {input_papers_file}")
    logger.info(f"Output LLM annotated file: {output_llm_annotated_jsonl}")
    logger.info(f"Force reannotate all: {force_reannotate_all}")
    logger.info(f"Using Ollama model: {os.environ.get('OLLAMA_MODEL_NAME', 'llama3')}")

    df_papers = load_jsonl_to_dataframe(input_papers_file)
    if df_papers.empty:
        logger.error(f"Input papers file '{input_papers_file}' is empty or does not exist. Exiting.")
        ensure_dir(os.path.dirname(output_llm_annotated_jsonl))
        save_jsonl_records([], output_llm_annotated_jsonl, append=False)
        return
    df_papers['doi'] = df_papers['doi'].apply(clean_doi)
    df_papers.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_papers)} unique papers from {input_papers_file}.")

    # Load existing output to identify already processed DOIs
    df_existing_output = load_jsonl_to_dataframe(output_llm_annotated_jsonl)
    existing_llm_annotated_dois = set()
    if not df_existing_output.empty:
        if 'doi' in df_existing_output.columns:
            df_existing_output['doi'] = df_existing_output['doi'].apply(clean_doi)

        # Identify papers that were already processed by LLM
        if 'llm_annotation_status' in df_existing_output.columns:
            existing_llm_annotated_dois = set(
                df_existing_output[
                    df_existing_output['llm_annotation_status'] == 'middle_llm_annotated'
                    ]['doi'].tolist()
            )
            logger.info(
                f"Found {len(existing_llm_annotated_dois)} DOIs already 'middle_llm_annotated' in '{output_llm_annotated_jsonl}'.")

    papers_to_process_df = df_papers.copy()
    initial_llm_process_count = len(papers_to_process_df)
    logger.info(f"Initial count of ALL papers identified for LLM processing: {initial_llm_process_count}")

    if not force_reannotate_all and not papers_to_process_df.empty:
        # Filter out already LLM-annotated papers
        papers_to_process_df = papers_to_process_df[
            ~papers_to_process_df['doi'].isin(existing_llm_annotated_dois)
        ].copy()
        logger.info(
            f"Filtered to {len(papers_to_process_df)} papers not yet LLM-annotated (or forced re-annotation is off).")

    if papers_to_process_df.empty:
        logger.info("No new papers found to process with LLM. Exiting.")
        # If no new papers, ensure output file exists and is not empty if original was not empty
        ensure_dir(os.path.dirname(output_llm_annotated_jsonl))
        if df_existing_output.empty:  # Only create empty if no existing output
            save_jsonl_records([], output_llm_annotated_jsonl, append=False)
        return

    papers_to_process_list = papers_to_process_df.to_dict(orient='records')
    logger.info(f"Starting LLM annotation for {len(papers_to_process_list)} papers.")

    ensure_dir(os.path.dirname(output_llm_annotated_jsonl))

    # Save existing output (excluding any that will be reprocessed by force_reannotate_all, if applicable)
    # This ensures papers already classified by LLM are retained
    if not df_existing_output.empty:
        if force_reannotate_all:
            # If re-annotating all, start with an empty file (all will be reprocessed)
            save_jsonl_records([], output_llm_annotated_jsonl, append=False)
        else:
            # Only save the ones that were already processed and are NOT being re-processed
            # This is key for incremental updates, so we don't lose previous LLM work
            already_processed_papers = df_existing_output[
                df_existing_output['doi'].isin(existing_llm_annotated_dois)].to_dict(orient='records')
            save_jsonl_records(already_processed_papers, output_llm_annotated_jsonl, append=False)
    else:
        # If no existing output, start with an empty file
        save_jsonl_records([], output_llm_annotated_jsonl, append=False)

    for i, article_data in enumerate(tqdm(papers_to_process_list, desc="LLM Broad Classifying Papers")):
        doi = article_data.get('doi')
        if not doi:
            logger.warning(f"Skipping paper due to missing DOI: {article_data.get('title', 'N/A')}")
            article_data['llm_annotation_status'] = 'skipped_no_doi'
            article_data['llm_annotation_timestamp'] = datetime.now().isoformat()
            article_data['llm_model_used'] = os.environ.get('OLLAMA_MODEL_NAME', 'llama3')
            save_jsonl_records([article_data], output_llm_annotated_jsonl, append=True)
            continue

        llm_output_text = None
        llm_output = {}  # Initialize to empty dict at the start of each loop
        llm_status = 'llm_error_general'
        llm_error_message = None

        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                prompt_text = generate_llm_prompt(article_data, llm_schema)

                response = ollama.chat(
                    model=os.environ.get('OLLAMA_MODEL_NAME', 'llama3'),
                    messages=[
                        {"role": "system",
                         "content": "You are an expert bioinformatician and machine learning researcher specializing in spatial omics data analysis. Your ONLY task is to output a JSON object according to the provided schema. Do NOT include any other text, preambles, or conversational filler."},
                        {"role": "user", "content": prompt_text}
                    ],
                    format='json'
                )

                llm_output_text = response['message']['content']

                try:
                    llm_output = json.loads(llm_output_text)
                    llm_status = 'middle_llm_annotated'
                    logger.info(f"Successfully broad classified DOI: {doi}")
                    break

                except json.JSONDecodeError as e:
                    llm_status = 'llm_error_json_parse'
                    llm_error_message = f"LLM returned malformed JSON for DOI {doi}: {e}. Raw response (first 500 chars): {llm_output_text[:500] if llm_output_text else 'N/A'}..."
                    logger.error(f"DOI {doi}: {llm_error_message}")
                    # Attempt a robust parse for common LLM markdown wrapping
                    cleaned_response_text = llm_output_text.strip()
                    if cleaned_response_text.startswith("```json"):
                        cleaned_response_text = cleaned_response_text[len("```json"):].strip()
                    if cleaned_response_text.endswith("```"):
                        cleaned_response_text = cleaned_response_text[:-len("```")].strip()
                    elif cleaned_response_text.startswith(
                            "```"):  # Handle cases where it starts with ``` but not ```json
                        cleaned_response_text = cleaned_response_text[len("```"):].strip()

                    try:  # Try parsing cleaned text
                        llm_output = json.loads(cleaned_response_text)
                        llm_status = 'middle_llm_annotated'
                        logger.warning(
                            f"DOI {doi}: Ollama response required cleaning. Cleaned response (first 200 chars): {cleaned_response_text[:200]}...")
                        break
                    except json.JSONDecodeError as clean_e:
                        llm_error_message += f" (Also failed after cleaning: {clean_e})"
                        logger.error(f"DOI {doi}: {llm_error_message}")


            except ollama.ResponseError as e:
                llm_status = 'llm_error_ollama_response'
                llm_error_message = f"Ollama response error for DOI {doi}: {e}. Status code: {e.status_code if hasattr(e, 'status_code') else 'N/A'}"
                logger.error(f"DOI {doi}: {llm_error_message}")
            except Exception as e:
                llm_status = 'llm_error_api'
                llm_error_message = f"General API call failed for DOI {doi}: {e}"
                logger.error(f"DOI {doi}: {llm_error_message}", exc_info=True)

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt) + random.uniform(0, 1))
            else:
                logger.critical(f"DOI {doi}: Max retries reached for API call error.")

        # --- Explicitly remove all existing 'llm_annot_' prefixed keys before adding new ones ---
        keys_to_remove = [key for key in article_data if key.startswith('llm_annot_')]
        for key in keys_to_remove:
            del article_data[key]
        logger.debug(f"DOI {doi}: Removed {len(keys_to_remove)} old 'llm_annot_' keys.")

        # --- STRICT FILTERING: Only add keys that are explicitly in the LLM schema ---
        schema_properties = llm_schema.get('properties', {})
        for key, value in (llm_output or {}).items():  # Use (llm_output or {}) to handle None case
            if key in schema_properties:
                article_data[f'llm_annot_{key}'] = value
            else:
                logger.warning(
                    f"DOI {doi}: LLM returned unexpected key '{key}' not in broad schema. Skipping this key.")

        # Ensure llm_annot_classification is a dictionary, even if empty, before score_and_split_broad_llm_output accesses it
        # This is the critical safeguard to prevent NoneType errors downstream
        if not isinstance(article_data.get('llm_annot_classification'), dict):
            article_data['llm_annot_classification'] = {}
            logger.warning(
                f"DOI {doi}: 'llm_annot_classification' was not a dictionary. Initialized to empty dict for downstream processing.")

        article_data['llm_annotation_status'] = llm_status
        article_data['llm_annotation_timestamp'] = datetime.now().isoformat()
        article_data['llm_model_used'] = os.environ.get('OLLAMA_MODEL_NAME', 'llama3')
        article_data['llm_annotation_error'] = llm_error_message  # Store error message or None

        save_jsonl_records([article_data], output_llm_annotated_jsonl, append=True)

    logger.info("--- LLM Broad Classification Process Complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs LLM-based broad classification of scientific papers using Ollama."
    )
    parser.add_argument(
        "--input_papers_file",
        type=str,
        required=True,
        help="Path to the input JSONL file of papers with extracted sections."
    )
    parser.add_argument(
        "--output_llm_annotated_jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file for LLM broad annotations."
    )
    parser.add_argument(
        "--llm_schema_path",
        type=str,
        required=True,
        help="Path to a JSON file containing the LLM response schema for broad classification."
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="If set, re-annotates all papers, ignoring existing 'middle_llm_annotated' status."
    )
    args = parser.parse_args()

    with open(args.llm_schema_path, 'r') as f:
        llm_response_schema_dict = json.load(f)

    run_llm_annotation(
        input_papers_file=args.input_papers_file,
        output_llm_annotated_jsonl=args.output_llm_annotated_jsonl,
        llm_schema=llm_response_schema_dict,
        force_reannotate_all=args.force_reannotate
    )