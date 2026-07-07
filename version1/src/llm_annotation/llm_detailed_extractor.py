# src/llm_annotation/llm_detailed_extractor.py
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

# Add project root to path for utility imports
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
    Generates a structured prompt for the LLM to extract detailed method information.
    """
    title = article_data.get('title', 'N/A')
    abstract = article_data.get('abstract', 'N/A')

    MAX_SECTION_CHARS_ABSTRACT = 2000
    MAX_SECTION_CHARS_SMALL = 1500
    MAX_SECTION_CHARS_LARGE = 4000

    sections_content = []
    # Only include sections if they exist AND are actual text content (not just boolean flags)
    if article_data.get('extracted_introduction_section') and isinstance(article_data['extracted_introduction_section'],
                                                                         str):
        sections_content.append(
            f"--- Introduction Section ---\n{truncate_section(article_data['extracted_introduction_section'], MAX_SECTION_CHARS_LARGE)}")
    if article_data.get('extracted_methods_section') and isinstance(article_data['extracted_methods_section'], str):
        sections_content.append(
            f"--- Methods Section ---\n{truncate_section(article_data['extracted_methods_section'], MAX_SECTION_CHARS_LARGE)}")
    if article_data.get('extracted_results_discussion_section') and isinstance(
            article_data['extracted_results_discussion_section'], str):
        sections_content.append(
            f"--- Results and Discussion Section ---\n{truncate_section(article_data['extracted_results_discussion_section'], MAX_SECTION_CHARS_LARGE)}")
    if article_data.get('extracted_conclusion_section') and isinstance(article_data['extracted_conclusion_section'],
                                                                       str):
        sections_content.append(
            f"--- Conclusion Section ---\n{truncate_section(article_data['extracted_conclusion_section'], MAX_SECTION_CHARS_SMALL)}")
    if article_data.get('extracted_data_availability_section') and isinstance(
            article_data['extracted_data_availability_section'], str):
        sections_content.append(
            f"--- Data Availability Section ---\n{truncate_section(article_data['extracted_data_availability_section'], MAX_SECTION_CHARS_SMALL)}")
    if article_data.get('extracted_code_availability_section') and isinstance(
            article_data['extracted_code_availability_section'], str):
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

    prompt = f"""
    This paper has been identified as a computational methods paper.
    Your task is to meticulously analyze the provided scientific paper content and extract specific, detailed information about the computational method into a JSON object.

    ---
    **Paper Content for Analysis:**

    {text_content}

    ---

    **Instructions for JSON Output:**
    * **Your response MUST be ONLY the JSON object.** Do NOT include any other text, preambles, or conversational filler.
    * **The JSON object you output must conform EXACTLY to the structure defined by the JSON schema provided below.**
    * For each property in the schema, you must provide the *extracted value*, not the schema definition of that property.
    * If a field is not applicable or information is not found, use `null` for single values or an empty array `[]` for list fields.
    * For `enum` fields, select values strictly from the provided lists. If 'Other' is chosen, provide a brief explanation in the `llm_notes` field.
    * **For 'primary_programming_languages'**: List all primary programming languages used for the method/software. Select from the provided enum. If not specified, use "Not Specified". If multiple, list all.

    **Example of Expected JSON Output Format (DO NOT output this example, fill with actual data):**
    ```json
    {{
      "package_algorithm_name": "SpaGCN",
      "pipeline_analysis_steps": ["Spatial Domain Identification", "Dimensionality Reduction"],
      "tested_data_modalities": ["Transcriptomics", "Imaging"],
      "tested_assay_types_platforms": ["Visium", "ST-seq"],
      "compared_algorithms_packages": [
        {{"name": "Seurat", "comparison_details": "Compared for clustering performance."}}
      ],
      "code_availability_details": {{"status": "Available", "link": "[https://github.com/example/spagcn](https://github.com/example/spagcn)"}},
      "data_used_details": [
        {{"dataset_name": "Mouse Brain Visium", "data_modalities": ["spatial transcriptomics"], "sample_info": {{"species": "Mus musculus", "tissue": "Brain"}}}}
      ],
      "primary_programming_languages": ["Python", "R"],
      "llm_notes": null
    }}
    ```

    **JSON Schema (This defines the structure of your output. Fill in the VALUES according to this structure):**
    {json.dumps(llm_response_schema, indent=2)}

    """
    return prompt


def run_llm_annotation(input_papers_file: str, output_llm_annotated_jsonl: str, llm_schema_path: str,
                       force_reannotate_all: bool = False, detailed_llm_target_score: int = 70):
    """
    Main function to orchestrate loading data, performing LLM annotation, and saving results.
    Args:
        input_papers_file: Path to the input JSONL file of computational papers with extracted sections.
        output_llm_annotated_jsonl: Path to the output JSONL file for LLM detailed annotations.
        llm_schema_path: Path to the JSON schema to use for this LLM call.
        force_reannotate_all: If True, re-annotates all papers, ignoring existing 'detailed_llm_annotated' status.
        detailed_llm_target_score: The target score for papers after detailed LLM annotation.
    """
    log_dir_path = "data/logs/llm_annotation_logs/"
    setup_logging(log_dir=log_dir_path, log_prefix="llm_detailed_extractor_log")

    logger.info("--- Starting LLM Detailed Extraction Process ---")
    logger.info(f"Input papers file: {input_papers_file}")
    logger.info(f"Output LLM annotated file: {output_llm_annotated_jsonl}")
    logger.info(f"Force reannotate all: {force_reannotate_all}")
    logger.info(f"Using Ollama model: {os.environ.get('OLLAMA_MODEL_NAME', 'llama3')}")
    logger.info(f"Assigned detailed LLM target score: {detailed_llm_target_score}")

    # Load the detailed LLM schema from the provided path
    try:
        with open(llm_schema_path, 'r') as f:
            llm_schema = json.load(f)

        expected_llm_annot_keys = {f"llm_annot_{prop}" for prop in llm_schema.get('properties', {}).keys()}
        logger.info(f"Loaded LLM Detailed Schema. Expected LLM annotation keys: {expected_llm_annot_keys}")
    except FileNotFoundError:
        logger.error(f"LLM Detailed Schema file not found at '{llm_schema_path}'. Cannot filter unexpected columns.")
        llm_schema = {}
        expected_llm_annot_keys = set()
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding LLM Detailed Schema from '{llm_schema_path}': {e}. Cannot filter unexpected columns.")
        llm_schema = {}
        expected_llm_annot_keys = set()

    df_papers = load_jsonl_to_dataframe(input_papers_file)
    if df_papers.empty:
        logger.error(f"Input papers file '{input_papers_file}' is empty or does not exist. Exiting.")
        ensure_dir(os.path.dirname(output_llm_annotated_jsonl))
        save_jsonl_records([], output_llm_annotated_jsonl, append=False)
        return
    df_papers['doi'] = df_papers['doi'].apply(clean_doi)
    df_papers.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_papers)} unique papers from {input_papers_file}.")

    # Load existing output to identify already processed DOIs and keep their data
    df_existing_output = load_jsonl_to_dataframe(output_llm_annotated_jsonl)
    existing_annotated_dois = set()
    if not df_existing_output.empty:
        if 'doi' in df_existing_output.columns:
            df_existing_output['doi'] = df_existing_output['doi'].apply(clean_doi)
        if 'llm_annotation_status' in df_existing_output.columns:
            existing_annotated_dois = set(
                df_existing_output[
                    df_existing_output['llm_annotation_status'] == 'detailed_llm_annotated'
                    ]['doi'].tolist()
            )
        logger.info(
            f"Found {len(existing_annotated_dois)} DOIs already 'detailed_llm_annotated' in '{output_llm_annotated_jsonl}'.")

    papers_to_process_df = df_papers.copy()
    initial_llm_process_count = len(papers_to_process_df)
    logger.info(f"Initial count of ALL papers identified for LLM processing: {initial_llm_process_count}")

    if not force_reannotate_all and not papers_to_process_df.empty:
        papers_to_process_df = papers_to_process_df[~papers_to_process_df['doi'].isin(existing_annotated_dois)].copy()
        logger.info(
            f"Filtered to {len(papers_to_process_df)} papers not yet 'detailed_llm_annotated' (or forced re-annotation is off).")

    if papers_to_process_df.empty:
        logger.info("No new papers found to process with LLM. Exiting.")
        ensure_dir(os.path.dirname(output_llm_annotated_jsonl))
        if not os.path.exists(output_llm_annotated_jsonl) or os.path.getsize(output_llm_annotated_jsonl) == 0:
            save_jsonl_records([], output_llm_annotated_jsonl, append=False)
        return

    papers_to_process_list = papers_to_process_df.to_dict(orient='records')
    logger.info(f"Starting LLM annotation for {len(papers_to_process_list)} papers.")

    ensure_dir(os.path.dirname(output_llm_annotated_jsonl))

    newly_processed_records = []  # Collect records that are newly processed in this run

    for i, article_data in enumerate(tqdm(papers_to_process_list, desc="LLM Detailed Extracting Papers")):
        doi = article_data.get('doi')
        if not doi:
            logger.warning(f"Skipping paper due to missing DOI: {article_data.get('title', 'N/A')}")
            article_data['llm_annotation_status'] = 'skipped_no_doi'
            article_data['llm_annotation_timestamp'] = datetime.now().isoformat()
            article_data['llm_model_used'] = os.environ.get('OLLAMA_MODEL_NAME', 'llama3')
            article_data['annotation_score'] = float(detailed_llm_target_score)
            newly_processed_records.append(article_data)  # Add to list of new records
            continue

        llm_output_text = None
        llm_output = None
        llm_status = 'llm_error_general'
        llm_error_message = None

        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                # Create a copy for prompt generation to avoid modifying original article_data prematurely
                article_data_for_prompt = article_data.copy()
                # Remove full text sections from the prompt data (they are not needed for prompt)
                for section_col in [
                    'extracted_introduction_section', 'extracted_methods_section',
                    'extracted_results_discussion_section', 'extracted_conclusion_section',
                    'extracted_data_availability_section', 'extracted_code_availability_section'
                ]:
                    if section_col in article_data_for_prompt:
                        # Only remove if it's actual text content, not already a boolean flag
                        if isinstance(article_data_for_prompt[section_col], str):
                            pass  # Keep for prompt generation
                        else:  # If it's already a boolean or other type, remove it from prompt data
                            del article_data_for_prompt[section_col]

                prompt_text = generate_llm_prompt(article_data_for_prompt, llm_schema)

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
                    llm_status = 'detailed_llm_annotated'
                    logger.info(f"Successfully detailed extracted DOI: {doi}")
                    break

                except json.JSONDecodeError:
                    cleaned_response_text = llm_output_text.strip()
                    if cleaned_response_text.startswith("```json"):
                        cleaned_response_text = cleaned_response_text[len("```json"):].strip()
                    if cleaned_response_text.endswith("```"):
                        cleaned_response_text = cleaned_response_text[:-len("```")].strip()
                    elif cleaned_response_text.startswith("```"):
                        cleaned_response_text = cleaned_response_text[len("```"):].strip()
                        if cleaned_response_text.startswith("json"):
                            cleaned_response_text = cleaned_response_text[len("json"):].strip()

                    try:
                        llm_output = json.loads(cleaned_response_text)
                        llm_status = 'detailed_llm_annotated'
                        logger.warning(
                            f"DOI {doi}: Ollama response required cleaning. Cleaned response (first 200 chars): {cleaned_response_text[:200]}...")
                        break
                    except json.JSONDecodeError as clean_e:
                        llm_status = 'llm_error_json_parse'
                        llm_error_message = f"LLM returned malformed JSON for DOI {doi} even after cleaning: {clean_e}. Raw response (first 200 chars): {llm_output_text[:200] if llm_output_text else 'N/A'}..."
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

        if llm_output is None:
            llm_output = {}

        # Remove all existing 'llm_annot_' prefixed keys before adding new ones
        keys_to_remove = [key for key in article_data if key.startswith('llm_annot_')]
        for key in keys_to_remove:
            del article_data[key]
        logger.debug(f"DOI {doi}: Removed {len(keys_to_remove)} old 'llm_annot_' keys from detailed extractor.")

        # Filter llm_output to only include keys defined in the schema properties
        filtered_llm_output = {}
        schema_properties = llm_schema.get('properties', {})
        for key, value in (llm_output or {}).items():
            if key in schema_properties:
                filtered_llm_output[key] = value
            else:
                logger.warning(f"DOI {doi}: LLM returned unexpected key '{key}' not in detailed schema. Skipping.")

        # Apply filtered LLM output to article_data with 'llm_annot_' prefix
        for key, value in filtered_llm_output.items():
            article_data[f'llm_annot_{key}'] = value

        article_data['llm_annotation_status'] = llm_status
        article_data['llm_annotation_timestamp'] = datetime.now().isoformat()
        article_data['llm_model_used'] = os.environ.get('OLLAMA_MODEL_NAME', 'llama3')
        article_data['llm_annotation_error'] = llm_error_message

        # Set detailed LLM target score if successfully annotated
        if llm_status == 'detailed_llm_annotated':
            article_data['annotation_score'] = float(detailed_llm_target_score)
        # If not successfully annotated, keep the previous score (e.g., broad LLM score)

        # Remove extracted section content (text) before saving.
        # The 'has_section' flags should already be present from extract_full_text_sections.py.
        for section_col in [
            'extracted_introduction_section', 'extracted_methods_section',
            'extracted_results_discussion_section', 'extracted_conclusion_section',
            'extracted_data_availability_section', 'extracted_code_availability_section'
        ]:
            if section_col in article_data and isinstance(article_data[section_col], str):
                del article_data[section_col]
                logger.debug(f"Removed text content for {section_col} from DOI {doi}.")

        newly_processed_records.append(article_data)  # Add to list of new records

    # Combine existing records with newly processed records and save the complete set
    all_records = df_existing_output.to_dict(orient='records') + newly_processed_records
    save_jsonl_records(all_records, output_llm_annotated_jsonl, append=False)  # Overwrite with complete set
    logger.info(
        f"Saved total {len(all_records)} records to {output_llm_annotated_jsonl} (overwritten with combined data).")
    logger.info("--- LLM Detailed Extraction Process Complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Performs LLM-based detailed extraction of scientific papers using Ollama."
    )
    parser.add_argument(
        "--input_papers_file",
        type=str,
        required=True,
        help="Path to the input JSONL file of computational papers for detailed extraction."
    )
    parser.add_argument(
        "--output_llm_annotated_jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file for LLM detailed annotations."
    )
    parser.add_argument(
        "--llm_schema_path",
        type=str,
        required=True,
        help="Path to a JSON file containing the LLM detailed response schema."
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="If set, re-annotates all papers, ignoring existing 'detailed_llm_annotated' status."
    )
    parser.add_argument(
        "--detailed_llm_target_score",
        type=int,
        default=70,
        help="The target score for papers after detailed LLM annotation."
    )
    args = parser.parse_args()

    run_llm_annotation(
        input_papers_file=args.input_papers_file,
        output_llm_annotated_jsonl=args.output_llm_annotated_jsonl,
        llm_schema_path=args.llm_schema_path,
        force_reannotate_all=args.force_reannotate,
        detailed_llm_target_score=args.detailed_llm_target_score
    )