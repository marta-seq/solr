# src/data_acquisition/repopulate_sections_from_llm_broad.py
import pandas as pd
import json
import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add project root to path for utility imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def repopulate_sections_from_llm_broad_main(
        input_llm_broad_file: str,
        output_papers_with_sections_file: str,
        output_sections_log_file: str
):
    """
    Repopulates extracted section data into papers_with_extracted_sections.jsonl
    and reconstructs full_text_extraction.log from llm_broad_annotated_papers.jsonl.
    This is a one-time recovery script.
    """
    setup_logging(log_prefix="repopulate_sections")
    logger.info("--- Starting Section Repopulation from LLM Broad Annotations ---")
    logger.info(f"Input LLM broad file: {input_llm_broad_file}")
    logger.info(f"Output papers with sections file: {output_papers_with_sections_file}")
    logger.info(f"Output sections log file: {output_sections_log_file}")

    # 1. Load the LLM broad annotated papers
    df_llm_broad = load_jsonl_to_dataframe(input_llm_broad_file)
    if df_llm_broad.empty:
        logger.error(f"Input LLM broad file '{input_llm_broad_file}' is empty or does not exist. Exiting.")
        ensure_dir(os.path.dirname(output_papers_with_sections_file))
        ensure_dir(os.path.dirname(output_sections_log_file))
        save_jsonl_records([], output_papers_with_sections_file, append=False)
        save_jsonl_records([], output_sections_log_file, append=False)
        return

    df_llm_broad['doi'] = df_llm_broad['doi'].apply(clean_doi)
    logger.info(f"Loaded {len(df_llm_broad)} records from LLM broad annotations.")

    # Prepare data for new papers_with_extracted_sections.jsonl
    repopulated_papers_with_sections = []
    # Prepare data for new full_text_extraction.log
    reconstructed_sections_log = []

    # Define the fields that represent extracted sections
    extracted_section_fields = [
        'extracted_introduction_section',
        'extracted_methods_section',
        'extracted_results_discussion_section',
        'extracted_conclusion_section',
        'extracted_code_availability_section',
        'extracted_data_availability_section'
    ]

    for index, row in df_llm_broad.iterrows():
        paper_data = row.to_dict()
        doi = paper_data.get('doi')

        if not doi:
            logger.warning(f"Skipping record due to missing DOI: {paper_data.get('title', 'N/A')}")
            continue

        # --- Reconstruct data for papers_with_extracted_sections.jsonl ---
        # Copy all original fields
        repopulated_paper = paper_data.copy()

        # Extract and rename section-related fields from llm_broad_annotated_papers
        # These fields are expected to be at the top level in the broad LLM output
        # and need to be copied directly.
        repopulated_paper['section_extraction_status'] = paper_data.get('section_extraction_status',
                                                                        'Unknown Status (Repopulated)')
        repopulated_paper['sections_extracted_successfully'] = paper_data.get('sections_extracted_successfully', False)
        repopulated_paper['section_extraction_error_message'] = paper_data.get('section_extraction_error_message', None)

        # Ensure all expected section fields exist, even if empty
        for field in extracted_section_fields:
            # The llm_broad_annotated_papers.jsonl typically has these fields directly
            # from the original extraction process.
            repopulated_paper[field] = paper_data.get(field, "")

        repopulated_papers_with_sections.append(repopulated_paper)

        # --- Reconstruct data for full_text_extraction.log ---
        # This log expects 'extracted_sections_data' as a nested dictionary
        reconstructed_log_entry = {
            "doi": doi,
            "section_extraction_status": paper_data.get('section_extraction_status',
                                                        'Unknown Status (Reconstructed Log)'),
            "extracted_sections_data": {},
            "error_message": paper_data.get('section_extraction_error_message', None)
        }

        # Populate extracted_sections_data for the log entry
        for field in extracted_section_fields:
            # Remove 'extracted_' prefix for the key in extracted_sections_data
            log_key = field.replace('extracted_', '')
            reconstructed_log_entry['extracted_sections_data'][log_key] = paper_data.get(field, "")

        reconstructed_sections_log.append(reconstructed_log_entry)

    # 2. Save the repopulated papers_with_extracted_sections.jsonl
    ensure_dir(os.path.dirname(output_papers_with_sections_file))
    save_jsonl_records(repopulated_papers_with_sections, output_papers_with_sections_file, append=False)
    logger.info(
        f"Successfully repopulated {len(repopulated_papers_with_sections)} records into {output_papers_with_sections_file}.")

    # 3. Save the reconstructed full_text_extraction.log
    ensure_dir(os.path.dirname(output_sections_log_file))
    save_jsonl_records(reconstructed_sections_log, output_sections_log_file, append=False)
    logger.info(
        f"Successfully reconstructed {len(reconstructed_sections_log)} records into {output_sections_log_file}.")

    logger.info("--- Section Repopulation Complete ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="One-time script to repopulate extracted sections and log from LLM broad annotations."
    )
    parser.add_argument(
        "--input_llm_broad_file",
        type=str,
        default="data/intermediate/llm_broad_annotated_papers.jsonl",
        help="Path to the input JSONL file with LLM broad annotated papers."
    )
    parser.add_argument(
        "--output_papers_with_sections_file",
        type=str,
        default="data/intermediate/papers_with_extracted_sections.jsonl",
        help="Path to the output JSONL file for papers with extracted sections."
    )
    parser.add_argument(
        "--output_sections_log_file",
        type=str,
        default="data/logs/full_text_extraction.log",
        help="Path to the output JSONL file for the full text extraction log."
    )
    args = parser.parse_args()

    repopulate_sections_from_llm_broad_main(
        input_llm_broad_file=args.input_llm_broad_file,
        output_papers_with_sections_file=args.output_papers_with_sections_file,
        output_sections_log_file=args.output_sections_log_file
    )