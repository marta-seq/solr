# src/data_processing/extract_sections.py
import pandas as pd
import argparse
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Set

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def extract_sections(input_raw_file: str, output_sections_file: str):
    """
    Extracts DOI, PMID, and all 'extracted_section' content from raw scraped papers
    and saves them to a separate JSONL file.
    Handles incremental updates: only processes new papers not already in the output file.
    """
    setup_logging(log_prefix="extract_sections")
    logger.info("--- Starting Section Extraction ---")
    logger.info(f"Input raw papers file: {input_raw_file}")
    logger.info(f"Output sections file: {output_sections_file}")

    # Load existing sections to identify already processed DOIs
    existing_sections_data = []
    existing_sections_dois = set()
    if os.path.exists(output_sections_file) and os.path.getsize(output_sections_file) > 0:
        try:
            # Load line by line to be robust to partial writes
            with open(output_sections_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)
                        existing_sections_data.append(record)
                        if 'doi' in record and record['doi']:
                            existing_sections_dois.add(clean_doi(record['doi']))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not parse line in existing sections file {output_sections_file} at line {line_num}: {line.strip()}")
            logger.info(
                f"Loaded {len(existing_sections_dois)} existing DOIs from {output_sections_file} for incremental processing.")
        except Exception as e:
            logger.warning(
                f"Error loading existing sections file {output_sections_file}: {e}. Will re-process all sections from input.")
            existing_sections_data = []
            existing_sections_dois = set()

    df_raw = load_jsonl_to_dataframe(input_raw_file)  # This will use the robust loader
    if df_raw.empty:
        logger.warning(f"Input raw papers file '{input_raw_file}' is empty. No sections to extract.")
        # If output file exists, ensure it's not deleted if it had existing data
        if not existing_sections_data:  # Only if no existing data was loaded
            os.makedirs(os.path.dirname(output_sections_file), exist_ok=True)
            save_jsonl_records([], output_sections_file, append=False)  # Create empty if truly no data
        return

    df_raw['doi'] = df_raw['doi'].apply(clean_doi)
    df_raw.dropna(subset=['doi'], inplace=True)  # Ensure valid DOIs for matching

    new_sections_to_add = []
    processed_new_dois = set()

    # Iterate through raw papers to find new ones and extract sections
    for index, row in df_raw.iterrows():
        doi = row.get('doi')
        if not doi:
            logger.debug(f"Skipping record due to missing DOI: {row.to_dict()}")
            continue

        cleaned_doi = clean_doi(doi)
        if cleaned_doi in existing_sections_dois or cleaned_doi in processed_new_dois:
            logger.debug(f"Skipping already processed DOI: {cleaned_doi}")
            continue

        section_record = {'doi': cleaned_doi, 'pmid': row.get('pmid')}

        # Extract all columns that start with 'extracted_' and end with '_section'
        extracted_section_cols = [col for col in row.index if col.startswith('extracted_') and col.endswith('_section')]
        for col in extracted_section_cols:
            section_record[col] = row.get(col)  # Get the content of the section

        # Also include the success flag and timestamp
        section_record['sections_extracted_successfully'] = row.get('sections_extracted_successfully', False)
        section_record[
            'section_extraction_timestamp'] = datetime.now().isoformat()  # Add a timestamp for when sections were extracted/processed

        new_sections_to_add.append(section_record)
        processed_new_dois.add(cleaned_doi)

    if new_sections_to_add:
        logger.info(f"Found {len(new_sections_to_add)} new papers with sections to extract.")
        # Combine existing data with new data
        final_sections_data = existing_sections_data + new_sections_to_add
        save_jsonl_records(final_sections_data, output_sections_file, append=False)  # Overwrite with full list
        logger.info(
            f"Updated {output_sections_file} with {len(new_sections_to_add)} new section records. Total: {len(final_sections_data)}.")
    else:
        logger.info("No new sections to extract. Output file remains unchanged.")

    logger.info("--- Section Extraction Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts and stores section content from raw papers incrementally."
    )
    parser.add_argument(
        "--input_raw_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing raw scraped papers (with sections)."
    )
    parser.add_argument(
        "--output_sections_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for extracted sections."
    )
    args = parser.parse_args()

    extract_sections(
        input_raw_file=args.input_raw_file,
        output_sections_file=args.output_sections_file
    )