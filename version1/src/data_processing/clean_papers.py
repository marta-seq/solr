# src/data_processing/clean_papers.py
import pandas as pd
import argparse
import logging
import os
from typing import List, Dict, Any

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def clean_papers(input_file: str, output_file: str):
    """
    Cleans raw scraped paper data:
    - Cleans DOIs.
    - Ensures essential columns exist.
    - Removes extracted section content to separate the data flow.
    """
    setup_logging(log_prefix="clean_papers")
    logger.info("--- Starting Paper Cleaning ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    df = load_jsonl_to_dataframe(input_file)  # This will now use the robust line-by-line loading
    if df.empty:
        logger.warning(f"Input file '{input_file}' is empty. Creating empty output file.")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_jsonl_records([], output_file, append=False)
        return

    initial_count = len(df)
    logger.info(f"Loaded {initial_count} raw papers.")

    # Clean DOIs
    if 'doi' in df.columns:
        df['doi'] = df['doi'].apply(clean_doi)
        # Drop rows where DOI becomes empty after cleaning
        df.dropna(subset=['doi'], inplace=True)
        logger.info(f"Cleaned DOIs. {len(df)} papers remaining after dropping empty DOIs.")
    else:
        logger.warning("DOI column not found in input data. Cannot clean DOIs.")
        # If DOI is critical for downstream, you might want to raise an error or return empty DF
        # For now, we'll proceed but rely on downstream to handle missing DOI if needed.

    # Ensure essential columns exist, fill with empty string if missing
    essential_cols = ['doi', 'pmid', 'title', 'abstract', 'year', 'authors', 'journal']
    for col in essential_cols:
        if col not in df.columns:
            df[col] = ''  # Use empty string for consistency
            logger.warning(f"Missing essential column: '{col}'. Added with empty string values.")

    # --- CRITICAL CHANGE: Remove extracted section content from the main DataFrame ---
    # These will now be handled by a separate extraction step.
    columns_to_drop = [col for col in df.columns if col.startswith('extracted_') and col.endswith('_section')]
    columns_to_drop.append('sections_extracted_successfully')  # Also remove this flag if present

    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True,
                errors='ignore')  # Use errors='ignore' to prevent crash if column not found
        logger.info(f"Removed section content columns from main data flow: {columns_to_drop}")
    else:
        logger.debug("No section content columns found to remove.")

    # Reorder columns for consistency (optional but good practice)
    # Get all current columns, then put essential_cols first, then others
    current_cols = df.columns.tolist()
    ordered_cols = [col for col in essential_cols if col in current_cols] + \
                   [col for col in current_cols if col not in essential_cols]
    df = df[ordered_cols]

    # Save cleaned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_jsonl_records(df.to_dict(orient='records'), output_file, append=False)
    logger.info(f"Cleaned {len(df)} papers. Output saved to {output_file}.")
    logger.info("--- Paper Cleaning Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cleans raw scraped paper data and removes section content."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing raw scraped papers."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for cleaned papers."
    )
    args = parser.parse_args()

    clean_papers(
        input_file=args.input_file,
        output_file=args.output_file
    )