# src/data_acquisition/clean_scraped_data.py
import pandas as pd
import argparse
import logging
import json # Ensure json is imported if needed for any direct json operations
import os
from datetime import datetime

# Add project root to path for utility imports
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir # <--- THIS IS THE CRITICAL LINE
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)

def clean_scraped_data(input_file: str, output_file: str, sample_size: int):
    """
    Cleans and deduplicates scraped paper data.
    - Loads raw scraped articles from a JSONL file.
    - Deduplicates based on DOI.
    - Applies sampling if a sample_size is specified.
    - Saves the cleaned data to a new JSONL file.
    """
    setup_logging(log_prefix="clean_scraped_data")
    logger.info("--- Starting Data Cleaning and Deduplication Process ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Requested sample size: {sample_size} (use -1 for all papers)")

    df = load_jsonl_to_dataframe(input_file)

    if df.empty:
        logger.warning(f"Input file '{input_file}' is empty or does not exist. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file)) # Ensure dir exists even for empty output
        save_jsonl_records([], output_file, append=False)
        return

    initial_record_count = len(df)
    logger.info(f"Loaded {initial_record_count} records.")

    # Clean DOIs and deduplicate
    if 'doi' in df.columns:
        df['doi'] = df['doi'].apply(clean_doi)
        df.drop_duplicates(subset=['doi'], inplace=True)
        logger.info(f"Removed duplicates based on DOI. Remaining records: {len(df)}.")
    else:
        logger.warning("No 'doi' column found for deduplication. Skipping DOI-based deduplication.")

    # Apply sampling if specified and valid
    if sample_size > 0 and sample_size < len(df):
        df_cleaned = df.sample(n=sample_size, random_state=42).copy()
        logger.info(f"Applied sampling. Selected {len(df_cleaned)} papers.")
    else:
        df_cleaned = df.copy()
        logger.info(f"No sampling applied (sample_size is -1 or greater than available papers). Keeping all {len(df_cleaned)} papers.")

    # Ensure 'annotation_score' column exists and is numeric
    if 'annotation_score' not in df_cleaned.columns:
        df_cleaned['annotation_score'] = 0.0
    df_cleaned['annotation_score'] = df_cleaned['annotation_score'].fillna(0.0).astype(float)

    ensure_dir(os.path.dirname(output_file)) # Ensure output directory exists
    save_jsonl_records(df_cleaned.to_dict(orient='records'), output_file, append=False)
    logger.info(f"Saved {len(df_cleaned)} cleaned records to {output_file}.")
    logger.info("--- Data Cleaning and Deduplication Process Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cleans and deduplicates scraped paper data."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing raw scraped articles."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for cleaned and deduplicated papers."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Number of papers to sample for processing. Use -1 for all papers."
    )
    args = parser.parse_args()

    clean_scraped_data(
        input_file=args.input_file,
        output_file=args.output_file,
        sample_size=args.sample_size
    )