# src/data_acquisition/merge_extracted_sections.py
import pandas as pd
import json
import os
import argparse
import logging
from typing import List, Dict, Any, Optional

# Import utility functions
from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
# IMPORTANT: Use save_jsonl_records for atomic writes when overwriting
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

# --- Configuration ---
DEFAULT_INPUT_PAPERS_FILE = "data/intermediate/filtered_and_scored_papers.jsonl" # Input from previous pipeline step
DEFAULT_INPUT_SECTIONS_LOG_FILE = "data/intermediate/extracted_sections_log.jsonl" # Input from full-text extractor
DEFAULT_OUTPUT_FILE = "data/intermediate/papers_with_extracted_sections.jsonl" # Output after merging
DEFAULT_LOGGING_DIR = "data/logs/section_merge_logs/"

# --- Section Keys (must match extractor) ---
SECTION_KEYS = [
    "introduction",
    "results_discussion",
    "conclusion",
    "methods",
    "code_availability",
    "data_availability"
]

# --- Logging Setup ---
logger = logging.getLogger(__name__)

def merge_sections_into_papers(
    df_papers: pd.DataFrame,
    df_sections_log: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges the best extracted sections from the log into the main papers DataFrame.
    """
    if df_papers.empty:
        logger.warning("Input papers DataFrame is empty. No sections to merge.")
        return pd.DataFrame()
    if df_sections_log.empty:
        logger.warning("Input sections log DataFrame is empty. No sections to merge.")
        # If no sections log, ensure section-related columns are initialized with default values
        df_papers_copy = df_papers.copy()
        df_papers_copy['sections_extracted_successfully'] = False
        df_papers_copy['is_open_access'] = False
        df_papers_copy['full_text_access_type'] = 'Unknown'
        df_papers_copy['section_extraction_status'] = 'Not Processed'
        for key in SECTION_KEYS:
            df_papers_copy[f'extracted_{key}_section'] = '' # Initialize text sections with empty string
        return df_papers_copy

    initial_papers_count = len(df_papers)
    logger.info(f"Starting merge of extracted sections for {initial_papers_count} papers.")

    # Ensure DOIs are clean for merging
    df_papers['doi'] = df_papers['doi'].apply(clean_doi)
    df_sections_log['doi'] = df_sections_log['doi'].apply(clean_doi)

    # Determine the "best" extraction attempt for each DOI
    status_rank_map = {
        'extracted from pdf (sections found)': 4,
        'extracted from html (sections found)': 3,
        'pdf processed but no relevant sections found': 2,
        'html processed (no sections found by heuristics)': 1,
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
        'serialization error': -3,
        'skipped pdf (pymupdf not installed)': -0.5,
        'skipped: empty doi': -0.6
    }
    df_sections_log['status_rank'] = df_sections_log['section_extraction_status'].str.lower().map(status_rank_map).fillna(-4)

    # Convert timestamp to datetime for proper sorting
    df_sections_log['extracted_at_timestamp'] = pd.to_datetime(df_sections_log['extracted_at_timestamp'], errors='coerce')

    # Group by DOI, find the max status_rank, then for ties, pick the latest timestamp
    idx_best_status = df_sections_log.groupby('doi')['status_rank'].idxmax()
    df_best_attempts = df_sections_log.loc[idx_best_status]

    idx_latest_timestamp = df_best_attempts.groupby('doi')['extracted_at_timestamp'].idxmax()
    df_sections_to_merge = df_best_attempts.loc[idx_latest_timestamp].copy()

    # Select columns to merge from the sections log
    merge_cols = ['doi', 'is_open_access', 'full_text_access_type', 'section_extraction_status']
    for key in SECTION_KEYS:
        merge_cols.append(f'extracted_{key}_section')

    df_sections_to_merge = df_sections_to_merge[merge_cols]

    # Add a flag indicating if sections were successfully extracted (i.e., not just a link or error)
    def check_sections_content(row):
        status = row.get('section_extraction_status', '').lower()
        if any(success_word in status for success_word in ['extracted from html', 'extracted from pdf']):
            return any(pd.notna(row.get(f'extracted_{key}_section')) and str(row.get(f'extracted_{key}_section', '')).strip() != '' for key in SECTION_KEYS)
        return False

    df_sections_to_merge['sections_extracted_successfully'] = df_sections_to_merge.apply(check_sections_content, axis=1)
    logger.info(f"Identified {df_sections_to_merge['sections_extracted_successfully'].sum()} DOIs with successfully extracted sections.")

    # Merge into the main papers DataFrame
    # Use a left merge to keep all original papers
    df_merged = pd.merge(df_papers, df_sections_to_merge, on='doi', how='left') # Removed suffixes as we'll handle explicitly

    # Define a dictionary of default fill values for columns that might have NaNs after merge
    # Use empty string for text sections, False for booleans, 'Unknown'/'Not Processed' for status/type
    fill_values = {
        'is_open_access': False,
        'sections_extracted_successfully': False,
        'full_text_access_type': 'Unknown',
        'section_extraction_status': 'Not Processed'
    }
    for key in SECTION_KEYS:
        fill_values[f'extracted_{key}_section'] = '' # Fill text sections with empty string

    # Apply fillna to all relevant columns at once
    for col_name, fill_value in fill_values.items():
        if col_name in df_merged.columns:
            # For boolean columns, ensure dtype is correct before filling
            if col_name in ['is_open_access', 'sections_extracted_successfully']:
                # Convert to boolean dtype first, then fillna for consistency
                df_merged[col_name] = df_merged[col_name].astype('boolean').fillna(fill_value)
            else:
                # For other types (strings), just fill NaNs
                df_merged[col_name] = df_merged[col_name].fillna(fill_value)
        else:
            # If a column doesn't exist after merge (e.g., no match for a paper), add it with default
            df_merged[col_name] = fill_value

    logger.info(f"Merged sections into {len(df_merged)} papers.")
    return df_merged


def main_merge_sections(input_papers_file: str, input_sections_log_file: str, output_file: str):
    """
    Main function to merge extracted full-text sections into the main papers dataset.
    """
    # Setup logging
    log_dir_path = DEFAULT_LOGGING_DIR
    setup_logging(log_dir=log_dir_path, log_prefix="section_merge_log")

    logger.info("--- Starting Section Merging Process ---")
    logger.info(f"Input papers file: {input_papers_file}")
    logger.info(f"Input sections log file: {input_sections_log_file}")
    logger.info(f"Output merged papers file: {output_file}")

    # Load papers data
    df_papers = load_jsonl_to_dataframe(input_papers_file)
    if df_papers.empty:
        logger.warning(f"Input papers file '{input_papers_file}' is empty or not found. Cannot merge sections.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False) # Create empty output file atomically
        return

    # Load sections log data
    df_sections_log = load_jsonl_to_dataframe(input_sections_log_file)
    if df_sections_log.empty:
        logger.warning(f"Input sections log file '{input_sections_log_file}' is empty or not found. Outputting original papers without sections.")
        ensure_dir(os.path.dirname(output_file))
        # If no sections log, just save the original papers, adding section columns with default values
        df_papers_copy = df_papers.copy()
        df_papers_copy['sections_extracted_successfully'] = False
        df_papers_copy['is_open_access'] = False
        df_papers_copy['full_text_access_type'] = 'Unknown'
        df_papers_copy['section_extraction_status'] = 'Not Processed'
        for key in SECTION_KEYS:
            df_papers_copy[f'extracted_{key}_section'] = '' # Initialize text sections with empty string
        # Use save_jsonl_records for atomic write
        save_jsonl_records(df_papers_copy.to_dict(orient='records'), output_file, append=False)
        return

    # Perform merging
    df_merged = merge_sections_into_papers(df_papers, df_sections_log)

    # Save the result
    ensure_dir(os.path.dirname(output_file))
    if not df_merged.empty:
        # Use save_jsonl_records for atomic write (append=False implies overwrite)
        save_jsonl_records(df_merged.to_dict(orient='records'), output_file, append=False)
        logger.info(f"Successfully saved {len(df_merged)} papers with merged sections to '{output_file}'.")
    else:
        logger.info(f"No articles remaining after merging. An empty file will be created at '{output_file}'.")
        save_jsonl_records([], output_file, append=False) # Ensure empty file is created atomically

    logger.info("--- Section Merging Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merges extracted full-text sections into the main papers dataset."
    )
    parser.add_argument(
        "--input_papers_file",
        type=str,
        default=DEFAULT_INPUT_PAPERS_FILE,
        help=f"Path to the input JSONL file of papers (default: {DEFAULT_INPUT_PAPERS_FILE})."
    )
    parser.add_argument(
        "--input_sections_log_file",
        type=str,
        default=DEFAULT_INPUT_SECTIONS_LOG_FILE,
        help=f"Path to the input JSONL file for extracted sections log (default: {DEFAULT_INPUT_SECTIONS_LOG_FILE})."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to the output JSONL file for papers with merged sections (default: {DEFAULT_OUTPUT_FILE})."
    )

    args = parser.parse_args()

    main_merge_sections(
        input_papers_file=args.input_papers_file,
        input_sections_log_file=args.input_sections_log_file,
        output_file=args.output_file
    )