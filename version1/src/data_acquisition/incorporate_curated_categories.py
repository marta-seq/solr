# src/data_acquisition/incorporate_curated_categories.py
# python -m src.data_acquisition.incorporate_curated_categories \
#     --input_file data/intermediate/cleaned_papers_metadata.jsonl \
#     --curated_categories_file data/inputs/paper_classification.xls \
#     --output_file data/intermediate/papers_with_curated_categories.jsonl
import pandas as pd
import json
import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

# Import utility functions
from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, append_to_jsonl, clean_doi

# --- Configuration ---
DEFAULT_INPUT_CLEANED_PAPERS = "data/intermediate/cleaned_papers_metadata.jsonl"
DEFAULT_INPUT_CURATED_CATEGORIES = "data/inputs/paper_classification.xls" # Path to your XLS file
DEFAULT_OUTPUT_FILE = "data/intermediate/papers_with_curated_categories.jsonl"
DEFAULT_LOGGING_DIR = "data/logs/curation_merge_logs/"

# --- Logging Setup ---
logger = logging.getLogger(__name__)

def incorporate_curated_categories(
    df_papers: pd.DataFrame,
    curated_categories_file: str
) -> pd.DataFrame:
    """
    Loads curated categories from an Excel file and merges them into the DataFrame.
    Prioritizes curated data and ensures high annotation/relevance scores for curated DOIs.
    """
    if df_papers.empty:
        logger.warning("Input DataFrame is empty. No curated categories to incorporate.")
        return df_papers.copy()

    initial_papers_count = len(df_papers)
    logger.info(f"Starting incorporation of curated categories for {initial_papers_count} articles.")

    # Load curated categories from XLS
    try:
        # Read the Excel file. Assuming the first sheet is relevant.
        df_curated = pd.read_excel(curated_categories_file)
        if df_curated.empty:
            logger.warning(f"Curated categories file '{curated_categories_file}' is empty. No categories will be merged.")
            return df_papers.copy()

        # Clean DOI in curated data for merging
        if 'doi' not in df_curated.columns:
            logger.error(f"Curated categories file '{curated_categories_file}' missing 'doi' column. Skipping merge.")
            return df_papers.copy()
        df_curated['doi'] = df_curated['doi'].apply(clean_doi)
        df_curated.drop_duplicates(subset=['doi'], inplace=True) # Ensure unique curated DOIs
        logger.info(f"Loaded {len(df_curated)} unique curated entries from '{curated_categories_file}'.")

    except FileNotFoundError:
        logger.error(f"Curated categories file not found: '{curated_categories_file}'. Skipping merge.")
        return df_papers.copy()
    except Exception as e:
        logger.error(f"Error loading or processing curated categories file '{curated_categories_file}': {e}", exc_info=True)
        return df_papers.copy()

    # Rename curated columns to avoid conflicts and indicate origin, or match target names
    # Assuming curated columns: 'category', 'pipeline_category', 'name'
    # Map them to target names: 'curated_category', 'curated_pipeline_category', 'curated_package_name'
    curated_col_map = {
        'category': 'curated_paper_category',
        'pipeline_category': 'curated_pipeline_category',
        'name': 'curated_package_method_name' # Assuming 'name' is the package/method name
    }
    df_curated = df_curated.rename(columns=curated_col_map)

    # Select only relevant columns from curated data for merging
    cols_to_merge = [col for col in curated_col_map.values() if col in df_curated.columns] + ['doi']
    df_curated_filtered = df_curated[cols_to_merge].copy()

    # Merge curated data into the main DataFrame
    # Use a left merge to keep all original papers, and merge curated data where DOIs match
    df_merged = pd.merge(df_papers, df_curated_filtered, on='doi', how='left', suffixes=('', '_curated_new'))
    logger.info(f"Merged curated categories. Total papers after merge: {len(df_merged)}.")

    # Identify which papers received curated data
    merged_dois = df_curated_filtered['doi'].tolist()
    df_merged['is_curated_entry'] = df_merged['doi'].isin(merged_dois)
    logger.info(f"Identified {df_merged['is_curated_entry'].sum()} papers that received curated categories.")

    # For papers that received curated data, ensure annotation_score and relevance_score are high
    # Also, if the 'status' was 'uncurated_new', you might want to update it to 'curated_method' here
    # However, the scraper should have already set 'curated_method' for these.
    # We'll just ensure scores are high.
    df_merged.loc[df_merged['is_curated_entry'], 'annotation_score'] = 100
    df_merged.loc[df_merged['is_curated_entry'], 'relevance_score'] = 100
    logger.info("Updated annotation_score and relevance_score for curated entries.")

    # You might want to update the 'status' if it wasn't already 'curated_method'
    # This ensures consistency if a DOI was scraped as 'uncurated_new' but then found in curated_categories.
    # df_merged.loc[df_merged['is_curated_entry'], 'status'] = 'curated_method'
    # (Commented out for now, as scraper should handle initial 'curated_method' status)

    # Drop the temporary flag column
    df_merged = df_merged.drop(columns=['is_curated_entry'])

    return df_merged


def main_incorporate_curated_categories(input_file: str, curated_categories_file: str, output_file: str):
    """
    Main function to incorporate manually curated paper categories into the dataset.
    """
    # Setup logging
    log_dir_path = DEFAULT_LOGGING_DIR
    setup_logging(log_dir=log_dir_path, log_prefix="curation_merge_log")

    logger.info("--- Starting Curated Categories Incorporation Process ---")
    logger.info(f"Input cleaned papers file: {input_file}")
    logger.info(f"Input curated categories file: {curated_categories_file}")
    logger.info(f"Output file: {output_file}")

    # Load cleaned papers data
    df_papers = load_jsonl_to_dataframe(input_file)
    if df_papers.empty:
        logger.warning(f"Input file '{input_file}' is empty or not found. No curated categories incorporated.")
        ensure_dir(os.path.dirname(output_file))
        with open(output_file, 'w') as f: # Create empty output file
            f.write('')
        return

    # Incorporate curated data
    df_result = incorporate_curated_categories(df_papers, curated_categories_file)

    # Save the result
    ensure_dir(os.path.dirname(output_file))
    if not df_result.empty:
        # Clear existing output file before appending, as this is a "new" version
        if os.path.exists(output_file):
            os.remove(output_file)
            logger.info(f"Cleared existing output file: {output_file}")
        append_to_jsonl(df_result.to_dict(orient='records'), output_file)
        logger.info(f"Successfully saved {len(df_result)} articles with curated categories to '{output_file}'.")
    else:
        logger.info(f"No articles remaining after processing. An empty file will be created at '{output_file}'.")
        with open(output_file, 'w') as f:
            f.write('')

    logger.info("--- Curated Categories Incorporation Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incorporates manually curated paper categories into the dataset."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=DEFAULT_INPUT_CLEANED_PAPERS,
        help=f"Path to the input JSONL file of cleaned papers (default: {DEFAULT_INPUT_CLEANED_PAPERS})."
    )
    parser.add_argument(
        "--curated_categories_file",
        type=str,
        default=DEFAULT_INPUT_CURATED_CATEGORIES,
        help=f"Path to the input XLS file with curated categories (default: {DEFAULT_INPUT_CURATED_CATEGORIES})."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to the output JSONL file with incorporated curated categories (default: {DEFAULT_OUTPUT_FILE})."
    )

    args = parser.parse_args()

    main_incorporate_curated_categories(
        input_file=args.input_file,
        curated_categories_file=args.curated_categories_file,
        output_file=args.output_file
    )
