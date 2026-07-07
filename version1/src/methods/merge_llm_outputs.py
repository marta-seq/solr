import pandas as pd
import json
import os
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Input Files (relative paths from my_project/scripts/methods/)
DOI_POOL_FILE = "../../data/doi_pool.jsonl"
WEAK_ANNOTATIONS_FILE = "../../data/weak_annotation_pool.jsonl"
EXTRACTED_SECTIONS_FILE = "../../data/methods/methods_section_extraction.jsonl"
LLM_MIDDLE_ANNOTATIONS_FILE = "../../data/methods/llm_annotated_papers.jsonl"  # Main LLM output
# Output Combined File
OUTPUT_COMPLETE_METHODS_FILE = "../../data/methods/complete_methods.jsonl"

# Default annotation score for entries processed by LLM (mid-review)
DEFAULT_LLM_ANNOTATION_SCORE = 2

# Prefix for columns coming from weak annotations to avoid name clashes
WEAK_ANNOTATION_PREFIX = "weak_annot_"

# --- Logging Setup ---
LOG_DIR = "../../data/logs/methods"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/merge_llm_outputs_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:  # Ensure handlers are not duplicated if script is run multiple times in same session
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# --- Helper Functions (re-used from previous script) ---
def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a JSONL file into a pandas DataFrame."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} at line {line_num}: {line.strip()}")
    else:
        logger.warning(f"File not found: {file_path}. Returning empty DataFrame.")
    return pd.DataFrame(data)


def clean_doi(doi_series: pd.Series) -> pd.Series:
    """Cleans DOI strings by removing URL prefixes, lowercasing, and stripping whitespace."""
    if not isinstance(doi_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    return doi_series.astype(str).str.replace('https://doi.org/', '', regex=False).str.lower().str.strip()


# --- Main Merging Logic ---
def merge_llm_data() -> pd.DataFrame:
    """
    Merges DOI pool, weak annotations, extracted sections, and LLM middle annotations
    into a single, comprehensive DataFrame, retaining only DOIs present in all three
    core annotation files (sections, middle annotations, weak annotations).
    """
    logger.info("--- Starting LLM Data Merging Process (Intersection Mode) ---")

    # 1. Load the three core annotation files and get their DOI sets
    df_sections = load_jsonl_to_dataframe(EXTRACTED_SECTIONS_FILE)
    df_llm_middle = load_jsonl_to_dataframe(LLM_MIDDLE_ANNOTATIONS_FILE)
    df_weak = load_jsonl_to_dataframe(WEAK_ANNOTATIONS_FILE)

    # Ensure DOIs are cleaned and unique in source dataframes
    # Also handle cases where a file might be empty or missing a 'doi' column
    for df in [df_sections, df_llm_middle, df_weak]:
        if not df.empty and 'doi' in df.columns:
            df['doi'] = clean_doi(df['doi'])
            # Keep last annotation if duplicates in source files, which is common
            df.drop_duplicates(subset=['doi'], keep='last', inplace=True)
        else:
            # If a core file is empty or missing 'doi', its DOI set will be empty
            logger.warning(
                f"One of the core annotation files is empty or missing 'doi' column. This will likely result in an empty final merged file if this is a critical input: {df}")

    # Get DOIs from each (handling potential empty dataframes)
    # If a DataFrame is empty or lacks a 'doi' column, its set will be empty.
    # An empty set in an intersection will result in an empty intersection.
    dois_sections = set(
        df_sections['doi'].tolist()) if not df_sections.empty and 'doi' in df_sections.columns else set()
    dois_llm_middle = set(
        df_llm_middle['doi'].tolist()) if not df_llm_middle.empty and 'doi' in df_llm_middle.columns else set()
    dois_weak = set(df_weak['doi'].tolist()) if not df_weak.empty and 'doi' in df_weak.columns else set()

    # Find the intersection of DOIs across all three core sets
    common_dois = dois_sections.intersection(dois_llm_middle, dois_weak)
    logger.info(f"Found {len(common_dois)} DOIs present in ALL three core annotation files.")

    if not common_dois:
        logger.warning("No DOIs found in the intersection of all core annotation files. The output file will be empty.")
        # Create and save an empty DataFrame if no common DOIs are found
        empty_df = pd.DataFrame(columns=['doi'])  # Ensures 'doi' column exists for consistency
        empty_df.to_json(OUTPUT_COMPLETE_METHODS_FILE, orient='records', lines=True, force_ascii=False,
                         date_format='iso')
        return empty_df

    # 2. Load DOI Pool and filter it by the common DOIs
    df_base = load_jsonl_to_dataframe(DOI_POOL_FILE)
    if df_base.empty:
        logger.critical(f"DOI pool file '{DOI_POOL_FILE}' is empty or not found. Cannot proceed with merging.")
        return pd.DataFrame()  # This will lead to an empty final df anyway

    df_base['doi'] = clean_doi(df_base['doi'])
    # Filter the base DOI pool to only include the common DOIs
    df_base = df_base[df_base['doi'].isin(common_dois)].copy()
    df_base.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Filtered DOI pool down to {len(df_base)} records based on the common DOIs intersection.")

    # 3. Perform Left Merges with the filtered base
    # Start with the filtered DOI pool as df_combined
    df_combined = df_base

    # Merge extracted sections
    df_combined = pd.merge(df_combined, df_sections, on='doi', how='left', suffixes=('', '_sections_DROP'))
    # Drop columns that were duplicated from the right DataFrame during merge (e.g., if common non-doi columns existed)
    df_combined.drop(columns=[col for col in df_combined.columns if '_sections_DROP' in str(col)], inplace=True)
    logger.info(f"Merged with sections. Current total records: {len(df_combined)}")

    # Merge LLM middle annotations
    df_combined = pd.merge(df_combined, df_llm_middle, on='doi', how='left', suffixes=('', '_llm_middle_DROP'))
    df_combined.drop(columns=[col for col in df_combined.columns if '_llm_middle_DROP' in str(col)], inplace=True)
    logger.info(f"Merged with LLM middle annotations. Current total records: {len(df_combined)}")

    # Merge weak annotations with prefixing
    if not df_weak.empty:
        # Prefix columns *before* merging to avoid name clashes with main LLM annotations
        cols_to_prefix = [col for col in df_weak.columns if col != 'doi']
        df_weak_prefixed = df_weak.rename(columns={col: WEAK_ANNOTATION_PREFIX + col for col in cols_to_prefix})
        logger.info(f"Applied prefix '{WEAK_ANNOTATION_PREFIX}' to weak annotation columns.")

        df_combined = pd.merge(df_combined, df_weak_prefixed, on='doi', how='left', suffixes=('', '_weak_DROP'))
        df_combined.drop(columns=[col for col in df_combined.columns if '_weak_DROP' in str(col)], inplace=True)
        logger.info(f"Merged with weak annotations. Current total records: {len(df_combined)}")
    else:
        logger.warning("Weak annotations DataFrame is empty, skipping merge for weak annotations.")

    # 4. Post-merge processing
    # Handle llm_annotation_date: Prefer date from 'llm_middle_annotations.jsonl' if present.
    # If not present or invalid from primary LLM annotations, set to current time.
    if 'llm_annotation_date' not in df_combined.columns:
        df_combined['llm_annotation_date'] = pd.NaT  # Initialize with Pandas Not a Time for datetime NaNs

    # Ensure llm_annotation_date is datetime type and fill missing with current time
    df_combined['llm_annotation_date'] = pd.to_datetime(df_combined['llm_annotation_date'], errors='coerce')
    # Fixing FutureWarning: use direct assignment instead of inplace=True
    df_combined['llm_annotation_date'] = df_combined['llm_annotation_date'].fillna(datetime.now())

    # Set default annotation_score for all records in this combined file.
    # This ensures newly added (common) records have a score of 2.
    df_combined['annotation_score'] = pd.to_numeric(df_combined.get('annotation_score', DEFAULT_LLM_ANNOTATION_SCORE),
                                                    errors='coerce')
    df_combined['annotation_score'] = df_combined['annotation_score'].fillna(DEFAULT_LLM_ANNOTATION_SCORE)
    df_combined['annotation_score'] = df_combined['annotation_score'].astype(int)

    # Initialize annotation_status (if not present)
    if 'annotation_status' not in df_combined.columns:
        df_combined['annotation_status'] = 'LLM_processed'  # Default status
    else:
        # Fill any NaNs in existing annotation_status and ensure string type
        df_combined['annotation_status'] = df_combined['annotation_status'].fillna('LLM_processed').astype(str)

    # Final deduplication by DOI (should be largely redundant if intersection and merges are correct, but good safeguard)
    df_combined.drop_duplicates(subset=['doi'], inplace=True, keep='last')
    logger.info(f"Final records after merging and deduplication: {len(df_combined)}")

    # Save the combined DataFrame
    try:
        df_combined.to_json(OUTPUT_COMPLETE_METHODS_FILE, orient='records', lines=True, force_ascii=False,
                            date_format='iso')
        logger.info(f"Combined data saved successfully to '{OUTPUT_COMPLETE_METHODS_FILE}'.")
    except Exception as e:
        logger.critical(f"Failed to save combined data to '{OUTPUT_COMPLETE_METHODS_FILE}': {e}")
        return pd.DataFrame()

    logger.info("--- LLM Data Merging Process (Intersection Mode) Finished ---")
    return df_combined


if __name__ == "__main__":
    merged_data = merge_llm_data()
    if not merged_data.empty:
        logger.info(
            f"Combined LLM outputs generated with {len(merged_data)} records. You can now run the master file update script.")
    else:
        logger.error(
            "Failed to generate combined LLM outputs or no common DOIs found across all sources. Output file might be empty.")
