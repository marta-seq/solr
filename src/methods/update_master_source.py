import pandas as pd
import json
import os
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import requests
import time # Import time for sleep


# --- Configuration ---
# Input Files (relative paths from the script location)
MASTER_FILE = "../../data/manual_curated_papers_methods.jsonl"
INPUT_LLM_COMBINED_FILE = "../../data/methods/complete_methods.jsonl"
INPUT_LLM_COMBINED_FILE = "../../data/llm_annotated_METHODS_llama.jsonl"

# Output Files
OUTPUT_MASTER_FILE = "../../data/methods/master_methods_manual_curation.jsonl"  # Overwrite the original master file
OUTPUT_NEW_PAPERS_FILE = "../../data/methods/new_llm_papers_to_master.jsonl"  # For papers added solely by LLM that were not in master

# Annotation Score Threshold: Papers with annotation_score >= this will NOT be updated by LLM
# A score of 3 or higher indicates manual curation and locks the record from LLM updates.
MANUAL_CURATION_SCORE_THRESHOLD = 3

# Interval (in days) after which citation counts should be re-fetched.
# Set to 0 to force update all citations every run (if they have a DOI).
CITATION_UPDATE_INTERVAL_DAYS = 7

# --- Logging Setup ---
# Log directory will be created if it doesn't exist
LOG_DIR = "../../data/logs/methods"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/manage_living_review_data_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure handlers are not duplicated if the script is run multiple times in the same session
if not logger.handlers:
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# --- Helper Functions ---
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


def save_dataframe_to_jsonl(df: pd.DataFrame, file_path: str, compression: str = 'infer'):
    """Saves a DataFrame to a JSONL file."""
    if not df.empty:
        df.to_json(file_path, orient='records', lines=True, force_ascii=False, date_format='iso',
                   compression=compression)
        logger.info(f"DataFrame saved successfully to '{file_path}'. Records: {len(df)}")
    else:
        # Create an empty file if the DataFrame is empty, to indicate no data
        open(file_path, 'a').close()  # 'a' mode creates if not exists, doesn't error if exists
        logger.warning(
            f"Empty DataFrame. No data saved to '{file_path}'. An empty file was created (if it didn't exist).")


# --- Main Logic for Updating Master File (Simplified Version) ---
def update_master_file_content() -> pd.DataFrame:
    """
    Updates the master curation file with new LLM annotations.
    If a master record is NOT manually curated (annotation_score < MANUAL_CURATION_SCORE_THRESHOLD),
    it will be entirely superseded by the corresponding LLM output record (if DOI matches).
    Manual curation (score >= MANUAL_CURATION_SCORE_THRESHOLD) is preserved.
    New DOIs from LLM output are added.
    """
    logger.info("--- Starting Master File Content Update Process (Simplified Logic) ---")

    # Load master and LLM output data
    df_master = load_jsonl_to_dataframe(MASTER_FILE)
    df_llm_output = load_jsonl_to_dataframe(INPUT_LLM_COMBINED_FILE)

    if df_llm_output.empty:
        logger.info("No new LLM data to process in 'complete_methods.jsonl'. Master file content remains unchanged.")
        return df_master  # Return original master if no LLM data

    # Clean DOIs for merging and ensure uniqueness within each source
    if not df_master.empty:
        df_master['doi'] = clean_doi(df_master['doi'])
        df_master.drop_duplicates(subset=['doi'], keep='last', inplace=True)
    df_llm_output['doi'] = clean_doi(df_llm_output['doi'])
    df_llm_output.drop_duplicates(subset=['doi'], keep='last',
                                  inplace=True)  # Ensure LLM output itself is unique by DOI

    logger.info(f"Master file initial records: {len(df_master)}")
    logger.info(f"LLM output records for processing: {len(df_llm_output)}")

    # Ensure 'llm_annotation_date' exists and is in datetime format in LLM output
    # This is still important for new records being added from LLM.
    if 'llm_annotation_date' in df_llm_output.columns:
        df_llm_output['llm_annotation_date'] = pd.to_datetime(df_llm_output['llm_annotation_date'], errors='coerce')
        df_llm_output['llm_annotation_date'] = df_llm_output['llm_annotation_date'].fillna(datetime.now())
    else:
        logger.warning(
            f"LLM output file '{INPUT_LLM_COMBINED_FILE}' is missing 'llm_annotation_date'. Adding current datetime to LLM output records.")
        df_llm_output['llm_annotation_date'] = datetime.now()

    # Ensure 'annotation_score' is an integer column, fillna default values
    if 'annotation_score' not in df_master.columns:
        df_master['annotation_score'] = 0
    df_master['annotation_score'] = pd.to_numeric(df_master['annotation_score'], errors='coerce').fillna(0).astype(int)

    if 'annotation_score' not in df_llm_output.columns:
        df_llm_output['annotation_score'] = 2  # Default to 2 for LLM processed if not provided
    df_llm_output['annotation_score'] = pd.to_numeric(df_llm_output['annotation_score'], errors='coerce').fillna(
        2).astype(int)

    # Set default annotation_status for LLM output records if missing
    if 'annotation_status' not in df_llm_output.columns:
        df_llm_output['annotation_status'] = 'LLM_processed'
    df_llm_output['annotation_status'] = df_llm_output['annotation_status'].fillna('LLM_processed').astype(str)

    # Step 1: Separate manually curated papers from df_master
    # These records will be preserved as-is, regardless of LLM output.
    df_master_manually_curated = df_master[
        df_master['annotation_score'] >= MANUAL_CURATION_SCORE_THRESHOLD
        ].copy()
    logger.info(f"Manually curated papers in master (will be preserved): {len(df_master_manually_curated)}")

    # Step 2: Get all LLM output records. These are the "desired" versions for non-manually curated papers,
    # and all new papers.
    df_records_from_llm = df_llm_output.copy()
    logger.info(f"LLM output records (potential updates/additions): {len(df_records_from_llm)}")

    # Step 3: Identify records from master that are *not* manually curated.
    # These are the candidates for being replaced by LLM output or kept if no LLM equivalent exists.
    df_master_llm_eligible = df_master[
        df_master['annotation_score'] < MANUAL_CURATION_SCORE_THRESHOLD
        ].copy()
    logger.info(f"LLM-eligible papers from master: {len(df_master_llm_eligible)}")

    # Step 4: Concatenate in specific order and deduplicate by DOI.
    # Order of concatenation is crucial for `drop_duplicates(keep='last')`:
    # 1. Manually curated records from master (always kept).
    # 2. LLM-eligible records from master (these are the 'old' versions that might be replaced).
    # 3. All records from LLM output (these are the 'new' versions that will supersede if DOI matches, or new papers).

    frames_to_concat = [
        df_master_manually_curated,
        df_master_llm_eligible,  # These might be duplicates of records in df_records_from_llm
        df_records_from_llm  # These will be the chosen ones for duplicates, or new ones
    ]

    df_final_master = pd.concat(frames_to_concat, ignore_index=True)

    # Drop duplicates based on 'doi', keeping the last occurrence.
    # This means for any DOI that appears in both df_master_llm_eligible and df_records_from_llm,
    # the version from df_records_from_llm will be kept.
    df_final_master.drop_duplicates(subset=['doi'], keep='last', inplace=True)

    # Step 5: Finalize DataFrame Structure (ensure all columns from all sources are present)
    all_possible_columns = sorted(list(set(df_master.columns.tolist() + df_llm_output.columns.tolist())))
    df_final_master = df_final_master.reindex(columns=all_possible_columns)

    # Step 6: Identify and save truly new LLM papers
    initial_master_dois = set(df_master['doi'].tolist()) if not df_master.empty else set()
    new_llm_dois_added = [
        doi for doi in df_llm_output['doi'].tolist()
        if doi not in initial_master_dois
    ]
    df_new_llm_papers = df_llm_output[df_llm_output['doi'].isin(new_llm_dois_added)].copy()
    save_dataframe_to_jsonl(df_new_llm_papers, OUTPUT_NEW_PAPERS_FILE)
    logger.info(f"Truly new LLM papers added to master: {len(new_llm_dois_added)}")

    logger.info(f"Final master file record count after content update: {len(df_final_master)}")
    logger.info("--- Master File Content Update Process Finished ---")
    return df_final_master


def add_citation_counts(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches and updates citation counts for DOIs in the master DataFrame
    using the OpenCitations API, based on the update interval.
    """
    logger.info("--- Starting Citation Count Update Process ---")

    if df_master.empty:
        logger.warning("Master DataFrame is empty. No citations to fetch.")
        return df_master

    # Ensure 'last_citations_update' is datetime for this function's operations
    if 'last_citations_update' in df_master.columns:
        df_master['last_citations_update'] = pd.to_datetime(df_master['last_citations_update'], errors='coerce')
    else:
        df_master['last_citations_update'] = pd.NaT  # Initialize if column doesn't exist (e.g., first run)

    # Identify DOIs that need update based on CITATION_UPDATE_INTERVAL_DAYS
    current_time = datetime.now()

    # Filter for DOIs with a valid DOI string
    df_with_dois = df_master[df_master['doi'].notna() & (df_master['doi'] != '')].copy()

    if CITATION_UPDATE_INTERVAL_DAYS == 0:
        # Force update all DOIs
        dois_to_update = df_with_dois
        logger.info(
            f"CITATION_UPDATE_INTERVAL_DAYS is 0. Attempting to update citations for all {len(dois_to_update)} records with DOIs.")
    else:
        # Update if last_citations_update is NaN or older than interval
        dois_to_update = df_with_dois[
            df_with_dois['last_citations_update'].isna() |
            ((current_time - df_with_dois['last_citations_update']).dt.days >= CITATION_UPDATE_INTERVAL_DAYS)
            ].copy()
        logger.info(
            f"Found {len(dois_to_update)} DOIs needing citation updates based on {CITATION_UPDATE_INTERVAL_DAYS} day interval.")

    if dois_to_update.empty:
        logger.info("No DOIs found needing citation updates based on current criteria.")
        return df_master  # Return original if no updates

    # Initialize 'citations' column if it doesn't exist
    if 'citations' not in df_master.columns:
        df_master['citations'] = np.nan

    # Iterate through DOIs to fetch citations with a progress bar
    for index, row in tqdm(dois_to_update.iterrows(), total=len(dois_to_update), desc="Fetching citations"):
        doi = str(row['doi'])  # Ensure DOI is string
        opencitations_url = f"https://opencitations.net/index/api/v1/citations/{doi}"

        try:
            response = requests.get(opencitations_url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            citations_data = response.json()

            # Using .loc for safe assignment based on DOI
            if citations_data and isinstance(citations_data, list):
                citation_count = len(citations_data)
                df_master.loc[df_master['doi'] == doi, 'citations'] = citation_count
                df_master.loc[df_master['doi'] == doi, 'last_citations_update'] = datetime.now()
                logger.debug(f"Updated citations for DOI {doi}: {citation_count}")
            else:
                df_master.loc[df_master['doi'] == doi, 'citations'] = 0
                df_master.loc[
                    df_master['doi'] == doi, 'last_citations_update'] = datetime.now()  # Still mark as updated
                logger.debug(f"No citations found or empty response for DOI {doi}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching citations for DOI {doi}: {e}")
            df_master.loc[df_master['doi'] == doi, 'citations'] = np.nan
            df_master.loc[df_master['doi'] == doi, 'last_citations_update'] = datetime.now()  # Mark as attempted
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from OpenCitations for DOI {doi}. Response: {response.text[:100]}...")
            df_master.loc[df_master['doi'] == doi, 'citations'] = np.nan
            df_master.loc[df_master['doi'] == doi, 'last_citations_update'] = datetime.now()  # Mark as attempted
        except Exception as e:
            logger.error(f"Unexpected error during citation fetch for DOI {doi}: {e}", exc_info=True)
            df_master.loc[df_master['doi'] == doi, 'citations'] = np.nan
            df_master.loc[df_master['doi'] == doi, 'last_citations_update'] = datetime.now()  # Mark as attempted

        # Be polite to the API to avoid hitting rate limits.
        time.sleep(0.1)

    df_master['citations'] = pd.to_numeric(df_master['citations'], errors='coerce').fillna(0).astype(int)

    logger.info(f"Citation counts update process completed.")
    return df_master


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("--- Starting Data Master Management Script ---")

    # Step 1: Update the master file content with new LLM annotations, preserving manual curation
    updated_df_master_content = update_master_file_content()

    # Step 2: Add/update citation counts if master data is available
    if updated_df_master_content is not None and not updated_df_master_content.empty:
        final_df_master = add_citation_counts(updated_df_master_content)
        # Save the final master file after citation updates
        save_dataframe_to_jsonl(final_df_master, OUTPUT_MASTER_FILE)
    else:
        logger.warning("No master data to process for citations. Ensure update_master_file_content ran successfully.")
        # If content update resulted in empty/None, ensure the master file is still handled (e.g., empty file created)
        save_dataframe_to_jsonl(pd.DataFrame(), OUTPUT_MASTER_FILE)  # Create empty master file if no data

    logger.info("--- Data Master Management Script Finished ---")