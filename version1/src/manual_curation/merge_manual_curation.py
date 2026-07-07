# src/manual_curation/merge_manual_curation.py
import pandas as pd
import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add the project root to sys.path to enable imports like 'src.utils.logging_setup'
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Use insert(0) to pr
# Import utility functions
from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def merge_manual_curation_main(
        input_llm_processed_file: str,
        input_manual_annotations_file: str,
        input_papers_to_delete_file: str,
        output_master_file: str,
        manual_annotation_score: float
):
    """
    Merges LLM-processed papers with manual annotations, handles deletions,
    and creates the final master dataset.
    Prioritizes manual annotations.
    """
    setup_logging(log_prefix="manual_curation_merge")
    logger.info("--- Starting Manual Curation Merge ---")
    logger.info(f"Input LLM processed file: {input_llm_processed_file}")
    logger.info(f"Input manual annotations file: {input_manual_annotations_file}")
    logger.info(f"Input papers to delete file: {input_papers_to_delete_file}")
    logger.info(f"Output master file: {output_master_file}")

    # 1. Load LLM-processed papers
    df_llm = load_jsonl_to_dataframe(input_llm_processed_file)
    if df_llm.empty:
        logger.warning("LLM processed DataFrame is empty. Master file will be empty or only contain manual entries.")
        df_llm = pd.DataFrame()  # Ensure it's an empty DataFrame if file was empty

    df_llm['doi'] = df_llm['doi'].apply(clean_doi)
    df_llm.set_index('doi', inplace=True)
    logger.info(f"Loaded {len(df_llm)} papers from LLM processed data.")

    # 2. Load papers to delete
    dois_to_delete = set()
    if os.path.exists(input_papers_to_delete_file) and os.path.getsize(input_papers_to_delete_file) > 0:
        try:
            df_delete = pd.read_csv(input_papers_to_delete_file)
            if 'doi' in df_delete.columns:
                dois_to_delete = set(df_delete['doi'].apply(clean_doi).tolist())
                logger.info(f"Loaded {len(dois_to_delete)} DOIs to delete from {input_papers_to_delete_file}.")
            else:
                logger.warning(f"'{input_papers_to_delete_file}' does not have a 'doi' column. Skipping deletion.")
        except Exception as e:
            logger.error(f"Error loading papers to delete from {input_papers_to_delete_file}: {e}", exc_info=True)
    else:
        logger.info(
            f"Papers to delete file '{input_papers_to_delete_file}' not found or empty. No papers will be deleted based on this file.")

    # Apply deletion filter to LLM data
    if not df_llm.empty:
        initial_count = len(df_llm)
        df_llm = df_llm[~df_llm.index.isin(dois_to_delete)]
        logger.info(f"Deleted {initial_count - len(df_llm)} papers from LLM data based on deletion list.")

    # 3. Load manual annotations
    df_manual = pd.DataFrame()
    if os.path.exists(input_manual_annotations_file) and os.path.getsize(input_manual_annotations_file) > 0:
        try:
            # Manual annotations are expected to be JSONL for consistency
            df_manual = load_jsonl_to_dataframe(input_manual_annotations_file)

            if not df_manual.empty:
                df_manual['doi'] = df_manual['doi'].apply(clean_doi)
                df_manual.set_index('doi', inplace=True)
                logger.info(f"Loaded {len(df_manual)} manual annotations from {input_manual_annotations_file}.")

                # Ensure manual entries have the correct score and review flag
                df_manual['llm_annotation_score'] = manual_annotation_score
                df_manual['manual_reviewed'] = True
                df_manual['manual_review_date'] = datetime.now().isoformat()
            else:
                logger.info(f"Manual annotations file '{input_manual_annotations_file}' is empty or has no valid data.")

        except Exception as e:
            logger.error(f"Error loading manual annotations from {input_manual_annotations_file}: {e}", exc_info=True)
            df_manual = pd.DataFrame()  # Ensure empty DataFrame on error
    else:
        logger.info(
            f"Manual annotations file '{input_manual_annotations_file}' not found or empty. No manual annotations will be merged.")

    # Apply deletion filter to manual annotations as well
    if not df_manual.empty:
        initial_count = len(df_manual)
        df_manual = df_manual[~df_manual.index.isin(dois_to_delete)]
        logger.info(f"Deleted {initial_count - len(df_manual)} papers from manual annotations based on deletion list.")

    # 4. Create the master DataFrame
    # Start with LLM data (which has been filtered by deletion list)
    df_master = df_llm.copy()

    # For papers that are manually reviewed, overwrite LLM data with manual data
    # This logic ensures manual entries take precedence.
    if not df_manual.empty:
        # Get DOIs that are in both the LLM processed data and manual data
        dois_in_both = df_master.index.intersection(df_manual.index)

        # Update columns in df_master for these common DOIs from df_manual
        # This ensures manual annotations overwrite LLM annotations for shared fields
        for col in df_manual.columns:  # Iterate through columns present in manual data
            if col in df_master.columns:  # Only update if column also exists in master
                df_master.loc[dois_in_both, col] = df_manual.loc[dois_in_both, col]
            else:  # If a column is in manual but not in master, add it and fill for common DOIs
                df_master[col] = None  # Initialize new column
                df_master.loc[dois_in_both, col] = df_manual.loc[dois_in_both, col]

        # Add any new manual annotations (DOIs present in manual but not in df_master)
        new_manual_dois = df_manual.index.difference(df_master.index)
        if not new_manual_dois.empty:
            df_master = pd.concat([df_master, df_manual.loc[new_manual_dois]], ignore_index=False)

        logger.info(f"Merged manual annotations. Total papers in master file: {len(df_master)}.")
    else:
        logger.info("No manual annotations to merge.")

    # Reset index before saving if DOI is a column
    df_master.reset_index(inplace=True)

    # Save the master file
    save_jsonl_records(df_master.to_dict(orient='records'), output_master_file, append=False)
    logger.info(f"Final master dataset saved to {output_master_file} with {len(df_master)} records.")
    logger.info("--- Manual Curation Merge Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merges LLM-processed papers with manual annotations and handles deletions."
    )
    parser.add_argument(
        "--input_llm_processed_file",
        type=str,
        required=True,
        help="Path to the input JSONL file with LLM annotated papers."
    )
    parser.add_argument(
        "--input_manual_annotations_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing manual annotations."
    )
    parser.add_argument(
        "--input_papers_to_delete_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing DOIs of papers to delete."
    )
    parser.add_argument(
        "--output_master_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for the final master dataset."
    )
    parser.add_argument(
        "--manual_annotation_score",
        type=float,
        default=1.0,
        help="The annotation score to assign to manually reviewed papers."
    )

    args = parser.parse_args()

    merge_manual_curation_main(
        input_llm_processed_file=args.input_llm_processed_file,
        input_manual_annotations_file=args.input_manual_annotations_file,
        input_papers_to_delete_file=args.input_papers_to_delete_file,
        output_master_file=args.output_master_file,
        manual_annotation_score=args.manual_annotation_score
    )