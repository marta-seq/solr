# src/data_processing/prepare_for_manual_review.py
import pandas as pd
import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def prepare_for_manual_review(input_computational_jsonl: str, input_non_computational_jsonl: str,
                              output_computational_jsonl: str, output_non_computational_jsonl: str,
                              broad_llm_target_score: int, detailed_llm_target_score: int):
    """
    Prepares papers for manual review by combining and initializing new papers
    into JSONL files. It appends new papers to existing JSONL files without
    overwriting manual edits.
    """
    setup_logging(log_prefix="prepare_manual_review")
    logger.info("--- Starting Preparation for Manual Review ---")
    logger.info(f"Input computational JSONL: {input_computational_jsonl}")
    logger.info(f"Input non-computational JSONL: {input_non_computational_jsonl}")
    logger.info(f"Output computational JSONL: {output_computational_jsonl}")
    logger.info(f"Output non-computational JSONL: {output_non_computational_jsonl}")
    logger.info(f"Broad LLM target score: {broad_llm_target_score}")
    logger.info(f"Detailed LLM target score: {detailed_llm_target_score}")

    # Ensure output directories exist
    ensure_dir(os.path.dirname(output_computational_jsonl))
    ensure_dir(os.path.dirname(output_non_computational_jsonl))

    # --- Helper to process and save DataFrame to JSONL ---
    def process_and_save_jsonl(df_new_llm_output: pd.DataFrame, output_jsonl_path: str,
                               is_computational_category: bool):
        logger.info(
            f"Processing for {'computational' if is_computational_category else 'non-computational'} category: {output_jsonl_path}")

        if df_new_llm_output.empty:
            logger.warning(f"Input DataFrame for {output_jsonl_path} is empty. No new papers to process.")
            # Ensure the output file exists, even if empty
            save_jsonl_records([], output_jsonl_path, append=False)
            return

        df_new_llm_output['doi'] = df_new_llm_output['doi'].apply(clean_doi)
        df_new_llm_output.drop_duplicates(subset=['doi'], inplace=True)
        logger.info(f"Cleaned and deduplicated input: {len(df_new_llm_output)} unique papers.")

        df_existing_manual_review = load_jsonl_to_dataframe(output_jsonl_path)
        existing_dois = set(df_existing_manual_review['doi'].tolist()) if not df_existing_manual_review.empty else set()
        logger.info(f"Found {len(existing_dois)} existing papers in manual review file '{output_jsonl_path}'.")

        # Filter for truly new papers not already in the manual review file
        df_to_add = df_new_llm_output[~df_new_llm_output['doi'].isin(existing_dois)].copy()
        logger.info(
            f"Identified {len(df_to_add)} new papers to add to manual review JSONL for {'computational' if is_computational_category else 'non-computational'} category.")

        if not df_to_add.empty:
            # Initialize manual review columns for new papers
            df_to_add['manual_review_status'] = 'pending'  # New papers are pending review
            df_to_add['manual_notes'] = ''
            df_to_add['manual_reviewer_id'] = ''
            df_to_add['manual_review_timestamp'] = ''

            # Set the initial annotation_score based on the LLM stage they came from
            for idx in df_to_add.index:
                llm_status = df_to_add.loc[idx, 'llm_annotation_status']
                if llm_status == 'detailed_llm_annotated':
                    df_to_add.loc[idx, 'annotation_score'] = float(detailed_llm_target_score)
                elif llm_status == 'middle_llm_annotated':  # Broad LLM annotated
                    df_to_add.loc[idx, 'annotation_score'] = float(broad_llm_target_score)
                else:
                    # If status is something else (e.g., 'llm_error_json_parse'), ensure score is float
                    df_to_add.loc[idx, 'annotation_score'] = float(
                        df_to_add.loc[idx, 'annotation_score']) if 'annotation_score' in df_to_add.columns and pd.notna(
                        df_to_add.loc[idx, 'annotation_score']) else 0.0

            # Combine existing and new papers.
            # Existing manual edits are preserved because df_existing_manual_review is loaded first.
            # New papers are appended.
            combined_records = df_existing_manual_review.to_dict(orient='records') + df_to_add.to_dict(orient='records')

            # Save the combined set, overwriting the file to ensure consistency
            save_jsonl_records(combined_records, output_jsonl_path, append=False)
            logger.info(
                f"Successfully updated '{output_jsonl_path}' with {len(df_to_add)} new papers. Total records: {len(combined_records)}.")
        else:
            logger.info(
                f"No new papers to add to '{output_jsonl_path}'. Current total records: {len(df_existing_manual_review)}.")
            # Ensure file exists even if no new papers were added and it was previously empty
            if not os.path.exists(output_jsonl_path) or os.path.getsize(output_jsonl_path) == 0:
                save_jsonl_records([], output_jsonl_path, append=False)

    # --- Process Computational Papers (from detailed LLM output) ---
    logger.info("Loading computational papers from detailed LLM output...")
    df_comp_new_llm = load_jsonl_to_dataframe(input_computational_jsonl)
    process_and_save_jsonl(df_comp_new_llm, output_computational_jsonl, is_computational_category=True)

    # --- Process Non-Computational Papers (from broad LLM split) ---
    logger.info("Loading non-computational papers from broad LLM split...")
    df_non_comp_new_llm = load_jsonl_to_dataframe(input_non_computational_jsonl)
    process_and_save_jsonl(df_non_comp_new_llm, output_non_computational_jsonl, is_computational_category=False)

    logger.info("--- Preparation for Manual Review Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepares papers for manual review by combining and initializing new papers into JSONL files."
    )
    parser.add_argument(
        "--input_computational_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file of computational papers (from detailed LLM output)."
    )
    parser.add_argument(
        "--input_non_computational_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file of non-computational papers (from broad LLM split)."
    )
    parser.add_argument(
        "--output_computational_jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file for computational papers for manual review."
    )
    parser.add_argument(
        "--output_non_computational_jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file for non-computational papers for manual review."
    )
    parser.add_argument(
        "--broad_llm_target_score",
        type=int,
        required=True,
        help="The target score assigned to papers after broad LLM annotation."
    )
    parser.add_argument(
        "--detailed_llm_target_score",
        type=int,
        required=True,
        help="The target score assigned to papers after detailed LLM annotation."
    )
    args = parser.parse_args()

    prepare_for_manual_review(
        input_computational_jsonl=args.input_computational_jsonl,
        input_non_computational_jsonl=args.input_non_computational_jsonl,
        output_computational_jsonl=args.output_computational_jsonl,
        output_non_computational_jsonl=args.output_non_computational_jsonl,
        broad_llm_target_score=args.broad_llm_target_score,
        detailed_llm_target_score=args.detailed_llm_target_score
    )