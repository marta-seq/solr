# src/data_processing/incorporate_manual_annotations.py
import pandas as pd
import json
import argparse
import os
import logging
import yaml
from datetime import datetime

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, clean_doi

logger = logging.getLogger(__name__)


def flatten_nested_data_for_final_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens nested dictionary and list columns in a DataFrame for final CSV output.
    This is a more comprehensive flattening for the final deployable CSV.
    """
    df_copy = df.copy()

    # Columns that might contain nested dictionaries or lists
    # This list should be comprehensive for all nested structures you want to flatten
    nested_cols_to_flatten = [
        'llm_annot_classification',
        'llm_annot_code_availability_details',
        'llm_annot_data_used_details',  # This is a list of dicts, needs special handling
        'llm_annot_compared_algorithms_packages',  # This is a list of dicts, needs special handling
        'llm_annot_pipeline_analysis_steps',
        'llm_annot_tested_data_modalities',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_primary_programming_languages',
        'llm_annot_assigned_categories'
        # Add any other nested columns you might have
    ]

    for col in nested_cols_to_flatten:
        if col in df_copy.columns and not df_copy[col].dropna().empty:
            # Handle dictionaries (e.g., llm_annot_classification, llm_annot_code_availability_details)
            if isinstance(df_copy[col].dropna().iloc[0], dict):
                try:
                    normalized_df = pd.json_normalize(df_copy[col].apply(lambda x: x if isinstance(x, dict) else {}))
                    normalized_df.columns = [f"{col}.{sub_col}" for sub_col in normalized_df.columns]
                    df_copy = pd.concat([df_copy.drop(columns=[col]), normalized_df], axis=1)
                except Exception as e:
                    logger.warning(f"Failed to normalize dictionary column '{col}': {e}. Converting to string.")
                    df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            # Handle lists of strings (e.g., pipeline_analysis_steps, assigned_categories)
            elif isinstance(df_copy[col].dropna().iloc[0], list) and \
                    all(isinstance(i, str) for i in df_copy[col].dropna().iloc[0]):
                df_copy[col] = df_copy[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
            # Handle lists of dictionaries (e.g., data_used_details, compared_algorithms_packages)
            elif isinstance(df_copy[col].dropna().iloc[0], list) and \
                    len(df_copy[col].dropna().iloc[0]) > 0 and \
                    isinstance(df_copy[col].dropna().iloc[0][0], dict):
                # Convert list of dicts to JSON string for CSV
                df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            # Handle any other list types by converting to string
            elif isinstance(df_copy[col].dropna().iloc[0], list):
                df_copy[col] = df_copy[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

    return df_copy


def incorporate_manual_annotations(input_original_jsonl: str, input_manual_csv: str,
                                   output_csv: str, manual_annotation_score: int):
    """
    Incorporates manual annotations from a CSV back into the original JSONL dataset,
    filters out papers marked for deletion, flattens the result,
    applies a high 'manual_annotation_score', and saves as CSV.
    """
    setup_logging(log_prefix="incorporate_manual_annotations")
    logger.info("--- Starting Incorporation of Manual Annotations ---")
    logger.info(f"Input original JSONL file: {input_original_jsonl}")
    logger.info(f"Input manual CSV file: {input_manual_csv}")
    logger.info(f"Output CSV file: {output_csv}")
    logger.info(f"Manual annotation score to apply: {manual_annotation_score}")

    # 1. Load original JSONL data (contains all fields, including full text sections)
    df_original = load_jsonl_to_dataframe(input_original_jsonl)
    if df_original.empty:
        logger.warning(f"Original JSONL file '{input_original_jsonl}' is empty. Creating empty output CSV.")
        ensure_dir(os.path.dirname(output_csv))
        pd.DataFrame(columns=['doi']).to_csv(output_csv, index=False)  # Ensure DOI column for empty CSV
        return

    df_original['doi'] = df_original['doi'].apply(clean_doi)
    df_original.set_index('doi', inplace=True)
    logger.info(f"Loaded {len(df_original)} records from original JSONL.")

    # 2. Load manual CSV (contains manual edits and deletion flags)
    df_manual = pd.DataFrame()
    if os.path.exists(input_manual_csv) and os.path.getsize(input_manual_csv) > 0:
        try:
            df_manual = pd.read_csv(input_manual_csv, dtype=str,
                                    keep_default_na=False)  # Read all as string to preserve exact input
            df_manual['doi'] = df_manual['doi'].apply(clean_doi)
            df_manual.set_index('doi', inplace=True)
            logger.info(f"Loaded {len(df_manual)} records from manual CSV.")
        except Exception as e:
            logger.error(
                f"Error loading manual CSV file '{input_manual_csv}': {e}. Proceeding without manual annotations.",
                exc_info=True)
    else:
        logger.warning(
            f"Manual CSV file '{input_manual_csv}' not found or empty. Proceeding without manual annotations.")

    # 3. Merge manual annotations into original data (prioritizing manual)
    # Create a copy to modify
    df_combined = df_original.copy()

    if not df_manual.empty:
        # Iterate through manually annotated papers
        for doi, manual_row in df_manual.iterrows():
            if doi in df_combined.index:
                # Update existing paper with manual data
                for col_name, value in manual_row.items():
                    if col_name in df_combined.columns:
                        # Direct overwrite for non-nested columns or flattened ones
                        df_combined.loc[doi, col_name] = value
                    else:
                        # Handle new columns from manual annotation (e.g., 'mark_for_deletion')
                        df_combined.loc[doi, col_name] = value

                # Apply manual annotation score and status
                df_combined.loc[doi, 'annotation_score'] = manual_annotation_score
                df_combined.loc[doi, 'manual_annotation_status'] = 'manually_annotated'
                df_combined.loc[doi, 'manual_annotation_timestamp'] = datetime.now().isoformat()
                logger.debug(f"DOI {doi}: Applied manual annotations and score {manual_annotation_score}.")
            else:
                logger.warning(
                    f"DOI {doi} from manual CSV not found in original JSONL data. Skipping manual update for this DOI.")

    # 4. Filter out papers marked for deletion
    initial_count = len(df_combined)
    if 'mark_for_deletion' in df_combined.columns:
        df_combined = df_combined[df_combined['mark_for_deletion'].astype(str).str.lower() != 'yes']
        logger.info(
            f"Filtered out {initial_count - len(df_combined)} papers marked for deletion. Remaining: {len(df_combined)}.")
    else:
        logger.warning("No 'mark_for_deletion' column found in combined DataFrame. No papers filtered for deletion.")

    # 5. Flatten the combined DataFrame for final CSV output
    df_final_csv = flatten_nested_data_for_final_csv(df_combined.reset_index())  # Reset index for flattening

    # Ensure consistent column order for final CSV
    # Define all possible columns that might appear in the final CSVs
    all_possible_output_cols = [
        'doi', 'pmid', 'title', 'abstract', 'year', 'journal', 'annotation_score',
        'manual_annotation_status', 'manual_annotation_timestamp', 'mark_for_deletion',
        'llm_annot_main_goal_of_paper',
        'llm_annot_classification.is_computational_methods_paper',
        'llm_annot_classification.primary_application_domain',
        'llm_annot_classification.assigned_categories',
        'llm_annot_llm_notes',  # Broad LLM notes
        'llm_annot_package_algorithm_name',
        'llm_annot_pipeline_analysis_steps',
        'llm_annot_tested_data_modalities',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_compared_algorithms_packages',  # This will be a JSON string of list of dicts
        'llm_annot_code_availability_details.status',
        'llm_annot_code_availability_details.link',
        'llm_annot_code_availability_details.license',
        'llm_annot_code_availability_details.version',
        'llm_annot_data_used_details',  # This will be a JSON string of list of dicts
        'llm_annot_primary_programming_languages',
        'llm_annot_llm_notes_detailed'  # Detailed LLM notes (if you want a separate one)
    ]

    # Add any missing expected columns to df_final_csv as None
    for col in all_possible_output_cols:
        if col not in df_final_csv.columns:
            df_final_csv[col] = None

    # Select and reorder columns for consistent output
    df_final_csv = df_final_csv[all_possible_output_cols]

    # 6. Save final CSV
    ensure_dir(os.path.dirname(output_csv))
    df_final_csv.to_csv(output_csv, index=False)
    logger.info(f"Final master CSV saved to {output_csv} with {len(df_final_csv)} entries.")
    logger.info("--- Incorporation of Manual Annotations Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incorporates manual annotations from CSV, filters for deletion, and saves final CSV."
    )
    parser.add_argument(
        "--input_original_jsonl",
        type=str,
        required=True,
        help="Path to the original JSONL file (e.g., intermediate_llm_detailed_annotated_papers.jsonl or intermediate_non_computational_papers.jsonl)."
    )
    parser.add_argument(
        "--input_manual_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing manual annotations."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file with incorporated manual annotations."
    )
    parser.add_argument(
        "--manual_annotation_score",
        type=int,
        required=True,
        help="The score to assign to manually annotated papers to protect them from overwrites."
    )
    args = parser.parse_args()

    incorporate_manual_annotations(
        input_original_jsonl=args.input_original_jsonl,
        input_manual_csv=args.input_manual_csv,
        output_csv=args.output_csv,
        manual_annotation_score=args.manual_annotation_score
    )
