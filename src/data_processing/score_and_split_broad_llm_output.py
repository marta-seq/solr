# src/data_processing/score_and_split_broad_llm_output.py
import pandas as pd
import argparse
import logging
import os
import json
from datetime import datetime

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


def score_and_split_broad_llm_output(input_file: str, output_computational_file: str,
                                     output_non_computational_file: str, broad_llm_target_score: int,
                                     llm_broad_schema_path: str):
    """
    Scores papers after broad LLM annotation and splits them into computational
    and non-computational categories based on LLM classification.
    """
    setup_logging(log_prefix="score_split_broad_llm")
    logger.info("--- Starting Score and Split Broad LLM Output Process ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output computational file: {output_computational_file}")
    logger.info(f"Output non-computational file: {output_non_computational_file}")
    logger.info(f"Broad LLM target score: {broad_llm_target_score}")

    df = load_jsonl_to_dataframe(input_file)
    if df.empty:
        logger.warning(f"Input file '{input_file}' is empty. Creating empty output files.")
        ensure_dir(os.path.dirname(output_computational_file))
        ensure_dir(os.path.dirname(output_non_computational_file))
        save_jsonl_records([], output_computational_file, append=False)
        save_jsonl_records([], output_non_computational_file, append=False)
        return

    df['doi'] = df['doi'].apply(clean_doi)

    # Load the broad LLM schema (for logging/validation, not direct use in parsing)
    try:
        with open(llm_broad_schema_path, 'r') as f:
            llm_broad_schema = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"LLM Broad Schema file not found at '{llm_broad_schema_path}'. Cannot validate LLM output structure.")
        llm_broad_schema = {}
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding LLM Broad Schema from '{llm_broad_schema_path}': {e}. Cannot validate LLM output structure.")
        llm_broad_schema = {}

    # Ensure 'annotation_score' column exists and is numeric
    if 'annotation_score' not in df.columns:
        df['annotation_score'] = 0.0
    df['annotation_score'] = df['annotation_score'].fillna(0.0).astype(float)

    computational_papers = []
    non_computational_papers = []

    for index, row in df.iterrows():
        is_computational = False
        doi = row.get('doi', 'N/A')

        llm_classification_raw = row.get('llm_annot_classification')
        llm_classification = {}

        if isinstance(llm_classification_raw, str):
            try:
                llm_classification = json.loads(llm_classification_raw)
                logger.debug(f"DOI {doi}: Successfully parsed llm_annot_classification from string.")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"DOI {doi}: Could not parse llm_annot_classification string as JSON: {e}. Raw: {llm_classification_raw[:100]}...")
                llm_classification = {}
        elif isinstance(llm_classification_raw, dict):
            llm_classification = llm_classification_raw
        elif llm_classification_raw is None:
            logger.warning(
                f"DOI {doi}: 'llm_annot_classification' is None. Defaulting to empty dict and non-computational.")
            llm_classification = {}
        else:
            logger.warning(
                f"DOI {doi}: 'llm_annot_classification' is unexpected type ({type(llm_classification_raw)}). Defaulting to empty dict and non-computational. Raw: {llm_classification_raw}")
            llm_classification = {}

        is_relevant_to_spatial_omics = llm_classification.get('is_relevant_to_spatial_omics_analysis')

        if isinstance(is_relevant_to_spatial_omics, bool):
            is_computational = is_relevant_to_spatial_omics
        else:
            logger.warning(
                f"DOI {doi}: 'is_relevant_to_spatial_omics_analysis' status is ambiguous ({is_relevant_to_spatial_omics}). Placing in non-computational.")

        # Set annotation score based on broad LLM classification
        if row.get('llm_annotation_status') == 'middle_llm_annotated':
            row['annotation_score'] = float(broad_llm_target_score)
            logger.debug(f"DOI {doi}: Set annotation score to {broad_llm_target_score} after broad LLM annotation.")
        else:
            logger.debug(f"DOI {doi}: Skipping score update for broad LLM due to status '{row.get('llm_annotation_status')}'.")


        if is_computational:
            computational_papers.append(row.to_dict())
        else:
            non_computational_papers.append(row.to_dict())

    df_computational = pd.DataFrame(computational_papers)
    df_non_computational = pd.DataFrame(non_computational_papers)

    ensure_dir(os.path.dirname(output_computational_file))
    ensure_dir(os.path.dirname(output_non_computational_file))

    save_jsonl_records(df_computational.to_dict(orient='records'), output_computational_file, append=False)
    logger.info(f"Saved {len(df_computational)} computational papers to {output_computational_file}.")

    save_jsonl_records(df_non_computational.to_dict(orient='records'), output_non_computational_file, append=False)
    logger.info(f"Saved {len(df_non_computational)} non-computational papers to {output_non_computational_file}.")

    logger.info("--- Score and Split Broad LLM Output Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scores papers and splits them into computational/non-computational categories."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file with LLM broad annotations."
    )
    parser.add_argument(
        "--output_computational_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for computational papers."
    )
    parser.add_argument(
        "--output_non_computational_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for non-computational papers."
    )
    parser.add_argument(
        "--broad_llm_target_score", # Changed from score_increment
        type=int,
        required=True,
        help="The target score for papers after broad LLM annotation."
    )
    parser.add_argument(
        "--llm_broad_schema_path",
        type=str,
        required=True,
        help="Path to the JSON schema for broad LLM classification."
    )
    args = parser.parse_args()

    score_and_split_broad_llm_output(
        input_file=args.input_file,
        output_computational_file=args.output_computational_file,
        output_non_computational_file=args.output_non_computational_file,
        broad_llm_target_score=args.broad_llm_target_score,
        llm_broad_schema_path=args.llm_broad_schema_path
    )