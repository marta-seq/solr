# src/llm_annotation/filter_computational_papers.py
import pandas as pd
import json
import os
import argparse
import logging
from typing import Dict, Any
import sys
# Add project root to path for utility imports (assuming script runs from project root or similar)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def filter_computational_papers_main(input_file: str, output_file: str):
    """
    Loads papers from the broad LLM annotation output and filters for those
    classified as computational methods papers.
    """
    setup_logging(log_prefix="filter_comp_papers")
    logger.info("--- Starting Computational Paper Filtering ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    df = load_jsonl_to_dataframe(input_file)
    if df.empty:
        logger.warning(f"Input file '{input_file}' is empty. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False)
        return

    df['doi'] = df['doi'].apply(clean_doi)

    # Filter for papers where 'llm_annot_classification.is_computational_methods_paper' is True
    # Handle cases where 'llm_annot_classification' might be missing or not a dict

    # First, ensure the classification column is treated as a dict or empty dict
    df['llm_annot_classification_safe'] = df['llm_annot_classification'].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    # Then, filter based on the 'is_computational_methods_paper' key
    computational_papers_df = df[
        df['llm_annot_classification_safe'].apply(
            lambda x: x.get('is_computational_methods_paper') == True
        )
    ].copy()

    # Drop the temporary safe column
    computational_papers_df.drop(columns=['llm_annot_classification_safe'], inplace=True, errors='ignore')

    if computational_papers_df.empty:
        logger.info("No computational methods papers found. Creating empty output file.")
    else:
        logger.info(f"Filtered {len(computational_papers_df)} computational methods papers.")

    ensure_dir(os.path.dirname(output_file))
    save_jsonl_records(computational_papers_df.to_dict(orient='records'), output_file, append=False)
    logger.info("--- Computational Paper Filtering Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filters papers classified as computational methods from LLM broad annotation output."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file with LLM broad annotated papers."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for computational methods papers."
    )
    args = parser.parse_args()
    filter_computational_papers_main(args.input_file, args.output_file)