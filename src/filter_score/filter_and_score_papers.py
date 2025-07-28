# src/filter_score/filter_and_score_papers.py
import pandas as pd
import argparse
import logging
import os
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


def filter_and_score_papers(input_file: str, output_file: str, base_annotation_score: int):
    """
    Filters papers based on 'kw_domain_inclusion' and assigns an initial annotation score.
    """
    setup_logging(log_prefix="filter_score")
    logger.info("--- Starting Filter and Score Papers Process ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Base annotation score: {base_annotation_score}")

    df = load_jsonl_to_dataframe(input_file)
    if df.empty:
        logger.warning(f"Input file '{input_file}' is empty. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False)
        return

    df['doi'] = df['doi'].apply(clean_doi)

    # Ensure 'annotation_score' column exists and is numeric
    if 'annotation_score' not in df.columns:
        df['annotation_score'] = 0.0
    df['annotation_score'] = df['annotation_score'].fillna(0.0).astype(float)

    # Filter based on kw_domain_inclusion
    # Papers with kw_domain_inclusion = True are kept
    # Papers with kw_domain_inclusion = False or missing/ambiguous are excluded

    # Convert kw_domain_inclusion to boolean, handling missing/non-boolean values as False
    df['kw_domain_inclusion_bool'] = df['kw_domain_inclusion'].apply(
        lambda x: True if isinstance(x, bool) and x else False
    )

    filtered_df = df[df['kw_domain_inclusion_bool']].copy()
    excluded_df = df[~df['kw_domain_inclusion_bool']].copy()

    logger.info(f"Filtered papers. Kept {len(filtered_df)} papers with kw_domain_inclusion=True.")
    logger.info(
        f"Excluded {len(excluded_df)} papers (kw_domain_inclusion was False or converted to False from None/NaN).")

    if filtered_df.empty:
        logger.warning("No records remain after filtering. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False)
        return

    # Assign base annotation score to filtered papers
    filtered_df['annotation_score'] = base_annotation_score
    logger.info(f"Assigned base annotation score of {base_annotation_score} to filtered papers.")

    # Update pipeline_category for filtered papers
    filtered_df['pipeline_category'] = 'keyword_filtered_and_scored'

    ensure_dir(os.path.dirname(output_file))
    save_jsonl_records(filtered_df.to_dict(orient='records'), output_file, append=False)
    logger.info(f"Saved {len(filtered_df)} filtered and scored papers to {output_file}.")
    logger.info("--- Filter and Score Papers Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filters papers based on 'kw_domain_inclusion' and assigns an initial annotation score."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file with extracted concepts."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for filtered and scored papers."
    )
    parser.add_argument(
        "--base_annotation_score",
        type=int,
        required=True,
        help="The base score to assign to papers that pass the initial filtering."
    )
    args = parser.parse_args()

    filter_and_score_papers(
        input_file=args.input_file,
        output_file=args.output_file,
        base_annotation_score=args.base_annotation_score
    )