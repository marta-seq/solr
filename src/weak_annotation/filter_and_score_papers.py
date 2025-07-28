# src/filter_score/filter_and_score_papers.py
import pandas as pd
import argparse
import logging
import os

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/home/martinha/PycharmProjects/phd/review"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records

logger = logging.getLogger(__name__)


def filter_and_score_papers(
        input_file: str,
        output_file: str
):
    """
    Filters papers to keep only those with kw_domain_inclusion = True.
    Sets relevance_score to 100 and adds 50 to annotation_score for these papers.
    All other papers are excluded.
    """
    setup_logging(log_prefix="filter_score")
    logger.info("--- Starting Paper Filtering and Scoring ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    df = load_jsonl_to_dataframe(input_file)
    if df.empty:
        logger.warning(f"Input file '{input_file}' is empty. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False)
        return

    initial_count = len(df)
    logger.info(f"Loaded {initial_count} records for filtering.")

    # DEBUGGING LOGS (these are 'debug' level, so they won't appear in default INFO logs)
    logger.debug(f"DEBUG (filter_score): DataFrame columns after load_jsonl_to_dataframe:\n{df.columns.tolist()}")
    if 'kw_domain_inclusion' in df.columns:
        logger.debug(
            f"DEBUG (filter_score): 'kw_domain_inclusion' dtype before conversion: {df['kw_domain_inclusion'].dtype}")
        logger.debug(
            f"DEBUG (filter_score): 'kw_domain_inclusion' value_counts before conversion:\n{df['kw_domain_inclusion'].value_counts(dropna=False)}")
        logger.debug(
            f"DEBUG (filter_score): First 10 'kw_domain_inclusion' values before conversion:\n{df['kw_domain_inclusion'].head(10).tolist()}")
    else:
        logger.debug("DEBUG (filter_score): 'kw_domain_inclusion' column NOT found after load_jsonl_to_dataframe.")

    # Ensure kw_domain_inclusion column exists and handle NaNs (False for unclassified)
    if 'kw_domain_inclusion' not in df.columns:
        df['kw_domain_inclusion'] = False  # Default to False if column is missing
        logger.warning("No 'kw_domain_inclusion' column found. Assuming False for all papers.")
    else:
        # Convert to boolean, treating NaN/None as False
        df['kw_domain_inclusion'] = df['kw_domain_inclusion'].fillna(False).astype(bool)
        logger.info(
            f"Detected {df['kw_domain_inclusion'].sum()} papers with kw_domain_inclusion=True before filtering.")

    # DEBUGGING LOGS AFTER CONVERSION (these are 'debug' level)
    if 'kw_domain_inclusion' in df.columns:
        logger.debug(
            f"DEBUG (filter_score): 'kw_domain_inclusion' dtype AFTER conversion: {df['kw_domain_inclusion'].dtype}")
        logger.debug(
            f"DEBUG (filter_score): 'kw_domain_inclusion' value_counts AFTER conversion:\n{df['kw_domain_inclusion'].value_counts(dropna=False)}")
        logger.debug(
            f"DEBUG (filter_score): First 10 'kw_domain_inclusion' values AFTER conversion:\n{df['kw_domain_inclusion'].head(10).tolist()}")

    # Filter to keep ONLY papers where kw_domain_inclusion is True
    df_filtered = df[df['kw_domain_inclusion'] == True].copy()
    kept_count = len(df_filtered)
    excluded_count = initial_count - kept_count

    logger.info(f"Filtered papers. Kept {kept_count} papers with kw_domain_inclusion=True.")
    logger.info(
        f"Excluded {excluded_count} papers (kw_domain_inclusion was False or converted to False from None/NaN).")

    # For the kept papers, set relevance_score to 100 and add 50 to annotation_score
    if not df_filtered.empty:
        df_filtered['relevance_score'] = 100
        # Ensure annotation_score is numeric before adding
        df_filtered['annotation_score'] = pd.to_numeric(df_filtered['annotation_score'], errors='coerce').fillna(
            0).astype(int)
        df_filtered['annotation_score'] += 50
        logger.info(f"Set relevance_score=100 and added 50 to annotation_score for {kept_count} included papers.")

    if df_filtered.empty:
        logger.warning("No records remain after filtering. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False)
        return

    ensure_dir(os.path.dirname(output_file))
    save_jsonl_records(df_filtered.to_dict(orient='records'), output_file, append=False)
    logger.info(f"Filtered and scored papers saved to {output_file}.")
    logger.info("--- Paper Filtering and Scoring Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filters papers to keep only those with kw_domain_inclusion = True and sets their scores."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing papers with extracted concepts."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for filtered and scored papers."
    )
    args = parser.parse_args()

    filter_and_score_papers(
        input_file=args.input_file,
        output_file=args.output_file
    )
