# /home/martinha/PycharmProjects/phd/review_test_run/src/data_acquisition/preprocess_curated_excel.py
import pandas as pd
import argparse
import logging
import os

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'review' is the actual project root containing 'src'
project_root = "/home/martinha/PycharmProjects/phd/review"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import clean_doi  # Only need clean_doi here

logger = logging.getLogger(__name__)


def preprocess_excel_to_csv(input_excel_path: str, output_csv_path: str):
    """
    Loads paper classification data from an Excel file, cleans DOIs,
    and saves the cleaned data to a CSV file.
    """
    setup_logging(log_prefix="preprocess_excel")
    logger.info("--- Starting Excel Preprocessing to CSV ---")
    logger.info(f"Input Excel file: {input_excel_path}")
    logger.info(f"Output Cleaned CSV file: {output_csv_path}")

    if not os.path.exists(input_excel_path):
        logger.error(f"Input Excel file not found: '{input_excel_path}'. Cannot proceed.")
        # Create an empty output CSV to prevent downstream errors if input is missing
        ensure_dir(os.path.dirname(output_csv_path))
        pd.DataFrame().to_csv(output_csv_path, index=False)
        return

    try:
        df = pd.read_excel(input_excel_path)
        logger.info(f"Loaded {len(df)} records from '{input_excel_path}'.")

        if 'doi' in df.columns:
            df['doi'] = df['doi'].apply(clean_doi)
            logger.info("Cleaned 'doi' column.")
        else:
            logger.warning("No 'doi' column found in Excel file. Skipping DOI cleaning.")

        ensure_dir(os.path.dirname(output_csv_path))
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Successfully saved cleaned data to '{output_csv_path}'.")

    except Exception as e:
        logger.error(f"Error during Excel preprocessing: {e}", exc_info=True)
        # Ensure an empty CSV is created on error to prevent downstream failures
        ensure_dir(os.path.dirname(output_csv_path))
        pd.DataFrame().to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses an Excel file containing curated paper classifications, cleans DOIs, and saves to CSV."
    )
    parser.add_argument(
        "--input_excel_file",
        type=str,
        required=True,
        help="Path to the input Excel file (e.g., paper_classification.xlsx)."
    )
    parser.add_argument(
        "--output_csv_file",
        type=str,
        required=True,
        help="Path to the output CSV file for cleaned data (e.g., paper_classification_cleaned.csv)."
    )
    args = parser.parse_args()

    preprocess_excel_to_csv(
        input_excel_path=args.input_excel_file,
        output_csv_path=args.output_csv_file
    )