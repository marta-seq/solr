# src/utils/data_helpers.py
import json
import pandas as pd
import re
import logging
import os
from typing import List, Dict, Any, Optional

# Add project root to path for utility imports (needed for relative imports below)
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- CRITICAL FIX: Correct import for ensure_dir ---
from src.utils.file_helpers import ensure_dir

logger = logging.getLogger(__name__)


def clean_doi(doi: str) -> str:
    """Cleans a DOI string by removing common prefixes and ensuring lowercase."""
    if pd.isna(doi):
        return ""
    doi = str(doi).strip().lower()
    # Remove common DOI prefixes
    doi = re.sub(r"^(doi:|https?://dx\.doi\.org/|https?://doi\.org/)", "", doi)
    return doi


def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads a JSONL file into a pandas DataFrame.
    Handles empty or non-existent files gracefully.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.info(f"File not found or empty: {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from line in {file_path}: {line[:100]}... Error: {e}")
                        continue
    except Exception as e:
        logger.error(f"Error reading JSONL file {file_path}: {e}")
        return pd.DataFrame()

    if not records:
        logger.info(f"No valid records found in {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records from {file_path}.")
    return df


def save_jsonl_records(records: List[Dict[str, Any]], file_path: str, append: bool = True):
    """
    Saves a list of dictionaries to a JSONL file.
    If append is True, appends to the file. Otherwise, overwrites it.
    Ensures directory exists.
    """
    ensure_dir(os.path.dirname(file_path))  # This line now has ensure_dir imported correctly

    mode = 'a' if append else 'w'
    if not records and not append and (not os.path.exists(file_path) or os.path.getsize(file_path) > 0):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass  # Truncate file
        logger.info(f"Created empty file {file_path} as no records to save.")
        return
    elif not records and append:
        logger.info(f"No records to append to {file_path}.")
        return
    elif not records:
        logger.info(f"No records to save to {file_path}.")
        return

    try:
        with open(file_path, mode, encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(records)} entries to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving records to {file_path}: {e}")


def load_curated_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads curated data from a CSV file, cleans DOIs, and handles potential errors.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.warning(f"Curated CSV file not found or empty: {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
        if 'doi' in df.columns:
            df['doi'] = df['doi'].apply(clean_doi)
            df.drop_duplicates(subset=['doi'], inplace=True)
            logger.info(f"Loaded {len(df)} unique records from curated CSV: {file_path}.")
        else:
            logger.warning(
                f"Curated CSV '{file_path}' does not contain a 'doi' column. Skipping DOI cleaning and deduplication.")
        return df
    except Exception as e:
        logger.error(f"Error loading or processing curated CSV '{file_path}': {e}")
        return pd.DataFrame()


# Keeping these functions for potential future use or if user changes mind about CSV,
# but they won't be used in the new JSONL-based manual review workflow.
def flatten_json_to_dataframe(df: pd.DataFrame, nested_cols: List[str], prefix: str = '') -> pd.DataFrame:
    """
    Flattens specified nested JSON columns into top-level columns in a DataFrame.
    """
    df_flat = df.copy()

    for col in nested_cols:
        if col in df_flat.columns and df_flat[col].apply(lambda x: isinstance(x, dict)).any():
            flat_data = df_flat[col].apply(lambda x: x if isinstance(x, dict) else {})
            temp_df = pd.json_normalize(flat_data, sep='_')
            temp_df.columns = [f"{prefix}{col}_{sub_col}" for sub_col in temp_df.columns]
            df_flat = pd.concat([df_flat.drop(columns=[col]), temp_df], axis=1)
            logger.debug(f"Flattened column: {col}")
        elif col in df_flat.columns and df_flat[col].apply(lambda x: isinstance(x, list)).any():
            logger.warning(f"Column '{col}' contains lists, not dictionaries. Skipping flattening for this column.")
        elif col in df_flat.columns:
            logger.debug(f"Column '{col}' is not a dictionary column or is empty. Skipping flattening.")
        else:
            logger.debug(f"Column '{col}' not found in DataFrame. Skipping flattening.")

    return df_flat


def nest_dataframe_to_json(df: pd.DataFrame, schema: Dict[str, Any],
                           top_level_prefix: str = 'llm_annot_') -> pd.DataFrame:
    """
    Nests flattened columns back into their original JSON structure based on a schema.
    """
    df_nested = df.copy()

    for prop_name, prop_details in schema.get('properties', {}).items():
        prefixed_prop_name = f"{top_level_prefix}{prop_name}"
        related_flat_columns = [col for col in df_nested.columns if col.startswith(f"{prefixed_prop_name}_")]

        if not related_flat_columns:
            if prefixed_prop_name in df_nested.columns:
                logger.debug(f"No flattened columns for '{prefixed_prop_name}'. Keeping existing column.")
            continue

        new_nested_series = []
        for index, row in df_nested.iterrows():
            nested_obj = {}
            for flat_col in related_flat_columns:
                sub_path_parts = flat_col[len(prefixed_prop_name) + 1:].split('_')

                current_level = nested_obj
                for i, part in enumerate(sub_path_parts):
                    if i == len(sub_path_parts) - 1:
                        current_level[part] = row[flat_col]
                    else:
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]
            new_nested_series.append(nested_obj)

        df_nested[prefixed_prop_name] = new_nested_series
        df_nested.drop(columns=related_flat_columns, inplace=True)
        logger.debug(f"Re-nested column: {prefixed_prop_name}")

    return df_nested