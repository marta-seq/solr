# explore_annotations.py

import pandas as pd
import json
import os
import logging
from collections import Counter
from datetime import datetime

# --- Configuration (should match your weak_annotator.py) ---
BASE_DIR = "../../data"
WEAK_ANNOTATION_POOL_FILE = os.path.join(BASE_DIR, "weak_annotation_pool.jsonl")

# --- Basic Logging for this script ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_weak_annotations_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads weak annotations from a .jsonl file into a Pandas DataFrame.
    Normalizes the 'weak_annotations' nested field.
    """
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} at line {line_num}: {line.strip()}")
    else:
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame() # Return an empty DataFrame

    if not data:
        logger.info(f"No data found in {file_path}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Load the main data
    df = pd.DataFrame(data)

    # Normalize the 'weak_annotations' column if it exists
    if 'weak_annotations' in df.columns:
        # json_normalize flattens the nested dictionary into new columns, prefixed by 'weak_annotations.'
        # If 'record_path' is specified, it flattens lists of dictionaries within the nested structure.
        # Here, 'weak_annotations' is a dict, so we just pass it as data to normalize.
        weak_anns_df = pd.json_normalize(df['weak_annotations'])

        # Concatenate the flattened weak_annotations with the original DataFrame,
        # dropping the original 'weak_annotations' column to avoid redundancy.
        df = pd.concat([df.drop('weak_annotations', axis=1), weak_anns_df], axis=1)

    logger.info(f"Successfully loaded {len(df)} records into DataFrame.")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")

    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting DataFrame exploration...")

    df_annotations = load_weak_annotations_to_dataframe(WEAK_ANNOTATION_POOL_FILE)

    if not df_annotations.empty:
        # Display basic information
        print("\n--- DataFrame Info ---")
        df_annotations.info()

        print("\n--- First 5 rows ---")
        print(df_annotations.head())

        print("\n--- Basic Statistics (e.g., relevance and paper type counts) ---")
        print("\nRelevance Score Counts:")
        print(df_annotations['relevance_score_llm'].value_counts())

        print("\nPaper Type Counts:")
        print(df_annotations['paper_type'].value_counts())

        # Example: Filter for 'method' papers and see their methodology types
        print("\n--- Methodology Type Counts for 'method' papers ---")
        method_papers = df_annotations[df_annotations['paper_type'] == 'method']
        if not method_papers.empty:
            print(method_papers['methodology_type'].value_counts())
        else:
            print("No 'method' papers found.")

        # Example: Count occurrences of each data modality
        print("\n--- Data Modalities Counts ---")
        # For list columns, you need to "explode" them or use apply with Counter
        all_modalities = Counter()
        for modalities_list in df_annotations['data_modalities_used'].dropna():
            all_modalities.update(modalities_list)
        print(all_modalities.most_common(10)) # Top 10 most common modalities

    logger.info("DataFrame exploration complete.")
    # --- How to "explore it a little bit" further: ---
    # 1. Run this script in an interactive Python environment (like a Jupyter Notebook or Google Colab).
    #    In such an environment, you can just run the cells, and then `df_annotations` will be available
    #    for you to type commands like:
    #    df_annotations.columns # See all column names
    #    df_annotations[df_annotations['relevance_score_llm'] == 'highly_relevant'].head() # Filter and view
    #    df_annotations.groupby('paper_type')['llm_model'].value_counts() # More complex grouping

    # 2. Add more print statements or save the DataFrame:
    #    If running as a standalone script, you can add more lines like:
    #    print("\n--- Papers marked as highly relevant ---")
    #    print(df_annotations[df_annotations['relevance_score_llm'] == 'highly_relevant'].to_string()) # to_string for full view

    #    You can also save it to other formats for external exploration:

    df_annotations.to_csv(os.path.join(BASE_DIR, "weak_annotations_flat.csv"), index=False)
    # df_annotations.to_excel(os.path.join(BASE_DIR, "weak_annotations_flat.xlsx"), index=False)
    logger.info(f"DataFrame saved to {os.path.join(BASE_DIR, 'weak_annotations_flat.csv')}")
