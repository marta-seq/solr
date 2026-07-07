import pandas as pd
import json
import os
import logging
from collections import Counter
from datetime import datetime
import numpy as np
import re
import hashlib # New import for hashing
import sys

# --- Helper Functions for Hashing and Map Handling ---

def calculate_file_md5(filepath):
    """Calculates the MD5 hash of a file's content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logger.warning(f"File not found for hashing: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error calculating MD5 for {filepath}: {e}")
        return None

def get_last_recorded_map_hash(record_filepath):
    """Loads the last recorded map hash from a JSON file."""
    if os.path.exists(record_filepath):
        try:
            with open(record_filepath, 'r') as f:
                data = json.load(f)
                return data.get('last_map_hash')
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not read or parse map hash record file {record_filepath}: {e}. Treating as no prior hash.")
            return None
    return None

def save_current_map_hash(record_filepath, current_hash):
    """Saves the current map hash to a JSON file."""
    try:
        with open(record_filepath, 'w') as f:
            json.dump({'last_map_hash': current_hash}, f)
        logger.debug(f"Saved current map hash {current_hash} to {record_filepath}")
    except Exception as e:
        logger.error(f"Error saving map hash record to {record_filepath}: {e}")

# Function to convert regex patterns to strings for logging
def serialize_map_for_logging(keyword_map):
    """Converts a keyword map (potentially containing re.Pattern objects) to a JSON-serializable dictionary."""
    serialized_map = {}
    for label, patterns in keyword_map.items():
        serialized_patterns = []
        for p in patterns:
            if isinstance(p, re.Pattern):
                serialized_patterns.append(p.pattern) # Get the regex string from the pattern object
            else:
                serialized_patterns.append(p)
        serialized_map[label] = serialized_patterns
    return serialized_map




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
    return df

# def apply_keyword_labels(term_list, keyword_map):
#     """
#     Replace terms in the list based on keyword presence, ignoring case,
#     hyphens, and parentheses.
#
#     Args:
#         term_list (list of str): Original list of terms.
#         keyword_map (dict): Keys are replacement labels, values are lists of keywords.
#
#     Returns:
#         List of str: List with keywords replaced by labels.
#     """
#     updated_terms = []
#
#     def normalize(text):
#         return re.sub(r'[^a-z0-9]', '', text.lower())
#
#     for term in term_list:
#         norm_term = normalize(term)
#         matched = False
#
#         for label, keywords in keyword_map.items():
#             for keyword in keywords:
#                 norm_keyword = normalize(keyword)
#                 if norm_keyword in norm_term:
#                     updated_terms.append(label)
#                     matched = True
#                     break
#             if matched:
#                 break
#
#         if not matched:
#             updated_terms.append(term)
#     updated_terms = list(set(updated_terms))
#     return updated_terms

def apply_keyword_labels(term_list, keyword_map):
    """
    Replace terms in the list based on keyword presence, ignoring case,
    hyphens, and parentheses. Handles both string and re.Pattern keywords.

    Args:
        term_list (list of str): Original list of terms.
        keyword_map (dict): Keys are replacement labels, values are lists of keywords (str or re.Pattern).

    Returns:
        List of str: List with keywords replaced by labels.
    """
    updated_terms = []

    def normalize_text(text):
        """Helper to normalize text for comparison."""
        return re.sub(r'[^a-z0-9]', '', text.lower())

    if not isinstance(term_list, list): # Ensure term_list is a list
        return []

    for term in term_list:
        norm_term = normalize_text(term)
        matched = False

        for label, keywords in keyword_map.items():
            for keyword_pattern in keywords:
                # Get the string representation of the keyword/pattern for normalization
                keyword_str = keyword_pattern.pattern if isinstance(keyword_pattern, re.Pattern) else keyword_pattern
                norm_keyword = normalize_text(keyword_str)

                # Use direct substring check after normalization as per your original apply_keyword_labels logic
                if norm_keyword in norm_term:
                    updated_terms.append(label)
                    matched = True
                    break # Found a match for this term in this label category
            if matched:
                break # Move to the next term in term_list

        if not matched:
            updated_terms.append(term)

    # Use set to ensure uniqueness of assigned labels (as per your original code's final set conversion)
    return list(set(updated_terms))


def harmonize_context(context_string, keyword_map):
    """
    Applies a given keyword map to a single biological_context string.
    Returns the first matching label from the map, or np.nan if no match.
    Handles various forms of missing/empty input.
    """
    # Normalize input: Treat actual None, numpy.nan, empty string, or whitespace string as missing
    if pd.isna(context_string) or not isinstance(context_string, str) or context_string.strip() == '':
        return np.nan

    # Further normalize for matching by replacing common missing string representations
    normalized_context = context_string.lower().replace('n/a', '').replace('none', '').strip()

    # If it becomes empty after normalization, it's still considered missing
    if not normalized_context:
        return np.nan

    for label, patterns in keyword_map.items():
        for pattern in patterns:
            # Check if the pattern is a pre-compiled regex object
            if isinstance(pattern, re.Pattern):
                if pattern.search(normalized_context):
                    return label
            # If it's a string, treat it as a literal keyword for direct substring search
            # (assuming explicit regex is used for complex patterns)
            elif isinstance(pattern, str):
                # For string patterns, default to full word boundary unless pattern implies partial
                if re.search(r'\b' + re.escape(pattern) + r'\b', normalized_context):
                     return label
                # Consider adding a fallback if you have string patterns that are meant for partial match
                # e.g., if pattern in normalized_context: return label
    return np.nan # No match found after checking all patterns

def print_unassigned_for_column(df, column_name):
    """
    Identifies and prints unique original 'biological_context' entries
    where the specified new column is NaN.
    """
    unassigned_mask = df[column_name].isna()
    unassigned_contexts = df[unassigned_mask]['biological_context']

    print(f"\n--- Unassigned 'biological_context' for '{column_name}' column ---")
    if not unassigned_contexts.empty:
        print(f"The following original 'biological_context' entries resulted in NaN for '{column_name}':")
        for i, context in enumerate(unassigned_contexts.drop_duplicates()):
            print(f"- {context}")
    else:
        print(f"All 'biological_context' entries were assigned a value for '{column_name}'.")



# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    BASE_DIR = "../../data"  # Make sure this path is correct for your environment
    WEAK_ANNOTATION_POOL_FILE = os.path.join(BASE_DIR, "weak_annotation_pool.jsonl")
    OUTPUT_CSV_FILE = os.path.join(BASE_DIR,
                                   "weak_annotation_pool_application_papers_harmonized.csv")  # Renamed for clarity

    # New configuration for mappings file and hash record
    KEYWORD_DEFS_FILE = os.path.join(BASE_DIR, "keyword_definitions.py")
    MAP_HASH_RECORD_FILE = os.path.join(BASE_DIR, ".map_hash_record.json")  # Hidden file for hash record

    # Log file named with timestamp
    LOG_FILE = os.path.join(BASE_DIR, f"living_review_annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # --- Set up Logging ---
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(LOG_FILE)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Prevent duplicate handlers if the script is run multiple times in the same session (e.g., in IDEs/Jupyter)
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)


    logger.info(f"Script started at {datetime.now()}")

    # --- Load Mappings from separate file ---
    # Add BASE_DIR to Python path to allow importing keyword_definitions as a module
    sys.path.insert(0, BASE_DIR)
    try:
        # import keyword_definitions as kd
        import harmonize_maps_application as kd

        # Assign the imported maps to the global scope for use in this script
        globals().update(
            {k: getattr(kd, k) for k in ['ANIMAL_MAP', 'TISSUE_MAP', 'DISEASE_MAP', 'keyword_map_methods', 'keyword_map_methods_general']}
        )
        logger.info(f"Successfully loaded keyword mappings from {KEYWORD_DEFS_FILE}")
    except ImportError as e:
        logger.error(f"Error loading keyword definitions from {KEYWORD_DEFS_FILE}: {e}")
        logger.error("Please ensure 'keyword_definitions.py' exists in the data directory and is syntactically correct.")
        sys.exit(1) # Exit if mappings cannot be loaded
    finally:
        sys.path.pop(0) # Remove BASE_DIR from path after import


    # --- Check for Changes in Keyword Mappings and Log if necessary ---
    current_map_hash = calculate_file_md5(KEYWORD_DEFS_FILE)
    last_recorded_map_hash = get_last_recorded_map_hash(MAP_HASH_RECORD_FILE)

    if current_map_hash is None:
        logger.error("Could not calculate hash for keyword definitions file. Cannot check for changes.")
    elif current_map_hash != last_recorded_map_hash:
        logger.info(f"\n--- Changes detected in {KEYWORD_DEFS_FILE} (or first run)! Logging current mappings at {datetime.now()} ---")
        # Log Keyword Mappings
        logger.info("\nANIMAL_MAP:")
        logger.info(json.dumps(serialize_map_for_logging(ANIMAL_MAP), indent=2))
        logger.info("\nTISSUE_MAP:")
        logger.info(json.dumps(serialize_map_for_logging(TISSUE_MAP), indent=2))
        logger.info("\nDISEASE_MAP:")
        logger.info(json.dumps(serialize_map_for_logging(DISEASE_MAP), indent=2))
        logger.info("\nkeyword_map_methods:")
        logger.info(json.dumps(keyword_map_methods, indent=2))
        logger.info("\nkeyword_map_methods_general:")
        logger.info(json.dumps(keyword_map_methods_general, indent=2))

        # Update the recorded hash
        save_current_map_hash(MAP_HASH_RECORD_FILE, current_map_hash)
        logger.info(f"Updated map hash record in {MAP_HASH_RECORD_FILE}")
    else:
        logger.info(f"\nNo changes detected in {KEYWORD_DEFS_FILE}. Using existing mappings.")

    # --- Rest of your data processing logic ---
    logger.info(f"Loading weak annotations from {WEAK_ANNOTATION_POOL_FILE}")
    df_annotations = load_weak_annotations_to_dataframe(WEAK_ANNOTATION_POOL_FILE)

    if df_annotations.empty:
        logger.error("No data loaded. Exiting script.")
        sys.exit(1) # Use sys.exit(1) for error exit

    logger.info("\n--- Basic Statistics (e.g., relevance and paper type counts) ---")
    logger.info("Relevance Score Counts:")
    logger.info(df_annotations['relevance_score_llm'].value_counts().to_string())
    logger.info("\nPaper Type Counts:")
    logger.info(df_annotations['paper_type'].value_counts().to_string())

    application_papers_df = df_annotations[df_annotations['paper_type'] == 'application'].copy()

    logger.info(f"\n--- Statistics for Application Papers ---")
    logger.info(f"Total application papers found: {len(application_papers_df)}")
    logger.info("\n--- Relevance Score Breakdown for Application Papers ---")

    relevance_counts_app = application_papers_df['relevance_score_llm'].value_counts()
    relevance_percentages_app = application_papers_df['relevance_score_llm'].value_counts(normalize=True) * 100

    logger.info("Relevance Score Counts for Application Papers:")
    for score in relevance_counts_app.index:
        count = relevance_counts_app[score]
        percentage = relevance_percentages_app[score]
        logger.info(f"- {score}: {percentage:.2f}% ({count} papers)")

    relevant_df = application_papers_df[
        ~application_papers_df['relevance_score_llm'].isin(['irrelevant', 'low_relevance', 'low-relevance'])
    ]
    logger.info(f"\nTotal relevant application papers: {len(relevant_df)}")

    relevant_df.loc[:, 'curated_data_modalities_used'] = relevant_df['data_modalities_used'].apply(
        lambda term_list: apply_keyword_labels(term_list, keyword_map_methods)
    )

    logger.info("\n--- Data Modalities Used Counts for Application Papers ---")
    application_modalities_counter = Counter()

    for modalities_list in relevant_df['curated_data_modalities_used'].dropna():
        if isinstance(modalities_list, list):
            application_modalities_counter.update(modalities_list)

    logger.info(f"Total unique data modalities identified: {len(application_modalities_counter)}")
    logger.info("Counts of each data modality used in Application Papers:")
    for modality, count in application_modalities_counter.most_common():
        logger.info(f"- {modality}: {count}")

    relevant_df.loc[:, 'modalities_general'] = relevant_df['curated_data_modalities_used'].apply(
        lambda term_list: apply_keyword_labels(term_list, keyword_map_methods_general)
    )

    # ***Harmonize biological context***
    logger.info("\n--- Biological Context Harmonization for Application Papers ---")

    is_missing_context = (
        relevant_df['biological_context']
        .astype(str)
        .replace({'N/A': '', 'None': ''})
        .str.strip()
        .eq('')
        | relevant_df['biological_context'].isna()
    )

    num_without_context = is_missing_context.sum()
    num_with_context = len(relevant_df) - num_without_context

    logger.info(f"Total relevant application papers: {len(relevant_df)}")
    logger.info(f"Papers with 'biological_context' provided: {num_with_context}")
    logger.info(f"Papers without 'biological_context' provided: {num_without_context}")
    if len(relevant_df) > 0:
        logger.info(f"Percentage with context: {num_with_context / len(relevant_df):.2%}")
        logger.info(f"Percentage without context: {num_without_context / len(relevant_df):.2%}")

    relevant_df['tissue'] = relevant_df['biological_context'].apply(
        lambda x: harmonize_context(x, TISSUE_MAP)
    )
    logger.info("\n--- Value Counts for 'tissue' ---")
    logger.info(relevant_df['tissue'].value_counts(dropna=False).to_string())
    print_unassigned_for_column(relevant_df, 'tissue')

    relevant_df['disease'] = relevant_df['biological_context'].apply(
        lambda x: harmonize_context(x, DISEASE_MAP)
    )
    logger.info("\n--- Value Counts for 'disease' ---")
    logger.info(relevant_df['disease'].value_counts(dropna=False).to_string())
    print_unassigned_for_column(relevant_df, 'disease')

    relevant_df['animal'] = relevant_df['biological_context'].apply(
        lambda x: harmonize_context(x, ANIMAL_MAP)
    )
    logger.info("\n--- Value Counts for 'animal' ---")
    logger.info(relevant_df['animal'].value_counts(dropna=False).to_string())
    print_unassigned_for_column(relevant_df, 'animal')

    try:
        relevant_df.to_csv(OUTPUT_CSV_FILE, index=False)
        logger.info(f"\nHarmonized data saved to: {OUTPUT_CSV_FILE}")
    except Exception as e:
        logger.error(f"Error saving harmonized data to CSV: {e}")

    logger.info(f"\nScript finished at {datetime.now()}")