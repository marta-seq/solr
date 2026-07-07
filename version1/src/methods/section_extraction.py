import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
# Input Master File (output from your previous script)
INPUT_MASTER_FILE = "../../data/manual_curated_papers_methods.jsonl"
# Output file for DataFrame with embeddings
OUTPUT_EMBEDDINGS_FILE = "../../data/methods/papers_with_embeddings.parquet"

# --- Embedding Model Configuration ---
# Options:
# 'general': 'all-MiniLM-L6-v2' (fast, good general performance)
# 'scibert': 'allenai/specter' (better for scientific text, specifically trained for scientific paper similarity)
EMBEDDING_MODEL_TYPE = 'scibert'  # Changed to 'scibert' as default, as it's often better for scientific texts

# Text Source Configuration: Choose which parts of the paper to use for embeddings
# Set to True to include, False to exclude
USE_TITLE_ABSTRACT = True
USE_EXTRACTED_SECTIONS = True  # Uses introduction, methods, conclusion, etc.
USE_LLM_ANNOTATIONS = True  # Uses various llm_annot_ and weak_annot_ fields

# --- Logging Setup ---
LOG_DIR = "../../data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/embedding_generation_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplicate logs if run multiple times in same session
if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Define relevant columns based on your actual JSON example ---

# These columns contain large blocks of text that were extracted from the paper.
LLM_EXTRACTED_SECTION_COLUMNS = [
    'extracted_introduction_section',
    'extracted_results_discussion_section',
    'extracted_conclusion_section',
    'extracted_methods_section',
    'extracted_code_availability_section',
    'extracted_data_availability_section'
]

# These are specific LLM-generated annotation fields from your example that contain useful textual information.
# We'll handle nested structures (lists of dicts, dicts) within prepare_text_for_embedding.
LLM_ANNOTATION_SPECIFIC_COLUMNS = [
    'llm_annot_additional_notes',
    'llm_annot_algorithms_bases_used',  # List of dicts, needs special parsing
    'llm_annot_code_availability_details',  # Dict, needs special parsing
    'llm_annot_compared_algorithms_packages',  # List of dicts, needs special parsing
    'llm_annot_data_used_details',  # List of dicts, needs special parsing
    'llm_annot_main_goal_of_paper',
    'llm_annot_package_algorithm_name',
    'llm_annot_pipeline_analysis_steps',  # List of strings
    'llm_annot_tested_assay_types_platforms',  # List of strings
    'llm_annot_tested_data_modalities',  # List of strings
    'llm_annot_tool_type',
    'weak_annot_biological_context',
    'weak_annot_data_modalities_used',  # List of strings
    'weak_annot_paper_type',
    'weak_annot_relevance_score_llm',
    'weak_annot_title',  # Might be redundant with main title, but LLM's understanding
    'weak_annot_weak_annotations'  # Nested dict, needs special parsing
]


# --- Helper Functions ---
def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a JSONL file into a pandas DataFrame."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} at line {line_num}: {line.strip()}")
    else:
        logger.error(f"Input file not found: {file_path}. Please ensure the master file exists.")
        return pd.DataFrame()  # Return empty DataFrame on critical error
    return pd.DataFrame(data)


def extract_text_from_complex_annotation(value):
    """
    Robustly extracts text from complex annotation structures (None, lists, dicts, bools, scalars).
    Avoids pd.isna() for the initial check to prevent potential type issues.
    """
    # Handle explicit None and numpy NaN (for float types that might contain NaNs)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''

    # Handle common empty container types explicitly as empty string
    if isinstance(value, (list, dict)):
        if not value:  # Checks if list or dict is empty
            return ''

        # If not empty, proceed to extract content
        if isinstance(value, list):
            sub_texts = []
            for item in value:
                if isinstance(item, dict):
                    # Extract text from dict values, filtering None/empty/N/A strings
                    dict_values = [
                        str(v).strip() for k, v in item.items()
                        if v is not None and str(v).strip() != 'N/A' and str(v).strip() != ''
                    ]
                    sub_texts.append(' '.join(dict_values))
                else:
                    sub_texts.append(str(item))
            return ' '.join(filter(None, sub_texts)).strip()

        elif isinstance(value, dict):  # If it's a non-empty dictionary
            dict_values = [
                str(v).strip() for k, v in value.items()
                if v is not None and str(v).strip() != 'N/A' and str(v).strip() != ''
            ]
            return ' '.join(dict_values).strip()

    # Handle boolean conversion
    if isinstance(value, bool):
        return 'Yes' if value else 'No'

    # For any other scalar type (int, str, etc.)
    return str(value).strip()


def prepare_text_for_embedding(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenates selected text fields for embedding."""
    logger.info("Preparing text for embedding based on configuration...")

    # Initialize a list to hold all series of text components
    text_components_series = []

    if USE_TITLE_ABSTRACT:
        if 'title' in df.columns:
            text_components_series.append(df['title'].fillna(''))
            logger.info("  - Including Title.")
        if 'abstract' in df.columns:
            text_components_series.append(df['abstract'].fillna(''))
            logger.info("  - Including Abstract.")

    if USE_EXTRACTED_SECTIONS:
        for col in LLM_EXTRACTED_SECTION_COLUMNS:
            if col in df.columns:
                text_components_series.append(df[col].fillna(''))
                logger.info(f"  - Including extracted section: '{col}'.")
            else:
                logger.debug(f"  - Extracted section column '{col}' not found.")

    if USE_LLM_ANNOTATIONS:
        for col in LLM_ANNOTATION_SPECIFIC_COLUMNS:
            if col in df.columns:
                # Apply the robust text extraction function
                text_components_series.append(df[col].apply(extract_text_from_complex_annotation))
                logger.info(f"  - Including LLM annotation column: '{col}'.")
            else:
                logger.debug(f"  - LLM annotation column '{col}' not found.")

        if not text_components_series:  # This check might be too broad if other sections were included
            logger.warning("  - No LLM annotation columns found in data despite USE_LLM_ANNOTATIONS being True.")

    if not text_components_series:
        logger.error(
            "No text sources selected or found for embedding. Please enable at least one option (USE_TITLE_ABSTRACT, USE_EXTRACTED_SECTIONS, USE_LLM_ANNOTATIONS) and ensure columns exist.")
        df['text_for_embedding'] = ''
        return df

    # Concatenate all chosen text components. Use a unique separator.
    # We use '. ' for readability, and then clean up multiple separators later.
    df['text_for_embedding'] = pd.concat(text_components_series, axis=1).agg('. '.join, axis=1)

    # Clean up the combined text
    df['text_for_embedding'] = df['text_for_embedding'].str.replace(r'\s+', ' ',
                                                                    regex=True).str.strip()  # Reduce multiple spaces
    df['text_for_embedding'] = df['text_for_embedding'].str.replace(r'\s*\.\s*\.', '. ',
                                                                    regex=True).str.strip()  # Remove double periods etc.
    df['text_for_embedding'] = df['text_for_embedding'].str.replace(r'\s*\.\s*', '. ',
                                                                    regex=True).str.strip()  # Ensure single space after period

    logger.info("Text preparation complete.")
    return df


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Embedding Generation Script ---")

    # 1. Load the master DataFrame
    df_papers = load_jsonl_to_dataframe(INPUT_MASTER_FILE)
    if df_papers.empty:
        logger.error("No data loaded. Exiting embedding generation.")
        exit()

    logger.info(f"Loaded {len(df_papers)} papers from '{INPUT_MASTER_FILE}'.")

    # 2. Prepare text for embedding
    df_papers = prepare_text_for_embedding(df_papers)

    # Filter out papers with empty text for embedding
    df_papers_with_text = df_papers[df_papers['text_for_embedding'].str.len() > 0].copy()
    if df_papers_with_text.empty:
        logger.warning(
            "No papers have sufficient text to generate embeddings. Check your data and text source configuration.")
        # Save an empty or original DataFrame if no embeddings can be generated
        df_papers.to_parquet(OUTPUT_EMBEDDINGS_FILE, index=False)
        logger.info("Script finished (no embeddings generated).")
        exit()

    logger.info(f"Proceeding with {len(df_papers_with_text)} papers for embedding generation.")

    # 3. Load the Sentence Transformer model
    if EMBEDDING_MODEL_TYPE == 'general':
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"Loading general purpose embedding model: {model_name}")
    elif EMBEDDING_MODEL_TYPE == 'scibert':
        model_name = 'allenai/specter'  # SPECTER is trained on scientific papers for citation prediction/similarity
        logger.info(f"Loading scientific embedding model: {model_name}")
    else:
        logger.error(f"Unknown EMBEDDING_MODEL_TYPE: '{EMBEDDING_MODEL_TYPE}'. Falling back to 'all-MiniLM-L6-v2'.")
        model_name = 'all-MiniLM-L6-v2'

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.critical(f"Failed to load Sentence Transformer model '{model_name}': {e}")
        logger.critical("Please check your internet connection and model name. Exiting.")
        exit()

    # 4. Generate Embeddings
    texts_to_embed = df_papers_with_text['text_for_embedding'].tolist()

    logger.info(f"Generating embeddings for {len(texts_to_embed)} papers using {model_name}...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    logger.info("Embeddings generation complete.")

    # 5. Add embeddings back to the DataFrame
    # Create a mapping from DOI to embedding array
    doi_to_embedding_map = dict(zip(df_papers_with_text['doi'], embeddings))

    # Apply the embeddings to the original df_papers, ensuring correct alignment
    df_papers['embedding'] = df_papers['doi'].map(doi_to_embedding_map).apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else None)

    # Fill None values (for papers that had empty text_for_embedding) with empty lists for consistency
    df_papers['embedding'] = df_papers['embedding'].apply(lambda x: [] if x is None else x)

    # 6. Save the DataFrame with embeddings
    try:
        # Create a new DataFrame with only DOI and embedding for clean storage
        df_embeddings_only = df_papers[['doi', 'embedding']].copy()

        # Save this smaller, cleaner DataFrame to Parquet
        df_embeddings_only.to_parquet(OUTPUT_EMBEDDINGS_FILE, index=False)
        logger.info(f"DataFrame with only DOI and embeddings saved successfully to '{OUTPUT_EMBEDDINGS_FILE}'.")

    except Exception as e:
        logger.critical(f"Failed to save DataFrame with embeddings to Parquet: {e}")

    logger.info("--- Embedding Generation Script Finished ---")