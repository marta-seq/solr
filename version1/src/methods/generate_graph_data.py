import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm
from umap import UMAP

# --- Configuration ---
INPUT_EMBEDDINGS_FILE = "../../data/methods/papers_with_embeddings.parquet"
# This should be your master JSONL file containing all original metadata and LLM annotations
INPUT_METADATA_FILE = "../../data/methods/master_methods_manual_curation.jsonl"
OUTPUT_GRAPH_DATA_FILE = "../../data/methods/graph_data.json"

# UMAP Configuration
UMAP_N_COMPONENTS = 2  # 2D for visualization
UMAP_N_NEIGHBORS = 15  # Balance local vs global structure (tune this!)
UMAP_MIN_DIST = 0.1  # How tightly points are clustered (tune this!)

# Similarity Graph Edge Configuration
TOP_K_SIMILAR_PAPERS = 10  # Number of strongest similarity links per paper (excluding self)
SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity to draw an edge (optional, for filtering very weak links)

# --- Logging Setup ---
LOG_DIR = "../../data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/graph_data_generation_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplicate logs if function is called multiple times
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


# --- Helper Functions for Text Extraction from Complex LLM Annotations ---
def extract_text_from_complex_annotation(value):
    """
    Robustly extracts text from complex annotation structures (None, lists, dicts, bools, scalars).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''

    # Handle cases where value might be a simple string representation of an empty list/dict
    if isinstance(value, str):
        if value.strip() in ['[]', '{}', 'N/A', 'None', '', 'nan']:  # Added 'nan' for robustness
            return ''
        return value.strip()  # Return the string itself if not empty/null-like

    if isinstance(value, (list, dict)):
        if not value:
            return ''

        if isinstance(value, list):
            sub_texts = []
            for item in value:
                if isinstance(item, dict):
                    dict_values = [
                        str(v).strip() for k, v in item.items()
                        if v is not None and str(v).strip() not in ['N/A', '', 'nan']  # Added 'nan'
                    ]
                    sub_texts.append(' '.join(dict_values))
                else:
                    sub_texts.append(str(item))
            return ' '.join(filter(None, sub_texts)).strip()

        elif isinstance(value, dict):
            dict_values = [
                str(v).strip() for k, v in value.items()
                if v is not None and str(v).strip() not in ['N/A', '', 'nan']  # Added 'nan'
            ]
            return ' '.join(dict_values).strip()

    if isinstance(value, bool):
        return 'Yes' if value else 'No'

    return str(value).strip()


# --- Main Data Generation Logic ---
def generate_graph_data():
    logger.info("--- Starting Graph Data Generation Script ---")

    # 1. Load papers with embeddings (DOI and embedding only from parquet)
    try:
        df_embeddings = pd.read_parquet(INPUT_EMBEDDINGS_FILE)
        logger.info(f"Loaded {len(df_embeddings)} papers with embeddings from '{INPUT_EMBEDDINGS_FILE}'.")
    except Exception as e:
        logger.critical(f"Failed to load embeddings from '{INPUT_EMBEDDINGS_FILE}': {e}")
        logger.critical("Please ensure the embedding generation script ran successfully and the file exists.")
        return

    # Filter out papers that don't have valid embeddings before merge (for efficiency)
    df_embeddings_valid = df_embeddings[
        df_embeddings['embedding'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)
    ].copy()

    if df_embeddings_valid.empty:
        logger.error("No papers with valid embeddings found in the parquet file. Cannot generate graph data.")
        return
    logger.info(f"Found {len(df_embeddings_valid)} papers with valid embeddings in parquet.")

    # 2. Load the master metadata file (JSONL with all annotations)
    try:
        df_metadata = pd.read_json(INPUT_METADATA_FILE, lines=True)
        # IMPORTANT: DO NOT set 'doi' as index here yet. Keep it as a column for the merge.
        logger.info(f"Loaded {len(df_metadata)} papers with metadata from '{INPUT_METADATA_FILE}'.")
    except Exception as e:
        logger.critical(f"Failed to load metadata from '{INPUT_METADATA_FILE}': {e}")
        logger.critical("Please ensure the metadata compilation scripts ran successfully (e.g., merge_llm_outputs.py).")
        return

    # 3. Merge the embeddings DataFrame with the metadata DataFrame
    # Ensure 'doi' is a column in df_metadata for the merge (it should be if not set as index above)
    # This merge will correctly bring all columns from df_metadata into df_final based on 'doi' column.
    df_final = pd.merge(df_embeddings_valid, df_metadata, on='doi', how='inner')

    if df_final.empty:
        logger.error("No common papers found after merging embeddings and metadata. Cannot generate graph data.")
        return

    logger.info(f"Successfully merged embeddings with metadata. Total papers for graph: {len(df_final)}.")

    # Now set 'doi' as the index for df_final AFTER the merge, for convenient iteration and UMAP.
    df_final.set_index('doi', inplace=True)

    # Prepare embeddings matrix for UMAP
    embeddings_matrix = np.array(df_final['embedding'].tolist())

    # 4. Perform UMAP dimensionality reduction
    logger.info(
        f"Performing UMAP reduction to {UMAP_N_COMPONENTS} dimensions (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
    reducer = UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                   random_state=42)
    umap_coords = reducer.fit_transform(embeddings_matrix)

    df_final['x_umap'] = umap_coords[:, 0]
    df_final['y_umap'] = umap_coords[:, 1]
    logger.info("UMAP reduction complete.")

    # 5. Prepare Node Data for Visualization
    nodes_data = []

    # Define a list of columns to include in node properties for hover/filtering
    # These columns are now expected to be present in df_final after the merge
    NODE_DISPLAY_ATTRIBUTES = [
        'title', 'abstract', 'year', 'citations', 'journal', 'source',
        'annotation_score', 'annotation_status',
        'llm_annot_package_algorithm_name',
        'llm_annot_pipeline_analysis_steps',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_tested_data_modalities',
        'llm_annot_tool_type',
        'weak_annot_biological_context',
        'weak_annot_data_modalities_used',
        'weak_annot_paper_type',
        'weak_annot_title',
        'llm_annot_main_goal_of_paper',
        'llm_annot_algorithms_bases_used',
        'llm_annot_code_availability_details',
        'llm_annot_compared_algorithms_packages',
        'llm_annot_data_used_details',
        'weak_annot_weak_annotations',
        'extracted_methods_section'
    ]

    # List of columns that contain potentially complex structures (lists/dicts)
    # that we want to save as 'raw_' for Streamlit filtering
    # These will be directly pulled from the row_series before string conversion
    RAW_DATA_COLUMNS_MAP = {
        'raw_pipeline_steps': 'llm_annot_pipeline_analysis_steps',
        'raw_tested_data_modalities': 'llm_annot_tested_data_modalities',
        'raw_tested_assay_types_platforms': 'llm_annot_tested_assay_types_platforms',
        'raw_tool_type': 'llm_annot_tool_type',  # Often a single value, but useful to keep raw
        'raw_author_keywords': 'author_keywords',
    }

    for doi, row_series in tqdm(df_final.iterrows(), total=len(df_final), desc="Preparing node data"):
        node_id = doi  # The index is now the DOI
        node_attrs = {
            'id': node_id,
            'doi': node_id,  # Explicitly add 'doi' attribute
            'x': row_series['x_umap'],
            'y': row_series['y_umap']
        }

        # Add display attributes (stringified for hover and general display)
        for attr in NODE_DISPLAY_ATTRIBUTES:
            # Use .get() to safely retrieve, then apply text extraction
            value = row_series.get(attr)
            node_attrs[attr] = extract_text_from_complex_annotation(value)

        # Populate raw_ fields directly from the row_series which contains original data types
        # Use .get() with an appropriate default (empty list for lists, None for scalars)
        for raw_key, original_col_name in RAW_DATA_COLUMNS_MAP.items():
            default_value = [] if 'pipeline' in raw_key or 'modalities' in raw_key or 'keywords' in raw_key else None
            node_attrs[raw_key] = row_series.get(original_col_name, default_value)

        nodes_data.append(node_attrs)
    logger.info("Node data prepared.")

    # 6. Prepare Similarity Edges
    edges_data = []
    # Get DOIs in the exact order of the embeddings_matrix for correct lookup
    dois_in_order = df_final.index.tolist()

    # Calculate cosine similarity for all pairs
    similarity_matrix = cosine_similarity(embeddings_matrix)

    logger.info(
        f"Generating similarity edges (top {TOP_K_SIMILAR_PAPERS} neighbors, threshold={SIMILARITY_THRESHOLD})...")
    for i in tqdm(range(len(dois_in_order)), desc="Generating edges"):
        source_doi = dois_in_order[i]
        similarities = similarity_matrix[i]

        top_k_indices = similarities.argsort()[-TOP_K_SIMILAR_PAPERS - 1:-1][::-1]

        for idx in top_k_indices:
            target_doi = dois_in_order[idx]
            score = similarities[idx]

            if score >= SIMILARITY_THRESHOLD:
                if source_doi < target_doi:  # Ensure consistent order for unique edges
                    edges_data.append({
                        'source': source_doi,
                        'target': target_doi,
                        'weight': float(score),
                        'type': 'semantic_similarity'
                    })
    logger.info(f"Generated {len(edges_data)} similarity edges.")

    # 7. Prepare Placeholder for Explicit Relationships (Future Work)
    explicit_edges_data = []  # Remains a placeholder for now
    logger.info("Placeholder for explicit relationship extraction prepared.")

    # 8. Save data to JSON
    graph_output = {
        'nodes': nodes_data,
        'edges': edges_data,
        'explicit_edges_placeholder': explicit_edges_data
    }

    try:
        with open(OUTPUT_GRAPH_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(graph_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Graph data saved successfully to '{OUTPUT_GRAPH_DATA_FILE}'.")
    except Exception as e:
        logger.critical(f"Failed to save graph data to JSON: {e}")

    logger.info("--- Graph Data Generation Script Finished ---")


if __name__ == "__main__":
    generate_graph_data()