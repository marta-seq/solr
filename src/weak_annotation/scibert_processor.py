import pandas as pd
import numpy as np
import datetime
import logging
import os
import sys
from sentence_transformers import SentenceTransformer, util
import torch

# --- Path Setup (Crucial for finding utils module) ---
# This block ensures the project's 'src' directory is in the Python path,
# allowing imports like 'utils.logger_setup' to work correctly.
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(current_script_path)))
path_to_add_to_sys_path = os.path.join(project_root, 'src')
if path_to_add_to_sys_path not in sys.path:
    sys.path.insert(0, path_to_add_to_sys_path)

from utils.logger_setup import setup_logging
from utils.doi import clean_doi  # Importing the clean_doi function from utils.doi
# ---------------------------
# Configuration Parameters
# ---------------------------
BASE_DIR = os.path.join(project_root, "data")
LOGGING_DIR = os.path.join(BASE_DIR, "logs/scibert_processor")

# --- Input Files ---
DOI_POOL_FILE = os.path.join(BASE_DIR, "doi_pool.jsonl")
SEED_DOI_FILE = os.path.join(BASE_DIR, "paper_classification.xlsx")

# --- Similarity Thresholds ---
RELEVANT_THRESHOLD = 0.80  # Score above this is considered 'relevant'
UNCERTAIN_THRESHOLD = 0.65  # Score between this and RELEVANT_THRESHOLD is 'uncertain'

# --- Categories that can have associated pipeline categories ---
# This list defines which main categories (from the 'category' column in seed data)
# should trigger a second-tier pipeline classification if a pipeline_category is provided.
# Ensure this list accurately reflects your data's structure and desired behavior.
CATEGORIES_WITH_PIPELINES = ['computational analysis - method']  # Add more if other categories have pipelines

# ---------------------------
# Initialize Logging (specific to this module)
# ---------------------------
logger = setup_logging(LOGGING_DIR, "scibert_process")

# ---------------------------
# Helper Function for DOI Cleaning
# ---------------------------
# def clean_doi(doi_str):
#     """Normalizes a DOI string by removing common prefixes."""
#     if pd.isna(doi_str):
#         return None
#     doi_str = str(doi_str).strip()
#     if doi_str.startswith("https://doi.org/"):
#         return doi_str[len("https://doi.org/"):]
#     return doi_str


# --- NEW HELPER FUNCTION FOR ROBUST SCALAR CONVERSION ---
def _to_scalar_float(value):
    """
    Ensures a value is a single Python float. Handles NumPy arrays (0D, 1D with 1 element),
    PyTorch tensors (0D, 1D with 1 element), and standard Python numerics.
    Raises an error if a multi-element array/tensor is passed, as that indicates a logic flaw.
    """
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if value.ndim == 0:
            return float(value.item())
        elif value.ndim == 1 and value.shape[0] == 1:
            return float(value.item())
        else:
            # This case should ideally not happen if similarity scores are correctly handled.
            # It means a multi-element array is being treated as a single scalar.
            logger.error(f"Attempted to convert multi-element array/tensor to scalar: {value}. "
                         f"Shape: {value.shape}. This indicates a potential logic error upstream.")
            # Forcing to first element here, but it's a fallback for a deeper issue.
            return float(value[0].item() if hasattr(value[0], 'item') else value[0])
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.error(f"Could not convert value to float: {value}, type: {type(value)}")
        raise


# ---------------------------
# Main SciBERT Processing Function
# ---------------------------
def run_scibert_analysis_pipeline() -> pd.DataFrame:
    """
    Runs the SciBERT embedding and similarity analysis on the paper pool.
    Returns a DataFrame with SciBERT-specific annotations for the pool papers.
    """
    logger.info("Starting SciBERT Similarity Analysis (Two-Tiered Classification with Multi-Label Seed Parsing).")

    # ---------------------------
    # Load SentenceTransformer Model
    # ---------------------------
    logger.info("Loading SciBERT (sentence-transformers 'allenai-specter') model...")
    try:
        model = SentenceTransformer('allenai-specter')
        logger.info("SciBERT model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load SentenceTransformer model: {e}. Ensure internet or model cached.")
        raise

    # ---------------------------
    # Load Full Pool of Papers to Annotate
    # ---------------------------
    logger.info(f"Loading pool papers from '{DOI_POOL_FILE}'.")
    try:
        pool_df = pd.read_json(DOI_POOL_FILE, lines=True)
        pool_df['doi'] = pool_df['doi'].apply(clean_doi)

        if 'author_keywords' not in pool_df.columns:
            pool_df['author_keywords'] = np.nan

        pool_df.dropna(subset=["doi"], inplace=True)

        initial_pool_count = len(pool_df)
        pool_df.drop_duplicates(subset=["doi"], inplace=True)
        if initial_pool_count - len(pool_df) > 0:
            logger.warning(f"Removed {initial_pool_count - len(pool_df)} duplicate DOIs from the pool.")

        pool_df['doi_lower'] = pool_df['doi'].str.lower()
        pool_lookup_df = pool_df.set_index('doi_lower')[['title', 'abstract']].copy()

        logger.info(f"Loaded {len(pool_df)} pool papers after cleaning and deduplication.")
    except FileNotFoundError:
        logger.critical(f"Pool data file not found: '{DOI_POOL_FILE}'.")
        raise
    except Exception as e:
        logger.critical(f"An error occurred while loading pool data: {e}.")
        raise

    # ---------------------------
    # Load Curated Seed Data & Enrich with Titles/Abstracts from Pool
    # AND handle multi-value categories (separated by ';')
    # ---------------------------
    logger.info(f"Loading seed data from '{SEED_DOI_FILE}' and expanding multi-value categories/pipelines.")
    try:
        seed_excel_df = pd.read_excel(SEED_DOI_FILE)
        seed_excel_df.columns = seed_excel_df.columns.str.strip()

        seed_excel_df['doi'] = seed_excel_df['doi'].apply(clean_doi)
        seed_excel_df.dropna(subset=["doi"], inplace=True)

        if seed_excel_df.empty:
            logger.critical("Seed data Excel is empty after cleaning DOIs. Cannot proceed with SciBERT.")
            raise ValueError("Empty seed data after cleaning.")

        seed_excel_df['doi_lower'] = seed_excel_df['doi'].str.lower()

        # Merge with pool_lookup_df to get title/abstract
        seed_df_raw = pd.merge(
            seed_excel_df,
            pool_lookup_df,
            left_on='doi_lower',
            right_index=True,
            how='left'
        )

        missing_seed_metadata = seed_df_raw[seed_df_raw['title'].isna() | seed_df_raw['abstract'].isna()]
        if not missing_seed_metadata.empty:
            logger.warning(
                f"Found {len(missing_seed_metadata)} seed papers whose titles/abstracts were not found in the DOI pool "
                f"(or were empty/NaN). These will be excluded from seed centroid calculation."
            )
            logger.debug(f"Missing seed DOIs (first 10): {missing_seed_metadata['doi'].head(10).tolist()}")
            seed_df_raw.dropna(subset=["title", "abstract"], inplace=True)

        if seed_df_raw.empty:
            logger.critical(
                "No valid seed papers with title/abstract found after matching with pool data. Cannot calculate centroids.")
            raise ValueError("No valid seed texts for embedding after matching.")

        # --- Multi-Value Category Parsing and Expansion ---
        expanded_seed_data = []
        logger.debug("Starting seed data expansion...")
        for idx, row in seed_df_raw.iterrows():
            doi = row['doi']
            title = row['title']
            abstract = row['abstract']

            original_categories_str = str(row['category']).strip() if pd.notna(row['category']) else ''
            original_pipelines_str = str(row['pipeline_category']).strip() if 'pipeline_category' in row and pd.notna(
                row['pipeline_category']) else ''

            categories_list = [c.strip() for c in original_categories_str.split(';') if c.strip()]
            pipelines_list = [p.strip() for p in original_pipelines_str.split(';') if p.strip()]

            logger.debug(
                f"  DOI: {doi}, Original Cats: '{original_categories_str}', Original Pipelines: '{original_pipelines_str}'")
            logger.debug(f"    Parsed Categories: {categories_list}, Parsed Pipelines: {pipelines_list}")

            if not categories_list:
                logger.warning(
                    f"Skipping seed paper {doi} (original row index {idx}) due to empty/invalid main category after parsing.")
                continue

            for cat in categories_list:
                if cat in CATEGORIES_WITH_PIPELINES and pipelines_list:
                    for pipe_cat in pipelines_list:
                        expanded_seed_data.append({
                            'doi': doi,
                            'title': title,
                            'abstract': abstract,
                            'category': cat,
                            'pipeline_category': pipe_cat
                        })
                    logger.debug(f"    Added {len(pipelines_list)} pipeline associations for main category '{cat}'.")
                else:
                    expanded_seed_data.append({
                        'doi': doi,
                        'title': title,
                        'abstract': abstract,
                        'category': cat,
                        'pipeline_category': ''
                    })
                    logger.debug(f"    Added main category '{cat}' (no pipeline or not pipeline-bearing).")

        seed_df_expanded = pd.DataFrame(expanded_seed_data)

        if seed_df_expanded.empty:
            logger.critical(
                "Expanded seed data is empty after multi-value parsing. No valid seed papers for centroids.")
            raise ValueError("Expanded seed data is empty.")

        seed_df_expanded['category'] = seed_df_expanded['category'].astype(str)
        seed_df_expanded['pipeline_category'] = seed_df_expanded['pipeline_category'].astype(str)

        logger.info(
            f"Loaded and expanded {len(seed_df_expanded)} seed paper associations from {len(seed_df_raw)} original seed papers.")
        logger.debug(f"Head of expanded seed data:\n{seed_df_expanded.head()}")
        logger.debug(f"Value counts of expanded 'category':\n{seed_df_expanded['category'].value_counts()}")
        logger.debug(
            f"Value counts of expanded 'pipeline_category':\n{seed_df_expanded['pipeline_category'].value_counts()}")

    except FileNotFoundError:
        logger.critical(f"Seed data file not found: '{SEED_DOI_FILE}'.")
        raise
    except Exception as e:
        logger.critical(f"An error occurred while loading or enriching seed data: {e}")
        raise

    # ---------------------------
    # Embed Seed Papers & Calculate Centroids (Two-Tiered)
    # Uses seed_df_expanded for embedding and centroid calculation
    # ---------------------------
    logger.info(
        f"Embedding {len(seed_df_expanded)} expanded seed paper associations and calculating centroids (two-tiered strategy)...")

    unique_seed_papers_for_embedding = seed_df_expanded[['doi', 'title', 'abstract']].drop_duplicates().reset_index(
        drop=True)

    seed_texts_for_embedding = unique_seed_papers_for_embedding["title"].fillna("") + ". " + \
                               unique_seed_papers_for_embedding["abstract"].fillna("")
    seed_texts_for_embedding = seed_texts_for_embedding.replace("", pd.NA).dropna()

    if seed_texts_for_embedding.empty:
        logger.critical(
            "No valid text found for unique seed papers to create embeddings after enrichment and cleaning.")
        raise ValueError("No valid seed texts for embedding.")

    logger.debug(f"Encoding {len(seed_texts_for_embedding)} unique seed texts...")
    seed_embeddings_tensor = model.encode(
        seed_texts_for_embedding.tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    logger.debug(f"Seed embeddings tensor shape: {seed_embeddings_tensor.shape}")

    seed_embedding_lookup = {
        doi: embedding for doi, embedding in zip(unique_seed_papers_for_embedding['doi'], seed_embeddings_tensor)
    }

    # --- Main Category Centroids ---
    main_category_centroids = {}
    for main_cat in seed_df_expanded['category'].unique():
        if not main_cat:
            continue

        dois_in_main_cat = seed_df_expanded[seed_df_expanded['category'] == main_cat]['doi'].unique()
        embeddings_for_this_main_cat = [seed_embedding_lookup[d] for d in dois_in_main_cat if
                                        d in seed_embedding_lookup]

        if embeddings_for_this_main_cat:
            main_category_centroids[main_cat] = torch.mean(torch.stack(embeddings_for_this_main_cat), dim=0)
            logger.debug(
                f"Main Category '{main_cat}': Centroid calculated from {len(embeddings_for_this_main_cat)} unique DOIs.")
        else:
            logger.warning(f"No embeddings found for main category '{main_cat}'. Centroid will not be calculated.")

    if not main_category_centroids:
        logger.critical("No main category centroids could be calculated. Cannot proceed.")
        raise ValueError("No main category centroids calculated.")

    main_category_labels = list(main_category_centroids.keys())
    main_centroid_embeddings_tensor = torch.stack(list(main_category_centroids.values()))
    logger.info(f"Calculated {len(main_category_labels)} main category centroids.")

    # --- Pipeline Category Centroids (Nested by Main Category) ---
    pipeline_category_centroids_by_main_category = {}

    for main_cat in CATEGORIES_WITH_PIPELINES:
        if main_cat not in seed_df_expanded['category'].unique():
            logger.debug(
                f"Main category '{main_cat}' (from CATEGORIES_WITH_PIPELINES) not found in expanded seed data, skipping pipeline centroid calculation for it.")
            continue

        logger.debug(f"Calculating pipeline centroids for main category: '{main_cat}'")
        pipeline_category_centroids_by_main_category[main_cat] = {}

        pipeline_cats_for_this_main = seed_df_expanded[
            (seed_df_expanded['category'] == main_cat) &
            (seed_df_expanded['pipeline_category'] != '')
            ]['pipeline_category'].unique()

        for pipe_cat in pipeline_cats_for_this_main:
            if not pipe_cat:
                continue

            dois_in_pipe_cat = seed_df_expanded[
                (seed_df_expanded['category'] == main_cat) &
                (seed_df_expanded['pipeline_category'] == pipe_cat)
                ]['doi'].unique()

            embeddings_for_this_pipe_cat = [seed_embedding_lookup[d] for d in dois_in_pipe_cat if
                                            d in seed_embedding_lookup]

            if embeddings_for_this_pipe_cat:
                pipeline_category_centroids_by_main_category[main_cat][pipe_cat] = torch.mean(
                    torch.stack(embeddings_for_this_pipe_cat), dim=0)
                logger.debug(
                    f"  Pipeline Category '{main_cat}::{pipe_cat}': Centroid calculated from {len(embeddings_for_this_pipe_cat)} unique DOIs.")
            else:
                logger.warning(
                    f"No embeddings found for pipeline category '{main_cat}::{pipe_cat}'. Centroid will not be calculated.")

    if not pipeline_category_centroids_by_main_category:
        logger.warning("No pipeline category centroids could be calculated from seed data.")
    else:
        total_pipeline_centroids = sum(len(v) for v in pipeline_category_centroids_by_main_category.values())
        logger.info(f"Calculated {total_pipeline_centroids} pipeline centroids across relevant main categories.")

    # ---------------------------
    # Embed Pool Papers
    # ---------------------------
    pool_texts_series = pool_df["title"].fillna("") + ". " + pool_df["abstract"].fillna("")

    valid_text_mask = pool_texts_series.replace("", pd.NA).notna()
    pool_df_for_embedding = pool_df[valid_text_mask].copy().reset_index(drop=True)

    logger.info(f"Embedding {len(pool_df_for_embedding)} pool papers (using title + abstract).")

    pool_texts_list = pool_texts_series[valid_text_mask].tolist()

    if pool_df_for_embedding.empty:
        logger.critical(
            "No valid text found for pool papers for embedding. This is often an issue with doi_pool.jsonl's 'title' or 'abstract' fields.")
        raise ValueError("No valid pool texts for embedding.")

    logger.debug(f"Encoding {len(pool_texts_list)} pool texts.")
    pool_embeddings = model.encode(pool_texts_list, convert_to_tensor=True, show_progress_bar=True)
    logger.debug(f"Pool paper embeddings created. Shape: {pool_embeddings.shape}")

    # ---------------------------
    # Classify and Annotate Pool Papers (Two-Tiered Logic)
    # ---------------------------
    def classify_and_annotate(paper_embedding):
        logger.debug(f"Starting classification for a paper.")

        # Step 1: Classify into Main Category
        main_similarity_scores = \
        util.cos_sim(paper_embedding.unsqueeze(0), main_centroid_embeddings_tensor).cpu().numpy()[0]
        logger.debug(
            f"main_similarity_scores type: {type(main_similarity_scores)}, shape: {main_similarity_scores.shape}")

        # Ensure max_main_sim is a scalar float
        max_main_sim = _to_scalar_float(np.max(main_similarity_scores))
        logger.debug(f"max_main_sim: {max_main_sim}, type: {type(max_main_sim)}")

        best_main_centroid_idx = np.argmax(main_similarity_scores)
        predicted_main_category = main_category_labels[best_main_centroid_idx]

        final_scibert_category = predicted_main_category
        final_scibert_pipeline_category = ""

        # Ensure final_scibert_score is a scalar float
        final_scibert_score = _to_scalar_float(round(max_main_sim, 4))

        # Initialize final_scibert_most_similar_centroid to its main category default
        final_scibert_most_similar_centroid = predicted_main_category

        logger.debug(f"Initial final_scibert_score: {final_scibert_score}, type: {type(final_scibert_score)}")
        logger.debug(f"RELEVANT_THRESHOLD: {RELEVANT_THRESHOLD}, UNCERTAIN_THRESHOLD: {UNCERTAIN_THRESHOLD}")

        # --- RELEVANCE CALCULATION (Main Category) ---
        relevance = "low-relevance_by_similarity"

        # Ensure current_score_for_relevance is a scalar float before comparison
        current_score_for_relevance = _to_scalar_float(final_scibert_score)

        logger.debug(
            f"Before main relevance check: current_score_for_relevance={current_score_for_relevance}, type={type(current_score_for_relevance)}")

        if current_score_for_relevance >= RELEVANT_THRESHOLD:
            relevance = "relevant_by_similarity"
        elif current_score_for_relevance >= UNCERTAIN_THRESHOLD:
            relevance = "uncertain_by_similarity"

        logger.debug(f"Relevance assigned based on main score: {relevance}")

        # --- TOP MAIN CATEGORIES ---
        top_categories_by_sim = []
        main_relevant_scores = []
        for idx, score_val in enumerate(main_similarity_scores):
            # Ensure score_scalar is a pure Python scalar float
            score_scalar = _to_scalar_float(score_val)

            logger.debug(
                f"  Main Loop: idx={idx}, score_val_type={type(score_val)}, score_scalar_type={type(score_scalar)}, score_scalar_value={score_scalar}")

            if score_scalar >= UNCERTAIN_THRESHOLD:
                main_relevant_scores.append((score_scalar, main_category_labels[idx]))
        main_relevant_scores.sort(key=lambda x: x[0], reverse=True)
        top_categories_by_sim = [label for score, label in main_relevant_scores]

        top_pipeline_categories_by_sim = []

        # Step 2: Conditional Classification for Pipeline Category
        if predicted_main_category in CATEGORIES_WITH_PIPELINES and \
                predicted_main_category in pipeline_category_centroids_by_main_category and \
                pipeline_category_centroids_by_main_category[predicted_main_category]:

            logger.debug(f"Attempting pipeline classification for main category: '{predicted_main_category}'")

            pipeline_centroids_for_this_main = pipeline_category_centroids_by_main_category[predicted_main_category]
            pipeline_category_labels_for_this_main = list(pipeline_centroids_for_this_main.keys())
            pipeline_centroid_embeddings_for_this_main = torch.stack(list(pipeline_centroids_for_this_main.values()))

            pipeline_similarity_scores = \
            util.cos_sim(paper_embedding.unsqueeze(0), pipeline_centroid_embeddings_for_this_main).cpu().numpy()[0]
            logger.debug(
                f"pipeline_similarity_scores type: {type(pipeline_similarity_scores)}, shape: {pipeline_similarity_scores.shape}")

            # Ensure max_pipeline_sim is a scalar float
            max_pipeline_sim = _to_scalar_float(np.max(pipeline_similarity_scores))
            logger.debug(f"max_pipeline_sim: {max_pipeline_sim}, type: {type(max_pipeline_sim)}")

            best_pipeline_centroid_idx = np.argmax(pipeline_similarity_scores)
            predicted_pipeline_category = pipeline_category_labels_for_this_main[best_pipeline_centroid_idx]

            final_scibert_pipeline_category = predicted_pipeline_category
            final_scibert_most_similar_centroid = f"{predicted_main_category}::{predicted_pipeline_category}"  # This line is now safe

            # Ensure final_scibert_score is a scalar float
            final_scibert_score = _to_scalar_float(
                round(max_pipeline_sim, 4))  # Update score with pipeline score as it's more specific

            # --- RE-EVALUATE RELEVANCE (Pipeline Category) ---
            relevance = "low-relevance_by_similarity"  # Reset for pipeline score

            # Ensure current_score_for_relevance is a scalar float before comparison
            current_score_for_relevance = _to_scalar_float(final_scibert_score)

            logger.debug(
                f"Before pipeline relevance check: current_score_for_relevance={current_score_for_relevance}, type={type(current_score_for_relevance)}")

            if current_score_for_relevance >= RELEVANT_THRESHOLD:
                relevance = "relevant_by_similarity"
            elif current_score_for_relevance >= UNCERTAIN_THRESHOLD:
                relevance = "uncertain_by_similarity"

            logger.debug(f"Relevance assigned based on pipeline score: {relevance}")

            # --- TOP PIPELINE CATEGORIES ---
            pipeline_relevant_scores = []
            for idx, score_val in enumerate(pipeline_similarity_scores):
                # Ensure score_scalar is a pure Python scalar float
                score_scalar = _to_scalar_float(score_val)

                logger.debug(
                    f"  Pipeline Loop: idx={idx}, score_val_type={type(score_val)}, score_scalar_type={type(score_scalar)}, score_scalar_value={score_scalar}")
                if score_scalar >= UNCERTAIN_THRESHOLD:
                    pipeline_relevant_scores.append((score_scalar, pipeline_category_labels_for_this_main[idx]))
            pipeline_relevant_scores.sort(key=lambda x: x[0], reverse=True)
            top_pipeline_categories_by_sim = [label for score, label in pipeline_relevant_scores]
        else:
            # If no pipeline classification, final_scibert_most_similar_centroid already defaults to main category
            logger.debug(
                f"Skipping pipeline classification for '{predicted_main_category}' as it's not a pipeline-bearing category or no pipeline centroids exist for it.")

        return pd.Series({
            "scibert_score": final_scibert_score,
            "scibert_most_similar_centroid": final_scibert_most_similar_centroid,
            "scibert_relevance": relevance,
            "scibert_category": final_scibert_category,
            "scibert_pipeline_category": final_scibert_pipeline_category,
            "scibert_top_categories_by_similarity": top_categories_by_sim,
            "scibert_top_pipeline_categories_by_similarity": top_pipeline_categories_by_sim
        })

    logger.info("Classifying and annotating pool papers based on two-tiered similarity scores.")
    results_list = []

    for i in range(len(pool_df_for_embedding)):
        current_doi = pool_df_for_embedding.loc[i, 'doi']
        current_embedding = pool_embeddings[i]

        try:
            row_results = classify_and_annotate(current_embedding)
            results_list.append(row_results)
        except Exception as e:
            logger.critical(
                f"Error during classification for index {i} (DOI: {current_doi if pd.notna(current_doi) else 'N/A'}): {e}. This paper will have error status.")
            results_list.append(pd.Series({
                "scibert_score": np.nan,
                "scibert_most_similar_centroid": "",  # Ensure this is always initialized in the error case too
                "scibert_relevance": "classification_error",
                "scibert_category": "",
                "scibert_pipeline_category": "",
                "scibert_top_categories_by_similarity": [],
                "scibert_top_pipeline_categories_by_similarity": []
            }))

    results_df = pd.DataFrame(results_list)
    results_df['doi'] = pool_df_for_embedding['doi'].values

    # ---------------------------
    # Prepare Output DataFrame (Crucial Merge Logic)
    # ---------------------------
    final_scibert_df = pd.merge(pool_df, results_df, on='doi', how='left', suffixes=('_original', ''))

    category_cols_to_fill = {
        'scibert_score': np.nan,
        'scibert_most_similar_centroid': '',
        'scibert_relevance': 'no_embedding_data',
        'scibert_category': '',
        'scibert_pipeline_category': '',
        'scibert_top_categories_by_similarity': [],
        'scibert_top_pipeline_categories_by_similarity': []
    }

    for col, fill_value in category_cols_to_fill.items():
        if col in final_scibert_df.columns:
            if isinstance(fill_value, list):
                # For columns expected to hold lists (like top_categories_by_similarity),
                # we explicitly set dtype to object and fill NaNs using .loc
                # This avoids the problematic pd.isna(x) on potentially array-like elements in .apply()

                # 1. Ensure the column is of object dtype to safely store lists and NaNs
                final_scibert_df[col] = final_scibert_df[col].astype(object)

                # 2. Identify rows where the value is truly NaN (from the left merge)
                nan_mask = final_scibert_df[col].isna()

                # 3. Assign the fill_value (e.g., []) to these NaN locations.
                # Use a list multiplication to create a list of 'fill_value' for all NaN rows
                final_scibert_df.loc[nan_mask, col] = [fill_value] * nan_mask.sum()

            else:
                # For scalar fill values, .fillna() is robust and appropriate.
                final_scibert_df[col] = final_scibert_df[col].fillna(fill_value)
        else:
            # If the column doesn't exist, create it and fill with the default value
            if isinstance(fill_value, list):
                # For list fill values, create an object Series
                final_scibert_df[col] = pd.Series([fill_value] * len(final_scibert_df), dtype=object)
            else:
                final_scibert_df[col] = fill_value

    if 'doi_lower_original' in final_scibert_df.columns:
        final_scibert_df.drop(columns='doi_lower_original', inplace=True)

    final_scibert_df["scibert_analysis_date"] = datetime.date.today().isoformat()

    seed_original_info = seed_excel_df[['doi', 'category', 'pipeline_category']].copy()
    seed_original_info.rename(columns={
        'category': 'original_curated_category',
        'pipeline_category': 'original_curated_pipeline_category'
    }, inplace=True)
    final_scibert_df = pd.merge(final_scibert_df, seed_original_info, on='doi', how='left')

    final_scibert_df["is_seed"] = final_scibert_df["doi"].isin(seed_excel_df["doi"]).astype(bool)

    logger.info("SciBERT results DataFrame prepared.")
    return final_scibert_df


if __name__ == "__main__":
    logger.info("SciBERT Processor started as a standalone script.")
    try:
        scibert_output_df = run_scibert_analysis_pipeline()
        intermediate_output_path = os.path.join(BASE_DIR,
                                                f"scibert_intermediate_results_{datetime.date.today().isoformat()}.jsonl")
        scibert_output_df.to_json(intermediate_output_path, orient='records', lines=True)
        logger.info(f"Intermediate SciBERT results saved to '{intermediate_output_path}' (when run standalone).")
    except Exception as e:
        logger.critical(f"SciBERT Processor failed: {e}")
        sys.exit(1)