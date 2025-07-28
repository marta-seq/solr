# File: src/main_orchestrator.py

import pandas as pd
import datetime
import os
import sys
import numpy as np

# --- Path Setup (Crucial for finding other modules) ---
current_script_path = os.path.abspath(__file__)
src_directory = os.path.dirname(current_script_path)
project_root = os.path.dirname(os.path.dirname(src_directory))

if src_directory not in sys.path:
    sys.path.insert(0, src_directory)

# Import functions from other modules
from scibert_processor import run_scibert_analysis_pipeline
from concept_extractor import extract_concepts_for_paper

# Import logger setup from utils (assuming it's in src/utils)
from utils.logger_setup import setup_logging
from utils.doi import clean_doi  # Importing the clean_doi function from utils.doi

# ---------------------------
# Configuration Parameters
# ---------------------------
BASE_DIR = os.path.join(project_root, "data")
LOGGING_DIR = os.path.join(BASE_DIR, "logs/weak_annotator")

DOI_POOL_FILE = os.path.join(BASE_DIR, "doi_pool.jsonl")
FINAL_OUTPUT_FILE = os.path.join(BASE_DIR, f"weak_annotated.jsonl")

# ---------------------------
# Initialize Logging
# ---------------------------
logger = setup_logging(LOGGING_DIR, "weak_annotator_pipeline")
logger.info("Logging initialized for the main annotation pipeline.")
logger.info(f"Log file directory: {LOGGING_DIR}")


# --- Helper Function for Removing kw_detail_ columns (if they exist) ---
def remove_kw_detail_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all columns from a DataFrame whose names start with 'kw_detail_'.
    These are often granular boolean flags from keyword extraction that are
    later consolidated into list-based columns.
    """
    columns_to_remove = [col for col in df.columns if col.startswith('kw_detail_')]

    if columns_to_remove:
        logger.info(f"Removing {len(columns_to_remove)} columns starting with 'kw_detail_'.")
        df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
    else:
        logger.info("No columns starting with 'kw_detail_' found to remove.")
        df_cleaned = df.copy()  # Return a copy to avoid unexpected side effects

    return df_cleaned


# ---------------------------
# Consolidation Logic
# ---------------------------
def consolidate_final_relevance(row) -> tuple[str, str]:
    """
    Consolidates the final relevance based on a defined hierarchy, prioritizing keywords.
    Returns a tuple: (relevance_status, relevance_source).
    """
    scibert_relevance = row['scibert_relevance']
    kw_inclusion = row['kw_domain_inclusion']
    kw_exclusion = row['kw_domain_exclusion']
    kw_pipeline_cats = row['kw_pipeline_category']

    # Rule 1: Strongest negative rule - if contains exclusion keywords
    if kw_exclusion:
        return "low-relevance", "keyword_exclusion"

    # Rule 2: Strong positive rule - if contains core inclusion keywords
    if kw_inclusion:
        return "relevant", "keyword_inclusion"

    # Rule 3: If keyword extraction found specific pipeline categories
    if kw_pipeline_cats and len(kw_pipeline_cats) > 0:
        return "relevant", "keyword_pipeline"

    # Rule 4: If SciBERT marked it uncertain, but keywords confirm relevance
    if scibert_relevance == "uncertain_by_similarity" and kw_inclusion:
        return "relevant", "hybrid_keyword_confirm"

    # Rule 5: If it does NOT contain core inclusion keywords AND SciBERT didn't find it clearly relevant
    if not kw_inclusion and scibert_relevance in ["low-relevance_by_similarity", "uncertain_by_similarity"]:
        return "low-relevance", "keyword_no_core_terms"

    # Rule 6: Default to SciBERT's assessment if no stronger keyword rules apply
    if scibert_relevance == "relevant_by_similarity":
        return "relevant", "scibert_similarity"
    elif scibert_relevance == "low-relevance_by_similarity":
        return "low-relevance", "scibert_low_similarity"
    elif scibert_relevance == "uncertain_by_similarity":
        return "uncertain", "scibert_uncertainty"

    # Fallback for unexpected scibert_relevance values
    return None, "unknown"


def consolidate_final_categories(row) -> dict:
    """
    Consolidates the final general and pipeline categories, and includes specific
    keyword-extracted concept lists (tissue, disease, animal, methods, application areas).
    """
    scibert_cat = row['scibert_category']
    scibert_pipe_cat = row['scibert_pipeline_category']

    kw_pipeline_cats = row['kw_pipeline_category']
    kw_paper_types = row['kw_paper_type']

    current_general_categories = []
    if pd.notna(scibert_cat) and str(scibert_cat).lower() not in ["", "nan", "none"]:
        current_general_categories.append(scibert_cat)

    current_pipeline_categories = []
    if pd.notna(scibert_pipe_cat) and str(scibert_pipe_cat).lower() not in ["", "nan", "none"]:
        current_pipeline_categories.append(scibert_pipe_cat)

    if "Methodology Paper" in kw_paper_types and "Computational Method" not in current_general_categories:
        current_general_categories.insert(0, "Computational Method")

    for pt in kw_paper_types:
        if pt not in current_general_categories:
            current_general_categories.append(pt)

    for kw_pc in kw_pipeline_cats:
        if kw_pc not in current_pipeline_categories:
            current_pipeline_categories.append(kw_pc)

    final_general_categories = list(dict.fromkeys(current_general_categories))
    final_pipeline_categories = list(dict.fromkeys(current_pipeline_categories))

    # Retrieve specific keyword lists directly from the row
    # Assuming extract_concepts_for_paper returns these as lists
    kw_tissue_val = row.get('kw_tissue', [])
    kw_disease_val = row.get('kw_disease', [])
    kw_animal_val = row.get('kw_organism', [])  # Renaming kw_organism to kw_animal
    kw_methods_val = row.get('kw_detected_methods', [])
    kw_application_areas_val = row.get('kw_application_area', [])

    return {
        "category": final_general_categories,
        "pipeline_category": final_pipeline_categories,

        "kw_paper_types": kw_paper_types,  # Keep original name as it's not consolidated
        "kw_pipeline_categories": kw_pipeline_cats,  # Keep original name

        "kw_tissue": kw_tissue_val,
        "kw_disease": kw_disease_val,
        "kw_animal": kw_animal_val,  # Renamed
        "kw_methods": kw_methods_val,  # Use kw_methods for clarity
        "kw_application_areas": kw_application_areas_val,
    }


# ---------------------------
# Main Orchestration Function
# ---------------------------
def run_full_pipeline():
    logger.info("Starting the full automated annotation pipeline.")

    # ---------------------------
    # Step 1: Run SciBERT Analysis
    # ---------------------------
    logger.info("Running SciBERT processor to get similarity and initial relevance scores...")
    # try:
    #     scibert_results_df = run_scibert_analysis_pipeline()
    #     logger.info(f"SciBERT analysis completed. {len(scibert_results_df)} papers processed by SciBERT.")
    # except Exception as e:
    #     logger.critical(f"SciBERT analysis failed: {e}. Aborting pipeline.")
    #     sys.exit(1)
    scibert_results_df = run_scibert_analysis_pipeline()
    logger.info(f"SciBERT analysis completed. {len(scibert_results_df)} papers processed by SciBERT.")
    # ---------------------------
    # Step 2: Apply Keyword-Based Concept Extraction
    # ---------------------------
    logger.info("Applying keyword-based concept extraction for each paper...")
    for col in ['title', 'abstract', 'author_keywords']:
        if col not in scibert_results_df.columns:
            scibert_results_df[col] = ''

    concept_results = scibert_results_df.apply(
        lambda row: extract_concepts_for_paper(
            title=row['title'],
            abstract=row['abstract'],
            author_keywords=row['author_keywords']
        ),
        axis=1
    )

    concept_results_df = pd.json_normalize(concept_results)
    full_annotated_df = pd.concat([scibert_results_df, concept_results_df], axis=1)
    logger.info("Keyword-based concept extraction completed.")

    # --- Clean up kw_detail_ columns before final consolidation ---
    full_annotated_df = remove_kw_detail_columns(full_annotated_df)

    # ---------------------------
    # Step 3: Consolidate Final Relevance and Categories
    # ---------------------------
    logger.info("Consolidating final relevance, categories, and keyword concepts...")

    # Define all possible columns that might be needed by consolidation functions
    # Ensure they are initialized if not present, to prevent KeyError.
    expected_cols_from_pipeline = [
        'scibert_relevance', 'scibert_score',
        'scibert_probable_category_single_best', 'scibert_pipeline_category',
        'kw_domain_inclusion', 'kw_domain_exclusion',
        'kw_pipeline_category', 'kw_paper_type',
        'kw_tissue', 'kw_organism', 'kw_disease',  # Individual KW biological concepts
        'kw_detected_methods', 'kw_application_area',  # Individual KW fields
        'is_seed', 'original_curated_category', 'original_curated_pipeline_category'
    ]

    for col in expected_cols_from_pipeline:
        if col not in full_annotated_df.columns:
            logger.warning(f"Missing expected column '{col}'. Initializing with default values.")
            if col.startswith('kw_') and col not in ['kw_domain_inclusion', 'kw_domain_exclusion']:
                full_annotated_df[col] = [[] for _ in range(len(full_annotated_df))]
            elif col in ['kw_domain_inclusion', 'kw_domain_exclusion']:
                full_annotated_df[col] = False
            elif col == 'is_seed':
                full_annotated_df[col] = False
            elif col == 'scibert_score':
                full_annotated_df[col] = 0.0  # Default score
            else:  # Other SciBERT or original curated columns
                full_annotated_df[col] = np.nan

    # Apply the relevance consolidation function and unpack the tuple results
    relevance_results = full_annotated_df.apply(consolidate_final_relevance, axis=1, result_type='expand')
    relevance_results.columns = ['relevance_status', 'relevance_source']
    final_output_df = pd.concat([full_annotated_df, relevance_results], axis=1)

    # Apply the category consolidation function
    consolidated_cat_info = final_output_df.apply(consolidate_final_categories, axis=1)
    consolidated_cat_df = pd.json_normalize(consolidated_cat_info)
    final_output_df = pd.concat([final_output_df, consolidated_cat_df], axis=1)

    logger.info("Final relevance, categories, and keyword concepts consolidation completed.")

    # --- Override automated annotations with manual curation for seed papers ---
    logger.info("Overriding automated annotations with manual curation for seed papers...")

    final_output_df['original_curated_category'] = final_output_df['original_curated_category'].replace(
        {np.nan: None}).astype(object)
    final_output_df['original_curated_pipeline_category'] = final_output_df[
        'original_curated_pipeline_category'].replace({np.nan: None}).astype(object)

    is_seed_mask = final_output_df['is_seed'] == True

    # Override relevance for seed papers
    final_output_df.loc[is_seed_mask, 'relevance_status'] = "relevant"
    final_output_df.loc[is_seed_mask, 'relevance_source'] = "manual_curation"
    logger.info(
        f"Set 'relevance_status' and 'relevance_source' for {is_seed_mask.sum()} seed papers to manual curation.")

    # Override category and pipeline_category (which are now lists) with curated values
    mask_category = is_seed_mask & final_output_df['original_curated_category'].notna() & (
                final_output_df['original_curated_category'] != '')
    final_output_df.loc[mask_category, 'category'] = final_output_df.loc[
        mask_category, 'original_curated_category'].apply(lambda x: [x] if pd.notna(x) else [])
    logger.info(f"Overrode 'category' for {mask_category.sum()} seed papers with single curated value list.")

    mask_pipeline_category = is_seed_mask & final_output_df['original_curated_pipeline_category'].notna() & (
                final_output_df['original_curated_pipeline_category'] != '')
    final_output_df.loc[mask_pipeline_category, 'pipeline_category'] = final_output_df.loc[
        mask_pipeline_category, 'original_curated_pipeline_category'].apply(lambda x: [x] if pd.notna(x) else [])
    logger.info(
        f"Overrode 'pipeline_category' for {mask_pipeline_category.sum()} seed papers with single curated value list.")

    logger.info("Manual curation override completed.")

    # --- Set annotation_score based on manual curation ---
    final_output_df['annotation_score'] = 1.0  # Default score for all papers
    final_output_df.loc[is_seed_mask, 'annotation_score'] = 2.0  # Override for seed papers
    logger.info(f"Set 'annotation_score' to 2.0 for {is_seed_mask.sum()} seed papers.")

    # --- Column Renaming ---
    logger.info("Renaming columns for clarity.")
    column_renames = {
        'scibert_category': 'scibert_category',
        'scibert_pipeline_category': 'scibert_pipeline_category',
        'kw_organism': 'kw_organism',  # Specific rename for biological concepts
        'kw_detected_methods': 'kw_methods',  # General method terms
        'kw_application_area': 'kw_application_areas',  # General application areas
    }
    final_output_df = final_output_df.rename(
        columns={k: v for k, v in column_renames.items() if k in final_output_df.columns})

    logger.info("Final processing step: Selecting and ordering final output columns.")

    # ---------------------------
    # Step 4: Export Final Results - Select and Order Columns
    # ---------------------------
    logger.info(f"Exporting final annotated data to '{FINAL_OUTPUT_FILE}'...")

    # Dynamically get all original columns from the initial DataFrame (scibert_results_df)
    # This ensures all metadata columns are carried through by default
    original_metadata_cols = [col for col in scibert_results_df.columns if col not in [
        'scibert_relevance', 'scibert_score', 'scibert_category',
        'scibert_pipeline_category',  # These are SciBERT outputs
        'doi_lower'  # This is internal temporary
    ]]
    # Filter out columns that will be added or processed later
    original_metadata_cols = [
        col for col in original_metadata_cols if col not in [
            'original_curated_category', 'original_curated_pipeline_category',  # Dropped after transfer
            'kw_domain_inclusion', 'kw_domain_exclusion', 'kw_pipeline_category', 'kw_paper_type',
            'kw_tissue', 'kw_organism', 'kw_disease', 'kw_detected_methods', 'kw_application_area'
            # Processed kw fields
        ]
    ]
    # Ensure is_seed is included if it's considered metadata
    if 'is_seed' not in original_metadata_cols:
        original_metadata_cols.append('is_seed')
    # Get all columns directly from the initial pool_df
    pool_df = pd.read_json(DOI_POOL_FILE, lines=True)
    pool_df['doi'] = pool_df['doi'].apply(clean_doi)
    # change this the pool and stuff should enter here not in the scibert processor
    all_initial_pool_columns = pool_df.columns.tolist()

    # Define the exact list of final columns you want and their order
    # Any columns not in this list will be dropped.
    # Use dict.fromkeys to maintain order and remove duplicates
    final_columns_order = list(dict.fromkeys([
        # All columns from the original DOI pool file will be added here first,
        # in their original order.
        *all_initial_pool_columns,
        # 'doi',
        # 'title',
        # 'abstract',
        # 'publication_date',
        # 'journal',
        # 'authors',
        # 'source_file',  # Assuming this might exist
        # 'pmid', 'pmcid', 'arxiv_id', 'url',  # Common IDs
        # 'author_keywords', 'mesh_terms',  # Important original content
        'is_seed',  # Keep this to indicate original curated status
        *original_metadata_cols,  # Include any other original columns found

        # Final Consolidated Outputs
        'relevance_status',  # e.g., 'relevant', 'low-relevance', 'uncertain'
        'relevance_source',  # e.g., 'keyword_inclusion', 'scibert_similarity', 'manual_curation'
        'annotation_score',  # Score based on automation vs. manual
        'category',  # Final consolidated list of general categories (now a list)
        'pipeline_category',  # Final consolidated list of pipeline categories (now a list)

        # SciBERT specific outputs (after renaming)
        'scibert_relevance',  # Original SciBERT relevance string
        'scibert_score',  # SciBERT similarity score
        'scibert_category',  # Renamed single best category
        'scibert_pipeline_category',  # Renamed single best pipeline category (still might be null)

        # Keyword-based flags (boolean flags from keyword processing)
        'kw_domain_inclusion',
        'kw_domain_exclusion',

        # Keyword-based matched concepts (these are lists of terms, after specific renames)
        'kw_paper_types',  # Original kw_paper_type
        'kw_pipeline_categories',  # Original kw_pipeline_category
        'kw_tissue',  # Specific KW for tissue
        'kw_disease',  # Specific KW for disease
        'kw_animal',  # Specific KW for animal (from kw_organism)
        'kw_methods',  # Specific KW for methods (from kw_detected_methods)
        'kw_application_areas',  # Specific KW for application areas (from kw_application_area)
    ]))

    # Columns to explicitly drop regardless of 'final_columns_order' for tidiness
    # These are typically intermediate or duplicates.
    temp_columns_to_drop_after_consolidation = [
        'scibert_most_similar_centroid',
        'doi_lower',
        # Original curated columns are dropped after their values are used
        'original_curated_category',
        'original_curated_pipeline_category',
        # Old consolidated columns that are now split or removed
        # 'kw_matched_biological_concepts', # This was a combined list, now split to individual kw_tissue, etc.
        # 'kw_matched_paper_types', # Redundant if kw_paper_types is kept as the final
        # 'kw_matched_pipeline_categories_labels', # Redundant if kw_pipeline_categories is kept
        # 'final_all_general_categories', # Redundant as 'category' is now the list
        # 'final_all_pipeline_categories', # Redundant as 'pipeline_category' is now the list
    ]

    # Remove columns that are explicitly not desired or are redundant
    final_output_df = final_output_df.drop(
        columns=[col for col in temp_columns_to_drop_after_consolidation if col in final_output_df.columns],
        errors='ignore'
    )

    # Now, select and reorder the final columns to exactly match 'final_columns_order'
    cols_to_select = [col for col in final_columns_order if col in final_output_df.columns]
    missing_cols = [col for col in final_columns_order if col not in final_output_df.columns]
    if missing_cols:
        logger.warning(f"Some desired final columns are missing from the DataFrame: {missing_cols}")

    final_output_df = final_output_df[cols_to_select]

    import collections
    column_counts = collections.Counter(final_output_df.columns)
    duplicate_columns = [col for col, count in column_counts.items() if count > 1]
    print(duplicate_columns)
    final_output_df = final_output_df.loc[:, ~final_output_df.columns.duplicated(keep='last')]

    try:
        final_output_df.to_json(FINAL_OUTPUT_FILE, orient='records', lines=True, indent=None)
        logger.info(f"Full annotation pipeline completed successfully. Output saved to '{FINAL_OUTPUT_FILE}'.")
        print(f"Pipeline completed. Final annotated data saved to {FINAL_OUTPUT_FILE}")
    except Exception as e:
        logger.critical(f"Failed to save final annotated file to '{FINAL_OUTPUT_FILE}': {e}")
        sys.exit(1)

    # ---------------------------
    # Logging Summary
    # ---------------------------
    n_total = len(final_output_df)
    n_relevant = (final_output_df["relevance_status"] == "relevant").sum()
    n_uncertain = (final_output_df["relevance_status"] == "uncertain").sum()
    n_low = (final_output_df["relevance_status"] == "low-relevance").sum()
    n_seed_papers = final_output_df["is_seed"].sum()

    logger.info(f"--- Full Pipeline Summary ---")
    logger.info(f"Total papers processed: {n_total}")
    logger.info(f"Final Relevant papers: {n_relevant}")
    logger.info(f"Final Uncertain papers: {n_uncertain}")
    logger.info(f"Final Low-relevance papers: {n_low}")
    logger.info(f"Papers identified as original seeds: {n_seed_papers}")


if __name__ == "__main__":
    run_full_pipeline()