# src/data_processing/integrate_manual_reviews.py
import pandas as pd
import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path for utility imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi, load_curated_csv_data

logger = logging.getLogger(__name__)


def integrate_manual_reviews(
        manual_comp_jsonl: str,
        manual_non_comp_jsonl: str,
        papers_to_delete_csv: str,
        final_master_comp_jsonl: str,
        final_master_non_computational_jsonl: str,
        final_manual_review_score: int
):
    """
    Integrates manual review edits and deletions to produce the final master
    computational and non-computational paper datasets.
    The master files are derived directly from the manual review files,
    excluding those marked for deletion.
    """
    setup_logging(log_prefix="integrate_manual_reviews")
    logger.info("--- Starting Manual Review Integration Process ---")

    logger.info(f"Manual computational JSONL: {manual_comp_jsonl}")
    logger.info(f"Manual non-computational JSONL: {manual_non_comp_jsonl}")
    logger.info(f"Papers to delete CSV: {papers_to_delete_csv}")
    logger.info(f"Final master computational JSONL: {final_master_comp_jsonl}")
    logger.info(f"Final master non-computational JSONL: {final_master_non_computational_jsonl}")
    logger.info(f"Final manual review score: {final_manual_review_score}")

    # 1. Load manual review JSONL files
    df_manual_comp = load_jsonl_to_dataframe(manual_comp_jsonl)
    if not df_manual_comp.empty:
        df_manual_comp['doi'] = df_manual_comp['doi'].apply(clean_doi)
        df_manual_comp.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_manual_comp)} manual computational papers.")

    df_manual_non_comp = load_jsonl_to_dataframe(manual_non_comp_jsonl)
    if not df_manual_non_comp.empty:
        df_manual_non_comp['doi'] = df_manual_non_comp['doi'].apply(clean_doi)
        df_manual_non_comp.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_manual_non_comp)} manual non-computational papers.")

    df_all_manual_review = pd.concat([df_manual_comp, df_manual_non_comp], ignore_index=True)
    if not df_all_manual_review.empty:
        df_all_manual_review['doi'] = df_all_manual_review['doi'].apply(clean_doi)
        df_all_manual_review.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Combined {len(df_all_manual_review)} unique papers from manual review files.")

    # 2. Load papers to delete from CSV
    df_to_delete = load_curated_csv_data(papers_to_delete_csv)  # Using load_curated_csv_data for CSV
    deleted_dois = set()
    if not df_to_delete.empty and 'doi' in df_to_delete.columns:
        deleted_dois = set(df_to_delete['doi'].apply(clean_doi).tolist())
        logger.info(f"Loaded {len(deleted_dois)} DOIs to delete from {papers_to_delete_csv}.")
    else:
        logger.info(f"No DOIs to delete found in {papers_to_delete_csv} or file is empty/missing 'doi' column.")

    # 3. Filter out deleted papers and apply final scores/statuses
    final_master_papers_list = []
    initial_papers_count = len(df_all_manual_review)
    logger.info(f"Starting with {initial_papers_count} papers before deletion and final processing.")

    for index, row in df_all_manual_review.iterrows():
        doi = row.get('doi')
        if not doi or doi in deleted_dois:
            logger.debug(f"Skipping DOI {doi} as it's marked for deletion or missing DOI.")
            continue

        current_paper = row.to_dict()

        # Set definitive score for manually reviewed papers (if not deleted)
        manual_status = current_paper.get('manual_review_status', '').lower()
        if manual_status in ['reviewed', 'reclassify_to_comp', 'reclassify_to_noncomp']:
            current_paper['annotation_score'] = float(final_manual_review_score)
            logger.debug(
                f"DOI {doi}: Set annotation_score to {final_manual_review_score} due to manual review status '{manual_status}'.")

            # Ensure llm_annot_classification is a dictionary before modifying
            if 'llm_annot_classification' not in current_paper or not isinstance(
                    current_paper['llm_annot_classification'], dict):
                current_paper['llm_annot_classification'] = {}

            # Update llm_annotation_status based on manual action
            if manual_status == 'reclassify_to_comp':
                current_paper['llm_annot_classification']['is_relevant_to_spatial_omics_analysis'] = True
                current_paper['llm_annotation_status'] = 'human_reclassified_to_comp'
                logger.info(f"DOI {doi} reclassified to computational by manual review.")
            elif manual_status == 'reclassify_to_noncomp':
                current_paper['llm_annot_classification']['is_relevant_to_spatial_omics_analysis'] = False
                current_paper['llm_annotation_status'] = 'human_reclassified_to_noncomp'
                logger.info(f"DOI {doi} reclassified to non-computational by manual review.")
            elif manual_status == 'reviewed':
                if current_paper.get('llm_annotation_status') == 'detailed_llm_annotated':
                    current_paper['llm_annotation_status'] = 'human_reviewed_detailed'
                elif current_paper.get('llm_annotation_status') == 'middle_llm_annotated':
                    current_paper['llm_annotation_status'] = 'human_reviewed_broad_only'
                else:
                    current_paper['llm_annotation_status'] = 'human_reviewed'
                logger.info(f"DOI {doi} broadly reviewed by human.")

        # Ensure extracted_sections (text) are removed and replaced by has_ flags
        for section_col in [
            'extracted_introduction_section', 'extracted_methods_section',
            'extracted_results_discussion_section', 'extracted_conclusion_section',
            'extracted_data_availability_section', 'extracted_code_availability_section'
        ]:
            if section_col in current_paper and isinstance(current_paper[section_col], str):
                del current_paper[section_col]
                logger.debug(f"Removed text content for {section_col} from DOI {doi}.")

        final_master_papers_list.append(current_paper)

    logger.info(f"After deletion, {len(final_master_papers_list)} papers remain for final master files.")

    # Define the comprehensive set of final output columns
    all_possible_kw_cols = set()
    for df_source in [df_manual_comp, df_manual_non_comp]:
        for col in df_source.columns:
            if col.startswith('kw_'):
                all_possible_kw_cols.add(col)

    final_output_columns_template = [
        'doi', 'title', 'abstract', 'annotation_score', 'pipeline_category',
        'llm_annotation_status', 'llm_model_used', 'llm_annotation_error',
        'llm_annot_classification',
        'llm_annot_summary_and_justification',
        'llm_annot_package_algorithm_name', 'llm_annot_pipeline_analysis_steps',
        'llm_annot_tested_data_modalities', 'llm_annot_tested_assay_types_platforms',
        'llm_annot_compared_algorithms_packages', 'llm_annot_code_availability_details',
        'llm_annot_data_used_details', 'llm_annot_primary_programming_languages',
        'llm_annot_llm_notes',
        'manual_review_status', 'manual_notes', 'manual_reviewer_id', 'manual_review_timestamp',
        'full_text_extraction_status', 'full_text_extraction_timestamp', 'full_text_extraction_error',
        'full_text_pdf_url',
        'has_introduction_section', 'has_methods_section', 'has_results_discussion_section',
        'has_conclusion_section', 'has_data_availability_section', 'has_code_availability_section'
    ]
    final_output_columns_template.extend(sorted(list(all_possible_kw_cols)))

    if not final_master_papers_list:
        df_final = pd.DataFrame(columns=final_output_columns_template)
        logger.warning("final_master_papers_list was empty. Created an empty DataFrame with all expected columns.")
    else:
        df_final = pd.DataFrame(final_master_papers_list)
        df_final = df_final.reindex(columns=final_output_columns_template, fill_value=None)
        logger.info(f"Created df_final with {len(df_final)} records and {len(df_final.columns)} columns.")

    # Convert complex JSON string columns back to actual Python objects if they were stringified during some process
    json_columns = [
        'llm_annot_classification',
        'llm_annot_code_availability_details',
        'llm_annot_compared_algorithms_packages',
        'llm_annot_data_used_details'
    ]
    for col in json_columns:
        if col in df_final.columns:
            # Ensure the value is a string before attempting json.loads
            df_final[col] = df_final[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            # Ensure it's a dict/list after parsing, if not, set to None/empty structure
            if col == 'llm_annot_classification':
                df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, dict) else {})
            elif col in ['llm_annot_compared_algorithms_packages', 'llm_annot_data_used_details']:
                df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, list) else [])
            else:  # For other dict types
                df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, dict) else {})

    # Split into final computational and non-computational based on 'is_relevant_to_spatial_omics_analysis'
    # Ensure 'llm_annot_classification' is a dictionary for .get()
    df_final_comp = df_final[
        df_final['llm_annot_classification'].apply(
            lambda x: isinstance(x, dict) and x.get('is_relevant_to_spatial_omics_analysis') == True)
    ].copy()
    df_final_non_comp = df_final[
        df_final['llm_annot_classification'].apply(
            lambda x: isinstance(x, dict) and x.get('is_relevant_to_spatial_omics_analysis') == False)
    ].copy()

    logger.info(f"Splitting results: {len(df_final_comp)} computational, {len(df_final_non_comp)} non-computational.")

    ensure_dir(os.path.dirname(final_master_comp_jsonl))
    ensure_dir(os.path.dirname(final_master_non_computational_jsonl))

    save_jsonl_records(df_final_comp.to_dict(orient='records'), final_master_comp_jsonl, append=False)
    logger.info(f"Saved {len(df_final_comp)} final computational papers to {final_master_comp_jsonl}.")

    save_jsonl_records(df_final_non_comp.to_dict(orient='records'), final_master_non_computational_jsonl, append=False)
    logger.info(
        f"Saved {len(df_final_non_comp)} final non-computational papers to {final_master_non_computational_jsonl}.")

    logger.info("--- Manual Review Integration Process Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrates manual review edits and deletions to produce the final master datasets."
    )
    parser.add_argument(
        "--manual_computational_jsonl",
        type=str,
        required=True,
        help="Path to the manually edited JSONL for computational papers."
    )
    parser.add_argument(
        "--manual_non_computational_jsonl",
        type=str,
        required=True,
        help="Path to the manually edited JSONL for non-computational papers."
    )
    parser.add_argument(
        "--papers_to_delete_csv",
        type=str,
        required=True,
        help="Path to the CSV file containing DOIs of papers to delete."
    )
    parser.add_argument(
        "--final_master_computational_jsonl",
        type=str,
        required=True,
        help="Path to the final output master computational papers JSONL file."
    )
    parser.add_argument(
        "--final_master_non_computational_jsonl",
        type=str,
        required=True,
        help="Path to the final output master non-computational papers JSONL file."
    )
    parser.add_argument(
        "--final_manual_review_score",
        type=int,
        required=True,
        help="The score to assign to papers that have undergone manual review."
    )
    args = parser.parse_args()

    integrate_manual_reviews(
        manual_comp_jsonl=args.manual_computational_jsonl,
        manual_non_comp_jsonl=args.manual_non_computational_jsonl,
        papers_to_delete_csv=args.papers_to_delete_csv,
        final_master_comp_jsonl=args.final_master_computational_jsonl,
        final_master_non_computational_jsonl=args.final_master_non_computational_jsonl,
        final_manual_review_score=args.final_manual_review_score
    )