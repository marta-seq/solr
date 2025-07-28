# src/manual_curation/seed_manual_annotations.py
import pandas as pd
import json
import os
import sys
from datetime import datetime
import logging
script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add the project root to sys.path to enable imports like 'src.utils.logging_setup'
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Use insert(0) to pr

from src.utils.logging_setup import setup_logging
from src.utils.data_helpers import load_jsonl_to_dataframe, clean_doi, save_jsonl_records
from src.utils.file_helpers import ensure_dir

logger = logging.getLogger(__name__)


def seed_manual_annotations_from_pipeline_output(
        input_pipeline_output_path: str,
        output_jsonl_path: str,
        manual_score: float
):
    """
    Reads the output of the 'incorporate_curated_categories' rule, transforms it
    into the manual_annotations.jsonl format, and saves it.
    This script is intended for ONE-TIME use to initialize the manual_annotations.jsonl.
    """
    setup_logging(log_prefix="seed_manual_annotations")
    logger.info(f"--- Seeding Manual Annotations from Pipeline Output: {input_pipeline_output_path} ---")

    df_pipeline = load_jsonl_to_dataframe(input_pipeline_output_path)
    if df_pipeline.empty:
        logger.error(
            f"Input pipeline output file '{input_pipeline_output_path}' is empty or not found. Cannot seed manual annotations.")
        return

    df_pipeline['doi'] = df_pipeline['doi'].apply(clean_doi)
    df_pipeline.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_pipeline)} records from {input_pipeline_output_path}.")

    manual_records = []
    for index, row in df_pipeline.iterrows():
        # Only include records that actually have curated data
        if pd.notna(row.get('curated_paper_category')) or pd.notna(row.get('curated_pipeline_category')):
            record = {'doi': row['doi']}

            # Map existing 'curated_' columns to 'final_' columns
            if 'curated_paper_category' in row and pd.notna(row['curated_paper_category']):
                # Assuming curated_paper_category is already a list or can be directly used
                # If it's a string like "Category1, Category2", you'd need to split it:
                if isinstance(row['curated_paper_category'], str):
                    record['final_paper_classification'] = [cat.strip() for cat in
                                                            row['curated_paper_category'].split(',')]
                else:
                    record['final_paper_classification'] = row['curated_paper_category']
            else:
                record['final_paper_classification'] = []

            if 'curated_pipeline_category' in row and pd.notna(row['curated_pipeline_category']):
                record['final_pipeline_step'] = str(row['curated_pipeline_category']).strip()
            else:
                record['final_pipeline_step'] = None

            # You can also map 'curated_package_method_name' if you want it to override LLM
            if 'curated_package_method_name' in row and pd.notna(row['curated_package_method_name']):
                record['llm_annot_package_algorithm_name'] = str(row['curated_package_method_name']).strip()
            else:
                record[
                    'llm_annot_package_algorithm_name'] = None  # Or keep it out if you only want to override if present

            record['manual_reviewed'] = True
            record['llm_annotation_score'] = manual_score
            record['manual_review_date'] = datetime.now().isoformat()
            manual_records.append(record)

    ensure_dir(os.path.dirname(output_jsonl_path))
    save_jsonl_records(manual_records, output_jsonl_path, append=False)
    logger.info(f"Seeded {len(manual_records)} manual annotations to {output_jsonl_path}.")
    logger.info("--- Manual Annotation Seeding Complete ---")


if __name__ == "__main__":
    # This path should match config["paths"]["intermediate_papers_with_curated_categories"]
    input_pipeline_output_path = "data/intermediate/filtered_and_scored_papers.jsonl"
    output_manual_jsonl_path = "data/manual_curation/manual_annotations.jsonl"
    manual_score_value = 1.0  # Highest score for manual entries

    seed_manual_annotations_from_pipeline_output(
        input_pipeline_output_path=input_pipeline_output_path,
        output_jsonl_path=output_manual_jsonl_path,
        manual_score=manual_score_value
    )