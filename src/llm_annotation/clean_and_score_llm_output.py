# src/post_llm_processing/clean_and_score_llm_output.py
import pandas as pd
import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Union

# Import utility functions
from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)


def clean_llm_output(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and normalizes LLM-generated fields within a single record.
    This is where you'd handle specific LLM quirks.
    """
    # Ensure classification is a dict, default to empty if not
    classification = record.get('llm_annot_classification')
    if not isinstance(classification, dict):
        classification = {}
        record['llm_annot_classification'] = classification

    # Ensure boolean fields are actually booleans
    if 'is_computational_methods_paper' in classification:
        comp_method = classification['is_computational_methods_paper']
        if isinstance(comp_method, str):
            classification['is_computational_methods_paper'] = comp_method.lower() == 'true'
        elif not isinstance(comp_method, bool):
            classification['is_computational_methods_paper'] = False  # Default to False if not boolean/string

    # Ensure list fields are lists, default to empty list if not
    list_fields = [
        'llm_annot_contribution_type',
        'llm_annot_algorithms_bases_used',
        'llm_annot_pipeline_analysis_steps',
        'llm_annot_tested_data_modalities',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_compared_algorithms_packages',
        'llm_annot_data_used_details'
    ]
    for field in list_fields:
        if not isinstance(record.get(field), list):
            record[field] = []

    # Ensure nested lists like assigned_categories are lists
    if 'assigned_categories' in classification and not isinstance(classification['assigned_categories'], list):
        classification['assigned_categories'] = []

    # Ensure nested objects like code_availability_details are dicts
    if not isinstance(record.get('llm_annot_code_availability_details'), dict):
        record['llm_annot_code_availability_details'] = {"status": "Unclear", "link": None, "license": None,
                                                         "version": None}

    # If the LLM still hallucinates prompt text into other fields, add specific cleaning here.
    # Example: If 'llm_annot_main_goal_of_paper' contains "A concise summary (1-3 sentences)..."
    # You might use regex or string methods to remove such patterns.
    # For now, we rely on the stricter prompt to prevent this.

    return record


def clean_keyword_paper_type(kw_paper_type_list: Union[List[str], str], mapping: Dict[str, List[str]]) -> List[str]:
    """
    Cleans and normalizes the kw_paper_type field based on a mapping.
    Handles cases where it might be a string (e.g., from a malformed JSONL line).
    """
    if isinstance(kw_paper_type_list, str):
        # Attempt to parse if it's a string representation of a list
        try:
            kw_paper_type_list = json.loads(kw_paper_type_list)
        except json.JSONDecodeError:
            kw_paper_type_list = [kw_paper_type_list]  # Treat as a single string if not JSON list

    if not isinstance(kw_paper_type_list, list):
        return []  # Return empty list if it's not a list after all attempts

    cleaned_types = []
    for item in kw_paper_type_list:
        if isinstance(item, str):
            # Check for direct matches or apply mapping
            if item in mapping:
                cleaned_types.extend(mapping[item])
            else:
                # Basic cleaning for individual items not in mapping
                cleaned_item = item.replace('computational analysis - ', '').replace(';', ',').strip()
                cleaned_types.append(cleaned_item)
    return sorted(list(set(cleaned_types)))  # Return unique and sorted


def calculate_annotation_score(
        record: Dict[str, Any],
        completeness_weight: float,
        method_bonus: float,
        section_bonus: float
) -> float:
    """
    Calculates an annotation score based on LLM output completeness and other factors.
    Score ranges from 0.0 to 1.0.
    """
    score = 0.0

    # Base score for successful LLM annotation
    if record.get('llm_annotation_status') == 'middle_llm_annotated':
        score += 0.5  # Base for being processed

        # 1. Completeness of key LLM fields
        llm_fields_to_check = [
            'llm_annot_main_goal_of_paper',
            'llm_annot_classification',
            'llm_annot_contribution_type'
        ]

        filled_fields = 0
        for field in llm_fields_to_check:
            val = record.get(field)
            if val is not None and (isinstance(val, list) and val) or (isinstance(val, dict) and val) or (
                    isinstance(val, str) and val.strip()):
                filled_fields += 1

        completeness_score = (filled_fields / len(llm_fields_to_check)) * completeness_weight
        score += completeness_score

        # 2. Bonus for computational methods papers
        classification = record.get('llm_annot_classification', {})
        if isinstance(classification, dict) and classification.get('is_computational_methods_paper') == True:
            score += method_bonus
            # If it's a method paper, check completeness of method-specific fields
            method_specific_fields = [
                'llm_annot_package_algorithm_name',
                'llm_annot_algorithms_bases_used',
                'llm_annot_pipeline_analysis_steps',
                'llm_annot_tested_data_modalities',
                'llm_annot_tested_assay_types_platforms',
                'llm_annot_compared_algorithms_packages',
                'llm_annot_code_availability_details',
                'llm_annot_data_used_details'
            ]
            filled_method_fields = 0
            for field in method_specific_fields:
                val = record.get(field)
                if val is not None and (isinstance(val, list) and val) or (isinstance(val, dict) and val) or (
                        isinstance(val, str) and val.strip()):
                    filled_method_fields += 1

            # Add a portion of method-specific completeness
            if method_specific_fields:  # Avoid division by zero
                score += (filled_method_fields / len(method_specific_fields)) * (method_bonus / 2)  # Smaller bonus

        # 3. Bonus for having full text sections available during extraction
        if record.get('sections_extracted_successfully') == True:
            score += section_bonus
    else:
        # If LLM annotation failed or was skipped, score is 0
        score = 0.0

    return min(1.0, round(score, 3))  # Cap at 1.0 and round for neatness


def clean_and_score_llm_output_main(
        input_file: str,
        output_file: str,
        completeness_weight: float,
        method_bonus: float,
        section_bonus: float,
        kw_paper_type_mapping_str: str  # Received as string from Snakemake
):
    """
    Main function to clean LLM annotations, normalize keyword types,
    and calculate an annotation score.
    """
    setup_logging(log_prefix="clean_score_llm")
    logger.info("--- Starting LLM Output Cleaning and Scoring ---")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")

    # Parse the kw_paper_type_mapping from string back to dict
    kw_paper_type_mapping = json.loads(kw_paper_type_mapping_str)

    df = load_jsonl_to_dataframe(input_file)
    if df.empty:
        logger.warning("Input DataFrame is empty. Creating empty output file.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records([], output_file, append=False)
        return

    processed_records = []
    for index, row in df.iterrows():
        record = row.to_dict()

        # Clean LLM output
        record = clean_llm_output(record)

        # Clean kw_paper_type
        if 'kw_paper_type' in record:
            record['kw_paper_type_cleaned'] = clean_keyword_paper_type(record['kw_paper_type'], kw_paper_type_mapping)
        else:
            record['kw_paper_type_cleaned'] = []

        # Calculate annotation score
        record['llm_annotation_score'] = calculate_annotation_score(
            record, completeness_weight, method_bonus, section_bonus
        )

        # Derive final paper classification and pipeline step from LLM output
        # These will be the default values, to be potentially overridden by manual curation
        classification = record.get('llm_annot_classification', {})
        record['final_paper_classification'] = classification.get('assigned_categories', [])

        # For pipeline step, we can take the first relevant one if it's a method paper
        if classification.get('is_computational_methods_paper') == True:
            pipeline_steps = record.get('llm_annot_pipeline_analysis_steps', [])
            if pipeline_steps:
                record['final_pipeline_step'] = pipeline_steps[0]  # Take the first one as primary
            else:
                record['final_pipeline_step'] = None
        else:
            record['final_pipeline_step'] = None  # Not applicable for non-method papers

        # Add a flag for manual review status (default to False)
        record['manual_reviewed'] = False
        record['manual_review_date'] = None  # Placeholder for when it's manually reviewed

        processed_records.append(record)

    save_jsonl_records(processed_records, output_file, append=False)
    logger.info(f"Finished cleaning and scoring. Saved {len(processed_records)} records to {output_file}.")
    logger.info("--- LLM Output Cleaning and Scoring Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean LLM annotations, normalize keyword types, and calculate annotation scores."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file with LLM annotated papers."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for cleaned and scored papers."
    )
    parser.add_argument(
        "--completeness_weight",
        type=float,
        default=0.6,
        help="Weight for the completeness of key LLM fields in score calculation."
    )
    parser.add_argument(
        "--method_bonus",
        type=float,
        default=0.2,
        help="Bonus score for papers classified as computational methods."
    )
    parser.add_argument(
        "--section_bonus",
        type=float,
        default=0.2,
        help="Bonus score for papers where full text sections were successfully extracted."
    )
    parser.add_argument(
        "--kw_paper_type_mapping",
        type=str,
        required=True,
        help="JSON string of the mapping for kw_paper_type cleaning."
    )

    args = parser.parse_args()

    clean_and_score_llm_output_main(
        input_file=args.input_file,
        output_file=args.output_file,
        completeness_weight=args.completeness_weight,
        method_bonus=args.method_bonus,
        section_bonus=args.section_bonus,
        kw_paper_type_mapping_str=args.kw_paper_type_mapping
    )