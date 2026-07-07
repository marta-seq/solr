# src/weak_annotation/concept_extractor.py
import re
import yaml  # Changed from json
import os
import pandas as pd  # Import pandas for checking pd.isna
import logging  # Added logging import
from typing import Optional, List, Dict, Any
# Add project root to path for utility imports
import sys
import argparse  # Import argparse for command-line argument parsing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/home/martinha/PycharmProjects/phd/review"  # Assuming this is your project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging  # Import setup_logging
from src.utils.file_helpers import ensure_dir  # Import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, load_curated_csv_data, \
    clean_doi  # Import data_helpers functions

logger = logging.getLogger(__name__)  # Initialize logger


# Define the path to the concepts configuration directory
# This will be set dynamically in the main function based on argparse
# Removed the problematic global CONCEPTS_CONFIG_DIR definition here.


def load_keywords_from_yaml(filepath: str) -> Dict[str, List[str]]:
    """Loads keyword mappings from a YAML file at the specified filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Keyword mapping file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compile_regex_map(raw_map: dict) -> dict:
    """
    Compiles raw regex pattern strings into re.Pattern objects.
    Assumes the input strings already contain any necessary word boundaries (e.g., \\b)
    and escape characters.
    """
    compiled_map = {}
    for label, patterns in raw_map.items():
        compiled_patterns = []
        for pattern_str in patterns:
            try:
                compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern_str}' for label '{label}': {e}")
        compiled_map[label] = compiled_patterns
    return compiled_map


# Global variables for compiled keywords, will be populated by the ConceptExtractorRunner
# This structure is necessary because extract_concepts_for_paper is a function, not a method
# and needs access to these compiled maps. They are loaded once at module import.
COMPILED_DOMAIN_INCLUSION_KEYWORDS = {}
COMPILED_DOMAIN_EXCLUSION_KEYWORDS = {}
COMPILED_TISSUE_MAP = {}
COMPILED_ANIMAL_MAP = {}
COMPILED_DISEASE_MAP = {}
COMPILED_PAPER_TYPES = {}
COMPILED_PIPELINE_TYPES_KEYWORDS = {}
COMPILED_GENERAL_METHODS = {}


# This block will run when the module is imported.
# It needs the CONCEPTS_CONFIG_DIR_ABS to be defined globally.
# Let's adjust the loading logic to happen within the ConceptExtractorRunner's __init__
# to ensure it uses the path passed via argparse.
# For now, we'll keep the global variables but load them via the runner.

# --- Helper function for matching ---
def match_first(text: str, pattern_map: Dict[str, List[re.Pattern]]) -> Optional[str]:
    """
    Returns the label of the first matching keyword category found in the text.
    """
    for label, patterns in pattern_map.items():
        if any(p.search(text) for p in patterns):
            return label
    return None


def match_all(text: str, pattern_map: Dict[str, List[re.Pattern]]) -> List[str]:
    """
    Returns a list of all matching keyword category labels found in the text.
    """
    matches = []
    for label, patterns in pattern_map.items():
        if any(p.search(text) for p in patterns):
            matches.append(label)
    return list(set(matches))  # Return unique matches


def extract_matching_keywords_detail(text: str, pattern_map: Dict[str, List[re.Pattern]]) -> Dict[str, List[str]]:
    """
    Extracts all specific text fragments that matched for each category.
    Returns a dictionary mapping category name to a list of unique matched text fragments.
    """
    matched_keywords_by_category = {}
    for category, patterns in pattern_map.items():
        matched_terms = []
        for p in patterns:
            # Find all non-overlapping matches for the current pattern
            found_terms = [match.group(0) for match in p.finditer(text)]
            if found_terms:
                matched_terms.extend(found_terms)
        if matched_terms:
            matched_keywords_by_category[category] = sorted(list(set(term.lower() for term in matched_terms)))
    return matched_keywords_by_category


def extract_concepts_for_paper(
        title: str,
        abstract: str,
        author_keywords: Optional[List[str]] = None,
        compiled_keywords: Dict[str, Dict[str, List[re.Pattern]]] = None  # Pass compiled keywords
) -> Dict[str, Any]:
    """
    Extracts various concepts (keywords/regex) from a paper's title, abstract, and author keywords.

    Args:
        title (str): The title of the paper.
        abstract (str): The abstract of the paper.
        author_keywords (Optional[List[str]]): List of keywords provided by authors.
                                               Can also be a string or NaN if read from a varied source.
        compiled_keywords (Dict[str, Dict[str, List[re.Pattern]]]): Dictionary of all compiled keyword maps.

    Returns:
        Dict[str, Any]: A dictionary containing extracted concepts and boolean flags for domain relevance.
    """
    if compiled_keywords is None:
        raise ValueError("Compiled keywords must be provided to extract_concepts_for_paper.")

    # Robustly handle author_keywords, converting to string if not already a list of strings
    keywords_text = ""
    if author_keywords is not None:
        if isinstance(author_keywords, list):
            keywords_text = " ".join(author_keywords)
        elif isinstance(author_keywords, str):
            keywords_text = author_keywords
        elif pd.isna(author_keywords):  # Handle NaN from pandas
            keywords_text = ""
        else:  # Attempt to convert other types to string
            keywords_text = str(author_keywords)

    # Combine all text fields, ensuring None/NaN values are treated as empty strings
    combined_text = f"{title or ''} {abstract or ''} {keywords_text}".strip()
    text_lower = combined_text.lower()

    # Keyword-based domain relevance checks (boolean flags)
    # Access compiled keywords from the passed dictionary
    kw_domain_inclusion = any(
        p.search(text_lower) for patterns in compiled_keywords['domain_inclusion_keywords'].values() for p in patterns)
    kw_domain_exclusion = any(
        p.search(text_lower) for patterns in compiled_keywords['domain_exclusion_keywords'].values() for p in patterns)

    # Extract all matching categories (list of labels)
    kw_tissue = match_all(text_lower, compiled_keywords['tissue_keywords'])
    kw_organism = match_all(text_lower, compiled_keywords['animal_keywords'])
    kw_disease = match_all(text_lower, compiled_keywords['disease_keywords'])
    kw_paper_type = match_all(text_lower, compiled_keywords['paper_type_keywords'])
    kw_pipeline_category = match_all(text_lower, compiled_keywords['pipeline_type_keywords'])
    kw_detected_methods = match_all(text_lower, compiled_keywords['general_method_keywords'])

    # Extract detailed matched text fragments within each category (for richer output/debugging)
    kw_detail_pipeline_categories = extract_matching_keywords_detail(text_lower,
                                                                     compiled_keywords['pipeline_type_keywords'])

    # Combine biological concept maps for a single detailed biological concepts field
    combined_biological_maps = {}
    combined_biological_maps.update(compiled_keywords['tissue_keywords'])
    combined_biological_maps.update(compiled_keywords['animal_keywords'])
    combined_biological_maps.update(compiled_keywords['disease_keywords'])
    kw_detail_biological_concepts = extract_matching_keywords_detail(text_lower, combined_biological_maps)

    return {
        "kw_domain_inclusion": kw_domain_inclusion,
        "kw_domain_exclusion": kw_domain_exclusion,
        "kw_tissue": kw_tissue,
        "kw_organism": kw_organism,
        "kw_disease": kw_disease,
        "kw_paper_type": kw_paper_type,
        "kw_pipeline_category": kw_pipeline_category,
        "kw_detected_methods": kw_detected_methods,
        # Detailed matches for specific categories, useful for granular analysis
        "kw_detail_pipeline_categories": kw_detail_pipeline_categories,
        "kw_detail_biological_concepts": kw_detail_biological_concepts
    }


class ConceptExtractorRunner:
    def __init__(self, concepts_config_dir: str, curated_csv_path: str):
        self.concepts_config_dir = concepts_config_dir
        self.curated_data = load_curated_csv_data(curated_csv_path)
        logger.info(f"Loaded {len(self.curated_data)} curated entries.")
        self.compiled_keywords = self._load_and_compile_all_keywords()

    def _load_and_compile_all_keywords(self) -> Dict[str, Dict[str, List[re.Pattern]]]:
        """Loads and compiles all keyword maps from the specified directory."""
        all_compiled_keywords = {}
        try:
            # Core domain relevance checks
            domain_inclusion_data = load_keywords_from_yaml(
                os.path.join(self.concepts_config_dir, "domain_inclusion_keywords.yaml"))
            if isinstance(domain_inclusion_data, dict) and len(domain_inclusion_data) == 1:
                all_compiled_keywords['domain_inclusion_keywords'] = compile_regex_map(domain_inclusion_data)
            else:
                logger.warning(
                    f"Expected 'domain_inclusion.yaml' to contain a single dictionary. Found: {domain_inclusion_data}. Skipping.")

            domain_exclusion_data = load_keywords_from_yaml(
                os.path.join(self.concepts_config_dir, "domain_exclusion_keywords.yaml"))
            if isinstance(domain_exclusion_data, dict) and len(domain_exclusion_data) == 1:
                all_compiled_keywords['domain_exclusion_keywords'] = compile_regex_map(domain_exclusion_data)
            else:
                logger.warning(
                    f"Expected 'domain_exclusion.yaml' to contain a single dictionary. Found: {domain_exclusion_data}. Skipping.")

            # Specific concept categories
            all_compiled_keywords['tissue_keywords'] = compile_regex_map(
                load_keywords_from_yaml(os.path.join(self.concepts_config_dir, "tissue_keywords.yaml")))
            all_compiled_keywords['animal_keywords'] = compile_regex_map(
                load_keywords_from_yaml(os.path.join(self.concepts_config_dir, "animal_keywords.yaml")))
            all_compiled_keywords['disease_keywords'] = compile_regex_map(
                load_keywords_from_yaml(os.path.join(self.concepts_config_dir, "disease_keywords.yaml")))
            all_compiled_keywords['paper_type_keywords'] = compile_regex_map(
                load_keywords_from_yaml(os.path.join(self.concepts_config_dir, "paper_type_keywords.yaml")))
            all_compiled_keywords['pipeline_type_keywords'] = compile_regex_map(
                load_keywords_from_yaml(os.path.join(self.concepts_config_dir, "pipeline_type_keywords.yaml")))
            all_compiled_keywords['general_method_keywords'] = compile_regex_map(
                load_keywords_from_yaml(os.path.join(self.concepts_config_dir, "general_method_keywords.yaml")))

            logger.info("All concept keyword files loaded and compiled successfully.")

        except FileNotFoundError as e:
            logger.critical(
                f"CRITICAL ERROR: Missing keyword mapping file: {e}. Ensure all YAML files are in '{self.concepts_config_dir}' directory and named correctly.",
                exc_info=True)
            raise
        except yaml.YAMLError as e:
            logger.critical(
                f"CRITICAL ERROR: Error parsing YAML keyword file: {e}. Check YAML file format in '{self.concepts_config_dir}'.",
                exc_info=True)
            raise
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: An unexpected error occurred during keyword map loading: {e}",
                            exc_info=True)
            raise
        return all_compiled_keywords

    def run(self, input_file: str, output_file: str):
        """
        Loads papers, extracts concepts, and saves the updated papers.
        Applies manual curation data if available.
        """
        logger.info(f"Loading papers from {input_file} for concept extraction.")
        df_papers = load_jsonl_to_dataframe(input_file)

        if df_papers.empty:
            logger.warning(f"No papers loaded from {input_file}. Creating empty output file.")
            ensure_dir(os.path.dirname(output_file))
            save_jsonl_records([], output_file, append=False)
            return

        processed_records = []
        for index, row in df_papers.iterrows():
            record = row.to_dict()

            # Apply manual curation data first if available
            cleaned_doi = clean_doi(record.get('doi', ''))
            if cleaned_doi and cleaned_doi in self.curated_data:
                curated_entry = self.curated_data[cleaned_doi]
                record.update(curated_entry)
                # Ensure kw_domain_inclusion from curation is prioritized, if it exists
                record['kw_domain_inclusion'] = curated_entry.get('kw_domain_inclusion',
                                                                  record.get('kw_domain_inclusion', False))
                # logger.debug(
                #     f"Applied manual curation for DOI: {cleaned_doi}. kw_domain_inclusion (from curation): {record['kw_domain_inclusion']}")

            # Then extract concepts from the record (which might override/add to curation)
            # Pass title, abstract, author_keywords and the compiled_keywords
            extracted_concepts = extract_concepts_for_paper(
                title=record.get('title', ''),
                abstract=record.get('abstract', ''),
                author_keywords=record.get('author_keywords'),
                compiled_keywords=self.compiled_keywords  # Pass the compiled keywords
            )
            record.update(extracted_concepts)  # Update the record with extracted concepts

            # No debug print here, as per user's request for July 23rd version

            processed_records.append(record)  # Append the updated record

        logger.info(f"Extracted concepts for {len(processed_records)} papers.")
        ensure_dir(os.path.dirname(output_file))
        save_jsonl_records(processed_records, output_file, append=False)
        logger.info(f"Papers with extracted concepts saved to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description="Extracts predefined concepts from paper titles and abstracts."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing cleaned papers."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file for papers with extracted concepts."
    )
    parser.add_argument(
        "--concepts_config_dir",
        type=str,
        required=True,
        help="Path to the directory containing YAML files with concept keywords."
    )
    args = parser.parse_args()

    setup_logging(log_prefix="concept_extractor")

    # Assuming paper_classification_cleaned.csv is in data/inputs/ relative to project root
    curated_csv_path = os.path.join(project_root, "data/inputs/paper_classification_cleaned.csv")

    runner = ConceptExtractorRunner(args.concepts_config_dir, curated_csv_path)
    runner.run(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
