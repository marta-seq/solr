# generate_annotation_statistics.py

import json
import os
import logging
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Tuple

# --- Configuration (should match your weak_annotator.py) ---
BASE_DIR = "../../data"
LOGGING_DIR = os.path.join(BASE_DIR, "logging_weak_annotator") # New logging dir for stats
WEAK_ANNOTATION_POOL_FILE = os.path.join(BASE_DIR, "weak_annotation_pool.jsonl")
STATS_OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_reports") # Directory for output reports

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_logging():
    """Sets up logging to console and a file with date/time in filename."""
    ensure_dir(LOGGING_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = os.path.join(LOGGING_DIR, f"stats_log_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized for Annotation Statistics.")
    logger.info(f"Log file: {log_file_name}")

def ensure_dir(path: str):
    """Ensures a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Reads all entries from a .jsonl file."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} at line {line_num}: {line.strip()}")
    return data

def generate_weak_annotation_statistics():
    """
    Generates and prints statistics from the weak_annotation_pool.jsonl file.
    Also saves the report to a file.
    """
    logger.info("\n--- Starting Annotation Statistics Generation ---")

    ensure_dir(STATS_OUTPUT_DIR) # Ensure output directory exists

    weak_annotations = read_jsonl_file(WEAK_ANNOTATION_POOL_FILE)
    total_annotated = len(weak_annotations)
    logger.info(f"Loaded {total_annotated} entries from {WEAK_ANNOTATION_POOL_FILE}.")

    if total_annotated == 0:
        logger.warning("No annotations found in the pool file. Exiting statistics generation.")
        return

    # --- Initialize Counters ---
    relevance_score_counts = Counter()
    paper_type_counts = Counter()
    methodology_type_counts = Counter()
    covered_analysis_steps_counts = Counter()
    code_availability_status_counts = Counter()
    review_type_counts = Counter()
    data_modalities_counts = Counter()
    llm_model_counts = Counter()
    is_method_paper_counts = Counter()
    is_method_comparison_paper_counts = Counter()

    # --- Process Each Annotation Entry ---
    for entry in weak_annotations:
        relevance_score_counts[entry.get('relevance_score_llm', 'N/A')] += 1
        paper_type = entry.get('paper_type', 'other')
        paper_type_counts[paper_type] += 1
        llm_model_counts[entry.get('llm_model', 'N/A')] += 1

        # Process weak_annotations sub-block
        weak_anns = entry.get('weak_annotations', {})

        is_method_paper_counts[str(weak_anns.get('is_method_paper', False))] += 1
        is_method_comparison_paper_counts[str(weak_anns.get('is_method_comparison_paper', False))] += 1

        # Specific details based on paper type or general application
        if paper_type == 'method' or paper_type == 'method_comparison':
            methodology_type_counts[weak_anns.get('methodology_type', 'N/A')] += 1
            for step in weak_anns.get('covered_analysis_steps', []):
                covered_analysis_steps_counts[step] += 1
            code_availability = weak_anns.get('code_availability', {})
            code_availability_status_counts[code_availability.get('status', 'N/A')] += 1

        if paper_type == 'review':
            review_type_counts[weak_anns.get('review_type', 'N/A')] += 1

        # Global counts for data modalities
        for modality in entry.get('data_modalities_used', []):
            data_modalities_counts[modality] += 1


    # --- Generate Report ---
    report_lines = []
    report_lines.append(f"--- Annotation Statistics Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    report_lines.append(f"Total weakly annotated papers: {total_annotated}")
    report_lines.append("\n--- Overall LLM Metrics ---")
    report_lines.append(f"LLM Models Used (count by model):\n{dict(llm_model_counts)}")

    report_lines.append("\n--- Relevance Scores ---")
    report_lines.append("Breakdown of 'relevance_score_llm':")
    for score, count in relevance_score_counts.most_common():
        report_lines.append(f"- {score}: {count} ({count/total_annotated:.2%})")

    report_lines.append("\n--- Paper Types ---")
    report_lines.append("Breakdown of 'paper_type':")
    for p_type, count in paper_type_counts.most_common():
        report_lines.append(f"- {p_type}: {count} ({count/total_annotated:.2%})")

    report_lines.append("\n--- General Weak Annotation Flags ---")
    report_lines.append(f"'is_method_paper' counts: {dict(is_method_paper_counts)}")
    report_lines.append(f"'is_method_comparison_paper' counts: {dict(is_method_comparison_paper_counts)}")

    report_lines.append("\n--- Data Modalities Used (across all papers) ---")
    report_lines.append("Top data modalities mentioned:")
    for modality, count in data_modalities_counts.most_common(15): # Top 15 for brevity
        report_lines.append(f"- {modality}: {count}")

    # --- Detailed Breakdowns by Paper Type (if applicable) ---
    if paper_type_counts.get('method', 0) > 0 or paper_type_counts.get('method_comparison', 0) > 0:
        report_lines.append("\n--- Methodology Details (for 'method' / 'method_comparison' papers) ---")
        if methodology_type_counts:
            report_lines.append(f"Methodology Types:\n{dict(methodology_type_counts)}")
        if covered_analysis_steps_counts:
            report_lines.append("Top Covered Analysis Steps:")
            for step, count in covered_analysis_steps_counts.most_common(15):
                report_lines.append(f"- {step}: {count}")
        if code_availability_status_counts:
            report_lines.append(f"Code Availability Status:\n{dict(code_availability_status_counts)}")

    if paper_type_counts.get('review', 0) > 0:
        report_lines.append("\n--- Review Type Details (for 'review' papers) ---")
        if review_type_counts:
            report_lines.append(f"Review Types:\n{dict(review_type_counts)}")

    # --- Print and Save Report ---
    report_content = "\n".join(report_lines)
    logger.info(report_content)

    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file_name = os.path.join(STATS_OUTPUT_DIR, f"weak_annotation_stats_{timestamp_file}.txt")
    with open(report_file_name, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Statistics report saved to: {report_file_name}")

    logger.info("--- Annotation Statistics Generation Complete ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    setup_logging()
    generate_weak_annotation_statistics()