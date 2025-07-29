# Snakefile for the PhD Review Pipeline - Phase 1: Scraping and Initial Cleaning

import yaml
import os
import json
import tempfile
from snakemake.io import protected

# Define working directory relative to Snakefile location
workdir: "/home/martinha/PycharmProjects/phd/review_test_run"

# --- Configuration ---
configfile: "config/pipeline_config.yaml"

# --- DEBUG: Print the loaded config to verify its contents ---
print("--- DEBUG: Snakemake Loaded Configuration ---")
print(json.dumps(config, indent=2))
print("--- END DEBUG ---")
# --- END DEBUG ---

# Define the final output files for THIS PHASE of the pipeline.
rule all:
    input:
        config["paths"]["intermediate_papers_filtered_scored"],
        config["paths"]["intermediate_extracted_sections_log"],
        config["paths"]["intermediate_llm_broad_annotated_papers"],
        config["paths"]["intermediate_computational_papers"],
        config["paths"]["intermediate_non_computational_papers"],
        config["paths"]["intermediate_llm_detailed_annotated_papers"],
        config["paths"]["manual_papers_to_delete_csv"],
        config["paths"]["final_master_computational_papers"],
        config["paths"]["final_master_non_computational_papers"]

# --- Data Acquisition ---

# Rule to preprocess paper_classification.xlsx into a cleaned CSV
rule preprocess_excel:
    input:
        excel_file=config["paths"]["paper_classification_excel"]
    output:
        protected(config["paths"]["paper_classification_cleaned_csv"])
    log:
        config["paths"]["log_dir"] + "preprocess_excel.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_acquisition.preprocess_curated_excel "
        "--input_excel_file {input.excel_file} "
        "--output_csv_file {output}"

# Rule for scraping PubMed articles
rule scrape_pubmed:
    output:
        protected(config["paths"]["raw_scraped_articles"])
    input:
        paper_classification_cleaned_csv=config["paths"]["paper_classification_cleaned_csv"]
    params:
        email=config["scraper"]["pubmed_email"],
        keywords_str=",".join(config["scraper"]["keywords"])
    log:
        config["paths"]["log_dir"] + "scrape_pubmed.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_acquisition.pubmed_scraper "
        "--paper_classification_cleaned_csv {input.paper_classification_cleaned_csv} "
        "--output_scraped_articles_file {output} "
        "--pubmed_email {params.email} "
        "--keywords '{params.keywords_str}'"


# --- Data Processing and Cleaning ---

# Rule for cleaning and deduplicating scraped data
rule clean_scraped_data:
    output:
        protected(config["paths"]["intermediate_cleaned_papers"])
    input:
        config["paths"]["raw_scraped_articles"]
    params:
        sample_size=config["clean_scraped_data"]["sample_size"]
    log:
        config["paths"]["log_dir"] + "clean_scraped_data.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_acquisition.clean_scraped_data "
        "--input_file {input} "
        "--output_file {output} "
        "--sample_size {params.sample_size}"

# Rule for extracting predefined concepts from articles
rule extract_concepts:
    output:
        protected(config["paths"]["intermediate_papers_with_extracted_concepts"])
    input:
        cleaned_papers=config["paths"]["intermediate_cleaned_papers"]
    params:
        concepts_dir_abs=os.path.join("/home/martinha/PycharmProjects/phd/review", config["concept_extractor"]["concepts_config_dir"])
    log:
        config["paths"]["log_dir"] + "extract_concepts.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.weak_annotation.concept_extractor "
        "--input_file {input.cleaned_papers} "
        "--output_file {output} "
        "--concepts_config_dir {params.concepts_dir_abs}"

# Rule for filtering and scoring papers based on kw_domain_inclusion
rule filter_and_score_papers:
    output:
        protected(config["paths"]["intermediate_papers_filtered_scored"])
    input:
        papers_with_concepts=config["paths"]["intermediate_papers_with_extracted_concepts"]
    params:
        base_annotation_score=config["filter_score"]["base_annotation_score"]
    log:
        config["paths"]["log_dir"] + "filter_and_score_papers.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.filter_score.filter_and_score_papers "
        "--input_file {input.papers_with_concepts} "
        "--output_file {output} "
        "--base_annotation_score {params.base_annotation_score}"

# Rule: Extract full-text sections
rule extract_full_text_sections:
    input:
        papers_to_process=config["paths"]["intermediate_papers_filtered_scored"]
    output:
        protected(config["paths"]["intermediate_extracted_sections_log"])
    params:
        unpaywall_email=config["full_text_extraction"]["unpaywall_email"]
    log:
        config["paths"]["log_dir"] + "extract_full_text_sections.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_acquisition.extract_full_text_sections "
        "--input_papers_file {input.papers_to_process} "
        "--output_sections_log_file {output} "
        "--unpaywall_email {params.unpaywall_email}"

# Rule: LLM Broad Classification
rule llm_broad_classify:
    input:
        papers_with_sections=config["paths"]["intermediate_extracted_sections_log"],
        llm_schema_file=config["llm_broad_classification"]["llm_schema_path"]
    output:
        protected(config["paths"]["intermediate_llm_broad_annotated_papers"])
    params:
        llm_schema_path_abs=os.path.join("/home/martinha/PycharmProjects/phd/review", config["llm_broad_classification"]["llm_schema_path"]),
        force_reannotate_flag="--force_reannotate" if config["llm_broad_classification"]["force_reannotate"] else "",
        ollama_model_name=config["llm_broad_classification"]["ollama_model_name"]
    log:
        config["paths"]["log_dir"] + "llm_broad_classifier.log"
    shell:
        "OLLAMA_MODEL_NAME={params.ollama_model_name} "
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.llm_annotation.llm_broad_classifier "
        "--input_papers_file {input.papers_with_sections} "
        "--output_llm_annotated_jsonl {output} "
        "--llm_schema_path {params.llm_schema_path_abs} "
        "{params.force_reannotate_flag}"

# Rule: Score and Split Broad LLM Output
rule score_and_split_broad_llm_output:
    output:
        computational_output=protected(config["paths"]["intermediate_computational_papers"]),
        non_computational_output=protected(config["paths"]["intermediate_non_computational_papers"])
    input:
        config["paths"]["intermediate_llm_broad_annotated_papers"]
    params:
        broad_llm_target_score=config["llm_broad_classification"]["broad_llm_target_score"],
        llm_broad_schema_path=config["llm_broad_classification"]["llm_schema_path"]
    log:
        config["paths"]["log_dir"] + "score_split_broad_llm_output.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_processing.score_and_split_broad_llm_output "
        f"--input_file {input} "
        f"--output_computational_file {output.computational_output} "
        f"--output_non_computational_file {output.non_computational_output} "
        f"--broad_llm_target_score {params.broad_llm_target_score} "
        f"--llm_broad_schema_path {params.llm_broad_schema_path}"

# Rule: LLM Annotation - SECOND PASS (Detailed Method Extraction for Computational Papers)
rule llm_annotate_detailed_methods:
    output:
        protected(config["paths"]["intermediate_llm_detailed_annotated_papers"])
    input:
        config["paths"]["intermediate_computational_papers"]
    params:
        force_reannotate_flag="--force_reannotate" if config["llm_detailed_annotation"]["force_reannotate"] else "",
        detailed_llm_target_score=config["llm_detailed_annotation"]["detailed_llm_target_score"],
        ollama_model_name=config["llm_detailed_annotation"]["ollama_model_name"],
        llm_detailed_schema_path=config["llm_detailed_annotation"]["llm_schema_path"]
    log:
        config["paths"]["log_dir"] + "llm_detailed_extractor.log"
    shell:
        "OLLAMA_MODEL_NAME={params.ollama_model_name} "
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.llm_annotation.llm_detailed_extractor "
        f"--input_papers_file {input} "
        f"--output_llm_annotated_jsonl {output} "
        f"--llm_schema_path {params.llm_detailed_schema_path} "
        f"--detailed_llm_target_score {params.detailed_llm_target_score} "
        f"{params.force_reannotate_flag}"

# Rule: Prepare for Manual Review (Generates JSONLs)
rule prepare_for_manual_review:
    output:
        comp_jsonl=protected(config["paths"]["manual_computational_for_review_jsonl"]),
        non_comp_jsonl=protected(config["paths"]["manual_non_computational_for_review_jsonl"])
    input:
        input_comp_jsonl=config["paths"]["intermediate_llm_detailed_annotated_papers"],
        input_non_comp_jsonl=config["paths"]["intermediate_non_computational_papers"]
    params:
        broad_llm_target_score=config["llm_broad_classification"]["broad_llm_target_score"],
        detailed_llm_target_score=config["llm_detailed_annotation"]["detailed_llm_target_score"]
    log:
        config["paths"]["log_dir"] + "prepare_for_manual_review.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_processing.prepare_for_manual_review "
        "--input_computational_jsonl {input.input_comp_jsonl} "
        "--input_non_computational_jsonl {input.input_non_comp_jsonl} "
        "--output_computational_jsonl {output.comp_jsonl} "
        "--output_non_computational_jsonl {output.non_comp_jsonl} "
        "--broad_llm_target_score {params.broad_llm_target_score} "
        "--detailed_llm_target_score {params.detailed_llm_target_score}"

# NEW RULE: Create empty papers_to_delete.csv
rule create_empty_papers_to_delete_csv:
    output:
        protected(config["paths"]["manual_papers_to_delete_csv"])
    log:
        config["paths"]["log_dir"] + "create_empty_papers_to_delete_csv.log"
    shell:
        "python -c \"import pandas as pd; import os; os.makedirs(os.path.dirname('{output}'), exist_ok=True); pd.DataFrame(columns=['doi', 'notes', 'reason_for_deletion']).to_csv('{output}', index=False)\""

# NEW RULE: Integrate Manual Reviews (Simplified)
rule integrate_manual_reviews:
    output:
        final_comp_jsonl=protected(config["paths"]["final_master_computational_papers"]),
        final_non_comp_jsonl=protected(config["paths"]["final_master_non_computational_papers"])
    input:
        manual_comp_jsonl=config["paths"]["manual_computational_for_review_jsonl"],
        manual_non_comp_jsonl=config["paths"]["manual_non_computational_for_review_jsonl"],
        papers_to_delete_csv=config["paths"]["manual_papers_to_delete_csv"]
    params:
        final_manual_review_score=config["manual_review"]["final_manual_review_score"]
    log:
        config["paths"]["log_dir"] + "integrate_manual_reviews.log"
    shell:
        "PYTHONPATH=/home/martinha/PycharmProjects/phd/review python -m src.data_processing.integrate_manual_reviews "
        f"--manual_computational_jsonl {{input.manual_comp_jsonl}} "
        f"--manual_non_computational_jsonl {{input.manual_non_comp_jsonl}} "
        f"--papers_to_delete_csv {{input.papers_to_delete_csv}} "
        f"--final_master_computational_jsonl {{output.final_comp_jsonl}} "
        f"--final_master_non_computational_jsonl {{output.final_non_comp_jsonl}} "
        f"--final_manual_review_score {{params.final_manual_review_score}}"

# NO TESTING RULES ARE INCLUDED IN THIS SNAKEFILE.
# ALL TESTS MUST BE RUN MANUALLY VIA `python -m unittest tests/your_test_file.py`