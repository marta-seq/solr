# run_pipeline.py

import yaml
import os
import subprocess
import logging
from datetime import datetime


# 1. run the scraping
cmd = [
    "python", "-m", "src.data_acquisition.pubmed_scraper",
    "--curated_dois_file", config["paths"]["curated_dois_file"],
    "--output_scraped_articles_file", config["paths"]["raw_scraped_articles"]
    "--pubmed_email", config["scraper"]["pubmed_email"],
    "--keywords",",".join(config["scraper"]["keywords"])
]

# 2. clean and deduplicate the scraped data
cmd += [
    "python", "-m", "src.data_acquisition.clean_scraped_data",
    "--input_file", config["paths"]["raw_scraped_articles"],
    "--output_file", config["paths"]["intermediate_cleaned_papers"]
]

# 3. incorporate manually curated categories
cmd += [
    "python", "-m", "src.data_acquisition.incorporate_curated_categories",
    "--input_file", config["paths"]["intermediate_cleaned_papers"],
    "--curated_categories_file", config["paths"]["curated_categories_file"],
    "--output_file", config["paths"]["intermediate_papers_with_curated_categories"]
]

# 4. extract concepts using keyword matching
cmd += [
    "python", "-m", "src.weak_annotation.concept_extractor",
    "--input_file", config["paths"]["intermediate_papers_with_curated_categories"],
    "--output_file", config["paths"]["intermediate_papers_with_extracted_concepts"],
    "--concepts_config_dir", config["concept_extractor"]["concepts_config_dir"]
]

# 5. filter papers and update annotation score based on weak annotation
cmd += [
    "python", "-m", "src.weak_annotation.filter_and_score_papers",
    "--input_file", config["paths"]["intermediate_papers_with_extracted_concepts"],
    "--output_file", config["paths"]["intermediate_filtered_and_scored_papers"],
    "--annotation_score", str(config["filter_score"]["annotation_score_value"])
]
# 6. extract full-text sections from papers
cmd += [
    "python", "-m", "src.data_acquisition.extract_full_text_sections",
    "--input_papers_file", config["paths"]["intermediate_filtered_and_scored_papers"],
    "--output_sections_log_file", config["paths"]["intermediate_extracted_sections_log"],
    "--unpaywall_email", config["full_text_extraction"]["unpaywall_email"]
]
# 7. merge extracted full-text sections into the main papers dataset

cmd += [
    "python", "-m", "src.data_acquisition.merge_extracted_sections",
    "--input_papers_file", config["paths"]["intermediate_filtered_and_scored_papers"],
    "--input_sections_log_file", config["paths"]["intermediate_extracted_sections_log"],
    "--output_file", config["paths"]["intermediate_papers_with_extracted_sections"]
]

# 8. run the LLM broad classification
cmd += [
    "python", "-m", "src.llm_annotation.llm_broad_classifier",
    "--input_papers_file", config["paths"]["intermediate_papers_with_extracted_sections"],
    "--output_llm_annotated_jsonl", config["paths"]["intermediate_llm_broad_annotated_papers"],
    "--llm_schema_path", config["llm"]["llm_broad_schema_path"],
    "--force_reannotate" if config["llm"]["force_reannotate"] else ""
]

    # run:
    #     with tempfile.NamedTemporaryFile(mode='w',delete=False,suffix='.json') as temp_schema_file:
    #         json.dump(config["llm"]["llm_broad_schema"],temp_schema_file,indent=2)
    #         temp_schema_file_path = temp_schema_file.name
    #
    #     shell_cmd = [
    #         "python", "-m", "src.llm_annotation.llm_broad_classifier",
    #         "--input_papers_file", str(input),
    #         "--output_llm_annotated_jsonl", str(output),
    #         "--llm_schema_path", temp_schema_file_path
    #     ]
    #     if params.force_reannotate_flag:
    #         shell_cmd.append(params.force_reannotate_flag)
    #     shell(" ".join(shell_cmd))
    #     os.remove(temp_schema_file_path)

# 9. score and split the LLM output into computational and non-computational papers
cmd += [
    "python", "-m", "src.data_processing.score_and_split_broad_llm_output",
    "--input_file", config["paths"]["intermediate_llm_broad_annotated_papers"],
    "--output_computational_file", config["paths"]["intermediate_computational_papers"],
    "--output_non_computational_file", config["paths"]["intermediate_non_computational_papers"],
    "--score_increment", str(config["llm"]["broad_llm_post_annotation_score_increment"]),
    "--llm_broad_schema_path", config["llm"]["llm_broad_schema_path"]
]

# 10. run the LLM detailed method extraction for computational papers
cmd += [
    "python", "-m", "src.llm_annotation.llm_detailed_extractor",
    "--input_papers_file", config["paths"]["intermediate_computational_papers"],
    "--output_llm_annotated_jsonl", config["paths"]["intermediate_llm_detailed_annotated_papers"],
    "--llm_schema_path", config["llm"]["llm_detailed_schema_path"],
    "--detailed_llm_score_increment", str(config["llm"]["detailed_llm_score_increment"]),
    "--force_reannotate" if config["llm"]["force_reannotate"] else ""
]

# run:
#         # Create a temporary file for the LLM detailed schema
#         with tempfile.NamedTemporaryFile(mode='w',delete=False,suffix='.json') as temp_schema_file:
#             json.dump(config["llm"]["llm_detailed_schema"],temp_schema_file,indent=2)
#             temp_schema_file_path = temp_schema_file.name
#
#         shell_cmd = [
#             "python", "-m", "src.llm_annotation.llm_detailed_extractor",
#             "--input_papers_file", str(input),
#             "--output_llm_annotated_jsonl", str(output),
#             "--llm_schema_path", temp_schema_file_path,
#             "--detailed_llm_score_increment", str(params.detailed_llm_score_increment)
#         ]
#         if params.force_reannotate_flag:
#             shell_cmd.append(params.force_reannotate_flag)
#         shell(" ".join(shell_cmd))
#         os.remove(temp_schema_file_path)

# 11. prepare papers for manual annotation
# get csv
# src/database/create_csv.py

# fetch citations
# build the graph for methods


# run the ccsv_to_deploy
# this script needs to be changed to protect entries already
# there as they are the manually annotated

# update readme based on the latest pipeline run utils/generate_readme_stats
# chrono scheduling

# TODO
# finish the pipeline part!
# needs to be easier and with less files. THIS!
# add the tests
# make timestamps all the same
# I think papers are not incrementally updated
# improve scheme
# add number of datasets in README
# add the data graph part link to the methods
# GRAPH KNOWLEDGE
# add edges based on similarity and cited by.
# similarity also optionally. use links form similarity, cited by, same data ...
