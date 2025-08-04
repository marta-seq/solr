# scripts/generate_readme_stats.py
import pandas as pd
import json
import os
from datetime import datetime
import yaml  # For loading pipeline_config.yaml

# Assuming the script is run from the project root by GitHub Actions
PROJECT_ROOT = os.getcwd()

# Load config to get paths
# CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "config", "pipeline_config.yaml")
# try:
#     with open(CONFIG_FILE_PATH, 'r') as f:
#         config = yaml.safe_load(f)
# except FileNotFoundError:
#     print(f"Error: Configuration file not found at: {CONFIG_FILE_PATH}")
#     exit(1)
# except yaml.YAMLError as e:
#     print(f"Error parsing configuration file: {e}")
#     exit(1)

# Paths to your data files (using paths from config)
# FIX: New primary data source for stats
# FINAL_DEPLOY_CSV = os.path.join(PROJECT_ROOT, config["paths"]["processed_final_deploy_csv"])
FINAL_DEPLOY_CSV = os.path.join(PROJECT_ROOT, "data/database/processed_final_deploy.csv")


def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except json.JSONDecodeError as e:
        print(f"Warning: Error decoding JSON from {file_path}: {e}. Returning empty list.")
        return []
    except Exception as e:
        print(f"Warning: Error reading JSONL file {file_path}: {e}. Returning empty list.")
        return []


def get_file_modification_date(file_path):
    """Gets the last modification date of a file in YYYY-MM-DD HH:MM:SS format."""
    if os.path.exists(file_path):
        timestamp = os.path.getmtime(file_path)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"


def get_latest_date_from_column(df: pd.DataFrame, column_name: str) -> str:
    """
    Parses a DataFrame column to datetime objects, handles errors,
    and returns the latest date in YYYY-MM-DD HH:MM:SS format.
    """
    if column_name not in df.columns:
        return "N/A"

    # Convert to datetime, coercing errors to NaT (Not a Time)
    dates = pd.to_datetime(df[column_name], errors='coerce')

    # Drop NaT values and find the maximum date
    latest_date = dates.dropna().max()

    if pd.isna(latest_date):
        return "N/A"
    return latest_date.strftime('%Y-%m-%d %H:%M:%S')


def generate_stats_table():
    """Generates the Markdown table string with pipeline statistics."""
    # FIX: Load data from processed_final_deploy.csv
    if not os.path.exists(FINAL_DEPLOY_CSV):
        print(f"Warning: Final deploy CSV not found: {FINAL_DEPLOY_CSV}. Cannot generate full stats.")
        # Return a table with N/A if the main data file is missing
        return "\n".join([
            "| Metric | Value |",
            "|---|---|",
            "| Total Papers | N/A |",
            "| Computational Papers | N/A |",
            "| Non-Computational Papers | N/A |",
            "| Last Scrape Date | N/A |",
            "| Last Annotation Date | N/A |",
            "| Last Manual Curation Date | N/A |"
        ])

    df = pd.read_csv(FINAL_DEPLOY_CSV, dtype=str, keep_default_na=False)  # Load as string to avoid type issues

    # FIX: Deduplicate based on 'doi' column
    if 'doi' in df.columns:
        initial_rows = len(df)
        df.drop_duplicates(subset=['doi'], inplace=True)
        if len(df) < initial_rows:
            print(f"Removed {initial_rows - len(df)} duplicate DOIs from {FINAL_DEPLOY_CSV}.")
    else:
        print(f"Warning: 'doi' column not found in {FINAL_DEPLOY_CSV}. Cannot deduplicate.")

    # FIX: Calculate total, computational, and non-computational papers
    total_papers = len(df)

    computational_papers = 0
    if 'is_computational' in df.columns:
        # Convert to boolean, handling various string representations of True/False
        df['is_computational'] = df['is_computational'].astype(str).str.lower().isin(['true', '1', 't', 'yes'])
        computational_papers = df['is_computational'].sum()
    else:
        print(f"Warning: 'is_computational' column not found in {FINAL_DEPLOY_CSV}.")

    non_computational_papers = total_papers - computational_papers

    # FIX: Get last update dates from columns within the DataFrame
    last_scrape_date = get_latest_date_from_column(df, 'scrape_date')
    last_annotation_date = get_latest_date_from_column(df, 'llm_annotation_timestamp')
    last_manual_curation_date = get_latest_date_from_column(df, 'MC_Date')

    stats = {
        "Total Papers": total_papers,
        "Computational Papers": computational_papers,
        "Non-Computational Papers": non_computational_papers,
        "Last Scrape Date": last_scrape_date,
        "Last Annotation Date": last_annotation_date,
        "Last Manual Curation Date": last_manual_curation_date  # New metric
    }

    table_rows = [
        "| Metric | Value |",
        "|---|---|"
    ]
    for metric, value in stats.items():
        table_rows.append(f"| {metric} | {value} |")

    return "\n".join(table_rows)


if __name__ == "__main__":
    stats_table_markdown = generate_stats_table()

    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    if not os.path.exists(readme_path):
        print("Error: README.md not found. Cannot update stats.")
        exit(1)

    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()

    # Define markers for the stats table section
    start_marker = "<!-- STATS_TABLE_START -->"
    end_marker = "<!-- STATS_TABLE_END -->"

    if start_marker in readme_content and end_marker in readme_content:
        # Replace the content between markers
        before_table = readme_content.split(start_marker)[0]
        after_table = readme_content.split(end_marker)[1]
        new_readme_content = f"{before_table}{start_marker}\n{stats_table_markdown}\n{end_marker}{after_table}"

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_readme_content)
        print("README.md updated with pipeline statistics.")
    else:
        print(f"Warning: Markers '{start_marker}' or '{end_marker}' not found in README.md. Stats table not updated.")
