import pandas as pd
import requests
import json
import time
import os
import re
import tempfile
import logging
import random
from datetime import datetime
from bs4 import BeautifulSoup
import fitz  # PyMuPDF, for PDF processing
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Your existing methods JSONL file that will be enhanced with sections
ORIGINAL_METHODS_FILE = "../../data/methods_papers2.jsonl"
# New file to store the log of all extraction attempts and the extracted sections
EXTRACTED_SECTIONS_LOG_FILE = "../../data/extracted_sections_log.jsonl"
LOG_DIR = "../../data/extraction_logs"
OUTPUT_METHODS_WITH_SECTIONS_FILE = "../../data/methods_papers_with_sections.jsonl" # This is the file that will receive the merged data

UNPAYWALL_EMAIL = "id9417@alunos.uminho.pt"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/112.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.65 Mobile Safari/537.36",
]
# --- Logging Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/section_extraction_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Section Keys ---
SECTION_KEYS = [
    "introduction",
    "results_discussion",
    "conclusion",
    "methods",
    "code_availability",
    "data_availability"
]

# --- Helper Functions ---

def load_jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} at line {line_num}: {line.strip()}")
    else:
        logger.info(f"File not found: {file_path}. Returning empty DataFrame.")
    return pd.DataFrame(data)

def save_jsonl_df(df, filepath, name="file", append=False):
    """Saves a DataFrame to a JSONL file, with an option to append."""
    if df.empty:
        logger.info(f"DataFrame for '{name}' is empty. Not saving to {filepath}.")
        if not append and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Removed empty '{name}' file: {filepath}")
            except Exception as e:
                logger.warning(f"Could not remove empty '{name}' file {filepath}: {e}")
        return

    mode = 'a' if append else 'w'
    try:
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        df.to_json(filepath, orient='records', lines=True, force_ascii=False, mode=mode)
        logger.info(f"Saved {len(df)} entries {'appended to' if append else 'to'} '{name}': {filepath}")
    except Exception as e:
        logger.critical(f"Failed to save '{name}' to {filepath}: {e}")


def clean_doi(doi_series):
    # Ensure it's a string series before applying string operations
    return doi_series.astype(str).str.replace('https://doi.org/', '', regex=False).str.lower().str.strip()

def clean_dataframe_dois(df: pd.DataFrame, doi_column: str = 'doi') -> pd.DataFrame:
    """Cleans and deduplicates DOIs in a DataFrame, ensuring consistency for merging."""
    if doi_column in df.columns:
        df[doi_column] = clean_doi(df[doi_column])
    return df

def extract_section(text, header_keywords, window=7000):
    stop_keywords = [
        "introduction", "background", "results", "discussion", "conclusion",
        "references", "acknowledgements", "supplementary", "funding", "appendix"
    ]
    stop_regex_parts = [re.escape(k) for k in stop_keywords]
    # Make sure to handle potential headers like "1. Introduction" or "Introduction."
    stop_regex = r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?(?:" + "|".join(stop_regex_parts) + r")\.?\s*(?:\n|$|\Z)"

    for header in header_keywords:
        # Improved regex to handle common header formats (e.g., "1. Introduction", "Introduction.")
        header_regex = rf"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?\b{re.escape(header)}\b\.?\s*(?:\n|$|\Z)"
        match = re.search(header_regex, text)
        if match:
            start_idx = match.end()
            sliced = text[start_idx:start_idx + window]
            stop_match = re.search(stop_regex, sliced)
            end_idx = start_idx + (
                stop_match.start() if stop_match else len(sliced))
            section_text = text[start_idx:end_idx].strip()

            section_text = re.sub(r'\n+', '\n', section_text).strip()

            if len(section_text.split()) > 30:
                return section_text

    return None

def extract_sections_from_text(text: str):
    sections = dict.fromkeys(SECTION_KEYS, None)

    headers_map = {
        'introduction': ["introduction", "background"],
        'results_discussion': ["results and discussion", "results", "discussion", "discussion and future work"],
        'conclusion': ["conclusion", "summary", "final remarks"],
        'methods': ["methods", "materials and methods", "experimental procedures", "study design"],
        'data_availability': ["data availability", "availability of data", "data and materials availability",
                              "data/code availability"],
        'code_availability': ["code availability", "availability of code", "software availability",
                              "data/code availability"]
    }

    for key, headers in headers_map.items():
        sections[key] = extract_section(text, headers)

    # Refined regex for availability sections to prevent matching within sentences too broadly
    # Look for keywords followed by very limited punctuation or spaces, then accessibility terms.
    if not sections["code_availability"]:
        mentions = re.findall(
            r"(?:code|software|repository)\W{0,5}\b(?:available|access|download|https?://(?:www\.)?github\.com|zenodo|bioconductor|figshare|bitbucket|gitlab|source code)\b",
            text, re.I)
        if mentions:
            unique_mentions = list(set(m.strip() for m in mentions if m.strip()))
            if unique_mentions:
                sections["code_availability"] = "Mentioned code availability sources: " + ", ".join(unique_mentions)

    if not sections["data_availability"]:
        mentions = re.findall(
            r"(?:data|dataset|database|repository)\W{0,5}\b(?:available|access|download|zenodo|10x genomics|figshare|dryad|geo|gse[0-9]+|ega|arrayexpress|genbank|sra|pdb)\b",
            text, re.I)
        if mentions:
            unique_mentions = list(set(m.strip() for m in mentions if m.strip()))
            if unique_mentions:
                sections["data_availability"] = "Mentioned data availability sources: " + ", ".join(unique_mentions)

    return sections


def extract_sections_from_pdf_file(pdf_path: str):
    sections = dict.fromkeys(SECTION_KEYS, None)
    try:
        doc = fitz.open(pdf_path)
        has_readable_text = False
        # Check first few pages for readability
        for page_num in range(min(5, len(doc))):
            if doc.load_page(page_num).get_text().strip():
                has_readable_text = True
                break

        if not has_readable_text:
            logger.warning(
                f"PDF content check: PDF contains no readable text (might be scanned or image-based) for {pdf_path}")
            return sections

        text = "\n".join(page.get_text() for page in doc)
        doc.close()

        if not text.strip():
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return sections

        return extract_sections_from_text(text)

    except fitz.FileDataError as e:
        logger.error(f"Error reading PDF {pdf_path} (File Data Error): {e}")
        return sections
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}", exc_info=True)
        return sections

def extract_sections_from_html_content(html_text: str):
    sections = dict.fromkeys(SECTION_KEYS, None)
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        if not text.strip():
            logger.warning("No text extracted from HTML content.")
            return sections

        return extract_sections_from_text(text)

    except Exception as e:
        logger.error(f"HTML parsing error: {e}", exc_info=True)
        return sections


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Section Extraction and Logging Process ---") # Updated message

    os.makedirs(LOG_DIR, exist_ok=True)
    session = requests.Session()

    try:
        df_methods = load_jsonl_to_dataframe(ORIGINAL_METHODS_FILE)
        df_methods['doi'] = clean_doi(df_methods['doi'])
        logger.info(f"Loaded {len(df_methods)} papers from {ORIGINAL_METHODS_FILE}.")

        if df_methods.empty:
            logger.error(f"Input file {ORIGINAL_METHODS_FILE} is empty. No papers to process. Exiting.")
            exit()

    except FileNotFoundError as e:
        logger.critical(e)
        exit()
    except Exception as e:
        logger.critical(f"An unexpected error occurred during initial file loading: {e}", exc_info=True)
        exit()

    # Load existing extraction log to determine which DOIs need processing
    df_extracted_log = load_jsonl_to_dataframe(EXTRACTED_SECTIONS_LOG_FILE)

    if not df_extracted_log.empty and 'doi' in df_extracted_log.columns:
        df_extracted_log['doi'] = clean_doi(df_extracted_log['doi'])
    else:
        df_extracted_log = pd.DataFrame(columns=['doi', 'extracted_at_timestamp', 'section_extraction_status',
                                                 'is_open_access', 'full_text_access_type'] +
                                                [f'extracted_{k}_section' for k in SECTION_KEYS])
        logger.info(f"Initialized empty DataFrame for {EXTRACTED_SECTIONS_LOG_FILE}.")


    processed_successful_dois = set()

    if not df_extracted_log.empty:
        df_extracted_log['extracted_at_timestamp'] = pd.to_datetime(df_extracted_log.get('extracted_at_timestamp'))

        status_rank_map = {
            'extracted from pdf (sections found)': 4,
            'extracted from html (sections found)': 3,
            'pdf processed but no relevant sections found': 2,
            'html processed (no sections found by heuristics)': 1,
            'oa_html_landing_page': 0.5,
            'oa_pdf_link': 0.5,
            'oa_pdf_fallback': 0.5,
            'oa but no direct location info from unpaywall': 0,
            'closed access or not found in unpaywall': 0,
            'unpaywall/network error': -1,
            'html fetch error': -1,
            'pdf download error': -1,
            'general error': -1,
            'initial: not processed': -2,
            'serialization error': -3
        }
        df_extracted_log['status_rank'] = df_extracted_log['section_extraction_status'].str.lower().map(status_rank_map).fillna(-4)

        idx = df_extracted_log.groupby('doi')['status_rank'].idxmax()
        df_best_attempts = df_extracted_log.loc[idx]

        idx_latest = df_best_attempts.groupby('doi')['extracted_at_timestamp'].idxmax()
        df_best_attempts = df_best_attempts.loc[idx_latest]

        # Determine which DOIs already have actual section content extracted successfully
        for _, row in df_best_attempts.iterrows():
            doi_val = row['doi']
            status = row.get('section_extraction_status', '').lower()
            if any(success_word in status for success_word in ['extracted from html', 'extracted from pdf']):
                if any(pd.notna(row.get(f'extracted_{key}_section')) and str(row.get(f'extracted_{key}_section', '')).strip() != '' for key in SECTION_KEYS if f'extracted_{key}_section' in row):
                    processed_successful_dois.add(doi_val)


    df_dois_to_process = df_methods[~df_methods['doi'].isin(processed_successful_dois)].copy()

    if df_dois_to_process.empty:
        logger.info("All papers in the methods file already have successfully extracted sections in the log file. No new extractions needed.")
    else:
        logger.info(
            f"Resuming processing from {len(processed_successful_dois)} successfully processed papers in log. "
            f"Remaining {len(df_dois_to_process)} DOIs to process for new section extraction attempts.")

        logger.info("\n--- Processing DOIs for Open Access and Section Extraction ---")

        for _, row in tqdm(df_dois_to_process.iterrows(), total=len(df_dois_to_process), desc="Extracting Sections"):
            doi = row['doi']

            current_paper_extraction_result = {
                'doi': doi,
                'extracted_at_timestamp': datetime.now().isoformat(),
                'is_open_access': False,
                'full_text_access_type': 'Unknown',
                'section_extraction_status': 'Initial: Not Processed',
            }
            for key in SECTION_KEYS:
                current_paper_extraction_result[f'extracted_{key}_section'] = None

            try:
                request_headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }

                unpaywall_api_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
                logger.debug(f"Querying Unpaywall for DOI: {doi}")

                time.sleep(random.uniform(5.0, 15.0))
                unpaywall_response = session.get(unpaywall_api_url, timeout=20)
                unpaywall_response.raise_for_status()
                unpaywall_data = unpaywall_response.json()

                if unpaywall_data and unpaywall_data.get('is_oa'):
                    current_paper_extraction_result['is_open_access'] = True
                    best_oa_location = unpaywall_data.get('best_oa_location')

                    if best_oa_location:
                        oa_url_for_pdf = best_oa_location.get('url_for_pdf')
                        oa_url_for_landing_page = best_oa_location.get('url_for_landing_page')

                        html_attempted = False
                        html_extraction_successful_and_found_sections = False
                        html_fetch_failed = False

                        if oa_url_for_landing_page and not (oa_url_for_pdf and "springer.com" in oa_url_for_pdf):
                            html_attempted = True
                            current_paper_extraction_result['full_text_access_type'] = 'OA_HTML_Landing_Page'
                            logger.debug(f"Attempting to fetch and process HTML for DOI: {doi} from {oa_url_for_landing_page}")

                            try:
                                time.sleep(random.uniform(5.0, 15.0))
                                html_response = session.get(oa_url_for_landing_page, timeout=30,
                                                            headers=request_headers)
                                html_response.raise_for_status()
                                if 'text/html' in html_response.headers.get('Content-Type', ''):
                                    sections = extract_sections_from_html_content(html_response.text)
                                    for key in SECTION_KEYS:
                                        current_paper_extraction_result[f'extracted_{key}_section'] = sections.get(key)

                                    if any(sections.get(key) for key in SECTION_KEYS):
                                        html_extraction_successful_and_found_sections = True
                                        status_msg = 'Extracted from HTML (sections found)'
                                    else:
                                        status_msg = 'HTML processed (no sections found by heuristics)'
                                    current_paper_extraction_result['section_extraction_status'] = status_msg
                                    logger.debug(f"HTML processing status for {doi}: {status_msg}")
                                else:
                                    html_fetch_failed = True
                                    status_msg = f"Fetched HTML but content type is not HTML ({html_response.headers.get('Content-Type', 'N/A')}) for DOI: {doi}"
                                    current_paper_extraction_result['section_extraction_status'] = status_msg
                                    logger.warning(status_msg)
                            except requests.exceptions.RequestException as e:
                                html_fetch_failed = True
                                status_msg = f"HTML fetch error for {doi}: {e}. URL: {oa_url_for_landing_page}"
                                current_paper_extraction_result['section_extraction_status'] = status_msg
                                logger.error(status_msg)
                            except Exception as e:
                                html_fetch_failed = True
                                status_msg = f"Unexpected error during HTML parsing for {doi}: {e}. URL: {oa_url_for_landing_page}"
                                current_paper_extraction_result['section_extraction_status'] = status_msg
                                logger.exception(status_msg)

                        if oa_url_for_pdf and (
                                not html_attempted or html_fetch_failed or not html_extraction_successful_and_found_sections):
                            if html_attempted and not html_extraction_successful_and_found_sections:
                                current_paper_extraction_result['full_text_access_type'] = 'OA_PDF_Fallback'
                            elif not html_attempted:
                                current_paper_extraction_result['full_text_access_type'] = 'OA_PDF_Link'

                            logger.debug(f"Attempting to download and process PDF for DOI: {doi} from {oa_url_for_pdf}")
                            tmp_file_path = None
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                    tmp_file_path = tmp_file.name
                                    pdf_headers = request_headers.copy()
                                    if oa_url_for_landing_page:
                                        pdf_headers['Referer'] = oa_url_for_landing_page

                                    time.sleep(random.uniform(5.0, 15.0))
                                    pdf_response = session.get(oa_url_for_pdf, timeout=45, headers=pdf_headers)
                                    pdf_response.raise_for_status()

                                    if 'application/pdf' in pdf_response.headers.get('Content-Type', ''):
                                        tmp_file.write(pdf_response.content)
                                        tmp_file.flush()
                                        logger.debug(f"Downloaded PDF to temporary file '{tmp_file_path}' for {doi}.")

                                        sections = extract_sections_from_pdf_file(tmp_file_path)
                                        for key in SECTION_KEYS:
                                            current_paper_extraction_result[f'extracted_{key}_section'] = sections.get(key)

                                        if any(sections.get(key) for key in SECTION_KEYS):
                                            current_paper_extraction_result[
                                                'section_extraction_status'] = 'Extracted from PDF (sections found)'
                                        else:
                                            current_paper_extraction_result[
                                                'section_extraction_status'] = 'PDF processed but no relevant sections found'
                                        logger.debug(
                                            f"PDF processing status for {doi}: {current_paper_extraction_result['section_extraction_status']}")
                                    else:
                                        status_msg = f"Fetched PDF URL but content type is not PDF ({pdf_response.headers.get('Content-Type', 'N/A')}) for DOI: {doi}"
                                        current_paper_extraction_result['section_extraction_status'] = status_msg
                                        logger.warning(status_msg)

                            except requests.exceptions.RequestException as e:
                                status_msg = f"PDF download error for {doi}: {e}. URL: {oa_url_for_pdf}"
                                current_paper_extraction_result['section_extraction_status'] = status_msg
                                logger.error(status_msg)
                            except Exception as e:
                                status_msg = f"Unexpected error during PDF processing for {doi}: {e}. URL: {oa_url_for_pdf}"
                                current_paper_extraction_result['section_extraction_status'] = status_msg
                                logger.exception(status_msg)
                            finally:
                                if tmp_file_path and os.path.exists(tmp_file_path):
                                    os.remove(tmp_file_path)
                                    logger.debug(f"Deleted temporary PDF: {tmp_file_path}")
                    else:
                        current_paper_extraction_result['full_text_access_type'] = 'OA_No_Direct_Link'
                        current_paper_extraction_result[
                            'section_extraction_status'] = 'OA but no direct location info from Unpaywall'
                        logger.debug(f"Paper is Open Access but no direct location info from Unpaywall for DOI: {doi}")
                else:
                    current_paper_extraction_result['is_open_access'] = False
                    current_paper_extraction_result['full_text_access_type'] = 'Closed_Access'
                    current_paper_extraction_result['section_extraction_status'] = 'Closed Access or not found in Unpaywall'
                    logger.debug(f"Paper is Closed Access or not found in Unpaywall for DOI: {doi}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Unpaywall API or network error for DOI {doi}: {e}")
                current_paper_extraction_result['section_extraction_status'] = f"Unpaywall/Network Error: {e}"
            except Exception as e:
                logger.error(f"General error processing DOI {doi}: {e}")
                current_paper_extraction_result['section_extraction_status'] = f"General Error: {e}"

            # Append this attempt's result to the log file immediately after processing each paper
            save_jsonl_df(pd.DataFrame([current_paper_extraction_result]), EXTRACTED_SECTIONS_LOG_FILE, "extraction log", append=True)

    # --- NO FINAL MERGE STEP AND SAVE TO A NEW FILE IF YOU ONLY WANT TWO FILES ---
    # The script now simply finishes after updating the EXTRACTED_SECTIONS_LOG_FILE.
    # The lines below this comment block are what you would remove if you want only two files.

    logger.info("--- Section extraction attempts finished. ---") # Updated message

    # You would COMMENT OUT or DELETE EVERYTHING FROM HERE DOWN IF YOU DO NOT WANT A THIRD FILE:
    # df_extracted_log_final = load_jsonl_to_dataframe(EXTRACTED_SECTIONS_LOG_FILE)
    #
    # if df_extracted_log_final.empty:
    #     logger.warning(f"No extraction log data found in {EXTRACTED_SECTIONS_LOG_FILE}. Cannot enhance methods file with extracted sections.")
    #     df_final_output = df_methods.copy()
    #     df_final_output['sections_extracted_successfully'] = False
    #     for key in SECTION_KEYS:
    #         df_final_output[f'extracted_{key}_section'] = None
    #     df_final_output['is_open_access'] = False
    #     df_final_output['full_text_access_type'] = 'Unknown'
    #     df_final_output['section_extraction_status'] = 'No extraction log entries'
    #     save_jsonl_df(df_final_output, OUTPUT_METHODS_WITH_SECTIONS_FILE, "output methods file (final)")
    #     logger.info("--- Script finished ---")
    #     exit()
    #
    # if 'doi' not in df_extracted_log_final.columns:
    #     logger.error(f"DOI column missing in {EXTRACTED_SECTIONS_LOG_FILE}. Cannot proceed with merging.")
    #     exit()
    # df_extracted_log_final['doi'] = clean_doi(df_extracted_log_final['doi'])
    #
    # df_extracted_log_final['extracted_at_timestamp'] = pd.to_datetime(df_extracted_log_final['extracted_at_timestamp'])
    # status_rank_map = {
    #     'extracted from pdf (sections found)': 4,
    #     'extracted from html (sections found)': 3,
    #     'pdf processed but no relevant sections found': 2,
    #     'html processed (no sections found by heuristics)': 1,
    #     'oa_html_landing_page': 0.5,
    #     'oa_pdf_link': 0.5,
    #     'oa_pdf_fallback': 0.5,
    #     'oa but no direct location info from unpaywall': 0,
    #     'closed access or not found in unpaywall': 0,
    #     'unpaywall/network error': -1,
    #     'html fetch error': -1,
    #     'pdf download error': -1,
    #     'general error': -1,
    #     'initial: not processed': -2,
    #     'serialization error': -3
    # }
    # df_extracted_log_final['status_rank'] = df_extracted_log_final['section_extraction_status'].str.lower().map(status_rank_map).fillna(-4)
    #
    # idx_best_status = df_extracted_log_final.groupby('doi')['status_rank'].idxmax()
    # df_best_attempts_for_merge = df_extracted_log_final.loc[idx_best_status]
    #
    # idx_latest_timestamp = df_best_attempts_for_merge.groupby('doi')['extracted_at_timestamp'].idxmax()
    # df_best_attempts_for_merge = df_best_attempts_for_merge.loc[idx_latest_timestamp]
    #
    # merge_cols = ['doi', 'is_open_access', 'full_text_access_type', 'section_extraction_status']
    # for key in SECTION_KEYS:
    #     merge_cols.append(f'extracted_{key}_section')
    #
    # df_sections_to_merge = df_best_attempts_for_merge[merge_cols].copy()
    #
    # def check_sections_content(row):
    #     return any(pd.notna(row.get(f'extracted_{key}_section')) and str(row.get(f'extracted_{key}_section', '')).strip() != '' for key in SECTION_KEYS)
    #
    # df_sections_to_merge['sections_extracted_successfully'] = df_sections_to_merge.apply(check_sections_content, axis=1)
    #
    # df_final_output = pd.merge(df_methods, df_sections_to_merge, on='doi', how='left', suffixes=('', '_new_extraction'))
    #
    # for col in merge_cols:
    #     if col != 'doi' and f'{col}_new_extraction' in df_final_output.columns:
    #         df_final_output[col] = df_final_output[f'{col}_new_extraction'].fillna(df_final_output.get(col, None))
    #         df_final_output.drop(columns=[f'{col}_new_extraction'], inplace=True)
    #     elif col != 'doi' and col not in df_final_output.columns:
    #         df_final_output[col] = None
    #
    # if 'sections_extracted_successfully' not in df_final_output.columns:
    #     df_final_output['sections_extracted_successfully'] = False
    # df_final_output['sections_extracted_successfully'] = df_final_output['sections_extracted_successfully'].fillna(False)
    #
    # save_jsonl_df(df_final_output, OUTPUT_METHODS_WITH_SECTIONS_FILE, "output methods file (final)") # THIS LINE IS GONE
    #
    # logger.info("--- Script finished ---")

    logger.info("--- Script finished. `methods_papers.jsonl` was not modified. Sections are in `extracted_sections_log.jsonl`. ---") # Final message