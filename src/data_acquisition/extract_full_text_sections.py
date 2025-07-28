# src/data_acquisition/extract_full_text_sections.py
import argparse
import json
import logging
import os
import time
import random
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional, List
import re
# Add project root to path for utility imports
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_jsonl_to_dataframe, save_jsonl_records, clean_doi

logger = logging.getLogger(__name__)

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extracts text from PDF content using PyMuPDF."""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

# --- Section Parsing ---
def parse_sections(text: str) -> Dict[str, str]:
    """
    Parses a given text to extract common scientific paper sections.
    Returns a dictionary with section names as keys and their content as values.
    """
    sections = {}
    # Common section headers (case-insensitive, with optional leading numbers/letters)
    # Using regex to capture content until the next header or end of document
    section_patterns = {
        'introduction': r'(?:^|\n)(?:(?:1\s*|I\s*\.\s*|INTRODUCTION|Introduction)\s*\n)(.*?)(?=\n(?:(?:2\s*|II\s*\.\s*|METHODS|MATERIALS AND METHODS|RESULTS|DISCUSSION|CONCLUSION|ACKNOWLEDGEMENTS|REFERENCES|SUPPLEMENTARY|APPENDIX)|$))',
        'methods': r'(?:^|\n)(?:(?:2\s*|II\s*\.\s*|METHODS|MATERIALS AND METHODS)\s*\n)(.*?)(?=\n(?:(?:3\s*|III\s*\.\s*|RESULTS|DISCUSSION|CONCLUSION|ACKNOWLEDGEMENTS|REFERENCES|SUPPLEMENTARY|APPENDIX)|$))',
        'results_discussion': r'(?:^|\n)(?:(?:3\s*|III\s*\.\s*|RESULTS|DISCUSSION|RESULTS AND DISCUSSION)\s*\n)(.*?)(?=\n(?:(?:4\s*|IV\s*\.\s*|CONCLUSION|ACKNOWLEDGEMENTS|REFERENCES|SUPPLEMENTARY|APPENDIX)|$))',
        'conclusion': r'(?:^|\n)(?:(?:4\s*|IV\s*\.\s*|CONCLUSION|CONCLUSIONS)\s*\n)(.*?)(?=\n(?:(?:5\s*|V\s*\.\s*|ACKNOWLEDGEMENTS|REFERENCES|SUPPLEMENTARY|APPENDIX)|$))',
        'data_availability': r'(?:^|\n)(?:DATA AVAILABILITY|Data Availability|DATA AVAILABILITY STATEMENT|Data Availability Statement)\s*\n(.*?)(?=\n(?:ACKNOWLEDGEMENTS|REFERENCES|SUPPLEMENTARY|APPENDIX|$))',
        'code_availability': r'(?:^|\n)(?:CODE AVAILABILITY|Code Availability|CODE AVAILABILITY STATEMENT|Code Availability Statement)\s*\n(.*?)(?=\n(?:ACKNOWLEDGEMENTS|REFERENCES|SUPPLEMENTARY|APPENDIX|$))',
    }

    # Clean up common PDF artifacts (e.g., headers/footers, multiple spaces)
    cleaned_text = re.sub(r'\s{2,}', ' ', text) # Replace multiple spaces with single space
    cleaned_text = re.sub(r'(\n\s*\n)+', '\n\n', cleaned_text) # Replace multiple newlines with double newline
    cleaned_text = cleaned_text.strip()

    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Basic word count filter to avoid capturing just a header or very short unrelated text
            if len(content.split()) > 30: # Only include if content is substantial
                sections[section_name] = content
                logger.debug(f"Found section: {section_name}")
            else:
                logger.debug(f"Skipped short section: {section_name}")
        else:
            logger.debug(f"Section not found: {section_name}")
    return sections

# --- Main Extraction Logic ---
def extract_full_text_sections(input_papers_file: str, output_sections_log_file: str, unpaywall_email: str):
    """
    Extracts full-text sections for papers with DOIs using Unpaywall and PDF parsing.
    Updates existing records or adds new ones to the output log file.
    """
    setup_logging(log_prefix="extract_full_text_sections")
    logger.info("--- Starting Full-Text Section Extraction Process ---")
    logger.info(f"Input papers file: {input_papers_file}")
    logger.info(f"Output sections log file: {output_sections_log_file}")
    logger.info(f"Unpaywall email: {unpaywall_email}")

    df_papers = load_jsonl_to_dataframe(input_papers_file)
    if df_papers.empty:
        logger.error(f"Input papers file '{input_papers_file}' is empty or does not exist. Exiting.")
        ensure_dir(os.path.dirname(output_sections_log_file))
        save_jsonl_records([], output_sections_log_file, append=False)
        return
    df_papers['doi'] = df_papers['doi'].apply(clean_doi)
    df_papers.drop_duplicates(subset=['doi'], inplace=True)
    logger.info(f"Loaded {len(df_papers)} unique papers from {input_papers_file}.")

    # Load existing log to identify already processed DOIs and keep their data
    df_existing_log = load_jsonl_to_dataframe(output_sections_log_file)
    existing_processed_dois = set()
    if not df_existing_log.empty:
        if 'doi' in df_existing_log.columns:
            df_existing_log['doi'] = df_existing_log['doi'].apply(clean_doi)
        # Consider a paper processed if it has a non-null 'extracted_introduction_section'
        # or if its 'full_text_extraction_status' is 'success'
        existing_processed_dois = set(
            df_existing_log[
                (df_existing_log['full_text_extraction_status'] == 'success') |
                (df_existing_log['extracted_introduction_section'].notna())
            ]['doi'].tolist()
        )
        logger.info(f"Found {len(existing_processed_dois)} DOIs already processed in '{output_sections_log_file}'.")

    papers_to_process_df = df_papers[~df_papers['doi'].isin(existing_processed_dois)].copy()
    initial_process_count = len(papers_to_process_df)
    logger.info(f"Identified {len(papers_to_process_df)} new papers for full-text extraction.")

    if papers_to_process_df.empty:
        logger.info("No new papers found to process for full-text extraction. Exiting.")
        # If no new papers, ensure the output file is still valid (might be empty or already complete)
        ensure_dir(os.path.dirname(output_sections_log_file))
        if not os.path.exists(output_sections_log_file) or os.path.getsize(output_sections_log_file) == 0:
            save_jsonl_records([], output_sections_log_file, append=False) # Create empty file if it doesn't exist
        return

    papers_to_process_list = papers_to_process_df.to_dict(orient='records')
    logger.info(f"Starting full-text extraction for {len(papers_to_process_list)} papers.")

    ensure_dir(os.path.dirname(output_sections_log_file))

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})

    newly_processed_records = [] # Collect records that are newly processed in this run

    for i, article_data in enumerate(tqdm(papers_to_process_list, desc="Extracting Full Text Sections")):
        doi = article_data.get('doi')
        if not doi:
            logger.warning(f"Skipping paper due to missing DOI: {article_data.get('title', 'N/A')}")
            article_data['full_text_extraction_status'] = 'skipped_no_doi'
            article_data['full_text_extraction_timestamp'] = datetime.now().isoformat()
            newly_processed_records.append(article_data)
            continue

        pdf_url = None
        full_text_content = ""
        extraction_status = 'failed'
        error_message = None

        try:
            # 1. Query Unpaywall for PDF URL
            unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={unpaywall_email}"
            response = session.get(unpaywall_url, timeout=10)
            response.raise_for_status()
            unpaywall_data = response.json()

            if unpaywall_data and unpaywall_data.get('best_oa_location'):
                pdf_url = unpaywall_data['best_oa_location'].get('url_for_pdf')
                if not pdf_url:
                    # Fallback to HTML if PDF not directly available but a landing page is
                    pdf_url = unpaywall_data['best_oa_location'].get('url')
                    logger.info(f"DOI {doi}: No direct PDF. Trying best OA URL: {pdf_url}")

            if pdf_url:
                logger.info(f"DOI {doi}: Found PDF URL: {pdf_url}")
                # 2. Download PDF content
                pdf_response = session.get(pdf_url, timeout=20)
                pdf_response.raise_for_status()

                # Check if content type is PDF
                content_type = pdf_response.headers.get('Content-Type', '').lower()
                if 'application/pdf' in content_type:
                    full_text_content = extract_text_from_pdf(pdf_response.content)
                    if full_text_content.strip():
                        extraction_status = 'success'
                        logger.info(f"DOI {doi}: Successfully extracted text from PDF.")
                    else:
                        extraction_status = 'failed_empty_pdf_text'
                        error_message = "PDF downloaded but no readable text extracted."
                        logger.warning(f"DOI {doi}: {error_message}")
                elif 'text/html' in content_type:
                    # If it's an HTML page, try to parse text from HTML
                    soup = BeautifulSoup(pdf_response.content, 'html.parser')
                    full_text_content = soup.get_text(separator='\n')
                    if full_text_content.strip():
                        extraction_status = 'success_from_html'
                        logger.info(f"DOI {doi}: Successfully extracted text from HTML.")
                    else:
                        extraction_status = 'failed_empty_html_text'
                        error_message = "HTML downloaded but no readable text extracted."
                        logger.warning(f"DOI {doi}: {error_message}")
                else:
                    extraction_status = 'failed_unsupported_content_type'
                    error_message = f"Unsupported content type from URL: {content_type}"
                    logger.warning(f"DOI {doi}: {error_message}")
            else:
                extraction_status = 'failed_no_pdf_url'
                error_message = "No direct PDF URL found via Unpaywall."
                logger.warning(f"DOI {doi}: {error_message}")

        except requests.exceptions.RequestException as e:
            extraction_status = 'failed_network_error'
            error_message = f"Network error during download for DOI {doi}: {e}"
            logger.error(f"DOI {doi}: {error_message}")
        except json.JSONDecodeError as e:
            extraction_status = 'failed_unpaywall_json'
            error_message = f"Unpaywall JSON decode error for DOI {doi}: {e}"
            logger.error(f"DOI {doi}: {error_message}")
        except Exception as e:
            extraction_status = 'failed_general_error'
            error_message = f"General error during processing for DOI {doi}: {e}"
            logger.error(f"DOI {doi}: {error_message}", exc_info=True)

        # Parse sections if text was successfully extracted
        sections = {}
        if extraction_status.startswith('success'):
            sections = parse_sections(full_text_content)
            logger.info(f"DOI {doi}: Parsed sections: {list(sections.keys())}")

        # Update the article_data dictionary
        article_data['full_text_extraction_status'] = extraction_status
        article_data['full_text_extraction_timestamp'] = datetime.now().isoformat()
        article_data['full_text_extraction_error'] = error_message
        article_data['full_text_pdf_url'] = pdf_url # Store the URL that was attempted

        # Store extracted sections (full text content)
        article_data['extracted_introduction_section'] = sections.get('introduction')
        article_data['extracted_methods_section'] = sections.get('methods')
        article_data['extracted_results_discussion_section'] = sections.get('results_discussion')
        article_data['extracted_conclusion_section'] = sections.get('conclusion')
        article_data['extracted_data_availability_section'] = sections.get('data_availability')
        article_data['extracted_code_availability_section'] = sections.get('code_availability')

        # Add boolean flags for section presence
        article_data['has_introduction_section'] = bool(sections.get('introduction'))
        article_data['has_methods_section'] = bool(sections.get('methods'))
        article_data['has_results_discussion_section'] = bool(sections.get('results_discussion'))
        article_data['has_conclusion_section'] = bool(sections.get('conclusion'))
        article_data['has_data_availability_section'] = bool(sections.get('data_availability'))
        article_data['has_code_availability_section'] = bool(sections.get('code_availability'))

        newly_processed_records.append(article_data) # Add to list of new records

        # Add a small delay to avoid overwhelming APIs
        time.sleep(random.uniform(0.5, 2.0))

    # Combine existing records with newly processed records and save the complete set
    all_records = df_existing_log.to_dict(orient='records') + newly_processed_records
    save_jsonl_records(all_records, output_sections_log_file, append=False) # Overwrite with complete set
    logger.info(f"Saved total {len(all_records)} records to {output_sections_log_file} (overwritten with combined data).")
    logger.info("--- Full-Text Section Extraction Process Complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts full-text sections from scientific papers using Unpaywall."
    )
    parser.add_argument(
        "--input_papers_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing paper metadata (including DOIs)."
    )
    parser.add_argument(
        "--output_sections_log_file",
        type=str,
        required=True,
        help="Path to the output JSONL file to log extracted sections and status."
    )
    parser.add_argument(
        "--unpaywall_email",
        type=str,
        required=True,
        help="Email address for Unpaywall API access."
    )
    args = parser.parse_args()

    extract_full_text_sections(
        input_papers_file=args.input_papers_file,
        output_sections_log_file=args.output_sections_log_file,
        unpaywall_email=args.unpaywall_email
    )