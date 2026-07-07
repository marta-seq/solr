# src/data_acquisition/pubmed_scraper.py
import requests
from Bio import Entrez
import json
import time
import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple, Optional
import argparse
import xml.etree.ElementTree as ET  # RE-ADDED: For robust XML parsing

# Add project root to path for module imports
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'review' is the actual project root containing 'src'
project_root = "/home/martinha/PycharmProjects/phd/review"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utility functions
from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_existing_dois_and_curated_pmids, append_to_jsonl, clean_doi, \
    load_curated_csv_data

# --- Configuration (Default values, can be overridden by CLI args) ---
DEFAULT_OUTPUT_SCRAPED_FILE = "data/raw/scraped_pubmed_articles.jsonl"
DEFAULT_LOGGING_DIR = "data/logs/scraper_logs/"
DEFAULT_PAPER_CLASSIFICATION_CLEANED_CSV = "data/inputs/paper_classification_cleaned.csv"

DEFAULT_KEYWORDS = [
    "spatial transcriptomics", "spatial proteomics", "spatial omics", "spatial metabolomics",
    "spatial data analysis",
    "spatial data preprocessing", "cell segmentation", "neighborhood analysis",
    "spatial variable genes", "niche analysis", "domain identification",

    "CODEX", "IMC", "Imaging Mass Cytometry", "MIBI",
    "MERFISH", "seqFISH", "Visium", "10x Genomics", "Xenium", "CosMx",
    "GeoMX", "Slide-seq", "Slide-seqV2", "HDST", "High Definition Spatial Transcriptomics",
    "MALSI-MSI", "MALDI Imaging Mass Spectrometry",
]

SLEEP_TIME_BETWEEN_REQUESTS = 2.0  # seconds

logger = logging.getLogger(__name__)


# --- Core Entrez Fetching Function (MODIFIED FOR ROBUST XML PARSING WITH ElementTree) ---
def fetch_article_details_from_pmids(
        pmids: List[str],
        query_label: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches full article details (DOI, Title, Year, PMID, Authors, Journal, Abstract,
    MeSH Terms, Keywords, PMCID, language, publication types, etc.) from PubMed.
    This function now uses ElementTree for robust XML parsing.
    """
    articles_details = []
    if not pmids:
        return []

    chunk_size = 500

    for i in range(0, len(pmids), chunk_size):
        chunk_pmids = pmids[i: i + chunk_size]
        try:
            logger.debug(f"DEBUG: Entrez.efetch PMIDs: '{','.join(chunk_pmids)}'")
            fetch_handle = Entrez.efetch(db="pubmed", id=','.join(chunk_pmids), retmode="xml")
            xml_data = fetch_handle.read()  # Read raw XML data
            fetch_handle.close()

            # Check if response is HTML (e.g., an error page)
            if not xml_data.strip().startswith(b'<?xml'):
                logger.error(
                    f"Entrez.efetch returned HTML (not XML) for PMIDs {chunk_pmids[:5]}... Response start: {xml_data[:200]}...")
                continue  # Skip to next chunk if response is not XML

            root = ET.fromstring(xml_data)  # Use ElementTree for parsing

            # Iterate through each PubmedArticle element
            for pubmed_article_elem in root.findall('PubmedArticle'):
                citation_elem = pubmed_article_elem.find('MedlineCitation')
                article_meta_elem = citation_elem.find('Article')
                pub_data_elem = pubmed_article_elem.find('PubmedData')

                # Initialize fields with defaults
                pmid, doi, pmcid, title, year, journal, abstract, language = '', '', '', '', '', '', '', ''
                authors = []
                mesh_terms = []
                author_keywords = []
                publication_types = []

                # PMID
                pmid_elem = citation_elem.find('PMID')
                if pmid_elem is not None:
                    pmid = pmid_elem.text

                # DOI, PMCID
                if pub_data_elem is not None:
                    article_id_list_elem = pub_data_elem.find('ArticleIdList')
                    if article_id_list_elem is not None:
                        for aid_elem in article_id_list_elem.findall('ArticleId'):
                            idtype = aid_elem.get('IdType', '').lower()
                            if idtype == 'doi':
                                doi = aid_elem.text.lower() if aid_elem.text else ''
                            elif idtype == 'pmc':
                                pmcid = aid_elem.text if aid_elem.text else ''

                # Title
                article_title_elem = article_meta_elem.find('ArticleTitle')
                if article_title_elem is not None:
                    title = article_title_elem.text.strip() if article_title_elem.text else ''

                # Year
                journal_issue_elem = article_meta_elem.find('Journal/JournalIssue')
                if journal_issue_elem is not None:
                    pub_date_elem = journal_issue_elem.find('PubDate')
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find('Year')
                        if year_elem is not None:
                            year = year_elem.text
                        else:
                            medline_date_elem = pub_date_elem.find('MedlineDate')
                            if medline_date_elem is not None and medline_date_elem.text:
                                year_match = re.search(r'^\d{4}', medline_date_elem.text)
                                if year_match:
                                    year = year_match.group(0)

                # Authors
                author_list_elem = article_meta_elem.find('AuthorList')
                if author_list_elem is not None:
                    for author_elem in author_list_elem.findall('Author'):
                        last_name_elem = author_elem.find('LastName')
                        initials_elem = author_elem.find('Initials')
                        collective_name_elem = author_elem.find('CollectiveName')

                        if last_name_elem is not None and initials_elem is not None:
                            authors.append(f"{last_name_elem.text}, {initials_elem.text}")
                        elif collective_name_elem is not None:
                            authors.append(collective_name_elem.text)

                # Journal
                journal_title_elem = article_meta_elem.find('Journal/Title')
                if journal_title_elem is not None:
                    journal = journal_title_elem.text

                # Abstract
                abstract_elem = article_meta_elem.find('Abstract/AbstractText')
                if abstract_elem is not None:
                    abstract_parts = []
                    if len(abstract_elem) > 0:
                        for part in abstract_elem:
                            if part.text:
                                abstract_parts.append(part.text.strip())
                    elif abstract_elem.text:
                        abstract_parts.append(abstract_elem.text.strip())
                    abstract = "\n".join(abstract_parts)

                # MeSH terms
                mesh_heading_list_elem = citation_elem.find('MeshHeadingList')
                if mesh_heading_list_elem is not None:
                    for mesh_heading_elem in mesh_heading_list_elem.findall('MeshHeading'):
                        descriptor_name_elem = mesh_heading_elem.find('DescriptorName')
                        if descriptor_name_elem is not None:
                            mesh_terms.append(descriptor_name_elem.text.strip())

                # Author keywords
                keyword_list_elem = citation_elem.find('KeywordList')
                if keyword_list_elem is not None:
                    for kw_elem in keyword_list_elem.findall('Keyword'):
                        if kw_elem.text:
                            author_keywords.append(kw_elem.text.strip())

                # Language
                language_elem = article_meta_elem.find('Language')
                if language_elem is not None:
                    language = language_elem.text

                # Publication types
                publication_type_list_elem = article_meta_elem.find('PublicationTypeList')
                if publication_type_list_elem is not None:
                    for pt_elem in publication_type_list_elem.findall('PublicationType'):
                        if pt_elem.text:
                            publication_types.append(pt_elem.text.strip())

                # Compile article metadata
                articles_details.append({
                    'doi': doi,
                    'pmid': pmid,
                    'pmcid': pmcid,
                    'title': title,
                    'year': year,
                    'authors': authors,
                    'journal': journal,
                    'abstract': abstract,
                    'mesh_terms': mesh_terms,
                    'author_keywords': author_keywords,
                    'language': language,
                    'publication_types': publication_types,
                    'scrape_date': datetime.now().isoformat(),
                    'source': 'PubMed',
                    'query_label': query_label or 'unknown',
                    'status': 'uncurated_new',
                    'relevance_score': None,
                    'annotation_score': 0,
                    'found_via': None
                })

            time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)
        except ET.ParseError as pe:
            logger.error(
                f"XML ParseError for chunk {i}-{i + len(chunk_pmids)} (PMIDs: {', '.join(chunk_pmids[:5])}...): {pe}",
                exc_info=True)
            logger.error(f"Problematic XML data snippet: {xml_data[:500]}...")
            time.sleep(5)
        except Exception as e:
            logger.error(
                f"Error fetching PubMed article details chunk {i}-{i + len(chunk_pmids)} (PMIDs: {', '.join(chunk_pmids[:5])}...): {e}",
                exc_info=True)
            time.sleep(5)

    return articles_details


# --- Consolidated Search Function (MODIFIED FOR ROBUST XML PARSING WITH ElementTree) ---
def search_pubmed_and_fetch(
        query_type: str,
        query_value: Any,
        existing_dois: Set[str],
        found_via_type: str,
        email: str
) -> List[Dict[str, Any]]:
    """
    Consolidated function to search PubMed and fetch new articles.
    This function now uses ElementTree for esearch XML parsing.
    """
    Entrez.email = email

    new_articles = []
    id_list = []
    log_message = ""

    try:
        if query_type == 'keywords':
            query_string = " OR ".join([f'("{kw}"[tiab] OR "{kw}"[mesh])' for kw in query_value])
            log_message = f"Searching PubMed by keywords: '{query_string}'"
            handle = Entrez.esearch(db="pubmed", term=query_string, retmax="1000000")
            xml_data = handle.read()  # Read raw XML data
            handle.close()

            if not xml_data.strip().startswith(b'<?xml'):
                logger.error(
                    f"Entrez.esearch returned HTML (not XML) for keyword search '{query_string[:50]}...'. Response start: {xml_data[:200]}...")
                return []

            root = ET.fromstring(xml_data)  # Parse with ElementTree
            id_list = [id_elem.text for id_elem in root.findall('IdList/Id') if id_elem.text]  # Extract IDs
        elif query_type == 'direct_query':
            log_message = f"Searching PubMed with direct query: '{query_value}'"
            handle = Entrez.esearch(db="pubmed", term=query_value, retmax="100000")
            xml_data = handle.read()  # Read raw XML data
            handle.close()

            if not xml_data.strip().startswith(b'<?xml'):
                logger.error(
                    f"Entrez.esearch returned HTML (not XML) for direct query '{query_value[:50]}...'. Response start: {xml_data[:200]}...")
                return []

            root = ET.fromstring(xml_data)  # Parse with ElementTree
            id_list = [id_elem.text for id_elem in root.findall('IdList/Id') if id_elem.text]  # Extract IDs
        elif query_type == 'similar_articles':
            pmids_to_process = list(query_value)
            if not pmids_to_process:
                logger.info("No PMIDs provided for similar articles search.")
                return []
            log_message = f"Searching PubMed for similar articles based on {len(pmids_to_process)} PMIDs"

            similar_pmids_found = set()
            elink_chunk_size = 50
            for i in range(0, len(pmids_to_process), elink_chunk_size):
                chunk_pmids = pmids_to_process[i: i + elink_chunk_size]
                elink_handle = Entrez.elink(db="pubmed", id=','.join(chunk_pmids), cmd="neighbor")
                elink_xml_data = elink_handle.read()  # Read raw XML data
                elink_handle.close()

                if not elink_xml_data.strip().startswith(b'<?xml'):
                    logger.error(
                        f"Entrez.elink returned HTML (not XML) for chunk {chunk_pmids[:5]}... Response start: {elink_xml_data[:200]}...")
                    continue

                elink_root = ET.fromstring(elink_xml_data)  # Parse with ElementTree

                for linkset_elem in elink_root.findall('LinkSet'):
                    for linksetdb_elem in linkset_elem.findall('LinkSetDb'):
                        if linksetdb_elem.find('DbTo').text == 'pubmed' and linksetdb_elem.find(
                                'LinkName').text == 'pubmed_pubmed_refs':
                            for link_elem in linksetdb_elem.findall('Link'):
                                id_elem = link_elem.find('Id')
                                if id_elem is not None and id_elem.text:
                                    similar_pmids_found.add(id_elem.text)
                time.sleep(SLEEP_TIME_BETWEEN_REQUESTS * 2)

            id_list = list(similar_pmids_found - query_value)
        else:
            logger.error(f"Unknown query type: {query_type}")
            return []

        logger.info(f"\n--- {log_message} ---")
        logger.info(f"Found {len(id_list)} articles on PubMed for this query.")

        if not id_list:
            return []

        all_found_articles_details = fetch_article_details_from_pmids(id_list, query_label=found_via_type)

        for article in all_found_articles_details:
            if article['doi'] and clean_doi(article['doi']) not in existing_dois:
                article['found_via'] = found_via_type
                new_articles.append(article)
                existing_dois.add(clean_doi(article['doi']))

    except ET.ParseError as pe:
        logger.error(f"XML ParseError during search for '{query_type}' with value '{str(query_value)[:100]}...': {pe}",
                     exc_info=True)
        if 'xml_data' in locals():
            logger.error(f"Problematic XML data snippet: {xml_data[:500]}...")
        elif 'elink_xml_data' in locals():
            logger.error(f"Problematic XML data snippet: {elink_xml_data[:500]}...")
        time.sleep(5)
    except Exception as e:
        logger.error(f"Error during PubMed search for '{query_type}' with value '{str(query_value)[:100]}...': {e}",
                     exc_info=True)

    logger.info(f"Added {len(new_articles)} new articles from this search.")
    return new_articles


def main_scraper(
        paper_classification_cleaned_csv: str,
        output_scraped_articles_file: str,
        pubmed_email: str,
        keywords: List[str]
):
    """
    Main function to orchestrate PubMed scraping.
    Now incorporates curated data from the cleaned CSV file.
    """
    Entrez.email = pubmed_email
    log_dir_path = DEFAULT_LOGGING_DIR
    setup_logging(log_dir=log_dir_path, log_prefix="scraping_pubmed_log")

    logger.info("--- Starting PubMed Scraping Process ---")
    logger.info(f"Using email: {pubmed_email}")
    logger.info(f"Paper Classification Cleaned CSV: {paper_classification_cleaned_csv}")
    logger.info(f"Output scraped articles file: {output_scraped_articles_file}")
    logger.info(f"Keywords for search: {', '.join(keywords)}")

    # --- Step 0: Load existing scraped data to manage incremental updates ---
    current_existing_dois_in_pool = set()
    all_scraped_records_so_far = []

    if os.path.exists(output_scraped_articles_file) and os.path.getsize(output_scraped_articles_file) > 0:
        df_existing = load_jsonl_to_dataframe(output_scraped_articles_file)
        all_scraped_records_so_far = df_existing.to_dict(orient='records')
        current_existing_dois_in_pool = set(clean_doi(r['doi']) for r in all_scraped_records_so_far if r.get('doi'))
        logger.info(f"Loaded {len(all_scraped_records_so_far)} existing records from {output_scraped_articles_file}.")
    else:
        logger.info(f"Output file '{output_scraped_articles_file}' does not exist or is empty. Starting fresh.")

    # --- Step 1: Process Curated DOIs from Cleaned CSV ---
    logger.info("\n--- Processing Curated DOIs from Cleaned CSV (paper_classification_cleaned.csv) ---")
    curated_data_from_csv = load_curated_csv_data(paper_classification_cleaned_csv)

    curated_pmids_for_similar_search = set()

    for doi_from_csv, curated_meta in curated_data_from_csv.items():
        cleaned_doi = clean_doi(doi_from_csv)
        logger.debug(f"DEBUG: Processing DOI from CSV: '{doi_from_csv}' -> Cleaned DOI: '{cleaned_doi}'")

        is_already_in_pool = False
        existing_record_index = -1
        for idx, record in enumerate(all_scraped_records_so_far):
            if clean_doi(record.get('doi')) == cleaned_doi:
                is_already_in_pool = True
                existing_record_index = idx
                break

        if not is_already_in_pool:
            logger.info(f"  Fetching details for NEW curated Cleaned CSV DOI: {cleaned_doi}")
            try:
                search_term = f"{cleaned_doi}[doi]"
                logger.debug(f"DEBUG: Entrez.esearch query term for cleaned DOI: '{search_term}'")
                handle = Entrez.esearch(db="pubmed", term=search_term, retmax="1")
                xml_data = handle.read()  # Read raw XML data
                handle.close()

                if not xml_data.strip().startswith(b'<?xml'):
                    logger.error(
                        f"Entrez.esearch returned HTML (not XML) for DOI {cleaned_doi}. Response start: {xml_data[:200]}...")
                    continue  # Skip this DOI if response is not XML

                root = ET.fromstring(xml_data)
                pmids_for_curated = [id_elem.text for id_elem in root.findall('IdList/Id') if id_elem.text]

                article_details = {}
                if pmids_for_curated:
                    details = fetch_article_details_from_pmids(pmids_for_curated, query_label='csv_curation')
                    if details:
                        article_details = details[0]
                        if article_details.get('pmid'):
                            curated_pmids_for_similar_search.add(article_details['pmid'])
                    else:
                        logger.warning(
                            f"    No PubMed details found for Cleaned CSV DOI: {cleaned_doi}. Using basic entry.")
                else:
                    logger.warning(
                        f"    PMID not found in PubMed for Cleaned CSV DOI: {cleaned_doi}. Using basic entry.")

                final_article_entry = {**article_details, **curated_meta}
                final_article_entry['doi'] = cleaned_doi
                final_article_entry['found_via'] = 'csv_curation'
                final_article_entry['scrape_date'] = datetime.now().isoformat()

                all_scraped_records_so_far.append(final_article_entry)
                current_existing_dois_in_pool.add(cleaned_doi)
                logger.info(f"    Added new curated Cleaned CSV paper: {cleaned_doi}.")

            except ET.ParseError as pe:
                logger.error(f"XML ParseError during esearch for Cleaned CSV DOI {cleaned_doi}: {pe}", exc_info=True)
                logger.error(f"Problematic XML data snippet: {xml_data[:500]}...")
            except Exception as e:
                logger.error(f"Error processing new curated Cleaned CSV DOI {cleaned_doi}: {e}", exc_info=True)
            time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)
        else:
            logger.info(f"  Cleaned CSV DOI {cleaned_doi} already in scraped pool. Updating metadata.")
            existing_record = all_scraped_records_so_far[existing_record_index]
            existing_record.update(curated_meta)
            existing_record['found_via'] = 'csv_curation'
            existing_record['status'] = 'curated_method'
            existing_record['scrape_date'] = datetime.now().isoformat()
            if existing_record.get('pmid'):
                curated_pmids_for_similar_search.add(existing_record['pmid'])
            logger.info(f"    Updated existing curated Cleaned CSV paper: {cleaned_doi}.")

    # --- Step 2: REMOVED Processing Curated DOIs from Text File (curated_dois.txt) ---

    # --- Step 3: Perform Keyword Search for new articles ---
    logger.info("\n--- Performing Keyword Search ---")
    new_keyword_articles = search_pubmed_and_fetch('keywords', keywords, current_existing_dois_in_pool,
                                                   'keyword_search', pubmed_email)
    all_scraped_records_so_far.extend(new_keyword_articles)

    # --- Step 4: Perform Similar Articles Search (based on all collected curated PMIDs) ---
    logger.info("\n--- Performing Similar Articles Search ---")
    valid_curated_pmids = {pmid for pmid in curated_pmids_for_similar_search if pmid}
    new_similar_articles = search_pubmed_and_fetch('similar_articles', valid_curated_pmids,
                                                   current_existing_dois_in_pool, 'similar_articles', pubmed_email)
    all_scraped_records_so_far.extend(new_similar_articles)

    # --- Final Save: Deduplicate and save all records atomically ---
    df_final_scraped = pd.DataFrame(all_scraped_records_so_far)
    if 'doi' in df_final_scraped.columns:
        df_final_scraped['doi'] = df_final_scraped['doi'].apply(clean_doi)
        df_final_scraped.drop_duplicates(subset=['doi'], keep='last', inplace=True)

    final_records_to_save = df_final_scraped.to_dict(orient='records')

    ensure_dir(os.path.dirname(output_scraped_articles_file))
    save_jsonl_records(final_records_to_save, output_scraped_articles_file, append=False)

    logger.info(f"\nScraping complete. Total unique DOIs in final output: {len(final_records_to_save)}")
    logger.info(f"Review '{output_scraped_articles_file}' for new uncurated entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Systematic literature scraping from PubMed to build a DOI pool."
    )
    parser.add_argument(
        "--paper_classification_cleaned_csv",
        type=str,
        default=DEFAULT_PAPER_CLASSIFICATION_CLEANED_CSV,
        help=f"Path to the cleaned CSV file with paper classifications (default: {DEFAULT_PAPER_CLASSIFICATION_CLEANED_CSV})."
    )
    parser.add_argument(
        "--output_scraped_articles_file",
        type=str,
        default=DEFAULT_OUTPUT_SCRAPED_FILE,
        help=f"Path to the output JSONL file for scraped articles (default: {DEFAULT_OUTPUT_SCRAPED_FILE})."
    )
    parser.add_argument(
        "--pubmed_email",
        type=str,
        required=True,
        help="Email address required by NCBI for Entrez API access."
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help=f"Comma-separated list of keywords for PubMed search (default: '{', '.join(DEFAULT_KEYWORDS)}')."
    )

    args = parser.parse_args()

    parsed_keywords = [kw.strip() for kw in args.keywords.split(',') if kw.strip()]

    main_scraper(
        paper_classification_cleaned_csv=args.paper_classification_cleaned_csv,
        output_scraped_articles_file=args.output_scraped_articles_file,
        pubmed_email=args.pubmed_email,
        keywords=parsed_keywords
    )
