"""
This Python script is designed for systematic literature scraping from PubMed.
It aims to build and maintain a comprehensive database of relevant research papers
in a 'doi_pool.jsonl' file.

Key functionalities include:
1.  **Initialization with Curated DOIs:** It first processes a 'curated_dois.txt'
    file containing a list of known, highly relevant DOIs (e.g., benchmark methods).
    These papers are added to the 'doi_pool.jsonl' with a 'curated_method' status,
    and their PubMed IDs (PMIDs) are fetched if available. Papers without PMIDs
    (e.g., preprints) are also tracked.
2.  **Keyword-Based Search:** It performs a broad search on PubMed using a predefined
    set of keywords related to spatial omics and analysis. New articles found
    are added to 'doi_pool.jsonl' with an 'uncurated_new' status.
3.  **Similar Articles Search:** For the papers identified as 'curated_method'
    (and having PMIDs), the script leverages PubMed's 'similar articles' feature
    (E-Link) to discover related research. These new findings are also added to
    'doi_pool.jsonl' as 'uncurated_new'.

The script prevents duplicate entries by maintaining a set of already processed DOIs.
All logging information (script progress, warnings, errors) is saved to a
dated log file within the 'data/logging_scrap' directory, alongside console output.

Only PubMed is queried in this version of the scraper.
author: AMSequeira
"""
import requests
from Bio import Entrez
import json
import time
import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple


EMAIL = "id9417@alunos.uminho.pt" # Required by NCBI for Entrez
Entrez.email = EMAIL # Set globally for all Entrez calls

BASE_DIR = "../../data" # Directory to store your data files
CURATED_DOIS_FILE = os.path.join(BASE_DIR, "curated_dois.txt") # Your initial curated DOIs
DOI_POOL_FILE = os.path.join(BASE_DIR, "doi_pool.jsonl") # Where new uncurated DOIs will be added
LOGGING_DIR = os.path.join(BASE_DIR, "logging_scrap") # Directory for log files

KEYWORDS = [
    "spatial transcriptomics", "spatial proteomics", "spatial omics", "spatial metabolomics",
    "spatial data analysis",
    "spatial data preprocessing", "cell segmentation", "neighborhood analysis",
     "spatial variable genes", "niche analysis", "domain identification",
]

# Set a reasonable sleep time to avoid hitting rate limits (especially for Entrez)
SLEEP_TIME_BETWEEN_REQUESTS = 0.5  # seconds

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():
    """Sets up logging to console and a file with date/time in filename."""
    ensure_dir(LOGGING_DIR)  # Ensure logging directory exists

    # Create file handler with a unique name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = os.path.join(LOGGING_DIR, f"scraping_log_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logging initialized.")
    logger.info(f"Log file: {log_file_name}")


def ensure_dir(path: str):
    """Ensures a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")


def load_existing_dois_and_curated_pmids(doi_pool_file: str) -> Tuple[Set[str], Set[str]]:
    """
    Loads all DOIs from the comprehensive DOI pool file and PMIDs for curated methods
    that are already in the pool.
    """
    existing_dois = set()
    curated_pmids = set()

    if os.path.exists(doi_pool_file):
        with open(doi_pool_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    if 'doi' in entry and entry['doi']:
                        existing_dois.add(entry['doi'].lower())

                    # Ensure the PMID is a non-empty string and status is 'curated_method'
                    if 'pmid' in entry and isinstance(entry['pmid'], str) and entry['pmid'] and \
                            entry.get('status') == 'curated_method':
                        curated_pmids.add(entry['pmid'])

                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {doi_pool_file} at line {line_num}: {line.strip()}")
        logger.info(f"Loaded {len(existing_dois)} total existing DOIs from {doi_pool_file}.")
        logger.info(f"Loaded {len(curated_pmids)} PMIDs for curated methods from {doi_pool_file}.")
    else:
        logger.info(f"'{doi_pool_file}' not found. Starting with empty DOI pool.")

    return existing_dois, curated_pmids


def fetch_article_details_from_pmids(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetches full article details (DOI, Title, Year, PMID, Authors, Journal, Abstract, MeSH Keywords)
    for a list of PMIDs."""
    articles_details = []

    if not pmids:
        return []

    chunk_size = 500

    for i in range(0, len(pmids), chunk_size):
        chunk_pmids = pmids[i: i + chunk_size]
        try:
            # retmode="xml" is good, but sometimes "Medline" can be easier to parse certain fields.
            # However, sticking to "xml" is generally more robust for structured data.
            fetch_handle = Entrez.efetch(db="pubmed", id=','.join(chunk_pmids), retmode="xml")
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()

            for pubmed_article in articles['PubmedArticle']:
                article_meta = pubmed_article['MedlineCitation']['Article']

                doi = ''
                pmid = ''
                authors = []
                journal_title = ''
                abstract_text = ''
                mesh_terms = []
                author_keywords = []  # Keywords supplied by authors

                # --- Extract PMID and DOI ---
                if 'ArticleIdList' in article_meta:
                    for article_id in article_meta['ArticleIdList']:
                        if article_id.attributes.get('IdType') == 'doi':
                            doi = str(article_id)
                        elif article_id.attributes.get('IdType') == 'pubmed':
                            pmid = str(article_id)

                if not doi and 'ELocationID' in article_meta:
                    for eloc_id in article_meta['ELocationID']:
                        if eloc_id.attributes.get('EIdType') == 'doi':
                            doi = str(eloc_id)
                            break

                if not pmid and 'PMID' in pubmed_article['MedlineCitation']:
                    pmid = str(pubmed_article['MedlineCitation']['PMID'])

                # --- Extract Title ---
                title = article_meta.get('ArticleTitle', 'No Title').replace('\n', ' ').strip()

                # --- Extract Year ---
                year = ''
                if 'Journal' in article_meta and 'JournalIssue' in article_meta['Journal']:
                    if 'PubDate' in article_meta['Journal']['JournalIssue']:
                        pub_date = article_meta['Journal']['JournalIssue']['PubDate']
                        if 'Year' in pub_date:
                            year = pub_date['Year']
                        elif 'MedlineDate' in pub_date:
                            year_match = re.search(r'^\d{4}', pub_date['MedlineDate'])
                            if year_match:
                                year = year_match.group(0)

                # --- Extract Authors ---
                if 'AuthorList' in article_meta:
                    for author in article_meta['AuthorList']:
                        if 'LastName' in author and 'Initials' in author:
                            authors.append(f"{author['LastName']}, {author['Initials']}")
                        elif 'CollectiveName' in author:  # For group authors
                            authors.append(author['CollectiveName'])
                        # You might want more sophisticated handling for ForeName, etc.

                # --- Extract Journal Title ---
                if 'Journal' in article_meta and 'Title' in article_meta['Journal']:
                    journal_title = article_meta['Journal']['Title']

                # --- Extract Abstract ---
                if 'Abstract' in article_meta:
                    abstract_parts = []
                    # Abstract can be structured into sections (e.g., copyright, object)
                    # We want the main abstract text.
                    if 'AbstractText' in article_meta['Abstract']:
                        # AbstractText can be a list of TagSet.Tag or just one.
                        # Some abstract parts might have 'Label' attributes (e.g., BACKGROUND, METHODS)
                        for abst_text_part in article_meta['Abstract']['AbstractText']:
                            abstract_parts.append(str(abst_text_part).strip())
                    abstract_text = "\n".join(abstract_parts).strip()

                # --- Extract MeSH Terms ---
                if 'MeshHeadingList' in pubmed_article['MedlineCitation']:
                    for mesh_heading in pubmed_article['MedlineCitation']['MeshHeadingList']:
                        if 'DescriptorName' in mesh_heading:
                            mesh_terms.append(str(mesh_heading['DescriptorName']).strip())

                # --- Extract Author Keywords (often in a 'KeywordList' if present) ---
                if 'KeywordList' in pubmed_article['MedlineCitation']:
                    for keyword_list in pubmed_article['MedlineCitation']['KeywordList']:
                        for keyword in keyword_list:
                            author_keywords.append(str(keyword).strip())

                articles_details.append({
                    'doi': doi,
                    'pmid': pmid,
                    'title': title,
                    'year': year,
                    'authors': authors,
                    'journal': journal_title,
                    'abstract': abstract_text,
                    'mesh_terms': mesh_terms,
                    'author_keywords': author_keywords,  # New field
                    'source': 'PubMed',
                })
            time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)
        except Exception as e:
            logger.error(
                f"Error fetching PubMed article details chunk {i}-{i + len(chunk_pmids)} (PMIDs: {', '.join(chunk_pmids[:5])}...): {e}",
                exc_info=True)
            time.sleep(5)
    return articles_details

def search_pubmed_by_keywords(keywords: List[str], existing_dois: Set[str], found_via_type: str) -> List[
    Dict[str, Any]]:
    """Searches PubMed for articles matching keywords and returns new ones."""
    new_articles = []

    query_string = " OR ".join([f'"{kw}"' for kw in keywords])
    logger.info(f"\n--- Searching PubMed by keywords: '{query_string}' ---")

    try:
        handle = Entrez.esearch(db="pubmed", term=query_string, retmax="1000000")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]
        logger.info(f"Found {len(id_list)} articles on PubMed for keyword search.")

        if not id_list:
            return []

        all_found_articles_details = fetch_article_details_from_pmids(id_list)

        for article in all_found_articles_details:
            if article['doi'] and article['doi'].lower() not in existing_dois:
                article['status'] = 'uncurated_new'
                article['relevance_score'] = None
                article['found_via'] = found_via_type
                new_articles.append(article)
                existing_dois.add(article['doi'].lower())

    except Exception as e:
        logger.error(f"Error during PubMed keyword search for '{query_string}': {e}", exc_info=True)

    logger.info(f"Added {len(new_articles)} new articles from PubMed keyword search.")
    return new_articles


def search_pubmed_similar_articles(curated_pmids: Set[str], existing_dois: Set[str], found_via_type: str) -> List[
    Dict[str, Any]]:
    """Searches PubMed for similar articles based on curated PMIDs."""
    new_articles = []
    logger.info(f"\n--- Searching PubMed for similar articles based on {len(curated_pmids)} curated PMIDs ---")

    pmids_to_process = list(curated_pmids)
    if not pmids_to_process:
        logger.info("No curated PMIDs to search similar articles for.")
        return []

    chunk_size = 50

    for i in range(0, len(pmids_to_process), chunk_size):
        chunk_pmids = pmids_to_process[i: i + chunk_size]
        try:
            handle = Entrez.elink(db="pubmed", id=','.join(chunk_pmids), cmd="neighbor")
            record = Entrez.read(handle)
            handle.close()

            similar_pmids = set()
            for linkset in record:
                if 'LinkSetDb' in linkset:
                    for linksetdb in linkset['LinkSetDb']:
                        if linksetdb.get('DbTo') == 'pubmed' and linksetdb.get('LinkName') == 'pubmed_pubmed_refs':
                            for link in linksetdb['Link']:
                                similar_pmids.add(link['Id'])

            similar_pmids = [p for p in similar_pmids if
                             p not in curated_pmids]  # Ensure we don't re-process original curated PMIDs here

            found_articles_details = fetch_article_details_from_pmids(similar_pmids)

            for article in found_articles_details:
                if article['doi'] and article['doi'].lower() not in existing_dois:
                    article['status'] = 'uncurated_new'
                    article['relevance_score'] = None
                    article['found_via'] = found_via_type
                    new_articles.append(article)
                    existing_dois.add(article['doi'].lower())

            time.sleep(SLEEP_TIME_BETWEEN_REQUESTS * 2)

        except Exception as e:
            logger.error(f"Error during PubMed similar articles search for chunk {chunk_pmids}: {e}", exc_info=True)
            time.sleep(5)

    logger.info(f"Added {len(new_articles)} new articles from PubMed similar articles search.")
    return new_articles


def append_to_doi_pool(articles: List[Dict[str, Any]], file_path: str):
    """Appends new articles to the DOI pool file."""
    if articles:
        with open(file_path, 'a') as f:
            for article in articles:
                f.write(json.dumps(article) + '\n')
        logger.info(f"Successfully appended {len(articles)} articles to {file_path}")
    else:
        logger.info("No new articles to append.")


def main_scraper():
    # Setup logging first
    setup_logging()

    ensure_dir(BASE_DIR)

    # --- Step 0: Ensure initial curated_dois.txt entries are in doi_pool.jsonl ---
    current_existing_dois_in_pool = set()
    if os.path.exists(DOI_POOL_FILE):
        with open(DOI_POOL_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'doi' in entry and entry['doi']:
                        current_existing_dois_in_pool.add(entry['doi'].lower())
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line during initial DOI pool load.")

    initial_curated_dois_from_file = set()
    if os.path.exists(CURATED_DOIS_FILE):
        with open(CURATED_DOIS_FILE, 'r') as f:
            for line in f:
                doi = line.strip().lower()
                if doi:
                    initial_curated_dois_from_file.add(doi)

    articles_to_initially_add = []
    logger.info("\n--- Initializing DOI pool with curated DOIs from text file ---")
    for doi_from_curated in initial_curated_dois_from_file:
        if doi_from_curated not in current_existing_dois_in_pool:
            logger.info(f"  Processing curated DOI not yet in pool: {doi_from_curated}")
            try:
                handle = Entrez.esearch(db="pubmed", term=f"{doi_from_curated}[doi]", retmax="1")
                record = Entrez.read(handle)
                handle.close()
                pmids_for_curated = record["IdList"]

                if pmids_for_curated:
                    details = fetch_article_details_from_pmids(pmids_for_curated)
                    if details:
                        article = details[0]
                        article['status'] = 'curated_method'
                        article['found_via'] = 'manual_curation'
                        articles_to_initially_add.append(article)
                        current_existing_dois_in_pool.add(doi_from_curated)
                        logger.info(f"    Added full details for {doi_from_curated} to initial add list.")
                    else:
                        logger.warning(
                            f"    No details fetched from PubMed for curated DOI: {doi_from_curated}. Adding basic entry.")
                        articles_to_initially_add.append({
                            'doi': doi_from_curated,
                            'title': 'Unknown Title (from curated_dois.txt)',
                            'year': 'Unknown',
                            'source': 'curated_txt',
                            'status': 'curated_method',
                            'relevance_score': None,
                            'pmid': None,
                            'found_via': 'manual_curation'
                        })
                        current_existing_dois_in_pool.add(doi_from_curated)
                else:
                    logger.warning(
                        f"    PMID not found in PubMed for curated DOI: {doi_from_curated}. Adding basic entry.")
                    articles_to_initially_add.append({
                        'doi': doi_from_curated,
                        'title': 'Unknown Title (from curated_dois.txt)',
                        'year': 'Unknown',
                        'source': 'curated_txt',
                        'status': 'curated_method',
                        'relevance_score': None,
                        'pmid': None,
                        'found_via': 'manual_curation'
                    })
                    current_existing_dois_in_pool.add(doi_from_curated)
            except Exception as e:
                logger.error(f"Error during initial fetch for curated DOI {doi_from_curated}: {e}", exc_info=True)
            time.sleep(SLEEP_TIME_BETWEEN_REQUESTS)

    append_to_doi_pool(articles_to_initially_add, DOI_POOL_FILE)
    logger.info(f"Initial DOI pool setup complete. Total entries now in pool: {len(current_existing_dois_in_pool)}")

    # --- Step 1: Load comprehensive existing DOIs and curated PMIDs from the now populated pool ---
    existing_dois_set, curated_pmids_set = load_existing_dois_and_curated_pmids(DOI_POOL_FILE)

    # 2. Perform Keyword Search
    new_keyword_articles = search_pubmed_by_keywords(KEYWORDS, existing_dois_set, 'keyword_search')
    append_to_doi_pool(new_keyword_articles, DOI_POOL_FILE)

    # 3. Perform Similar Articles Search (only for curated ones)
    new_similar_articles = search_pubmed_similar_articles(curated_pmids_set, existing_dois_set, 'similar_articles')
    append_to_doi_pool(new_similar_articles, DOI_POOL_FILE)

    logger.info(f"\nScraping complete.")
    final_dois_count, _ = load_existing_dois_and_curated_pmids(DOI_POOL_FILE)
    logger.info(f"Total unique DOIs currently tracked: {len(final_dois_count)}")
    logger.info(f"Review '{DOI_POOL_FILE}' for new uncurated entries.")
if __name__ == "__main__":
    # For initial testing, you might uncomment this to clear the DOI pool for a fresh run
    # BE CAREFUL: This will delete your existing doi_pool.jsonl
    if os.path.exists(DOI_POOL_FILE):
        os.remove(DOI_POOL_FILE)
        print(f"Removed existing {DOI_POOL_FILE} for a fresh start.")

    main_scraper()