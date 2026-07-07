"""
PubMed Scraper for Spatial Omics Literature

Scrapes papers using three strategies:
1. Curated seed papers (CSV) - Loads manually curated method/dataset papers
2. Similar articles - Finds related papers via PubMed's similarity algorithm
3. Keyword search - Searches spatial omics terms (technologies, methods, analysis)

Outputs:
- data/raw/scraped_pubmed_articles.jsonl - All papers with metadata
- data/raw/scraping_metadata.json - Tracks scraping history for incremental updates
- data/logs/scraper_logs/scraping_pubmed_log_YYYYMMDD_HHMMSS.log - Execution logs

Modes:
- full: Complete scrape from scratch (~1-2 hours, ~10k papers)
- incremental: Only new papers since last run (~2-5 min, ~10-50 papers)
- date_range: Custom date range

Usage:
    python3 pubmed_scraper_new.py --pubmed_email you@email.com --mode full
"""
import requests
from Bio import Entrez
import time
import os
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/home/martinha/PycharmProjects/phd/review"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging_setup import setup_logging
from src.utils.file_helpers import ensure_dir
from src.utils.data_helpers import load_curated_csv_data, clean_doi

DEFAULT_OUTPUT_SCRAPED_FILE = "data/raw/scraped_pubmed_articles.jsonl"
DEFAULT_METADATA_FILE = "data/raw/scraping_metadata.json"
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
SLEEP_TIME = 0.35
logger = logging.getLogger(__name__)


def load_curated_csv_data(file_path: str) -> Dict[str, Dict]:
    """
    Loads curated data from a CSV file and returns a dictionary.
    Key: cleaned DOI
    Value: dictionary of metadata (category, pipeline_category, name, etc.)
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        logger.warning(f"Curated CSV file not found or empty: {file_path}. Returning empty dict.")
        return {}

    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

        if 'doi' not in df.columns:
            logger.error(f"Curated CSV '{file_path}' does not contain a 'doi' column!")
            return {}

        # Clean DOIs
        df['doi'] = df['doi'].apply(clean_doi)

        # Remove rows with empty DOIs
        df = df[df['doi'] != '']

        # Drop duplicates
        df.drop_duplicates(subset=['doi'], inplace=True)

        logger.info(f"Loaded {len(df)} unique records from curated CSV: {file_path}.")

        # Convert to dictionary format: {doi: {metadata}}
        result = {}
        for _, row in df.iterrows():
            doi = row['doi']
            # Create metadata dict - get column names and values properly
            metadata = {}
            for col in df.columns:
                if col != 'doi':  # Skip the DOI column
                    metadata[col] = row[col]

            # Add status markers
            metadata['status'] = 'curated_method'
            result[doi] = metadata

        return result

    except Exception as e:
        logger.error(f"Error loading curated CSV '{file_path}': {e}")
        return {}

class ScrapingMetadata:
    """Track scraping runs and enable incremental updates."""

    def __init__(self, metadata_file: str):
        self.metadata_file = metadata_file
        self.data = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'last_full_scrape': None,
            'last_incremental_scrape': None,
            'total_papers_scraped': 0,
            'scraping_history': []
        }

    def save(self):
        ensure_dir(os.path.dirname(self.metadata_file))
        with open(self.metadata_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record_run(self, mode: str, papers_added: int):
        timestamp = datetime.now().isoformat()
        if mode == 'full':
            self.data['last_full_scrape'] = timestamp
        else:
            self.data['last_incremental_scrape'] = timestamp

        self.data['total_papers_scraped'] += papers_added
        self.data['scraping_history'].append({
            'timestamp': timestamp,
            'mode': mode,
            'papers_added': papers_added
        })
        self.save()

    def get_last_scrape_date(self, mode: str = 'incremental') -> Optional[str]:
        if mode == 'full':
            return self.data.get('last_full_scrape')
        return self.data.get('last_incremental_scrape')


def build_date_query(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Build PubMed date range query."""
    if not start_date:
        return ""

    end_date = end_date or datetime.now().strftime("%Y/%m/%d")
    return f" AND ({start_date}[PDAT]:{end_date}[PDAT])"


def fetch_article_details(pmids: List[str], query_label: Optional[str] = None) -> List[Dict]:
    """Fetch article metadata from PubMed."""
    articles = []
    if not pmids:
        return []

    for i in range(0, len(pmids), 500):
        chunk = pmids[i:i + 500]
        try:
            handle = Entrez.efetch(db="pubmed", id=','.join(chunk), retmode="xml")
            root = ET.fromstring(handle.read())
            handle.close()

            for article_elem in root.findall('PubmedArticle'):
                citation = article_elem.find('MedlineCitation')
                article_meta = citation.find('Article')
                pub_data = article_elem.find('PubmedData')

                pmid = citation.find('PMID').text if citation.find('PMID') is not None else ''
                doi = pmcid = ''

                if pub_data is not None:
                    for aid in pub_data.findall('.//ArticleId'):
                        idtype = aid.get('IdType', '').lower()
                        if idtype == 'doi':
                            doi = aid.text.lower() if aid.text else ''
                        elif idtype == 'pmc':
                            pmcid = aid.text if aid.text else ''

                title_elem = article_meta.find('ArticleTitle')
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ''

                year = ''
                pub_date = article_meta.find('.//PubDate/Year')
                if pub_date is not None:
                    year = pub_date.text
                else:
                    medline_date = article_meta.find('.//PubDate/MedlineDate')
                    if medline_date is not None and medline_date.text:
                        match = re.search(r'^\d{4}', medline_date.text)
                        if match:
                            year = match.group(0)

                authors = []
                for author in article_meta.findall('.//Author'):
                    last = author.find('LastName')
                    init = author.find('Initials')
                    coll = author.find('CollectiveName')
                    if last is not None and init is not None:
                        authors.append(f"{last.text}, {init.text}")
                    elif coll is not None:
                        authors.append(coll.text)

                journal_elem = article_meta.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''

                abstract_parts = []
                for abs_elem in article_meta.findall('.//Abstract/AbstractText'):
                    if abs_elem.text:
                        abstract_parts.append(abs_elem.text.strip())
                abstract = "\n".join(abstract_parts)

                mesh_terms = [
                    desc.text.strip()
                    for desc in citation.findall('.//MeshHeading/DescriptorName')
                    if desc.text
                ]

                keywords = [
                    kw.text.strip()
                    for kw in citation.findall('.//KeywordList/Keyword')
                    if kw.text
                ]

                language_elem = article_meta.find('Language')
                language = language_elem.text if language_elem is not None else ''

                pub_types = [
                    pt.text.strip()
                    for pt in article_meta.findall('.//PublicationTypeList/PublicationType')
                    if pt.text
                ]

                articles.append({
                    'doi': doi,
                    'pmid': pmid,
                    'pmcid': pmcid,
                    'title': title,
                    'year': year,
                    'authors': authors,
                    'journal': journal,
                    'abstract': abstract,
                    'mesh_terms': mesh_terms,
                    'author_keywords': keywords,
                    'language': language,
                    'publication_types': pub_types,
                    'scrape_date': datetime.now().isoformat(),
                    'source': 'PubMed',
                    'query_label': query_label or 'unknown',
                    'status': 'uncurated_new',
                    'relevance_score': None,
                    'annotation_score': 0,
                })

            time.sleep(SLEEP_TIME)
        except Exception as e:
            logger.error(f"Error fetching chunk {i}-{i + len(chunk)}: {e}")
            time.sleep(5)

    return articles


def search_pubmed(query_type: str, query_value, existing_dois: Set[str],
                  found_via: str, email: str, date_filter: str = "") -> List[Dict]:
    """Search PubMed and return new articles."""
    Entrez.email = email
    new_articles = []

    try:
        if query_type == 'keywords':
            base_query = " OR ".join([f'("{kw}"[tiab] OR "{kw}"[mesh])' for kw in query_value])
            query = base_query + date_filter
            logger.info(f"Keyword query: {query}")
            handle = Entrez.esearch(db="pubmed", term=query, retmax="1000000")

        elif query_type == 'direct_query':
            query = query_value + date_filter
            logger.info(f"Direct query: {query}")
            handle = Entrez.esearch(db="pubmed", term=query, retmax="100000")

        elif query_type == 'similar':
            similar_pmids = set()
            query_pmids = list(query_value)
            logger.info(f"Finding similar articles for {len(query_pmids)} seed PMIDs")

            for i in range(0, len(query_pmids), 50):
                chunk = query_pmids[i:i + 50]
                elink_handle = Entrez.elink(db="pubmed", id=','.join(chunk), cmd="neighbor")
                root = ET.fromstring(elink_handle.read())
                elink_handle.close()

                for link in root.findall('.//LinkSetDb[LinkName="pubmed_pubmed_refs"]/Link/Id'):
                    if link.text:
                        similar_pmids.add(link.text)
                time.sleep(SLEEP_TIME * 2)

            id_list = list(similar_pmids - query_value)
            logger.info(f"Found {len(id_list)} similar articles")

            all_articles = fetch_article_details(id_list, found_via)

            for article in all_articles:
                if article['doi'] and clean_doi(article['doi']) not in existing_dois:
                    article['found_via'] = found_via
                    new_articles.append(article)
                    existing_dois.add(clean_doi(article['doi']))
            return new_articles
        else:
            logger.error(f"Unknown query type: {query_type}")
            return []

        root = ET.fromstring(handle.read())
        handle.close()
        id_list = [id_elem.text for id_elem in root.findall('IdList/Id') if id_elem.text]

        logger.info(f"Found {len(id_list)} articles for {query_type}")

        if not id_list:
            return []

        all_articles = fetch_article_details(id_list, found_via)

        for article in all_articles:
            if article['doi'] and clean_doi(article['doi']) not in existing_dois:
                article['found_via'] = found_via
                new_articles.append(article)
                existing_dois.add(clean_doi(article['doi']))

    except Exception as e:
        logger.error(f"Error during search: {e}")

    logger.info(f"Added {len(new_articles)} new articles")
    return new_articles


def main_scraper(
        paper_classification_csv: str,
        output_file: str,
        metadata_file: str,
        email: str,
        keywords: List[str],
        mode: str = 'incremental',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
):
    """Main scraper with incremental update support."""
    Entrez.email = email
    setup_logging(log_dir=DEFAULT_LOGGING_DIR, log_prefix="scraping_pubmed_log")

    logger.info(f"--- Starting PubMed Scraping (Mode: {mode}) ---")

    metadata = ScrapingMetadata(metadata_file)

    # Load existing data
    existing_dois = set()
    all_records = []

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df = pd.read_json(output_file, lines=True)
        all_records = df.to_dict(orient='records')
        existing_dois = {clean_doi(r['doi']) for r in all_records if r.get('doi')}
        logger.info(f"Loaded {len(all_records)} existing records")

    # Determine date filter
    # Determine date filter
    date_filter = ""
    if mode == 'incremental':
        last_scrape = metadata.get_last_scrape_date('incremental')
        if not last_scrape:
            # Use last_full_scrape as fallback
            last_scrape = metadata.get_last_scrape_date('full')
            if last_scrape:
                logger.info(f"No incremental scrape found, using last full scrape date: {last_scrape}")
            else:
                logger.warning("No previous scrape found, searching last 30 days")
                last_scrape = (datetime.now() - timedelta(days=30)).isoformat()

        # Add 1 day buffer to avoid missing papers
        last_date = datetime.fromisoformat(last_scrape) - timedelta(days=1)
        start_date = last_date.strftime("%Y/%m/%d")
        logger.info(f"Incremental mode: searching from {start_date}")
        date_filter = build_date_query(start_date, None)

    if mode in ['incremental', 'date_range'] and start_date:
        date_filter = build_date_query(start_date, end_date)

    # Process curated papers from CSV
    logger.info("--- Processing Curated CSV ---")
    curated_data = load_curated_csv_data(paper_classification_csv)
    curated_pmids = set()

    for doi, meta in curated_data.items():
        cleaned_doi = clean_doi(doi)

        existing_idx = next(
            (i for i, r in enumerate(all_records) if clean_doi(r.get('doi')) == cleaned_doi),
            None
        )

        if existing_idx is None:
            # NEW PAPER - try to fetch from PubMed
            try:
                handle = Entrez.esearch(db="pubmed", term=f"{cleaned_doi}[doi]", retmax="1")
                root = ET.fromstring(handle.read())
                handle.close()
                pmids = [id_elem.text for id_elem in root.findall('IdList/Id')]

                if pmids:
                    # Found in PubMed - fetch full details
                    details = fetch_article_details(pmids, 'csv_curation')
                    if details:
                        article = {**details[0], **meta}
                        article['doi'] = cleaned_doi
                        article['found_via'] = 'csv_curation'
                        article['status'] = 'curated_method'
                    else:
                        # Fetch failed - create basic entry
                        article = {
                            'doi': cleaned_doi,
                            'pmid': pmids[0],
                            'title': meta.get('name', ''),
                            'year': '',
                            'authors': [],
                            'journal': '',
                            'abstract': '',
                            'source': 'CSV',
                            'scrape_date': datetime.now().isoformat(),
                            **meta,
                            'found_via': 'csv_curation',
                            'status': 'curated_method'
                        }

                    if article.get('pmid'):
                        curated_pmids.add(article['pmid'])
                else:
                    # NOT in PubMed - add with CSV metadata only
                    article = {
                        'doi': cleaned_doi,
                        'pmid': '',
                        'title': meta.get('name', ''),
                        'year': '',
                        'authors': [],
                        'journal': '',
                        'abstract': '',
                        'source': 'CSV',
                        'scrape_date': datetime.now().isoformat(),
                        **meta,
                        'found_via': 'csv_curation',
                        'status': 'curated_method'
                    }

                all_records.append(article)
                existing_dois.add(cleaned_doi)
                logger.debug(f"Added new curated paper: {cleaned_doi}")

                time.sleep(SLEEP_TIME)

            except Exception as e:
                # Error - still add paper with CSV data
                logger.warning(f"Error searching PubMed for {cleaned_doi}: {e}. Adding with CSV data only.")
                article = {
                    'doi': cleaned_doi,
                    'pmid': '',
                    'title': meta.get('name', ''),
                    'year': '',
                    'authors': [],
                    'journal': '',
                    'abstract': '',
                    'source': 'CSV',
                    'scrape_date': datetime.now().isoformat(),
                    **meta,
                    'found_via': 'csv_curation',
                    'status': 'curated_method'
                }
                all_records.append(article)
                existing_dois.add(cleaned_doi)
        else:
            # Paper ALREADY EXISTS - update it and collect PMID
            all_records[existing_idx].update(meta)
            all_records[existing_idx]['found_via'] = 'csv_curation'
            all_records[existing_idx]['status'] = 'curated_method'

            # CRITICAL: Collect PMID for similar search
            if all_records[existing_idx].get('pmid'):
                curated_pmids.add(all_records[existing_idx]['pmid'])
                logger.debug(f"Updated existing curated paper: {cleaned_doi}, collected PMID")

    logger.info(f"Processed {len(curated_data)} curated DOIs from CSV")
    logger.info(f"Collected {len(curated_pmids)} PMIDs for similar article search")

    initial_count = len(all_records)

    # Keyword search
    logger.info("--- Keyword Search ---")
    new_articles = search_pubmed('keywords', keywords, existing_dois, 'keyword_search',
                                 email, date_filter)
    all_records.extend(new_articles)

    # Similar articles search (only if full mode or significant new papers)
    if mode == 'full' or len(new_articles) > 50:
        logger.info("--- Similar Articles Search ---")
        new_similar = search_pubmed('similar', curated_pmids, existing_dois,
                                    'similar_articles', email)
        all_records.extend(new_similar)

    # Save results
    df_final = pd.DataFrame(all_records)
    if 'doi' in df_final.columns:
        df_final['doi'] = df_final['doi'].apply(clean_doi)
        df_final.drop_duplicates(subset=['doi'], keep='last', inplace=True)

    ensure_dir(os.path.dirname(output_file))
    df_final.to_json(output_file, orient='records', lines=True)

    papers_added = len(df_final) - initial_count
    metadata.record_run(mode, papers_added)

    logger.info(f"Scraping complete. Total DOIs: {len(df_final)}, New: {papers_added}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PubMed for spatial omics literature")
    parser.add_argument("--paper_classification_cleaned_csv", type=str,
                        default=DEFAULT_PAPER_CLASSIFICATION_CLEANED_CSV)
    parser.add_argument("--output_scraped_articles_file", type=str,
                        default=DEFAULT_OUTPUT_SCRAPED_FILE)
    parser.add_argument("--metadata_file", type=str, default=DEFAULT_METADATA_FILE)
    parser.add_argument("--pubmed_email", type=str, required=True)
    parser.add_argument("--keywords", type=str, default=",".join(DEFAULT_KEYWORDS))
    parser.add_argument("--mode", type=str, choices=['full', 'incremental', 'date_range'],
                        default='incremental',
                        help="Scraping mode: full=everything, incremental=since last run, date_range=custom dates")
    parser.add_argument("--start_date", type=str,
                        help="Start date for date_range mode (format: YYYY/MM/DD)")
    parser.add_argument("--end_date", type=str,
                        help="End date for date_range mode (format: YYYY/MM/DD)")

    args = parser.parse_args()
    keywords = [kw.strip() for kw in args.keywords.split(',') if kw.strip()]

    main_scraper(
        args.paper_classification_cleaned_csv,
        args.output_scraped_articles_file,
        args.metadata_file,
        args.pubmed_email,
        keywords,
        args.mode,
        args.start_date,
        args.end_date
    )



# # Make sure directories exist
# mkdir -p data/raw data/logs/scraper_logs data/inputs
#
# # Run full scrape
# python3 src/data_acquisition/pubmed_scraper_new.py --pubmed_email "id9417@alunos.uminho.pt" --mode full 2>&1 | tee logs/full_scrape_$(date +%Y%m%d).log
# # The 2>&1 | tee logs/... part saves all output to a log file so you can check progress later.
#
#
#
# # In another terminal, watch the logs
# tail -f data/logs/scraper_logs/*.log
#
# # Or check how many papers so far
# wc -l data/raw/scraped_pubmed_articles.jsonl

#
# python3 << 'EOF'
# import pandas as pd
# import json
#
# df = pd.read_json('data/raw/scraped_pubmed_articles.jsonl', lines=True)
#
# print("=" * 60)
# print("PUBMED SCRAPING SUMMARY")
# print("=" * 60)
# print(f"\nTotal papers scraped: {len(df)}")
# print(f"Unique DOIs: {df['doi'].nunique()}")
# print(f"Papers with DOI: {df['doi'].notna().sum()}")
# print(f"Papers without DOI: {df['doi'].isna().sum()}")
#
# print("\n--- Papers by Year ---")
# print(df['year'].value_counts().sort_index().tail(10))
#
# print("\n--- Papers by Source ---")
# print(df['found_via'].value_counts())
#
# print("\n--- Papers by Status ---")
# print(df['status'].value_counts())
#
# print("\n--- Top Journals ---")
# print(df['journal'].value_counts().head(10))
#
# print("\n--- Sample Paper ---")
# sample = df.iloc[0]
# print(f"Title: {sample['title'][:80]}...")
# print(f"DOI: {sample['doi']}")
# print(f"Year: {sample['year']}")
# print(f"Journal: {sample['journal']}")
#
# print("\n--- Data Quality ---")
# print(f"Papers with abstract: {df['abstract'].notna().sum()} ({df['abstract'].notna().sum()/len(df)*100:.1f}%)")
# print(f"Papers with authors: {df['authors'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()}")
# print(f"Papers with MeSH terms: {df['mesh_terms'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()}")
#
# print("=" * 60)
# EOF