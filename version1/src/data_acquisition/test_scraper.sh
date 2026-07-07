#!/bin/bash
# test_scraper.sh - Test the PubMed scraper with limited scope

set -e

PROJECT_ROOT="/home/martinha/PycharmProjects/phd/review"
cd "$PROJECT_ROOT"

echo "=== Testing PubMed Scraper ==="
echo ""

# Test 1: Check if script exists
echo "Test 1: Checking if scraper exists..."
if [ -f "src/data_acquisition/pubmed_scraper.py" ]; then
    echo "✓ Scraper file found"
else
    echo "✗ Scraper file NOT found at src/data_acquisition/pubmed_scraper.py"
    exit 1
fi

# Test 2: Check Python dependencies
echo ""
echo "Test 2: Checking Python dependencies..."
python3 -c "from Bio import Entrez; import pandas as pd; import xml.etree.ElementTree as ET" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All dependencies available"
else
    echo "✗ Missing dependencies. Install with:"
    echo "  pip install biopython pandas"
    exit 1
fi

# Test 3: Check directory structure
echo ""
echo "Test 3: Checking directory structure..."
mkdir -p data/raw data/logs/scraper_logs data/inputs
echo "✓ Directories created"

# Test 4: Run scraper with limited keywords (date range mode for safety)
echo ""
echo "Test 4: Running scraper in test mode (last 7 days only)..."
echo "This should only fetch a few papers..."
echo ""

# Calculate date 7 days ago
START_DATE=$(date -d "7 days ago" +"%Y/%m/%d" 2>/dev/null || date -v-7d +"%Y/%m/%d")
END_DATE=$(date +"%Y/%m/%d")

python3 src/data_acquisition/pubmed_scraper.py \
    --pubmed_email "your@email.com" \
    --mode date_range \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --keywords "Visium,spatial transcriptomics"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Scraper ran successfully!"
    echo ""
    echo "Check the output:"
    echo "  - Scraped data: data/raw/scraped_pubmed_articles.jsonl"
    echo "  - Metadata: data/raw/scraping_metadata.json"
    echo "  - Logs: data/logs/scraper_logs/"

    # Show how many papers were found
    if [ -f "data/raw/scraped_pubmed_articles.jsonl" ]; then
        PAPER_COUNT=$(wc -l < data/raw/scraped_pubmed_articles.jsonl)
        echo ""
        echo "Papers scraped: $PAPER_COUNT"
        echo ""
        echo "First paper (sample):"
        head -n 1 data/raw/scraped_pubmed_articles.jsonl | python3 -m json.tool | head -n 20
    fi
else
    echo ""
    echo "✗ Scraper failed. Check logs in data/logs/scraper_logs/"
    exit 1
fi

echo ""
echo "=== All tests passed! ==="
echo ""
echo "Next steps:"
echo "1. Review the scraped data in data/raw/scraped_pubmed_articles.jsonl"
echo "2. Try running with --mode incremental"
echo "3. Set up the full automation pipeline"