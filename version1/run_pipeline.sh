#!/bin/bash
# run_pipeline.sh - Complete automated living review pipeline
# This will eventually orchestrate: scraping → preprocessing → analysis → visualization
# For now: only scraping is implemented

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT="/home/martinha/PycharmProjects/phd/live_review"
PYTHON_BIN="$(which python3)"
EMAIL="id9417@alunos.uminho.pt"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

# Default mode (can be overridden)
RUN_MODE="${1:-incremental}"  # Use first argument or default to incremental. It will run from the last scrap date

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log_step() {
    echo "" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    log_message "$1"
    echo "========================================" | tee -a "$MAIN_LOG"
}

check_success() {
    if [ $? -eq 0 ]; then
        log_message "✓ $1 completed"
    else
        log_message "✗ $1 failed"
        return 1
    fi
}

# ============================================================================
# SETUP
# ============================================================================

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/data/raw"
mkdir -p "$PROJECT_ROOT/data/processed"
mkdir -p "$PROJECT_ROOT/data/reports"

cd "$PROJECT_ROOT"

log_step "Starting Living Review Pipeline"
log_message "Mode: $RUN_MODE"
log_message "Email: $EMAIL"
log_message "Timestamp: $TIMESTAMP"

# ============================================================================
# STEP 1: DATA ACQUISITION - PubMed
# ============================================================================

log_step "STEP 1: Scraping PubMed"

$PYTHON_BIN src/data_acquisition/pubmed_scraper_new.py \
    --pubmed_email "$EMAIL" \
    --mode "$RUN_MODE" \
    2>&1 | tee -a "$MAIN_LOG"

check_success "PubMed scraping" || exit 1

if [ -f "data/raw/scraped_pubmed_articles.jsonl" ]; then
    PUBMED_COUNT=$(wc -l < data/raw/scraped_pubmed_articles.jsonl)
    log_message "Total PubMed papers: $PUBMED_COUNT"
fi

# ============================================================================
# STEP 2: DATA ACQUISITION - BioRxiv/MedRxiv
# ============================================================================

log_step "STEP 2: Scraping BioRxiv/MedRxiv"

if [ -f "src/data_acquisition/biorxiv_scraper.py" ]; then
    $PYTHON_BIN src/data_acquisition/biorxiv_scraper.py \
        --mode "$RUN_MODE" \
        --keywords "spatial transcriptomics,spatial omics,Visium,MERFISH,Xenium" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "BioRxiv scraping" || log_message "⚠ BioRxiv failed, continuing..."

    if [ -f "data/raw/scraped_biorxiv_articles.jsonl" ]; then
        BIORXIV_COUNT=$(wc -l < data/raw/scraped_biorxiv_articles.jsonl)
        log_message "Total BioRxiv papers: $BIORXIV_COUNT"
    fi
else
    log_message "⚠ BioRxiv scraper not found yet, skipping..."
fi

# ============================================================================
# STEP 3: PREPROCESSING - Merge & Deduplicate
# TODO: Implement merge_sources.py
# ============================================================================

log_step "STEP 3: Merging and Deduplicating [TODO]"

if [ -f "src/preprocessing/merge_sources.py" ]; then
    $PYTHON_BIN src/preprocessing/merge_sources.py \
        --pubmed "data/raw/scraped_pubmed_articles.jsonl" \
        --biorxiv "data/raw/scraped_biorxiv_articles.jsonl" \
        --output "data/processed/merged_papers.jsonl" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "Merging sources" || log_message "⚠ Merge failed, continuing..."
else
    log_message "⚠ Merge script not implemented yet, skipping..."
fi

# ============================================================================
# STEP 4: PREPROCESSING - Data Cleaning
# TODO: Implement clean_data.py
# ============================================================================

log_step "STEP 4: Data Cleaning [TODO]"

if [ -f "src/preprocessing/clean_data.py" ]; then
    $PYTHON_BIN src/preprocessing/clean_data.py \
        --input "data/processed/merged_papers.jsonl" \
        --output "data/processed/cleaned_papers.jsonl" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "Data cleaning" || log_message "⚠ Cleaning failed, continuing..."
else
    log_message "⚠ Cleaning script not implemented yet, skipping..."
fi

# ============================================================================
# STEP 5: CLASSIFICATION - Categorize Papers
# TODO: Implement classify_papers.py
# ============================================================================

log_step "STEP 5: Classifying Papers [TODO]"

if [ -f "src/analysis/classify_papers.py" ]; then
    $PYTHON_BIN src/analysis/classify_papers.py \
        --input "data/processed/cleaned_papers.jsonl" \
        --output "data/processed/classified_papers.jsonl" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "Classification" || log_message "⚠ Classification failed, continuing..."
else
    log_message "⚠ Classification script not implemented yet, skipping..."
fi

# ============================================================================
# STEP 6: ANALYSIS - Extract Methods & Datasets
# TODO: Implement extract_methods.py
# ============================================================================

log_step "STEP 6: Extracting Methods & Datasets [TODO]"

if [ -f "src/analysis/extract_methods.py" ]; then
    $PYTHON_BIN src/analysis/extract_methods.py \
        --input "data/processed/classified_papers.jsonl" \
        --output "data/processed/methods_catalog.json" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "Method extraction" || log_message "⚠ Extraction failed, continuing..."
else
    log_message "⚠ Method extraction script not implemented yet, skipping..."
fi

# ============================================================================
# STEP 7: STATISTICS - Generate Reports
# TODO: Implement generate_stats.py
# ============================================================================

log_step "STEP 7: Generating Statistics [TODO]"

if [ -f "src/analysis/generate_stats.py" ]; then
    $PYTHON_BIN src/analysis/generate_stats.py \
        --input "data/processed/classified_papers.jsonl" \
        --output "data/reports/stats_${TIMESTAMP}.json" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "Statistics generation" || log_message "⚠ Stats failed, continuing..."
else
    log_message "⚠ Stats script not implemented yet, skipping..."
fi

# ============================================================================
# STEP 8: VISUALIZATION - Generate Plots
# TODO: Implement generate_plots.py
# ============================================================================

log_step "STEP 8: Generating Visualizations [TODO]"

if [ -f "src/visualization/generate_plots.py" ]; then
    $PYTHON_BIN src/visualization/generate_plots.py \
        --input "data/processed/classified_papers.jsonl" \
        --output_dir "data/reports/figures" \
        2>&1 | tee -a "$MAIN_LOG"

    check_success "Visualization" || log_message "⚠ Visualization failed, continuing..."
else
    log_message "⚠ Visualization script not implemented yet, skipping..."
fi

# ============================================================================
# STEP 9: VERSION CONTROL - Commit Changes (Optional)
# ============================================================================

log_step "STEP 9: Version Control"

if git rev-parse --git-dir > /dev/null 2>&1; then
    if git diff --quiet data/ 2>/dev/null; then
        log_message "No changes to commit"
    else
        log_message "Committing changes to git..."
        git add data/raw/*.jsonl data/processed/*.jsonl data/reports/*.json 2>/dev/null || true
        git commit -m "Automated update: $(date +'%Y-%m-%d %H:%M')" 2>&1 | tee -a "$MAIN_LOG" || true
        log_message "✓ Changes committed locally"

        # Uncomment to auto-push to remote:
        # git push origin main 2>&1 | tee -a "$MAIN_LOG"
    fi
else
    log_message "⚠ Not a git repository, skipping version control"
fi

# ============================================================================
# CLEANUP & SUMMARY
# ============================================================================

log_step "Pipeline Summary"

log_message "Completed steps:"
log_message "  ✓ PubMed scraping"
[ -f "src/data_acquisition/biorxiv_scraper.py" ] && log_message "  ✓ BioRxiv scraping"
log_message ""
log_message "Pending steps (not implemented yet):"
log_message "  ⚠ Merge & deduplicate"
log_message "  ⚠ Data cleaning"
log_message "  ⚠ Paper classification"
log_message "  ⚠ Method extraction"
log_message "  ⚠ Statistics generation"
log_message "  ⚠ Visualization"
log_message ""
log_message "Pipeline finished at $(date)"
log_message "Full log: $MAIN_LOG"

# Clean up old logs (keep last 30 days)
find "$LOG_DIR" -name "pipeline_*.log" -mtime +30 -delete 2>/dev/null || true

echo ""
echo "========================================" | tee -a "$MAIN_LOG"
echo "Pipeline Complete!" | tee -a "$MAIN_LOG"
echo "Check log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"