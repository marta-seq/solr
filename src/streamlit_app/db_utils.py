# src/streamlit_app/db_utils.py
import streamlit as st
import pandas as pd
import os

# Define the base path for your project (assuming this script is in src/streamlit_app/)
BASE_PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define specific data paths relative to the BASE_PROJECT_PATH
PAPERS_CSV_PATH = os.path.join(BASE_PROJECT_PATH, 'data', 'database', 'processed_final_deploy.csv')
INTERNAL_DATASETS_XLSX_PATH = os.path.join(BASE_PROJECT_PATH, 'data', 'inputs', 'internal_datasets.xlsx')

# --- Data Loading Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_raw_papers_data():
    """Loads raw papers data from 'processed_final.csv'."""
    if not os.path.exists(PAPERS_CSV_PATH):
        st.error(f"Error: 'processed_final.csv' not found at {PAPERS_CSV_PATH}. Please ensure it's in data/database/")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PAPERS_CSV_PATH)
        # Ensure 'year' is numeric and handle potential NaNs for plotting
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        # Ensure dates are datetime objects for plotting
        for col in ['scrape_date', 'full_text_extraction_timestamp', 'llm_annotation_timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading papers data from CSV: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_internal_datasets_data():
    """Loads internal datasets data from 'internal_datasets.xlsx'."""
    if not os.path.exists(INTERNAL_DATASETS_XLSX_PATH):
        st.error(f"Error: 'internal_datasets.xlsx' not found at {INTERNAL_DATASETS_XLSX_PATH}. Please ensure it's in data/inputs/")
        return pd.DataFrame()
    try:
        df = pd.read_excel(INTERNAL_DATASETS_XLSX_PATH)
        # Ensure 'year', 'n_patients', 'n_samples', 'n_regions' are numeric
        for col in ['year', 'n_patients', 'n_samples', 'n_regions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        return df
    except Exception as e:
        st.error(f"Error loading internal datasets data from XLSX: {e}")
        return pd.DataFrame()

# --- Paper Categorization and Utility Functions ---

@st.cache_data(ttl=3600)
def get_categorized_papers():
    """
    Loads all papers and categorizes them into computational and non-computational
    based on the 'is_computational' column.
    """
    df_papers = load_raw_papers_data()
    if df_papers.empty:
        return pd.DataFrame(), pd.DataFrame() # Return two empty DataFrames

    # Rule for distinguishing computational papers: directly use 'is_computational' column
    if 'is_computational' in df_papers.columns:
        df_computational = df_papers[df_papers['is_computational'] == True].copy()
        df_non_computational = df_papers[df_papers['is_computational'] == False].copy()
    else:
        st.warning("Column 'is_computational' not found. All papers treated as non-computational for categorization.")
        df_computational = pd.DataFrame() # No computational papers if column is missing
        df_non_computational = df_papers.copy() # All papers are non-computational by default

    return df_computational, df_non_computational

def get_exploded_counts(df, column_name, title_prefix="Top"):
    """
    Helper function to explode a comma-separated column and return value counts.
    Handles missing columns gracefully.
    """
    if column_name not in df.columns:
        st.info(f"Column '{column_name}' not found for plotting.")
        return pd.DataFrame()

    # Dropna first, then convert to string, split, and explode
    exploded_series = df[column_name].dropna().astype(str).str.split(', ').explode()

    if exploded_series.empty:
        st.info(f"No data found in '{column_name