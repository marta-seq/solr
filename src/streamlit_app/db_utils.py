import pandas as pd
import os
import json
import streamlit as st # Import streamlit for caching

# --- Configuration for Data Paths ---
# Define paths relative to the script's location (src/streamlit_app/db_utils.py)
# This assumes:
# /home/martinha/PycharmProjects/phd/review/
# ├── src/
# │   └── streamlit_app/
# │       └── db_utils.py
# └── data/
#     ├── database/
#     │   └── processed_final.csv  <--- Your main papers data
#     ├── methods/
#     │   └── graph_data.json      <--- Your graph data
#     └── inputs/
#         └── internal_datasets.xlsx <--- Your internal datasets
#
# So, to go from 'src/streamlit_app' to 'data', we go up one level (..) then into 'data'
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

PAPERS_CSV_PATH = os.path.join(BASE_DATA_DIR, 'database', 'processed_final.csv') # Corrected filename and path
GRAPH_DATA_JSON_PATH = os.path.join(BASE_DATA_DIR, 'methods', 'graph_data.json') # Corrected path
INTERNAL_DATASETS_XLSX_PATH = os.path.join(BASE_DATA_DIR, 'inputs', 'internal_datasets.xlsx') # Added for completeness

# --- Data Loading Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_raw_papers_data() -> pd.DataFrame:
    """
    Loads raw papers data from 'processed_final.csv'.
    """
    if not os.path.exists(PAPERS_CSV_PATH):
        st.error(f"Error: 'processed_final.csv' not found at {PAPERS_CSV_PATH}. Please ensure the file exists and the path is correct.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(PAPERS_CSV_PATH)
        # Ensure 'year' is numeric and handle potential NaNs for plotting
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        # Ensure dates are datetime objects for plotting
        for col in ['scrape_date', 'full_text_extraction_timestamp', 'llm_annotation_timestamp', 'MC_Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"Successfully loaded {len(df)} papers from {PAPERS_CSV_PATH}")
        return df
    except Exception as e:
        st.error(f"Error loading papers data from CSV: {e}. Please check the CSV format.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_internal_datasets_data() -> pd.DataFrame:
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
        print(f"Successfully loaded {len(df)} internal datasets from {INTERNAL_DATASETS_XLSX_PATH}")
        return df
    except Exception as e:
        st.error(f"Error loading internal datasets data from XLSX: {e}. Please check the XLSX format.")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_categorized_papers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw papers data and categorizes them into computational
    and non-computational papers based on the 'is_computational' column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing
        (df_computational_papers, df_non_computational_papers).
        Returns empty DataFrames if raw data cannot be loaded or categorized.
    """
    df_papers_raw = load_raw_papers_data()

    if df_papers_raw.empty:
        print("No raw papers data available for categorization.")
        return pd.DataFrame(), pd.DataFrame()

    # Rule for distinguishing computational papers: directly use 'is_computational' column
    if 'is_computational' in df_papers_raw.columns:
        df_computational = df_papers_raw[df_papers_raw['is_computational'] == True].copy()
        df_non_computational = df_papers_raw[df_papers_raw['is_computational'] == False].copy()
        print(f"Categorized {len(df_computational)} computational papers and {len(df_non_computational)} non-computational papers.")
    else:
        st.warning("Column 'is_computational' not found in papers data. All papers treated as non-computational for categorization.")
        df_computational = pd.DataFrame() # No computational papers if column is missing
        df_non_computational = df_papers_raw.copy() # All papers are non-computational by default

    return df_computational, df_non_computational

@st.cache_data # Consider caching this if the input DataFrame doesn't change often
def get_exploded_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Explodes a DataFrame column (assuming it contains lists/strings of items)
    and counts the occurrences of each item.
    """
    if df.empty or column not in df.columns:
        print(f"Warning: DataFrame is empty or column '{column}' not found for exploding counts.")
        return pd.DataFrame(columns=['item', 'count'])

    # Attempt to convert string representation of lists to actual lists
    # This is a common issue with CSVs where lists are saved as "['item1', 'item2']"
    # Use ast.literal_eval for safer evaluation if strings are valid Python literals
    # Otherwise, simple split might be needed. For now, assume simple split for comma-separated.
    # If your columns like kw_tissue, kw_disease are actual string representations of lists,
    # you might need: df_copy[column] = df_copy[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    # For now, sticking to .str.split(', ') as it was in previous versions.
    exploded_series = df[column].dropna().astype(str).str.split(', ').explode()

    if exploded_series.empty:
        print(f"No data found in '{column}' after processing for exploded counts.")
        return pd.DataFrame(columns=['item', 'count'])

    counts = exploded_series.value_counts().reset_index()
    counts.columns = ['item', 'count']
    return counts

@st.cache_data # Consider caching this if graph data doesn't change often
def load_graph_data() -> dict | None:
    """
    Loads graph data from 'graph_data.json'.
    """
    if not os.path.exists(GRAPH_DATA_JSON_PATH):
        st.error(f"Error: 'graph_data.json' not found at {GRAPH_DATA_JSON_PATH}. Please run generate_graph_data.py first.")
        return None
    try:
        with open(GRAPH_DATA_JSON_PATH, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        print(f"Successfully loaded graph data from {GRAPH_DATA_JSON_PATH}")
        return graph_data
    except FileNotFoundError:
        print(f"Error: graph_data.json not found at {GRAPH_DATA_JSON_PATH}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {GRAPH_DATA_JSON_PATH}. Check file format.")
        return None
    except Exception as e:
        st.error(f"Error loading graph data: {e}")
        return None

