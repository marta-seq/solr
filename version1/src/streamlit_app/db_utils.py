import pandas as pd
import os
import json
import streamlit as st # Import streamlit for caching
import ast
# --- Configuration for Data Paths ---
# Define paths relative to the script's location (src/streamlit_app/db_utils.py)
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

# Corrected filename as per your last message
PAPERS_CSV_PATH = os.path.join(BASE_DATA_DIR, 'database', 'processed_final_deploy.csv')
GRAPH_DATA_JSON_PATH = os.path.join(BASE_DATA_DIR, 'methods', 'graph_data.json')
INTERNAL_DATASETS_XLSX_PATH = os.path.join(BASE_DATA_DIR, 'inputs', 'internal_datasets.xlsx')

# --- Data Loading Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_raw_papers_data(delimiter=',') -> pd.DataFrame: # Added delimiter parameter
    """
    Loads raw papers data from 'processed_final_deploy.csv'.
    Includes enhanced debugging and delimiter option.
    """
    print(f"DEBUG: Attempting to load papers from absolute path: {PAPERS_CSV_PATH} with delimiter: '{delimiter}'")

    if not os.path.exists(PAPERS_CSV_PATH):
        st.error(f"Error: '{os.path.basename(PAPERS_CSV_PATH)}' not found at {PAPERS_CSV_PATH}. Please ensure the file exists and the path is correct.")
        print(f"DEBUG: File NOT found at {PAPERS_CSV_PATH}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(PAPERS_CSV_PATH, delimiter=delimiter) # Use the passed delimiter
        print(f"DEBUG: Successfully read CSV. DataFrame is empty: {df.empty}")
        if df.empty:
            st.warning(f"Warning: The CSV file at {PAPERS_CSV_PATH} was read, but the resulting DataFrame is empty.")
            return pd.DataFrame()

        print(f"DEBUG: Columns loaded: {df.columns.tolist()}")
        print(f"DEBUG: First 5 rows of loaded data:\n{df.head().to_string()}")

        # Ensure 'year' is numeric and handle potential NaNs for plotting
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        else:
            print("DEBUG: 'year' column not found in loaded DataFrame.")

        # Ensure dates are datetime objects for plotting
        for col in ['scrape_date', 'full_text_extraction_timestamp', 'llm_annotation_timestamp', 'MC_Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                print(f"DEBUG: '{col}' column not found in loaded DataFrame.")

        print(f"Successfully loaded {len(df)} papers from {PAPERS_CSV_PATH}")
        return df
    except pd.errors.EmptyDataError:
        st.error(f"Error: The CSV file at {PAPERS_CSV_PATH} is empty.")
        print(f"DEBUG: EmptyDataError: CSV file is empty.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading papers data from CSV: {e}. Please check the CSV format and content.")
        print(f"DEBUG: General Exception during CSV loading: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_internal_datasets_data() -> pd.DataFrame:
    """Loads internal datasets data from 'internal_datasets.xlsx'."""
    print(f"DEBUG: Attempting to load internal datasets from absolute path: {INTERNAL_DATASETS_XLSX_PATH}")
    if not os.path.exists(INTERNAL_DATASETS_XLSX_PATH):
        st.error(f"Error: 'internal_datasets.xlsx' not found at {INTERNAL_DATASETS_XLSX_PATH}. Please ensure it's in data/inputs/")
        print(f"DEBUG: File NOT found at {INTERNAL_DATASETS_XLSX_PATH}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(INTERNAL_DATASETS_XLSX_PATH)
        print(f"DEBUG: Successfully read XLSX. DataFrame is empty: {df.empty}")
        if df.empty:
            st.warning(f"Warning: The XLSX file at {INTERNAL_DATASETS_XLSX_PATH} was read, but the resulting DataFrame is empty.")
            return pd.DataFrame()
        print(f"DEBUG: Columns loaded for internal datasets: {df.columns.tolist()}")

        # Ensure 'year', 'n_patients', 'n_samples', 'n_regions' are numeric
        for col in ['year', 'n_patients', 'n_samples', 'n_regions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            else:
                print(f"DEBUG: '{col}' column not found in loaded internal datasets DataFrame.")
        print(f"Successfully loaded {len(df)} internal datasets from {INTERNAL_DATASETS_XLSX_PATH}")
        return df
    except Exception as e:
        st.error(f"Error loading internal datasets data from XLSX: {e}. Please check the XLSX format.")
        print(f"DEBUG: General Exception during XLSX loading: {e}")
        return pd.DataFrame()
def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a list or converts a comma-separated
    string into a list. Handles cases where the input is already a list or non-string.
    This is crucial because JSON fields might store lists as strings (e.g., "['item1', 'item2']")
    and we need to parse them into actual Python lists for filtering.
    """
    if isinstance(val, str):
        try:
            evaluated = ast.literal_eval(val)
            if isinstance(evaluated, list):
                return evaluated
            else:
                # If it's a string that's not a list representation, treat as a single item list
                return [str(evaluated)]
        except (ValueError, SyntaxError):
            # Fallback for comma-separated strings that aren't valid literal_eval
            return [item.strip() for item in val.split(',') if item.strip()]
    elif isinstance(val, list):
        return val
    return [] # Return empty list for NaN or other unexpected types

@st.cache_data(ttl=3600)
def get_categorized_papers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw papers data and categorizes them into computational
    and non-computational papers based on the 'is_computational' column.
    """
    df_papers_raw = load_raw_papers_data() # Now defaults to comma, can be overridden if needed

    if df_papers_raw.empty:
        print("DEBUG: No raw papers data available for categorization (DataFrame was empty).")
        return pd.DataFrame(), pd.DataFrame()

    # Rule for distinguishing computational papers: directly use 'is_computational' column
    if 'is_computational' in df_papers_raw.columns:
        df_computational = df_papers_raw[df_papers_raw['is_computational'] == True].copy()
        df_non_computational = df_papers_raw[df_papers_raw['is_computational'] == False].copy()
        print(f"DEBUG: Categorized {len(df_computational)} computational papers and {len(df_non_computational)} non-computational papers.")
    else:
        st.warning("Column 'is_computational' not found in papers data. All papers treated as non-computational for categorization.")
        print("DEBUG: 'is_computational' column NOT found in DataFrame during categorization step.")
        df_computational = pd.DataFrame() # No computational papers if column is missing
        df_non_computational = df_papers_raw.copy() # All papers are non-computational by default

    return df_computational, df_non_computational

@st.cache_data # Consider caching this if the input DataFrame doesn't change often
def get_exploded_counts(df, column_name):
    """
    Explodes a column containing lists, counts the occurrences of each item,
    and returns a DataFrame suitable for Plotly bar charts.
    Ensures the output DataFrame has 'Count' and a display-friendly column name.
    """
    # Define a display-friendly name for the category column
    display_column_name = column_name.replace('kw_', '').replace('llm_annot_', '').replace('_', ' ').title()

    if df.empty or column_name not in df.columns or df[column_name].isnull().all():
        # Return an empty DataFrame with the expected column names
        return pd.DataFrame(columns=[display_column_name, 'Count'])

    df_copy = df.copy()
    # Ensure the column contains actual lists; this is a safeguard, ideally done during initial load
    df_copy[column_name] = df_copy[column_name].apply(safe_literal_eval)

    # Filter out empty lists before exploding to avoid empty strings/NaNs from explode
    exploded_series = df_copy[column_name].explode()

    # Drop any potential NaN values that might result from empty lists or None after explode
    exploded_series = exploded_series.dropna()

    if exploded_series.empty:
        # Return an empty DataFrame with the expected column names if no valid data after explode
        return pd.DataFrame(columns=[display_column_name, 'Count'])

    counts = exploded_series.value_counts().reset_index()
    counts.columns = [display_column_name, 'Count']  # Rename columns for Plotly Express

    # Ensure the 'Count' column is numeric
    counts['Count'] = pd.to_numeric(counts['Count'], errors='coerce').fillna(0)

    return counts.sort_values(by='Count', ascending=False)

@st.cache_data # Consider caching this if graph data doesn't change often
def load_graph_data() -> dict | None:
    """
    Loads graph data from 'graph_data.json'.
    """
    print(f"DEBUG: Attempting to load graph data from absolute path: {GRAPH_DATA_JSON_PATH}")
    if not os.path.exists(GRAPH_DATA_JSON_PATH):
        st.error(f"Error: 'graph_data.json' not found at {GRAPH_DATA_JSON_PATH}. Please run generate_graph_data.py first.")
        print(f"DEBUG: File NOT found at {GRAPH_DATA_JSON_PATH}")
        return None
    try:
        with open(GRAPH_DATA_JSON_PATH, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        print(f"Successfully loaded graph data from {GRAPH_DATA_JSON_PATH}")
        return graph_data
    except FileNotFoundError:
        print(f"DEBUG: FileNotFoundError: graph_data.json not found at {GRAPH_DATA_JSON_PATH}")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {GRAPH_DATA_JSON_PATH}. Check file format.")
        print(f"DEBUG: JSONDecodeError: Could not decode JSON from {GRAPH_DATA_JSON_PATH}.")
        return None
    except Exception as e:
        st.error(f"Error loading graph data: {e}")
        print(f"DEBUG: General Exception during graph data loading: {e}")
        return None

