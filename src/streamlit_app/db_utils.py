import pandas as pd
import os
import json
import streamlit as st # Import streamlit for caching

# Define paths relative to the script's location
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, '..', 'data')
RAW_PAPERS_PATH = os.path.join(DATA_DIR, 'raw_papers.csv')
GRAPH_DATA_PATH = os.path.join(DATA_DIR, 'graph_data.json')

@st.cache_data # Add caching back
def load_raw_papers_data() -> pd.DataFrame:
    """
    Loads raw papers data from a CSV file.
    Expects the CSV to have a 'category' column.
    """
    try:
        df = pd.read_csv(RAW_PAPERS_PATH)
        print(f"Successfully loaded raw papers data from {RAW_PAPERS_PATH}")
        return df
    except FileNotFoundError:
        print(f"Error: raw_papers.csv not found at {RAW_PAPERS_PATH}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading raw papers data: {e}")
        return pd.DataFrame()

@st.cache_data # Add caching back
def get_categorized_papers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw papers data and categorizes them into computational
    and non-computational papers based on the 'category' column.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing
        (df_computational_papers, df_non_computational_papers).
        Returns empty DataFrames if raw data cannot be loaded or categorized.
    """
    df_papers_raw = load_raw_papers_data()

    if df_papers_raw.empty:
        print("No raw papers data available for categorization.")
        return pd.DataFrame(), pd.DataFrame()

    # Ensure 'category' column exists
    if 'category' not in df_papers_raw.columns:
        print("Warning: 'category' column not found in raw papers data. Cannot categorize.")
        # If no category column, return all as computational for now, or handle as per app logic
        return df_papers_raw, pd.DataFrame()

    df_computational = df_papers_raw[df_papers_raw['category'] == 'Computational'].copy()
    df_non_computational = df_papers_raw[df_papers_raw['category'] == 'Non-Computational'].copy()

    print(f"Categorized {len(df_computational)} computational papers and {len(df_non_computational)} non-computational papers.")
    return df_computational, df_non_computational

@st.cache_data # Consider caching this if the input DataFrame doesn't change often
def get_exploded_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Explodes a DataFrame column (assuming it contains lists/strings of items)
    and counts the occurrences of each item.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=['item', 'count'])

    # Convert string representation of lists to actual lists
    # This handles cases where lists are stored as strings like "['item1', 'item2']"
    df_copy = df.copy()
    df_copy[column] = df_copy[column].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Explode the list column and count occurrences
    exploded_df = df_copy.explode(column)
    counts = exploded_df[column].value_counts().reset_index()
    counts.columns = ['item', 'count']
    return counts

@st.cache_data # Consider caching this if graph data doesn't change often
def load_graph_data() -> dict | None:
    """
    Loads graph data from a JSON file.
    """
    try:
        with open(GRAPH_DATA_PATH, 'r') as f:
            graph_data = json.load(f)
        print(f"Successfully loaded graph data from {GRAPH_DATA_PATH}")
        return graph_data
    except FileNotFoundError:
        print(f"Error: graph_data.json not found at {GRAPH_DATA_PATH}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {GRAPH_DATA_PATH}. Check file format.")
        return None
    except Exception as e:
        print(f"Error loading graph data: {e}")
        return None

