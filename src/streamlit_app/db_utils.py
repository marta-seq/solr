import pandas as pd
import json
import os

# --- Configuration for Data Paths ---
# Define the base directory for your data.
# This assumes your 'data' folder is a sibling to 'streamlit_app'
# For example:
# solr/
# └── src/
#     └── streamlit_app/
#     └── data/
#         └── methods/
#             └── graph_data.json
#         └── raw_papers.csv (or .json, etc.)

# Adjust this path if your 'data' directory is located elsewhere relative to db_utils.py
# If db_utils.py is in 'streamlit_app' and 'data' is also in 'streamlit_app':
# DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# If 'data' is one level up from 'streamlit_app':
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

RAW_PAPERS_PATH = os.path.join(DATA_DIR, "raw_papers.csv") # Example: assuming a CSV file
GRAPH_DATA_PATH = os.path.join(DATA_DIR, "methods", "graph_data.json")

# --- Mock Data Generation (for testing if you don't have actual files yet) ---
def _generate_mock_papers_data():
    """Generates a mock DataFrame for raw papers."""
    data = {
        'paper_id': [f'P{i:03d}' for i in range(1, 101)],
        'title': [f'Paper Title {i}' for i in range(1, 101)],
        'abstract': [f'Abstract for paper {i} discussing various topics.' for i in range(1, 101)],
        'category': ['Computational' if i % 2 == 0 else 'Non-Computational' for i in range(1, 101)],
        'keywords': [
            ['ML', 'AI', 'Algorithm'] if i % 3 == 0 else
            ['Survey', 'Review'] if i % 3 == 1 else
            ['Data', 'Analysis']
            for i in range(1, 101)
        ],
        'publication_year': [2020 + (i % 5) for i in range(1, 101)]
    }
    df = pd.DataFrame(data)
    # Save to CSV for persistent mock data
    os.makedirs(os.path.dirname(RAW_PAPERS_PATH), exist_ok=True)
    df.to_csv(RAW_PAPERS_PATH, index=False)
    print(f"Mock raw papers data saved to: {RAW_PAPERS_PATH}")
    return df

def _generate_mock_graph_data():
    """Generates mock graph data for methods."""
    mock_data = {
        "nodes": [
            {"id": "MethodA", "label": "Method A", "group": "Computational"},
            {"id": "MethodB", "label": "Method B", "group": "Non-Computational"},
            {"id": "MethodC", "label": "Method C", "group": "Computational"},
            {"id": "MethodD", "label": "Method D", "group": "Non-Computational"},
        ],
        "edges": [
            {"from": "MethodA", "to": "MethodC"},
            {"from": "MethodB", "to": "MethodD"},
            {"from": "MethodA", "to": "MethodB"},
        ]
    }
    os.makedirs(os.path.dirname(GRAPH_DATA_PATH), exist_ok=True)
    with open(GRAPH_DATA_PATH, 'w') as f:
        json.dump(mock_data, f, indent=4)
    print(f"Mock graph data saved to: {GRAPH_DATA_PATH}")
    return mock_data

# Call mock data generation if files don't exist (optional, for initial setup)
if not os.path.exists(RAW_PAPERS_PATH):
    _generate_mock_papers_data()
if not os.path.exists(GRAPH_DATA_PATH):
    _generate_mock_graph_data()

# --- Database Utility Functions ---

def load_raw_papers_data() -> pd.DataFrame:
    """
    Loads raw papers data from a specified source (e.g., CSV, JSON, database).

    Returns:
        pd.DataFrame: A DataFrame containing the raw papers data.
    """
    try:
        # Example: Loading from a CSV file
        df = pd.read_csv(RAW_PAPERS_PATH)
        print(f"Successfully loaded raw papers data from {RAW_PAPERS_PATH}")
        return df
    except FileNotFoundError:
        print(f"Error: Raw papers data file not found at {RAW_PAPERS_PATH}. Generating mock data.")
        return _generate_mock_papers_data()
    except Exception as e:
        print(f"An error occurred while loading raw papers data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def get_categorized_papers(category: str = None) -> pd.DataFrame:
    """
    Retrieves papers, optionally filtered by category.

    Args:
        category (str, optional): The category to filter by (e.g., 'Computational', 'Non-Computational').
                                  If None, all papers are returned.

    Returns:
        pd.DataFrame: A DataFrame of categorized papers.
    """
    papers_df = load_raw_papers_data()
    if not papers_df.empty and category:
        # Ensure 'category' column exists before filtering
        if 'category' in papers_df.columns:
            filtered_df = papers_df[papers_df['category'] == category].copy()
            print(f"Filtered papers for category: {category}. Found {len(filtered_df)} papers.")
            return filtered_df
        else:
            print("Warning: 'category' column not found in papers data. Returning all papers.")
            return papers_df
    print(f"Returning all {len(papers_df)} papers (no category filter or empty data).")
    return papers_df

def get_exploded_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Explodes a list-like column (e.g., 'keywords') in a DataFrame and counts occurrences.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to explode (expected to contain lists).

    Returns:
        pd.DataFrame: A DataFrame with counts of each item in the exploded column.
                      Columns: 'item', 'count'.
    """
    if df.empty or column not in df.columns:
        print(f"Warning: DataFrame is empty or column '{column}' not found for exploding.")
        return pd.DataFrame(columns=['item', 'count'])

    # Ensure the column contains iterable items (e.g., lists)
    # If your 'keywords' column is stored as a string representation of a list,
    # you might need to convert it first: df[column].apply(ast.literal_eval)
    # For this example, we assume it's already a list or can be iterated.
    exploded_series = df[column].explode()
    counts = exploded_series.value_counts().reset_index()
    counts.columns = ['item', 'count']
    print(f"Exploded counts for column '{column}': {len(counts)} unique items found.")
    return counts

def load_graph_data():
    """
    Loads graph data from a JSON file.

    Returns:
        dict: The loaded graph data (nodes and edges).
    """
    try:
        with open(GRAPH_DATA_PATH, 'r') as f:
            graph_data = json.load(f)
        print(f"Successfully loaded graph data from {GRAPH_DATA_PATH}")
        return graph_data
    except FileNotFoundError:
        print(f"Error: Graph data file not found at {GRAPH_DATA_PATH}. Generating mock data.")
        return _generate_mock_graph_data()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {GRAPH_DATA_PATH}. Check file format.")
        return {"nodes": [], "edges": []}
    except Exception as e:
        print(f"An error occurred while loading graph data: {e}")
        return {"nodes": [], "edges": []}

# Example Usage (for testing db_utils.py directly)
if __name__ == "__main__":
    print("\n--- Testing load_raw_papers_data ---")
    papers = load_raw_papers_data()
    print(papers.head())

    print("\n--- Testing get_categorized_papers (Computational) ---")
    comp_papers = get_categorized_papers(category='Computational')
    print(comp_papers.head())

    print("\n--- Testing get_exploded_counts (Keywords) ---")
    keyword_counts = get_exploded_counts(papers, 'keywords')
    print(keyword_counts.head())

    print("\n--- Testing load_graph_data ---")
    graph = load_graph_data()
    print(f"Graph nodes: {len(graph.get('nodes', []))}, edges: {len(graph.get('edges', []))}")
