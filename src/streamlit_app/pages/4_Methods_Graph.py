import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import ast # For safely evaluating string representations of lists
import numpy as np
import colorsys # For generating distinct colors
import hashlib # For consistent color hashing
import json # For handling JSON specific errors

# --- Configuration ---
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Paper Network Explorer")

# Define the path to your processed data file.
# IMPORTANT: This is now set to 'graph_data.json' as per your reminder!
PROCESSED_DATA_FILE = 'data/methods/graph_data.json'

# --- Helper Functions ---

def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a list or converts a comma-separated
    string into a list. Handles cases where the input is already a list.
    This is often needed if JSON values that should be arrays are sometimes strings.
    """
    if isinstance(val, str):
        try:
            # Try to evaluate as a Python literal (e.g., "['a', 'b']")
            evaluated = ast.literal_eval(val)
            if isinstance(evaluated, list):
                return evaluated
            else:
                # If it's not a list after eval, treat as single item in a list
                return [str(evaluated)]
        except (ValueError, SyntaxError):
            # If literal_eval fails, try splitting by comma (common for simple strings)
            return [item.strip() for item in val.split(',') if item.strip()]
    elif isinstance(val, list):
        return val
    return [] # Default to empty list if not string or list

def get_hashed_color(text: str) -> str:
    """
    Generates a consistent hex color from a string using hashing.
    This helps ensure that the same category always gets the same color.
    """
    if not text:
        return "#CCCCCC" # Default grey for empty/None
    # Use MD5 hash to get a consistent integer from the string
    hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    # Use HSV to get distinct colors, then convert to RGB hex
    # Adjust hue, saturation, and value for better visual appeal and distinction
    hue = (hash_value % 360) / 360.0
    saturation = 0.7 # Keep saturation high for vivid colors
    value = 0.85 # Keep value high for brightness
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"#{r:02x}{b:02x}{g:02x}" # Adjusted to f"#{r:02x}{b:02x}{g:02x}" for slightly different color distribution

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the paper data from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the paper data.

    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    try:
        # --- MAJOR CHANGE: Reading JSON instead of CSV ---
        df = pd.read_json(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the JSON is in the correct directory.")
        st.stop() # Stop the app if file is not found
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Please check if it's a valid JSON file. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the JSON file: {e}")
        st.stop()

    # Define columns that are expected to be lists and apply safe_literal_eval
    # This is still important as some JSON fields might contain string representations of lists
    list_cols = [
        'kw_pipeline_category',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_tested_data_modalities',
        'llm_annot_compared_algorithms_packages',
        'similar_papers'
    ]
    for col in list_cols:
        if col in df.columns:
            # Apply safe_literal_eval only if the column contains strings that need parsing
            # If the JSON already provides actual lists, this will just return them as is.
            df[col] = df[col].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)
        else:
            # If a list column is missing, create an empty list column to prevent errors
            df[col] = [[] for _ in range(len(df))]
            st.warning(f"Column '{col}' not found in your data. It will be treated as empty.")


    # Ensure 'citations' column is numeric and handle potential missing values
    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df['citations'] = 0 # Default to 0 if 'citations' column is missing
        st.warning("Column 'citations' not found in your data. Node sizes will default to a minimum size.")

    # Ensure 'year' column is numeric
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    else:
        df['year'] = 2000 # Default year if missing
        st.warning("Column 'year' not found in your data. Defaulting to year 2000.")

    # Ensure 'doi', 'title', 'abstract' columns exist
    for col in ['doi', 'title', 'abstract']:
        if col not in df.columns:
            df[col] = '' # Default to empty string if missing
            st.warning(f"Column '{col}' not found in your data. It will be empty in display.")

    return df

df = load_data(PROCESSED_DATA_FILE)

# --- Sidebar Filters ---
st.sidebar.header("Filter Papers")

# Year Range Filter (between X and Y year)
# Check if df is empty before getting min/max year
if not df.empty and 'year' in df.columns:
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    selected_year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year) # Default to the full range
    )
else:
    st.sidebar.info("No data loaded or 'year' column missing to set year filter.")
    selected_year_range = (0, 9999) # Default wide range if no data

# Extract all unique values for multiselect filters
# Ensure sorting for consistent display
all_categories = sorted(list(set(item for sublist in df['kw_pipeline_category'] for item in sublist))) if 'kw_pipeline_category' in df.columns else []
all_assay_types = sorted(list(set(item for sublist in df['llm_annot_tested_assay_types_platforms'] for item in sublist))) if 'llm_annot_tested_assay_types_platforms' in df.columns else []
all_data_modalities = sorted(list(set(item for sublist in df['llm_annot_tested_data_modalities'] for item in sublist))) if 'llm_annot_tested_data_modalities' in df.columns else []

# Multiselect filters for categories, assay types, and data modalities
selected_categories = st.sidebar.multiselect(
    "Filter by Pipeline Category",
    options=all_categories,
    default=all_categories # Select all by default
)

selected_assay_types = st.sidebar.multiselect(
    "Filter by Assay Types/Platforms",
    options=all_assay_types,
    default=all_assay_types # Select all by default
)

selected_data_modalities = st.sidebar.multiselect(
    "Filter by Data Modalities",
    options=all_data_modalities,
    default=all_data_modalities # Select all by default
)

st.sidebar.header("Graph Options")
# Toggle for showing similarity edges
show_similarity_edges = st.sidebar.checkbox("Show Similarity Edges", value=True)

# --- Apply Filters ---
filtered_df = df[
    (df['year'] >= selected_year_range[0]) &
    (df['year'] <= selected_year_range[1])
]

# Apply category filter: paper included if ANY of its categories are selected
if selected_categories and 'kw_pipeline_category' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['kw_pipeline_category'].apply(lambda x: any(cat in selected_categories for cat in x))
    ]

# Apply assay types filter: paper included if ANY of its assay types are selected
if selected_assay_types and 'llm_annot_tested_assay_types_platforms' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['llm_annot_tested_assay_types_platforms'].apply(lambda x: any(assay in selected_assay_types for assay in x))
    ]

# Apply data modalities filter: paper included if ANY of its modalities are selected
if selected_data_modalities and 'llm_annot_tested_data_modalities' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['llm_annot_tested_data_modalities'].apply(lambda x: any(modality in selected_data_modalities for modality in x))
    ]

st.write(f"Displaying **{len(filtered_df)}** papers based on current filters.")

# --- Graph Visualization ---
if not filtered_df.empty:
    # Initialize pyvis network
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
    net.toggle_physics(True) # Enable physics for better node layout and movement

    # Calculate min/max citation counts for node size normalization
    # Use 'citations' column for sizing
    max_citations = filtered_df['citations'].max() if 'citations' in filtered_df.columns else 0
    min_citations = filtered_df['citations'].min() if 'citations' in filtered_df.columns else 0

    # Create a color map for categories using the hashing function
    category_colors = {cat: get_hashed_color(cat) for cat in all_categories}

    # Add nodes to the graph
    for idx, row in filtered_df.iterrows():
        node_id = row['doi']
        title = row['title']
        citations = row['citations'] if 'citations' in row else 0 # Use 'citations' column
        year = row['year']
        # Convert lists to comma-separated strings for display in tooltip
        categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
        assay_types = ", ".join(row['llm_annot_tested_assay_types_platforms']) if row['llm_annot_tested_assay_types_platforms'] else "N/A"
        data_modalities = ", ".join(row['llm_annot_tested_data_modalities']) if row['llm_annot_tested_data_modalities'] else "N/A"
        abstract = row['abstract']

        # Node Size: Logarithmic scaling for better visual distinction
        # Add a small constant (1) to citations to handle log(0) and ensure minimum size
        # Scale the size to a reasonable range (e.g., 10 to 40)
        if max_citations > 0:
            size = 10 + (np.log1p(citations) / np.log1p(max_citations + 1)) * 30
        else:
            size = 10 # Default minimum size if no citations or all are zero

        # Node Color: Based on the first category for the node's main color.
        node_color = "#CCCCCC" # Default grey if no categories
        if row['kw_pipeline_category']:
            node_color = category_colors.get(row['kw_pipeline_category'][0], "#CCCCCC")

        # Tooltip for node details (displayed on hover)
        tooltip = f"""
        <b>Title:</b> {title}<br>
        <b>DOI:</b> {node_id}<br>
        <b>Citations:</b> {citations}<br>
        <b>Year:</b> {year}<br>
        <b>Categories:</b> {categories}<br>
        <b>Assay Types/Platforms:</b> {assay_types}<br>
        <b>Data Modalities:</b> {data_modalities}<br>
        <b>Abstract:</b> {abstract[:300]}...
        """

        net.add_node(
            node_id,
            label=title,
            title=tooltip,
            size=size,
            color=node_color,
            font={'color': 'black', 'size': 12},
            borderWidth=1,
            borderWidthSelected=3,
            shape='dot'
        )

    # Add edges based on similarity if the toggle is enabled
    if show_similarity_edges and 'similar_papers' in filtered_df.columns:
        for idx, row in filtered_df.iterrows():
            source_doi = row['doi']
            # Ensure the source node is in the current graph (i.e., it passed filters)
            if source_doi in [node['id'] for node in net.nodes]:
                for target_doi in row['similar_papers']:
                    # Ensure the target node also exists in the filtered DataFrame and the graph
                    if target_doi in filtered_df['doi'].values and target_doi in [node['id'] for node in net.nodes]:
                        net.add_edge(source_doi, target_doi, title="Similarity", color="#888888", width=1.5)
    elif not show_similarity_edges:
        st.info("Similarity edges are currently turned off.")
    elif 'similar_papers' not in filtered_df.columns:
        st.warning("Cannot show similarity edges: 'similar_papers' column not found in your data.")


    # Generate and display the graph HTML
    try:
        path = "paper_network.html"
        net.save_graph(path)

        with open(path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()
        st.components.v1.html(html_content, height=780, scrolling=True)
    except Exception as e:
        st.error(f"An error occurred while rendering the graph: {e}")
        st.info("This might happen if there are no papers matching your filters to display, or if the 'doi' column is missing/empty.")

else:
    st.info("No papers match the selected filters. Please adjust your criteria in the sidebar.")

