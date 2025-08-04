import logging
import streamlit as st
import pandas as pd
import networkx as nx # Included for potential future use, not directly used by Pyvis for graph creation
from pyvis.network import Network
import ast # For safely evaluating string representations of lists
import numpy as np
import colorsys # For generating distinct colors
import hashlib # For consistent color hashing
import json # For handling JSON specific errors
import os

# Define the path to your processed data file.
# Ensure this path is correct relative to where your Streamlit app is run.
PROCESSED_DATA_FILE = 'data/methods/graph_data.json'

# --- Helper Functions ---

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

def get_hashed_color(text: str) -> str:
    """
    Generates a consistent hex color from a string using hashing.
    This ensures that the same category always has the same color across the graph.
    """
    if not text:
        return "#CCCCCC" # Default color for empty string/missing category
    hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    # Use HSV to get more distinct colors
    hue = (hash_value % 360) / 360.0
    saturation = 0.7 # Keep saturation high for vivid colors
    value = 0.85     # Keep value high for brightness
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"#{r:02x}{g:02x}{b:02x}"

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the paper data from a JSON file.
    The JSON file is expected to contain a dictionary with 'nodes' and 'edges' keys.
    'nodes' should point to a list of dictionaries (for paper data).
    'edges' should point to a list of dictionaries (for explicit graph edges).
    Applies safe_literal_eval to list-like columns and handles missing columns.

    Returns:
        tuple: A tuple containing two Pandas DataFrames (nodes_df, edges_df).
               Stops Streamlit execution if file cannot be loaded or essential keys are missing.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_json_data = json.load(f) # Load the entire JSON dictionary
            logging.debug(f"Loaded JSON data from '{file_path}'. Keys: {full_json_data.keys()}")

        # Create DataFrame for nodes from the 'nodes' key
        if 'nodes' in full_json_data and isinstance(full_json_data['nodes'], list):
            df_nodes = pd.DataFrame(full_json_data['nodes'])
            df_nodes['id'] = df_nodes['id'].astype(str) # Ensure 'doi' is a string for node IDs
            df_nodes['doi'] = df_nodes['id'].astype(str) # Ensure 'doi' is a string for node IDs
            logging.debug(f"Nodes DataFrame created with shape: {df_nodes.shape}.")
            logging.debug(f"Columns in nodes DataFrame: {df_nodes.columns.tolist()}.")
        else:
            st.error(f"Error: JSON data from '{file_path}' does not contain a 'nodes' key or 'nodes' is not a list.")
            st.stop()

        # Create DataFrame for edges from the 'edges' key (if present and valid)
        df_edges = pd.DataFrame() # Initialize as empty
        if 'links' in full_json_data and isinstance(full_json_data['links'], list):
            df_edges = pd.DataFrame(full_json_data['links'])
            logging.debug(f"Edges DataFrame created with shape: {df_edges.shape}.")
        else:
            st.warning(f"JSON data from '{file_path}' does not contain an 'edges' key or 'edges' is not a list. No explicit edges will be loaded from 'edges' key.")

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the JSON is in the correct directory.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Please check if it's a valid JSON file. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the JSON file: {e}")
        st.stop()

    # --- Preprocessing for df_nodes ---
    # Columns expected to contain lists (or string representations of lists)
    list_cols = [
        'kw_pipeline_category',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_tested_data_modalities',
        'llm_annot_compared_algorithms_packages',
        # 'similar_papers', # This column is now treated as a node property, not for direct edge drawing
        'kw_tissue'
    ]
    for col in list_cols:
        if col in df_nodes.columns:
            # Apply safe_literal_eval to ensure all are proper lists for filtering
            # Fill NaN/None with empty strings before applying safe_literal_eval
            df_nodes[col] = df_nodes[col].fillna('').apply(safe_literal_eval)
        else:
            # Add missing list columns as empty lists to prevent errors in filtering
            df_nodes[col] = [[] for _ in range(len(df_nodes))]
            st.warning(f"Column '{col}' not found in your nodes data. It will be treated as empty lists for filtering.")

    # Handle 'citations' column for node sizing
    if 'citations' in df_nodes.columns:
        df_nodes['citations'] = pd.to_numeric(df_nodes['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['citations'] = 0 # Default citations if column is missing
        st.warning("Column 'citations' not found in your nodes data. Node sizes will default to a minimum size.")

    # Handle 'year' column for filtering
    if 'year' in df_nodes.columns:
        df_nodes['year'] = pd.to_numeric(df_nodes['year'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['year'] = 2000 # Default year if column is missing
        st.warning("Column 'year' not found in your nodes data. Defaulting to year 2000 for all papers.")

    # Ensure 'doi', 'title', 'abstract' columns exist and are strings
    for col in ['doi', 'title', 'abstract']:
        if col not in df_nodes.columns:
            df_nodes[col] = ''
            st.warning(f"Column '{col}' not found in your nodes data. It will be empty in display.")
        # Ensure 'doi' is always a string, as it's used as the node ID
        if col == 'doi':
            df_nodes[col] = df_nodes[col].astype(str)

    return df_nodes, df_edges

# Load data when the page is run
# Now load both nodes and edges DataFrames
df_nodes, df_edges = load_data(PROCESSED_DATA_FILE)

# --- Streamlit Page Content ---

# Header for this specific page (appropriate for a multi-page app)
st.header("Paper Network Visualization")

# --- Sidebar Filters ---
st.sidebar.header("Filter Papers")

# Year Range Slider
if not df_nodes.empty and 'year' in df_nodes.columns and df_nodes['year'].min() != df_nodes['year'].max():
    min_year_data, max_year_data = int(df_nodes['year'].min()), int(df_nodes['year'].max())
    selected_year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year_data,
        max_value=max_year_data,
        value=(min_year_data, max_year_data)
    )
elif not df_nodes.empty and 'year' in df_nodes.columns:
    st.sidebar.info(f"Only one year ({int(df_nodes['year'].min())}) available in data. No range slider needed.")
    selected_year_range = (int(df_nodes['year'].min()), int(df_nodes['year'].max()))
else:
    st.sidebar.info("No data loaded or 'year' column missing to set year filter.")
    selected_year_range = (0, 9999) # Broad default range if no year data

# Get all unique values for multiselect filters from the entire dataset (df_nodes)
all_categories = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in sublist))) if 'kw_pipeline_category' in df_nodes.columns else []
all_assay_types = sorted(list(set(item for sublist in df_nodes['llm_annot_tested_assay_types_platforms'] for item in sublist))) if 'llm_annot_tested_assay_types_platforms' in df_nodes.columns else []
all_data_modalities = sorted(list(set(item for sublist in df_nodes['llm_annot_tested_data_modalities'] for item in sublist))) if 'llm_annot_tested_data_modalities' in df_nodes.columns else []

# Multiselect filters for list-based columns
selected_categories = st.sidebar.multiselect(
    "Filter by Pipeline Category",
    options=all_categories,
    default=all_categories # Default to all selected for broad initial view
)

selected_assay_types = st.sidebar.multiselect(
    "Filter by Assay Types/Platforms",
    options=all_assay_types,
    default=all_assay_types
)

selected_data_modalities = st.sidebar.multiselect(
    "Filter by Data Modalities",
    options=all_data_modalities,
    default=all_data_modalities
)

st.sidebar.header("Graph Options")
# Toggle for similarity edges
show_similarity_edges = st.sidebar.checkbox("Show Similarity Edges", value=True)

# --- Apply Filters ---
# Filter the nodes DataFrame first
filtered_nodes_df = df_nodes[
    (df_nodes['year'] >= selected_year_range[0]) &
    (df_nodes['year'] <= selected_year_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Apply multiselect filters: check if ANY selected item is in the paper's list
if selected_categories and 'kw_pipeline_category' in filtered_nodes_df.columns:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['kw_pipeline_category'].apply(lambda x: any(cat in selected_categories for cat in x))
    ]

if selected_assay_types and 'llm_annot_tested_assay_types_platforms' in filtered_nodes_df.columns:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['llm_annot_tested_assay_types_platforms'].apply(lambda x: any(assay in selected_assay_types for assay in x))
    ]

if selected_data_modalities and 'llm_annot_tested_data_modalities' in filtered_nodes_df.columns:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['llm_annot_tested_data_modalities'].apply(lambda x: any(modality in selected_data_modalities for modality in x))
    ]

st.write(f"Displaying **{len(filtered_nodes_df)}** papers based on current filters.")

# --- Graph Visualization ---
if not filtered_nodes_df.empty:
    # Initialize Pyvis network
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
    net.toggle_physics(True) # Enable physics for better layout

    # Calculate max citations for node sizing
    max_citations = filtered_nodes_df['citations'].max() if 'citations' in filtered_nodes_df.columns else 0
    # Add 1 to max_citations to avoid log(0) if max_citations is 0, and to scale better
    max_citations_for_scaling = max_citations + 1

    # Generate colors for ALL possible categories in the entire dataset to ensure consistency
    all_categories_for_colors = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in sublist)))
    category_colors = {cat: get_hashed_color(cat) for cat in all_categories_for_colors}

    # Add nodes to the network
    for idx, row in filtered_nodes_df.iterrows():
        node_id = row['doi'] # This is correctly using DOI as the node ID
        title = row['title']
        citations = row['citations'] if 'citations' in row else 0
        year = row['year']
        # Join lists for display in tooltip to show all values
        categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
        assay_types = ", ".join(row['llm_annot_tested_assay_types_platforms']) if row['llm_annot_tested_assay_types_platforms'] else "N/A"
        data_modalities = ", ".join(row['llm_annot_tested_data_modalities']) if row['llm_annot_tested_data_modalities'] else "N/A"
        abstract = row['abstract']

        # Node size based on citations (logarithmic scale for better visual distinction)
        # Scale size between a min (e.g., 10) and max (e.g., 40)
        if max_citations_for_scaling > 1: # Check if there's actual variation in citations
            size = 10 + (np.log1p(citations) / np.log1p(max_citations_for_scaling)) * 30
        else:
            size = 20 # Default size if all citations are 0 or constant

        # Node color based on the FIRST pipeline category (Pyvis nodes have one background color)
        node_color = "#CCCCCC" # Default grey if no category
        if row['kw_pipeline_category']:
            node_color = category_colors.get(row['kw_pipeline_category'][0], "#CCCCCC")

        # Tooltip content
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
            label=title, # Main label on the node
            title=tooltip, # Content for the hover tooltip
            size=size,
            color=node_color,
            font={'color': 'black', 'size': 12}, # Font color for the label
            borderWidth=1,
            borderWidthSelected=3,
            shape='dot' # Simple dot shape
        )

    # Add edges based on the df_edges DataFrame, filtered by current nodes
    if show_similarity_edges and not df_edges.empty:
        # Create a set of DOIs currently in the filtered graph for efficient lookup
        current_graph_dois = set(filtered_nodes_df['doi'].tolist())

        # Filter df_edges to only include edges where both source and target are in the current filtered nodes
        # Ensure 'source' and 'target' columns exist in df_edges
        if 'source' in df_edges.columns and 'target' in df_edges.columns:
            filtered_edges_to_draw = df_edges[
                (df_edges['source'].isin(current_graph_dois)) &
                (df_edges['target'].isin(current_graph_dois))
            ]

            for idx, edge_row in filtered_edges_to_draw.iterrows():
                source_node = edge_row['source']
                target_node = edge_row['target']
                # Add edge properties if they exist, e.g., 'value' for width or title
                edge_value = edge_row.get('value', 1.0) # Default value if not present
                edge_title = f"Similarity: {edge_value:.2f}" # Default title

                # Check if edge already exists to prevent duplicates in undirected graph
                # Pyvis handles this somewhat, but explicit check can be safer
                if not net.has_edge(source_node, target_node) and not net.has_edge(target_node, source_node):
                    net.add_edge(
                        source_node,
                        target_node,
                        title=edge_title,
                        color="#888888", # Default edge color
                        width=1.5 + (edge_value * 2) # Example: scale width by value
                    )
            if filtered_edges_to_draw.empty:
                st.info("No similarity edges found between the currently filtered papers.")
        else:
            st.warning("Cannot show similarity edges: 'source' or 'target' column missing in your edges data.")
    elif not show_similarity_edges:
        st.info("Similarity edges are currently turned off. Toggle them on in the sidebar.")
    elif df_edges.empty:
        st.warning("Cannot show similarity edges: No edge data loaded from your JSON file.")


    # Save and display the graph HTML
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
