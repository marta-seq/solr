import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import ast # For safely evaluating string representations of lists
import numpy as np
import colorsys # For generating distinct colors
import hashlib # For consistent color hashing
import json # For handling JSON specific errors

# Define the path to your processed data file.
# Adjust this path if your 'data' directory is located differently relative to this page file.
PROCESSED_DATA_FILE = 'data/methods/graph_data.json'

# --- Helper Functions ---

def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a list or converts a comma-separated
    string into a list. Handles cases where the input is already a list.
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
            # Fallback for comma-separated strings
            return [item.strip() for item in val.split(',') if item.strip()]
    elif isinstance(val, list):
        return val
    return []

def get_hashed_color(text: str) -> str:
    """
    Generates a consistent hex color from a string using hashing.
    """
    if not text:
        return "#CCCCCC" # Default color for empty string
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
    """
    try:
        df = pd.read_json(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the JSON is in the correct directory.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Please check if it's a valid JSON file. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the JSON file: {e}")
        st.stop()

    list_cols = [
        'kw_pipeline_category',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_tested_data_modalities',
        'llm_annot_compared_algorithms_packages',
        'similar_papers',
        'kw_tissue'
    ]
    for col in list_cols:
        if col in df.columns:
            # Apply safe_literal_eval to ensure all are lists
            df[col] = df[col].apply(safe_literal_eval)
        else:
            df[col] = [[] for _ in range(len(df))]
            st.warning(f"Column '{col}' not found in your data. It will be treated as empty lists.")

    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df['citations'] = 0
        st.warning("Column 'citations' not found in your data. Node sizes will default to a minimum size.")

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    else:
        df['year'] = 2000 # Default year if column is missing
        st.warning("Column 'year' not found in your data. Defaulting to year 2000 for all papers.")

    for col in ['doi', 'title', 'abstract']:
        if col not in df.columns:
            df[col] = ''
            st.warning(f"Column '{col}' not found in your data. It will be empty in display.")

    return df

# Load data when the page is run
df = load_data(PROCESSED_DATA_FILE)

# --- Streamlit Page Content ---

# Header for this specific page
st.header("Paper Network Visualization")

# --- Sidebar Filters ---
st.sidebar.header("Filter Papers")

# Year Range Slider
if not df.empty and 'year' in df.columns:
    min_year_data, max_year_data = int(df['year'].min()), int(df['year'].max())
    selected_year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year_data,
        max_value=max_year_data,
        value=(min_year_data, max_year_data)
    )
else:
    st.sidebar.info("No data loaded or 'year' column missing to set year filter.")
    selected_year_range = (0, 9999) # Broad range if no year data

# Get all unique values for multiselect filters from the entire dataset (df)
all_categories = sorted(list(set(item for sublist in df['kw_pipeline_category'] for item in sublist))) if 'kw_pipeline_category' in df.columns else []
all_assay_types = sorted(list(set(item for sublist in df['llm_annot_tested_assay_types_platforms'] for item in sublist))) if 'llm_annot_tested_assay_types_platforms' in df.columns else []
all_data_modalities = sorted(list(set(item for sublist in df['llm_annot_tested_data_modalities'] for item in sublist))) if 'llm_annot_tested_data_modalities' in df.columns else []

# Multiselect filters
selected_categories = st.sidebar.multiselect(
    "Filter by Pipeline Category",
    options=all_categories,
    default=all_categories # Default to all selected
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
show_similarity_edges = st.sidebar.checkbox("Show Similarity Edges", value=True)

# --- Apply Filters ---
filtered_df = df[
    (df['year'] >= selected_year_range[0]) &
    (df['year'] <= selected_year_range[1])
].copy() # Use .copy() to avoid SettingWithCopyWarning

# Apply multiselect filters
if selected_categories and 'kw_pipeline_category' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['kw_pipeline_category'].apply(lambda x: any(cat in selected_categories for cat in x))
    ]

if selected_assay_types and 'llm_annot_tested_assay_types_platforms' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['llm_annot_tested_assay_types_platforms'].apply(lambda x: any(assay in selected_assay_types for assay in x))
    ]

if selected_data_modalities and 'llm_annot_tested_data_modalities' in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df['llm_annot_tested_data_modalities'].apply(lambda x: any(modality in selected_data_modalities for modality in x))
    ]

st.write(f"Displaying **{len(filtered_df)}** papers based on current filters.")

# --- Graph Visualization ---
if not filtered_df.empty:
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
    net.toggle_physics(True)

    max_citations = filtered_df['citations'].max() if 'citations' in filtered_df.columns else 0
    min_citations = filtered_df['citations'].min() if 'citations' in filtered_df.columns else 0

    # Generate colors for ALL possible categories to ensure consistency
    all_categories_for_colors = sorted(list(set(item for sublist in df['kw_pipeline_category'] for item in sublist)))
    category_colors = {cat: get_hashed_color(cat) for cat in all_categories_for_colors}

    for idx, row in filtered_df.iterrows():
        node_id = row['doi']
        title = row['title']
        citations = row['citations'] if 'citations' in row else 0
        year = row['year']
        # Join lists for display in tooltip
        categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
        assay_types = ", ".join(row['llm_annot_tested_assay_types_platforms']) if row['llm_annot_tested_assay_types_platforms'] else "N/A"
        data_modalities = ", ".join(row['llm_annot_tested_data_modalities']) if row['llm_annot_tested_data_modalities'] else "N/A"
        abstract = row['abstract']

        # Node size based on citations (logarithmic scale for better visual distinction)
        if max_citations > 0:
            size = 10 + (np.log1p(citations) / np.log1p(max_citations + 1)) * 30
        else:
            size = 10 # Default size if no citations or max_citations is 0

        # Node color based on the first pipeline category (if available)
        node_color = "#CCCCCC" # Default grey
        if row['kw_pipeline_category'] and row['kw_pipeline_category'][0]:
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
            label=title,
            title=tooltip,
            size=size,
            color=node_color,
            font={'color': 'black', 'size': 12},
            borderWidth=1,
            borderWidthSelected=3,
            shape='dot'
        )

    # Add similarity edges based on toggle
    if show_similarity_edges and 'similar_papers' in filtered_df.columns:
        for idx, row in filtered_df.iterrows():
            source_doi = row['doi']
            if source_doi in [node['id'] for node in net.nodes]: # Ensure source node exists in current graph
                for target_doi in row['similar_papers']:
                    if target_doi in filtered_df['doi'].values and target_doi in [node['id'] for node in net.nodes]: # Ensure target node exists
                        net.add_edge(source_doi, target_doi, title="Similarity", color="#888888", width=1.5)
    elif not show_similarity_edges:
        st.info("Similarity edges are currently turned off.")
    elif 'similar_papers' not in filtered_df.columns:
        st.warning("Cannot show similarity edges: 'similar_papers' column not found in your data.")

    # Save and display the graph
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

