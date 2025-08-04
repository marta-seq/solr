import logging
import streamlit as st
import pandas as pd
from pyvis.network import Network
import ast
import numpy as np
import colorsys
import hashlib
import json
import os

# Define the path to your processed data file.
PROCESSED_DATA_FILE = 'data/methods/graph_data.json'

# --- Helper Functions ---

def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            evaluated = ast.literal_eval(val)
            if isinstance(evaluated, list):
                return evaluated
            else:
                return [str(evaluated)]
        except (ValueError, SyntaxError):
            return [item.strip() for item in val.split(',') if item.strip()]
    elif isinstance(val, list):
        return val
    return []

def get_hashed_color(text: str) -> str:
    if not text:
        return "#CCCCCC"
    hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    hue = (hash_value % 360) / 360.0
    saturation = 0.7
    value = 0.85
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"#{r:02x}{g:02x}{b:02x}"

# --- Load Data ---

@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_json_data = json.load(f)
            logging.debug(f"Loaded JSON data from '{file_path}'. Keys: {full_json_data.keys()}")

        if 'nodes' in full_json_data and isinstance(full_json_data['nodes'], list):
            df_nodes = pd.DataFrame(full_json_data['nodes'])
            df_nodes['id'] = df_nodes['id'].astype(str)
            df_nodes['doi'] = df_nodes['id'].astype(str)
            logging.debug(f"Nodes DataFrame created with shape: {df_nodes.shape}.")
            logging.debug(f"Columns in nodes DataFrame: {df_nodes.columns.tolist()}.")
        else:
            st.error(f"Error: JSON data from '{file_path}' does not contain a 'nodes' key or 'nodes' is not a list.")
            st.stop()

        df_edges = pd.DataFrame()
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

    list_cols = [
        'kw_pipeline_category',
        'kw_detected_methods',
        'llm_annot_tested_data_modalities',
        'llm_annot_compared_algorithms_packages',
        'kw_tissue'
    ]

    for col in list_cols:
        if col in df_nodes.columns:
            df_nodes[col] = df_nodes[col].fillna('').apply(safe_literal_eval)
        else:
            df_nodes[col] = [[] for _ in range(len(df_nodes))]
            st.warning(f"Column '{col}' not found in your nodes data. It will be treated as empty lists for filtering.")

    if 'citations' in df_nodes.columns:
        df_nodes['citations'] = pd.to_numeric(df_nodes['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['citations'] = 0
        st.warning("Column 'citations' not found in your nodes data. Node sizes will default to a minimum size.")

    if 'year' in df_nodes.columns:
        df_nodes['year'] = pd.to_numeric(df_nodes['year'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['year'] = 2000
        st.warning("Column 'year' not found in your nodes data. Defaulting to year 2000 for all papers.")

    for col in ['doi', 'title', 'abstract']:
        if col not in df_nodes.columns:
            df_nodes[col] = ''
            st.warning(f"Column '{col}' not found in your nodes data. It will be empty in display.")
        if col == 'doi':
            df_nodes[col] = df_nodes[col].astype(str)

    return df_nodes, df_edges

df_nodes, df_edges = load_data(PROCESSED_DATA_FILE)

# --- Streamlit Page Content ---

st.header("Paper Network Visualization")

# --- Sidebar Filters ---
st.sidebar.header("Filter Papers")

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
    selected_year_range = (0, 9999)

all_categories = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in sublist))) if 'kw_pipeline_category' in df_nodes.columns else []
all_assay_types = sorted(list(set(item for sublist in df_nodes['kw_detected_methods'] for item in sublist))) if 'kw_detected_methods' in df_nodes.columns else []
all_data_modalities = sorted(list(set(item for sublist in df_nodes['llm_annot_tested_data_modalities'] for item in sublist))) if 'llm_annot_tested_data_modalities' in df_nodes.columns else []

selected_categories = st.sidebar.multiselect(
    "Filter by Pipeline Category",
    options=all_categories,
    default=all_categories
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
filtered_nodes_df = df_nodes[
    (df_nodes['year'] >= selected_year_range[0]) &
    (df_nodes['year'] <= selected_year_range[1])
].copy()

if selected_categories and 'kw_pipeline_category' in filtered_nodes_df.columns:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['kw_pipeline_category'].apply(lambda x: any(cat in selected_categories for cat in x))
    ]

if selected_assay_types and 'kw_detected_methods' in filtered_nodes_df.columns:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['kw_detected_methods'].apply(lambda x: any(assay in selected_assay_types for assay in x))
    ]

if selected_data_modalities and 'llm_annot_tested_data_modalities' in filtered_nodes_df.columns:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['llm_annot_tested_data_modalities'].apply(lambda x: any(modality in selected_data_modalities for modality in x))
    ]

st.write(f"Displaying **{len(filtered_nodes_df)}** papers based on current filters.")

# --- Graph Visualization ---
if not filtered_nodes_df.empty:
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
    net.toggle_physics(True)

    max_citations = filtered_nodes_df['citations'].max() if 'citations' in filtered_nodes_df.columns else 0
    max_citations_for_scaling = max_citations + 1

    all_categories_for_colors = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in sublist)))
    category_colors = {cat: get_hashed_color(cat) for cat in all_categories_for_colors}

    for idx, row in filtered_nodes_df.iterrows():
        node_id = row['doi']
        title = row['title']
        citations = row['citations'] if 'citations' in row else 0
        year = row['year']
        categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
        assay_types = ", ".join(row['kw_detected_methods']) if row['kw_detected_methods'] else "N/A"
        data_modalities = ", ".join(row['llm_annot_tested_data_modalities']) if row['llm_annot_tested_data_modalities'] else "N/A"
        abstract = row['abstract']

        if max_citations_for_scaling > 1:
            size = 10 + (np.log1p(citations) / np.log1p(max_citations_for_scaling)) * 30
        else:
            size = 20

        node_color = "#CCCCCC"
        if row['kw_pipeline_category']:
            node_color = category_colors.get(row['kw_pipeline_category'][0], "#CCCCCC")

        tooltip = f"""
        <b>Title:</b> {title}<br>
        <b>DOI:</b> {node_id}<br>
        <b>Citations:</b> {citations}<br>
        <b>Year:</b> {year}<br>
        <b>Categories:</b> {categories}<br>
        <b>Assay Types/Platforms:</b> {assay_types}<br>
        <b>Data Modalities:</b> {data_modalities}<br>
        """

        net.add_node(
            node_id,
            label="",
            title=tooltip,
            size=size,
            color=node_color,
            font={'color': 'black', 'size': 12},
            borderWidth=1,
            borderWidthSelected=3,
            shape='dot'
        )

    similarity_threshold = 0.75
    if show_similarity_edges and not df_edges.empty:
        current_graph_dois = set(filtered_nodes_df['doi'].tolist())

        filtered_edges_to_draw = df_edges[
            (df_edges['source'].isin(current_graph_dois)) &
            (df_edges['target'].isin(current_graph_dois)) &
            (df_edges['value'] > similarity_threshold)
        ]

        for idx, edge_row in filtered_edges_to_draw.iterrows():
            source_node = edge_row['source']
            target_node = edge_row['target']
            edge_value = edge_row.get('value', 1.0)
            edge_title = f"Similarity: {edge_value:.2f}"

            net.add_edge(
                source_node,
                target_node,
                title=edge_title,
                color="#888888",
                width=1.5 + (edge_value * 2)
            )

        if filtered_edges_to_draw.empty:
            st.info("No similarity edges found between the currently filtered papers.")
    elif not show_similarity_edges:
        st.info("Similarity edges are currently turned off. Toggle them on in the sidebar.")
    elif df_edges.empty:
        st.warning("Cannot show similarity edges: No edge data loaded from your JSON file.")

    # Generate legend HTML
    legend_html = """
    <div style="display: flex; flex-wrap: wrap;">
    """
    for category, color in category_colors.items():
        legend_html += f"""
        <div style="margin: 5px;">
            <div style="background-color: {color}; width: 20px; height: 20px; display: inline-block;"></div>
            <span>{category}</span>
        </div>
        """
    legend_html += "</div>"

    # Display legend
    st.components.v1.html(legend_html, height=100)

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
