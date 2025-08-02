# Methods_Graph.py

import streamlit as st
import json
import pandas as pd
from pyvis.network import Network
import os
import logging
from datetime import datetime

# --- Configuration ---
GRAPH_DATA_JSON_PATH = "./data/methods/graph_data.json" # Assuming this path is correct for spatial data methods
TEMP_HTML_PATH = "./graph_visualization_methods.html"

# --- Logging Setup ---
LOG_DIR = "./data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/spatial_data_graph_page_%Y%m%d_%H%M%S.log") # Updated log filename

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

# --- Function to Load Graph Data ---
@st.cache_data
def load_graph_data(file_path):
    """Loads graph data from a JSON file."""
    if not os.path.exists(file_path):
        logger.error(f"Graph data file not found at: {file_path}")
        st.error(f"Graph data file not found. Please ensure '{file_path}' exists.")
        return None, None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded graph data from '{file_path}'.")
        return data['nodes'], data['links']
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{file_path}': {e}")
        st.error(f"Error loading graph data: Invalid JSON format. {e}")
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading graph data: {e}")
        st.error(f"An unexpected error occurred while loading graph data: {e}")
        return None, None

# --- Main Streamlit Page Content ---
def app():
    # REMOVED: st.set_page_config(...) - This is now in your main app file

    st.title("Interactive Graph of Computational Spatial Data Analysis Methods")
    st.markdown("""
    This interactive graph visualizes connections between computational spatial data analysis papers based on their semantic similarity.
    Use the sidebar to search for papers and filter the graph. Click on nodes to see paper details.
    """)

    nodes, links = load_graph_data(GRAPH_DATA_JSON_PATH)

    if nodes is None or links is None:
        st.stop()

    df_nodes = pd.DataFrame(nodes)
    df_links = pd.DataFrame(links)

    # --- Sidebar for Search and Filters ---
    st.sidebar.header("Search & Filters")

    search_query = st.sidebar.text_input("Search by Title or Authors", "")

    min_year = int(df_nodes['year'].min()) if 'year' in df_nodes.columns and df_nodes['year'].notna().any() else 1900
    max_year = int(df_nodes['year'].max()) if 'year' in df_nodes.columns and df_nodes['year'].notna().any() else datetime.now().year
    year_range = st.sidebar.slider(
        "Filter by Year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    selected_paper_types = []
    if 'kw_paper_type' in df_nodes.columns:
        all_paper_types = df_nodes['kw_paper_type'].dropna().unique().tolist()
        if all_paper_types:
            selected_paper_types = st.sidebar.multiselect(
                "Filter by Paper Type",
                options=all_paper_types,
                default=all_paper_types
            )

    selected_pipeline_categories = []
    if 'kw_pipeline_category' in df_nodes.columns:
        all_pipeline_categories = df_nodes['kw_pipeline_category'].dropna().unique().tolist()
        if all_pipeline_categories:
            selected_pipeline_categories = st.sidebar.multiselect(
                "Filter by Pipeline Category",
                options=all_pipeline_categories,
                default=all_pipeline_categories
            )

    filtered_nodes_df = df_nodes.copy()

    if search_query:
        filtered_nodes_df = filtered_nodes_df[
            filtered_nodes_df['title'].str.contains(search_query, case=False, na=False) |
            filtered_nodes_df['authors'].str.contains(search_query, case=False, na=False)
        ]

    if 'year' in filtered_nodes_df.columns:
        filtered_nodes_df = filtered_nodes_df[
            (filtered_nodes_df['year'] >= year_range[0]) &
            (filtered_nodes_df['year'] <= year_range[1])
        ]

    if selected_paper_types and 'kw_paper_type' in filtered_nodes_df.columns:
        filtered_nodes_df = filtered_nodes_df[
            filtered_nodes_df['kw_paper_type'].isin(selected_paper_types)
        ]

    if selected_pipeline_categories and 'kw_pipeline_category' in filtered_nodes_df.columns:
        filtered_nodes_df = filtered_nodes_df[
            filtered_nodes_df['kw_pipeline_category'].isin(selected_pipeline_categories)
        ]

    filtered_node_ids = set(filtered_nodes_df['id'].tolist())

    filtered_links_df = df_links[
        (df_links['source'].isin(filtered_node_ids)) &
        (df_links['target'].isin(filtered_node_ids))
    ]

    st.sidebar.info(f"Showing {len(filtered_node_ids)} papers and {len(filtered_links_df)} connections.")

    # --- Graph Visualization ---
    st.subheader("Graph Visualization")

    if not filtered_nodes_df.empty:
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=True,
            cdn_resources='remote'
        )

        net.toggle_physics(True)
        net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)

        for _, node in filtered_nodes_df.iterrows():
            title_html = f"""
            <b>Title:</b> {node['title']}<br>
            <b>Authors:</b> {node['authors']}<br>
            <b>Year:</b> {node['year']}<br>
            <b>Journal:</b> {node['journal']}<br>
            <b>Paper Type:</b> {node.get('kw_paper_type', 'N/A')}<br>
            <b>Pipeline Category:</b> {node.get('kw_pipeline_category', 'N/A')}<br>
            <b>DOI:</b> <a href="https://doi.org/{node['id']}" target="_blank">{node['id']}</a><br>
            """
            tooltip = f"{node['title']} ({node['year']})"

            size = 10 + (node['year'] - min_year) / (max_year - min_year + 1) * 15 if (max_year - min_year) > 0 else 15
            color = "#FF6347" if node.get('kw_paper_type') == 'Review' else "#6A5ACD"

            net.add_node(
                node['id'],
                label=node['title'],
                title=title_html,
                x=node['x'] * 1000,
                y=node['y'] * 1000,
                size=size,
                color=color,
                paper_data=node.to_dict()
            )

        for _, link in filtered_links_df.iterrows():
            net.add_edge(
                link['source'],
                link['target'],
                value=link['value'],
                title=f"Similarity: {link['value']:.2f}",
                color="#888888"
            )

        net.show_buttons(filter_=['physics', 'selection', 'nodes', 'edges', 'manipulation'])
        net.options.interaction.hover = True
        net.options.interaction.tooltipDelay = 300
        net.options.interaction.zoomView = True
        net.options.interaction.dragNodes = True
        net.options.interaction.dragView = True

        net.save_graph(TEMP_HTML_PATH)

        with open(TEMP_HTML_PATH, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=750, scrolling=True)

        st.subheader("Selected Paper Details")
        st.info("Click on a node in the graph to see its details here (requires custom component for direct interaction). "
                "Hover over nodes for quick details.")

    else:
        st.warning("No papers match the current filters. Adjust your search or filters.")

    if os.path.exists(TEMP_HTML_PATH):
        os.remove(TEMP_HTML_PATH)
        logger.info(f"Cleaned up temporary HTML file: {TEMP_HTML_PATH}")

    st.markdown("---")
    st.markdown("Developed for Computational Spatial Data Analysis") # Updated footer text

if __name__ == "__main__":
    app()
