# # Methods_Graph.py
#
# import streamlit as st
# import json
# import pandas as pd
# from pyvis.network import Network
# import os
# import logging
# from datetime import datetime
#
# # --- Configuration ---
# GRAPH_DATA_JSON_PATH = "./data/methods/graph_data.json" # Assuming this path is correct for spatial data methods
# TEMP_HTML_PATH = "./graph_visualization_methods.html"
#
# # --- Logging Setup ---
# LOG_DIR = "./data/logs"
# os.makedirs(LOG_DIR, exist_ok=True)
# log_filename = datetime.now().strftime(f"{LOG_DIR}/spatial_data_graph_page_%Y%m%d_%H%M%S.log") # Updated log filename
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
# if logger.handlers:
#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)
#
# file_handler = logging.FileHandler(log_filename)
# file_handler.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
#
# logger.addHandler(file_handler)
#
# # --- Function to Load Graph Data ---
# @st.cache_data
# def load_graph_data(file_path):
#     """Loads graph data from a JSON file."""
#     if not os.path.exists(file_path):
#         logger.error(f"Graph data file not found at: {file_path}")
#         st.error(f"Graph data file not found. Please ensure '{file_path}' exists.")
#         return None, None
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         logger.info(f"Successfully loaded graph data from '{file_path}'.")
#         return data['nodes'], data['links']
#     except json.JSONDecodeError as e:
#         logger.error(f"Error decoding JSON from '{file_path}': {e}")
#         st.error(f"Error loading graph data: Invalid JSON format. {e}")
#         return None, None
#     except Exception as e:
#         logger.error(f"An unexpected error occurred while loading graph data: {e}")
#         st.error(f"An unexpected error occurred while loading graph data: {e}")
#         return None, None
#
# # --- Main Streamlit Page Content ---
# def app():
#     # REMOVED: st.set_page_config(...) - This is now in your main app file
#
#     st.title("Interactive Graph of Computational Spatial Data Analysis Methods")
#     st.markdown("""
#     This interactive graph visualizes connections between computational spatial data analysis papers based on their semantic similarity.
#     Use the sidebar to search for papers and filter the graph. Click on nodes to see paper details.
#     """)
#
#     nodes, links = load_graph_data(GRAPH_DATA_JSON_PATH)
#
#     if nodes is None or links is None:
#         st.stop()
#
#     df_nodes = pd.DataFrame(nodes)
#     df_links = pd.DataFrame(links)
#
#     # --- Sidebar for Search and Filters ---
#     st.sidebar.header("Search & Filters")
#
#     search_query = st.sidebar.text_input("Search by Title or Authors", "")
#
#     min_year = int(df_nodes['year'].min()) if 'year' in df_nodes.columns and df_nodes['year'].notna().any() else 1900
#     max_year = int(df_nodes['year'].max()) if 'year' in df_nodes.columns and df_nodes['year'].notna().any() else datetime.now().year
#     year_range = st.sidebar.slider(
#         "Filter by Year",
#         min_value=min_year,
#         max_value=max_year,
#         value=(min_year, max_year)
#     )
#
#     selected_paper_types = []
#     if 'kw_paper_type' in df_nodes.columns:
#         all_paper_types = df_nodes['kw_paper_type'].dropna().unique().tolist()
#         if all_paper_types:
#             selected_paper_types = st.sidebar.multiselect(
#                 "Filter by Paper Type",
#                 options=all_paper_types,
#                 default=all_paper_types
#             )
#
#     selected_pipeline_categories = []
#     if 'kw_pipeline_category' in df_nodes.columns:
#         all_pipeline_categories = df_nodes['kw_pipeline_category'].dropna().unique().tolist()
#         if all_pipeline_categories:
#             selected_pipeline_categories = st.sidebar.multiselect(
#                 "Filter by Pipeline Category",
#                 options=all_pipeline_categories,
#                 default=all_pipeline_categories
#             )
#
#     filtered_nodes_df = df_nodes.copy()
#
#     if search_query:
#         filtered_nodes_df = filtered_nodes_df[
#             filtered_nodes_df['title'].str.contains(search_query, case=False, na=False) |
#             filtered_nodes_df['authors'].str.contains(search_query, case=False, na=False)
#         ]
#
#     if 'year' in filtered_nodes_df.columns:
#         filtered_nodes_df = filtered_nodes_df[
#             (filtered_nodes_df['year'] >= year_range[0]) &
#             (filtered_nodes_df['year'] <= year_range[1])
#         ]
#
#     if selected_paper_types and 'kw_paper_type' in filtered_nodes_df.columns:
#         filtered_nodes_df = filtered_nodes_df[
#             filtered_nodes_df['kw_paper_type'].isin(selected_paper_types)
#         ]
#
#     if selected_pipeline_categories and 'kw_pipeline_category' in filtered_nodes_df.columns:
#         filtered_nodes_df = filtered_nodes_df[
#             filtered_nodes_df['kw_pipeline_category'].isin(selected_pipeline_categories)
#         ]
#
#     filtered_node_ids = set(filtered_nodes_df['id'].tolist())
#
#     filtered_links_df = df_links[
#         (df_links['source'].isin(filtered_node_ids)) &
#         (df_links['target'].isin(filtered_node_ids))
#     ]
#
#     st.sidebar.info(f"Showing {len(filtered_node_ids)} papers and {len(filtered_links_df)} connections.")
#
#     # --- Graph Visualization ---
#     st.subheader("Graph Visualization")
#
#     if not filtered_nodes_df.empty:
#         net = Network(
#             height="750px",
#             width="100%",
#             bgcolor="#222222",
#             font_color="white",
#             notebook=True,
#             cdn_resources='remote'
#         )
#
#         net.toggle_physics(True)
#         net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
#
#         for _, node in filtered_nodes_df.iterrows():
#             title_html = f"""
#             <b>Title:</b> {node['title']}<br>
#             <b>Authors:</b> {node['authors']}<br>
#             <b>Year:</b> {node['year']}<br>
#             <b>Journal:</b> {node['journal']}<br>
#             <b>Paper Type:</b> {node.get('kw_paper_type', 'N/A')}<br>
#             <b>Pipeline Category:</b> {node.get('kw_pipeline_category', 'N/A')}<br>
#             <b>DOI:</b> <a href="https://doi.org/{node['id']}" target="_blank">{node['id']}</a><br>
#             """
#             tooltip = f"{node['title']} ({node['year']})"
#
#             size = 10 + (node['year'] - min_year) / (max_year - min_year + 1) * 15 if (max_year - min_year) > 0 else 15
#             color = "#FF6347" if node.get('kw_paper_type') == 'Review' else "#6A5ACD"
#
#             net.add_node(
#                 node['id'],
#                 label=node['title'],
#                 title=title_html,
#                 x=node['x'] * 1000,
#                 y=node['y'] * 1000,
#                 size=size,
#                 color=color,
#                 paper_data=node.to_dict()
#             )
#
#         for _, link in filtered_links_df.iterrows():
#             net.add_edge(
#                 link['source'],
#                 link['target'],
#                 value=link['value'],
#                 title=f"Similarity: {link['value']:.2f}",
#                 color="#888888"
#             )
#
#         net.show_buttons(filter_=['physics', 'selection', 'nodes', 'edges', 'manipulation'])
#         net.options.interaction.hover = True
#         net.options.interaction.tooltipDelay = 300
#         net.options.interaction.zoomView = True
#         net.options.interaction.dragNodes = True
#         net.options.interaction.dragView = True
#
#         net.save_graph(TEMP_HTML_PATH)
#
#         with open(TEMP_HTML_PATH, 'r', encoding='utf-8') as f:
#             html_content = f.read()
#         st.components.v1.html(html_content, height=750, scrolling=True)
#
#         st.subheader("Selected Paper Details")
#         st.info("Click on a node in the graph to see its details here (requires custom component for direct interaction). "
#                 "Hover over nodes for quick details.")
#
#     else:
#         st.warning("No papers match the current filters. Adjust your search or filters.")
#
#     if os.path.exists(TEMP_HTML_PATH):
#         os.remove(TEMP_HTML_PATH)
#         logger.info(f"Cleaned up temporary HTML file: {TEMP_HTML_PATH}")
#
#     st.markdown("---")
#     st.markdown("Developed for Computational Spatial Data Analysis") # Updated footer text
#
# if __name__ == "__main__":
#     app()
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import ast # For safely evaluating string representations of lists
import numpy as np
import colorsys # For generating distinct colors
import hashlib # For consistent color hashing

# --- Configuration ---
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Paper Network Explorer")

# --- Helper Functions ---

# Function to safely convert string representations of lists to actual lists
def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a list or converts a comma-separated
    string into a list. Handles cases where the input is already a list.
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

# Function to generate a distinct color based on a string (for categories)
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
    return f"#{r:02x}{g:02x}{b:02x}"

# --- Load Data (Example - replace with your actual data loading) ---
@st.cache_data
def load_data():
    """
    Loads and preprocesses the paper data.
    Replace this with your actual data loading from a CSV or other source.
    """
    # This is dummy data for demonstration.
    # Ensure your actual DataFrame has the following columns:
    # 'doi': Unique identifier for each paper
    # 'title': Title of the paper
    # 'citationCount': Number of citations (integer)
    # 'year': Publication year (integer)
    # 'kw_pipeline_category': List of strings (e.g., ['Category A', 'Category B'])
    # 'llm_annot_tested_assay_types_platforms': List of strings (e.g., ['Assay X', 'Platform 1'])
    # 'llm_annot_tested_data_modalities': List of strings (e.g., ['Modality M', 'Modality N'])
    # 'llm_annot_compared_algorithms_packages': List of strings (e.g., ['Algo A', 'Package 1'])
    # 'similar_papers': List of DOIs of similar papers
    # 'abstract': Abstract of the paper (string)
    data = {
        'doi': [f'doi_{i}' for i in range(100)],
        'title': [f'Paper Title {i}' for i in range(100)],
        'citationCount': np.random.randint(0, 1000, 100),
        'year': np.random.randint(2010, 2025, 100),
        'kw_pipeline_category': [
            ['Computational', 'Category A'] if i % 5 == 0 else
            ['Category B'] if i % 7 == 0 else
            ['Computational'] if i % 3 == 0 else
            ['Category C']
            for i in range(100)
        ],
        'llm_annot_tested_assay_types_platforms': [
            ['Visium', 'Platform 1'] if i % 4 == 0 else
            ['Assay Y'] if i % 6 == 0 else
            ['Platform 2', 'Visium'] if i % 8 == 0 else
            ['Assay Z']
            for i in range(100)
        ],
        'llm_annot_tested_data_modalities': [
            ['Imaging', 'Genomics'] if i % 3 == 0 else
            ['Proteomics'] if i % 5 == 0 else
            ['Imaging'] if i % 7 == 0 else
            ['Transcriptomics']
            for i in range(100)
        ],
        'llm_annot_compared_algorithms_packages': [
            ['Algo A', 'Package 1'] if i % 2 == 0 else
            ['Algo B'] if i % 4 == 0 else
            ['Package 2']
            for i in range(100)
        ],
        # 'similar_papers' should contain DOIs that this paper is similar to.
        # For demonstration, creating some random connections.
        'similar_papers': [
            [f'doi_{j}' for j in np.random.choice(range(100), size=np.random.randint(0, 3), replace=False) if j != i]
            for i in range(100)
        ],
        'abstract': [f'This is an abstract for paper {i}. It discusses various topics related to its categories, modalities, and assays. ' * 2 for i in range(100)]
    }
    df = pd.DataFrame(data)

    # Apply safe_literal_eval to all columns that are expected to be lists
    list_cols = [
        'kw_pipeline_category',
        'llm_annot_tested_assay_types_platforms',
        'llm_annot_tested_data_modalities',
        'llm_annot_compared_algorithms_packages',
        'similar_papers'
    ]
    for col in list_cols:
        df[col] = df[col].apply(safe_literal_eval)

    # Ensure citationCount is numeric and handle potential missing values
    df['citationCount'] = pd.to_numeric(df['citationCount'], errors='coerce').fillna(0).astype(int)

    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Papers")

# Year Range Filter (between X and Y year)
min_year, max_year = int(df['year'].min()), int(df['year'].max())
selected_year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year) # Default to the full range
)

# Extract all unique values for multiselect filters
# Ensure sorting for consistent display
all_categories = sorted(list(set(item for sublist in df['kw_pipeline_category'] for item in sublist)))
all_assay_types = sorted(list(set(item for sublist in df['llm_annot_tested_assay_types_platforms'] for item in sublist)))
all_data_modalities = sorted(list(set(item for sublist in df['llm_annot_tested_data_modalities'] for item in sublist)))

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
if selected_categories:
    filtered_df = filtered_df[
        filtered_df['kw_pipeline_category'].apply(lambda x: any(cat in selected_categories for cat in x))
    ]

# Apply assay types filter: paper included if ANY of its assay types are selected
if selected_assay_types:
    filtered_df = filtered_df[
        filtered_df['llm_annot_tested_assay_types_platforms'].apply(lambda x: any(assay in selected_assay_types for assay in x))
    ]

# Apply data modalities filter: paper included if ANY of its modalities are selected
if selected_data_modalities:
    filtered_df = filtered_df[
        filtered_df['llm_annot_tested_data_modalities'].apply(lambda x: any(modality in selected_data_modalities for modality in x))
    ]

st.write(f"Displaying **{len(filtered_df)}** papers based on current filters.")

# --- Graph Visualization ---
if not filtered_df.empty:
    # Initialize pyvis network
    # `notebook=True` and `cdn_resources='remote'` are important for Streamlit compatibility
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
    net.toggle_physics(True) # Enable physics for better node layout and movement

    # Calculate min/max citation counts for node size normalization
    max_citations = filtered_df['citationCount'].max()
    min_citations = filtered_df['citationCount'].min()

    # Create a color map for categories using the hashing function
    category_colors = {cat: get_hashed_color(cat) for cat in all_categories}

    # Add nodes to the graph
    for idx, row in filtered_df.iterrows():
        node_id = row['doi']
        title = row['title']
        citations = row['citationCount']
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
        # The full list of categories will be in the tooltip.
        node_color = "#CCCCCC" # Default grey if no categories
        if row['kw_pipeline_category']:
            # Use the color associated with the first category in the list
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
            label=title, # Text displayed on the node
            title=tooltip, # HTML content for the tooltip
            size=size, # Node size based on citations
            color=node_color, # Node color based on category
            font={'color': 'black', 'size': 12}, # Font color for the label
            borderWidth=1, # Border around the node
            borderWidthSelected=3, # Thicker border when selected
            shape='dot' # 'dot' or 'circle' are good for size variations
        )

    # Add edges based on similarity if the toggle is enabled
    if show_similarity_edges:
        for idx, row in filtered_df.iterrows():
            source_doi = row['doi']
            # Ensure the source node is in the current graph (i.e., it passed filters)
            if source_doi in [node['id'] for node in net.nodes]:
                for target_doi in row['similar_papers']:
                    # Ensure the target node also exists in the filtered DataFrame and the graph
                    if target_doi in filtered_df['doi'].values and target_doi in [node['id'] for node in net.nodes]:
                        net.add_edge(source_doi, target_doi, title="Similarity", color="#888888", width=1.5)

    # Generate and display the graph HTML
    try:
        # Save the network to an HTML file
        path = "paper_network.html"
        net.save_graph(path)

        # Read the HTML file and display it in Streamlit using components.v1.html
        with open(path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()
        st.components.v1.html(html_content, height=780, scrolling=True)
    except Exception as e:
        st.error(f"An error occurred while rendering the graph: {e}")
        st.info("This might happen if there are no papers matching your filters to display.")

else:
    st.info("No papers match the selected filters. Please adjust your criteria in the sidebar.")

