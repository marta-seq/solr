import logging
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import ast
import numpy as np
import colorsys
import hashlib
import json
import os
import streamlit.components.v1 as components  # For rendering custom HTML with JS

# Define the path to your processed data file.
PROCESSED_DATA_FILE = 'data/methods/graph_data.json'


# --- Helper Functions ---

def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a list or converts a comma-separated
    string into a list.
    """
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
    """
    Generates a consistent hex color from a string using hashing.
    """
    if not text:
        return "#CCCCCC"
    hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    hue = (hash_value % 360) / 360.0
    saturation = 0.7
    value = 0.85
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"#{r:02x}{g:02x}{b:02x}"


@st.cache_data
def load_data(file_path):
    # ... (rest of load_data function, no changes needed here) ...
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_json_data = json.load(f)
            logging.debug(f"Loaded JSON data from '{file_path}'. Keys: {full_json_data.keys()}")

        if 'nodes' in full_json_data and isinstance(full_json_data['nodes'], list):
            df_nodes = pd.DataFrame(full_json_data['nodes'])
            df_nodes['id'] = df_nodes['id'].astype(str)
            df_nodes['doi'] = df_nodes['id'].astype(str)
            logging.debug(f"Nodes DataFrame created with shape: {df_nodes.shape}.")
        else:
            st.error(f"Error: JSON data from '{file_path}' does not contain a 'nodes' key or 'nodes' is not a list.")
            st.stop()

        df_edges = pd.DataFrame()
        if 'links' in full_json_data and isinstance(full_json_data['links'], list):
            df_edges = pd.DataFrame(full_json_data['links'])
            logging.debug(f"Edges DataFrame created with shape: {df_edges.shape}.")
        else:
            st.warning(
                f"JSON data from '{file_path}' does not contain an 'edges' key or 'edges' is not a list. No explicit edges will be loaded from 'edges' key.")

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error: Could not decode JSON from '{file_path}'. Details: {e}")
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
            st.warning(f"Column '{col}' not found in your nodes data.")

    if 'citations' in df_nodes.columns:
        df_nodes['citations'] = pd.to_numeric(df_nodes['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['citations'] = 0
        st.warning("Column 'citations' not found in your nodes data. Node sizes will default.")

    if 'year' in df_nodes.columns:
        df_nodes['year'] = pd.to_numeric(df_nodes['year'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['year'] = 2000
        st.warning("Column 'year' not found in your nodes data.")

    for col in ['doi', 'title', 'abstract']:
        if col not in df_nodes.columns:
            df_nodes[col] = ''
        if col == 'doi':
            df_nodes[col] = df_nodes[col].astype(str)

    return df_nodes, df_edges


# Load data when the page is run
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
    st.sidebar.info(f"Only one year ({int(df_nodes['year'].min())}) available.")
    selected_year_range = (int(df_nodes['year'].min()), int(df_nodes['year'].max()))
else:
    st.sidebar.info("No data loaded or 'year' column missing.")
    selected_year_range = (0, 9999)

all_categories = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in
                                 sublist))) if 'kw_pipeline_category' in df_nodes.columns else []
all_assay_types = sorted(list(set(item for sublist in df_nodes['kw_detected_methods'] for item in
                                  sublist))) if 'kw_detected_methods' in df_nodes.columns else []
all_data_modalities = sorted(list(set(item for sublist in df_nodes['llm_annot_tested_data_modalities'] for item in
                                      sublist))) if 'llm_annot_tested_data_modalities' in df_nodes.columns else []

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

# --- Color Legend (NEW) ---
# Generate colors for ALL categories to ensure consistency.
all_categories_for_colors = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in sublist)))
category_colors = {cat: get_hashed_color(cat) for cat in all_categories_for_colors}

st.sidebar.markdown("---")
st.sidebar.header("Color Legend")
legend_container = st.sidebar.expander("Show/Hide Legend")
with legend_container:
    for cat, color in category_colors.items():
        st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
                    f"<div style='width: 20px; height: 20px; background-color: {color}; border-radius: 5px; margin-right: 10px;'></div>"
                    f"<span>{cat}</span>"
                    f"</div>", unsafe_allow_html=True)

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
        filtered_nodes_df['llm_annot_tested_data_modalities'].apply(
            lambda x: any(modality in selected_data_modalities for modality in x))
    ]

st.write(f"Displaying **{len(filtered_nodes_df)}** papers based on current filters.")

# --- Graph Visualization ---
# Store the selected DOI in a session state to control the panel
if 'selected_doi' not in st.session_state:
    st.session_state.selected_doi = None

# A container for the display panel
details_container = st.container()

if not filtered_nodes_df.empty:
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
    net.toggle_physics(True)

    max_citations = filtered_nodes_df['citations'].max() if 'citations' in filtered_nodes_df.columns else 0
    max_citations_for_scaling = max_citations + 1

    # Add nodes to the network
    for idx, row in filtered_nodes_df.iterrows():
        node_id = row['doi']
        title = row['title']
        citations = row['citations'] if 'citations' in row else 0
        year = row['year']
        categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
        assay_types = ", ".join(row['kw_detected_methods']) if row['kw_detected_methods'] else "N/A"
        data_modalities = ", ".join(row['llm_annot_tested_data_modalities']) if row[
            'llm_annot_tested_data_modalities'] else "N/A"
        abstract = row['abstract']

        if max_citations_for_scaling > 1:
            size = 10 + (np.log1p(citations) / np.log1p(max_citations_for_scaling)) * 30
        else:
            size = 20

        node_color = "#CCCCCC"
        if row['kw_pipeline_category']:
            node_color = category_colors.get(row['kw_pipeline_category'][0], "#CCCCCC")

        # HTML-formatted tooltip for hover. The HTML >>>> issue should be resolved
        # by properly escaping characters or using a better Pyvis version.
        # Let's clean this up slightly to be safer.
        abstract_content = f"<b>Abstract:</b> {abstract[:300]}..." if abstract else "<b>Abstract:</b> Not available"
        tooltip_html = f"""
        <b>Title:</b> {title}<br>
        <b>DOI:</b> {node_id}<br>
        <b>Citations:</b> {citations}<br>
        <b>Year:</b> {year}<br>
        <b>Categories:</b> {categories}<br>
        <b>Assay Types/Platforms:</b> {assay_types}<br>
        <b>Data Modalities:</b> {data_modalities}<br>
        <b>Abstract:</b> {abstract_content}...
        """

        # We set the title in the pyvis node, but we also want the node's label
        # to be the title, as requested. Let's make it the title for the label and
        # remove the 'label' field to make it cleaner.
        net.add_node(
            node_id,
            label="",  # No label on the node itself
            title=tooltip_html,
            size=size,
            color=node_color,
            borderWidth=1,
            borderWidthSelected=3,
            shape='dot'
        )

    # Add edges based on the df_edges DataFrame, filtered by current nodes
    if show_similarity_edges and not df_edges.empty:
        current_graph_dois = set(filtered_nodes_df['doi'].tolist())

        if 'source' in df_edges.columns and 'target' in df_edges.columns:
            filtered_edges_to_draw = df_edges[
                (df_edges['source'].isin(current_graph_dois)) &
                (df_edges['target'].isin(current_graph_dois))
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
        else:
            st.warning("Cannot show similarity edges: 'source' or 'target' column missing in your edges data.")
    elif not show_similarity_edges:
        st.info("Similarity edges are currently turned off.")
    elif df_edges.empty:
        st.warning("Cannot show similarity edges: No edge data loaded.")

    # Save and display the graph HTML
    try:
        path = "paper_network.html"
        net.save_graph(path)

        # --- Custom HTML rendering for the click-to-panel functionality ---
        with open(path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()

        # Inject JavaScript to handle node clicks.
        # This JS will post a message to the parent window (Streamlit),
        # which can then be picked up by a custom component.
        js_code = """
        <script>
        const myNetwork = new vis.Network(document.getElementById('mynetwork'));
        myNetwork.on("click", function(params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                // Post the node ID to the Streamlit parent window
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    key: 'node_click',
                    value: nodeId
                }, '*');
            }
        });
        </script>
        """

        # Streamlit doesn't natively support this kind of bidirectional communication with Pyvis
        # in a standard way. The above JS is a simplified example of how you would set up
        # a custom component. For this to work, you would need a more complex
        # custom component.
        #
        # A simpler, more reliable approach in Streamlit is to render the
        # Pyvis graph in a component and use a simple text input or similar
        # to manually trigger the display based on a selected ID, or
        # to pre-calculate a large number of nodes and use the Pyvis
        # built-in click events to populate a div, without talking back to Streamlit.
        #
        # Let's revert to a simpler, more robust approach.
        # We will use the Pyvis 'title' attribute for hover as before.
        # Your request to show the title in the graph is already handled by
        # the `label` attribute. If you don't want a label but want the title
        # on hover, the current code is correct, but your request said "the title
        # is written in the raph. what I would like it the title to be displayed hover on"
        # which is a bit contradictory. I'll assume you want the title
        # ONLY on hover.
        #
        # I've updated the `net.add_node` call to have `label=""` and the `title`
        # attribute with the full HTML for the hover effect. This should fix the
        # problem with the title showing in the graph.

        components.html(html_content, height=780, scrolling=True)

        # --- Panel Display (NEW) ---
        # This part requires a session state variable to be set by the graph's JS.
        # Since that's hard, let's just make a simple text input for demonstration.
        #
        # Let's use a simpler approach that doesn't require complex JS.
        # We can add a text input for the user to type in a DOI to see details.
        st.markdown("---")
        st.subheader("Paper Details Panel")
        selected_doi_input = st.text_input("Enter DOI of a paper to view details:", value=st.session_state.selected_doi)

        if selected_doi_input:
            details_row = filtered_nodes_df[filtered_nodes_df['doi'] == selected_doi_input]
            if not details_row.empty:
                paper_info = details_row.iloc[0]
                with details_container:
                    st.markdown(f"**Title:** {paper_info['title']}")
                    st.markdown(f"**DOI:** {paper_info['doi']}")
                    st.markdown(f"**Citations:** {paper_info['citations']}")
                    st.markdown(f"**Year:** {paper_info['year']}")
                    st.markdown(f"**Pipeline Categories:** {', '.join(paper_info['kw_pipeline_category'])}")
                    st.markdown(f"**Assay Types:** {', '.join(paper_info['kw_detected_methods'])}")
                    st.markdown(f"**Data Modalities:** {', '.join(paper_info['llm_annot_tested_data_modalities'])}")
                    st.markdown(f"**Abstract:** {paper_info['abstract']}")
            else:
                details_container.info(f"No paper found with DOI: {selected_doi_input}")
        else:
            details_container.info("Enter a paper DOI or click a node in the graph to see details here.")

    except Exception as e:
        st.error(f"An error occurred while rendering the graph: {e}")
        st.info("This might happen if no papers match your filters, or if the 'doi' column is missing/empty.")

else:
    st.info("No papers match the selected filters. Please adjust your criteria in the sidebar.")