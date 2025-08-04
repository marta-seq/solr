import streamlit as st
import pandas as pd
from pyvis.network import Network
import ast  # For safely evaluating string representations of lists
import numpy as np
import colorsys  # For generating distinct colors
import hashlib  # For consistent color hashing
import json  # For handling JSON specific errors
import os
import streamlit.components.v1 as components  # For rendering custom HTML with JS

# Define the path to your processed data file.
PROCESSED_DATA_FILE = 'data/methods/graph_data.json'


# --- Helper Functions ---

def safe_literal_eval(val):
    """
    Safely evaluates a string representation of a list or converts a comma-separated
    string into a list. Handles cases where the input is already a list or non-string.
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
    return []  # Return empty list for NaN or other unexpected types


def get_hashed_color(text: str) -> str:
    """
    Generates a consistent hex color from a string using hashing.
    This ensures that the same category always has the same color across the graph.
    """
    if not text:
        return "#CCCCCC"  # Default color for empty string/missing category
    hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
    # Use HSV to get more distinct colors
    hue = (hash_value % 360) / 360.0
    saturation = 0.7  # Keep saturation high for vivid colors
    value = 0.85  # Keep value high for brightness
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
    return f"#{r:02x}{g:02x}{b:02x}"


@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the paper data from a JSON file.
    The JSON file is expected to contain a dictionary with 'nodes' and 'links' keys.
    'nodes' should point to a list of dictionaries (for paper data).
    'links' should point to a list of dictionaries (for explicit graph edges).
    Applies safe_literal_eval to list-like columns and handles missing columns.

    Returns:
        tuple: A tuple containing two Pandas DataFrames (nodes_df, edges_df).
               Stops Streamlit execution if file cannot be loaded or essential keys are missing.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_json_data = json.load(f)  # Load the entire JSON dictionary

        # Debug print: Raw nodes loaded from JSON - REMOVED
        # if 'nodes' in full_json_data and isinstance(full_json_data['nodes'], list):
        #     st.write(f"Raw nodes loaded from JSON: {len(full_json_data['nodes'])}")
        # else:
        #     st.error(f"Error: JSON data from '{file_path}' does not contain a 'nodes' key or 'nodes' is not a list.")
        #     st.stop()

        # Create DataFrame for nodes from the 'nodes' key
        df_nodes = pd.DataFrame(full_json_data['nodes'])
        # Ensure 'id' is a string for node IDs and remove duplicates based on 'id'
        df_nodes['id'] = df_nodes['id'].astype(str)
        df_nodes.drop_duplicates(subset=['id'], inplace=True)  # Remove duplicates
        df_nodes['doi'] = df_nodes['id'].astype(str)  # Ensure 'doi' is a string for node IDs
        # Debug print: Nodes DataFrame shape after loading and deduplication - REMOVED
        # st.write(f"Nodes DataFrame shape after loading and deduplication: {df_nodes.shape}")

        # Create DataFrame for edges from the 'links' key (if present and valid)
        df_edges = pd.DataFrame()  # Initialize as empty
        if 'links' in full_json_data and isinstance(full_json_data['links'], list):
            df_edges = pd.DataFrame(full_json_data['links'])
        else:
            st.warning(
                f"JSON data from '{file_path}' does not contain a 'links' key or 'links' is not a list. No explicit edges will be loaded.")

    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the JSON is in the correct directory.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(
            f"Error: Could not decode JSON from '{file_path}'. Please check if it's a valid JSON file. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the JSON file: {e}")
        st.stop()

    # --- Preprocessing for df_nodes ---
    # Columns expected to contain lists (or string representations of lists)
    list_cols = [
        'kw_pipeline_category',
        'kw_detected_methods',
        'llm_annot_compared_algorithms_packages',
        'kw_tissue'
    ]

    for col in list_cols:
        if col in df_nodes.columns:
            df_nodes[col] = df_nodes[col].fillna('').apply(safe_literal_eval)
        else:
            df_nodes[col] = [[] for _ in range(len(df_nodes))]
            st.warning(f"Column '{col}' not found in your nodes data. It will be treated as empty lists for filtering.")

    # Handle 'citations' column for node sizing
    if 'citations' in df_nodes.columns:
        df_nodes['citations'] = pd.to_numeric(df_nodes['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['citations'] = 0  # Default citations if column is missing
        st.warning("Column 'citations' not found in your nodes data. Node sizes will default to a minimum size.")

    # Handle 'year' column for filtering
    if 'year' in df_nodes.columns:
        df_nodes['year'] = pd.to_numeric(df_nodes['year'], errors='coerce').fillna(0).astype(int)
        # Debug print: Min/Max year after loading and conversion - REMOVED
        # if not df_nodes.empty:
        #     st.write(f"Min year in loaded data: {df_nodes['year'].min()}, Max year in loaded data: {df_nodes['year'].max()}")
    else:
        df_nodes['year'] = 2000  # Default year if column is missing
        st.warning("Column 'year' not found in your nodes data. Defaulting to year 2000 for all papers.")

    # Handle 'x' and 'y' coordinates for node positioning
    for coord_col in ['x', 'y']:
        if coord_col in df_nodes.columns:
            df_nodes[coord_col] = pd.to_numeric(df_nodes[coord_col], errors='coerce').fillna(0)  # Fill NaN with 0
        else:
            # If 'x' or 'y' are missing, create them with default values (e.g., random or 0)
            st.warning(
                f"Column '{coord_col}' not found in your nodes data. Nodes will be positioned at 0 for this coordinate.")
            df_nodes[coord_col] = 0  # Default to 0 if column is missing

    # Ensure 'doi', 'title', 'abstract' columns exist and are strings
    for col in ['doi', 'title', 'abstract']:
        if col not in df_nodes.columns:
            df_nodes[col] = ''
            st.warning(f"Column '{col}' not found in your nodes data. It will be empty in display.")
        if col == 'doi':
            df_nodes[col] = df_nodes[col].astype(str)

    return df_nodes, df_edges


# Load data when the page is run
df_nodes, df_edges = load_data(PROCESSED_DATA_FILE)

# --- Streamlit Page Content ---

st.header("Spatial Omics Paper Network")

# --- Search Box (NEW) ---
search_query = st.text_input("Search papers by Title or DOI", "")

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
    st.sidebar.info(f"Only one year ({int(df_nodes['year'].min())}) available in data.")
    selected_year_range = (int(df_nodes['year'].min()), int(df_nodes['year'].max()))
else:
    st.sidebar.info("No data loaded or 'year' column missing to set year filter.")
    selected_year_range = (0, 9999)  # Broad default range if no year data

# Get all unique values for checkbox filters from the entire dataset (df_nodes)
all_categories = sorted(list(set(item for sublist in df_nodes['kw_pipeline_category'] for item in
                                 sublist))) if 'kw_pipeline_category' in df_nodes.columns else []
all_assay_types = sorted(list(set(item for sublist in df_nodes['kw_detected_methods'] for item in
                                  sublist))) if 'kw_detected_methods' in df_nodes.columns else []

# Checkbox filters (NEW: using expanders and individual checkboxes)
selected_categories = []
with st.sidebar.expander("Filter by Pipeline Category"):
    for cat in all_categories:
        # Default to False (unchecked)
        if st.checkbox(cat, value=False, key=f"cat_{cat}"):
            selected_categories.append(cat)

selected_assay_types = []
with st.sidebar.expander("Filter by Assay Types/Platforms"):
    for assay in all_assay_types:
        # Default to False (unchecked)
        if st.checkbox(assay, value=False, key=f"assay_{assay}"):
            selected_assay_types.append(assay)

st.sidebar.header("Graph Options")
# Toggle for similarity edges
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
# Start with the full dataset and apply filters sequentially
filtered_nodes_df = df_nodes.copy()

# Apply year filter
filtered_nodes_df = filtered_nodes_df[
    (filtered_nodes_df['year'] >= selected_year_range[0]) &
    (filtered_nodes_df['year'] <= selected_year_range[1])
    ]

# Apply multiselect/checkbox filters: if NO categories are selected, it means NO filter is applied for this category.
# If categories ARE selected, then filter to include papers with ANY of the selected categories.
if selected_categories:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['kw_pipeline_category'].apply(lambda x: any(cat in selected_categories for cat in x))
    ]

if selected_assay_types:
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['kw_detected_methods'].apply(lambda x: any(assay in selected_assay_types for assay in x))
    ]

# Apply search query filter
if search_query:
    search_query_lower = search_query.lower()
    filtered_nodes_df = filtered_nodes_df[
        filtered_nodes_df['title'].str.lower().str.contains(search_query_lower) |
        filtered_nodes_df['doi'].str.lower().str.contains(search_query_lower)
        ]

st.write(f"Displaying **{len(filtered_nodes_df)}** papers based on current filters.")

# --- Graph Visualization and Details Panel Layout ---
# Use columns to place the graph on the left and the details panel on the right
graph_col, details_col = st.columns([0.8, 0.2])  # Adjusted column ratio for wider graph

# Initialize session state for selected DOI (for the details panel)
if 'selected_doi' not in st.session_state:
    st.session_state.selected_doi = None

# Get selected DOI from query params if available (from node click)
query_params = st.query_params
if "selected_doi" in query_params:
    # Ensure it's a string, as query_params can sometimes return a list if multiple values exist
    st.session_state.selected_doi = str(query_params["selected_doi"])
    # Clear query param to prevent re-triggering on subsequent runs unless clicked again
    st.query_params["selected_doi"] = None  # Directly modify query_params dict

with graph_col:
    if not filtered_nodes_df.empty:
        net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)


        # Physics is already disabled, ensuring static nodes.
        # net.toggle_physics(False)

        # Define citation-based sizes - ADJUSTED SIZES
        def get_node_size(citations):
            if citations <= 5:
                return 5  # Smaller base size
            elif 5 < citations <= 25:
                return 8
            elif 25 < citations <= 50:
                return 12
            elif 50 < citations <= 100:
                return 16
            else:  # citations > 100
                return 20  # Max size


        # --- Node Display Limit ---
        max_nodes_to_display = st.sidebar.slider(
            "Max Nodes to Display (for performance)",
            min_value=100,
            max_value=len(filtered_nodes_df) if len(filtered_nodes_df) > 100 else 100,
            # Adjust max based on filtered count
            value=min(1000, len(filtered_nodes_df))  # Default to 1000 or actual count if less
        )

        nodes_to_draw = filtered_nodes_df.head(max_nodes_to_display)
        if len(filtered_nodes_df) > max_nodes_to_display:
            st.warning(
                f"Displaying only the first {max_nodes_to_display} of {len(filtered_nodes_df)} filtered papers for performance. Adjust 'Max Nodes to Display' in the sidebar.")

        # Add nodes to the network
        for idx, row in nodes_to_draw.iterrows():  # Use nodes_to_draw
            node_id = row['doi']
            title = row['title']
            citations = row['citations'] if 'citations' in row else 0
            year = row['year']
            categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
            assay_types = ", ".join(row['kw_detected_methods']) if row['kw_detected_methods'] else "N/A"
            abstract = row['abstract']

            # Get x and y coordinates, default to 0 if not present
            node_x = row.get('x', 0)
            node_y = row.get('y', 0)

            # Assign size based on citation ranges
            size = get_node_size(citations)

            # Node color based on the FIRST pipeline category
            node_color = "#CCCCCC"  # Default grey if no category
            if row['kw_pipeline_category']:
                node_color = category_colors.get(row['kw_pipeline_category'][0], "#CCCCCC")

            # HTML-formatted tooltip for hover - ONLY TITLE
            tooltip_html = f"<b>Title:</b> {title}"  # FIX: Only title on hover

            net.add_node(
                node_id,
                label="",  # FIX: No label on the node itself
                title=tooltip_html,  # Content for the hover tooltip (now only title)
                size=size,
                color=node_color,
                borderWidth=1,
                borderWidthSelected=3,
                shape='dot',
                x=node_x,  # FIX: Use x coordinate from data
                y=node_y,  # FIX: Use y coordinate from data
                fixed=True  # FIX: Make nodes static
            )

        # Add edges based on the df_edges DataFrame, filtered by current nodes
        if show_similarity_edges and not df_edges.empty:
            # Create a set of DOIs currently in the filtered graph for efficient lookup
            current_graph_dois = set(nodes_to_draw['doi'].tolist())  # Use nodes_to_draw DOIs

            # Filter df_edges to only include edges where both source and target are in the current filtered nodes
            if 'source' in df_edges.columns and 'target' in df_edges.columns:
                filtered_edges_to_draw = df_edges[
                    (df_edges['source'].isin(current_graph_dois)) &
                    (df_edges['target'].isin(current_graph_dois))
                    ]

                for idx, edge_row in filtered_edges_to_draw.iterrows():
                    source_node = edge_row['source']
                    target_node = edge_row['target']
                    edge_value = edge_row.get('value', 1.0)  # Default value if not present
                    edge_title = f"Similarity: {edge_value:.2f}"  # Default title

                    net.add_edge(
                        source_node,
                        target_node,
                        title=edge_title,
                        color="#888888",  # Default edge color
                        width=1.5 + (edge_value * 2)  # Example: scale width by value
                    )
                if filtered_edges_to_draw.empty:
                    st.info("No similarity edges found between the currently displayed papers.")
            else:
                st.warning("Cannot show similarity edges: 'source' or 'target' column missing in your edges data.")
        elif not show_similarity_edges:
            st.info("Similarity edges are currently turned off.")
        elif df_edges.empty:
            st.warning("Cannot show similarity edges: No edge data loaded from your JSON file.")

        # Generate HTML directly from network object
        try:
            html_content = net.generate_html(notebook=True)  # Generate HTML directly

            # Inject JavaScript to handle node clicks and update query parameters
            js_injection = f"""
            <style>
                /* FIX: Remove border and padding from the network container */
                #mynetwork {{
                    border: none !important;
                    padding: 0 !important;
                }}
            </style>
            <script type="text/javascript">
                var network = null;
                function initializeNetwork() {{
                    var container = document.getElementById('mynetwork');
                    var data = {{
                        nodes: new vis.DataSet({json.dumps(net.nodes)}),
                        edges: new vis.DataSet({json.dumps(net.edges)})
                    }};
                    var options = {json.dumps(json.loads(net.options.to_json()))}; 

                    // FIX: Explicitly set node options to ensure no labels and control interaction
                    options.nodes = options.nodes || {{}};
                    options.nodes.label = ''; // Ensure no label is displayed on the node
                    options.nodes.font = options.nodes.font || {{}};
                    options.nodes.font.size = 0; // Make font size 0 to ensure it's hidden
                    options.interaction = options.interaction || {{}};
                    options.interaction.tooltipDelay = 100; // Small delay for hover tooltip
                    options.interaction.hover = true; // Enable hover
                    options.interaction.hoverConnectedEdges = false; // Don't highlight connected edges on hover
                    options.interaction.selectConnectedEdges = false; // Don't select connected edges on click
                    options.interaction.zoomView = true; // Allow zooming
                    options.interaction.dragView = true; // Allow dragging the view

                    // FIX: Ensure physics is explicitly disabled in vis.js options
                    options.physics = {{ enabled: false }};

                    network = new vis.Network(container, data, options);

                    network.on("click", function (params) {{
                        if (params.nodes.length > 0) {{
                            var nodeId = params.nodes[0];
                            console.log("Node clicked:", nodeId); // Debug log

                            // Update URL query parameter without reloading the whole page
                            var url = new URL(window.location.href);
                            url.searchParams.set('selected_doi', nodeId);
                            window.history.pushState({{{{path:url.href}}}},'',url.href); // Escaped curly braces

                            // Send message to parent (Streamlit) to trigger re-run and update state
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                key: 'selected_node_trigger', // Dummy key to trigger re-run
                                value: nodeId
                            }}, '*');

                            // FIX: Attempt to hide the native Pyvis tooltip on click
                            var tooltip = document.querySelector('.vis-tooltip');
                            if (tooltip) {{
                                tooltip.style.display = 'none';
                            }}
                            // Also hide any active tooltips from Pyvis directly
                            network.hidePopup();
                        }}
                    }});

                    // FIX: Ensure native tooltip is hidden when graph is loaded or interacted with
                    network.on("stabilized", function() {{
                        var tooltip = document.querySelector('.vis-tooltip');
                        if (tooltip) {{
                            tooltip.style.display = 'none';
                        }}
                        network.hidePopup(); // Also hide any active popups
                    }});
                    network.on("afterDrawing", function() {{
                        var tooltip = document.querySelector('.vis-tooltip');
                        if (tooltip) {{
                            tooltip.style.display = 'none';
                        }}
                        network.hidePopup(); // Also hide any active popups
                    }});
                }}
                document.addEventListener('DOMContentLoaded', initializeNetwork);
                if (document.readyState === 'complete') {{
                    initializeNetwork();
                }}
            </script>
            """
            html_content = html_content.replace('</body>', js_injection + '</body>')

            components.html(html_content, height=780, scrolling=True)
        except Exception as e:
            st.error(f"An error occurred while rendering the graph: {e}")
            st.info(
                "This might happen if there are no papers matching your filters to display, or if the 'doi' column is missing/empty.")

    else:
        st.info("No papers match the selected filters. Please adjust your criteria in the sidebar.")

# --- Details Panel (Right Column) ---
with details_col:
    st.subheader("Paper Details")
    if st.session_state.selected_doi:
        paper_info = filtered_nodes_df[filtered_nodes_df['doi'] == st.session_state.selected_doi]
        if not paper_info.empty:
            paper = paper_info.iloc[0]
            st.markdown(f"**Title:** {paper['title']}")
            st.markdown(f"**DOI:** {paper['doi']}")
            st.markdown(f"**Citations:** {paper['citations']}")
            st.markdown(f"**Year:** {paper['year']}")
            st.markdown(f"**Pipeline Categories:** {', '.join(paper['kw_pipeline_category'])}")
            st.markdown(f"**Assay Types/Platforms:** {', '.join(paper['kw_detected_methods'])}")
            st.markdown(f"**Abstract:** {paper['abstract']}")

            # Button to close/clear the details panel
            if st.button("Close Details"):
                st.session_state.selected_doi = None
                st.query_params["selected_doi"] = None  # Also clear URL param
                st.rerun()  # Force rerun to clear the panel
        else:
            st.info(
                "Select a paper from the graph to see its details here, or the selected paper is no longer in the filtered view.")
    else:
        st.info("Click on a node in the graph to display its details here.")
