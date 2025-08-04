import streamlit as st
import pandas as pd
from pyvis.network import Network
import ast  # For safely evaluating string representations of lists
import numpy as np
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

        # Create DataFrame for nodes from the 'nodes' key
        if 'nodes' in full_json_data and isinstance(full_json_data['nodes'], list):
            df_nodes = pd.DataFrame(full_json_data['nodes'])
            # Ensure 'id' is a string for node IDs and remove duplicates based on 'id'
            df_nodes['id'] = df_nodes['id'].astype(str)
            df_nodes.drop_duplicates(subset=['id'], inplace=True)  # Remove duplicates
            df_nodes['doi'] = df_nodes['id'].astype(str)  # Ensure 'doi' is a string for node IDs
        else:
            st.error(f"Error: JSON data from '{file_path}' does not contain a 'nodes' key or 'nodes' is not a list.")
            st.stop()

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
    # Handle 'citations' column for node sizing
    if 'citations' in df_nodes.columns:
        df_nodes['citations'] = pd.to_numeric(df_nodes['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df_nodes['citations'] = 0  # Default citations if column is missing
        st.warning("Column 'citations' not found in your nodes data. Node sizes will default to a minimum size.")

    # Handle 'year' column (even if not used for filtering, ensure it's numeric for consistency)
    if 'year' in df_nodes.columns:
        df_nodes['year'] = pd.to_numeric(df_nodes['year'], errors='coerce').fillna(0).astype(int)
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

    # Ensure 'title' and 'abstract' columns exist and are strings
    for col in ['title', 'abstract']:
        if col not in df_nodes.columns:
            df_nodes[col] = ''
            st.warning(f"Column '{col}' not found in your nodes data. It will be empty in display.")

    return df_nodes, df_edges


# Load data when the page is run
df_nodes, df_edges = load_data(PROCESSED_DATA_FILE)

# --- Streamlit Page Content ---

st.header("Spatial Omics Paper Network")

# --- Graph Visualization ---
# No sidebar, no filters, no search box, no details panel
# Display all unique nodes loaded

st.write(f"Displaying **{len(df_nodes)}** papers.")  # Display total unique papers loaded

if not df_nodes.empty:
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)


    # Physics is explicitly disabled in JS options below

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


    # Add nodes to the network
    for idx, row in df_nodes.iterrows():  # Iterate over all unique nodes
        node_id = row['doi']
        title = row['title']
        citations = row['citations'] if 'citations' in row else 0

        # Get x and y coordinates, default to 0 if not present
        node_x = row.get('x', 0)
        node_y = row.get('y', 0)

        # Assign size based on citation ranges
        size = get_node_size(citations)

        # Uniform node color
        node_color = "#888888"  # A neutral grey

        # HTML-formatted tooltip for hover - ONLY TITLE
        tooltip_html = f"<b>Title:</b> {title}"

        net.add_node(
            node_id,
            label="",  # No label on the node itself
            title=tooltip_html,  # Content for the hover tooltip (only title)
            size=size,
            color=node_color,
            borderWidth=1,
            borderWidthSelected=3,
            shape='dot',
            x=node_x,  # Use x coordinate from data
            y=node_y,  # Use y coordinate from data
            fixed=True  # Make nodes static
        )

    # Add edges based on the df_edges DataFrame, ensuring only existing nodes are connected
    if not df_edges.empty:
        current_graph_dois = set(df_nodes['doi'].tolist())  # All unique nodes loaded

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
                color="#D3D3D3",  # Light grey for edges
                width=1.5 + (edge_value * 2)  # Example: scale width by value
            )
        if filtered_edges_to_draw.empty:
            st.info("No similarity edges found between the displayed papers.")
    else:
        st.info("No edge data loaded from your JSON file.")

    # Generate HTML directly from network object
    try:
        html_content = net.generate_html(notebook=True)  # Generate HTML directly

        # Inject JavaScript and CSS
        js_css_injection = f"""
        <style>
            /* Remove border and padding from the network container */
            #mynetwork {{
                border: none !important;
                padding: 0 !important;
            }}
            /* Hide the native Pyvis tooltip by default */
            .vis-tooltip {{
                display: none !important;
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

                // Explicitly set node options to ensure no labels and control interaction
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

                // Ensure physics is explicitly disabled in vis.js options
                options.physics = {{ enabled: false }};

                network = new vis.Network(container, data, options);

                // No click event listener needed as there's no details panel
                // network.on("click", function (params) {{ ... }});

                // Ensure native tooltip is hidden when graph is loaded or interacted with
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
        html_content = html_content.replace('</body>', js_css_injection + '</body>')

        components.html(html_content, height=780, scrolling=True)
    except Exception as e:
        st.error(f"An error occurred while rendering the graph: {e}")
        st.info("This might happen if there are no papers matching your data.")

else:
    st.info("No papers loaded or available to display from your data file.")
```