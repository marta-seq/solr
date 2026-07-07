# streamlit_app.py
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np  # For log1p
import os
from streamlit_plotly_events import plotly_events  # <-- NEW IMPORT

# --- Configuration ---
# Adjust this path if your 'data' directory is located differently relative to where you run the Streamlit app
GRAPH_DATA_FILE = "data/methods/graph_data.json"

# Node sizing parameters (can be adjusted via Streamlit sliders in the app)
DEFAULT_BASE_NODE_SIZE = 8
DEFAULT_CITATIONS_SCALE_FACTOR = 100
DEFAULT_CITATIONS_MULTIPLIER = 5


# --- Helper Functions ---
@st.cache_data  # Cache data loading for performance
def load_graph_data(file_path):
    """Loads graph data from the JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Graph data file not found at: {file_path}")
        st.warning("Please ensure `generate_graph_data.py` has been run successfully to create this file.")
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['nodes'], data['edges']
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from '{file_path}': {e}. The file might be corrupted or malformed.")
        return [], []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading graph data: {e}")
        return [], []


def get_short_hover_text(node):
    """Generates concise hover text (only title) for a node."""
    title = node.get('title', 'N/A')
    return f"<b>{title}</b>"


# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="Methodology Graph Explorer")

st.title("Methodology Graph Explorer")

nodes, edges = load_graph_data(GRAPH_DATA_FILE)

if not nodes:
    st.warning("No graph data available to display. Please check the console for error messages.")
    st.stop()

# --- Prepare Data for Filtering ---
df_nodes = pd.DataFrame(nodes)

# Extract all unique values for filtering from raw lists
# Use .explode() for lists in columns for easier filtering of individual tags
df_exploded_pipeline = df_nodes.explode('raw_pipeline_steps')
all_pipeline_steps = sorted(df_exploded_pipeline['raw_pipeline_steps'].dropna().unique())

df_exploded_modalities = df_nodes.explode('raw_tested_data_modalities')
all_data_modalities = sorted(df_exploded_modalities['raw_tested_data_modalities'].dropna().unique())

all_years = sorted(df_nodes['year'].dropna().astype(str).unique(),
                   reverse=True)  # Convert year to string for consistent filter

# --- Sidebar Filters ---
st.sidebar.header("Graph Filters & Options")

# Search Bar
search_query = st.sidebar.text_input("Search by Title/Abstract", "")

# Year Filter
selected_years = st.sidebar.multiselect("Filter by Year", all_years, default=all_years)

# Pipeline Steps Filter
selected_pipeline_steps = st.sidebar.multiselect("Filter by Pipeline Steps", all_pipeline_steps)

# Data Modalities Filter
selected_data_modalities = st.sidebar.multiselect("Filter by Data Modalities", all_data_modalities)

# Coloring Option
color_by_option = st.sidebar.selectbox(
    "Color Nodes By",
    ["None", "Tool Type", "Year", "Annotation Status"],  # Add more options if desired
    index=1  # Default to Tool Type
)

# Size Option
st.sidebar.subheader("Node Sizing Options")
size_by_citations = st.sidebar.checkbox("Size Nodes by Citations", value=True)
base_node_size = st.sidebar.slider("Base Node Size", 3, 20, DEFAULT_BASE_NODE_SIZE)
if size_by_citations:
    citations_scale_factor = st.sidebar.slider("Citation Scale Factor", 1, 500, DEFAULT_CITATIONS_SCALE_FACTOR)
    citations_multiplier = st.sidebar.slider("Citation Size Multiplier", 1, 20, DEFAULT_CITATIONS_MULTIPLIER)

# --- Filter Nodes ---
filtered_nodes = []
for node in nodes:
    keep_node = True

    # Year filter
    if selected_years and str(node.get('year')) not in selected_years:
        keep_node = False

    # Pipeline Steps filter (check if any selected step is in node's raw_pipeline_steps)
    if selected_pipeline_steps:
        node_steps = node.get('raw_pipeline_steps', [])
        if not any(step in node_steps for step in selected_pipeline_steps):
            keep_node = False

    # Data Modalities filter (check if any selected modality is in node's raw_tested_data_modalities)
    if selected_data_modalities:
        node_modalities = node.get('raw_tested_data_modalities', [])
        if not any(modality in node_modalities for modality in selected_data_modalities):
            keep_node = False

    # Search filter
    if search_query:
        title = node.get('title', '').lower()
        abstract = node.get('abstract', '').lower()
        search_lower = search_query.lower()
        if search_lower not in title and search_lower not in abstract:
            keep_node = False

    if keep_node:
        filtered_nodes.append(node)

if not filtered_nodes:
    st.info("No nodes match the current filter criteria.")
    st.stop()

# --- Create Graph ---
x_coords = [node['x'] for node in filtered_nodes]
y_coords = [node['y'] for node in filtered_nodes]
node_ids = [node['id'] for node in filtered_nodes]

# Node Colors
node_colors = []
color_map = {}  # To store color mapping for legend

if color_by_option == "None":
    node_colors = ['lightgray'] * len(filtered_nodes)
    color_map['All Nodes'] = 'lightgray'
elif color_by_option == "Tool Type":
    unique_tool_types = sorted(list(set(node.get('llm_annot_tool_type', 'Unknown') for node in filtered_nodes)))
    colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf']  # D3 color palette
    for i, tt in enumerate(unique_tool_types):
        color_map[tt] = colors_palette[i % len(colors_palette)]
    node_colors = [color_map.get(node.get('llm_annot_tool_type', 'Unknown')) for node in filtered_nodes]
elif color_by_option == "Year":
    # Create a continuous colormap or distinct for years
    unique_years_int = sorted(
        list(set(int(node.get('year', 0)) for node in filtered_nodes if str(node.get('year')).isdigit())))
    if len(unique_years_int) > 1:
        min_year, max_year = min(unique_years_int), max(unique_years_int)
        from plotly.colors import get_colorscale, sample_colorscale

        colorscale = get_colorscale('Plasma')
        for year_val in unique_years_int:
            norm_year = (year_val - min_year) / (max_year - min_year) if max_year > min_year else 0
            color_map[str(year_val)] = sample_colorscale(colorscale, norm_year)[0]
        node_colors = [color_map.get(str(node.get('year', 0))) for node in filtered_nodes]
    else:
        node_colors = ['lightgray'] * len(filtered_nodes)
        color_map[str(unique_years_int[0]) if unique_years_int else 'N/A'] = 'lightgray'
elif color_by_option == "Annotation Status":
    unique_statuses = sorted(list(set(node.get('annotation_status', 'Unknown') for node in filtered_nodes)))
    colors_palette = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    for i, status in enumerate(unique_statuses):
        color_map[status] = colors_palette[i % len(colors_palette)]
    node_colors = [color_map.get(node.get('annotation_status', 'Unknown')) for node in filtered_nodes]

# Node Sizes
node_sizes = []
for node in filtered_nodes:
    size = base_node_size  # Start with base size
    if size_by_citations:
        citations_val = node.get('citations')
        citations_for_log = 0
        if citations_val is not None:
            try:
                citations_for_log = float(citations_val)
            except (ValueError, TypeError):
                citations_for_log = 0
        # Add a scaled value based on citations, log1p helps with large ranges
        size += np.log1p(citations_for_log / citations_scale_factor) * citations_multiplier
    node_sizes.append(size)

# Create Plotly traces
# Nodes Trace
node_trace = go.Scatter(
    x=x_coords,
    y=y_coords,
    mode='markers',
    hoverinfo='text',
    text=[get_short_hover_text(node) for node in filtered_nodes],  # Short hover text
    marker=dict(
        size=node_sizes,
        color=node_colors,
        line_width=1,
        line_color='black',
        sizemode='diameter'
    ),
    customdata=[node for node in filtered_nodes]  # Store full node data here for click events
)

data = [node_trace]  # Only nodes for simplicity for now

# Create Plotly Figure
fig = go.Figure(data=data,
                layout=go.Layout(
                    title=dict(
                        text='<br>UMAP Visualization of Methodologies',
                        font=dict(size=16)
                    ),
                    showlegend=True,  # Display legend for colors
                    hovermode='closest',  # For the small hover tooltip
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(
                        text="UMAP projection of papers based on embedding similarity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=700  # Fixed height for the graph
                ))

# Add a legend dynamically for colors
if color_by_option != "None":
    for label, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],  # Invisible trace for legend entry
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            name=label,
            legendgroup=label,
            showlegend=True
        ))

# --- Display Graph and Details ---
# Use columns for layout: graph on left, details on right
graph_column, detail_column = st.columns([0.7, 0.3])  # Adjust column ratios for your screen

with graph_column:
    st.subheader("Interactive Graph")
    # NEW: Use plotly_events to capture clicks
    selected_points = plotly_events(
        fig,
        select_event=True,  # Capture selection events (clicks)
        key="plotly_graph_events"  # Unique key for the component
    )

# Detail view for selected node
with detail_column:
    st.subheader("Selected Paper Details")

    display_node = None

    if selected_points:
        # If a point was clicked, use its customdata
        # customdata is a list of node dictionaries, indexed by the point_index
        point_index = selected_points[0]['pointIndex']
        display_node = node_trace.customdata[point_index]
    elif search_query:
        # If no click, but a search query is active, try to find the best match for initial display
        for node in filtered_nodes:
            if search_query.lower() == node.get('title', '').lower():
                display_node = node
                break
        if not display_node:
            for node in filtered_nodes:
                if search_query.lower() in node.get('title', '').lower():
                    display_node = node
                    break
        if not display_node:
            for node in filtered_nodes:
                if search_query.lower() in node.get('abstract', '').lower():
                    display_node = node
                    break

    if not display_node and filtered_nodes:
        # Fallback to the first filtered node if no click and no search match
        display_node = filtered_nodes[0]

    if display_node:
        st.markdown(f"### {display_node.get('title', 'No Title')}")
        st.markdown(f"**DOI:** [{display_node.get('doi', 'N/A')}](https://doi.org/{display_node.get('doi', '')})")
        st.markdown(f"**Year:** {display_node.get('year', 'N/A')}")
        st.markdown(f"**Citations:** {display_node.get('citations', 'N/A')}")
        st.markdown(f"**Journal:** {display_node.get('journal', 'N/A')}")
        st.markdown(f"**Tool Type:** {display_node.get('llm_annot_tool_type', 'N/A')}")

        st.markdown("---")
        st.markdown("**Abstract:**")
        st.text_area("Abstract Content", value=display_node.get('abstract', 'N/A'), height=200, disabled=True)

        st.markdown("---")
        st.markdown("**Raw Pipeline Steps:**")
        st.write(display_node.get('raw_pipeline_steps', []))

        st.markdown("**Raw Data Modalities:**")
        st.write(display_node.get('raw_tested_data_modalities', []))

        st.markdown("**Raw Author Keywords:**")
        st.write(display_node.get('raw_author_keywords', []))

        # Additional LLM annotated fields
        st.markdown("---")
        st.markdown("**LLM Annotated Details:**")
        st.markdown(f"**Main Goal:** {display_node.get('llm_annot_main_goal_of_paper', 'N/A')}")
        st.markdown(f"**Algorithms Used:** {display_node.get('llm_annot_algorithms_bases_used', 'N/A')}")
        st.markdown(f"**Code Availability:** {display_node.get('llm_annot_code_availability_details', 'N/A')}")
        st.markdown(f"**Compared Algorithms:** {display_node.get('llm_annot_compared_algorithms_packages', 'N/A')}")
    else:
        st.info("Click a node on the graph or use filters/search to see details here.")

# Add instructions on how to run
st.sidebar.markdown("---")
st.sidebar.markdown("**How to Interact:**")
st.sidebar.markdown("- Use filters and search to narrow down nodes.")
st.sidebar.markdown("- **Click a node on the graph to see its details in the right panel.**")
st.sidebar.markdown("- Hover over nodes on the graph for a quick summary (only title).")