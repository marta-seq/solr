import streamlit as st
import pandas as pd
import json
from pyvis.network import Network
import numpy as np
import ast
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the data
@st.cache_data
def load_data(file_path):
    nodes = []
    links = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                if 'nodes' in data:
                    nodes.extend(data['nodes'])
                if 'links' in data:
                    links.extend(data['links'])
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON on line {line_number}: {e}")
                continue
    return pd.DataFrame(nodes), pd.DataFrame(links)

# Define the path to your data file
DATA_FILE = 'data/methods/graph_data.json'

# Load the data
try:
    df_nodes, df_links = load_data(DATA_FILE)
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# Define the column names
PIPELINE_PART_COLUMN = 'kw_pipeline_category'  # Replace with your actual column name
CATEGORY_COLUMN = 'category'  # Replace with your actual column name
DETECTED_METHODS_COLUMN = 'kw_detected_methods'  # Replace with your actual column name

# Ensure the columns are treated as lists
def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(val, list):
        return val
    return []

if PIPELINE_PART_COLUMN in df_nodes.columns:
    df_nodes[PIPELINE_PART_COLUMN] = df_nodes[PIPELINE_PART_COLUMN].apply(safe_literal_eval)
if CATEGORY_COLUMN in df_nodes.columns:
    df_nodes[CATEGORY_COLUMN] = df_nodes[CATEGORY_COLUMN].apply(safe_literal_eval)
if DETECTED_METHODS_COLUMN in df_nodes.columns:
    df_nodes[DETECTED_METHODS_COLUMN] = df_nodes[DETECTED_METHODS_COLUMN].apply(safe_literal_eval)

# Function to get unique values from a list column
def get_unique_values(df, column):
    if column not in df.columns:
        return []
    unique_values = set()
    for sublist in df[column]:
        if isinstance(sublist, list):
            for item in sublist:
                unique_values.add(item)
    return sorted(unique_values)

# Get unique values for filters
unique_categories = get_unique_values(df_nodes, CATEGORY_COLUMN)
unique_methods = get_unique_values(df_nodes, DETECTED_METHODS_COLUMN)
unique_years = sorted(df_nodes['year'].unique()) if 'year' in df_nodes.columns else []

# Streamlit app
st.title("Paper Network Visualization")

# Sidebar filters
st.sidebar.header("Filters")

# Year filter
selected_years = st.sidebar.multiselect(
    "Filter by Year",
    options=unique_years,
    default=unique_years
)

# Category filter
selected_categories = st.sidebar.multiselect(
    "Filter by Category",
    options=unique_categories,
    default=unique_categories
)

# Detected Methods filter
selected_methods = st.sidebar.multiselect(
    "Filter by Detected Methods",
    options=unique_methods,
    default=unique_methods
)

# Toggle for edges
show_edges = st.sidebar.checkbox("Show Edges", value=True)

# Apply filters
filtered_df = df_nodes.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
if selected_categories:
    filtered_df = filtered_df[filtered_df[CATEGORY_COLUMN].apply(lambda x: any(cat in selected_categories for cat in x))]
if selected_methods:
    filtered_df = filtered_df[filtered_df[DETECTED_METHODS_COLUMN].apply(lambda x: any(method in selected_methods for method in x))]

st.write(f"Displaying {len(filtered_df)} papers based on current filters.")

# Generate the graph
net = Network(height="750px", width="100%", notebook=True, cdn_resources='remote', directed=False)
net.toggle_physics(True)

# Function to determine node size based on citations
def get_node_size(citations):
    if citations < 5:
        return 10
    elif citations < 25:
        return 20
    elif citations < 75:
        return 30
    elif citations < 150:
        return 40
    else:
        return 50

# Generate colors for categories
unique_pipeline_parts = get_unique_values(df_nodes, PIPELINE_PART_COLUMN)
color_map = {part: f"#{np.random.choice(range(256), 3).tobytes().hex()}" for part in unique_pipeline_parts}

# Add nodes to the network
for idx, row in filtered_df.iterrows():
    node_id = row['id']
    title = row['title']
    citations = row.get('citations', 0)
    x = row.get('x', None)
    y = row.get('y', None)
    pipeline_part = row[PIPELINE_PART_COLUMN][0] if row[PIPELINE_PART_COLUMN] else None

    size = get_node_size(citations)
    color = color_map.get(pipeline_part, "#CCCCCC")

    node_kwargs = {
        'id': node_id,
        'label': "",
        'title': title,
        'size': size,
        'color': color
    }
    if x is not None and y is not None:
        node_kwargs['x'] = x
        node_kwargs['y'] = y

    net.add_node(**node_kwargs)

# Add edges to the network
if show_edges and not df_links.empty:
    for idx, row in df_links.iterrows():
        source = row['source']
        target = row['target']
        if source in filtered_df['id'].values and target in filtered_df['id'].values:
            net.add_edge(source, target)

# Save and display the graph
try:
    path = "paper_network.html"
    net.save_graph(path)
    with open(path, 'r', encoding='utf-8') as html_file:
        html_content = html_file.read()
    st.components.v1.html(html_content, height=750, scrolling=True)
except Exception as e:
    st.error(f"An error occurred while rendering the graph: {e}")

# Display detailed information on click (placeholder for actual implementation)
st.sidebar.header("Detailed Information")
st.sidebar.write("Click on a node to see detailed information here.")
