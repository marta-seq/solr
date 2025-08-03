import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import ast # For safely evaluating string representations of lists
import numpy as np
import colorsys # For generating distinct colors
import hashlib # For consistent color hashing
import json # For handling JSON specific errors
import plotly.express as px # Import plotly express for bar charts
import plotly.graph_objects as go # Import plotly graph objects for layout updates

# --- Configuration ---
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Paper Network Explorer")

# Define the path to your processed data file.
# --- IMPORTANT CHANGE: Updated file path as per your instruction ---
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

def get_exploded_counts(df, column_name):
    """
    Explodes a list-like column and returns a DataFrame with 'item' and 'count'
    for use in bar charts.
    """
    if column_name not in df.columns or df[column_name].isnull().all():
        return pd.DataFrame(columns=['item', 'count'])

    exploded_series = df[column_name].explode()
    if exploded_series.empty:
        return pd.DataFrame(columns=['item', 'count'])

    # Ensure the exploded items are strings to prevent issues with value_counts
    exploded_series = exploded_series.astype(str)

    counts = exploded_series.value_counts().reset_index()
    # IMPORTANT: Ensure these column names are lowercase 'item' and 'count'
    counts.columns = ['item', 'count']
    return counts

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
            df[col] = df[col].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)
        else:
            df[col] = [[] for _ in range(len(df))]
            st.warning(f"Column '{col}' not found in your data. It will be treated as empty.")

    # Confirming 'citations' column handling, as it's correctly named.
    if 'citations' in df.columns:
        df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
    else:
        df['citations'] = 0
        st.warning("Column 'citations' not found in your data. Node sizes will default to a minimum size.")

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    else:
        df['year'] = 2000
        st.warning("Column 'year' not found in your data. Defaulting to year 2000.")

    for col in ['doi', 'title', 'abstract']:
        if col not in df.columns:
            df[col] = ''
            st.warning(f"Column '{col}' not found in your data. It will be empty in display.")

    return df

df = load_data(PROCESSED_DATA_FILE)

# --- Streamlit Layout ---
st.title("Paper Network and Analytics Dashboard")

# Create tabs for different views
tab1, tab2 = st.tabs(["Network Graph", "Data Analytics"])

with tab2: # Data Analytics Tab
    st.header("Data Distribution Insights")

    df_papers_raw = df.copy()

    chart_columns = {
        'kw_pipeline_category': 'Pipeline Category',
        'llm_annot_tested_assay_types_platforms': 'Assay Type/Platform',
        'llm_annot_tested_data_modalities': 'Data Modality',
        'kw_tissue': 'Tissue'
    }

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    chart_cols_list = [col1, col2, col3, col4]
    for i, (col_name, chart_title_suffix) in enumerate(chart_columns.items()):
        with chart_cols_list[i % len(chart_cols_list)]:
            st.subheader(f"Top 10 {chart_title_suffix}")
            counts_df = get_exploded_counts(df_papers_raw, col_name)
            if not counts_df.empty:
                # --- FIX: Using 'count' for x-axis and 'item' for y-axis ---
                fig = px.bar(
                    counts_df.head(10),
                    x='count', # Corrected to lowercase 'count'
                    y='item',  # Corrected to lowercase 'item'
                    title=f"Top 10 {chart_title_suffix}",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No data available for {chart_title_suffix}.")

with tab1: # Network Graph Tab
    st.header("Paper Network Visualization")

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Papers")

    if not df.empty and 'year' in df.columns:
        min_year, max_year = int(df['year'].min()), int(df['year'].max())
        selected_year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        st.sidebar.info("No data loaded or 'year' column missing to set year filter.")
        selected_year_range = (0, 9999)

    all_categories = sorted(list(set(item for sublist in df['kw_pipeline_category'] for item in sublist))) if 'kw_pipeline_category' in df.columns else []
    all_assay_types = sorted(list(set(item for sublist in df['llm_annot_tested_assay_types_platforms'] for item in sublist))) if 'llm_annot_tested_assay_types_platforms' in df.columns else []
    all_data_modalities = sorted(list(set(item for sublist in df['llm_annot_tested_data_modalities'] for item in sublist))) if 'llm_annot_tested_data_modalities' in df.columns else []

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
    filtered_df = df[
        (df['year'] >= selected_year_range[0]) &
        (df['year'] <= selected_year_range[1])
    ]

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

        all_categories_for_colors = sorted(list(set(item for sublist in df['kw_pipeline_category'] for item in sublist)))
        category_colors = {cat: get_hashed_color(cat) for cat in all_categories_for_colors}

        for idx, row in filtered_df.iterrows():
            node_id = row['doi']
            title = row['title']
            citations = row['citations'] if 'citations' in row else 0
            year = row['year']
            categories = ", ".join(row['kw_pipeline_category']) if row['kw_pipeline_category'] else "N/A"
            assay_types = ", ".join(row['llm_annot_tested_assay_types_platforms']) if row['llm_annot_tested_assay_types_platforms'] else "N/A"
            data_modalities = ", ".join(row['llm_annot_tested_data_modalities']) if row['llm_annot_tested_data_modalities'] else "N/A"
            abstract = row['abstract']

            if max_citations > 0:
                size = 10 + (np.log1p(citations) / np.log1p(max_citations + 1)) * 30
            else:
                size = 10

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

        if show_similarity_edges and 'similar_papers' in filtered_df.columns:
            for idx, row in filtered_df.iterrows():
                source_doi = row['doi']
                if source_doi in [node['id'] for node in net.nodes]:
                    for target_doi in row['similar_papers']:
                        if target_doi in filtered_df['doi'].values and target_doi in [node['id'] for node in net.nodes]:
                            net.add_edge(source_doi, target_doi, title="Similarity", color="#888888", width=1.5)
        elif not show_similarity_edges:
            st.info("Similarity edges are currently turned off.")
        elif 'similar_papers' not in filtered_df.columns:
            st.warning("Cannot show similarity edges: 'similar_papers' column not found in your data.")

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

