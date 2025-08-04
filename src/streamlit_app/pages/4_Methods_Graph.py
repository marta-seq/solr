import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import ast  # Import ast for safe_literal_eval
from typing import List, Dict, Any, Optional
import logging

# Configure page
st.set_page_config(page_title="Methods Graph", layout="wide")


# Cache data loading for performance
@st.cache_data
def load_graph_data():
    """Load and preprocess graph data"""
    try:
        with open('data/methods/graph_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {"nodes": [], "links": []}


# Helper function for safe literal evaluation
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
            # This handles cases like "item1, item2" as a single string
            return [item.strip() for item in val.split(',') if item.strip()]
    elif isinstance(val, list):
        return val
    return []  # Return empty list for NaN or other unexpected types


@st.cache_data
def preprocess_data(data):
    """Preprocess data for efficient filtering and extract unique values"""
    nodes_df = pd.DataFrame(data['nodes'])
    links_df = pd.DataFrame(data['links'])

    # Ensure 'id' is string and remove duplicates based on 'id'
    if 'id' in nodes_df.columns:
        nodes_df['id'] = nodes_df['id'].astype(str)
        nodes_df.drop_duplicates(subset=['id'], inplace=True)
    else:
        st.error("Node data must contain an 'id' column for deduplication.")
        st.stop()

    # Handle missing values and convert lists for relevant columns
    list_cols_to_process = [
        "kw_pipeline_category",
        "kw_detected_methods",
        "authors"  # For display in details
    ]

    for col in list_cols_to_process:
        if col in nodes_df.columns:
            # Apply safe_literal_eval directly to parse string representations of lists
            nodes_df[col] = nodes_df[col].apply(safe_literal_eval)
        else:
            nodes_df[col] = [[] for _ in range(len(nodes_df))]  # Add empty list column if missing

    # Ensure 'title', 'abstract' columns exist and are strings
    for col in ['title', 'abstract']:
        if col not in nodes_df.columns:
            nodes_df[col] = ''
            st.warning(f"Column '{col}' not found in node data. It will be empty in display.")
        nodes_df[col] = nodes_df[col].astype(str).fillna('')  # Ensure string and no NaNs

    # Handle 'citations' column for node sizing
    if 'citations' in nodes_df.columns:
        nodes_df['citations'] = pd.to_numeric(nodes_df['citations'], errors='coerce').fillna(0).astype(int)
    else:
        nodes_df['citations'] = 0
        st.warning("Column 'citations' not found. Node sizes will default based on 0 citations.")

    # Handle 'year' column for filtering
    if 'year' in nodes_df.columns:
        nodes_df['year'] = pd.to_numeric(nodes_df['year'], errors='coerce').fillna(0).astype(int)
    else:
        nodes_df['year'] = 0  # Default year to 0 if missing
        st.warning("Column 'year' not found. Year filter might not function as expected.")

    # Handle 'x' and 'y' coordinates for node positioning
    for coord_col in ['x', 'y']:
        if coord_col in nodes_df.columns:
            nodes_df[coord_col] = pd.to_numeric(nodes_df[coord_col], errors='coerce').fillna(0)
        else:
            nodes_df[coord_col] = 0  # Default to 0 if missing
            st.warning(f"Column '{coord_col}' not found. Nodes will be positioned at 0 for this coordinate.")

    # Set 'id' as index for easier lookup
    nodes_df.set_index('id', inplace=True)

    # Ensure 'doi' column exists and is a copy of the unique 'id' (index)
    nodes_df['doi'] = nodes_df.index.astype(str)

    # Extract unique values for filtering from the processed list columns
    unique_categories = sorted(list(set(item for sublist in nodes_df["kw_pipeline_category"] for item in sublist)))
    unique_methods = sorted(list(set(item for sublist in nodes_df["kw_detected_methods"] for item in sublist)))

    # Collect only the FIRST category for coloring and legend
    unique_pipeline_parts_for_color = set()
    for category_list in nodes_df["kw_pipeline_category"]:
        if category_list:  # If the list is not empty
            unique_pipeline_parts_for_color.add(category_list[0])
    unique_pipeline_parts_for_color = sorted(list(unique_pipeline_parts_for_color))

    return {
        'nodes_df': nodes_df,
        'links_df': links_df,
        'unique_categories': unique_categories,
        'unique_methods': unique_methods,
        'unique_pipeline_parts_for_color': unique_pipeline_parts_for_color,
        'category_col': "kw_pipeline_category",
        'methods_col': "kw_detected_methods",
        'pipeline_col': "kw_pipeline_category"  # Column used for coloring
    }


def get_node_color(pipeline_parts, color_map):
    """Get color for node based on pipeline parts (first one in list)"""
    if isinstance(pipeline_parts, list) and len(pipeline_parts) > 0:
        return color_map.get(pipeline_parts[0], '#808080')  # Default grey
    else:
        return '#808080'  # Gray for None/empty


def get_node_size(citations):
    """Get node size based on citation count (5 classes)"""
    if citations <= 4:
        return 5
    elif 5 <= citations <= 24:
        return 8
    elif 25 <= citations <= 49:
        return 12
    elif 50 <= citations <= 99:
        return 16
    else:  # citations >= 100
        return 20


def filter_nodes(nodes_df, year_range, selected_categories, selected_methods,
                 category_col, methods_col):
    """Filter nodes based on criteria"""
    filtered_df = nodes_df.copy()

    # Year filter
    if 'year' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_range[0]) &
            (filtered_df['year'] <= year_range[1])
            ]

    # Category filter: Only apply if categories are selected
    if selected_categories and category_col in filtered_df.columns:
        mask = filtered_df[category_col].apply(
            lambda x: any(cat in x for cat in selected_categories)
        )
        filtered_df = filtered_df[mask]

    # Methods filter: Only apply if methods are selected
    if selected_methods and methods_col in filtered_df.columns:
        mask = filtered_df[methods_col].apply(
            lambda x: any(method in x for method in selected_methods)
        )
        filtered_df = filtered_df[mask]

    return filtered_df


def create_graph(filtered_nodes_df, links_df, show_edges, processed_data):
    """Create the interactive graph using Plotly"""
    if filtered_nodes_df.empty:
        return go.Figure()

    # Create color mapping for pipeline categories
    unique_pipeline_parts = processed_data['unique_pipeline_parts_for_color']
    # Use a qualitative color scale, extend if needed
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    color_map = dict(zip(unique_pipeline_parts, colors[:len(unique_pipeline_parts)]))

    # Prepare node data
    node_x = filtered_nodes_df['x'].tolist()
    node_y = filtered_nodes_df['y'].tolist()
    node_colors = [get_node_color(row[processed_data['pipeline_col']], color_map) for idx, row in
                   filtered_nodes_df.iterrows()]
    node_sizes = [get_node_size(row['citations']) for idx, row in filtered_nodes_df.iterrows()]
    node_text = filtered_nodes_df['title'].tolist()  # Only title for hover
    node_ids = filtered_nodes_df.index.tolist()  # Use index (which is 'id') for customdata

    fig = go.Figure()

    # Add edges if enabled
    if show_edges and not links_df.empty:
        edge_x = []
        edge_y = []

        # Filter links to only include those between currently displayed nodes
        # Ensure source/target are in the filtered_nodes_df index
        valid_links_df = links_df[
            links_df['source'].isin(filtered_nodes_df.index) &
            links_df['target'].isin(filtered_nodes_df.index)
            ]

        for _, link in valid_links_df.iterrows():
            source_id = link.get('source')
            target_id = link.get('target')

            source_row = filtered_nodes_df.loc[source_id]
            target_row = filtered_nodes_df.loc[target_id]

            edge_x.extend([source_row.get('x', 0), target_row.get('x', 0), None])
            edge_y.extend([source_row.get('y', 0), target_row.get('y', 0), None])

        if edge_x:  # Only add if there are edges to show
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='rgba(125,125,125,0.3)'),
                hoverinfo='none',
                showlegend=False,
                name='edges'
            ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color='white'),
            opacity=0.8
        ),
        text=node_text,
        hovertemplate='<b>%{text}</b>',  # Only title on hover
        customdata=node_ids,  # Pass node IDs for click handling
        name='nodes'
    ))

    # Remove Plotly's built-in legend and legend items, we'll create a custom one
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )

    return fig


def display_node_details(node_data, processed_data):
    """Display detailed information about selected node"""
    st.subheader(f"📄 {node_data.get('title', 'No title')}")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Year:** {node_data.get('year', 'N/A')}")
        st.write(f"**Citations:** {node_data.get('citations', 'N/A')}")

        # Authors
        authors = node_data.get('authors', 'N/A')
        if isinstance(authors, list):
            authors = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
        st.write(f"**Authors:** {authors}")

    with col2:
        # Pipeline parts
        pipeline_parts = node_data.get(processed_data['pipeline_col'])
        if isinstance(pipeline_parts, list):
            pipeline_parts = ', '.join(pipeline_parts)
        st.write(f"**Pipeline Categories:** {pipeline_parts}")

        # Categories (using category_col from processed_data)
        categories = node_data.get(processed_data['category_col'])
        if isinstance(categories, list):
            categories = ', '.join(categories)
        st.write(f"**All Categories:** {categories}")

    # Methods
    methods = node_data.get(processed_data['methods_col'])
    if isinstance(methods, list) and methods:
        st.write("**Detected Methods:**")
        st.write(', '.join(methods))

    # Abstract if available
    if 'abstract' in node_data:
        st.write("**Abstract:**")
        st.write(node_data['abstract'])


# Main app
def main():
    st.title("🔬 Methods Graph Visualization")

    # Load and preprocess data
    with st.spinner("Loading graph data..."):
        raw_data = load_graph_data()
        processed_data = preprocess_data(raw_data)

    nodes_df = processed_data['nodes_df']
    links_df = processed_data['links_df']

    # Create layout columns
    left_sidebar = st.sidebar
    main_col, right_panel = st.columns([3, 1])  # Adjusting column ratio for better graph spread

    # Sidebar filters
    with left_sidebar:
        st.header("🔍 Filters")

        # Year filter
        if 'year' in nodes_df.columns and not nodes_df['year'].empty and nodes_df['year'].min() != nodes_df[
            'year'].max():
            min_year = int(nodes_df['year'].min())
            max_year = int(nodes_df['year'].max())
            year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year),
                key="year_filter"
            )
        else:
            st.info("Year data not available or uniform. Year filter disabled.")
            year_range = (0, 9999)  # Broad range if no year data

        # Category filter
        st.subheader("Pipeline Categories")
        selected_categories = st.multiselect(
            "Select categories:",
            options=processed_data['unique_categories'],
            # Default to empty list, so no filter applied initially
            default=[],
            key="category_filter"
        )

        # Methods filter
        st.subheader("Detected Methods")
        selected_methods = st.multiselect(
            "Select methods:",
            options=processed_data['unique_methods'],
            # Default to empty list, so no filter applied initially
            default=[],
            key="methods_filter"
        )

        # Edge toggle
        show_edges = st.checkbox("Show similarity edges", value=True, key="edge_toggle")

        st.markdown("---")
        st.subheader("📊 Statistics")

    # Filter nodes
    filtered_nodes = filter_nodes(
        nodes_df, year_range, selected_categories, selected_methods,
        processed_data['category_col'], processed_data['methods_col']
    )

    # Display paper count in sidebar
    with left_sidebar:
        st.metric("Papers Displayed", len(filtered_nodes))

    # Main graph
    with main_col:
        if not filtered_nodes.empty:
            fig = create_graph(filtered_nodes, links_df, show_edges, processed_data)

            # Configure Plotly chart for click events
            plotly_config = {
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                                           'autoscale', 'resetscale', 'hoverclosest', 'v1hovermode', 'togglespikelines',
                                           'togglehover', 'sendDataToCloud', 'resetViews'],
                'displaylogo': False,
                'staticPlot': False  # Allow interaction
            }

            # Capture click events and update session state
            clicked_data = st.plotly_chart(
                fig,
                use_container_width=True,
                key="main_graph",
                on_click="select",  # Use on_click to capture selection
                config=plotly_config
            )

            # Check if a node was clicked and update selected_node
            if clicked_data and clicked_data['points']:
                # The customdata contains the original node 'id' (DOI)
                clicked_node_id = clicked_data['points'][0]['customdata']
                if st.session_state.selected_node != clicked_node_id:
                    st.session_state.selected_node = clicked_node_id
                    st.rerun()  # Rerun to update the right panel

        else:
            st.warning("No papers match the current filters.")

    # Right panel for node details
    with right_panel:
        st.header("📋 Details")

        # Store selected node in session state
        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = None

        # Display details if a node is selected
        if st.session_state.selected_node is not None and st.session_state.selected_node in filtered_nodes.index:
            node_data = filtered_nodes.loc[st.session_state.selected_node]
            display_node_details(node_data.to_dict(), processed_data)
        else:
            st.info("Click on a node to see details.")

        # --- Custom Color Legend ---
        # FIX: Changed subheader to markdown for smaller title
        st.markdown("**🎨 Color Legend (Pipeline Category)**")
        unique_pipeline_parts = processed_data['unique_pipeline_parts_for_color']
        colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
        color_map = dict(zip(unique_pipeline_parts, colors[:len(unique_pipeline_parts)]))

        for category, color in color_map.items():
            st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
                        f"<div style='width: 15px; height: 15px; background-color: {color}; border-radius: 3px; margin-right: 8px;'></div>"
                        f"<span>{category}</span>"
                        f"</div>", unsafe_allow_html=True)

        # Show legend for node sizes
        # FIX: Changed subheader to markdown for smaller title
        st.markdown("**📏 Node Size Legend (Citations)**")
        st.write("• Size 5: 0-4 citations")
        st.write("• Size 8: 5-24 citations")
        st.write("• Size 12: 25-49 citations")
        st.write("• Size 16: 50-99 citations")
        st.write("• Size 20: 100+ citations")


if __name__ == "__main__":
    main()
