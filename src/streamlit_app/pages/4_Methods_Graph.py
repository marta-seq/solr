import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Configure page
st.set_page_config(page_title="Methods Graph", layout="wide")


# Cache data loading for performance
@st.cache_data
def load_graph_data():
    """Load and preprocess graph data"""
    try:
        with open('data/methods/graph_data.json.jsonl', 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {"nodes": [], "links": []}


@st.cache_data
def preprocess_data(data):
    """Preprocess data for efficient filtering"""
    nodes_df = pd.DataFrame(data['nodes'])
    links_df = pd.DataFrame(data['links'])

    # Handle missing values and convert lists
    for col in nodes_df.columns:
        if nodes_df[col].dtype == 'object':
            nodes_df[col] = nodes_df[col].fillna('None')

    # Extract unique values for filtering
    category_col = "CATEGORY_COLUMN_NAME"  # Replace with actual column name
    methods_col = "DETECTED_METHODS_COLUMN_NAME"  # Replace with actual column name
    pipeline_col = "PIPELINE_PART_COLUMN_NAME"  # Replace with actual column name

    # Get unique values from list columns
    unique_categories = set()
    unique_methods = set()
    unique_pipeline_parts = set()

    for idx, row in nodes_df.iterrows():
        # Handle category column
        if category_col in row and row[category_col] is not None:
            if isinstance(row[category_col], list):
                unique_categories.update(row[category_col])
            else:
                unique_categories.add(str(row[category_col]))
        else:
            unique_categories.add('None')

        # Handle methods column
        if methods_col in row and row[methods_col] is not None:
            if isinstance(row[methods_col], list):
                unique_methods.update(row[methods_col])
            else:
                unique_methods.add(str(row[methods_col]))
        else:
            unique_methods.add('None')

        # Handle pipeline parts column for coloring
        if pipeline_col in row and row[pipeline_col] is not None:
            if isinstance(row[pipeline_col], list):
                unique_pipeline_parts.update(row[pipeline_col])
            else:
                unique_pipeline_parts.add(str(row[pipeline_col]))
        else:
            unique_pipeline_parts.add('None')

    return {
        'nodes_df': nodes_df,
        'links_df': links_df,
        'unique_categories': sorted(list(unique_categories)),
        'unique_methods': sorted(list(unique_methods)),
        'unique_pipeline_parts': sorted(list(unique_pipeline_parts)),
        'category_col': category_col,
        'methods_col': methods_col,
        'pipeline_col': pipeline_col
    }


def get_node_color(pipeline_parts, color_map):
    """Get color for node based on pipeline parts"""
    if isinstance(pipeline_parts, list) and len(pipeline_parts) > 0:
        return color_map.get(pipeline_parts[0], '#808080')
    elif pipeline_parts and pipeline_parts != 'None':
        return color_map.get(str(pipeline_parts), '#808080')
    else:
        return '#808080'  # Gray for None/empty


def get_node_size(citations):
    """Get node size based on citation count"""
    try:
        cit_count = int(citations) if citations else 0
    except:
        cit_count = 0

    if cit_count == 0:
        return 8
    elif cit_count < 5:
        return 12
    elif cit_count < 25:
        return 16
    elif cit_count < 75:
        return 20
    elif cit_count < 150:
        return 24
    else:
        return 28


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

    # Category filter
    if selected_categories and category_col in filtered_df.columns:
        mask = filtered_df[category_col].apply(
            lambda x: any(cat in (x if isinstance(x, list) else [str(x)])
                          for cat in selected_categories) if x is not None else False
        )
        filtered_df = filtered_df[mask]

    # Methods filter
    if selected_methods and methods_col in filtered_df.columns:
        mask = filtered_df[methods_col].apply(
            lambda x: any(method in (x if isinstance(x, list) else [str(x)])
                          for method in selected_methods) if x is not None else False
        )
        filtered_df = filtered_df[mask]

    return filtered_df


def create_graph(filtered_nodes_df, links_df, show_edges, processed_data):
    """Create the interactive graph"""
    if filtered_nodes_df.empty:
        return go.Figure()

    # Create color mapping
    unique_pipeline_parts = processed_data['unique_pipeline_parts']
    colors = px.colors.qualitative.Set3[:len(unique_pipeline_parts)]
    color_map = dict(zip(unique_pipeline_parts, colors))

    # Get filtered node IDs
    filtered_node_ids = set(filtered_nodes_df.index)

    # Prepare node data
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_ids = []

    for idx, row in filtered_nodes_df.iterrows():
        node_x.append(row.get('x', 0))
        node_y.append(row.get('y', 0))

        # Color based on pipeline parts
        pipeline_parts = row.get(processed_data['pipeline_col'])
        node_colors.append(get_node_color(pipeline_parts, color_map))

        # Size based on citations
        citations = row.get('citations', 0)
        node_sizes.append(get_node_size(citations))

        # Hover text (title only)
        title = row.get('title', 'No title')
        node_text.append(title)
        node_ids.append(idx)

    fig = go.Figure()

    # Add edges if enabled
    if show_edges and not links_df.empty:
        edge_x = []
        edge_y = []

        for _, link in links_df.iterrows():
            source_id = link.get('source')
            target_id = link.get('target')

            # Only show edges between filtered nodes
            if source_id in filtered_node_ids and target_id in filtered_node_ids:
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
        hovertemplate='<b>%{text}</b><extra></extra>',
        customdata=node_ids,
        name='nodes'
    ))

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Methods Network Graph",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=14)
            )
        ],
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
        st.write(f"**Pipeline Parts:** {pipeline_parts}")

        # Categories
        categories = node_data.get(processed_data['category_col'])
        if isinstance(categories, list):
            categories = ', '.join(categories)
        st.write(f"**Categories:** {categories}")

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
    main_col, right_panel = st.columns([3, 1])

    # Sidebar filters
    with left_sidebar:
        st.header("🔍 Filters")

        # Year filter
        if 'year' in nodes_df.columns:
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
            year_range = (2000, 2024)

        # Category filter
        st.subheader("Categories")
        selected_categories = st.multiselect(
            "Select categories:",
            options=processed_data['unique_categories'],
            key="category_filter"
        )

        # Methods filter
        st.subheader("Detected Methods")
        selected_methods = st.multiselect(
            "Select methods:",
            options=processed_data['unique_methods'],
            key="methods_filter"
        )

        # Edge toggle
        show_edges = st.checkbox("Show edges", value=True, key="edge_toggle")

        st.markdown("---")

    # Filter nodes
    filtered_nodes = filter_nodes(
        nodes_df, year_range, selected_categories, selected_methods,
        processed_data['category_col'], processed_data['methods_col']
    )

    # Display paper count
    with left_sidebar:
        st.metric("📊 Papers Displayed", len(filtered_nodes))

    # Main graph
    with main_col:
        if not filtered_nodes.empty:
            fig = create_graph(filtered_nodes, links_df, show_edges, processed_data)

            # Handle click events
            clicked_point = st.plotly_chart(
                fig,
                use_container_width=True,
                key="main_graph"
            )

        else:
            st.warning("No papers match the current filters.")

    # Right panel for node details
    with right_panel:
        st.header("📋 Details")

        # Store selected node in session state
        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = None

        # You would need to implement click handling here
        # This is a simplified version - in practice you'd need to capture
        # the clicked point from the plotly chart

        if st.session_state.selected_node is not None:
            node_data = filtered_nodes.loc[st.session_state.selected_node]
            display_node_details(node_data.to_dict(), processed_data)
        else:
            st.info("Click on a node to see details")

            # Show legend
            st.subheader("🎨 Legend")
            st.write("**Node Size (Citations):**")
            st.write("• Small: 0-4 citations")
            st.write("• Medium: 5-24 citations")
            st.write("• Large: 25-74 citations")
            st.write("• X-Large: 75-149 citations")
            st.write("• XX-Large: 150+ citations")


if __name__ == "__main__":
    main()