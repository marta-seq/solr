import json
import plotly.graph_objects as go
import numpy as np
import os
import logging
from datetime import datetime

# --- Configuration ---
INPUT_GRAPH_DATA_FILE = "../../data/methods/graph_data.json"
OUTPUT_HTML_FILE = "../../visualizations/visualized_graph.html"

# Node sizing parameters
BASE_NODE_SIZE = 8
CITATIONS_SCALE_FACTOR = 100
CITATIONS_MULTIPLIER = 5

# --- Logging Setup ---
LOG_DIR = "../../data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = datetime.now().strftime(f"{LOG_DIR}/static_visualization_%Y%m%d_%H%M%S.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.handlers:
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# --- Helper to format hover text ---
def get_hover_text(node):
    """
    Generates concise hover text for a node.
    Adjust this to show only the most important info in the static hover.
    """
    title = node.get('title', 'N/A')
    doi = node.get('doi', 'N/A')
    year = node.get('year', 'N/A')
    citations = node.get('citations', 'N/A')
    tool_type = node.get('llm_annot_tool_type', 'N/A')

    # Process raw lists for display
    pipeline_steps = node.get('raw_pipeline_steps', [])
    modalities = node.get('raw_tested_data_modalities', [])

    hover_text = f"<b>Title:</b> {title}<br>"
    hover_text += f"<b>DOI:</b> {doi}<br>"
    hover_text += f"<b>Year:</b> {year}<br>"
    hover_text += f"<b>Citations:</b> {citations}<br>"
    hover_text += f"<b>Tool Type:</b> {tool_type}<br>"
    if pipeline_steps:
        hover_text += f"<b>Pipeline Steps:</b> {', '.join(pipeline_steps)}<br>"
    if modalities:
        hover_text += f"<b>Data Modalities:</b> {', '.join(modalities)}<br>"

    return hover_text


# --- Main Visualization Logic ---
def visualize_graph(last_update_date: str = None): # Added last_update_date as an argument
    logger.info("--- Starting Static Graph Visualization Script ---")

    # 1. Load graph data
    try:
        with open(INPUT_GRAPH_DATA_FILE, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        nodes_data = graph_data['nodes']
        edges_data = graph_data['edges']
        logger.info(f"Loaded {len(nodes_data)} nodes and {len(edges_data)} edges from '{INPUT_GRAPH_DATA_FILE}'.")
    except FileNotFoundError:
        logger.critical(
            f"Graph data file not found: '{INPUT_GRAPH_DATA_FILE}'. Please run generate_graph_data.py first.")
        return
    except json.JSONDecodeError as e:
        logger.critical(f"Error decoding JSON from '{INPUT_GRAPH_DATA_FILE}': {e}. File might be corrupted.")
        return
    except Exception as e:
        logger.critical(f"An unexpected error occurred while loading graph data: {e}")
        return

    if not nodes_data:
        logger.warning("No nodes found in graph data. Cannot visualize an empty graph.")
        return

    # Prepare node coordinates and attributes
    x_coords = [node['x'] for node in nodes_data]
    y_coords = [node['y'] for node in nodes_data]

    # --- Node Sizing based on Citations ---
    node_sizes = []
    for node in nodes_data:
        size = BASE_NODE_SIZE
        citations_val = node.get('citations')

        # Safely convert citations to a number
        citations_for_log = 0
        if citations_val is not None:
            try:
                citations_for_log = float(citations_val)
            except (ValueError, TypeError):
                # Handle cases where citations might be non-numeric or empty string
                logger.warning(
                    f"Invalid citation value '{citations_val}' for DOI: {node.get('doi', 'N/A')}. Defaulting to 0.")
                citations_for_log = 0

        # Apply log transformation for smoother scaling
        # Adding 1 to avoid log(0) and ensure even low citation counts contribute to size
        scaled_citations = np.log1p(citations_for_log / CITATIONS_SCALE_FACTOR) * CITATIONS_MULTIPLIER
        size += scaled_citations
        node_sizes.append(size)
    logger.info("Node sizes calculated based on citations.")

    # --- Node Coloring based on Tool Type ---
    node_colors = []
    color_map = {}

    # Define a distinct color palette (you can customize these hex codes)
    colors_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    unique_tool_types = sorted(list(set(node.get('llm_annot_tool_type', 'Unknown') for node in nodes_data)))

    for i, tt in enumerate(unique_tool_types):
        color_map[tt] = colors_palette[i % len(colors_palette)]  # Cycle through colors if more types than colors

    for node in nodes_data:
        tool_type = node.get('llm_annot_tool_type', 'Unknown')
        color = color_map.get(tool_type)
        if color is None:
            logger.warning(f"Tool type '{tool_type}' not found in color map. Assigning default gray.")
            color = 'lightgray'
        node_colors.append(color)
    logger.info("Node colors determined by 'llm_annot_tool_type'.")

    # Create node trace
    node_trace = go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        hoverinfo='text',
        text=[get_hover_text(node) for node in nodes_data],
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line_width=1,
            line_color='black',
            sizemode='diameter'
        ),
        showlegend=False
    )

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in edges_data:
        source_node = next((n for n in nodes_data if n['id'] == edge['source']), None)
        target_node = next((n for n in nodes_data if n['id'] == edge['target']), None)

        if source_node and target_node:
            edge_x.extend([source_node['x'], target_node['x'], None])
            edge_y.extend([source_node['y'], target_node['y'], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Prepare annotations list
    annotations_list = [
        dict(
            text="UMAP projection of papers based on embedding similarity",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002
        )
    ]

    # Add the "Last update:" annotation if date is provided
    if last_update_date:
        annotations_list.append(
            dict(
                text=f"Last update: {last_update_date}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=0.04,  # Adjust y to be slightly above the UMAP annotation
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color="gray") # Smaller, muted font
            )
        )
        logger.info(f"Added 'Last update: {last_update_date}' annotation to the graph.")


    # Create Plotly Figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='<br>UMAP Visualization of Methodologies',
                            font=dict(size=16)
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=annotations_list, # Use the list of annotations
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Add custom legend entries for node colors
    for tool_type, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            name=f"Tool Type: {tool_type}",
            legendgroup=tool_type,
            showlegend=True
        ))
    logger.info("Custom legend entries added for tool types.")

    # Save to HTML
    try:
        fig.write_html(OUTPUT_HTML_FILE, auto_open=True)
        logger.info(f"Static graph visualization saved to '{OUTPUT_HTML_FILE}' and opened in browser.")
    except Exception as e:
        logger.critical(f"Failed to save or open HTML file: {e}")

    logger.info("--- Static Graph Visualization Script Finished ---")


if __name__ == "__main__":
    # Example of how to call it with a date
    current_date = datetime.now().strftime("%Y-%m-%d")
    visualize_graph(last_update_date=current_date)
    # Or, you can pass a specific date string:
    # visualize_graph(last_update_date="2024-03-15")