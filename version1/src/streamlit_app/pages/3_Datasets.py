# src/streamlit_app/pages/3_Datasets.py
import streamlit as st
import pandas as pd
from db_utils import load_internal_datasets_data

st.set_page_config(layout="wide")

st.title("Internal Datasets")
st.markdown("Browse and search your internal datasets.")

df_datasets = load_internal_datasets_data()

if not df_datasets.empty:
    st.subheader(f"Total Internal Datasets: {len(df_datasets)}")

    # Search bar
    search_query = st.text_input("Search by any column", "")
    if search_query:
        search_lower = search_query.lower()
        # Search across all string columns
        df_datasets = df_datasets[
            df_datasets.apply(lambda row: row.astype(str).str.lower().str.contains(search_lower, na=False).any(), axis=1)
        ]
        st.info(f"Showing {len(df_datasets)} results for '{search_query}'.")

    # Filters (example: by year, organism, dataset_type if available)
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'year' in df_datasets.columns and not df_datasets['year'].isnull().all():
            all_years = sorted(df_datasets['year'].dropna().astype(int).unique().tolist(), reverse=True)
            selected_years = st.multiselect("Filter by Year", all_years, default=[])
            if selected_years:
                df_datasets = df_datasets[df_datasets['year'].isin(selected_years)]
        else:
            st.info("Year data not available for filtering.")
    with col2:
        if 'organism' in df_datasets.columns and not df_datasets['organism'].isnull().all():
            all_organisms = sorted(df_datasets['organism'].dropna().unique().tolist())
            selected_organisms = st.multiselect("Filter by Organism", all_organisms, default=[])
            if selected_organisms:
                df_datasets = df_datasets[df_datasets['organism'].isin(selected_organisms)]
        else:
            st.info("Organism data not available for filtering.")
    with col3:
        if 'dataset_type' in df_datasets.columns and not df_datasets['dataset_type'].isnull().all():
            all_types = sorted(df_datasets['dataset_type'].dropna().unique().tolist())
            selected_types = st.multiselect("Filter by Dataset Type", all_types, default=[])
            if selected_types:
                df_datasets = df_datasets[df_datasets['dataset_type'].isin(selected_types)]
        else:
            st.info("Dataset Type data not available for filtering.")

    st.dataframe(df_datasets, use_container_width=True)
else:
    st.warning("No internal datasets data available to display. Please ensure 'internal_datasets.xlsx' is present and accessible.")