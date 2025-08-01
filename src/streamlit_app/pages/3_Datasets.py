# src/streamlit_app/pages/3_Datasets.py
import streamlit as st
import pandas as pd
from db_utils import get_db_engine, load_internal_datasets_data

st.set_page_config(layout="wide")

st.title("Internal Datasets")
st.markdown("Browse and search your internal datasets.")

engine = get_db_engine()

if engine:
    df_datasets = load_internal_datasets_data(engine)

    if not df_datasets.empty:
        st.subheader(f"Total Internal Datasets: {len(df_datasets)}")

        # Search bar
        search_query = st.text_input("Search by Dataset Name or Description", "")
        if search_query:
            search_lower = search_query.lower()
            df_datasets = df_datasets[
                df_datasets['dataset_name'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['dataset_description'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['omics'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['omics_detail'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['method'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['organism'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['tissue'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['disease'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['cohorts'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['dataset_type'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['markers'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['clinical_data'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['notes'].str.lower().str.contains(search_lower, na=False) |
                df_datasets['notes2'].str.lower().str.contains(search_lower, na=False)
            ]
            st.info(f"Showing {len(df_datasets)} results for '{search_query}'.")

        # Filters (example: by year if available in internal_datasets)
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
        st.warning("No internal datasets data available to display. Please ensure the database is populated.")