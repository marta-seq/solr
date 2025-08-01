# src/streamlit_app/pages/1_Non_Computational_Papers.py
import streamlit as st
import pandas as pd
from db_utils import get_db_engine, load_papers_data

st.set_page_config(layout="wide")

st.title("Non-Computational Papers")
st.markdown("Browse and search papers that are not classified as computational.")

engine = get_db_engine()

if engine:
    df_papers = load_papers_data(engine)

    if not df_papers.empty:
        # Filter for non-computational papers
        df_non_comp = df_papers[df_papers['is_computational'] == False].copy()

        if df_non_comp.empty:
            st.info("No non-computational papers found in the database.")
        else:
            st.subheader(f"Total Non-Computational Papers: {len(df_non_comp)}")

            # Search bar
            search_query = st.text_input("Search by Title or Abstract", "")
            if search_query:
                search_lower = search_query.lower()
                df_non_comp = df_non_comp[
                    df_non_comp['title'].str.lower().str.contains(search_lower, na=False) |
                    df_non_comp['abstract'].str.lower().str.contains(search_lower, na=False)
                ]
                st.info(f"Showing {len(df_non_comp)} results for '{search_query}'.")

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                if 'year' in df_non_comp.columns and not df_non_comp['year'].isnull().all():
                    all_years = sorted(df_non_comp['year'].dropna().astype(int).unique().tolist(), reverse=True)
                    selected_years = st.multiselect("Filter by Year", all_years, default=[])
                    if selected_years:
                        df_non_comp = df_non_comp[df_non_comp['year'].isin(selected_years)]
                else:
                    st.info("Year data not available for filtering.")

            with col2:
                if 'journal' in df_non_comp.columns and not df_non_comp['journal'].isnull().all():
                    all_journals = sorted(df_non_comp['journal'].dropna().unique().tolist())
                    selected_journals = st.multiselect("Filter by Journal", all_journals, default=[])
                    if selected_journals:
                        df_non_comp = df_non_comp[df_non_comp['journal'].isin(selected_journals)]
                else:
                    st.info("Journal data not available for filtering.")


            st.dataframe(df_non_comp, use_container_width=True)
    else:
        st.warning("No paper data available to display. Please ensure the database is populated.")