# src/streamlit_app/pages/1_Non_Computational_Papers.py
import streamlit as st
import pandas as pd
from db_utils import get_categorized_papers

st.set_page_config(layout="wide")

st.title("Non-Computational Papers")
st.markdown("Browse and search papers that are not classified as computational.")

# Get categorized papers
_, df_non_computational_papers = get_categorized_papers()

if df_non_computational_papers.empty:
    st.info("No non-computational papers found in the data.")
else:
    st.subheader(f"Total Non-Computational Papers: {len(df_non_computational_papers)}")

    # Search bar
    search_query = st.text_input("Search by Title or Abstract", "")
    if search_query:
        search_lower = search_query.lower()
        df_non_computational_papers = df_non_computational_papers[
            df_non_computational_papers['title'].str.lower().str.contains(search_lower, na=False) |
            df_non_computational_papers['abstract'].str.lower().str.contains(search_lower, na=False)
        ]
        st.info(f"Showing {len(df_non_computational_papers)} results for '{search_query}'.")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if 'year' in df_non_computational_papers.columns and not df_non_computational_papers['year'].isnull().all():
            all_years = sorted(df_non_computational_papers['year'].dropna().astype(int).unique().tolist(), reverse=True)
            selected_years = st.multiselect("Filter by Year", all_years, default=[])
            if selected_years:
                df_non_computational_papers = df_non_computational_papers[df_non_computational_papers['year'].isin(selected_years)]
        else:
            st.info("Year data not available for filtering.")

    with col2:
        if 'journal' in df_non_computational_papers.columns and not df_non_computational_papers['journal'].isnull().all():
            all_journals = sorted(df_non_computational_papers['journal'].dropna().unique().tolist())
            selected_journals = st.multiselect("Filter by Journal", all_journals, default=[])
            if selected_journals:
                df_non_computational_papers = df_non_computational_papers[df_non_computational_papers['journal'].isin(selected_journals)]
        else:
            st.info("Journal data not available for filtering.")

    st.dataframe(df_non_computational_papers, use_container_width=True)