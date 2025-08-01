# src/streamlit_app/pages/2_Computational_Methods.py
import streamlit as st
import pandas as pd
from db_utils import get_db_engine, load_computational_methods_data, load_papers_data

st.set_page_config(layout="wide")

st.title("Computational Methods")
st.markdown("Explore the unique computational methods identified in papers.")

engine = get_db_engine()

if engine:
    df_methods = load_computational_methods_data(engine)
    df_papers = load_papers_data(engine) # Load papers to link methods to papers

    if not df_methods.empty:
        st.subheader(f"Total Unique Methods: {len(df_methods)}")

        # Search bar
        search_query = st.text_input("Search by Method Name", "")
        if search_query:
            search_lower = search_query.lower()
            df_methods = df_methods[
                df_methods['name'].str.lower().str.contains(search_lower, na=False)
            ]
            st.info(f"Showing {len(df_methods)} results for '{search_query}'.")

        # Optional: Display papers linked to a selected method
        if not df_methods.empty:
            selected_method_name = st.selectbox("Select a method to see related papers:",
                                                [''] + sorted(df_methods['name'].tolist()))
            if selected_method_name:
                # Find the method_id for the selected method
                selected_method_id = df_methods[df_methods['name'] == selected_method_name]['method_id'].iloc[0]

                # Query paper_method_links to find DOIs related to this method
                query = text(f"""
                    SELECT p.doi, p.title, p.year, p.journal
                    FROM papers p
                    JOIN paper_method_links pml ON p.doi = pml.paper_doi
                    WHERE pml.method_id = :method_id;
                """)
                df_related_papers = pd.read_sql_query(query, engine, params={'method_id': selected_method_id})

                st.subheader(f"Papers using '{selected_method_name}':")
                if not df_related_papers.empty:
                    st.dataframe(df_related_papers, use_container_width=True)
                else:
                    st.info(f"No papers found using '{selected_method_name}'.")

        st.dataframe(df_methods, use_container_width=True)
    else:
        st.warning("No computational methods data available to display. Please ensure the database is populated.")