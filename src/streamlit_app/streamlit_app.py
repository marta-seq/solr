# src/streamlit_app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from db_utils import get_db_engine, load_papers_data, load_computational_methods_data, load_internal_datasets_data

# Set Streamlit page configuration
st.set_page_config(
    page_title="PhD Review Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Welcome to the PhD Review Dashboard!")
st.markdown("Explore and analyze papers and datasets related to spatial omics and computational methods.")

# Get database engine
engine = get_db_engine()

if engine:
    # Load all papers data
    df_papers = load_papers_data(engine)
    df_methods = load_computational_methods_data(engine)
    df_datasets = load_internal_datasets_data(engine)

    if not df_papers.empty:
        st.header("Overall Statistics")

        col1, col2, col3 = st.columns(3)

        total_papers = len(df_papers)
        # Use .sum() on boolean column directly for True counts
        computational_papers = df_papers['is_computational'].sum() if 'is_computational' in df_papers.columns else 0
        non_computational_papers = total_papers - computational_papers

        with col1:
            st.metric("Total Papers", total_papers)
        with col2:
            st.metric("Computational Papers", computational_papers)
        with col3:
            st.metric("Non-Computational Papers", non_computational_papers)

        st.markdown("---")
        st.header("Paper Distribution by Year")

        if 'year' in df_papers.columns and not df_papers['year'].isnull().all():
            # Filter out future years or non-numeric years
            current_year = pd.Timestamp.now().year
            df_years = df_papers[df_papers['year'].notna() & (df_papers['year'] <= current_year)].copy()
            df_years['year'] = df_years['year'].astype(int) # Ensure year is int for grouping

            papers_by_year = df_years['year'].value_counts().sort_index().reset_index()
            papers_by_year.columns = ['Year', 'Number of Papers']

            fig_year = px.bar(papers_by_year, x='Year', y='Number of Papers',
                              title='Number of Papers by Publication Year',
                              labels={'Number of Papers': 'Count'},
                              color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_year, use_container_width=True)
        else:
            st.info("Year data not available or not suitable for plotting.")

        st.markdown("---")
        st.header("Latest Data Updates")

        col_date1, col_date2 = st.columns(2)
        if 'scrape_date' in df_papers.columns and not df_papers['scrape_date'].isnull().all():
            latest_scrape = df_papers['scrape_date'].max().strftime('%Y-%m-%d')
            col_date1.metric("Latest Scrape Date", latest_scrape)
        else:
            col_date1.info("Scrape date data not available.")

        if 'llm_annotation_timestamp' in df_papers.columns and not df_papers['llm_annotation_timestamp'].isnull().all():
            latest_llm_annotation = df_papers['llm_annotation_timestamp'].max().strftime('%Y-%m-%d %H:%M')
            col_date2.metric("Latest LLM Annotation", latest_llm_annotation)
        else:
            col_date2.info("LLM annotation timestamp not available.")

        st.markdown("---")
        st.header("Distribution by Organism and Disease")

        col_org, col_disease = st.columns(2)

        if 'kw_organism' in df_papers.columns and not df_papers['kw_organism'].isnull().all():
            # Handle comma-separated lists for organisms
            df_organisms_exploded = df_papers['kw_organism'].dropna().str.split(', ').explode()
            organism_counts = df_organisms_exploded.value_counts().reset_index()
            organism_counts.columns = ['Organism', 'Count']
            fig_organism = px.bar(organism_counts.head(10), x='Count', y='Organism', orientation='h',
                                  title='Top 10 Organisms',
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_organism.update_layout(yaxis={'categoryorder':'total ascending'})
            with col_org:
                st.plotly_chart(fig_organism, use_container_width=True)
        else:
            with col_org:
                st.info("Organism data not available for plotting.")

        if 'kw_disease' in df_papers.columns and not df_papers['kw_disease'].isnull().all():
            # Handle comma-separated lists for diseases
            df_diseases_exploded = df_papers['kw_disease'].dropna().str.split(', ').explode()
            disease_counts = df_diseases_exploded.value_counts().reset_index()
            disease_counts.columns = ['Disease', 'Count']
            fig_disease = px.bar(disease_counts.head(10), x='Count', y='Disease', orientation='h',
                                 title='Top 10 Diseases',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_disease.update_layout(yaxis={'categoryorder':'total ascending'})
            with col_disease:
                st.plotly_chart(fig_disease, use_container_width=True)
        else:
            with col_disease:
                st.info("Disease data not available for plotting.")

    else:
        st.warning("No paper data loaded. Please ensure the database is populated.")