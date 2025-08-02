# src/streamlit_app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from db_utils import get_categorized_papers, load_raw_papers_data, get_exploded_counts

# Set Streamlit page configuration
st.set_page_config(
    page_title="Computational Spatial Data Analysis Platform", # Overall app title
    layout="wide",
    initial_sidebar_state="expanded",
    # icon="📊" # Optional: You might want a spatial data related emoji here
)
st.title("Welcome to the Computational Spatial Data Analysis Explorer!")
st.markdown("""
This platform helps you explore and visualize methods related to computational spatial data analysis.
Use the sidebar to navigate through different sections, including an interactive graph of methods.
""")
# Load and categorize papers once
df_papers_raw = load_raw_papers_data()
df_computational_papers, df_non_computational_papers = get_categorized_papers()

if df_papers_raw.empty:
    st.error("No paper data loaded. Please ensure 'processed_final.csv' is present and accessible.")
    st.stop() # Stop execution if no data

st.header("Overall Statistics")

col1, col2, col3 = st.columns(3)

total_papers = len(df_papers_raw)
computational_count = len(df_computational_papers)
non_computational_count = len(df_non_computational_papers)

with col1:
    st.metric("Total Papers", total_papers)
with col2:
    st.metric("Computational Papers", computational_count)
with col3:
    st.metric("Non-Computational Papers", non_computational_count)

st.markdown("---")

# --- Date Information ---
st.header("Latest Data Updates")
col_date1, col_date2, col_date3 = st.columns(3)

if 'scrape_date' in df_papers_raw.columns and not df_papers_raw['scrape_date'].isnull().all():
    latest_scrape = df_papers_raw['scrape_date'].max().strftime('%Y-%m-%d')
    col_date1.metric("Latest Scrape Date", latest_scrape)
else:
    col_date1.info("Scrape date data not available.")

if 'llm_annotation_timestamp' in df_papers_raw.columns and not df_papers_raw['llm_annotation_timestamp'].isnull().all():
    latest_llm_annotation = df_papers_raw['llm_annotation_timestamp'].max().strftime('%Y-%m-%d %H:%M')
    col_date2.metric("Latest LLM Annotation", latest_llm_annotation)
else:
    col_date2.info("LLM annotation timestamp not available.")

# Assuming 'full_text_extraction_timestamp' is a general update date
if 'full_text_extraction_timestamp' in df_papers_raw.columns and not df_papers_raw['full_text_extraction_timestamp'].isnull().all():
    latest_update = df_papers_raw['full_text_extraction_timestamp'].max().strftime('%Y-%m-%d %H:%M')
    col_date3.metric("Last Data Update", latest_update)
else:
    col_date3.info("Full text extraction timestamp not available.")

st.markdown("---")

# --- Pie Chart: Computational vs. Non-Computational ---
st.header("Paper Type Distribution")
paper_type_data = pd.DataFrame({
    'Category': ['Computational', 'Non-Computational'],
    'Count': [computational_count, non_computational_count]
})
fig_pie = px.pie(paper_type_data, values='Count', names='Category',
                 title='Distribution of Computational vs. Non-Computational Papers',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# --- Line Chart: Papers by Year (Overall, Computational, Non-Computational) ---
st.header("Papers by Publication Year")
if 'year' in df_papers_raw.columns and not df_papers_raw['year'].isnull().all():
    current_year = pd.Timestamp.now().year
    df_years_all = df_papers_raw[df_papers_raw['year'].notna() & (df_papers_raw['year'] <= current_year)].copy()
    df_years_comp = df_computational_papers[df_computational_papers['year'].notna() & (df_computational_papers['year'] <= current_year)].copy()
    df_years_non_comp = df_non_computational_papers[df_non_computational_papers['year'].notna() & (df_non_computational_papers['year'] <= current_year)].copy()

    if not df_years_all.empty:
        papers_by_year_all = df_years_all['year'].value_counts().sort_index().reset_index()
        papers_by_year_all.columns = ['Year', 'Total Papers']

        papers_by_year_comp = df_years_comp['year'].value_counts().sort_index().reset_index()
        papers_by_year_comp.columns = ['Year', 'Computational Papers']

        papers_by_year_non_comp = df_years_non_comp['year'].value_counts().sort_index().reset_index()
        papers_by_year_non_comp.columns = ['Year', 'Non-Computational Papers']

        # Merge all dataframes for plotting
        merged_years = papers_by_year_all.merge(papers_by_year_comp, on='Year', how='outer')
        merged_years = merged_years.merge(papers_by_year_non_comp, on='Year', how='outer').fillna(0)
        merged_years = merged_years.sort_values('Year')

        fig_line = px.line(merged_years, x='Year', y=['Total Papers', 'Computational Papers', 'Non-Computational Papers'],
                           title='Number of Papers by Publication Year',
                           labels={'value': 'Number of Papers', 'variable': 'Paper Type'},
                           color_discrete_map={
                               'Total Papers': 'blue',
                               'Computational Papers': 'red',
                               'Non-Computational Papers': 'green'
                           })
        fig_line.update_traces(mode='lines+markers')
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No valid year data found for plotting.")
else:
    st.info("Year data not available or not suitable for plotting.")

st.markdown("---")

# --- Graphics on kw_tissue and kw_disease ---
st.header("Distribution by Tissue and Disease")
col_tissue, col_disease = st.columns(2)

with col_tissue:
    tissue_counts = get_exploded_counts(df_papers_raw, 'kw_tissue')
    if not tissue_counts.empty:
        fig_tissue = px.bar(tissue_counts.head(10), x='Count', y='Tissue', orientation='h',
                            title='Top 10 Tissues',
                            color_discrete_sequence=px.colors.qualitative.Set3)
        fig_tissue.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_tissue, use_container_width=True)
    else:
        st.info("Tissue data not available for plotting.")

with col_disease:
    disease_counts = get_exploded_counts(df_papers_raw, 'kw_disease')
    if not disease_counts.empty:
        fig_disease = px.bar(disease_counts.head(10), x='Count', y='Disease', orientation='h',
                             title='Top 10 Diseases',
                             color_discrete_sequence=px.colors.qualitative.Set3)
        fig_disease.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_disease, use_container_width=True)
    else:
        st.info("Disease data not available for plotting.")

st.markdown("---")

# --- Graphics on Computational Methods Specifics (from computational papers) ---
st.header("Computational Papers: Method Specifics")

col_modalities, col_paper_type = st.columns(2)
with col_modalities:
    modalities_counts = get_exploded_counts(df_computational_papers, 'llm_annot_tested_data_modalities')
    if not modalities_counts.empty:
        fig_modalities = px.bar(modalities_counts.head(10), x='Count', y='Tested Data Modalities', orientation='h',
                                title='Top 10 Tested Data Modalities',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_modalities.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_modalities, use_container_width=True)
    else:
        st.info("Tested Data Modalities data not available for plotting in computational papers.")

with col_paper_type:
    paper_type_counts = get_exploded_counts(df_computational_papers, 'kw_paper_type')
    if not paper_type_counts.empty:
        fig_paper_type = px.bar(paper_type_counts.head(10), x='Count', y='Paper Type', orientation='h',
                                title='Top 10 Paper Types',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_paper_type.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_paper_type, use_container_width=True)
    else:
        st.info("Paper Type data not available for plotting in computational papers.")

col_pipeline_category, col_detected_methods = st.columns(2)
with col_pipeline_category:
    pipeline_category_counts = get_exploded_counts(df_computational_papers, 'kw_pipeline_category')
    if not pipeline_category_counts.empty:
        fig_pipeline_category = px.bar(pipeline_category_counts.head(10), x='Count', y='Pipeline Category', orientation='h',
                                       title='Top 10 Pipeline Categories',
                                       color_discrete_sequence=px.colors.qualitative.T10)
        fig_pipeline_category.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_pipeline_category, use_container_width=True)
    else:
        st.info("Pipeline Category data not available for plotting in computational papers.")

with col_detected_methods:
    detected_methods_counts = get_exploded_counts(df_computational_papers, 'kw_detected_methods')
    if not detected_methods_counts.empty:
        fig_detected_methods = px.bar(detected_methods_counts.head(10), x='Count', y='Detected Methods', orientation='h',
                                      title='Top 10 Detected Methods',
                                      color_discrete_sequence=px.colors.qualitative.T10)
        fig_detected_methods.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_detected_methods, use_container_width=True)
    else:
        st.info("Detected Methods data not available for plotting in computational papers.")