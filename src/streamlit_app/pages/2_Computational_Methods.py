# # src/streamlit_app/pages/2_Computational_Papers.py
# import streamlit as st
# import pandas as pd
# from db_utils import get_categorized_papers
#
# st.set_page_config(layout="wide")
#
# st.title("Computational Papers")
# st.markdown("Browse and search papers that are classified as computational.")
#
# # Get categorized papers
# df_computational_papers, _ = get_categorized_papers()
#
# if df_computational_papers.empty:
#     st.info("No computational papers found in the data.")
# else:
#     st.subheader(f"Total Computational Papers: {len(df_computational_papers)}")
#
#     # Search bar
#     search_query = st.text_input("Search by Title or Abstract", "")
#     if search_query:
#         search_lower = search_query.lower()
#         df_computational_papers = df_computational_papers[
#             df_computational_papers['title'].str.lower().str.contains(search_lower, na=False) |
#             df_computational_papers['abstract'].str.lower().str.contains(search_lower, na=False)
#         ]
#         st.info(f"Showing {len(df_computational_papers)} results for '{search_query}'.")
#
#     # Filters
#     col1, col2 = st.columns(2)
#     with col1:
#         if 'year' in df_computational_papers.columns and not df_computational_papers['year'].isnull().all():
#             all_years = sorted(df_computational_papers['year'].dropna().astype(int).unique().tolist(), reverse=True)
#             selected_years = st.multiselect("Filter by Year", all_years, default=[])
#             if selected_years:
#                 df_computational_papers = df_computational_papers[df_computational_papers['year'].isin(selected_years)]
#         else:
#             st.info("Year data not available for filtering.")
#
#     with col2:
#         if 'journal' in df_computational_papers.columns and not df_computational_papers['journal'].isnull().all():
#             all_journals = sorted(df_computational_papers['journal'].dropna().unique().tolist())
#             selected_journals = st.multiselect("Filter by Journal", all_journals, default=[])
#             if selected_journals:
#                 df_computational_papers = df_computational_papers[df_computational_papers['journal'].isin(selected_journals)]
#         else:
#             st.info("Journal data not available for filtering.")
#
#     st.dataframe(df_computational_papers, use_container_width=True)
import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming db_utils is in the same directory as your main streamlit_app.py
# If db_utils is in a 'src' folder at the root, you might need:
import db_utils # Adjust import based on your project structure

# --- Load Data (specific to this page) ---
@st.cache_data(ttl=3600)
def get_computational_data():
    # Only load computational data relevant for this page
    df_computational, _ = db_utils.get_categorized_papers()
    return df_computational

df_computational = get_computational_data()

# --- Page Content ---
st.header("Computational Methods Papers")

if df_computational.empty:
    st.warning("No computational papers found or loaded. Please check your data source.")
else:
    st.write(f"Displaying {len(df_computational)} computational papers.")

    # Filter for kw_pipeline_category presence
    # This checkbox controls the table display below
    filter_pipeline_category = st.checkbox("Show only papers with 'Pipeline Category' (kw_pipeline_category)", value=False)

    filtered_df_computational = df_computational.copy()

    if filter_pipeline_category:
        # Filter rows where 'kw_pipeline_category' is not null and not an empty string
        # Also handle cases where it might be a string of just whitespace
        filtered_df_computational = filtered_df_computational[
            filtered_df_computational['kw_pipeline_category'].notna() &
            (filtered_df_computational['kw_pipeline_category'].astype(str).str.strip() != '')
        ]
        st.info(f"Filtered to {len(filtered_df_computational)} papers with a 'Pipeline Category'.")

    # Display the filtered DataFrame
    # This is the table that gets filtered
    st.dataframe(filtered_df_computational[[
        'title', 'year', 'keywords', 'computational_methods', 'kw_pipeline_category',
        'kw_task_category', 'kw_data_type', 'kw_model_type', 'is_computational'
    ]].set_index('title'))

    st.subheader("Computational Methods Distribution")
    if 'computational_methods' in df_computational.columns:
        df_methods = db_utils.get_exploded_counts(df_computational, 'computational_methods')
        if not df_methods.empty:
            fig_methods = px.bar(df_methods.head(10), x='item', y='count',
                                 title='Top 10 Computational Methods', labels={'item': 'Method', 'count': 'Count'})
            st.plotly_chart(fig_methods, use_container_width=True)
        else:
            st.info("No computational methods data available.")
    else:
        st.info("Computational methods column not found.")

    st.subheader("Pipeline Category Distribution")
    if 'kw_pipeline_category' in df_computational.columns:
        df_pipeline = db_utils.get_exploded_counts(df_computational, 'kw_pipeline_category')
        if not df_pipeline.empty:
            fig_pipeline = px.bar(df_pipeline.head(10), x='item', y='count',
                                  title='Top 10 Pipeline Categories', labels={'item': 'Category', 'count': 'Count'})
            st.plotly_chart(fig_pipeline, use_container_width=True)
        else:
            st.info("No pipeline category data available.")
    else:
        st.info("Pipeline category column not found.")
