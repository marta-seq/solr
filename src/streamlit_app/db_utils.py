# src/streamlit_app/db_utils.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables from .env file (ensure it's in the project root)
# When deployed on Streamlit Cloud, these will come from Streamlit Secrets.
load_dotenv()

@st.cache_resource # Cache the database connection engine to avoid reconnecting on every rerun
def get_db_engine():
    """Establishes and returns a SQLAlchemy engine for PostgreSQL."""
    db_name = os.getenv('POSTGRES_DB')
    db_user = os.getenv('POSTGRES_USER')
    db_password = os.getenv('POSTGRES_PASSWORD')
    db_host = os.getenv('POSTGRES_HOST')
    db_port = os.getenv('POSTGRES_PORT')

    if not all([db_name, db_user, db_password, db_host, db_port]):
        st.error("Database connection failed: One or more environment variables (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_PORT) are not set.")
        st.stop() # Stop the app if crucial env vars are missing
        return None

    try:
        # Construct the connection string
        connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)

        # Test connection by executing a simple query
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        st.success("Successfully connected to the database!")
        return engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.info("Please ensure your database is running and accessible (check credentials, host, port, and firewall rules if deploying to cloud).")
        st.stop() # Stop the app if connection fails
        return None

@st.cache_data(ttl=3600) # Cache data for 1 hour (adjust as needed)
def load_papers_data(engine):
    """Loads all papers data from the 'papers' table."""
    try:
        df = pd.read_sql_table('papers', engine)
        # Ensure 'year' is numeric and handle potential NaNs for plotting
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
        # Ensure dates are datetime objects for plotting
        for col in ['scrape_date', 'full_text_extraction_timestamp', 'llm_annotation_timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading papers data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_computational_methods_data(engine):
    """Loads all computational methods data from the 'computational_methods' table."""
    try:
        df = pd.read_sql_table('computational_methods', engine)
        return df
    except Exception as e:
        st.error(f"Error loading computational methods data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_internal_datasets_data(engine):
    """Loads all internal datasets data from the 'internal_datasets' table."""
    try:
        df = pd.read_sql_table('internal_datasets', engine)
        # Ensure 'year', 'n_patients', 'n_samples', 'n_regions' are numeric
        for col in ['year', 'n_patients', 'n_samples', 'n_regions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        return df
    except Exception as e:
        st.error(f"Error loading internal datasets data: {e}")
        return pd.DataFrame()