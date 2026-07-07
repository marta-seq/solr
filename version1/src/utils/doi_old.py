import pandas as pd

def clean_doi(doi_str):
    """Normalizes a DOI string by removing common prefixes."""
    if pd.isna(doi_str):
        return None
    doi_str = str(doi_str).strip()
    if doi_str.startswith("https://doi.org/"):
        return doi_str[len("https://doi.org/"):]
    return doi_str