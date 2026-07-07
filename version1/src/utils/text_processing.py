# src/utils/text_processing.py
import re
from typing import List, Any

# Simple list of common English stop words for basic title cleaning
# Extend this list if needed for your domain
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "for", "nor", "on", "at", "to", "from",
    "in", "of", "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "not", "no", "yes", "can", "will",
    "would", "should", "could", "this", "that", "these", "those", "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def normalize_title(title: Any) -> str:
    """
    Normalizes a title string for comparison:
    - Converts to string, handles non-string inputs (e.g., lists, None)
    - Converts to lowercase
    - Removes punctuation
    - Removes extra whitespace
    - Removes common stop words (optional, but good for fuzzy matching)
    """
    if isinstance(title, list):
        title = title[0] if title else ""
    elif title is None:
        title = ""
    else:
        title = str(title)

    title = title.lower()
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    # Remove digits
    title = re.sub(r'\d+', '', title)
    # Remove extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    # Remove stop words
    words = title.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)

def normalize_authors(authors: List[str]) -> str:
    """
    Normalizes a list of author names for comparison.
    - Converts to lowercase.
    - Removes initials (e.g., "Doe, J" -> "doe").
    - Sorts names alphabetically.
    - Joins with a delimiter.
    """
    if not isinstance(authors, list):
        return ""

    normalized_names = []
    for author in authors:
        author_str = str(author).lower().strip()
        # Attempt to extract last name if format is "Lastname, Initials"
        match = re.match(r'([a-z\s-]+),?\s*([a-z\s\.]*)', author_str)
        if match:
            last_name = match.group(1).replace(',', '').strip()
            normalized_names.append(last_name)
        else:
            # Fallback for other formats, just use the whole name after cleaning
            normalized_names.append(re.sub(r'[^\w\s]', '', author_str).strip())

    normalized_names = [name for name in normalized_names if name] # Remove empty strings
    normalized_names.sort() # Sort for consistent order
    return "_".join(normalized_names) # Join with underscore for a single string

# Example usage (for testing this module directly)
if __name__ == "__main__":
    test_titles = [
        "A Novel Method for Spatial Transcriptomics Analysis",
        ["A Novel Method for Spatial Transcriptomics Analysis"], # List format
        "A Novel Method for Spatial Transcriptomics Analysis. (Preprint)",
        "Novel Method: Spatial Transcriptomics Analysis",
        None,
        "",
        "The application of a new tool in spatial omics data"
    ]
    print("--- Normalized Titles ---")
    for t in test_titles:
        print(f"Original: '{t}' -> Normalized: '{normalize_title(t)}'")

    test_authors = [
        ["Sequeira, A", "Doe, J"],
        ["Doe, J", "Sequeira, A.S."],
        ["Smith, P", "Jones, R.T.", "Collective Group"]
    ]
    print("\n--- Normalized Authors ---")
    for a in test_authors:
        print(f"Original: {a} -> Normalized: '{normalize_authors(a)}'")