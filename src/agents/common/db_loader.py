"""
db_loader.py
Loads the most recent dated CSVs written by src/preprocessing/ (methods_metadata_*.csv,
datasets_*.csv), the same "find most recent dated file" convention as
02_fetch_metadata.py, and builds the shared lookup structures every
department needs: a DoiIndex (dedup) and an IdAllocator (new IDs).
"""

from pathlib import Path

import pandas as pd

from . import config
from .doi_utils import DoiIndex
from .id_allocator import IdAllocator


def _latest(pattern: str) -> Path:
    matches = sorted(config.PROCESSED_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern} in {config.PROCESSED_DIR}")
    return matches[-1]


def load_methods() -> pd.DataFrame:
    """Prefers methods_metadata_*.csv (has abstract/title/etc.); falls back to
    methods_*.csv (excluding the metadata variant) if metadata hasn't been run."""
    try:
        f = _latest("methods_metadata_*.csv")
    except FileNotFoundError:
        candidates = [p for p in config.PROCESSED_DIR.glob("methods_*.csv")
                      if "metadata" not in p.name]
        if not candidates:
            raise
        f = sorted(candidates)[-1]
    return pd.read_csv(f, dtype=str)


def load_datasets() -> pd.DataFrame:
    return pd.read_csv(_latest("datasets_*.csv"), dtype=str)


def _is_true(val) -> bool:
    return str(val).strip().lower() == "true"


class Database:
    """One place holding everything an agent needs to check 'does this already
    exist' and 'what's the next free ID' - built fresh at the start of a run."""

    def __init__(self):
        self.methods = load_methods()
        self.datasets = load_datasets()
        self.doi_index = DoiIndex()
        self.id_allocator = IdAllocator()
        self._build()

    def _build(self):
        # DOIs: methods sheet + both DOI columns in datasets sheet
        self.doi_index.load_from_dataframe(self.methods, "DOI", "entry_id")
        self.doi_index.load_from_dataframe(self.datasets, "data_DOI", "entry_id")
        self.doi_index.load_from_dataframe(self.datasets, "paper_DOI", "entry_id")

        # ID allocator needs every entry_id seen anywhere, real or placeholder
        all_ids = list(self.methods["entry_id"].dropna()) + list(self.datasets["entry_id"].dropna())
        self.id_allocator.register_existing(all_ids)

    def real_methods(self) -> pd.DataFrame:
        """Methods rows excluding placeholders (is_placeholder == True)."""
        return self.methods[~self.methods["is_placeholder"].apply(_is_true)]
