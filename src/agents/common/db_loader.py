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
from . import staging
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
        # DOIs: methods sheet + both DOI columns in datasets sheet.
        # IMPORTANT: a dataset row's paper_DOI is the DOI of the *paper* that
        # produced it - it must be indexed under that paper's paper_ENTRY_ID,
        # NOT the dataset's own entry_id, or a lookup for the paper's DOI
        # silently returns the dataset's ID instead (caught via testing -
        # this DOI legitimately appears in both methods.DOI for M_PR_3 and
        # datasets.paper_DOI for D_SP_IMC_5, which is the dataset it produced).
        self.doi_index.load_from_dataframe(self.methods, "DOI", "entry_id")
        self.doi_index.load_from_dataframe(self.datasets, "data_DOI", "entry_id")
        self.doi_index.load_from_dataframe(self.datasets, "paper_DOI", "paper_ENTRY_ID")

        # ID allocator needs every entry_id seen anywhere, real or placeholder
        all_ids = list(self.methods["entry_id"].dropna()) + list(self.datasets["entry_id"].dropna())
        self.id_allocator.register_existing(all_ids)

        self._ingest_staged_candidates()

    def _ingest_staged_candidates(self):
        """If you're re-running the pipeline after a previous run stopped
        partway through (e.g. MAX_PAPERS_PER_RUN cut it short, or you're
        resuming days later after a rate-limit pause), earlier runs may have
        already staged new entries that aren't in methods_metadata.csv/
        datasets.csv yet (nothing gets merged into the master files
        automatically). Without this, a later run wouldn't know about them
        and could create a SECOND, differently-numbered entry for the same
        DOI. Caught before it caused real duplicates - ingest everything
        already sitting in staging.xlsx into the live DoiIndex/IdAllocator
        before processing anything. Once you run merge_candidates.py and it
        archives staging.xlsx, this naturally has nothing left to ingest."""
        staged = staging.load_all_candidates_for_run()
        ingested = 0
        for rec in staged:
            if rec.get("action") != "create_entry":
                continue
            entry_id = rec.get("entry_id")
            if not entry_id:
                continue
            self.id_allocator.register_existing([entry_id])

            if rec.get("target_sheet") == "papers":
                doi = rec.get("DOI", "")
                if doi:
                    self.doi_index.add(doi, entry_id)
            else:
                data_doi = rec.get("data_DOI", "")
                if data_doi:
                    self.doi_index.add(data_doi, entry_id)
                paper_doi = rec.get("paper_DOI", "")
                paper_entry_id = rec.get("paper_ENTRY_ID", "")
                if paper_doi and paper_entry_id:
                    self.doi_index.add(paper_doi, paper_entry_id)
            ingested += 1

        if ingested:
            print(f"[db_loader] Ingested {ingested} already-staged candidates "
                  f"from previous unmerged run(s) (avoids duplicate creation on re-run).")

    def real_methods(self) -> pd.DataFrame:
        """Methods rows excluding placeholders (is_placeholder == True)."""
        return self.methods[~self.methods["is_placeholder"].apply(_is_true)]
