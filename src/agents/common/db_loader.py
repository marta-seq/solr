"""
db_loader.py
Loads the most recent dated CSVs written by src/preprocessing/ (methods_metadata_*.csv,
datasets_*.csv), the same "find most recent dated file" convention as
02_fetch_metadata.py, and builds the shared lookup structures every
department needs: a DoiIndex (dedup), a NameIndex (name-based dedup, added
2026-09-27), and an IdAllocator (new IDs).
"""

import re
from pathlib import Path

import pandas as pd

from . import config
from . import staging
from .doi_utils import DoiIndex
from .id_allocator import IdAllocator

_NAME_SEPARATORS_RE = re.compile(r"[\s/\-_+&(),]+")


def _normalize_method_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def _name_candidates(name: str) -> set:
    """Every normalized form worth INDEXING for a given existing name: its
    own full normalized form (separators stripped, not split - e.g.
    'Mask R-CNN' -> 'maskrcnn'), PLUS each individual separator-split
    sub-token (/, -, _, +, &, whitespace, parens, commas - e.g.
    'Mesmer/DeepCell' -> also 'mesmer', 'deepcell'). Indexing the whole
    form as well as the sub-tokens matters because a real duplicate can
    show up with the separator on EITHER side of the pair (existing
    'MaskRCNN' + new citation 'Mask R-CNN', or existing 'Mesmer/DeepCell'
    + new citation 'Mesmer') - only indexing sub-tokens would catch the
    second case but miss the first, since 'maskrcnn' is never a sub-token
    of 'Mask R-CNN' on its own (it splits into 'mask'/'r'/'cnn'). A length
    guard (>=4 normalized characters) avoids over-matching on short generic
    tokens ("R", "ML" alone)."""
    whole = _normalize_method_name(name)
    subtokens = {re.sub(r"[^a-z0-9]", "", t.lower())
                 for t in _NAME_SEPARATORS_RE.split(str(name or "")) if t.strip()}
    return {c for c in ({whole} | subtokens) if len(c) >= 4}


class NameIndex:
    """
    In-memory lookup for method-name-based dedup, added 2026-09-27:
    DoiIndex above only catches a duplicate when both mentions share the
    exact same DOI - a real duplicate is missed whenever a comparison
    method is cited casually by name with no resolvable citation, or under
    a slightly different combined name (e.g. a new citation naming just
    "Mesmer" when the existing DB entry is titled "Mesmer/DeepCell").
    Real live examples that motivated this: Mesmer, CellProfiler,
    PhenoGraph, and Mask R-CNN were each independently created twice by
    compared_methods_agent.py's citation-chasing.

    A target name matches if its FULL normalized form (see
    _normalize_method_name) is one of an existing entry's indexed
    candidate forms (see _name_candidates) - this catches "exact same
    name, different casing/punctuation" in EITHER direction (existing
    "MaskRCNN" + new "Mask R-CNN", or the reverse) and "one name is a
    combined/aliased form containing the other" (e.g. "Mesmer" inside
    "Mesmer/DeepCell"). Deliberately conservative, not exhaustive - two
    genuinely different tools that happen to share no common form still
    won't be caught; treat this as a maintenance-reducing net, not a
    guarantee, same spirit as DoiIndex's own docstring.
    """

    def __init__(self):
        self._form_to_id = {}

    def load_from_dataframe(self, df: pd.DataFrame, name_col: str = "name", id_col: str = "entry_id"):
        for _, row in df.iterrows():
            entry_id = row.get(id_col, "")
            if pd.isna(entry_id):
                continue
            entry_id = str(entry_id).strip()
            if not entry_id or entry_id.lower() in ("", "nan", "na"):
                continue
            name = row.get(name_col, "")
            if pd.isna(name) or not str(name).strip():
                continue
            for form in _name_candidates(name):
                self._form_to_id.setdefault(form, entry_id)

    def add(self, name: str, entry_id: str):
        for form in _name_candidates(name):
            self._form_to_id.setdefault(form, entry_id)

    def lookup(self, name: str):
        """Returns an existing entry_id if `name`'s full normalized form
        matches any indexed candidate form, else None."""
        full = _normalize_method_name(name)
        if len(full) < 4:
            return None
        return self._form_to_id.get(full)


def _latest(pattern: str) -> Path:
    # Sort key normalizes "-" to "_" - plain sorted() puts "-" before "_" in
    # ASCII, so e.g. "methods_2026-08-29.csv" would sort (and get picked as
    # "latest") BEFORE "methods_2026_07_07.csv", silently loading a stale
    # two-months-older file. Same bug/fix as 01_parse_excel.py's and
    # 03_export_json.py's find_latest() (fixed there 2026-09-01) - this
    # module's own copy was missed at the time. Caught 2026-09-17 while
    # building the literature-search scanner: every agent using
    # db_loader.Database() (compared_methods_agent.py, data_fetch_agent.py,
    # this scanner) was silently deduping against a stale DB whenever both
    # filename date-formats coexisted in data/processed/.
    matches = sorted(config.PROCESSED_DIR.glob(pattern), key=lambda p: p.stem.replace("-", "_"))
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
        self.name_index = NameIndex()
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

        # Method names, for name-based dedup (see NameIndex) - methods sheet
        # only, datasets have no equivalent "name" concept.
        self.name_index.load_from_dataframe(self.methods, "name", "entry_id")

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

            if rec.get("target_sheet") in ("method_pub", "AP_pub"):
                doi = rec.get("DOI", "")
                if doi:
                    self.doi_index.add(doi, entry_id)
                name = rec.get("name", "")
                if name:
                    self.name_index.add(name, entry_id)
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
