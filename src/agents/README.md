# src/agents — Phase 2 review department

Status as of this drop. Everything is built. Update this file as things change.

## Built and tested (against the real methods_metadata/datasets CSVs)

- `common/config.py` — tunable knobs: MAX_HOPS, MAX_PAPERS_PER_RUN, model
  fallback chain, paths. Reads contact email from `SOLR_EMAIL` env var
  (falls back to a placeholder if unset - fill in the fallback if you're
  not using that var name).
- `common/doi_utils.py` — `normalize_doi` (mirrors `01_parse_excel.py`
  exactly - keep both in sync if either changes), `DoiIndex` for dedup.
- `common/id_allocator.py` — next free `M_PR_x` / `D_SP_IMC_x` etc, by
  scanning existing IDs rather than trusting a stored counter.
- `common/paper_fetcher.py` — tiered full-text fetch (Unpaywall -> Europe
  PMC -> bioRxiv -> Crossref abstract-only), cached to `data/paper_cache/`.
  `get_agent_text(paper, section_name)` is the one function agents should
  call to pull a section with sensible fallback + truncation tracking.
  Needs `pdfplumber` installed (`pip install pdfplumber --break-system-packages`).
- `common/llm_client.py` — OpenRouter wrapper with model fallback chain.
  Needs `OPENROUTER_API_KEY` env var. Verify the free-model slugs in
  `config.py` are still live at openrouter.ai/models (Price: Free) before a
  big run - the roster rotates.
- `common/reference_list_parser.py` — parses a paper's own reference list
  into {marker: full_reference_text}, so a numbered in-text citation like
  "[12]" gets resolved to the EXACT reference as printed, not an LLM's
  recollection of it. Falls back cleanly (empty dict) for author-year style
  papers with no numbered references.
- `common/reference_resolver.py` — `resolve_citation()` is the main entry
  point: tries the reference list first (instant + exact if the entry
  already prints a DOI), then Crossref search on that exact text, and only
  falls back to resolving the LLM's recalled citation_text if no reference
  list was available at all.
- `common/staging.py` — candidates get written to a continuously-saved
  Excel workbook (`data/agent_review/staging_<date>.xlsx`), not the master
  file. Saved to disk after every single candidate.
- `common/db_loader.py` — loads latest `methods_metadata_*.csv` /
  `datasets_*.csv`, builds the shared `DoiIndex` + `IdAllocator`.
- `mailroom/triage.py` — `build_seed_queue()` (paper pool -> excludes
  non-method + already-reviewed [manual OR auto] -> 84 seed papers on your
  current data) and `build_data_pool()` (dataset pool -> excludes
  already-reviewed, modality-aware metadata gap check -> 17 rows on your
  current data).
- `methods_desk/compared_methods_agent.py` — `process_paper(db, paper_entry, fetched=None, reference_map=None)`.
  Reads the methods section, asks the LLM which methods are compared
  against (returning a reference marker like "12" where possible, citation
  text only as fallback for author-year papers), resolves via
  `reference_resolver.resolve_citation()`, matches against the live
  `DoiIndex` or creates a new `M_PR` entry, updates the source paper's
  `Method_comparison_P_ENTRY_ID`, stages everything, and returns newly
  created entries for the orchestrator to queue at depth+1.
- `data_desk/data_fetch_agent.py` — `process_paper(db, paper_entry, fetched=None, reference_map=None)`.
  Classifies each data-availability mention as self-generated vs
  reused-external, resolves the reused case via the same
  `reference_resolver.resolve_citation()`, matches against `data_DOI` or
  `data_accession_number` or creates a new `D_*` entry (+ stub `Application`
  paper, `AP_` prefix, if the origin paper isn't in the DB yet). Single
  pass, no recursion.
- `data_desk/intern_agent.py` — `process_entry(db, data_item)`. Works off
  `build_data_pool()`'s worklist, prefers the precomputed `abstract` column
  in `methods_metadata.csv` (free) over a fresh fetch, resolving
  `paper_ENTRY_ID` via the DOI index first if it's blank on the dataset row
  but `paper_DOI` matches a known paper. Fills ONLY fields the LLM found
  explicit textual support for - partial fills are expected and correct,
  not a bug.
- `run_pipeline.py` — the orchestrator. `main()` loads the DB, runs the
  paper queue (methods desk + data desk together, one fetch per paper
  shared between both), then runs the data pool (intern agent), then
  reports a summary. **`MAX_PAPERS_PER_RUN` is a single budget SHARED
  across the paper queue and the data pool, in that order** - a run with
  budget 40 that uses 25 on papers only has 15 left for the data pool, it
  does NOT get 40+40.
  Run with: `python -m src.agents.run_pipeline`

**Everything above has been dry-run tested with mocked fetch/LLM/Crossref
calls against your real CSVs, including a full end-to-end run of
`run_pipeline.main()`. The actual network calls (OpenRouter, Crossref,
Unpaywall/Europe PMC/bioRxiv) are UNTESTED against the real internet - this
dev sandbox has no network access. First real run should use a small
`MAX_PAPERS_PER_RUN` (5-10) to sanity-check before a big batch.**

## Real bugs found via testing against your actual data (all fixed)

- `paper_fetcher.py`: `fetch_paper()` was orphaned from its own `def` line.
- `db_loader.py`: a dataset's `paper_DOI` was indexed under the dataset's
  own `entry_id` instead of `paper_ENTRY_ID`, silently corrupting DOI
  lookups for the paper that DOI actually belongs to.
- `doi_utils.py`: `DoiIndex.load_from_dataframe` treated pandas' `NaN`
  (a truthy float in plain Python) as a valid entry_id, indexing real DOIs
  against the literal string `"nan"`.
- `run_pipeline.py`: `MAX_PAPERS_PER_RUN` was applied as a separate budget
  per phase instead of one shared budget across the whole run, silently
  doubling real LLM/API usage vs. what the config implies.

## Data-quality things noticed while testing (not fixed here - your call)

- A few rows in `Data_SP`/`Data_ST` have scratch notes as their `entry_id`
  (e.g. "add spacejam hackaton data") instead of a real `D_*` ID - these
  currently pass `is_valid_id()` and get treated as real dataset rows by
  `build_data_pool()`. Worth a stricter ID-shape check in `triage.py` if
  you want them excluded.
- Many `Data_*` rows have `paper_DOI` filled in but `paper_ENTRY_ID` blank,
  even when that DOI matches a real paper elsewhere in the sheet.
  `intern_agent.py` resolves this via the DOI index at read-time, but the
  underlying link is implicit rather than stored - might be worth fixing
  at the source in `01_parse_excel.py` too.
- `M_PR_4`'s existing `Method_comparison_P_ENTRY_ID` value contains a
  scratch note ("do not show nay data, ...") alongside real IDs - agents
  preserve existing content and append rather than overwrite, so this is
  harmless but will look odd in the staged output until cleaned up by hand.

## Before running anything for real

```bash
pip install pdfplumber --break-system-packages
export OPENROUTER_API_KEY="..."          # openrouter.ai/keys
# confirm SOLR_EMAIL is set, or edit the fallback in common/config.py
```

`data/paper_cache/` and `data/agent_review/` are created automatically on
first import (see `config.py`) - empty on a fresh checkout.

Recommend starting with a small `MAX_PAPERS_PER_RUN` (5-10) for the first
real run. Check `data/agent_review/staging_<date>.xlsx` afterward before
scaling up - nothing gets merged into your master Excel automatically.
