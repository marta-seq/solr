# src/agents — Phase 2 review department

Status as of this drop. Update this file as pieces get added.

## Built and tested (against the real methods_metadata/datasets CSVs)

- `common/config.py` — tunable knobs: MAX_HOPS, MAX_PAPERS_PER_RUN, model
  fallback chain, paths. **TODO: set your real contact email** (search for
  CONTACT_EMAIL) — confirm whether to read it from the same env var as your
  existing PubMed setup, or keep it separate.
- `common/doi_utils.py` — `normalize_doi` (mirrors `01_parse_excel.py`
  exactly - keep both in sync if either changes), `DoiIndex` for dedup.
- `common/id_allocator.py` — next free `M_PR_x` / `D_SP_IMC_x` etc, by
  scanning existing IDs rather than trusting a stored counter.
- `common/paper_fetcher.py` — tiered full-text fetch (Unpaywall -> Europe
  PMC -> bioRxiv -> Crossref abstract-only), cached to `data/paper_cache/`.
  Needs `pdfplumber` installed (`pip install pdfplumber --break-system-packages`).
- `common/llm_client.py` — OpenRouter wrapper with model fallback chain.
  Needs `OPENROUTER_API_KEY` env var. Verify the free-model slugs in
  `config.py` are still live at openrouter.ai/models (Price: Free) before a
  big run - the roster rotates.
- `common/staging.py` — candidates get written to a continuously-saved
  Excel workbook (`data/agent_review/staging_<date>.xlsx`), not the master
  file. Saved to disk after every single candidate.
- `common/db_loader.py` — loads latest `methods_metadata_*.csv` /
  `datasets_*.csv`, builds the shared `DoiIndex` + `IdAllocator`.
- `mailroom/triage.py` — `build_seed_queue()` (paper pool -> excludes
  non-method + already-reviewed [manual or auto] -> 84 seed papers on your
  current data) and `build_data_pool()` (dataset pool -> excludes
  already-reviewed, modality-aware metadata gap check -> 17 rows on your
  current data).

## Not built yet - empty package stubs only

- `methods_desk/compared_methods_agent.py` - reads a paper's methods
  section, extracts compared-methods references, resolves them to DOIs,
  matches/creates `M_PR` entries, stages candidates. Recursive (feeds new
  entries back into the seed queue at depth+1, capped by MAX_HOPS).
- `data_desk/data_fetch_agent.py` - reads a paper's data-availability
  section, classifies each dataset mention as self-generated vs
  reused-external, resolves the reused case to a DOI, matches/creates
  `D_*` entries (+ stub `Application` paper if the source paper doesn't
  exist yet), stages candidates. Single pass, no recursion.
- `data_desk/intern_agent.py` - works off `build_data_pool()`'s worklist,
  fills empty metadata fields from the associated paper's abstract, stages
  candidates. Independent of the paper-recursion track entirely.
- `common/reference_resolver.py` - shared by both desk agents: given a
  citation's title/author text, searches Crossref and fuzzy-matches to a
  DOI. Not written yet - needed before either desk agent can be finished.
- `run_pipeline.py` - orchestrator that ties triage -> desks -> staging
  together, respecting MAX_HOPS/MAX_PAPERS_PER_RUN.

## Before running anything

```bash
pip install pdfplumber --break-system-packages
export OPENROUTER_API_KEY="..."          # openrouter.ai/keys
# confirm/set your contact email in common/config.py
```

`data/paper_cache/` and `data/agent_review/` are created automatically on
first import (see `config.py`) - empty on a fresh checkout.
