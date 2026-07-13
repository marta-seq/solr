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
  Needs `pdfplumber` installed (`pip install pdfplumber beautifulsoup4 lxml --break-system-packages`).
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
- `common/staging.py` — candidates get written to a SINGLE ongoing Excel
  workbook (`data/agent_review/staging.xlsx` - NOT date-stamped anymore),
  saved to disk after every single candidate. Keeps accumulating across
  every run until you actually merge it - `merge_candidates.py` should
  archive/clear it once merged, so the next run starts clean. Each row's
  own `curation_date` column preserves per-candidate timing, which is why
  the file itself doesn't need to be split by day.
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
  does NOT get 40+40. Per-paper log now shows encountered/resolved/new
  counts for both desks separately (e.g. "encountered 2 compared method(s)
  -> 2 resolved, 1 genuinely NEW"), not just the new-entry count.
  Run with: `python -m src.agents.run_pipeline [--max-papers N]`
- `common/config.py`'s `REQUIRE_ISOLATED_SECTION` (default True): both desk
  agents now skip the LLM call ENTIRELY when the target section wasn't
  cleanly isolated (source must be exactly `section:methods` /
  `section:data_availability`, not `full_text_fallback` or `abstract_only`)
  - avoids spending scarce free-tier queries on unfocused/low-confidence
    text. Set False to also try the LLM on those lower-confidence inputs.
  The log now states this UNAMBIGUOUSLY - "SKIPPED - LLM was NOT called
  (reason)" vs "LLM WAS called - encountered N...". A prior version left
  this to be inferred from a 0 count (which could mean either "skipped" or
  "called and found nothing"), which caused real confusion during testing
  even when the gate was working correctly - now every result explicitly
  says which case it is.

- `merge_candidates.py` — the missing piece of the cycle. Finds the most
  recent `datasets_curated_*.xlsx` in `data/data_curated/`, applies every
  staged candidate into a NEW file (`datasets_curated_autoreview_<date>.xlsx`
  - never overwrites your original), then archives `staging.xlsx` so the
  next pipeline run starts clean. Rules: never touches a row with
  `REVIEW_STATUS == "manual"` (checked directly, on top of triage already
  excluding these); last-written-wins for any cell two candidates both
  touch (staging.xlsx is chronological, so processing it in order gives
  this naturally); `needs_review` entries merge in alongside `auto` ones,
  same file, sort/filter by `REVIEW_STATUS` in Excel yourself.
  **`Data_multi`'s real columns don't match `data_fetch_agent.py`'s field
  names at all** (its DOI columns are `'DOI '`/`'DOI'`, not `data_DOI`/
  `paper_DOI`) - rather than guess and risk writing into the wrong column
  of a confusingly-structured sheet, anything targeting `Data_multi` goes
  into a separate `NEEDS_MANUAL_PLACEMENT` sheet instead of being
  auto-merged. Column-name matching is whitespace-insensitive (handles
  real header quirks like `'data_DOI '` with a trailing space). **Tested
  against your actual real master file** (`datasets_curated_2026_07_07.xlsx`)
  - verified a papers update, a new papers entry, a new Data_ST entry
  (confirming the trailing-space column match), and a Data_multi candidate
  correctly routing to NEEDS_MANUAL_PLACEMENT instead of risking corruption.
  Run with: `python -m src.agents.merge_candidates`

**Everything above (except merge_candidates.py, tested against your real
master file directly) has been dry-run tested with mocked fetch/LLM/
Crossref calls against your real CSVs, including a full end-to-end run of
`run_pipeline.main()`. The actual network calls (OpenRouter, Crossref,
Unpaywall/Europe PMC/bioRxiv) ARE now live-tested via your real runs on
gaia, surfacing and fixing many real bugs listed below.**

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
- `reference_resolver.py`: Crossref search returned figure/table/component
  DOIs (not just article DOIs) that could out-score the real article on
  title similarity, causing wrong matches/duplicates (e.g. SSAM/Baysor
  each got a new entry instead of matching their existing ones). Fixed by
  filtering to accepted article/preprint Crossref types only.
- `paper_fetcher.py` (caught on a real live run - the big one): blind
  regex tag-stripping on Unpaywall HTML and Europe PMC XML left huge
  amounts of website chrome mixed into "full text" - navigation menus,
  CRediT author-contribution-role lists (e.g. "Methodology" appearing as
  a role label, not a heading), figure-download widgets. This confused
  section-heading detection AND got sent straight to the LLM, wasting
  real calls on unreliable input. Fixed two ways: (1) Europe PMC's real
  JATS XML `<sec>` tags are now parsed directly for genuine section
  structure instead of regex-guessing on flattened text; (2) Unpaywall
  HTML is now cleaned with BeautifulSoup (nav/header/footer/script/style
  removed, `<article>` content preferred) instead of blind tag-stripping.
  Also added `is_probably_real_content()` as a pre-LLM quality gate in
  both desk agents, so garbage text never even reaches the LLM call -
  verified it correctly blocks on realistic chrome-polluted text while
  still passing real methods text through. Further improved: Unpaywall's
  HTML path now parses real DOM structure two ways before falling back to
  regex-on-flattened-text - Tier 1 uses actual `<h1>`-`<h6>` heading tags
  (classified by their own short text), Tier 2 (only if Tier 1 finds
  nothing) looks for short "pseudo-heading" elements some publishers use
  instead of real heading tags (e.g. `<p><strong>Methods</strong></p>`).
  Verified all three tiers against realistic HTML.
- `db_loader.py`: a fresh `Database()` never re-read already-staged
  candidates, so re-running the pipeline - even the SAME day - could create
  a SECOND, differently-numbered entry for a DOI an earlier run already
  staged. First fix only scoped this to "today's" file, which was still
  wrong: resuming across a day boundary (very likely given free-tier daily
  rate limits) would miss yesterday's staged candidates entirely. Real fix:
  switched `staging.py` to a single ongoing `staging.xlsx` (not date-
  stamped) and ingest everything in it on startup, regardless of when it
  was written. Verified a later run correctly sees an earlier run's staged
  entry and allocates the next free ID instead of colliding or duplicating.

**Migration note if you already have a run in progress:** a running Python
process has the old code loaded in memory and will keep writing to
`staging_<date>.xlsx` regardless of this fix. Once that run finishes,
rename its output file to `staging.xlsx` (dropping the date) before running
again, so the next run picks up everything already staged.

- `mailroom/triage.py`: `build_seed_queue()` now ALSO excludes papers
  already attempted this session (anything staged in `staging.xlsx` by
  either desk, regardless of outcome) - without this, re-running the
  pipeline before merging staging.xlsx into the master CSV would restart
  from the SAME seed papers instead of continuing to the next unprocessed
  ones (REVIEW_STATUS only changes once you actually merge, so nothing
  else marked those papers as "done"). Verified: simulated 15
  already-attempted papers, confirmed the next run's queue correctly
  picked up at paper #16 with zero overlap.

- `reference_resolver.py`/`compared_methods_agent.py` (caught from real
  staged output, several distinct issues): (1) the "embedded DOI in
  reference list" path trusted confidence 1.0 unconditionally, so a
  reference-list marker-to-entry mapping error produced confidently WRONG
  entries stamped `auto` (e.g. "ImageJ" resolved to an unrelated book
  chapter, "StarDist" resolved to an unrelated 3D-printer paper) - now
  verifies the embedded DOI's real Crossref title actually appears in its
  own reference text AND that the method name the LLM identified appears
  somewhere in what was resolved, downgrading confidence when either check
  fails. Verified: catches the ImageJ-style case (zero text overlap)
  correctly; honestly does NOT catch the specific StarDist case, where the
  wrong paper's own title coincidentally contains the literal word
  "Stardist" as an unrelated product name - no simple text check can tell
  that apart from the real tool, so that class of error still needs a
  human glance. (2) prompt was too permissive - accepted generic author-
  year citations (e.g. "Zeisel et al.") as if they were named methods; now
  explicitly requires method_name to be an actual tool/algorithm proper
  name. (3) the LLM sometimes put marker-like text ("ref. 41(") into
  citation_text instead of citation_marker despite the prompt asking
  otherwise, silently skipping the reliable reference-list lookup in favor
  of an unreliable Crossref search on garbled text - added a lenient
  fallback regex to recover the marker from citation_text when this
  happens. All three verified against the exact real examples that
  surfaced them.

- `doi_utils.py`: `normalize_doi` didn't strip trailing sentence punctuation
  swept up when extracting a DOI from natural-language reference text (e.g.
  "...see https://doi.org/10.1038/s41467-021-23807-4." - the period ends
  the SENTENCE, not the DOI). Caught from real staged output: EVERY single
  embedded-DOI extraction had a trailing period, meaning a correct DOI
  match against an existing entry (which correctly has no trailing period)
  would silently fail as two different-looking strings for what should be
  the identical DOI - producing duplicate entries for methods (Baysor,
  pciSeq, SCS) that likely already existed. Fixed and verified against all
  6 real examples that surfaced this, plus confirmed a legitimate DOI with
  genuine internal periods (a bioRxiv date-style DOI) is untouched.

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
pip install pdfplumber beautifulsoup4 lxml --break-system-packages
export OPENROUTER_API_KEY="..."          # openrouter.ai/keys
# confirm SOLR_EMAIL is set, or edit the fallback in common/config.py
```

`data/paper_cache/` and `data/agent_review/` are created automatically on
first import (see `config.py`) - empty on a fresh checkout.

Recommend starting with a small `MAX_PAPERS_PER_RUN` (5-10) for the first
real run. Check `data/agent_review/staging.xlsx` afterward before
scaling up - nothing gets merged into your master Excel automatically.

## The full cycle: run -> review -> merge -> re-preprocess -> repeat

**1. Run the pipeline** (as many times as you want, across as many days as
you want - it always continues from where it left off, never restarts):
```bash
python -m src.agents.run_pipeline                  # uses MAX_PAPERS_PER_RUN from config.py
python -m src.agents.run_pipeline --max-papers 15   # or override it for one run
```
Everything proposed goes into `data/agent_review/staging.xlsx` - your
master Excel is never touched at this stage.

**2. When you're ready to actually merge what's in staging.xlsx:**
```bash
python -m src.agents.merge_candidates
```
This finds your most recent `datasets_curated_*.xlsx` in
`data/data_curated/`, applies every staged candidate into a **brand new**
file - `datasets_curated_autoreview_<date>.xlsx` - and never overwrites
your original. It also archives `staging.xlsx` to
`staging_merged_<date>.xlsx`, so the next `run_pipeline` run starts clean
instead of re-ingesting already-merged candidates.

The console output tells you exactly what happened:
```
[merge] Using master file: datasets_curated_2026_07_07.xlsx
[merge] Wrote datasets_curated_autoreview_2026-07-13.xlsx:
[merge]   42 candidates applied
[merge]   0 skipped (manually-reviewed rows protected)
[merge]   3 routed to NEEDS_MANUAL_PLACEMENT sheet
[merge] Archived staging.xlsx -> staging_merged_2026-07-13.xlsx
```

**3. Open `datasets_curated_autoreview_<date>.xlsx` and review it by hand.**
Sort/filter by `REVIEW_STATUS` to find `needs_review` and `auto` rows.
Check the `AUTO_CURATION_AGENT`/`AUTO_CURATION_MODEL`/`AUTO_CONFIDENCE`/
`AUTO_NOTES` columns added alongside your normal columns - they tell you
which agent/model proposed each auto-added cell and why. Also check the
`NEEDS_MANUAL_PLACEMENT` sheet if the merge output mentioned any - those
are `Data_multi` candidates that couldn't be safely auto-merged (see
"Built and tested" above for why) and need you to place them by hand.
Fix wrong DOIs/prefixes, delete anything wrong, reassign real categories
instead of the generic `M_AUTO` placeholder prefix - whatever needs doing.

**4. Once you're happy with it, re-run your existing preprocessing chain**
pointed at that reviewed file, to regenerate the CSVs the pipeline actually
reads:
```bash
python src/preprocessing/01_parse_excel.py
python src/preprocessing/02_fetch_metadata.py
python src/preprocessing/03_export_json.py
```
(adjust paths/args to however you normally point these at a specific input
file - the point is just: the *reviewed* `autoreview` file becomes the new
`methods_metadata.csv`/`datasets.csv` source, not the old one.)

**5. Run `run_pipeline.py` again.** `db_loader.py` always grabs the most
recently *dated* CSV, so it automatically picks up your reviewed baseline -
nothing to configure. The cycle repeats from step 1.

**You never have to merge after every single run.** Staging accumulates
safely across as many `run_pipeline` runs as you want; merge whenever you
actually want a reviewable checkpoint, not on any fixed schedule.
