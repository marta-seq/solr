# SOLR — Spatial Omics Living Review

A curated, graph-based living review of computational methods and datasets for spatial omics analysis. Built out of frustration with static reviews that become outdated before they are published.

<br>

<p align="center">
  <a href="https://marta-seq.github.io/solr/" target="_blank">
    <img src="https://img.shields.io/badge/View_on_GitHub_Pages-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="View on GitHub Pages">
  </a>
</p>

<img src="figures/img_2.png" alt="Navigating the sea of papers" width="700">

---

## What is this?

The spatial omics methods landscape moves faster than any static review can keep up with. SOLR is an attempt to fix that — a living, browsable graph of computational methods, benchmarking relationships, and datasets, maintained by someone who actually used these tools during a PhD.

The core idea: methods are only meaningful in context. Which datasets were used to benchmark them? Against which other methods? A tool validated only on mouse brain cortex is a different thing from one validated on human tumour tissue. SOLR makes that context visible.

## What's inside

- **Methods graph** — computational methods grouped by pipeline stage (preprocessing, cell segmentation, phenotyping, niche analysis, spatially variable genes, cell-cell communication), with edges showing which methods were compared against each other and on what data
- **Dataset registry** — curated spatial omics datasets with tissue type, disease, organism, modality, technology, and number of markers/genes — with links to download, nothing hosted here
- **Methods book** — a narrative overview of the methods landscape, written during a PhD in spatial omics

## Current status

Past the "manually curated only" stage — a real, running agent pipeline now does most
of the first-pass curation, with Marta reviewing and correcting before anything
reaches the live site. In order of what actually happens to a paper:

1. **Ingestion** (3 ways in): manual entry, an automated literature scanner (PubMed/
   bioRxiv/medRxiv), or citation-chasing (an agent reading one paper discovers another
   it compares against or reuses data from).
2. **Agent curation** ("the desks"): a cheap audit gate checks a paper is genuinely
   computational and figures out its spatial modality; method papers go to a desk that
   extracts compared methods and datasets; application papers go to a separate desk
   that extracts tissue/organism/disease metadata and checks (strictly) whether the
   paper generated genuinely new data itself.
3. **Manual review**: everything an agent proposes is staged, never applied directly —
   Marta reviews a batch, corrects what's wrong, and only then promotes it to become
   the new source-of-truth spreadsheet.
4. **Export + publish**: the spreadsheet is parsed, enriched with Crossref/PubMed
   metadata, exported to JSON, embedded for the semantic graph layer, checked by an
   automated gate, and published by a weekly job (see below).

See **[`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md)** for the full box-diagram version
of this flow, file by file.

Hosted as static files on GitHub Pages — no backend, no server, no database; the
frontend just reads pre-generated JSON.

## Automation (weekly update)

A weekly job on the server that runs the pipeline keeps the site current without
touching the human-reviewed spreadsheet:

1. **Scrape + agents** (`src/agents/orchestrate.py`) — scan PubMed/bioRxiv/medRxiv since
   the last run and run the curation desks on the new papers. Output goes to
   `staging.xlsx` only. All LLM stages run on local models (Ollama).
2. **Build** (`src/publish/build_site.py`) — apply the accumulated staging rows onto a
   *copy* of the master, run steps 01–04 on it and write the site JSON to a scratch
   folder. The master is never modified.
3. **Gate** (`src/publish/gate.py`) — deterministic checks (no LLM) comparing the new JSON
   with what is currently published: valid JSON, no duplicate ids, row counts did not
   drop, no more than a fixed number of new rows, no reviewed row lost, no new dangling
   links, files did not shrink. If any check fails, nothing is published.
4. **Publish** (`src/publish/publish.py`) — a plain script (not an LLM) that copies only
   `docs/data/*.json` into a dedicated clone, commits as `solr-bot`, tags the commit
   `data-YYYY-MM-DD` and pushes to `main`. It refuses to push if anything else changed.
5. **Digest** (`src/publish/weekly.py`, `notify.py`) — a status file, plus an optional
   email, with new/unreviewed row counts and a reminder when the last manual review is
   old.

Rows added automatically are published flagged as *unreviewed*; the site footer shows the
last update date and the date of the last human review. To roll back a bad publish,
revert the `solr-bot` commit (or reset to the previous `data-*` tag).

Manual review is unchanged: merge the staged candidates into an `autoreview` file, review
it, promote it to the master, then publish with
`python -m src.publish.build_site --push`.

## What's next

- Extending the review beyond spatial proteomics to other spatial modalities
  (transcriptomics, metabolomics, epigenomics, multi-omics), with per-modality method
  tables.
- Email digest for the weekly job.
- Exploring alternatives to Excel as the master spreadsheet grows past what is
  comfortable to review by hand.

## Structure

```
data/
  data_curated/        ← source of truth (Excel) - exactly one active file at a time
  data_curated_backup/ ← every superseded version, dated (the real version history)
  processed/           ← cleaned CSVs from the export pipeline
  agent_review/        ← staging.xlsx - agent proposals pending manual review
  site_build/          ← scratch output of the weekly build (not published directly)
  weekly/              ← weekly run logs, backups, STATUS.txt, last_digest.txt
  manual_papers/       ← PDFs of paywalled papers, used as full text on the next run
src/
  preprocessing/        Phase 1 - the export pipeline
    01_parse_excel.py       clean and export from Excel
    02_fetch_metadata.py    enrich with Crossref/PubMed metadata
    03_export_json.py       export to JSON for the frontend
    04_compute_embeddings.py embeds + positions the semantic graph layer
  agents/               Phase 2/3 - the automated ingestion + curation pipeline
    common/                 shared infra (LLM provider config, paper fetcher/
                             cache, staging writer, DB indexes)
    living_ingestion/       mailroom/ (work-queue triage), literature_search/
                             (the scanner), merge_candidates.py
    agent_curation/         methods_desk/, data_desk/, application_desk/ -
                             see PIPELINE_OVERVIEW.md for what each does
    run_pipeline.py         runs the curation desks (shared --max-papers budget)
    orchestrate.py          one command for a cycle: preflight, scrape, agents -> staging
  publish/              weekly automation
    build_site.py           staging + master -> 01..04 -> site JSON (scratch folder)
    gate.py                 deterministic checks before publishing
    publish.py              commits/pushes docs/data/*.json as solr-bot
    weekly.py, notify.py    the cron entry point, status file and digest email
docs/                  ← static site (GitHub Pages)
  index.html
  data/                     methods.json / datasets.json / stats.json /
                             embeddings.json - what the frontend actually reads
  js/
  book/
version1/              ← archived first prototype
```

## Contributing

If you want to suggest a missing method, flag an error, or add a dataset — open an issue or a pull request. The source of truth is the curated spreadsheet; contributions to that are the most valuable.

---

*Started during a PhD in spatial omics. The man in the boat knows the feeling.*

![Man in boat on a sea of papers](figures/img_1.png)