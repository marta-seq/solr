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
4. **Export + publish**: the promoted spreadsheet is parsed, enriched with Crossref/
   PubMed metadata, exported to JSON, embedded for the semantic graph layer, and
   pushed live.

See **[`PIPELINE_OVERVIEW.md`](PIPELINE_OVERVIEW.md)** for the full box-diagram version
of this flow, file by file.

Hosted as static files on GitHub Pages — no backend, no server, no database; the
frontend just reads pre-generated JSON.

## What's next

Open items (always check `context/TODO.md` for the live version of this if you have
Claude Code context access) include: scheduling the ingestion + agent-curation cycle
to run on its own on a recurring basis, an API-cost/provider strategy now that free-
tier quotas are a real constraint at this volume, and exploring alternatives to Excel
as the master spreadsheet grows past what's comfortable to review by hand.

## Structure

```
data/
  data_curated/        ← source of truth (Excel) - exactly one active file at a time
  data_curated_backup/ ← every superseded version, dated (the real version history)
  processed/           ← cleaned CSVs from the export pipeline
  agent_review/        ← staging.xlsx - agent proposals pending manual review
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
    run_pipeline.py         orchestrates all of the above
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