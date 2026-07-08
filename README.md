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

Phase 1 — manually curated static version. A core set of well-curated papers with real comparison edges, the rest as stubs. Hosted as static files on GitHub Pages, no backend.

## What's next

Phase 2 will introduce an automated review pipeline — a set of agents that scrape bioRxiv and PubMed, categorise new papers, extract which methods they compare against and which datasets they use, and populate the database automatically. The goal is a graph that stays current without manual effort, while preserving manual curation quality markers so users know what to trust.

## Structure

```
data/
  data_curated/        ← source of truth (Excel)
  data_curated_backup/ ← versioned backups
  processed/           ← cleaned CSVs and JSON outputs
src/
  preprocessing/
    01_parse_excel.py  ← clean and export from Excel
    02_fetch_metadata.py ← enrich with Crossref/PubMed metadata
    03_export_json.py  ← export to JSON for the frontend
docs/                  ← static site (GitHub Pages)
  index.html
  data/
  book/
version1/              ← archived first prototype
```

## Contributing

If you want to suggest a missing method, flag an error, or add a dataset — open an issue or a pull request. The source of truth is the curated spreadsheet; contributions to that are the most valuable.

---

*Started during a PhD in spatial omics. The man in the boat knows the feeling.*

![Man in boat on a sea of papers](figures/img_1.png)