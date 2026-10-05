# SOLR — pipeline overview

How a paper actually gets from "somewhere on the internet" to a row on the live site.
A text-box version of the same flow described in more detail in the code and (if
you have Claude Code context access) `context/ARCHITECTURE.md` / `context/pipeline_steps.md`.

## 1. Three ways a paper enters the system

```
  ┌─────────────────┐   ┌─────────────────────┐   ┌───────────────────────┐
  │   MANUAL ENTRY   │   │ LITERATURE SCANNER  │   │   CITATION-CHASING    │
  │                  │   │                      │   │                        │
  │ Marta adds a row │   │ scan.py scrapes      │   │ a desk agent, while    │
  │ directly to the  │   │ PubMed / bioRxiv /   │   │ reading paper X,       │
  │ master Excel      │   │ medRxiv on a keyword │   │ finds it compares      │
  │ (title/DOI known) │   │ filter, tags each    │   │ itself against paper Y │
  │                  │   │ hit method/app/      │   │ or reuses dataset Z -  │
  │                  │   │ review + category    │   │ Y/Z get pulled in too  │
  └────────┬─────────┘   └──────────┬───────────┘   └───────────┬────────────┘
           │                        │                            │
           └────────────────────────┴────────────────────────────┘
                                     │
                                     ▼
                     staged as a method_pub or AP_pub row
                  (REVIEW_STATUS: manual / scraped / needs_review)
```

## 2. Agent processing (Phase 3 — "the desks")

Every row, regardless of how it arrived, flows into one shared triage step, then
splits by paper type:

```
                     ┌───────────────────────────────┐
                     │   category_audit_agent.py     │
                     │  (cheap title+abstract check:  │
                     │  genuinely computational? what │
                     │  spatial modality?)            │
                     └───────────────┬─────────────────┘
                                     │
                flagged "not a method"│  looks correct
                (wet-lab/hardware)    │
                     ▼                ▼
              needs_review      is this method_pub or AP_pub?
              (stop here)              │
                        ┌──────────────┴───────────────┐
                        ▼                                ▼
          ┌─────────────────────────┐      ┌───────────────────────────────┐
          │      METHODS DESK        │      │      APPLICATION DESK          │
          │  (method_pub only)        │      │  (AP_pub only)                 │
          │                           │      │                                 │
          │ compared_methods_agent.py │      │ application_review_agent.py    │
          │  → what methods does this │      │  → animal/tissue/disease/      │
          │    paper benchmark itself │      │    technique (metadata call)   │
          │    against? resolves DOIs,│      │  → did THIS paper itself       │
          │    creates new M_AUTO_*   │      │    generate new data? (strict, │
          │    entries, recurses      │      │    separate call - NOT just    │
          │    1 hop into them        │      │    "does it use data")         │
          │                           │      │                                 │
          │ data_fetch_agent.py       │      │ Both sub-tasks independently   │
          │  → what datasets did this │      │ gated/retried - one running out│
          │    paper use (own or      │      │ of quota doesn't block the     │
          │    reused)? creates/links │      │ other.                          │
          │    D_* entries            │      │                                 │
          └─────────────┬─────────────┘      └───────────────┬─────────────────┘
                        │                                      │
                        └──────────────────┬───────────────────┘
                                            ▼
                         staging.xlsx (every proposed change,
                         nothing touches the master Excel directly)
```

## 3. Manual review and merge

```
        staging.xlsx
             │
             ▼
   merge_candidates.py
   (applies staged changes into a NEW
    datasets_curated_autoreview_<date>.xlsx -
    never touches the current master)
             │
             ▼
   Marta reviews the autoreview file row by row
   (this is where judgment calls, corrections, and
    "actually this isn't self-generated data" catches happen -
    see notes.md / context/ANNOTATION_REVIEW_LOG.md)
             │
             ▼
   promoted: renamed to become the new master
   (old master archived to data_curated_backup/)
```

## 4. Export and publish

```
   new master .xlsx
        │
        ▼
   01_parse_excel.py   → cleans/normalizes, writes methods_*.csv / datasets_*.csv
        │
        ▼
   02_fetch_metadata.py → enriches with Crossref/PubMed (abstract, citations, year)
        │
        ▼
   03_export_json.py   → writes docs/data/{methods,datasets,stats}.json
        │
        ▼
   04_compute_embeddings.py (gaia only - needs sentence-transformers/umap-learn)
                            → docs/data/embeddings.json (positions the
                              Applications Graph)
        │
        ▼
   Marta commits + pushes (never done by an AI session - see CLAUDE.md)
        │
        ▼
   GitHub Pages serves docs/ directly (classic branch-deploy, no build step)
        │
        ▼
   marta-seq.github.io/solr/ - Methods tab/graph, Datasets tab, Applications
   tab/graph, Methods Book - all reading the same docs/data/*.json
```

## Where things actually live

```
data/data_curated/        the ONE current master .xlsx (source of truth)
data/data_curated_backup/ every superseded master, dated (the real version history -
                           the Excel itself is never committed to git)
data/processed/           CSV intermediates from 01/02 (also not committed)
data/agent_review/        staging.xlsx (pending) + archived merged staging files

src/preprocessing/        Phase 1 - the 01→02→03→04 export pipeline
src/agents/
  common/                 shared infra: LLM client + provider config, paper
                           fetcher/cache, staging writer, DB loader/indexes
  living_ingestion/
    mailroom/              triage.py - builds each phase's work queue
    literature_search/      the PubMed/bioRxiv/medRxiv scanner
    merge_candidates.py     staging.xlsx -> new autoreview file
  agent_curation/
    methods_desk/           category_audit_agent.py, compared_methods_agent.py
    data_desk/              data_fetch_agent.py, intern_agent.py
    application_desk/       application_review_agent.py
  run_pipeline.py          orchestrates all of the above, one shared
                           --max-papers budget across phases

docs/                     the live site (hand-built SPA, no framework/build step)
  data/                    methods.json / datasets.json / stats.json /
                           embeddings.json - what the frontend actually reads
  js/                      one file per tab/feature (graph.js, applications.js,
                           methods.js, book.js, ...)
```
