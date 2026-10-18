"""
manual_papers.py
A drop-in folder for papers the automatic fetcher cannot reach (paywalls). Download the PDF however you can (institutional login in a
browser, Zotero, a colleague) and put it in data/manual_papers/ - ANY file name works:

    1. a DOI in the file name wins (10.1038_s41592-025-02826-9.pdf, 10.1038%2Fs41592...pdf), else
    2. the DOI printed in the document itself (first two PDF pages / first 4000 chars of a .txt/.md) - the most frequent one is used, because
       a first page mostly carries its OWN DOI (header/footer) and only occasionally cites others.

paper_fetcher.fetch_paper() checks this folder BEFORE any online source and overrides a cached "no text" / abstract-only result as soon as a
matching file appears (no waiting for the negative-cache expiry). Positive results are cached as source "manual_file".

Commands (from the project root):
    python -m src.agents.common.manual_papers needed   # writes data/manual_papers/NEEDED.html + NEEDED.csv: papers still without full text,
                                                       # with clickable DOI links, queue (method/application) and entry ids
    python -m src.agents.common.manual_papers status   # what the folder contains, which DOI each file was matched to, files with no DOI found

Folder override for tests: SOLR_MANUAL_PAPERS_DIR. The folder is data (not code): don't commit PDFs (publisher copyright) - add data/manual_papers/ to
your ignore list if the data/ directory is tracked.
"""

import csv
import html
import io
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

from . import config
from .doi_utils import normalize_doi

logging.getLogger("pdfminer").setLevel(logging.ERROR)    # pdfminer warns about harmless colour-space quirks on every page
MANUAL_DIR = Path(os.environ.get("SOLR_MANUAL_PAPERS_DIR", config.ROOT / "data" / "manual_papers"))
INDEX_FILE = MANUAL_DIR / ".index.json"
SUFFIXES = {".pdf", ".txt", ".md"}
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>\]\)]+")
_NAME_DOI_RE = re.compile(r"(10\.\d{4,9})[_/](.+)")


def _pdf_text(path: Path, max_pages=None) -> str:
    """PDF -> text. Order of preference (compared on 8 real journal PDFs, 2026-10-18): poppler's `pdftotext` (reads two-column layouts in order,
    keeps word spaces), PyMuPDF, then pdfplumber (worst: glued words, interleaved columns, broke data-availability detection on 5/8)."""
    import shutil
    import subprocess
    if shutil.which("pdftotext"):
        cmd = ["pdftotext"] + (["-l", str(max_pages)] if max_pages else []) + [str(path), "-"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
            if out.strip():
                return out
        except (subprocess.SubprocessError, OSError):
            pass
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            pages = doc if max_pages is None else [doc[i] for i in range(min(max_pages, len(doc)))]
            out = "\n".join(pg.get_text("text") for pg in pages)
        if out.strip():
            return out
    except Exception:
        pass
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("no PDF text extractor available: install poppler-utils (pdftotext), pymupdf or pdfplumber")
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in (pdf.pages if max_pages is None else pdf.pages[:max_pages]):
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def pdf_bytes_to_text(data: bytes) -> str:
    """Same extraction for PDF bytes downloaded online (Unpaywall, bioRxiv)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(data)
        tmp.flush()
        return _pdf_text(Path(tmp.name))


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _doi_key(doi: str) -> str:
    return normalize_doi(doi).lower()


def _doi_from_name(stem: str) -> str:
    m = _NAME_DOI_RE.search(stem.replace("%2F", "/").replace("%2f", "/"))
    return _doi_key(m.group(1) + "/" + m.group(2)) if m else ""


def _doi_from_content(path: Path) -> str:
    try:
        head = _pdf_text(path, max_pages=2) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:
        return ""
    found = [_doi_key(d) for d in _DOI_RE.findall(head)]
    found = [d for d in found if d]
    if not found:
        return ""
    best = Counter(found).most_common()
    top = best[0][1]
    return next(d for d in found if dict(best)[d] == top)      # most frequent, earliest on ties


def _load_index() -> dict:
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_index() -> dict:
    """{filename: {"mtime", "size", "doi", "by"}} for every supported file; only new/changed files are re-read."""
    if not MANUAL_DIR.exists():
        return {}
    old, new = _load_index(), {}
    for f in sorted(MANUAL_DIR.iterdir()):
        if not f.is_file() or f.suffix.lower() not in SUFFIXES or f.name.startswith(("~", ".")):
            continue
        st = f.stat()
        prev = old.get(f.name)
        if prev and prev.get("mtime") == st.st_mtime and prev.get("size") == st.st_size:
            new[f.name] = prev
            continue
        doi, by = _doi_from_name(f.stem), "file name"
        if not doi:
            doi, by = _doi_from_content(f), "document text"
        new[f.name] = {"mtime": st.st_mtime, "size": st.st_size, "doi": doi, "by": by if doi else ""}
    try:
        INDEX_FILE.write_text(json.dumps(new, indent=1), encoding="utf-8")
    except OSError:
        pass
    return new


_memo = {"stamp": None, "map": {}}


def _doi_map() -> dict:
    if not MANUAL_DIR.exists():
        return {}
    stamp = (MANUAL_DIR.stat().st_mtime, len(os.listdir(MANUAL_DIR)))
    if _memo["stamp"] != stamp:
        _memo["map"] = {v["doi"]: MANUAL_DIR / name for name, v in build_index().items() if v.get("doi")}
        _memo["stamp"] = stamp
    return _memo["map"]


def find(doi: str):
    """Path of the manually supplied file for this DOI, or None. Exact match first; else tolerant of trailing junk the PDF text glues onto
    a DOI (Oxford prints '.../aanag166/8778543', others '...-1761oconnovember7,2023'): the file's DOI may START WITH the wanted DOI,
    provided the wanted DOI is long enough (>= 18 chars) not to collide with a shorter, different DOI."""
    key = _doi_key(doi)
    if not key:
        return None
    m = _doi_map()
    if key in m:
        return m[key]
    if len(key) >= 18:
        for cand, path in m.items():
            if cand.startswith(key):
                return path
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────
def _status():
    idx = build_index()
    if not idx:
        print(f"{MANUAL_DIR} is empty or missing. Put PDFs (any file name) in it.")
        return
    unmatched = [n for n, v in idx.items() if not v.get("doi")]
    print(f"{len(idx)} file(s) in {MANUAL_DIR}, {len(idx) - len(unmatched)} matched to a DOI")
    for n, v in idx.items():
        print(f"  {'OK ' if v.get('doi') else '?? '} {n}  ->  {v.get('doi') or 'NO DOI FOUND (rename it to <doi with / as _>.pdf)'}"
              f"{'  [' + v['by'] + ']' if v.get('by') else ''}")


def _needed():
    from . import paper_fetcher
    from .db_loader import Database
    db = Database()
    rows = []
    for _, r in db.methods.iterrows():
        doi = normalize_doi(r.get("DOI", ""))
        if not doi or str(r.get("is_placeholder", "")).lower() == "true" or find(doi):
            continue
        cache = paper_fetcher._cache_path(doi)
        if not cache.exists():
            continue                                            # never attempted yet: the pipeline will try online first
        try:
            c = json.loads(cache.read_text())
        except ValueError:
            continue
        if c.get("source") == "none":
            have = "nothing"
        elif not c.get("is_full_text"):
            have = "abstract only"
        else:
            continue
        rows.append({"entry_id": r.get("entry_id", ""), "queue": str(r.get("paper_type", "")), "have": have, "doi": doi,
                     "title": str(r.get("title", "") or "")[:200], "journal": str(r.get("journal", "") or ""),
                     "file_name": doi.replace("https://doi.org/", "").replace("/", "_") + ".pdf"})
    rows.sort(key=lambda x: (x["queue"], x["have"], x["entry_id"]))
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_DIR / "NEEDED.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["entry_id", "queue", "have", "doi", "title", "journal", "file_name"])
        w.writeheader()
        w.writerows(rows)
    body = "".join(f"<tr><td>{html.escape(str(r['entry_id']))}</td><td>{html.escape(r['queue'])}</td><td>{r['have']}</td>"
                   f"<td><a href=\"{html.escape(r['doi'])}\" target=\"_blank\" rel=\"noopener\">{html.escape(r['doi'].replace('https://doi.org/', ''))}</a></td>"
                   f"<td>{html.escape(r['title'])}</td><td>{html.escape(r['journal'])}</td></tr>" for r in rows)
    (MANUAL_DIR / "NEEDED.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Papers needing a manual PDF</title>"
        "<style>body{font:14px sans-serif;margin:2em}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}</style>"
        f"<h2>{len(rows)} papers without full text</h2><p>Open each DOI while logged in via your institution, save the PDF into "
        f"<code>{html.escape(str(MANUAL_DIR))}</code> (any file name). Then re-run the pipeline.</p>"
        f"<table><tr><th>entry</th><th>queue</th><th>have</th><th>DOI</th><th>title</th><th>journal</th></tr>{body}</table>", encoding="utf-8")
    print(f"{len(rows)} papers need a manual PDF ({sum(r['have'] == 'nothing' for r in rows)} with nothing, "
          f"{sum(r['have'] == 'abstract only' for r in rows)} abstract only) -> {MANUAL_DIR / 'NEEDED.html'} and NEEDED.csv")
    by = Counter((r["queue"], r["have"]) for r in rows)
    for k, n in sorted(by.items()):
        print(f"   {k[0]:12s} {k[1]:14s} {n}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"needed": _needed, "status": _status}.get(cmd, lambda: print(__doc__))()
