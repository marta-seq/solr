"""
04_compute_embeddings.py
Computes a title(+abstract) embedding for every non-placeholder paper
(methods and applications) and reduces it to 2D via UMAP, for the
Applications Graph (docs/js/applications.js), which has no pipeline
category to anchor node positions with the way the Methods graph does.

Uses a small generic sentence-transformers model (all-MiniLM-L6-v2) rather
than SPECTER2 for now - SPECTER2 needs the `adapters` library and a task
adapter on top of the base model, more setup than justified for a first
pass. Swapping the model later is a one-line change (MODEL_NAME below); the
UMAP/export logic doesn't change.

Reads docs/data/methods.json (already has id/title/abstract/paper_type/
is_placeholder from 03_export_json.py) rather than the raw CSVs, since
that's already the assembled, cleaned text per entry.

Stable layout (2026-10-20): UMAP is global, so one changed abstract or one new paper moves EVERY point - the Applications Graph
would jump after each weekly update. If the env vars SOLR_PREV_EMBEDDINGS / SOLR_PREV_METHODS point at the previously published
embeddings.json / methods.json, papers whose id exists there with unchanged text KEEP their position; new or changed papers are
placed at the similarity-weighted mean position of their K nearest kept papers. A full UMAP refit happens only when more than
FULL_REFIT_FRAC of the papers are new/changed (or no previous files are given, or SOLR_FULL_EMBEDDINGS=1).

Usage:
    python src/preprocessing/04_compute_embeddings.py
"""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import umap

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = Path(os.environ.get("SOLR_OUT_DIR") or ROOT / "docs" / "data")   # override: weekly site build, 2026-10-19
MODEL_NAME = "all-MiniLM-L6-v2"

# Final coordinate spread, chosen to be roughly comparable to the Methods
# graph's existing category-centroid canvas (docs/js/state.js notes that's
# "a ~1400x800 virtual canvas") - not load-bearing, cosmetic only, tune
# freely by re-running this script.
TARGET_SPREAD = 500

PREV_EMB = os.environ.get("SOLR_PREV_EMBEDDINGS")
PREV_METHODS = os.environ.get("SOLR_PREV_METHODS")
FULL_REFIT_FRAC = 0.10    # refit everything if more than this share of papers is new/changed
K_NEIGHBORS = 5           # new/changed papers go to the weighted mean of their K nearest kept papers


def _text(e):
    return f"{e['title']}. {e.get('abstract') or ''}".strip()


def incremental_coords(entries, texts, vectors):
    """Coordinates reusing the previous layout, or None if a full refit is needed (see module docstring)."""
    if os.environ.get("SOLR_FULL_EMBEDDINGS") == "1" or not (PREV_EMB and PREV_METHODS):
        return None
    try:
        prev_emb = json.loads(Path(PREV_EMB).read_text(encoding="utf-8"))
        prev_rows = json.loads(Path(PREV_METHODS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    prev_text = {e["id"]: _text(e) for e in prev_rows if not e.get("is_placeholder") and e.get("title")}
    kept = [i for i, (e, tx) in enumerate(zip(entries, texts)) if e["id"] in prev_emb and prev_text.get(e["id"]) == tx]
    moved = [i for i in range(len(entries)) if i not in set(kept)]
    if len(kept) < 20 or len(moved) > FULL_REFIT_FRAC * len(entries):
        print(f"{len(moved)} of {len(entries)} papers new/changed -> full UMAP refit")
        return None
    coords = np.zeros((len(entries), 2))
    for i in kept:
        coords[i] = prev_emb[entries[i]["id"]]
    kept_vec, kept_xy = vectors[kept], coords[kept]
    for i in moved:
        sims = kept_vec @ vectors[i]                    # vectors are L2-normalised -> cosine similarity
        top = np.argsort(-sims)[:K_NEIGHBORS]
        w = np.clip(sims[top], 1e-6, None)
        coords[i] = (kept_xy[top] * w[:, None]).sum(axis=0) / w.sum()
    print(f"Stable layout: kept {len(kept)} positions, placed {len(moved)} new/changed papers near their nearest neighbours")
    return coords

def main():
    methods = json.loads((OUT_DIR / "methods.json").read_text(encoding="utf-8"))
    entries = [m for m in methods if not m.get("is_placeholder") and m.get("title")]
    print(f"Embedding {len(entries)} papers ({sum(1 for e in entries if e['paper_type']=='application')} applications, "
          f"{sum(1 for e in entries if e['paper_type']=='method')} methods)...")

    texts = [f"{e['title']}. {e.get('abstract') or ''}".strip() for e in entries]

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    coords = incremental_coords(entries, texts, vectors)
    if coords is None:
        print("Reducing to 2D via UMAP...")
        n_neighbors = min(15, len(entries) - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
        coords = reducer.fit_transform(vectors)

        # Standardize then scale to TARGET_SPREAD so the frontend doesn't need
        # its own arbitrary scale constant - coords are ready to plot directly.
        coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
        coords = coords * TARGET_SPREAD

    embeddings = {e["id"]: [round(float(x), 2), round(float(y), 2)] for e, (x, y) in zip(entries, coords)}

    out_path = OUT_DIR / "embeddings.json"
    out_path.write_text(json.dumps(embeddings, indent=2), encoding="utf-8", newline="\n")
    print(f"Wrote {out_path} ({len(embeddings)} entries)")

if __name__ == "__main__":
    main()
