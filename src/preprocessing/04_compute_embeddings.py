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

Usage:
    python src/preprocessing/04_compute_embeddings.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import umap

ROOT     = Path(__file__).resolve().parents[2]
OUT_DIR  = ROOT / "docs" / "data"
MODEL_NAME = "all-MiniLM-L6-v2"

# Final coordinate spread, chosen to be roughly comparable to the Methods
# graph's existing category-centroid canvas (docs/js/state.js notes that's
# "a ~1400x800 virtual canvas") - not load-bearing, cosmetic only, tune
# freely by re-running this script.
TARGET_SPREAD = 500

def main():
    methods = json.loads((OUT_DIR / "methods.json").read_text(encoding="utf-8"))
    entries = [m for m in methods if not m.get("is_placeholder") and m.get("title")]
    print(f"Embedding {len(entries)} papers ({sum(1 for e in entries if e['paper_type']=='application')} applications, "
          f"{sum(1 for e in entries if e['paper_type']=='method')} methods)...")

    texts = [f"{e['title']}. {e.get('abstract') or ''}".strip() for e in entries]

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

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
