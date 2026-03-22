import os
import json
import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

_ROOT = os.path.join(os.path.dirname(__file__), "..")
VECTOR_DB_DIR = os.path.join(_ROOT, "vector-db")
INDEX_FILE = os.path.join(VECTOR_DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singletons — loaded once on first search call
_index = None
_metadata = None
_model = None


def _load():
    global _index, _metadata, _model
    if _index is None:
        _index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, encoding="utf-8") as f:
            _metadata = json.load(f)
        _model = SentenceTransformer(MODEL_NAME, token=False)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search vector-db for the top_k most similar notes to query.

    Returns a list of dicts: [{filename, content, score}, ...]
    """
    _load()
    vec = _model.encode(query, show_progress_bar=False).astype("float32").reshape(1, -1)
    distances, indices = _index.search(vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        record = _metadata[idx]
        results.append({"filename": record["filename"], "content": record["content"], "score": float(dist)})
    return results
