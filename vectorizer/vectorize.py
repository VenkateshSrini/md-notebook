import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "..", "source-md")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vector-db")
INDEX_FILE = os.path.join(VECTOR_DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"


def _load_md_files():
    texts, records = [], []
    all_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".md"))
    for filename in all_files:
        path = os.path.join(SOURCE_DIR, filename)
        content = open(path, encoding="utf-8").read().strip()
        if not content:
            print(f"  [SKIP] Empty file: {filename}")
            continue
        texts.append(content)
        records.append({"filename": filename, "content": content})
    return texts, records


def run():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, token=False)

    texts, records = _load_md_files()
    total = len(records)

    print(f"Embedding {total} files...")
    embeddings = []
    for i, text in enumerate(texts, start=1):
        embedding = model.encode(text, show_progress_bar=False)
        embeddings.append(embedding)
        print(f"\rVectorizing... [{i}/{total}]", end="", flush=True)

    print(f"\nBuilding FAISS index...")
    matrix = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Done. {total} files vectorized.")
