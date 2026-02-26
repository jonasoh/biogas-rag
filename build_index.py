"""
build_index.py – Embed handbook chunks and store them in a FAISS index.

Usage:
    python build_index.py \
        --chunks data/chunks.json \
        --outdir data \
        --model  Alibaba-NLP/gte-multilingual-base \
        --batch-size 32
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np


# MPS (Apple Silicon) triggers out-of-bounds errors with some
# trust_remote_code models (e.g. gte-multilingual-base), so only
# auto-select CUDA; fall back to CPU otherwise.
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"
except Exception:
    DEVICE = "cpu"


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Encode *texts* and return L2-normalised float32 vectors."""
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    print(f"Loading model '{model_name}' …")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=DEVICE)

    print(f"Embedding {len(texts)} chunks in batches of {batch_size} …")
    all_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vecs.append(vecs)

    matrix = np.vstack(all_vecs).astype("float32")

    # L2-normalise so that inner product == cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix /= norms

    return matrix


def build_faiss_index(vectors: np.ndarray):
    """Build a flat inner-product FAISS index from *vectors*."""
    import faiss

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a FAISS index from handbook chunks.")
    p.add_argument("--chunks", default="data/chunks.json")
    p.add_argument("--outdir", default="data")
    p.add_argument("--model", default="Alibaba-NLP/gte-multilingual-base")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    import faiss

    args = parse_args()

    print(f"Loading chunks from {args.chunks} …")
    chunks: List[Dict] = read_json(args.chunks)
    print(f"  {len(chunks)} chunks loaded")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, args.model, args.batch_size)
    print(f"Embedding shape: {vectors.shape}")

    index = build_faiss_index(vectors)
    print(f"FAISS index contains {index.ntotal} vectors (dim={vectors.shape[1]})")

    os.makedirs(args.outdir, exist_ok=True)

    index_path = os.path.join(args.outdir, "handbook.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index written to {index_path}")

    # Metadata file: all fields except 'text'
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    meta_path = os.path.join(args.outdir, "chunk_metadata.json")
    write_json(meta_path, metadata)
    print(f"Chunk metadata written to {meta_path}")


if __name__ == "__main__":
    main()
