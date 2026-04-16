"""
build_index.py – Embed all chunks (handbook + lectures) and store them in a FAISS index.

Loads data/chunks.json (handbook) and, if present, data/lecture_chunks.json (lectures),
merges them, and builds a single combined FAISS index.

Usage:
    python build_index.py \
        --chunks data/chunks.json \
        --lecture-chunks data/lecture_chunks.json \
        --outdir data \
        --batch-size 32
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import faiss

from config import DEVICE, EMBEDDING_MODEL


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


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build a flat inner-product FAISS index from *vectors*."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a FAISS index from all chunks.")
    p.add_argument("--chunks", default="data/chunks.json")
    p.add_argument("--lecture-chunks", default="data/lecture_chunks.json")
    p.add_argument("--outdir", default="data")
    p.add_argument("--model", default=EMBEDDING_MODEL)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    import faiss

    args = parse_args()

    print(f"Loading handbook chunks from {args.chunks} …")
    chunks: List[Dict] = read_json(args.chunks)
    print(f"  {len(chunks)} handbook chunks loaded")

    if os.path.exists(args.lecture_chunks):
        print(f"Loading lecture chunks from {args.lecture_chunks} …")
        lecture_chunks: List[Dict] = read_json(args.lecture_chunks)
        print(f"  {len(lecture_chunks)} lecture chunks loaded")
        chunks = chunks + lecture_chunks
    else:
        print(f"No lecture chunks found at {args.lecture_chunks}, skipping")

    print(f"Total chunks to index: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts, args.model, args.batch_size)
    print(f"Embedding shape: {vectors.shape}")

    index = build_faiss_index(vectors)
    print(f"FAISS index contains {index.ntotal} vectors (dim={vectors.shape[1]})")

    os.makedirs(args.outdir, exist_ok=True)

    index_path = os.path.join(args.outdir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"FAISS index written to {index_path}")

    # Combined chunks.json (used by app.py to load texts at query time)
    chunks_path = os.path.join(args.outdir, "chunks.json")
    write_json(chunks_path, chunks)
    print(f"Combined chunks written to {chunks_path}")

    # Metadata file: all fields except 'text'
    metadata = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    meta_path = os.path.join(args.outdir, "chunk_metadata.json")
    write_json(meta_path, metadata)
    print(f"Chunk metadata written to {meta_path}")


if __name__ == "__main__":
    main()
