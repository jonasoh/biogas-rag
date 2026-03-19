"""
query.py -- Interactive query utility for the handbook FAISS index.

Usage:
    # single query
    python query.py "vad är metanogener?"

    # interactive loop
    python query.py

    # options
    python query.py --top-k 8 --snippet 300 "toxiska ämnen i biogasprocessen"
"""

import argparse
import json
import os
import sys

# sentence_transformers must be imported before faiss on macOS: both link
# against Apple's Accelerate framework and faiss-first causes a segfault.
try:
    import torch

    # MPS causes a segfault with gte-multilingual-base; use CPU or CUDA only.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

from sentence_transformers import SentenceTransformer  # noqa: E402 (must precede faiss)

import faiss  # noqa: E402
import numpy as np

DEFAULT_INDEX = "data/handbook.faiss"
DEFAULT_META = "data/chunk_metadata.json"
DEFAULT_CHUNKS = "data/chunks.json"
DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
#DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

def load_assets(index_path, meta_path, chunks_path, model_name):
    print(f"Loading index …", end=" ", flush=True)
    index = faiss.read_index(index_path)
    print(f"{index.ntotal} vectors")

    print(f"Loading metadata …", end=" ", flush=True)
    meta = json.load(open(meta_path, encoding="utf-8"))
    texts = [c["text"] for c in json.load(open(chunks_path, encoding="utf-8"))]
    print(f"{len(meta)} chunks")

    print(f"Loading model '{model_name}' …", end=" ", flush=True)
    model = SentenceTransformer(model_name, trust_remote_code=True, device=DEVICE)
    print("ready\n")

    return index, meta, texts, model


def search(query: str, index, meta, texts, model, top_k: int, snippet: int):
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)

    scores, ids = index.search(vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
        m = meta[idx]
        text_preview = texts[idx][:snippet].replace("\n", " ")
        if len(texts[idx]) > snippet:
            text_preview += " …"
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": m["chunk_id"],
                "chapter": m["chapter_number"],
                "chapter_title": m["chapter_title"],
                "section": m["section_title"],
                "pages": f"{m['page_start']}–{m['page_end']}",
                "type": m["type_of_content"],
                "text": text_preview,
            }
        )
    return results


def print_results(results):
    for r in results:
        ch = f"Ch{r['chapter']} · " if r["chapter"] else ""
        print(
            f"  [{r['rank']}] score={r['score']:.4f}  {ch}{r['type']}  p.{r['pages']}"
        )
        if r["chapter_title"]:
            print(f"      {r['chapter_title']} › {r['section']}")
        else:
            print(f"      {r['section']}")
        print(f"      {r['text']}")
        print()


def parse_args():
    p = argparse.ArgumentParser(description="Query the handbook FAISS index.")
    p.add_argument("query", nargs="?", help="Query string (omit for interactive mode)")
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--snippet", type=int, default=200, help="Characters of text to show per result"
    )
    return p.parse_args()


def main():
    args = parse_args()

    for path in (args.index, args.meta, args.chunks):
        if not os.path.exists(path):
            print(
                f"Error: {path} not found. Run build_index.py first.", file=sys.stderr
            )
            sys.exit(1)

    index, meta, texts, model = load_assets(
        args.index, args.meta, args.chunks, args.model
    )

    if args.query:
        results = search(
            args.query, index, meta, texts, model, args.top_k, args.snippet
        )
        print(f'Results for: "{args.query}"\n')
        print_results(results)
        return

    # Interactive loop
    print(f"Top-{args.top_k} results per query. Empty line to quit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        results = search(query, index, meta, texts, model, args.top_k, args.snippet)
        print()
        print_results(results)


if __name__ == "__main__":
    main()
