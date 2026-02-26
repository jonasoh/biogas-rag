"""
test_rag.py – Simple RAG pipeline for the biogas handbook.

Retrieves relevant chunks from the FAISS index, then sends them as
context to a local Ollama instance (llama3.1) to generate an answer.

Usage:
    python test_rag.py "vad är metanogener?"
    python test_rag.py --debug "toxiska ämnen"
    python test_rag.py                        # interactive mode
"""

import argparse
import json
import os
import sys

import httpx
import numpy as np

from query import (
    DEFAULT_CHUNKS,
    DEFAULT_INDEX,
    DEFAULT_META,
    DEFAULT_MODEL,
    load_assets,
)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LLM_MODEL = "llama3.1:latest"
TOP_K = 5

SYSTEM_PROMPT = """\
Du är en kunnig handledare i biogasteknik. Du svarar på svenska.

Regler:
- Svara ENBART utifrån den kontext som ges nedan.
- Om kontexten inte innehåller tillräcklig information för att besvara frågan, \
säg tydligt att du inte kan besvara frågan med den tillgängliga informationen.
- Hänvisa till kapitel- och sidnummer när det är möjligt.
- Var koncis och saklig."""

USER_TEMPLATE = """\
### Kontext
{context}

### Fråga
{question}"""


def retrieve(query: str, index, meta, texts, model, top_k: int):
    """Search the FAISS index and return results with full chunk text."""
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)

    scores, ids = index.search(vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
        m = meta[idx]
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
                "text": texts[idx],  # full text, no truncation
            }
        )
    return results


def build_context(results: list[dict]) -> str:
    """Format search results into a context string for the LLM."""
    parts = []
    for r in results:
        header = f"[Kap. {r['chapter']}, s. {r['pages']}] {r['chapter_title']}"
        if r["section"]:
            header += f" › {r['section']}"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def ask_llm(question: str, context: str, debug: bool = False) -> str:
    """Send the question + context to Ollama and return the response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, question=question),
        },
    ]

    if debug:
        print("=" * 72)
        print("DEBUG: MESSAGES SENT TO LLM")
        print("=" * 72)
        for msg in messages:
            print(f"\n--- [{msg['role'].upper()}] ---")
            print(msg["content"])
        print("=" * 72)
        print()

    resp = httpx.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def rag_query(question: str, index, meta, texts, model, debug: bool = False) -> str:
    """Full RAG pipeline: retrieve then generate."""
    results = retrieve(question, index, meta, texts, model, TOP_K)

    print(f"\n--- Hämtade {len(results)} relevanta avsnitt ---")
    for r in results:
        ch = f"Kap. {r['chapter']}" if r["chapter"] else ""
        print(f"  [{r['rank']}] {ch} s. {r['pages']}  (score={r['score']:.4f})")
    print()

    context = build_context(results)
    return ask_llm(question, context, debug=debug)


def parse_args():
    p = argparse.ArgumentParser(description="RAG pipeline for the biogas handbook.")
    p.add_argument("query", nargs="*", help="Query string (omit for interactive mode)")
    p.add_argument(
        "--debug", action="store_true", help="Print full LLM prompt to console"
    )
    return p.parse_args()


def main():
    args = parse_args()

    for path in (DEFAULT_INDEX, DEFAULT_META, DEFAULT_CHUNKS):
        if not os.path.exists(path):
            print(
                f"Error: {path} not found. Run build_index.py first.", file=sys.stderr
            )
            sys.exit(1)

    index, meta, texts, model = load_assets(
        DEFAULT_INDEX, DEFAULT_META, DEFAULT_CHUNKS, DEFAULT_MODEL
    )
    # load_assets returns snippet-truncated texts; reload full texts
    texts = [c["text"] for c in json.load(open(DEFAULT_CHUNKS, encoding="utf-8"))]

    if args.query:
        question = " ".join(args.query)
        answer = rag_query(question, index, meta, texts, model, debug=args.debug)
        print(answer)
        return

    # Interactive loop
    print("Ställ en fråga om biogashandboken. Tom rad för att avsluta.\n")
    while True:
        try:
            question = input("Fråga> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer = rag_query(question, index, meta, texts, model, debug=args.debug)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
