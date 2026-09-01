"""
rag.py -- RAG pipeline for the biogas handbook.

Retrieves relevant chunks from the FAISS index, then sends them as
context to a local LLM to generate an answer.

Usage:
    python rag.py "vad är metanogener?"
    python rag.py --debug "toxiska ämnen"
    python rag.py                        # interactive mode
"""

import argparse
import json
import os
import sys

import httpx
import numpy as np

from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    INDEX_PATH,
    LLM_MODEL,
    LLM_TIMEOUT,
    LLM_URL,
    METADATA_PATH,
    TOP_K,
)
from query import load_assets

SYSTEM_PROMPT = """\
Du är en assistent som hjälper studenter att hitta information i kursmaterialet för kursen \
Driftoptimering av lantbruksbaserad biogasproduktion vid Sveriges Lantbruksuniversitet (SLU). \
Du svarar på svenska.

Kursmaterialet består av en handbok och föreläsningar. \
Dina svar visas tillsammans med relevanta avsnitt ur kursmaterialet, inklusive bilder och tabeller. \
Kursmaterialet finns också tillgängligt på [kursens hemsida i Canvas](https://canvas.slu.se/courses/15835).

För beräkning av driftparametrar (HRT, OLR, TS, VS, utrötningsgrad, \
HRT), kväve i processen (TAN, FAN, kvävemineralisering) och \
processparametrar (SMP, volymetrisk metanproduktion) finns också \
ett beräkningsverktyg tillgängligt på \
[denna länk](https://biogas.molsci.slu.se/calc/). Hänvisa till detta om \
användaren frågar om beräkning av dessa parametrar.

Kursmaterialet innehåller övningsfrågor (kapitel 1–7) och tillhörande svar. \
Dessa kan ingå i kontexten nedan, markerade som [Övning, Kap. N] respektive \
[Svar på övningar, Kap. N]. Om övningar eller svar finns med i kontexten, \
hänvisa till dem med sidnummer, t.ex. [s. 140](handbok:140). \
Om användaren frågar om en beräkning eller ett begrepp som övas i kursen, \
hänvisa till relevanta övningsfrågor.

Du svarar ENBART utifrån den kontext som ges nedan. Kort och koncist. Om kontexten inte \
innehåller tillräcklig information för att besvara frågan, \
säg att du inte kan besvara frågan med den tillgängliga informationen.

Dina svar kommer att formateras i en webapp där användaren också kan se kursmaterialet. \
Det är därför mycket viktigt att du svarar exakt enligt de riktlinjer som anges nedan.

# Riktlinjer för formatering

- Istället för 'kontext', använd 'handboken', 'biogashandboken' eller 'föreläsningen' \
beroende på källan.

## Hänvisning till handboken (source: 'handbook')

- Du anger alltid källhänvisningar när det är möjligt och de måste ALLTID vara korrekt formaterade \
för att användaren ska kunna hitta informationen.
- När du hänvisar till sidor i handboken, använd ALLTID formatet \
[s. N](handbok:N) för en enskild sida eller [s. N–M](handbok:N-M) för ett sidintervall. \
Formatera i löpande text, t.ex. 'beskrivs i handboken på [s. 45–52](handbok:45-52)'.
- Skriv inte ut sidnummer utan hänvisningar.
- Skriv inte ut hela tabeller utan hänvisa till dem i handboken.

## Hänvisning till föreläsningar (source: 'lecture')

- När du hänvisar till en föreläsning MÅSTE du ALLTID använda formatet '[Föreläsning: Föreläsningsnamn (UTAN TIDSSTÄMPEL)](url)'.
- Skriv ALDRIG ut tidsstämpeln för föreläsningar. Hänvisa bara till föreläsningen.
- Hänvisa inte till samma föreläsning flera gånger i samma svar.

## Hänvisning till kursinformation (source: 'course_info')

- När du hänvisar till kursinformation räcker det att skriva 'enligt kursinformationen' eller 'i kursen'.
- Du behöver inte ange sidnummer eller tidsstämpel.

## Hänvisning till ordlista (source: 'glossary')

- Skriv bara svaret direkt utan att referera till ordlistan.

## Hänvisning till övningar och svar (type: 'exercise' / 'solution')

- Hänvisa till övningsfrågor med sidnummer i handboken, t.ex. 'övningsfrågorna på [s. 140](handbok:140)'.
- Hänvisa till svar på övningar på samma sätt, t.ex. 'svaren finns på [s. 150](handbok:150)'.
- Ange aldrig svar på övningsfrågor ordagrant — hänvisa istället till sidorna i handboken.
"""

USER_TEMPLATE = """\
### Kontext
{context}

### Fråga
{question}"""


def retrieve(query: str, index, meta, texts, model, top_k: int):
    """Search the vector index and return results with full chunk text."""
    vec = model.encode([query], convert_to_numpy=True).astype("float32")
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)

    all_scores = (vec @ index.T)[0]
    ids = np.argsort(all_scores)[::-1][:top_k]
    scores = all_scores[ids]

    results = []
    for rank, (idx, score) in enumerate(zip(ids, scores), 1):
        m = meta[idx]
        source = m.get("source", "handbook")
        if source == "lecture":
            result = {
                "rank": rank,
                "score": float(score),
                "chunk_id": m["chunk_id"],
                "source": "lecture",
                "lecture_title": m["lecture_title"],
                "url": m.get("url", ""),
                "timestamp_start": m.get("timestamp_start", ""),
                "timestamp_end": m.get("timestamp_end", ""),
                "type": m["type_of_content"],
                "text": texts[idx],
                "table_markdown": None,
                "table_legend": None,
            }
        elif source == "course_info":
            result = {
                "rank": rank,
                "score": float(score),
                "chunk_id": m["chunk_id"],
                "source": "course_info",
                "section_title": m.get("section_title", ""),
                "type": m["type_of_content"],
                "text": texts[idx],
                "table_markdown": None,
                "table_legend": None,
            }
        else:
            result = {
                "rank": rank,
                "score": float(score),
                "chunk_id": m["chunk_id"],
                "source": "handbook",
                "chapter": m["chapter_number"],
                "chapter_title": m["chapter_title"],
                "section": m["section_title"],
                "pages": f"{m['page_start']}–{m['page_end']}",
                "type": m["type_of_content"],
                "text": texts[idx],
                "table_markdown": m.get("table_markdown"),
                "table_legend": m.get("table_legend"),
            }
        results.append(result)
    return results


def _make_injected_chunk(m: dict, texts: list[str], idx: int) -> dict:
    return {
        "rank": 0,
        "score": -1.0,
        "chunk_id": m["chunk_id"],
        "source": "handbook",
        "chapter": m["chapter_number"],
        "chapter_title": m["chapter_title"],
        "section": m["section_title"],
        "pages": f"{m['page_start']}–{m['page_end']}",
        "type": m["type_of_content"],
        "text": texts[idx],
        "table_markdown": m.get("table_markdown"),
        "table_legend": m.get("table_legend"),
        "injected": True,
    }


def enrich_with_exercises(
    results: list[dict],
    meta: list[dict],
    texts: list[str],
) -> list[dict]:
    """Ensure exercise and solution chunks always appear together.

    - When a solution chunk is retrieved, inject the matching exercise(s) before it.
    - When an exercise chunk is retrieved, inject the matching solution(s) after it.
    """
    existing_ids = {r["chunk_id"] for r in results}
    solution_chapters = {r["chapter"] for r in results if r["type"] == "solution"}
    exercise_chapters = {r["chapter"] for r in results if r["type"] == "exercise"}

    if not solution_chapters and not exercise_chapters:
        return results

    # Exercises to inject before matching solutions (keyed by chapter)
    exercises_by_chapter: dict[int, list[dict]] = {}
    # Solutions to inject after matching exercises (keyed by chapter)
    solutions_by_chapter: dict[int, list[dict]] = {}

    for idx, m in enumerate(meta):
        ch = m.get("chapter_number")
        ctype = m["type_of_content"]
        if ch is None or m["chunk_id"] in existing_ids:
            continue
        if ctype == "exercise" and ch in solution_chapters:
            exercises_by_chapter.setdefault(ch, []).append(
                _make_injected_chunk(m, texts, idx)
            )
        elif ctype == "solution" and ch in exercise_chapters:
            solutions_by_chapter.setdefault(ch, []).append(
                _make_injected_chunk(m, texts, idx)
            )

    if not exercises_by_chapter and not solutions_by_chapter:
        return results

    enriched = []
    injected_ex_chapters: set[int] = set()
    injected_sol_chapters: set[int] = set()
    for r in results:
        # Inject exercises before the first solution of their chapter
        if r["type"] == "solution" and r["chapter"] not in injected_ex_chapters:
            enriched.extend(exercises_by_chapter.get(r["chapter"], []))
            injected_ex_chapters.add(r["chapter"])
        enriched.append(r)
        # Inject solutions after the first exercise of their chapter
        if r["type"] == "exercise" and r["chapter"] not in injected_sol_chapters:
            enriched.extend(solutions_by_chapter.get(r["chapter"], []))
            injected_sol_chapters.add(r["chapter"])

    return enriched


def build_context(results: list[dict]) -> str:
    """Format search results into a context string for the LLM."""
    parts = []
    for r in results:
        if r.get("source") == "lecture":
            ts = r.get("timestamp_start", "")
            url = r.get("url", "")
            title = r.get("lecture_title", "")
            if url:
                header = f"[Föreläsning: {title}, {ts}]({url})"
            else:
                header = f"Föreläsning: {title}, {ts}"
            content = r["text"]
        elif r.get("source") == "course_info":
            header = f"[Kursinformation: {r.get('section_title', '')}]"
            content = r["text"]
        else:
            type_ = r.get("type", "")
            if type_ == "exercise":
                header = f"[Övning, Kap. {r['chapter']}, s. {r['pages']}] {r['chapter_title']}"
            elif type_ == "solution":
                header = f"[Svar på övningar, Kap. {r['chapter']}, s. {r['pages']}] {r['chapter_title']}"
            else:
                header = f"[Kap. {r['chapter']}, s. {r['pages']}] {r['chapter_title']}"
                if r["section"]:
                    header += f" › {r['section']}"

            if r.get("table_markdown"):
                # For table chunks: show the actual table, not the summary
                legend = r.get("table_legend", "")
                content = f"{legend}\n\n{r['table_markdown']}"
            else:
                content = r["text"]

        parts.append(f"{header}\n{content}")
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
        LLM_URL,
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]["message"]["content"]


def rag_query(question: str, index, meta, texts, model, debug: bool = False) -> str:
    """Full RAG pipeline: retrieve then generate."""
    results = retrieve(question, index, meta, texts, model, TOP_K)
    results = enrich_with_exercises(results, meta, texts)

    print(f"\n--- Hämtade {len(results)} relevanta avsnitt ---")
    for r in results:
        src = r.get("source", "handbook")
        if src == "lecture":
            loc = r.get("lecture_title", "")
        elif src == "course_info":
            loc = r.get("section_title", "course_info")
        else:
            ch = f"Kap. {r.get('chapter', '')} " if r.get("chapter") else ""
            loc = f"{ch}s. {r.get('pages', '')}"
        print(f"  [{r['rank']}] {src} · {loc}  (score={r['score']:.4f})")
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

    for path in (INDEX_PATH, METADATA_PATH, CHUNKS_PATH):
        if not os.path.exists(path):
            print(
                f"Error: {path} not found. Run build_index.py first.", file=sys.stderr
            )
            sys.exit(1)

    index, meta, texts, model = load_assets(
        INDEX_PATH, METADATA_PATH, CHUNKS_PATH, EMBEDDING_MODEL
    )
    # load_assets returns snippet-truncated texts; reload full texts
    texts = [c["text"] for c in json.load(open(CHUNKS_PATH, encoding="utf-8"))]

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
