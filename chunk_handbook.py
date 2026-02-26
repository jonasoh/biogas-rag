"""
chunk_handbook.py – Split the biogas handbook into overlapping text chunks
with structured metadata, respecting chapter boundaries.

Usage:
    python chunk_handbook.py \
        --input  raw_data/handbook_notables.json \
        --meta   metadata/handbook/handbook_chapters.json \
        --outdir data \
        --chunk-size 1000 \
        --overlap 150
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_BLANK = re.compile(r"\n{3,}")


def strip_html(text: str) -> str:
    """Remove HTML-like tags and collapse excessive blank lines."""
    text = _HTML_TAG.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Page collection
# ---------------------------------------------------------------------------


def collect_pages(
    pages_dict: Dict[str, str],
    page_start: int,
    page_end: int,
) -> List[Tuple[int, str]]:
    """Return [(page_num, cleaned_text), ...] for pages in [page_start, page_end]."""
    result = []
    for p in range(page_start, page_end + 1):
        raw = pages_dict.get(str(p), "")
        if not raw:
            continue
        cleaned = strip_html(raw)
        if cleaned:
            result.append((p, cleaned))
    return result


# ---------------------------------------------------------------------------
# Sliding-window chunker
# ---------------------------------------------------------------------------


def page_at_offset(page_breaks: List[Tuple[int, int]], offset: int) -> int:
    """Return the page number corresponding to *offset* in the concatenated text."""
    page = page_breaks[0][1]
    for pos, pnum in page_breaks:
        if pos <= offset:
            page = pnum
        else:
            break
    return page


def sliding_window_chunks(
    collected: List[Tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> List[Tuple[str, int, int]]:
    """
    Produce (text, page_start, page_end) tuples for a list of (page_num, text) pairs.
    Hard boundary: the entire input is treated as one unit (caller enforces chapter
    boundaries by only passing pages within a single chapter/segment).
    """
    if not collected:
        return []

    # Build concatenated text and track where each page starts.
    page_breaks: List[Tuple[int, int]] = []
    parts: List[str] = []
    cursor = 0
    for page_num, text in collected:
        page_breaks.append((cursor, page_num))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" separator

    full_text = "\n".join(p for _, p in collected)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    split_texts = splitter.split_text(full_text)

    chunks: List[Tuple[str, int, int]] = []
    search_start = 0

    for chunk_text in split_texts:
        # Find the exact start position of this chunk in the full text to map pages
        start = full_text.find(chunk_text, search_start)
        if start == -1:
            start = search_start

        end = start + len(chunk_text)

        p_start = page_at_offset(page_breaks, start)
        p_end = page_at_offset(page_breaks, end - 1)
        chunks.append((chunk_text, p_start, p_end))

        # Advance search_start, but keep it before the overlap region
        search_start = start + 1

    return chunks


# ---------------------------------------------------------------------------
# Section lookup
# ---------------------------------------------------------------------------


def build_page_to_section(sections: List[Dict]) -> Dict[int, str]:
    """
    Build a mapping page_num → section_title.
    Handles inverted page ranges by clamping page_end = max(page_start, page_end).
    """
    mapping: Dict[int, str] = {}
    for sec in sections:
        s_start = sec["page_start"]
        s_end = sec["page_end"]
        if s_end < s_start:
            print(
                f"  [warn] section '{sec['title']}': inverted range "
                f"{s_start}..{s_end}, clamping to {s_start}..{s_start}",
                file=sys.stderr,
            )
            s_end = s_start
        for p in range(s_start, s_end + 1):
            mapping[p] = sec["title"]
    return mapping


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------


def make_chunk_id(source: str, segment: str, idx: int) -> str:
    safe = re.sub(r"[^a-z0-9]", "_", segment.lower())
    return f"{source}_{safe}_{idx:04d}"


def chunks_for_segment(
    pages_dict: Dict[str, str],
    page_start: int,
    page_end: int,
    chunk_size: int,
    overlap: int,
    source: str,
    chapter_number: Optional[int],
    chapter_title: Optional[str],
    section_title: str,
    type_of_content: str,
    section_map: Optional[Dict[int, str]] = None,
    segment_id: str = "",
) -> List[Dict]:
    """Build chunks for a contiguous page range."""
    if page_end < page_start:
        print(
            f"  [warn] skipping inverted range {page_start}..{page_end} "
            f"for '{section_title}'",
            file=sys.stderr,
        )
        return []

    collected = collect_pages(pages_dict, page_start, page_end)
    raw_chunks = sliding_window_chunks(collected, chunk_size, overlap)

    result = []
    for idx, (text, p_start, p_end) in enumerate(raw_chunks):
        sec_label = section_title
        if section_map and p_start in section_map:
            sec_label = section_map[p_start]
        chunk_id = make_chunk_id(source, segment_id or section_title, idx)
        result.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "section_title": sec_label,
                "page_start": p_start,
                "page_end": p_end,
                "type_of_content": type_of_content,
                "text": text,
            }
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_chunks(
    pages_dict: Dict[str, str],
    meta: Dict,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    source = meta.get("source_id", "handbook")
    all_chunks: List[Dict] = []

    # ---- 1. Chapter text ------------------------------------------------
    chapters = meta.get("chapters", [])

    # Glossary starts at page 113 (Ordförklaringar). Cap all chapter page
    # ranges below this so back-matter is not double-counted.
    GLOSSARY_START = 113

    for ch in chapters:
        ch_num = ch["chapter_number"]
        ch_title = ch["chapter_title"]
        ch_start = ch["page_start"]
        ch_end = ch["page_end"]

        if ch_end < ch_start:
            print(
                f"[warn] chapter {ch_num} ('{ch_title}') has inverted range "
                f"{ch_start}..{ch_end}; will attempt sections individually",
                file=sys.stderr,
            )

        sections = ch.get("sections", [])
        section_map = build_page_to_section(sections)

        # Determine the actual usable page range for this chapter.
        # Use max page found in sections as fallback when chapter page_end is bad.
        if ch_end < ch_start:
            valid_pages = [max(s["page_start"], s["page_end"]) for s in sections]
            ch_end = max(valid_pages) if valid_pages else ch_start

        # Never extend into back-matter (glossary starts at GLOSSARY_START).
        ch_end = min(ch_end, GLOSSARY_START - 1)

        print(f"Chunking chapter {ch_num}: '{ch_title}' (pages {ch_start}–{ch_end})")
        segment_id = f"ch{ch_num}"
        ch_chunks = chunks_for_segment(
            pages_dict=pages_dict,
            page_start=ch_start,
            page_end=ch_end,
            chunk_size=chunk_size,
            overlap=overlap,
            source=source,
            chapter_number=ch_num,
            chapter_title=ch_title,
            section_title=sections[0]["title"] if sections else ch_title,
            type_of_content="text",
            section_map=section_map,
            segment_id=segment_id,
        )
        print(f"  → {len(ch_chunks)} chunks")
        all_chunks.extend(ch_chunks)

    # ---- 2. Glossary (Ordförklaringar) and References (Litteratur) -------
    # These appear as sections inside chapter 7 in the metadata but are
    # better treated as standalone back-matter.
    # Pages 113–116 = glossary, 117–138 = references (before exercises start).
    exercises_meta = meta.get("exercises", {})
    ex_start = exercises_meta.get("start_page", 139)

    glossary_start, glossary_end = 113, 116
    references_start, references_end = 117, ex_start - 1

    print(f"Chunking glossary (pages {glossary_start}–{glossary_end})")
    all_chunks.extend(
        chunks_for_segment(
            pages_dict=pages_dict,
            page_start=glossary_start,
            page_end=glossary_end,
            chunk_size=chunk_size,
            overlap=overlap,
            source=source,
            chapter_number=None,
            chapter_title=None,
            section_title="Ordförklaringar",
            type_of_content="glossary",
            segment_id="glossary",
        )
    )

    print(f"Chunking references (pages {references_start}–{references_end})")
    all_chunks.extend(
        chunks_for_segment(
            pages_dict=pages_dict,
            page_start=references_start,
            page_end=references_end,
            chunk_size=chunk_size,
            overlap=overlap,
            source=source,
            chapter_number=None,
            chapter_title=None,
            section_title="Litteratur",
            type_of_content="references",
            segment_id="references",
        )
    )

    # ---- 3. Exercises ---------------------------------------------------
    per_chapter_ex = exercises_meta.get("per_chapter", [])
    solutions_meta = exercises_meta.get("solutions", {})
    solutions_start_page = solutions_meta.get("start_page", 149)

    for i, ex in enumerate(per_chapter_ex):
        ch_num = ex["chapter_number"]
        ex_page = ex["page"]
        ch_title = next(
            (c["chapter_title"] for c in chapters if c["chapter_number"] == ch_num),
            None,
        )
        # Page range: from this exercise page to (next exercise page - 1) or
        # (solutions_start_page - 1).
        if i + 1 < len(per_chapter_ex):
            ex_page_end = per_chapter_ex[i + 1]["page"] - 1
        else:
            ex_page_end = solutions_start_page - 1

        print(
            f"Chunking exercises ch{ch_num} "
            f"(pages {ex_page}–{ex_page_end}): '{ex['title']}'"
        )
        all_chunks.extend(
            chunks_for_segment(
                pages_dict=pages_dict,
                page_start=ex_page,
                page_end=ex_page_end,
                chunk_size=chunk_size,
                overlap=overlap,
                source=source,
                chapter_number=ch_num,
                chapter_title=ch_title,
                section_title=ex["title"],
                type_of_content="exercise",
                segment_id=f"ex_ch{ch_num}",
            )
        )

    # ---- 4. Solutions ---------------------------------------------------
    per_chapter_sol = solutions_meta.get("per_chapter", [])
    # Estimate last page of the handbook from the highest page key in pages_dict.
    max_page = max(int(k) for k in pages_dict.keys())

    for i, sol in enumerate(per_chapter_sol):
        ch_num = sol["chapter_number"]
        sol_page = sol["page"]
        ch_title = next(
            (c["chapter_title"] for c in chapters if c["chapter_number"] == ch_num),
            None,
        )
        if i + 1 < len(per_chapter_sol):
            sol_page_end = per_chapter_sol[i + 1]["page"] - 1
        else:
            sol_page_end = max_page

        print(
            f"Chunking solutions ch{ch_num} "
            f"(pages {sol_page}–{sol_page_end}): '{sol['title']}'"
        )
        all_chunks.extend(
            chunks_for_segment(
                pages_dict=pages_dict,
                page_start=sol_page,
                page_end=sol_page_end,
                chunk_size=chunk_size,
                overlap=overlap,
                source=source,
                chapter_number=ch_num,
                chapter_title=ch_title,
                section_title=sol["title"],
                type_of_content="solution",
                segment_id=f"sol_ch{ch_num}",
            )
        )

    return all_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk the biogas handbook for RAG.")
    p.add_argument("--input", default="raw_data/handbook_notables.json")
    p.add_argument("--meta", default="metadata/handbook/handbook_chapters.json")
    p.add_argument("--outdir", default="data")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading pages from {args.input}")
    pages_dict: Dict[str, str] = read_json(args.input)
    print(f"  {len(pages_dict)} pages loaded")

    print(f"Loading metadata from {args.meta}")
    meta: Dict = read_json(args.meta)

    chunks = build_chunks(
        pages_dict=pages_dict,
        meta=meta,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    out_path = os.path.join(args.outdir, "chunks.json")
    os.makedirs(args.outdir, exist_ok=True)
    write_json(out_path, chunks)

    print(f"\nDone. {len(chunks)} chunks written to {out_path}")

    # Summary
    from collections import Counter

    type_counts = Counter(c["type_of_content"] for c in chunks)
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")


if __name__ == "__main__":
    main()
