#!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TocEntry:
    title: str
    page: int


@dataclass
class Chapter:
    chapter_number: int
    chapter_title: str
    page_start: int
    toc_source_pages: List[int]
    entries: List[TocEntry]
    page_end: Optional[int] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def extract_pages(book: Dict[str, Any], pages: List[int]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pages:
        key = str(p)
        if key not in book:
            raise KeyError(f"missing page {p} in input json")
        out[key] = book[key]
    return out


def strip_simple_html(text: str) -> str:
    # minimal removal for tags like <heading>...</heading>
    text = re.sub(r"</?heading>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)
    return text


def normalize_toc_text(raw: str) -> str:
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = strip_simple_html(t)
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


_chapter_header_re = re.compile(r"^\*\*(\d+)\.\s*(.+?)\s+(\d+)\*\*$")

_line_with_page_re = re.compile(r"^(.*?)\s+(\d+)$")

SPECIAL_CHAPTER_TITLES = {"Ordförklaringar", "Litteratur"}


def parse_toc_lines(
    normalized_text: str,
    source_page_numbers: List[int],
) -> Tuple[List[Chapter], List[TocEntry], Dict[str, Any]]:
    lines = [ln.strip() for ln in normalized_text.split("\n")]

    chapters: List[Chapter] = []
    back_matter: List[TocEntry] = []
    exercises_bucket: Dict[str, Any] = {
        "start_page": None,
        "per_chapter": [],
        "solutions": {
            "start_page": None,
            "per_chapter": [],
        },
    }

    current: Optional[Chapter] = None
    current_source_pages: List[int] = []
    last_chapter_number = 0

    def finalize_current() -> None:
        nonlocal current
        if current is not None:
            chapters.append(current)
            current = None

    # heuristic buckets
    in_exercises = False
    in_solutions = False

    for ln in lines:
        if not ln:
            continue

        m_ch = _chapter_header_re.match(ln)
        if m_ch:
            finalize_current()
            ch_num = int(m_ch.group(1))
            ch_title = m_ch.group(2).strip()
            ch_page = int(m_ch.group(3))
            current = Chapter(
                chapter_number=ch_num,
                chapter_title=ch_title,
                page_start=ch_page,
                toc_source_pages=list(source_page_numbers),
                entries=[],
            )
            last_chapter_number = ch_num
            in_exercises = False
            in_solutions = False
            continue

        m_lp = _line_with_page_re.match(ln)
        if m_lp:
            title = m_lp.group(1).strip()
            page = int(m_lp.group(2))

            low = title.lower()

            # bucket switches
            if title == "Övningsuppgifter":
                in_exercises = True
                in_solutions = False
                exercises_bucket["start_page"] = page
                continue

            if title == "Svar på övningsuppgifterna":
                in_solutions = True
                in_exercises = False
                exercises_bucket["solutions"]["start_page"] = page
                continue

            if (
                not in_exercises
                and not in_solutions
                and title in SPECIAL_CHAPTER_TITLES
            ):
                finalize_current()
                last_chapter_number += 1
                current = Chapter(
                    chapter_number=last_chapter_number,
                    chapter_title=title,
                    page_start=page,
                    toc_source_pages=list(source_page_numbers),
                    entries=[],
                )
                continue

            if in_exercises:
                m_ex_ch = re.match(r"^Övningsuppgifter Kapitel (\d+)$", title)
                if m_ex_ch:
                    exercises_bucket["per_chapter"].append(
                        {
                            "chapter_number": int(m_ex_ch.group(1)),
                            "title": title,
                            "page": page,
                        }
                    )
                else:
                    # keep non-chapter exercise lines as generic entries
                    exercises_bucket["per_chapter"].append(
                        {"chapter_number": None, "title": title, "page": page}
                    )
                continue

            if in_solutions:
                m_sol_ch = re.match(
                    r"^Svar på övningsuppgifterna till Kapitel (\d+)$", title
                )
                if m_sol_ch:
                    exercises_bucket["solutions"]["per_chapter"].append(
                        {
                            "chapter_number": int(m_sol_ch.group(1)),
                            "title": title,
                            "page": page,
                        }
                    )
                else:
                    exercises_bucket["solutions"]["per_chapter"].append(
                        {"chapter_number": None, "title": title, "page": page}
                    )
                continue

            # normal toc lines
            if current is not None:
                current.entries.append(TocEntry(title=title, page=page))
            else:
                back_matter.append(TocEntry(title=title, page=page))

            continue

        # if a line doesn't end with a page number, ignore it for now
        # you can tighten this later if needed

    finalize_current()

    debug = {
        "line_count": len(lines),
        "chapter_count": len(chapters),
        "back_matter_count": len(back_matter),
        "exercises_seen": exercises_bucket["start_page"] is not None,
        "solutions_seen": exercises_bucket["solutions"]["start_page"] is not None,
    }

    return chapters, back_matter, {"exercises": exercises_bucket, "debug": debug}


def to_serializable_chapters(chapters: List[Chapter]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ch in chapters:
        d = asdict(ch)
        d["entries"] = [asdict(e) for e in ch.entries]
        out.append(d)
    return out


def to_serializable_entries(entries: List[TocEntry]) -> List[Dict[str, Any]]:
    return [asdict(e) for e in entries]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="raw_data/handbook.json")
    ap.add_argument("--outdir", default="metadata/handbook")
    ap.add_argument("--source-id", default="handbook")
    ap.add_argument("--title", default="handbook")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    book = read_json(args.input)

    toc_pages = [4, 5]
    extracted = extract_pages(book, toc_pages)
    write_json(os.path.join(args.outdir, "00_extracted_pages_4_5.json"), extracted)

    raw_concat = "\n\n".join([extracted[str(p)] for p in toc_pages])
    write_text(os.path.join(args.outdir, "01_toc_raw.txt"), raw_concat)

    normalized = normalize_toc_text(raw_concat)
    write_text(os.path.join(args.outdir, "02_toc_normalized.txt"), normalized)

    chapters, back_matter, extra = parse_toc_lines(normalized, toc_pages)

    # Calculate page_end for each chapter and section
    for i, ch in enumerate(chapters):
        if i + 1 < len(chapters):
            ch.page_end = chapters[i + 1].page_start - 1
        else:
            # Last chapter ends at first back_matter page or end of document
            if back_matter:
                ch.page_end = back_matter[0].page - 1
            else:
                ch.page_end = None

    parsed = {
        "source_id": args.source_id,
        "toc_pages": toc_pages,
        "chapters": to_serializable_chapters(chapters),
        "back_matter": to_serializable_entries(back_matter),
        **extra,
    }
    write_json(os.path.join(args.outdir, "03_toc_parsed.json"), parsed)

    # minimal chapter metadata file to drive later chunking
    chapter_metadata = {
        "source_id": args.source_id,
        "source_type": "textbook",
        "title": args.title,
        "toc_pages": toc_pages,
        "chapters": [
            {
                "chapter_number": ch.chapter_number,
                "chapter_title": ch.chapter_title,
                "page_start": ch.page_start,
                "page_end": ch.page_end,
                "sections": [
                    {
                        "title": e.title,
                        "page_start": e.page,
                        "page_end": (
                            ch.entries[ch.entries.index(e) + 1].page - 1
                            if ch.entries.index(e) + 1 < len(ch.entries)
                            else ch.page_end
                        ),
                    }
                    for e in ch.entries
                ],
            }
            for ch in chapters
        ],
        "back_matter": to_serializable_entries(back_matter),
        "exercises": extra["exercises"],
    }
    write_json(os.path.join(args.outdir, "handbook_chapters.json"), chapter_metadata)


if __name__ == "__main__":
    main()
