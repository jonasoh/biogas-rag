"""
chunk_lectures.py – Parse WebVTT subtitle files and split into overlapping
text chunks with structured metadata, one lecture at a time.

Usage:
    python chunk_lectures.py \
        --subs-dir  raw_data/lecture_subs \
        --meta      raw_data/lecture_subs/lecture_metadata.csv \
        --outdir    data \
        --chunk-size 1500 \
        --overlap 150
"""

import argparse
import csv
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def slugify(name: str) -> str:
    """Convert a lecture title to a safe ASCII slug for use in chunk IDs."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return name.strip("_")


# ---------------------------------------------------------------------------
# VTT parsing
# ---------------------------------------------------------------------------

_TIMESTAMP_LINE_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}|\d{2}:\d{2})\.\d{3}\s*-->\s*(\d{1,2}:\d{2}:\d{2}|\d{2}:\d{2})\.\d{3}"
)
_INLINE_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_ts(ts: str) -> str:
    """Ensure timestamp is HH:MM:SS (add leading 00: if only MM:SS given)."""
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00:{parts[0]}:{parts[1]}"
    return ts


def parse_vtt(path: str) -> List[Tuple[str, str, str]]:
    """
    Parse a WebVTT file and return a list of (start_ts, end_ts, text) tuples.
    Timestamps are HH:MM:SS strings. Inline tags are stripped from cue text.
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    cues: List[Tuple[str, str, str]] = []
    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Find the timestamp line (may be preceded by a cue identifier)
        ts_line_idx = None
        for i, line in enumerate(lines):
            if _TIMESTAMP_LINE_RE.search(line):
                ts_line_idx = i
                break

        if ts_line_idx is None:
            continue

        m = _TIMESTAMP_LINE_RE.search(lines[ts_line_idx])
        start_ts = _normalize_ts(m.group(1))
        end_ts = _normalize_ts(m.group(2))

        text_lines = lines[ts_line_idx + 1 :]
        text = " ".join(
            _INLINE_TAG_RE.sub("", line).strip()
            for line in text_lines
            if line.strip()
        )

        if text:
            cues.append((start_ts, end_ts, text))

    return cues


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_lecture(
    cues: List[Tuple[str, str, str]],
    lecture_title: str,
    url: str,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """
    Split the cues of a single lecture into overlapping text chunks.
    Each chunk records the timestamp range it covers.
    """
    if not cues:
        return []

    # Build concatenated transcript and record cue start offsets for timestamp lookup.
    # full_text = cue_texts joined by "\n", so cue i starts at cursor_i.
    cue_positions: List[Tuple[int, str, str]] = []  # (char_start, ts_start, ts_end)
    parts: List[str] = []
    cursor = 0
    for ts_start, ts_end, text in cues:
        cue_positions.append((cursor, ts_start, ts_end))
        parts.append(text)
        cursor += len(text) + 1  # +1 for the "\n" separator

    full_text = "\n".join(parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    split_texts = splitter.split_text(full_text)

    def ts_for_pos(pos: int, use_end: bool = False) -> str:
        """Return the start or end timestamp of the cue that covers *pos*."""
        ts = cue_positions[0][2 if use_end else 1]
        for cue_start, cue_ts_start, cue_ts_end in cue_positions:
            if cue_start <= pos:
                ts = cue_ts_end if use_end else cue_ts_start
            else:
                break
        return ts

    slug = slugify(lecture_title)
    chunks: List[Dict] = []
    search_start = 0

    for i, chunk_text in enumerate(split_texts):
        start = full_text.find(chunk_text, search_start)
        if start == -1:
            start = search_start
        end = start + len(chunk_text) - 1

        chunks.append(
            {
                "chunk_id": f"lecture_{slug}_{i:04d}",
                "source": "lecture",
                "lecture_title": lecture_title,
                "url": url,
                "timestamp_start": ts_for_pos(start, use_end=False),
                "timestamp_end": ts_for_pos(end, use_end=True),
                "type_of_content": "lecture_transcript",
                "text": chunk_text,
            }
        )

        search_start = start + 1

    return chunks


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_lecture_metadata(meta_path: str) -> Dict[str, str]:
    """Load lecture_metadata.csv → {lecture_title: url}."""
    result: Dict[str, str] = {}
    with open(meta_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            result[row["lecture"].strip()] = row["url"].strip()
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk WebVTT lecture subtitles.")
    p.add_argument("--subs-dir", default="raw_data/lecture_subs")
    p.add_argument("--meta", default="raw_data/lecture_subs/lecture_metadata.csv")
    p.add_argument("--outdir", default="data")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=160)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading lecture metadata from {args.meta} …")
    meta_by_title = load_lecture_metadata(args.meta)
    print(f"  {len(meta_by_title)} lectures in metadata")

    vtt_files = sorted(
        f for f in os.listdir(args.subs_dir) if f.lower().endswith(".vtt")
    )
    print(f"Found {len(vtt_files)} VTT files in {args.subs_dir}")

    all_chunks: List[Dict] = []

    for vtt_file in vtt_files:
        lecture_title = os.path.splitext(vtt_file)[0]
        url = meta_by_title.get(lecture_title, "")
        if not url:
            print(f"  WARNING: No metadata found for '{lecture_title}', URL will be empty")

        vtt_path = os.path.join(args.subs_dir, vtt_file)
        cues = parse_vtt(vtt_path)
        print(f"  {vtt_file}: {len(cues)} cues")

        chunks = chunk_lecture(cues, lecture_title, url, args.chunk_size, args.overlap)
        print(f"    → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal lecture chunks: {len(all_chunks)}")

    out_path = os.path.join(args.outdir, "lecture_chunks.json")
    write_json(out_path, all_chunks)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
