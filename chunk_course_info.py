"""
chunk_course_info.py – Build course info chunks from CSV sources.

Reads raw_data/course_overview.csv (lecture list) and raw_data/Ordlista.csv
(terminology grouped by theme) and writes data/course_info_chunks.json.

Produces 5 chunks:
  course_info_0000  course_overview     – list of all lectures with URLs
  course_info_0001  terminology         – biokemi och mikrobiologi
  course_info_0002  terminology         – processparametrar och mätvärden
  course_info_0003  terminology         – anläggningstyper och processer
  course_info_0004  terminology         – substrat, produkter och certifiering

Usage:
    python chunk_course_info.py \
        --overview  raw_data/course_overview.csv \
        --ordlista  raw_data/Ordlista.csv \
        --outdir    data
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


GRUPP_TITLES = {
    "biokemi":    "Begrepp: Biokemi och mikrobiologi",
    "parametrar": "Begrepp: Processparametrar och mätvärden",
    "anlaggning": "Begrepp: Anläggningstyper och processer",
    "substrat":   "Begrepp: Substrat, produkter och certifiering",
}

GRUPP_ORDER = ["biokemi", "parametrar", "anlaggning", "substrat"]


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def build_overview_chunk(overview_path: str) -> Dict:
    lines = ["Kursen 'Driftoptimering av lantbruksbaserad biogasproduktion' på       SLU innehåller följande föreläsningar:"]
    with open(overview_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            nummer = row["nummer"].strip()
            titel = row["titel"].strip()
            url = row["url"].strip()
            beskrivning = row.get("beskrivning", "").strip()
            entry = f"{nummer}. {titel} ({url})"
            if beskrivning:
                entry += f" – {beskrivning}"
            lines.append(entry)

    return {
        "chunk_id": "course_info_0000",
        "source": "course_info",
        "type_of_content": "course_overview",
        "section_title": "Kursöversikt",
        "text": "\n".join(lines),
    }


def build_terminology_chunks(ordlista_path: str) -> List[Dict]:
    groups: dict[str, list[str]] = defaultdict(list)

    with open(ordlista_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            grupp = row["grupp"].strip()
            term = row["term"].strip()
            definition = row["definition"].strip()
            if grupp and term:
                groups[grupp].append(f"{term} – {definition}")

    chunks = []
    for i, grupp in enumerate(GRUPP_ORDER, start=1):
        title = GRUPP_TITLES[grupp]
        entries = groups.get(grupp, [])
        text = f"{title}:\n" + "\n".join(entries)
        chunks.append({
            "chunk_id": f"course_info_{i:04d}",
            "source": "course_info",
            "type_of_content": "terminology",
            "section_title": title,
            "text": text,
        })

    return chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build course info chunks for RAG.")
    p.add_argument("--overview", default="raw_data/course_overview.csv")
    p.add_argument("--ordlista", default="raw_data/Ordlista.csv")
    p.add_argument("--outdir", default="data")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Building course overview chunk from {args.overview} …")
    overview_chunk = build_overview_chunk(args.overview)
    print(f"  {len(overview_chunk['text'].splitlines())} lines")

    print(f"Building terminology chunks from {args.ordlista} …")
    term_chunks = build_terminology_chunks(args.ordlista)
    for c in term_chunks:
        n = len(c["text"].splitlines()) - 1
        print(f"  {c['chunk_id']}  {c['section_title']}  ({n} terms)")

    all_chunks = [overview_chunk] + term_chunks
    out_path = os.path.join(args.outdir, "course_info_chunks.json")
    write_json(out_path, all_chunks)
    print(f"Written {len(all_chunks)} course info chunks to {out_path}")


if __name__ == "__main__":
    main()
