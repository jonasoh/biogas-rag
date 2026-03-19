#!/usr/bin/env python3
"""
Extract HTML tables from handbook.json and create a separate handbook_tables.json file.
Also removes the tables from the main handbook text.
"""

import json
import re
from pathlib import Path


def extract_tables_and_legends(text):
    """
    Extract tables and their legends from handbook text.
    Returns a list of dicts with table info and the cleaned text.
    """
    tables = []

    # Pattern to match <table>...</table> followed by <table-legend>...</table-legend>
    pattern = r"(<table>.*?</table>)\n<table-legend>(.*?)</table-legend>"

    matches = list(re.finditer(pattern, text, re.DOTALL))

    for match in matches:
        table_html = match.group(1)
        legend = match.group(2)

        table_info = {"table": table_html, "legend": legend}
        tables.append(table_info)

    # Remove tables and legends from text
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)

    return tables, cleaned_text


def extract_context(page_text, page_num):
    """
    Extract chapter and section information from the page text.
    """
    chapter = None
    section = None

    # Look for <heading> tags (chapters)
    heading_match = re.search(r"<heading>(.*?)</heading>", page_text, re.DOTALL)
    if heading_match:
        chapter = heading_match.group(1).strip()

    # Look for <subheading> tags (sections)
    subheading_match = re.search(
        r"<subheading>(.*?)</subheading>", page_text, re.DOTALL
    )
    if subheading_match:
        section = subheading_match.group(1).strip()

    return chapter, section


def main():
    # Read the handbook
    handbook_path = Path("raw_data/handbook.json")

    with open(handbook_path, "r", encoding="utf-8") as f:
        handbook = json.load(f)

    all_tables = []
    modified_handbook = {}

    # Process each page
    for page_num, content in handbook.items():
        chapter, section = extract_context(content, page_num)
        tables, cleaned_content = extract_tables_and_legends(content)

        # Store modified content (text without tables)
        modified_handbook[page_num] = cleaned_content

        # Store tables with metadata
        for table_info in tables:
            table_entry = {
                "page": int(page_num),
                "chapter": chapter,
                "section": section,
                "table": table_info["table"],
                "legend": table_info["legend"],
            }
            all_tables.append(table_entry)

    # Write the tables to a new file
    tables_output_path = Path("data/handbook_tables.json")
    tables_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tables_output_path, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(all_tables)} tables")
    print(f"Tables saved to: {tables_output_path}")

    # Write the modified handbook (without tables) to a new file
    notables_path = Path("raw_data/handbook_notables.json")
    with open(notables_path, "w", encoding="utf-8") as f:
        json.dump(modified_handbook, f, ensure_ascii=False, indent=2)

    print(f"Handbook without tables saved to: {notables_path}")
    print("\nTables extracted:")
    for i, table in enumerate(all_tables, 1):
        print(f"{i}. Page {table['page']}: {table['legend'][:60]}...")


if __name__ == "__main__":
    main()
