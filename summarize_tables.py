"""
summarize_tables.py – Convert extracted HTML tables to markdown and
generate natural-language summaries via a local LLM.

Usage:
    python summarize_tables.py                      # full run
    python summarize_tables.py --dry-run             # show markdown only, skip LLM
    python summarize_tables.py --input data/handbook_tables.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

LLM_URL = "http://127.0.0.1:8000/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3.5-35B-A3B"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

SUMMARY_PROMPT = """\
Du är en expert på biogasteknik. Nedan finns en tabell ur en biogashandbok.
Skriv en sammanfattning på svenska (2-4 meningar) som beskriver vad tabellen visar,
vilka data den innehåller, och vilka centrala värden eller slutsatser som kan dras.
Inkludera nyckelord som gör tabellen sökbar via en sökmotor.

Tabell: {legend}

{table_markdown}"""


# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

_TAG = re.compile(r"<(/?)(\w+)[^>]*>")

# Unicode sub/superscript maps (common chemistry characters)
_SUB_MAP = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUP_MAP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _convert_sub_sup(html: str) -> str:
    """Replace <sub>X</sub> and <sup>X</sup> with unicode equivalents."""

    def _replace(m: re.Match) -> str:
        tag = m.group(1)
        content = m.group(2)
        char_map = _SUB_MAP if tag == "sub" else _SUP_MAP
        return content.translate(char_map)

    html = re.sub(r"<sub>(.*?)</sub>", lambda m: _replace(type("M", (), {"group": lambda s, n: {1: "sub", 2: m.group(1)}[n]})()), html)
    html = re.sub(r"<sup>(.*?)</sup>", lambda m: _replace(type("M", (), {"group": lambda s, n: {1: "sup", 2: m.group(1)}[n]})()), html)
    return html


def _strip_tags(html: str) -> str:
    """Remove all remaining HTML tags."""
    return re.sub(r"<[^>]+>", "", html).strip()


def _parse_rows(table_html: str) -> list[list[str]]:
    """Parse HTML table into a list of rows, each a list of cell strings."""
    table_html = _convert_sub_sup(table_html)
    rows = []
    for tr_match in re.finditer(r"<tr>(.*?)</tr>", table_html, re.DOTALL):
        tr = tr_match.group(1)
        cells = re.findall(r"<(?:th|td)>(.*?)</(?:th|td)>", tr, re.DOTALL)
        cells = [_strip_tags(c).strip() for c in cells]
        rows.append(cells)
    return rows


def _is_single_column_list(rows: list[list[str]]) -> bool:
    """Check if this is a single-column list (no real table structure)."""
    return all(len(row) == 1 for row in rows)


def html_table_to_markdown(table_html: str, legend: str = "") -> str:
    """Convert an HTML table to markdown."""
    rows = _parse_rows(table_html)
    if not rows:
        return ""

    # Single-column tables → bullet list
    if _is_single_column_list(rows):
        items = [row[0] for row in rows if row[0]]
        return "\n".join(f"- {item}" for item in items)

    # Determine if first row is a header (came from <th> tags)
    first_row_html = re.search(r"<tr>(.*?)</tr>", _convert_sub_sup(table_html), re.DOTALL)
    has_header = bool(first_row_html and "<th>" in first_row_html.group(1))

    # Normalise column count
    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    # Calculate column widths
    col_widths = [
        max(len(rows[i][j]) for i in range(len(rows)))
        for j in range(max_cols)
    ]
    col_widths = [max(w, 3) for w in col_widths]

    def fmt_row(row: list[str]) -> str:
        cells = [cell.ljust(w) for cell, w in zip(row, col_widths)]
        return "| " + " | ".join(cells) + " |"

    lines = []
    if has_header:
        lines.append(fmt_row(rows[0]))
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        data_rows = rows[1:]
    else:
        # No header — use first row as header anyway (best we can do)
        lines.append(fmt_row(rows[0]))
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        data_rows = rows[1:]

    for row in data_rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------


def summarize_table(legend: str, table_markdown: str) -> str:
    """Call the local LLM to generate a natural-language summary."""
    prompt = SUMMARY_PROMPT.format(legend=legend, table_markdown=table_markdown)
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                LLM_URL,
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Strip <think> blocks if present (Qwen reasoning)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except (httpx.HTTPError, KeyError) as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] LLM error: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return f"[Sammanfattning kunde inte genereras för: {legend}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize extracted handbook tables.")
    p.add_argument("--input", default="data/handbook_tables.json")
    p.add_argument("--output", default="data/handbook_tables_enriched.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show markdown conversion only, skip LLM calls",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tables = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not tables:
        print(f"No tables found in {args.input}. Run extract_tables.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(tables)} tables from {args.input}")
    enriched = []

    for i, t in enumerate(tables, 1):
        legend = t["legend"]
        print(f"\n[{i}/{len(tables)}] Page {t['page']}: {legend[:70]}")

        md = html_table_to_markdown(t["table"], legend)
        print(md)

        if args.dry_run:
            summary = "(dry run — skipped)"
        else:
            print("  Summarizing via LLM …", end=" ", flush=True)
            summary = summarize_table(legend, md)
            print("done")
            print(f"  Summary: {summary[:120]}…" if len(summary) > 120 else f"  Summary: {summary}")

        enriched.append(
            {
                "page": t["page"],
                "chapter": t["chapter"],
                "section": t["section"],
                "table_html": t["table"],
                "legend": legend,
                "table_markdown": md,
                "summary": summary,
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(enriched)} enriched tables written to {args.output}")


if __name__ == "__main__":
    main()
