# A Continuing Education Tool for Farm-Scale Anaerobic Digestion

A retrieval-augmented generation (RAG) assistant designed for an online course aimed at farmers who are interested in or currently operating farm-scale anaerobic digesters. The system lets users ask questions in natural language and receive answers grounded in course materials: Microbiological Handbook for Biogas Plants [(swe)](https://www.avfallsverige.se/rapporter-utveckling/rapporter/biogasprocessens-mikrobiologi-handbok/) [(eng)](https://www.offgridding.com/uploads/1/0/1/5/10150858/microbiologicalhandbook_biogas.pdf) by Anna Schnürer and Åsa Jarvis, as well as video lecture transcripts.

The system can show video lectures and handbook pages directly in the interface. All answers include citations to the relevant source materials. It can also recommend relevant exercises from the handbook, and refer to an external (web-based calculation tool)[https://github.com/jonasoh/biogas-educational-calculator] for key process parameters.

<img width="1181" height="713" alt="image" src="https://github.com/user-attachments/assets/da5f0e6f-54cc-4611-918e-73864fc49648" />

## Hardware requirements

The system is designed to run locally on a modest machine. We use a GMKtec EVO-X2 with 128 GB unified RAM, but it should work on machines with less memory as well, such as a Mac Mini (not tested).

## Preparing the Data

The handbook PDF was transcribed to JSON using Google Gemini with a prompt that extracts page-by-page text, marks headings with `<heading>` and `<subheading>` tags, formats tables as HTML, and replaces figures with descriptive markdown image tags.

Lecture transcripts are provided as `.vtt` subtitle files and chunked with timestamp metadata preserved. Transcripts were generated using [MacWhisper](https://goodsnooze.gumroad.com/l/macwhisper).

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) or similar recommended for environment management
- An external LLM endpoint compatible with the OpenAI chat completions API (we run Qwen-3.6-35B-A3B using [llama.cpp](https://github.com/ggml-org/llama.cpp))
- The handbook PDF placed in `static/` (required for the in-browser PDF viewer)

### Environment

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create a `.env` file:

```
ADMIN_EMAIL=your@email.com
ADMIN_PASSWORD=yourpassword
LLM_URL=http://127.0.0.1:11434/v1/chat/completions   # or wherever your LLM is
LLM_MODEL=your-model-name
```

### Build the Vector Index

Place the preprocessed data files in `raw_data/` (handbook JSON, lecture `.vtt` files, course CSVs), then run the full pipeline:

```bash
# Extract HTML tables from the handbook and remove them from the main text
python extract_tables.py
# → data/handbook_tables.json (tables with legends and chapter context)
# → raw_data/handbook_notables.json (handbook text without tables)

# Generate LLM summaries of each table (requires a running LLM endpoint)
python summarize_tables.py
# → data/handbook_tables_enriched.json
# Use --dry-run to preview markdown conversion without calling the LLM

# Chunk all source materials
python chunk_handbook.py    # uses handbook_notables.json
python chunk_lectures.py
python chunk_course_info.py

# Embed all chunks and write the vector index
python build_index.py
```

This produces `data/vectors.npy`, `data/chunks.json`, and `data/chunk_metadata.json`.

#### Why tables are handled separately

The handbook's HTML tables do not chunk well as plain text. `extract_tables.py` pulls them out of the main text before chunking so they don't pollute neighbouring passages. `summarize_tables.py` then converts each table to Markdown and asks the LLM for a short natural-language summary, making the table content discoverable through semantic search.

### PDF for the In-Browser Viewer

The handbook PDF must be available to the web server for the [PDF.js](https://mozilla.github.io/pdf.js/) viewer to work. Place the file at:

```
static/handbok.pdf
```

Prebuilt PDF.js should be placed in `static/pdfjs`.

### Run the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser.

## CLI Usage

You can query the system directly from the terminal without the web interface:

```bash
python rag.py "How does temperature affect methane production?"
python rag.py --debug "question"   # shows the full LLM prompt
python rag.py                      # interactive mode
```
