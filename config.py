"""
config.py -- Central configuration for the biogas RAG system.

All tunable parameters in one place. Change embedding model, LLM model,
retrieval settings, etc. here instead of hunting through multiple files.
"""

import os

# ── Device selection ────────────────────────────────────────────────────
# MPS (Apple Silicon) causes segfaults with gte-multilingual-base;
# only use CUDA or CPU.

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ── Embedding model ─────────────────────────────────────────────────────

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# ── LLM ─────────────────────────────────────────────────────────────────

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8000/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
LLM_TIMEOUT = 120.0

# ── Retrieval ───────────────────────────────────────────────────────────

TOP_K = 8

# ── Data paths ──────────────────────────────────────────────────────────

INDEX_PATH = "data/index.faiss"
METADATA_PATH = "data/chunk_metadata.json"
CHUNKS_PATH = "data/chunks.json"
LECTURE_CHUNKS_PATH = "data/lecture_chunks.json"
