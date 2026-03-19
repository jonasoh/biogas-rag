"""
app.py -- FastAPI web interface for the biogas RAG handbook.

Run:
    python app.py
    # or: uvicorn app:app --host 0.0.0.0 --port 8080
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager

import aiosqlite
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel

from query import DEFAULT_CHUNKS, DEFAULT_INDEX, DEFAULT_META, DEFAULT_MODEL, load_assets
from test_rag import (
    LLAMA_URL,
    LLM_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    USER_TEMPLATE,
    build_context,
    retrieve,
)

# ── Configuration ────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-fallback-key-change-in-production")
COOKIE_NAME = "session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
DB_PATH = "biogas.db"

serializer = URLSafeTimedSerializer(SECRET_KEY)

# ── Session helpers ──────────────────────────────────────────────────────


def set_session_cookie(response, user_id: int, name: str):
    token = serializer.dumps({"uid": user_id, "name": name})
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        return serializer.loads(token, max_age=COOKIE_MAX_AGE)
    except BadSignature:
        return None


# ── Database helpers ─────────────────────────────────────────────────────


async def log_query(db: aiosqlite.Connection, user_id: int, question: str, answer: str):
    await db.execute(
        "INSERT INTO query_logs (user_id, question, answer) VALUES (?, ?, ?)",
        (user_id, question, answer),
    )
    await db.commit()


# ── Lifespan (startup / shutdown) ────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load RAG assets (FAISS index, metadata, chunks, embedding model)
    index, meta, texts, model = load_assets(
        DEFAULT_INDEX, DEFAULT_META, DEFAULT_CHUNKS, DEFAULT_MODEL
    )
    app.state.index = index
    app.state.meta = meta
    app.state.texts = texts
    app.state.model = model

    # Open SQLite database
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    NOT NULL UNIQUE,
            allow_logging INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    await db.execute(
        """CREATE TABLE IF NOT EXISTS query_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(id),
            question   TEXT    NOT NULL,
            answer     TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    await db.commit()
    app.state.db = db

    # Shared async HTTP client for LLM requests
    app.state.http = httpx.AsyncClient(timeout=120.0)

    print("Server ready.")
    yield

    await app.state.http.aclose()
    await app.state.db.close()


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)
app.mount("/static/pdfjs", StaticFiles(directory="static/pdfjs"), name="pdfjs")

# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/static/handbok.pdf")
async def handbok_pdf():
    return FileResponse("static/handbok.pdf", media_type="application/pdf")


class SignupBody(BaseModel):
    name: str
    email: str
    allow_logging: bool = False


@app.post("/api/signup")
async def signup(body: SignupBody, request: Request):
    name = body.name.strip()
    email = body.email.strip().lower()
    if not name or not email or "@" not in email:
        raise HTTPException(400, "Name and valid email are required.")

    db: aiosqlite.Connection = request.app.state.db
    try:
        cursor = await db.execute(
            "INSERT INTO users (name, email, allow_logging) VALUES (?, ?, ?)",
            (name, email, int(body.allow_logging)),
        )
        await db.commit()
        user_id = cursor.lastrowid
    except aiosqlite.IntegrityError:
        raise HTTPException(409, "Email already registered. Use login instead.")

    resp = JSONResponse({"ok": True, "name": name})
    set_session_cookie(resp, user_id, name)
    return resp


class LoginBody(BaseModel):
    email: str


@app.post("/api/login")
async def login(body: LoginBody, request: Request):
    email = body.email.strip().lower()
    db: aiosqlite.Connection = request.app.state.db
    row = await db.execute_fetchall(
        "SELECT id, name FROM users WHERE email = ?", (email,)
    )
    if not row:
        raise HTTPException(404, "Email not found. Sign up first.")

    user_id, name = row[0][0], row[0][1]
    resp = JSONResponse({"ok": True, "name": name})
    set_session_cookie(resp, user_id, name)
    return resp


@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME)
    return resp


@app.get("/api/query")
async def query_sse(q: str, request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not logged in.")

    q = q.strip()
    if not q or len(q) > 2000:
        raise HTTPException(400, "Query must be 1-2000 characters.")

    # Retrieve relevant chunks (blocking call → thread pool)
    loop = asyncio.get_event_loop()
    st = request.app.state
    results = await loop.run_in_executor(
        None,
        lambda: retrieve(q, st.index, st.meta, st.texts, st.model, TOP_K),
    )
    context = build_context(results)

    # Build LLM messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=q)},
    ]

    # Look up whether user opted into logging
    db: aiosqlite.Connection = st.db
    row = await db.execute_fetchall(
        "SELECT allow_logging FROM users WHERE id = ?", (user["uid"],)
    )
    allow_logging = bool(row and row[0][0])

    async def event_stream():
        full_answer = []
        try:
            async with st.http.stream(
                "POST",
                LLAMA_URL,
                json={"model": LLM_MODEL, "messages": messages, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        full_answer.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'LLM server unavailable.'})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            yield "data: [DONE]\n\n"
            if allow_logging and full_answer:
                await log_query(db, user["uid"], q, "".join(full_answer))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
