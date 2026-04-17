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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel

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
from rag import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_context,
    enrich_with_exercises,
    retrieve,
)

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-fallback-key-change-in-production")
COOKIE_NAME = "session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
DB_PATH = "biogas.db"

ADMIN_COOKIE_NAME = "admin_session"
ADMIN_COOKIE_MAX_AGE = 60 * 60 * 8  # 8 hours
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

serializer = URLSafeTimedSerializer(SECRET_KEY)

# ── Session helpers ──────────────────────────────────────────────────────


def set_session_cookie(response, user_id: int, email: str):
    token = serializer.dumps({"uid": user_id, "email": email})
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


# ── Admin session helpers ────────────────────────────────────────────────


def set_admin_cookie(response):
    token = serializer.dumps({"admin": True})
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        max_age=ADMIN_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


def get_current_admin(request: Request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE_NAME)
    if not token:
        return False
    try:
        return bool(serializer.loads(token, max_age=ADMIN_COOKIE_MAX_AGE).get("admin"))
    except BadSignature:
        return False


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
        INDEX_PATH, METADATA_PATH, CHUNKS_PATH, EMBEDDING_MODEL
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
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT,
            email             TEXT    NOT NULL UNIQUE,
            allow_logging     INTEGER NOT NULL DEFAULT 0,
            has_seen_welcome  INTEGER NOT NULL DEFAULT 0,
            created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
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
    app.state.http = httpx.AsyncClient(timeout=LLM_TIMEOUT)

    print("Server ready.")
    yield

    await app.state.http.aclose()
    await app.state.db.close()


# ── FastAPI app ──────────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)
app.mount("/static/pdfjs", StaticFiles(directory="static/pdfjs"), name="pdfjs")
app.mount("/static/images", StaticFiles(directory="static/images"), name="images")

# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("static/admin.html")


@app.get("/static/handbok.pdf")
async def handbok_pdf():
    return FileResponse("static/handbok.pdf", media_type="application/pdf")


class LoginBody(BaseModel):
    email: str
    allow_logging: bool = False


@app.post("/api/login")
async def login(body: LoginBody, request: Request):
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Ange en giltig e-postadress.")

    db: aiosqlite.Connection = request.app.state.db
    row = await db.execute_fetchall(
        "SELECT id, has_seen_welcome FROM users WHERE email = ?", (email,)
    )
    if not row:
        raise HTTPException(404, "E-postadressen är inte registrerad.")

    user_id = row[0][0]
    has_seen_welcome = bool(row[0][1])
    await db.execute(
        "UPDATE users SET allow_logging = ? WHERE id = ?",
        (int(body.allow_logging), user_id),
    )
    await db.commit()

    resp = JSONResponse({"ok": True, "email": email, "has_seen_welcome": has_seen_welcome})
    set_session_cookie(resp, user_id, email)
    return resp


@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME)
    return resp


@app.post("/api/welcome/seen")
async def welcome_seen(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not logged in.")
    db: aiosqlite.Connection = request.app.state.db
    await db.execute(
        "UPDATE users SET has_seen_welcome = 1 WHERE id = ?", (user["uid"],)
    )
    await db.commit()
    return JSONResponse({"ok": True})


# ── Admin API ────────────────────────────────────────────────────────────


class AdminLoginBody(BaseModel):
    email: str
    password: str


@app.post("/api/admin/login")
async def admin_login(body: AdminLoginBody):
    if not ADMIN_EMAIL or not ADMIN_PASSWORD:
        raise HTTPException(503, "Admin not configured.")
    if body.email.strip().lower() != ADMIN_EMAIL.lower() or body.password != ADMIN_PASSWORD:
        raise HTTPException(401, "Ogiltiga uppgifter.")
    resp = JSONResponse({"ok": True})
    set_admin_cookie(resp)
    return resp


@app.post("/api/admin/logout")
async def admin_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(ADMIN_COOKIE_NAME)
    return resp


@app.get("/api/admin/users")
async def admin_list_users(request: Request):
    if not get_current_admin(request):
        raise HTTPException(401, "Not authorized.")
    db: aiosqlite.Connection = request.app.state.db
    rows = await db.execute_fetchall(
        "SELECT id, name, email, allow_logging, created_at FROM users ORDER BY created_at DESC"
    )
    users = [
        {
            "id": r[0],
            "name": r[1],
            "email": r[2],
            "allow_logging": bool(r[3]),
            "created_at": r[4],
        }
        for r in rows
    ]
    return JSONResponse(users)


class AddUserBody(BaseModel):
    name: str
    email: str


@app.post("/api/admin/users")
async def admin_add_user(body: AddUserBody, request: Request):
    if not get_current_admin(request):
        raise HTTPException(401, "Not authorized.")
    name = body.name.strip()
    email = body.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Ange en giltig e-postadress.")
    db: aiosqlite.Connection = request.app.state.db
    try:
        await db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)", (name or None, email)
        )
        await db.commit()
    except aiosqlite.IntegrityError:
        raise HTTPException(409, "E-postadressen är redan registrerad.")
    return JSONResponse({"ok": True})


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, request: Request):
    if not get_current_admin(request):
        raise HTTPException(401, "Not authorized.")
    db: aiosqlite.Connection = request.app.state.db
    await db.execute("DELETE FROM query_logs WHERE user_id = ?", (user_id,))
    result = await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(404, "User not found.")
    return JSONResponse({"ok": True})


# ── Query ────────────────────────────────────────────────────────────────


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
    results = enrich_with_exercises(results, st.meta, st.texts)
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
                LLM_URL,
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
