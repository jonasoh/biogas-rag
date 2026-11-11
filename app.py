"""
app.py -- FastAPI web interface for the biogas RAG system.

Run:
    python app.py
    # or: uvicorn app:app --host 0.0.0.0 --port 8080
"""

import asyncio
import json
import os
import secrets
import time
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
import analytics
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
DB_PATH = os.environ.get("DB_PATH", "biogas.db")

ADMIN_COOKIE_NAME = "admin_session"
ADMIN_COOKIE_MAX_AGE = 60 * 60 * 8  # 8 hours
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

# Tag each logged question with theme/outcome using the LLM, right after it
# has been answered. Costs one short extra LLM call per logged question.
ANALYTICS_TAGGING = os.environ.get("ANALYTICS_TAGGING", "1") != "0"
PROMPT_VERSION = analytics.prompt_version(SYSTEM_PROMPT)

serializer = URLSafeTimedSerializer(SECRET_KEY)

# ── Session helpers ──────────────────────────────────────────────────────


def new_session_id() -> str:
    return secrets.token_hex(8)


def set_session_cookie(response, user_id: int, email: str, sid: str | None = None):
    # "sid" identifies one login (one browser sign-in), so analytics can
    # count sessions separately from accounts.
    token = serializer.dumps({"uid": user_id, "email": email, "sid": sid or new_session_id()})
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


async def log_query(db: aiosqlite.Connection, **fields) -> int:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    cur = await db.execute(
        f"INSERT INTO query_logs ({cols}) VALUES ({marks})", tuple(fields.values())
    )
    await db.commit()
    return cur.lastrowid


# Background tasks (logging/tagging) must be referenced until done, or they
# can be garbage-collected mid-flight.
_background_tasks: set[asyncio.Task] = set()


def run_in_background(coro):
    task = asyncio.get_running_loop().create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def log_and_tag(app_state, fields: dict):
    db = app_state.db
    log_id = await log_query(db, **fields)
    if fields["status"] == "ok" and ANALYTICS_TAGGING:
        await analytics.tag_log_row(
            db, app_state.http, LLM_URL, LLM_MODEL, log_id,
            fields["question"], fields["answer"],
        )


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
    await analytics.migrate(db)
    app.state.db = db
    app.state.backfill = analytics.Backfill()

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
        "SELECT id, name, email, allow_logging, has_seen_welcome, created_at FROM users ORDER BY created_at DESC"
    )
    users = [
        {
            "id": r[0],
            "name": r[1],
            "email": r[2],
            "allow_logging": bool(r[3]),
            "has_seen_welcome": bool(r[4]),
            "created_at": r[5],
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
    # Query logs are kept for analytics. users.id is AUTOINCREMENT, so the
    # deleted id is never reused and the logs can't be misattributed.
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

    started = time.monotonic()

    # Retrieve relevant chunks (blocking call → thread pool)
    loop = asyncio.get_running_loop()
    st = request.app.state
    results = await loop.run_in_executor(
        None,
        lambda: retrieve(q, st.index, st.meta, st.texts, st.model, TOP_K),
    )
    results = enrich_with_exercises(results, st.meta, st.texts)
    context = build_context(results)
    retrieval_json, top_score = analytics.summarize_retrieval(results)

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

    # Sessions created before session ids existed get one now.
    sid = user.get("sid") or new_session_id()

    async def event_stream():
        full_answer = []
        status = "ok"
        try:
            async with st.http.stream(
                "POST",
                LLM_URL,
                json={"model": LLM_MODEL, "messages": messages, "stream": True},
            ) as resp:
                if resp.status_code != 200:
                    status = "error"
                    yield f"data: {json.dumps({'error': 'LLM server error.'})}\n\n"
                else:
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
            if status == "ok" and not full_answer:
                status = "error"
            yield "data: [DONE]\n\n"
        except httpx.ConnectError:
            status = "error"
            yield f"data: {json.dumps({'error': 'LLM server unavailable.'})}\n\n"
            yield "data: [DONE]\n\n"
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError):
            status = "error"
            yield f"data: {json.dumps({'error': 'LLM request failed.'})}\n\n"
            yield "data: [DONE]\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            # Client went away mid-answer.
            status = "aborted"
            raise
        finally:
            # Logged from a separate task: this generator may be in a
            # cancelled scope where further awaits would be interrupted.
            if allow_logging:
                run_in_background(
                    log_and_tag(
                        st,
                        {
                            "user_id": user["uid"],
                            "session_id": sid,
                            "question": q,
                            "answer": "".join(full_answer),
                            "status": status,
                            "latency_ms": int((time.monotonic() - started) * 1000),
                            "llm_model": LLM_MODEL,
                            "prompt_version": PROMPT_VERSION,
                            "retrieval": retrieval_json,
                            "top_score": top_score,
                            "refused": int(
                                analytics.looks_refused("".join(full_answer))
                            ),
                        },
                    )
                )

    response = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    if not user.get("sid"):
        set_session_cookie(response, user["uid"], user["email"], sid)
    return response


# ── Admin analytics ──────────────────────────────────────────────────────


@app.get("/api/admin/analytics")
async def admin_analytics(request: Request, days: int | None = None):
    if not get_current_admin(request):
        raise HTTPException(401, "Not authorized.")
    st = request.app.state
    rows = await analytics.fetch_rows(st.db, days)
    data = analytics.compute_analytics(rows, st.meta)
    data["backfill"] = st.backfill.status()
    data["tagging_enabled"] = ANALYTICS_TAGGING
    data["prompt_version"] = PROMPT_VERSION
    return JSONResponse(data)


@app.get("/api/admin/questions/{log_id}")
async def admin_question(log_id: int, request: Request):
    if not get_current_admin(request):
        raise HTTPException(401, "Not authorized.")
    st = request.app.state
    rows = await st.db.execute_fetchall(
        """SELECT q.id, q.question, q.answer, q.created_at, q.session_id, q.status,
                  q.latency_ms, q.llm_model, q.prompt_version, q.retrieval, q.top_score,
                  a.theme, a.kind, a.outcome
           FROM query_logs q LEFT JOIN query_analysis a ON a.log_id = q.id
           WHERE q.id = ?""",
        (log_id,),
    )
    if not rows:
        raise HTTPException(404, "Not found.")
    r = dict(rows[0])
    units = analytics.course_units(st.meta)
    r["retrieval"] = [
        {"chunk_id": cid, "score": score, "unit": units.get(cid, (None, cid))[1]}
        for cid, score in (json.loads(r["retrieval"]) if r["retrieval"] else [])
    ]
    return JSONResponse(r)


class BackfillBody(BaseModel):
    retag_all: bool = False


@app.post("/api/admin/analytics/backfill")
async def admin_backfill(body: BackfillBody, request: Request):
    if not get_current_admin(request):
        raise HTTPException(401, "Not authorized.")
    st = request.app.state
    if st.backfill.running:
        raise HTTPException(409, "En uppdatering pågår redan.")

    async def retrieve_fn(question: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: retrieve(question, st.index, st.meta, st.texts, st.model, TOP_K),
        )

    st.backfill.running = True  # set before scheduling to avoid double starts
    run_in_background(
        st.backfill.run(st.db, st.http, LLM_URL, LLM_MODEL, retrieve_fn, body.retag_all)
    )
    return JSONResponse(st.backfill.status())


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
