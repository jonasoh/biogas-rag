"""
analytics.py -- Question analytics for the admin interface.

Two layers:

1. Cheap, deterministic signals stored when a query is logged
   (see ``log_query`` in app.py): retrieved chunks + scores, a heuristic
   refusal flag, status, latency, model and prompt version.

2. LLM tagging, run in the background after logging (and as a backfill
   from the admin page): a normalised theme, the kind of question
   (course content / course admin / small talk) and whether it was
   answered. Stored in ``query_analysis`` so everything can be re-tagged
   later without touching the raw log.

``compute_analytics`` aggregates both layers into the payload rendered by
the "Analys" tab in static/admin.html.
"""

import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import aiosqlite
import httpx

# ── Schema ───────────────────────────────────────────────────────────────

# Columns added to query_logs on top of the original
# (id, user_id, question, answer, created_at). Added with ALTER TABLE so
# existing databases are migrated in place; old rows get NULLs.
QUERY_LOG_EXTRA_COLUMNS = {
    "session_id": "TEXT",  # random id per login (one browser sign-in)
    "status": "TEXT",  # ok | error | aborted
    "latency_ms": "INTEGER",
    "llm_model": "TEXT",
    "prompt_version": "TEXT",  # short hash of SYSTEM_PROMPT
    "retrieval": "TEXT",  # JSON: [[chunk_id, score], ...] in rank order
    "top_score": "REAL",
    "refused": "INTEGER",  # heuristic, from answer text
}

CREATE_QUERY_ANALYSIS = """CREATE TABLE IF NOT EXISTS query_analysis (
    log_id      INTEGER PRIMARY KEY REFERENCES query_logs(id) ON DELETE CASCADE,
    theme       TEXT,
    kind        TEXT,     -- content | admin | smalltalk
    outcome     TEXT,     -- answered | partial | refused
    tagger      TEXT,     -- model used for tagging
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
)"""


async def migrate(db: aiosqlite.Connection):
    cols = {r[1] for r in await db.execute_fetchall("PRAGMA table_info(query_logs)")}
    for name, type_ in QUERY_LOG_EXTRA_COLUMNS.items():
        if name not in cols:
            await db.execute(f"ALTER TABLE query_logs ADD COLUMN {name} {type_}")
    await db.execute(CREATE_QUERY_ANALYSIS)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at)"
    )
    await db.commit()


def prompt_version(system_prompt: str) -> str:
    return hashlib.sha1(system_prompt.encode()).hexdigest()[:8]


# ── Heuristic refusal detection ──────────────────────────────────────────

# Phrases the assistant uses when the context doesn't cover the question.
# Only the opening of the answer is checked: a refusal is stated up front,
# whereas later caveats ("handboken nämner inte ...") usually follow a real
# answer.
_REFUSAL_RE = re.compile(
    r"kan inte (besvara|hitta|svara)"
    r"|(innehåller|nämner|specificerar|anger) (inte|ingen|inga)"
    r"|finns (inte|ingen|inga) (information|uppgifter)",
    re.IGNORECASE,
)


def looks_refused(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer[:250]))


# ── LLM tagging ──────────────────────────────────────────────────────────

KINDS = ("content", "admin", "smalltalk")
OUTCOMES = ("answered", "partial", "refused")

TAG_SYSTEM_PROMPT = """\
Du analyserar frågor som studenter ställt till en AI-assistent i en högskolekurs \
om driftoptimering av lantbruksbaserad biogasproduktion. Assistenten svarar bara \
utifrån kursmaterialet (en handbok om biogasprocessens mikrobiologi samt föreläsningar). \
Analysen används av läraren för att se vad studenterna undrar över och vad som saknas \
i kursmaterialet.

Du får en fråga och BÖRJAN av assistentens svar (svaret är avkortat). Returnera ett \
JSON-objekt med:

"theme": det konkreta ämne studenten frågar om, som ett kort tema på svenska \
(1–4 ord, rättstavat, versal i början). Var konkret: temat ska vara det specifika \
begrepp, material eller problem frågan gäller, t.ex. "FOS/TAC", "Kogödsel som substrat", \
"HRT och OLR", "Ammoniakhämning", "Järntillsats", "Affärsmodeller", "Skattereduktion", \
"Opponering". Använd INTE breda samlingsrubriker som "Biogasprocessen", \
"Biogasprocessens mikrobiologi", "Substrat" eller "Processdrift" när frågan gäller \
något mer specifikt. Om ett befintligt tema nedan gäller SAMMA specifika ämne, \
använd det exakt (stavning och versaler). Annars, skapa ett nytt.

"kind": \
- "content": en fråga om ett ämne, även om ämnet inte tas upp i kursen (t.ex. \
ekonomi, juridik, teknik) och även om studenten frågar var i materialet något står. \
- "admin": frågor om själva kursens genomförande: grupper, opponering, schema, \
inlämningar, examination, var man hittar referenslistan eller en viss referens. \
- "smalltalk": hälsningar, test, nonsens.

"outcome": \
- "answered": assistenten ger ett sakligt svar på frågan. Detta är det normala fallet. \
Att svarsutdraget är avkortat betyder INTE att svaret är ofullständigt. \
- "partial": assistenten säger UTTRYCKLIGEN att materialet saknar en del av det \
som efterfrågas (t.ex. "det finns inget exakt värde", "kontexten specificerar inte") \
men svarar ändå på en del eller ger närliggande information. \
- "refused": assistenten säger att den inte kan besvara frågan. \
För smalltalk, välj "answered" om assistenten svarade.

Svara ENDAST med JSON, t.ex. {"theme": "HRT och OLR", "kind": "content", "outcome": "answered"}"""

TAG_USER_TEMPLATE = """\
Befintliga teman: {themes}

Fråga: {question}

Svar (början): {answer}"""

# How many existing themes to show the tagger (most frequent first).
MAX_THEMES_IN_PROMPT = 150


def _parse_tag(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    theme = str(obj.get("theme", "")).strip().strip(".")[:60]
    kind = obj.get("kind")
    outcome = obj.get("outcome")
    if not theme or kind not in KINDS or outcome not in OUTCOMES:
        return None
    return {"theme": theme[0].upper() + theme[1:], "kind": kind, "outcome": outcome}


async def existing_themes(db: aiosqlite.Connection) -> list[str]:
    # The most frequent themes, but listed alphabetically: a stable order keeps
    # the prompt prefix identical between calls so the LLM server's prompt
    # cache can be reused (prompt processing dominates the cost of a tag).
    rows = await db.execute_fetchall(
        "SELECT theme FROM query_analysis WHERE theme IS NOT NULL "
        "GROUP BY theme ORDER BY COUNT(*) DESC LIMIT ?",
        (MAX_THEMES_IN_PROMPT,),
    )
    return sorted((r[0] for r in rows), key=str.casefold)


async def tag_question(
    http: httpx.AsyncClient,
    llm_url: str,
    llm_model: str,
    question: str,
    answer: str,
    themes: list[str],
) -> dict | None:
    """Ask the LLM for theme/kind/outcome. Returns None on any failure."""
    user = TAG_USER_TEMPLATE.format(
        themes=", ".join(themes) if themes else "(inga ännu)",
        question=question,
        answer=(answer[:600] or "(inget svar)"),
    )
    try:
        resp = await http.post(
            llm_url,
            json={
                "model": llm_model,
                "messages": [
                    {"role": "system", "content": TAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": 200,
                "stream": False,
                "response_format": {"type": "json_object"},
                # Qwen3-style models: skip the reasoning phase for this task.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None
    return _parse_tag(content)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def snap_theme(theme: str, themes: list[str]) -> str:
    """Map a case variant or typo of an existing theme onto that theme.

    Only near-identical spellings are merged (1 edit, or 2 for long names);
    "Fårgödsel som substrat" must not become "Kogödsel som substrat".
    """
    key = theme.casefold()
    # Short names are often acronyms (BMP vs RMP): exact matches only.
    max_dist = 0 if len(key) < 6 else 1 if len(key) < 12 else 2
    best, best_d = theme, max_dist + 1
    for t in themes:
        d = _edit_distance(key, t.casefold())
        if d < best_d:
            best, best_d = t, d
    return best


async def save_tag(db: aiosqlite.Connection, log_id: int, tag: dict, tagger: str):
    await db.execute(
        "INSERT OR REPLACE INTO query_analysis (log_id, theme, kind, outcome, tagger) "
        "VALUES (?, ?, ?, ?, ?)",
        (log_id, tag["theme"], tag["kind"], tag["outcome"], tagger),
    )
    await db.commit()


async def tag_log_row(db, http, llm_url, llm_model, log_id, question, answer) -> bool:
    themes = await existing_themes(db)
    tag = await tag_question(http, llm_url, llm_model, question, answer, themes)
    if tag is None:
        return False
    tag["theme"] = snap_theme(tag["theme"], themes)
    await save_tag(db, log_id, tag, llm_model)
    return True


# ── Retrieval summary (shared by logging and backfill) ──────────────────


def summarize_retrieval(results: list[dict]) -> tuple[str, float | None]:
    """Compact JSON of the ranked (non-injected) retrieval + top score."""
    ranked = [r for r in results if not r.get("injected")]
    payload = json.dumps([[r["chunk_id"], round(r["score"], 4)] for r in ranked])
    top = max((r["score"] for r in ranked), default=None)
    return payload, top


# ── Backfill ─────────────────────────────────────────────────────────────


class Backfill:
    """Single background job filling in missing retrieval data and tags."""

    def __init__(self):
        self.running = False
        self.phase = ""
        self.done = 0
        self.total = 0
        self.failed = 0
        self.error = ""

    def status(self) -> dict:
        return {
            "running": self.running,
            "phase": self.phase,
            "done": self.done,
            "total": self.total,
            "failed": self.failed,
            "error": self.error,
        }

    async def run(self, db, http, llm_url, llm_model, retrieve_fn, retag_all: bool):
        """retrieve_fn(question) -> awaitable list of retrieval results."""
        self.running, self.error, self.failed = True, "", 0
        try:
            # 1. Retrieval data for rows logged before it was recorded.
            rows = await db.execute_fetchall(
                "SELECT id, question, answer FROM query_logs WHERE retrieval IS NULL"
            )
            self.phase, self.done, self.total = "retrieval", 0, len(rows)
            for log_id, question, answer in rows:
                results = await retrieve_fn(question)
                retrieval, top = summarize_retrieval(results)
                await db.execute(
                    "UPDATE query_logs SET retrieval = ?, top_score = ?, "
                    "refused = COALESCE(refused, ?) WHERE id = ?",
                    (retrieval, top, int(looks_refused(answer)), log_id),
                )
                self.done += 1
            await db.commit()

            # 2. LLM tags. Oldest first so themes grow in a stable order.
            if retag_all:
                await db.execute("DELETE FROM query_analysis")
                await db.commit()
            rows = await db.execute_fetchall(
                "SELECT q.id, q.question, q.answer FROM query_logs q "
                "LEFT JOIN query_analysis a ON a.log_id = q.id "
                "WHERE a.log_id IS NULL AND COALESCE(q.status, 'ok') = 'ok' "
                "ORDER BY q.id"
            )
            self.phase, self.done, self.total = "tagging", 0, len(rows)
            for log_id, question, answer in rows:
                ok = await tag_log_row(
                    db, http, llm_url, llm_model, log_id, question, answer
                )
                self.done += 1
                if not ok:
                    self.failed += 1
                    # LLM down or misbehaving: stop rather than hammering it.
                    if self.failed >= 5 and self.failed == self.done:
                        self.error = "LLM-taggning misslyckades; är LLM-servern igång?"
                        break
        except Exception as e:  # surface in the UI instead of dying silently
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False
            self.phase = ""


# ── Aggregation ──────────────────────────────────────────────────────────

# A question "hits" a chapter/lecture if one of its top-N retrieved
# chunks belongs to it.
COVERAGE_TOP_N = 3


def _chunk_unit(m: dict) -> tuple[str, str] | None:
    """Map chunk metadata to the course unit it belongs to."""
    source = m.get("source", "handbook")
    if source == "lecture":
        return ("lecture", m["lecture_title"])
    if source == "course_info":
        return ("course_info", "Kursinformation")
    if m.get("type_of_content") == "references":
        return None  # reference lists say nothing about the topic
    ch = m.get("chapter_number")
    if ch is None:
        return None
    return ("handbook", f"Kap. {ch}: {m.get('chapter_title', '')}")


def course_units(meta: list[dict]) -> dict[str, tuple[str, str]]:
    """chunk_id -> (source, unit label)."""
    out = {}
    for m in meta:
        unit = _chunk_unit(m)
        if unit:
            out[m["chunk_id"]] = unit
    return out


def _outcome(row) -> str:
    if row["outcome"]:
        return row["outcome"]
    if row["refused"] is not None:
        return "refused" if row["refused"] else "answered"
    return "refused" if looks_refused(row["answer"]) else "answered"


async def fetch_rows(db: aiosqlite.Connection, days: int | None):
    where, params = "", ()
    if days:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        where = "WHERE q.created_at >= ?"
        params = (since.strftime("%Y-%m-%d %H:%M:%S"),)
    return await db.execute_fetchall(
        f"""SELECT q.id, q.user_id, q.session_id, q.question, q.answer, q.created_at,
                   q.status, q.latency_ms, q.prompt_version, q.retrieval, q.top_score,
                   q.refused, a.theme, a.kind, a.outcome
            FROM query_logs q LEFT JOIN query_analysis a ON a.log_id = q.id
            {where}
            ORDER BY q.created_at""",
        params,
    )


def _week(ts: str) -> str:
    d = datetime.strptime(ts[:10], "%Y-%m-%d")
    return (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")


def compute_analytics(rows, meta: list[dict]) -> dict:
    units = course_units(meta)
    all_units = sorted(set(units.values()))

    ok_rows = [r for r in rows if (r["status"] or "ok") == "ok"]
    errors = len(rows) - len(ok_rows)

    def session(r):
        # Rows logged before session ids existed count as one session per account.
        return r["session_id"] or f"user:{r['user_id']}"

    # ── Per-question records ──
    items = []
    for r in ok_rows:
        items.append(
            {
                "id": r["id"],
                "q": r["question"],
                "ts": r["created_at"],
                "user": r["user_id"],
                "session": session(r),
                "theme": r["theme"],
                "kind": r["kind"] or "content",
                "outcome": _outcome(r),
                "tagged": r["theme"] is not None,
                "retrieval": json.loads(r["retrieval"]) if r["retrieval"] else None,
                "top_score": r["top_score"],
            }
        )
    content = [i for i in items if i["kind"] == "content"]

    # ── Summary ──
    oc = Counter(i["outcome"] for i in content)
    summary = {
        "questions": len(items),
        "content_questions": len(content),
        "users": len({i["user"] for i in items}),
        "sessions": len({i["session"] for i in items}),
        "answered": oc["answered"],
        "partial": oc["partial"],
        "refused": oc["refused"],
        "smalltalk": sum(i["kind"] == "smalltalk" for i in items),
        "admin": sum(i["kind"] == "admin" for i in items),
        "untagged": sum(not i["tagged"] for i in items),
        "errors": errors,
        "first": items[0]["ts"] if items else None,
        "last": items[-1]["ts"] if items else None,
    }

    # ── Weekly series (all kinds, by outcome) ──
    weekly = defaultdict(Counter)
    for i in content:
        weekly[_week(i["ts"])][i["outcome"]] += 1
    weeks = []
    if weekly:
        start = datetime.strptime(min(weekly), "%Y-%m-%d")
        end = datetime.strptime(max(weekly), "%Y-%m-%d")
        while start <= end:  # include empty weeks so gaps are visible
            k = start.strftime("%Y-%m-%d")
            weeks.append({"week": k, **{o: weekly[k][o] for o in OUTCOMES}})
            start += timedelta(days=7)

    # ── Themes ──
    by_theme = defaultdict(list)
    for i in items:
        if i["theme"]:
            by_theme[(i["theme"], i["kind"])].append(i)
    themes = []
    for (theme, kind), qs in by_theme.items():
        c = Counter(q["outcome"] for q in qs)
        # Course units the answered questions in this theme hit, for context.
        hit = Counter()
        for q in qs:
            for chunk_id, _ in (q["retrieval"] or [])[:COVERAGE_TOP_N]:
                if chunk_id in units:
                    hit[units[chunk_id][1]] += 1
        if c["refused"] and c["answered"] + c["partial"] == 0:
            verdict = "gap"  # never answered: probably not in the material
        elif c["refused"]:
            verdict = "inconsistent"  # sometimes answered: phrasing/retrieval issue
        elif c["partial"] > c["answered"]:
            verdict = "thin"  # mostly partial answers: material is thin
        else:
            verdict = "ok"
        themes.append(
            {
                "theme": theme,
                "kind": kind,
                "total": len(qs),
                "users": len({q["user"] for q in qs}),
                "sessions": len({q["session"] for q in qs}),
                "answered": c["answered"],
                "partial": c["partial"],
                "refused": c["refused"],
                "verdict": verdict,
                "last": max(q["ts"] for q in qs),
                "units": [u for u, _ in hit.most_common(3)],
                "questions": [
                    {"id": q["id"], "q": q["q"], "outcome": q["outcome"], "ts": q["ts"]}
                    for q in sorted(qs, key=lambda q: q["ts"], reverse=True)
                ],
            }
        )
    themes.sort(key=lambda t: (-t["total"], t["theme"]))

    # ── Coverage: which course units do (content) questions land in ──
    coverage = {u: Counter() for u in all_units}
    for i in content:
        seen = set()
        for chunk_id, _ in (i["retrieval"] or [])[:COVERAGE_TOP_N]:
            u = units.get(chunk_id)
            if u and u not in seen:
                seen.add(u)
                coverage[u][i["outcome"]] += 1
    coverage_out = [
        {
            "source": src,
            "unit": label,
            "total": sum(c.values()),
            **{o: c[o] for o in OUTCOMES},
        }
        for (src, label), c in coverage.items()
    ]

    def unit_sort(u):
        m = re.match(r"Kap\. (\d+)", u["unit"])
        return (u["source"], int(m.group(1)) if m else 0, u["unit"])

    coverage_out.sort(key=unit_sort)

    # ── Question list (no answers; fetched on demand) ──
    questions = [
        {
            "id": i["id"],
            "q": i["q"],
            "ts": i["ts"],
            "theme": i["theme"],
            "kind": i["kind"],
            "outcome": i["outcome"],
            "top_score": i["top_score"],
        }
        for i in reversed(items)
    ]

    return {
        "summary": summary,
        "weeks": weeks,
        "themes": themes,
        "coverage": coverage_out,
        "questions": questions,
    }
