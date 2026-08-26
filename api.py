# Agentic RAG - FastAPI server.
#   * POST /v1/chat/completions  OpenAI-compatible (stream + non-stream)
#   * POST /ingest               upload a file/dir archive
#   * GET  /health, /stats
#
# Run:  uvicorn api:app --reload --port 8000

import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import logging_config
from agents import OrchestratorAgent
from config import settings
import db
import ingest
import memory
import mongo

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_config.setup_logging()
    # Create the schema (idempotent) + Mongo indexes. This lives in the
    # lifespan because newer FastAPI/Starlette no longer fire
    # @app.on_event("startup") when a lifespan is provided.
    db.init_db()
    mongo.init_db()  # safe no-op if MongoDB is not running
    log.info("🚀 Server started")
    yield
    log.info("🛑 Server stopped")


app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
orchestrator = OrchestratorAgent()

# Serve the web UI.
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def _count(table: str) -> int:
    with db.get_conn().cursor() as cur:
        cur.execute(f"SELECT count(*) AS n FROM {table}")
        return cur.fetchone()["n"]


# ---------------------------------------------------------------------------
# Health / stats
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pgvector": db.USE_PGVECTOR,
        "documents": _count("documents"),
        "chunks": _count("chunks"),
        "conversations": mongo.count_conversations(),
        "messages": mongo.count_messages(),
    }


@app.get("/api/config")
def app_config():
    """Client-facing upload settings (used by the web UI)."""
    return {
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "max_upload_files": settings.MAX_UPLOAD_FILES,
    }


@app.get("/documents")
def documents(collection: str | None = None):
    """List ingested documents with chunk counts (for the UI sidebar).
    Optionally filter to a single collection."""
    coll_filter = ""
    params: list = []
    if collection:
        cid = db.get_collection_id(collection)
        if cid is None:
            return []
        coll_filter = "WHERE d.collection_id = %s"
        params = [cid]
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT d.id, d.title, d.source_type, d.source_path, d.created_at,
                      c2.name AS collection, count(c.id) AS chunks
               FROM documents d
               LEFT JOIN chunks c ON c.document_id = d.id
               LEFT JOIN collections c2 ON c2.id = d.collection_id
               {coll_filter}
               GROUP BY d.id, c2.name ORDER BY d.id DESC""",
            params,
        )
        return cur.fetchall()


@app.get("/collections")
def collections():
    """List all collections with doc/chunk counts."""
    return db.list_collections()


class CollectionRequest(BaseModel):
    name: str
    description: str | None = None


@app.post("/collections")
def create_collection(req: CollectionRequest):
    """Create a collection (no-op if it already exists)."""
    existing = db.get_collection_id(req.name)
    if existing:
        return {"id": existing, "name": req.name, "created": False}
    cid = db.get_or_create_collection(req.name)
    return {"id": cid, "name": req.name, "created": True}


@app.get("/images/{image_id}")
def get_image(image_id: int):
    """Serve a stored image by id (used by the UI to show source images)."""
    with db.get_conn().cursor() as cur:
        cur.execute("SELECT data, mime_type FROM images WHERE id = %s", (image_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=bytes(row["data"]), media_type=row["mime_type"])


# ---------------------------------------------------------------------------
# Ingestion (with live progress)
# ---------------------------------------------------------------------------

# In-memory upload progress trackers keyed by client-supplied upload_id.
# Written from the ingest threadpool thread, read by the progress poll endpoint
# on the event loop thread — plain dict get/set is fine under the GIL here.
UPLOAD_PROGRESS: dict[str, dict] = {}


def _progress_state(upload_id: str) -> dict:
    # Bounded: never keep more than a few hundred finished trackers around.
    if len(UPLOAD_PROGRESS) > 300:
        stale = [k for k, v in UPLOAD_PROGRESS.items()
                 if v.get("status") in ("done", "error")][:len(UPLOAD_PROGRESS) - 50]
        for k in stale:
            UPLOAD_PROGRESS.pop(k, None)
    return UPLOAD_PROGRESS.get(upload_id, {
        "percent": 0, "phase": "starting", "message": "Starting…", "status": "running",
    })


@app.post("/ingest")
def ingest_endpoint(file: UploadFile = File(...), title: str | None = Form(None),
                    collection: str = Form("default"),
                    upload_id: str | None = Query(None),
                    update: bool = Query(False, description="delta-update an existing "
                        "doc: reuse unchanged chunks, embed only changed ones"),
                    user_id: str | None = Form(None, description="owner of the doc "
                        "(used by metadata filtering)"),
                    request: Request = None):
    # Ownership: a logged-in non-admin's uploads are owned by them — the form
    # user_id is ignored so a user can't tag a doc as someone else's. Admins
    # and anonymous callers keep the explicit user_id (admin may tag for any
    # user; anonymous leaves the doc public).
    owner = mongo.user_from_token(_bearer(request)) if request is not None else None
    if owner and not owner.get("is_admin"):
        user_id = owner["id"]
    # Record WHO ingested the doc: an authenticated admin/user gets their id,
    # anonymous stays NULL. Ownerless docs with ingested_by set (admin/CLI) are
    # the shared corpus every normal user can see; anonymous uploads (both
    # NULL) stay hidden from normal users.
    ingested_by = owner["id"] if owner else None
    # Use the real uploaded filename as the document title (not the temp name).
    real_name = file.filename or "upload"
    suffix = os.path.splitext(real_name)[1] or ".txt"
    uid = upload_id or uuid.uuid4().hex
    if upload_id:
        UPLOAD_PROGRESS[uid] = {"percent": 5, "phase": "uploading",
                                "message": f"Uploading {real_name}…", "status": "running",
                                "file": real_name}

    # Enforce the per-file size limit BEFORE reading the body into memory.
    if settings.MAX_UPLOAD_MB:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        if size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"'{real_name}' is {size / (1024 * 1024):.1f} MB — exceeds the "
                       f"{settings.MAX_UPLOAD_MB} MB upload limit",
            )

    file.file.seek(0)
    data = file.file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        def on_progress(p: dict):
            if upload_id:
                UPLOAD_PROGRESS[uid] = {**p, "status": "running", "file": real_name}

        doc_id, n, info = ingest.ingest_file(tmp_path, title=title or real_name,
                                             collection=collection or "default",
                                             on_progress=on_progress,
                                             update_existing=update,
                                             user_id=user_id,
                                             ingested_by=ingested_by)
        # info["mode"] distinguishes: "ingested" (fresh), "updated" (delta
        # update — only changed chunks embedded), "skipped" (duplicate name),
        # "empty" (no extractable text).
        mode = info.get("mode", "ingested")
        resp = {"document_id": doc_id, "chunks": n, "filename": real_name,
                "collection": collection or "default",
                "skipped": mode == "skipped",
                "updated": mode == "updated"}
        if mode == "updated":
            resp["reused"] = info.get("reused", 0)
            resp["removed"] = info.get("removed", 0)
        if mode == "empty":
            resp["note"] = "no extractable content"
        if upload_id:
            UPLOAD_PROGRESS[uid] = {"percent": 100, "phase": "done",
                                    "message": "Done", "status": "done",
                                    "file": real_name, "result": resp}
        return resp
    except ValueError as exc:  # unsupported file type
        if upload_id:
            UPLOAD_PROGRESS[uid] = {"percent": 100, "phase": "error",
                                    "message": str(exc), "status": "error", "file": real_name}
        raise HTTPException(status_code=415, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # malformed/corrupt file and other failures
        if upload_id:
            UPLOAD_PROGRESS[uid] = {"percent": 100, "phase": "error",
                                    "message": str(exc), "status": "error", "file": real_name}
        raise HTTPException(status_code=400, detail=f"failed to ingest '{real_name}': {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/ingest/progress/{upload_id}")
def ingest_progress(upload_id: str):
    return _progress_state(upload_id)


# ---------------------------------------------------------------------------
# Chat (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    conversation_id: str | None = None
    collection: str | None = None
    # Optional metadata filter: {user_id, date_from, date_to, tags, tags_mode}.
    filters: dict | None = None


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ---------------------------------------------------------------------------
# Auth (basic username + password, bearer session tokens)
# ---------------------------------------------------------------------------

def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _auth_user(request: Request) -> dict:
    """Resolve the authenticated user from the token (401 if missing/invalid)."""
    user = mongo.user_from_token(_bearer(request))
    if not user:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


@app.post("/api/register")
def register(req: RegisterRequest):
    try:
        return mongo.register_user(req.username, req.display_name, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/login")
def login(req: LoginRequest):
    res = mongo.login_user(req.username, req.password)
    if not res:
        raise HTTPException(status_code=401, detail="invalid username or password")
    return res


@app.post("/api/logout")
def logout(request: Request):
    mongo.logout(_bearer(request))
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    return _auth_user(request)


@app.post("/api/password")
def change_password(req: ChangePasswordRequest, request: Request):
    """Change the signed-in user's password (verifies the current one first)."""
    user = _auth_user(request)  # 401 if not authenticated
    res = mongo.change_password(user["id"], req.current_password, req.new_password,
                                keep_token=_bearer(request))
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "could not change password"))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat history (per-user, ownership enforced)
# ---------------------------------------------------------------------------

@app.get("/conversations")
def conversations(request: Request):
    user = _auth_user(request)
    return memory.get_conversations(user["id"])


@app.get("/conversations/{cid}")
def conversation_detail(cid: str, request: Request):
    user = _auth_user(request)
    msgs = memory.get_conversation_messages(cid, user["id"])
    if msgs is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if msgs is False:
        raise HTTPException(status_code=403, detail="this conversation belongs to another user")
    return {"id": cid, "messages": msgs}


@app.delete("/conversations/{cid}")
def conversation_delete(cid: str, request: Request):
    user = _auth_user(request)
    res = memory.delete_conversation(cid, user["id"])
    if res == "not_found":
        raise HTTPException(status_code=404, detail="conversation not found")
    if res == "forbidden":
        raise HTTPException(status_code=403, detail="this conversation belongs to another user")
    return {"deleted": cid}


def _tracked_stream(events, conv: str | None = None):
    """Yield orchestration events, logging when the stream has fully finished."""
    t0 = time.perf_counter()
    try:
        for ev in events:
            yield ev
    finally:
        log.info("✅ Stream finished | conv=%s | %.1fs", conv, time.perf_counter() - t0)


def _sse(model: str, events, conv: str | None = None):
    """Wrap orchestration events into OpenAI chat.completion.chunk SSE lines."""
    yield f'data: {json.dumps({"id": "chatcmpl-agenticrag", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'
    for ev in events:
        if ev["type"] == "content":
            payload = {"id": "chatcmpl-agenticrag", "object": "chat.completion.chunk",
                       "model": model, "choices": [{"index": 0,
                       "delta": {"content": ev["delta"]}, "finish_reason": None}]}
            yield f"data: {json.dumps(payload)}\n\n"
        elif ev["type"] == "status":
            payload = {"id": "chatcmpl-agenticrag", "object": "chat.completion.chunk",
                       "model": model, "choices": [{"index": 0, "delta": {},
                       "finish_reason": None}], "status": ev["status"]}
            yield f"data: {json.dumps(payload)}\n\n"
        elif ev["type"] == "sources":
            final = {"id": "chatcmpl-agenticrag", "object": "chat.completion.chunk",
                     "model": model, "choices": [{"index": 0, "delta": {},
                     "finish_reason": "stop"}], "sources": ev["sources"],
                     "conversation_id": conv}
            yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_endpoint(req: ChatRequest, request: Request):
    query = (req.messages[-1].get("content") if req.messages else "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="message content is required")
    model = req.model or settings.LLM_MODEL
    # Chat auth is optional: a valid bearer token scopes the chat to that user;
    # without one the chat still works but is not saved to any user's history.
    user = mongo.user_from_token(_bearer(request))
    # Cache scope: admins and anonymous users share the global/shared cache
    # bucket; regular users get their own per-user cache so their private
    # uploads and related cached answers never leak across accounts.
    cache_scope = None if (not user or user.get("is_admin")) else user["id"]
    # Retrieval scope: the shared corpus (admin/CLI-ingested, ownerless docs)
    # is visible to everyone. A logged-in non-admin additionally sees their
    # own uploads. Users' private uploads are visible to NOBODY else — not
    # even the admin.
    filters = dict(req.filters or {})
    if user and not user.get("is_admin"):
        filters["user_id"] = [None, user["id"]]
    else:
        filters["user_id"] = [None]
    log.info("▶️  User query | user=%s stream=%s collection=%s filters=%s | %r",
             (user or {}).get("username"), req.stream, req.collection,
             filters, query[:120])
    if req.conversation_id is not None:
        owner = memory.conversation_owner(req.conversation_id)
        if owner is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        if user and owner and owner != user["id"]:
            raise HTTPException(status_code=403, detail="this conversation belongs to another user")
        conv = req.conversation_id
    else:
        conv = memory.create_conversation((query or "New chat")[:60],
                                          user_id=user["id"] if user else None)

    if req.stream:
        events = orchestrator.run_stream(query, conversation_id=conv, collection=req.collection,
                                         filters=filters,
                                         user_id=cache_scope)
        log.info("… streaming response | conv=%s", conv)
        return StreamingResponse(_sse(model, _tracked_stream(events, conv), conv),
                                 media_type="text/event-stream")

    res = orchestrator.run(query, conversation_id=conv, collection=req.collection,
                           filters=filters,
                           user_id=cache_scope)
    log.info("✅ Response sent | type=%s sources=%d chars=%d",
             res.get("type"), len(res.get("sources") or []),
             len(res.get("answer") or ""))
    return {
        "id": "chatcmpl-agenticrag",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": res["answer"]},
                     "finish_reason": "stop"}],
        "sources": res["sources"],
        "type": res["type"],
        "conversation_id": conv,
        "collection": req.collection,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
