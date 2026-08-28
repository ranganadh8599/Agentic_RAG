# Agentic RAG - document management & ingestion endpoints.

import os
import tempfile
import uuid
from threading import Lock

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

import app.database.mongo as mongo
import app.database.postgres as db
import app.ingestion.pipeline as ingest
from app.api.dependencies import bearer_token, page_params
from app.core.config import settings

router = APIRouter()


def _visibility(user) -> tuple[list[str], list]:
    """SQL visibility clauses that mirror chat retrieval scoping.

    Anonymous / admin -> shared corpus only (ownerless docs ingested by an
                         admin/CLI: user_id IS NULL AND ingested_by IS NOT NULL).
    Regular user      -> shared corpus OR their own uploads.
    A user's private documents are NEVER visible to anonymous callers.

    Returns a single grouped clause (shared OR own) plus params, so callers can
    safely AND it with additional filters (collection, image id, ...)."""
    parts = ["(d.user_id IS NULL AND d.ingested_by IS NOT NULL)"]
    params: list = []
    if user and not user.get("is_admin"):
        parts.append("d.user_id = %s")
        params.append(user["id"])
    return [("(" + " OR ".join(parts) + ")")], params


@router.get("/documents")
def documents(request: Request, collection: str | None = None,
              limit: int | None = Query(None, ge=1),
              offset: int | None = Query(None, ge=0)):
    """List documents visible to the caller, with chunk counts.

    Visibility mirrors chat retrieval (see _visibility): a logged-in regular
    user sees the shared corpus + their own uploads; anonymous/admin see the
    shared corpus only. A user's private uploads never leak to other users or
    anonymous callers via this listing.
    """
    user = mongo.user_from_token(bearer_token(request))
    where, params = _visibility(user)
    if collection:
        cid = db.get_collection_id(collection)
        if cid is None:
            return []
        where.append("d.collection_id = %s")
        params.append(cid)
    page_limit, page_offset = page_params(limit, offset)
    sql = f"""SELECT d.id, d.title, d.source_type, d.source_path, d.created_at,
                    c2.name AS collection, count(c.id) AS chunks
             FROM documents d
             LEFT JOIN chunks c ON c.document_id = d.id
             LEFT JOIN collections c2 ON c2.id = d.collection_id
             WHERE {' AND '.join(where)}
             GROUP BY d.id, c2.name ORDER BY d.id DESC"""
    if page_limit is not None:
        sql += " LIMIT %s OFFSET %s"
        params += [page_limit, page_offset]
    with db.get_conn().cursor() as cur:
        cur.execute(sql, params or None)
        return cur.fetchall()


@router.get("/images/{image_id}")
def get_image(image_id: int, request: Request):
    """Serve a stored image by id — only if the owning document is visible to
    the caller (same scoping as /documents). Prevents IDOR on image ids."""
    user = mongo.user_from_token(bearer_token(request))
    where, vis_params = _visibility(user)
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT i.data, i.mime_type
                FROM images i JOIN documents d ON d.id = i.document_id
                WHERE i.id = %s AND ({' AND '.join(where)})""",
            (image_id, *vis_params),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=bytes(row["data"]), media_type=row["mime_type"])


# ---------------------------------------------------------------------------
# Ingestion (with live progress)
# ---------------------------------------------------------------------------

# In-memory upload progress trackers keyed by client-supplied upload_id.
# Written from the ingest threadpool thread, read by the progress poll endpoint
# on the event loop thread. A lock guards the read+prune compound operation;
# individual dict get/set is atomic under the GIL.
#
# LIMITATION: this is per-process state. Under `uvicorn --workers N` (or
# multiple replicas) the ingest may run in a different worker than the poll
# request, so progress can come back stale/empty. That is fine for the default
# single-worker deployment; for multi-worker setups, move progress to a shared
# store (e.g. Redis or a progress table in Postgres) and read from there.
UPLOAD_PROGRESS: dict[str, dict] = {}
_progress_lock = Lock()


def _progress_state(upload_id: str) -> dict:
    # Bounded: never keep more than a few hundred finished trackers around.
    with _progress_lock:
        if len(UPLOAD_PROGRESS) > 300:
            stale = [k for k, v in UPLOAD_PROGRESS.items()
                     if v.get("status") in ("done", "error")][:len(UPLOAD_PROGRESS) - 50]
            for k in stale:
                UPLOAD_PROGRESS.pop(k, None)
        return UPLOAD_PROGRESS.get(upload_id, {
            "percent": 0, "phase": "starting", "message": "Starting…", "status": "running",
        })


@router.post("/ingest")
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
    owner = mongo.user_from_token(bearer_token(request)) if request is not None else None
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


@router.get("/ingest/progress/{upload_id}")
def ingest_progress(upload_id: str):
    return _progress_state(upload_id)
