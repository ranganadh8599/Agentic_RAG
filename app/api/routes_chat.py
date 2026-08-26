# Agentic RAG - OpenAI-compatible chat endpoint (stream + non-stream).
#
#   POST /v1/chat/completions  OpenAI-compatible (stream + non-stream)

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import app.database.mongo as mongo
import app.memory.conversation as memory
from app.api.dependencies import bearer_token
from app.core.config import settings
from app.schemas.chat import ChatRequest

log = logging.getLogger("api")

router = APIRouter()


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
        elif ev["type"] == "replace":
            # Client signal: drop everything streamed so far (the Critic loop
            # is replacing the flawed first draft, not appending to it).
            payload = {"id": "chatcmpl-agenticrag", "object": "chat.completion.chunk",
                       "model": model, "choices": [{"index": 0, "delta": {},
                       "finish_reason": None}], "replace": True}
            yield f"data: {json.dumps(payload)}\n\n"
        elif ev["type"] == "unverified":
            # Client signal: the Critic agent could not run, so the answer was
            # delivered without grounding verification.
            payload = {"id": "chatcmpl-agenticrag", "object": "chat.completion.chunk",
                       "model": model, "choices": [{"index": 0, "delta": {},
                       "finish_reason": None}], "unverified": True}
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


@router.post("/v1/chat/completions")
def chat_endpoint(req: ChatRequest, request: Request):
    query = (req.messages[-1].get("content") if req.messages else "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="message content is required")
    model = req.model or settings.LLM_MODEL
    # Chat auth is optional: a valid bearer token scopes the chat to that user;
    # without one the chat still works but is not saved to any user's history.
    user = mongo.user_from_token(bearer_token(request))
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
        # Documented behavior (Option A): conversation ownership is tied to an
        # authenticated user. Anonymous conversations have user_id=NULL, so
        # conversation_owner returns None and they are NOT resumable by id
        # (404). Resume requires an authenticated session.
        if owner is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        if user and owner and owner != user["id"]:
            raise HTTPException(status_code=403, detail="this conversation belongs to another user")
        conv = req.conversation_id
    else:
        conv = memory.create_conversation((query or "New chat")[:60],
                                          user_id=user["id"] if user else None)

    orchestrator = request.app.state.orchestrator
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
        "unverified": res.get("unverified", False),
        "conversation_id": conv,
        "collection": req.collection,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
