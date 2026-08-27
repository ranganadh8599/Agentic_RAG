# Agentic RAG - chat history endpoints (per-user, ownership enforced).

from fastapi import APIRouter, HTTPException, Query, Request

import app.memory.conversation as memory
from app.api.dependencies import page_params, require_user

router = APIRouter()


@router.get("/conversations")
def conversations(request: Request,
                  limit: int | None = Query(None, ge=1),
                  offset: int | None = Query(None, ge=0)):
    user = require_user(request)
    page_limit, page_offset = page_params(limit, offset)
    return memory.get_conversations(user["id"], page_limit, page_offset)


@router.get("/conversations/{cid}")
def conversation_detail(cid: str, request: Request):
    user = require_user(request)
    msgs = memory.get_conversation_messages(cid, user["id"])
    if msgs is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if msgs is False:
        raise HTTPException(status_code=403, detail="this conversation belongs to another user")
    return {"id": cid, "messages": msgs}


@router.delete("/conversations/{cid}")
def conversation_delete(cid: str, request: Request):
    user = require_user(request)
    res = memory.delete_conversation(cid, user["id"])
    if res == "not_found":
        raise HTTPException(status_code=404, detail="conversation not found")
    if res == "forbidden":
        raise HTTPException(status_code=403, detail="this conversation belongs to another user")
    return {"deleted": cid}
