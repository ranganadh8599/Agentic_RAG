# Agentic RAG - collection management endpoints.

from fastapi import APIRouter, Query
from pydantic import BaseModel

import app.database.postgres as db
from app.api.dependencies import page_params

router = APIRouter()


class CollectionRequest(BaseModel):
    name: str
    description: str | None = None


@router.get("/collections")
def collections(limit: int | None = Query(None, ge=1),
                offset: int | None = Query(None, ge=0)):
    """List all collections with doc/chunk counts (optionally paginated)."""
    page_limit, page_offset = page_params(limit, offset)
    return db.list_collections(page_limit, page_offset)


@router.post("/collections")
def create_collection(req: CollectionRequest):
    """Create a collection (no-op if it already exists)."""
    existing = db.get_collection_id(req.name)
    if existing:
        return {"id": existing, "name": req.name, "created": False}
    cid = db.get_or_create_collection(req.name)
    return {"id": cid, "name": req.name, "created": True}
