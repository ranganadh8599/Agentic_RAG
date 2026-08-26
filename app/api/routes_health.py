# Agentic RAG - health & client-facing config endpoints.

from fastapi import APIRouter

import db
import mongo
from app.api.dependencies import count_rows
from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "pgvector": db.USE_PGVECTOR,
        "documents": count_rows("documents"),
        "chunks": count_rows("chunks"),
        "conversations": mongo.count_conversations(),
        "messages": mongo.count_messages(),
    }


@router.get("/api/config")
def app_config():
    """Client-facing upload settings (used by the web UI)."""
    return {
        "max_upload_mb": settings.MAX_UPLOAD_MB,
        "max_upload_files": settings.MAX_UPLOAD_FILES,
    }
