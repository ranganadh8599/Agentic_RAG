# Agentic RAG - FastAPI application entry point.
#
# Assembles the app: logging, lifespan (DB init), CORS, the static web UI, and
# the route routers (which live under app/api/routes_*.py).
#
# Run:  uvicorn app.main:app --reload --port 8000

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.orchestrator import OrchestratorAgent
from app.api import (routes_auth, routes_chat, routes_collections,
                     routes_conversations, routes_documents, routes_health)
from app.core.config import settings
from app.core.logging import setup_logging
import app.database.mongo as mongo
import app.database.postgres as db

log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Create the schema (idempotent) + Mongo indexes. This lives in the
    # lifespan because newer FastAPI/Starlette no longer fire
    # @app.on_event("startup") when a lifespan is provided.
    db.init_db()
    mongo.init_db()  # safe no-op if MongoDB is not running
    log.info("🚀 Server started")
    yield
    log.info("🛑 Server stopped")
    db.close_pool()


app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
# Shared orchestrator instance; the chat router reads it via request.app.state.
app.state.orchestrator = OrchestratorAgent()

# CORS: the web UI is served same-origin, so no CORS headers are needed for it.
# When a separate frontend or a browser-based OpenAI-compatible client calls the
# API, allow the configured origin(s) (CORS_ORIGINS, comma-separated, default
# "*"). Auth uses bearer tokens (not cookies), so allow_credentials stays off
# and a browser will never auto-attach credentials to a cross-origin request.
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )


@app.middleware("http")
async def security_headers(request, call_next):
    """Set hardening headers on every response (OWASP System Configuration).

    The Content-Security-Policy is the main XSS defense-in-depth: it keeps the
    marked CDN script but blocks inline scripts, and forbids object/embed/frame
    embedding. script-src is restricted to self + the marked CDN."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    )
    return response

# Serve the web UI (static/ sits at the repository root, one level up from
# this package).
STATIC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static"))
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Route routers (paths are distinct, so order doesn't matter).
app.include_router(routes_health.router)
app.include_router(routes_documents.router)
app.include_router(routes_collections.router)
app.include_router(routes_auth.router)
app.include_router(routes_conversations.router)
app.include_router(routes_chat.router)
