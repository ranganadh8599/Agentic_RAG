"""Shared fixtures for the Agentic RAG test suite.

Hermeticity: the whole pytest session runs with MOCK models and deterministic
retrieval (query expansion / reranker / query rewrite off), so DB-backed tests
are fast and repeatable. Real-model RAG evaluation lives under
tests/evaluation/ and is run explicitly (``pytest -m evaluation``).
"""
import os
import sys
import tempfile
import uuid

import pytest

# Make the repo root importable even when pytest is invoked from a subdir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Hermetic session defaults. Set BEFORE app.core.config loads .env (config
# uses load_dotenv(override=False), so these win over .env).
os.environ.setdefault("EMBEDDING_MODEL", "mock")     # deterministic, offline
os.environ.setdefault("LLM_MODEL", "mock")
os.environ.setdefault("VISION_MODEL", "mock")
os.environ.setdefault("USE_QUERY_EXPANSION", "0")    # deterministic retrieval
os.environ.setdefault("USE_RERANKER", "0")           # never load the cross-encoder
os.environ.setdefault("USE_SPARSE_SEARCH", "0")      # sparse schema absent on CI box
os.environ.setdefault("USE_KEYWORD_SEARCH", "1")     # real FTS channel on
os.environ.setdefault("USE_ASYMMETRIC_PREFIX", "0")
os.environ.setdefault("USE_QUERY_REWRITE", "0")
os.environ.setdefault("GENERAL_STRONG_THRESHOLD", "0.1")
os.environ.setdefault("RELEVANCE_FLOOR", "-1.0")     # mock-embedding results always pass
os.environ.setdefault("METADATA_FILTER_MODE", "pre")
os.environ.setdefault("PBKDF2_ITERATIONS", "1000")   # keep auth hashing fast in tests


@pytest.fixture(scope="session")
def db_ready():
    """Initialize the Postgres schema once per session (idempotent).

    Requires the local Postgres + pgvector service to be running (see
    README / environment notes). Returns the `app.database.postgres` module.
    """
    import app.database.postgres as db
    db.init_db()
    return db


@pytest.fixture(scope="session")
def mongo_ready():
    """Ensure MongoDB indexes exist (no-op if Mongo is down)."""
    import app.database.mongo as mongo
    mongo.init_db()
    return mongo


@pytest.fixture(scope="session")
def client(db_ready, mongo_ready):
    """FastAPI TestClient for in-process API tests (DB + Mongo already init).

    Created without the context manager so the lifespan doesn't re-init the
    schema on every test; `db_ready`/`mongo_ready` handle setup.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def unique_collection(db_ready):
    """Create a throwaway collection and drop it (and its docs) afterwards."""
    import app.database.postgres as db
    name = f"test_{uuid.uuid4().hex[:12]}"
    coll_id = db.get_or_create_collection(name)
    yield name
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM documents WHERE collection_id = %s", (coll_id,))
        cur.execute("DELETE FROM collections WHERE id = %s", (coll_id,))


@pytest.fixture
def ingest_text(db_ready):
    """Ingest a tiny text file into a collection; return (doc_id, chunk_count)."""
    import app.ingestion.pipeline as ingest

    def _ingest(text, collection, title="sample.txt", user_id=None):
        fd, path = tempfile.mkstemp(suffix=".txt", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            doc_id, n, _info = ingest.ingest_file(
                path, title=title, collection=collection, skip_duplicates=True,
                user_id=user_id, ingested_by="test")
            return doc_id, n
        finally:
            os.unlink(path)
    return _ingest

