# Agentic RAG - PostgreSQL storage layer.
# Uses pgvector when the extension is available, otherwise falls back to
# storing embeddings as JSONB and computing cosine similarity in Python.

import json

import numpy as np
import psycopg
from psycopg.rows import dict_row

from config import settings

USE_PGVECTOR = False
DB_READY = False
# SQL type name used for embedding columns and for `<=> %s::<cast>` in queries.
# "vector" (HNSW cap 2000 dims) or "halfvec" (HNSW cap 4000 dims, fp16).
VEC_CAST = "vector"


def get_conn():
    """Open a fresh connection (autocommit, dict rows)."""
    conn = psycopg.connect(settings.DATABASE_URL, row_factory=dict_row)
    conn.autocommit = True
    return conn


def init_db() -> bool:
    """Create the schema. Returns True if pgvector is in use, False for JSONB fallback."""
    global USE_PGVECTOR, DB_READY, VEC_CAST
    conn = get_conn()
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        USE_PGVECTOR = True
    except psycopg.Error:
        USE_PGVECTOR = False

    # pgvector's HNSW index supports at most 2000 dims for `vector`. Embedding
    # models like gemini-embedding-2 are 3072-dim, so use `halfvec` there
    # (HNSW cap 4000 dims; half-precision is fine for cosine similarity).
    VEC_CAST = ("halfvec" if USE_PGVECTOR
                and settings.EMBEDDING_DIM > settings.HNSW_VECTOR_DIM_LIMIT else "vector")
    vec_type = f"{VEC_CAST}({settings.EMBEDDING_DIM})" if USE_PGVECTOR else "jsonb"

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS collections (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS documents (
            id          SERIAL PRIMARY KEY,
            collection_id INT REFERENCES collections(id) ON DELETE SET NULL,
            title       TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_path TEXT,
            metadata    JSONB DEFAULT '{{}}'::jsonb,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id          SERIAL PRIMARY KEY,
            document_id INT REFERENCES documents(id) ON DELETE CASCADE,
            content     TEXT NOT NULL,
            chunk_index INT,
            language    TEXT,
            metadata    JSONB DEFAULT '{{}}'::jsonb,
            embedding   {vec_type}
        );
        CREATE TABLE IF NOT EXISTS images (
            id          SERIAL PRIMARY KEY,
            document_id INT REFERENCES documents(id) ON DELETE CASCADE,
            page        INT,
            mime_type   TEXT NOT NULL DEFAULT 'image/jpeg',
            data        BYTEA NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id              SERIAL PRIMARY KEY,
            query_embedding {vec_type},
            query           TEXT NOT NULL,
            response        TEXT NOT NULL,
            model           TEXT,
            sources         JSONB DEFAULT '{{}}'::jsonb,
            collection_id   INT REFERENCES collections(id) ON DELETE CASCADE,
            hit_count       INT DEFAULT 0,
            created_at      TIMESTAMPTZ DEFAULT now(),
            last_used_at    TIMESTAMPTZ DEFAULT now()
        );
        ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS sources JSONB DEFAULT '{{}}'::jsonb;
        ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS collection_id INT REFERENCES collections(id) ON DELETE CASCADE;
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection_id INT REFERENCES collections(id) ON DELETE SET NULL;
    """)

    if USE_PGVECTOR:
        # Migrate columns created as vector(N) -> halfvec(N) when dims > 2000,
        # otherwise HNSW indexing is impossible (vector HNSW caps at 2000 dims).
        for tbl, col in [("chunks", "embedding"),
                         ("semantic_cache", "query_embedding")]:
            try:
                row = conn.execute(
                    f"SELECT data_type FROM information_schema.columns "
                    f"WHERE table_name='{tbl}' AND column_name='{col}'").fetchone()
                if row and row["data_type"] != VEC_CAST:
                    conn.execute(
                        f"ALTER TABLE {tbl} ALTER COLUMN {col} TYPE {vec_type} "
                        f"USING {col}::{vec_type}")
            except psycopg.Error:
                pass

        # HNSW indexes for fast approximate cosine search.
        for tbl, col in [("chunks", "embedding"),
                         ("semantic_cache", "query_embedding")]:
            try:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{tbl}_{col} "
                    f"ON {tbl} USING hnsw ({col} {VEC_CAST}_cosine_ops)"
                )
            except psycopg.Error:
                pass

    # Ensure a 'default' collection exists and backfill legacy documents.
    try:
        default_id = get_or_create_collection("default")
        conn.execute("UPDATE documents SET collection_id = %s WHERE collection_id IS NULL",
                     (default_id,))
    except psycopg.Error:
        pass

    # Atomic duplicate guard: unique per (collection, lower(title)). Concurrent
    # uploads of the same file can otherwise both pass the SELECT check before
    # either INSERT commits (check-then-act race) and create duplicate rows.
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_title_collection "
            "ON documents (collection_id, lower(title))"
        )
    except psycopg.Error:
        pass

    DB_READY = True
    return USE_PGVECTOR


# ---------------------------------------------------------------------------
# Collections (document namespaces)
# ---------------------------------------------------------------------------

def get_collection_id(name: str | None) -> int | None:
    """Look up a collection by name (returns None if it doesn't exist)."""
    if not name:
        return None
    with get_conn().cursor() as cur:
        cur.execute("SELECT id FROM collections WHERE name = %s", (name,))
        row = cur.fetchone()
        return row["id"] if row else None


def get_or_create_collection(name: str) -> int:
    """Get a collection id by name, creating it if it doesn't exist."""
    with get_conn().cursor() as cur:
        cur.execute("SELECT id FROM collections WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur.execute("INSERT INTO collections (name) VALUES (%s) RETURNING id", (name,))
        return cur.fetchone()["id"]


def list_collections() -> list[dict]:
    """List collections with document/chunk counts."""
    with get_conn().cursor() as cur:
        cur.execute(
            """SELECT c.id, c.name, c.description, c.created_at,
                      (SELECT count(*) FROM documents d WHERE d.collection_id = c.id) AS docs,
                      (SELECT count(*) FROM chunks ch JOIN documents d ON d.id = ch.document_id
                        WHERE d.collection_id = c.id) AS chunks
               FROM collections c ORDER BY c.name"""
        )
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def to_db_vec(embedding) -> str:
    """Convert a python list of floats into a value the DB column accepts."""
    if embedding is None:
        return None
    if USE_PGVECTOR:
        # pgvector accepts the literal form '[0.1,0.2,...]' directly.
        return "[" + ",".join(f"{float(x):.6f}" for x in embedding) + "]"
    return json.dumps([float(x) for x in embedding])


def from_db_vec(raw):
    """Convert a stored embedding into a numpy array."""
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw.astype("float32")
    if isinstance(raw, str):
        return np.array(json.loads(raw), dtype="float32")
    return np.array(raw, dtype="float32")


def cosine(a, b) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def to_json(obj) -> str:
    return json.dumps(obj)
