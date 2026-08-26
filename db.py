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
# True when the schema supports BM25-style sparse retrieval (pgvector sparsevec).
SPARSE_READY = False
# SQL type name used for embedding columns and for `<op> %s::<cast>` in queries.
# "vector" (HNSW cap 2000 dims) or "halfvec" (HNSW cap 4000 dims, fp16).
# The operator/opclass is chosen from settings.EMBEDDING_METRIC (cosine/dot/l2).
VEC_CAST = "vector"

# Similarity metric -> pgvector operators and HNSW opclass suffixes.
#   cosine: <=> (distance, 0 = identical)       -> _cosine_ops (any embeddings)
#   dot:    <#> (negated inner product)          -> _ip_ops     (normalized embs)
#   l2:     <-> (Euclidean distance)             -> _l2_ops     (raw/geometric)
_OP_BY_METRIC = {"cosine": "<=>", "dot": "<#>", "l2": "<->"}
_OPS_BY_METRIC = {"cosine": "_cosine_ops", "dot": "_ip_ops", "l2": "_l2_ops"}


def op_name() -> str:
    """pgvector operator used in ORDER BY (lower value = closer)."""
    return _OP_BY_METRIC.get(settings.EMBEDDING_METRIC, "<=>")


def hnsw_ops() -> str:
    """HNSW opclass suffix matching the configured metric."""
    return _OPS_BY_METRIC.get(settings.EMBEDDING_METRIC, "_cosine_ops")


def dist_expr(col: str, placeholder: str) -> str:
    """SQL fragment `col <op> placeholder` for ORDER BY (lower = closer)."""
    return f"{col} {op_name()} {placeholder}"


def score_expr(col: str, placeholder: str) -> str:
    """SQL fragment turning the raw distance into a higher-is-better score.
    cosine -> 1 - distance; dot -> inner product; l2 -> -distance."""
    if settings.EMBEDDING_METRIC == "cosine":
        return f"1 - ({dist_expr(col, placeholder)})"
    return f"-({dist_expr(col, placeholder)})"


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
            user_id     TEXT,
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
            embedding   {vec_type},
            content_hash TEXT
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
            user_id         TEXT,
            hit_count       INT DEFAULT 0,
            created_at      TIMESTAMPTZ DEFAULT now(),
            last_used_at    TIMESTAMPTZ DEFAULT now()
        );
        ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS sources JSONB DEFAULT '{{}}'::jsonb;
        ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS collection_id INT REFERENCES collections(id) ON DELETE CASCADE;
        ALTER TABLE semantic_cache ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE retrieval_cache ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection_id INT REFERENCES collections(id) ON DELETE SET NULL;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingested_by TEXT;
        -- Existing ownerless docs were all admin/CLI uploads: treat them as the
        -- shared corpus so normal users can still see them.
        UPDATE documents SET ingested_by = 'admin'
        WHERE user_id IS NULL AND ingested_by IS NULL;

        CREATE TABLE IF NOT EXISTS retrieval_cache (
            id              SERIAL PRIMARY KEY,
            query           TEXT NOT NULL,
            query_embedding {vec_type},
            results         JSONB NOT NULL DEFAULT '[]'::jsonb,
            best_score      DOUBLE PRECISION NOT NULL DEFAULT 0,
            hits            INT NOT NULL DEFAULT 1,
            collection_id   INT REFERENCES collections(id) ON DELETE CASCADE,
            user_id         TEXT,
            created_at      TIMESTAMPTZ DEFAULT now(),
            last_used_at    TIMESTAMPTZ DEFAULT now()
        );
        ALTER TABLE retrieval_cache ADD COLUMN IF NOT EXISTS results JSONB NOT NULL DEFAULT '[]'::jsonb;
        ALTER TABLE retrieval_cache ADD COLUMN IF NOT EXISTS best_score DOUBLE PRECISION NOT NULL DEFAULT 0;
    """)

    if USE_PGVECTOR:
        # Migrate columns created as vector(N) -> halfvec(N) when dims > 2000,
        # otherwise HNSW indexing is impossible (vector HNSW caps at 2000 dims).
        for tbl, col in [("chunks", "embedding"),
                         ("semantic_cache", "query_embedding"),
                         ("retrieval_cache", "query_embedding")]:
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
                         ("semantic_cache", "query_embedding"),
                         ("retrieval_cache", "query_embedding")]:
            try:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{tbl}_{col} "
                    f"ON {tbl} USING hnsw ({col} {VEC_CAST}{hnsw_ops()})"
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

    # Exact-query dedup for the retrieval cache (per collection + user; a
    # user_id of NULL means anonymous/public — those all share one row).
    try:
        conn.execute("DROP INDEX IF EXISTS uq_retrieval_cache_query")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_retrieval_cache_query "
            "ON retrieval_cache (COALESCE(collection_id, 0), COALESCE(user_id, ''), lower(query))"
        )
    except psycopg.Error:
        pass

    # Delta-update lookups: find chunks of a document by content hash.
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash "
            "ON chunks (document_id, content_hash)"
        )
    except psycopg.Error:
        pass

    # Metadata filtering: user_id on documents + GIN over chunk tags.
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id)")
    except psycopg.Error:
        pass
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin "
            "ON chunks USING gin (metadata jsonb_path_ops)"
        )
    except psycopg.Error:
        pass

    # --- BM25 sparse retrieval (pgvector sparsevec, pgvector >= 0.7) ---------
    # Optional: disabled gracefully when sparsevec is unavailable.
    global SPARSE_READY
    try:
        conn.execute(
            """ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sparse_embedding sparsevec;
               ALTER TABLE chunks ADD COLUMN IF NOT EXISTS token_count INT;
               CREATE SEQUENCE IF NOT EXISTS sparse_vocab_idx_seq;
               CREATE TABLE IF NOT EXISTS sparse_vocab (
                   term text PRIMARY KEY,
                   idx int NOT NULL UNIQUE DEFAULT nextval('sparse_vocab_idx_seq')
               );
               CREATE TABLE IF NOT EXISTS sparse_term_stats (
                   term text PRIMARY KEY, df int NOT NULL DEFAULT 0
               );
               CREATE TABLE IF NOT EXISTS sparse_corpus_stats (
                   id int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                   chunk_count int NOT NULL DEFAULT 0,
                   total_tokens bigint NOT NULL DEFAULT 0
               );
               INSERT INTO sparse_corpus_stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
               CREATE INDEX IF NOT EXISTS idx_chunks_sparse
                   ON chunks USING hnsw (sparse_embedding sparsevec_ip_ops);
            """
        )
        SPARSE_READY = True
    except psycopg.Error:
        SPARSE_READY = False

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


def similarity(a, b) -> float:
    """Score for the configured metric (higher = more similar).
    cosine -> [-1,1]; dot -> inner product; l2 -> -Euclidean distance."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    if settings.EMBEDDING_METRIC == "l2":
        return -float(np.linalg.norm(a - b))
    if settings.EMBEDDING_METRIC == "dot":
        return float(np.dot(a, b))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def to_json(obj) -> str:
    return json.dumps(obj)
