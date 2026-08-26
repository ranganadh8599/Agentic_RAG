# Agentic RAG - retrieval & semantic cache (stored in Postgres instead of Redis).
#
# Two caches:
#   * semantic_cache     - full LLM answers for near-identical queries
#   * retrieval_cache    - reranked chunk lists for popular queries
# Both are scoped per user (NULL = shared/global bucket) so per-user private
# documents can never leak across accounts via a cache hit.

import logging

import db
from app.core.config import settings

log = logging.getLogger("retrieval")


def _as_scopes(user_id):
    """Normalize a cache scope into a list of buckets to check:
    None -> [None] (shared/global), a string -> [that id], a list -> itself."""
    if user_id is None:
        return [None]
    if isinstance(user_id, (list, tuple)):
        return list(user_id)
    return [user_id]


def _scope_clause(scopes):
    """Build a WHERE fragment matching any of the given cache scopes. None in
    the list means the shared/global bucket (user_id IS NULL)."""
    ids = [s for s in scopes if s is not None]
    if ids and None in scopes:
        return "(user_id = ANY(%s) OR user_id IS NULL)", [ids]
    if ids:
        return "user_id = ANY(%s)", [ids]
    return "user_id IS NULL", []


def _results_shared_safe(results, user_id):
    """True when storing results under the given cache bucket cannot leak a
    private document. The global bucket (user_id=None) is shared by every
    user, so it may only hold results with no owned (user_id != NULL) docs;
    per-user buckets are private to their owner, so always safe."""
    if user_id is not None or not results:
        return True
    doc_ids = [r.get("doc_id") for r in results if r.get("doc_id")]
    if not doc_ids:
        return True
    try:
        with db.get_conn().cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM documents "
                "WHERE id = ANY(%s) AND user_id IS NOT NULL",
                (doc_ids,))
            return cur.fetchone()["n"] == 0
    except Exception:  # noqa: BLE001
        return False


def semantic_cache_lookup(query_emb, collection_id: int | None = None, user_id=None):
    if not db.USE_PGVECTOR:
        return None
    _ph = f"%s::{db.VEC_CAST}"
    scope_sql, scope_params = _scope_clause(_as_scopes(user_id))
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT id, response, sources, {db.score_expr('query_embedding', _ph)} AS sim
               FROM semantic_cache
               WHERE collection_id IS NOT DISTINCT FROM %s
                 AND {scope_sql}
               ORDER BY {db.dist_expr('query_embedding', _ph)}
               LIMIT 1""",
            (db.to_db_vec(query_emb), collection_id, *scope_params, db.to_db_vec(query_emb)),
        )
        row = cur.fetchone()
    if row and row["sim"] is not None and row["sim"] >= settings.SEMANTIC_CACHE_THRESHOLD:
        return row
    return None


def semantic_cache_store(query: str, query_emb, response: str, model: str, sources=None,
                         collection_id: int | None = None, user_id: str | None = None):
    try:
        with db.get_conn().cursor() as cur:
            cur.execute(
                "INSERT INTO semantic_cache (query_embedding, query, response, model, sources, collection_id, user_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (db.to_db_vec(query_emb), query, response, model,
                 db.to_json(sources) if sources is not None else "{}", collection_id, user_id),
            )
    except Exception:  # noqa: BLE001
        pass


def retrieval_cache_lookup(query_emb, collection_id: int | None = None, user_id=None):
    """Return cached reranked results for a near-identical popular query.

    Serves the exact reranked chunk list, so repeat questions skip hybrid
    search + RRF + cross-encoder entirely (the LLM answer is still generated
    fresh from those chunks downstream). Scoped per user (NULL = anonymous).
    """
    if not db.USE_PGVECTOR or not settings.RETRIEVAL_CACHE_ENABLED:
        return None
    _ph = f"%s::{db.VEC_CAST}"
    scope_sql, scope_params = _scope_clause(_as_scopes(user_id))
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT id, results, best_score,
                       {db.score_expr('query_embedding', _ph)} AS sim
               FROM retrieval_cache
               WHERE collection_id IS NOT DISTINCT FROM %s
                 AND {scope_sql}
                 AND {db.score_expr('query_embedding', _ph)} >= %s
               ORDER BY {db.dist_expr('query_embedding', _ph)}
               LIMIT 1""",
            (db.to_db_vec(query_emb), collection_id, *scope_params, db.to_db_vec(query_emb),
             settings.RETRIEVAL_CACHE_THRESHOLD, db.to_db_vec(query_emb)),
        )
        row = cur.fetchone()
    if not row:
        return None
    try:  # popularity bump (best effort)
        with db.get_conn().cursor() as cur:
            cur.execute(
                "UPDATE retrieval_cache SET hits = hits + 1, last_used_at = now() WHERE id = %s",
                (row["id"],),
            )
    except Exception:  # noqa: BLE001
        pass
    return {"results": row["results"] or [],
            "best_score": float(row["best_score"] or 0.0)}


def retrieval_cache_store(query: str, query_emb, results, best_score: float,
                          collection_id: int | None = None, user_id: str | None = None):
    """Upsert reranked results for a query; bump hits when the exact query repeats.

    Unique per (collection, user, lower(query)) — user_id NULL is the anonymous/
    public bucket."""
    if not settings.RETRIEVAL_CACHE_ENABLED:
        return
    try:
        with db.get_conn().cursor() as cur:
            cur.execute(
                """INSERT INTO retrieval_cache (query, query_embedding, results, best_score, collection_id, user_id)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (COALESCE(collection_id, 0), COALESCE(user_id, ''), lower(query))
                   DO UPDATE SET query_embedding = EXCLUDED.query_embedding,
                                 results = EXCLUDED.results,
                                 best_score = EXCLUDED.best_score,
                                 hits = retrieval_cache.hits + 1,
                                 last_used_at = now()""",
                (query, db.to_db_vec(query_emb), db.to_json(results),
                 best_score, collection_id, user_id),
            )
    except Exception:  # noqa: BLE001
        return
    _trim_retrieval_cache()


def _trim_retrieval_cache():
    """Keep retrieval_cache under RETRIEVAL_CACHE_MAX_ENTRIES: evict the least
    popular / least recently used entries (only when over capacity)."""
    try:
        with db.get_conn().cursor() as cur:
            n = cur.execute("SELECT count(*) AS n FROM retrieval_cache").fetchone()["n"]
            if n > settings.RETRIEVAL_CACHE_MAX_ENTRIES:
                cur.execute(
                    """DELETE FROM retrieval_cache
                       WHERE id IN (
                           SELECT id FROM retrieval_cache
                           ORDER BY hits ASC, last_used_at ASC
                           LIMIT %s
                       )""",
                    (n - settings.RETRIEVAL_CACHE_MAX_ENTRIES,),
                )
    except Exception:  # noqa: BLE001
        pass


def clear_retrieval_cache():
    """Drop all cached retrieval results. Call after re-ingesting documents so
    cached chunk lists never point at stale/removed chunks."""
    try:
        with db.get_conn().cursor() as cur:
            cur.execute("DELETE FROM retrieval_cache")
    except Exception:  # noqa: BLE001
        pass
