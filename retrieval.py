# Agentic RAG - hybrid retrieval.
# Combines:
#   * semantic cache with a cosine threshold
#   * LLM query expansion + weighted reciprocal-rank fusion
#   * asymmetric query/document prefixes (when enabled)
#   * vector (pgvector) + keyword (Postgres full-text) hybrid search

import logging
import re
import time

import db
import rerank
import sparse
from functools import lru_cache

from config import settings
from llm import embed_texts, chat_text
from logging_config import fmt_table
from prompts import EXPANSION_PROMPT

log = logging.getLogger("retrieval")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=settings.QUERY_EMBED_CACHE_SIZE)
def embed_query(query: str):
    """Embed a query, cached so repeated or similar queries never re-embed
    the same text."""
    text = query
    if settings.USE_ASYMMETRIC_PREFIX:
        text = settings.QUERY_PREFIX + text
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Semantic cache (stored in Postgres instead of Redis)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Retrieval-results cache (popular queries -> cached reranked chunks)
# ---------------------------------------------------------------------------

def retrieval_cache_lookup(query_emb, collection_id: int | None = None,
                           user_id=None):
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


# ---------------------------------------------------------------------------
# Metadata filtering (pre-filter before ANN / post-filter trim)
# ---------------------------------------------------------------------------

def _norm_filters(filters):
    """Normalize a metadata filter dict -> dict or None.

    Supported keys:
      user_id             - documents owned by a user (documents.user_id)
      date_from / date_to - document ingest date range (documents.created_at)
      tags                - list[str]; chunk metadata tags
      tags_mode           - 'any' (default) | 'all'
    """
    if not filters:
        return None
    f = {}
    uid = filters.get("user_id")
    if isinstance(uid, (list, tuple)):
        # A list of allowed owners; None in the list = the shared/admin-ingested
        # (ownerless) docs a normal user is allowed to see.
        vals = [None if v in (None, "", "None") else str(v) for v in uid]
        if vals:
            f["user_id"] = vals
    elif uid:
        f["user_id"] = str(uid)
    dfrom = filters.get("date_from")
    if dfrom:
        f["date_from"] = str(dfrom)
    dto = filters.get("date_to")
    if dto:
        f["date_to"] = str(dto)
    tags = filters.get("tags")
    if tags:
        f["tags"] = [str(t) for t in tags]
        f["tags_mode"] = "all" if str(filters.get("tags_mode", "any")).lower() == "all" else "any"
    return f or None


def _filter_where(filters):
    """Build (where_clause, params) for PRE-filtering the chunks JOIN documents
    query before the ANN scan. user_id/date live on documents, tags on chunk
    metadata (JSONB containment)."""
    where, params = [], []
    if not filters:
        return "", []
    uid = filters.get("user_id")
    if isinstance(uid, (list, tuple)):
        parts = []
        allowed = [v for v in uid if v is not None]
        if None in uid:
            # Ownerless docs ingested by an admin/CLI = the shared corpus that
            # every normal user may retrieve.
            parts.append("(d.user_id IS NULL AND d.ingested_by IS NOT NULL)")
        if allowed:
            parts.append("d.user_id = ANY(%s)")
            params.append(allowed)
        if parts:
            where.append("(" + " OR ".join(parts) + ")")
    elif uid:
        where.append("d.user_id = %s")
        params.append(uid)
    dfrom = filters.get("date_from")
    if dfrom:
        where.append("d.created_at >= %s")
        params.append(dfrom)
    dto = filters.get("date_to")
    if dto:
        where.append("d.created_at <= %s")
        params.append(dto)
    tags = filters.get("tags")
    if tags:
        op = "?&" if filters.get("tags_mode") == "all" else "?|"
        where.append(f"c.metadata->'tags' {op} %s")
        params.append(tags)
    return (" AND " + " AND ".join(where)) if where else "", params


def _is_post_filter(filters) -> bool:
    """True when filters should be applied AFTER retrieval (trim) instead of
    BEFORE the ANN scan. Post-filtering guarantees recall (fetch then trim) but
    scans more; pre-filtering is faster but selective filters can cut ANN
    results (recall risk) — the speed/recall tradeoff the caller chooses."""
    return bool(filters) and settings.METADATA_FILTER_MODE == "post"


def _passes_filter(row, filters) -> bool:
    """Python-side filter for POST-filtering. Row must carry user_id, created_at
    and metadata (the search queries add those columns in post mode)."""
    if not filters:
        return True
    uid = filters.get("user_id")
    if isinstance(uid, (list, tuple)):
        row_uid = row.get("user_id")
        if row_uid is None:
            if None not in uid or not row.get("ingested_by"):
                return False
        elif row_uid not in [v for v in uid if v is not None]:
            return False
    elif uid and row.get("user_id") != uid:
        return False
    dfrom, dto = filters.get("date_from"), filters.get("date_to")
    created = row.get("created_at")
    if (dfrom or dto) and created is not None:
        cdate = str(created)[:10]
        if dfrom and cdate < str(dfrom)[:10]:
            return False
        if dto and cdate > str(dto)[:10]:
            return False
    tags = filters.get("tags")
    if tags:
        row_tags = (row.get("metadata") or {}).get("tags") or []
        if filters.get("tags_mode") == "all":
            if not all(t in row_tags for t in tags):
                return False
        elif not any(t in row_tags for t in tags):
            return False
    return True


# ---------------------------------------------------------------------------
# Individual retrievers
# ---------------------------------------------------------------------------

def vector_search(query_emb, top_k: int, collection_id: int | None = None, filters=None):
    if db.USE_PGVECTOR:
        coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
        post = _is_post_filter(filters)
        filter_where, filter_params = _filter_where(filters) if not post else ("", [])
        # Extra columns needed to trim in Python when post-filtering.
        select_cols = ", d.user_id, d.created_at, d.ingested_by" if post else ""
        oversample = settings.METADATA_FILTER_OVERSAMPLE if post else 1
        # Placeholders: SELECT vec, [WHERE collection], [WHERE filters], ORDER BY vec, LIMIT
        params = [db.to_db_vec(query_emb)]
        if collection_id is not None:
            params.append(collection_id)
        params.extend(filter_params)
        params.append(db.to_db_vec(query_emb))
        params.append(top_k * settings.RETRIEVAL_MULTIPLIER * oversample)
        _ph = f"%s::{db.VEC_CAST}"
        with db.get_conn().cursor() as cur:
            cur.execute(
                f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type{select_cols},
                          {db.score_expr('c.embedding', _ph)} AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE 1=1 {coll_filter}{filter_where}
                   ORDER BY {db.dist_expr('c.embedding', _ph)}
                   LIMIT %s""",
                params,
            )
            rows = cur.fetchall()
        results = [(float(r["score"] or 0.0), r) for r in rows]
        if post:
            results = [(s, r) for s, r in results if _passes_filter(r, filters)]
        # Relevance threshold: drop weak matches below the floor.
        return [(s, r) for s, r in results if s >= settings.RELEVANCE_FLOOR]

    # JSONB fallback: fetch all embeddings, score in Python.
    with db.get_conn().cursor() as cur:
        cur.execute(
            """SELECT c.id, c.content, c.metadata, c.document_id, d.id AS doc_id,
                      d.title, d.source_type, d.collection_id, d.user_id,
                      d.created_at, c.embedding
               FROM chunks c JOIN documents d ON d.id = c.document_id"""
        )
        rows = cur.fetchall()
    if collection_id is not None:
        rows = [r for r in rows if r.get("collection_id") == collection_id]
    scored = []
    for r in rows:
        if not _passes_filter(r, filters):
            continue
        v = db.from_db_vec(r["embedding"])
        if v is None:
            continue
        s = db.similarity(query_emb, v)
        if s >= settings.RELEVANCE_FLOOR:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return scored[: top_k * 2]


def keyword_search(query: str, top_k: int, collection_id: int | None = None, filters=None):
    """Full-text search over chunk content AND the document title.

    Matching the title makes file-name queries work ("describe chart.png",
    "what is in report.pdf?") even when the chunk text itself is OCR noise.
    Title matches are boosted so they surface above weak content matches."""
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    post = _is_post_filter(filters)
    filter_where, filter_params = _filter_where(filters) if not post else ("", [])
    select_cols = ", d.user_id, d.created_at, d.ingested_by" if post else ""
    oversample = settings.METADATA_FILTER_OVERSAMPLE if post else 1
    params = [query, query, query, query]
    if collection_id is not None:
        params.append(collection_id)
    params.extend(filter_params)
    params.append(top_k * oversample)
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type{select_cols},
                      ts_rank(to_tsvector('english', c.content),
                              plainto_tsquery('english', %s))
                      + {settings.KEYWORD_TITLE_BOOST} * ts_rank(to_tsvector('english', d.title),
                                                               plainto_tsquery('english', %s)) AS score
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE (to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
                  OR to_tsvector('english', d.title) @@ plainto_tsquery('english', %s))
                 {coll_filter}{filter_where}
               ORDER BY score DESC
               LIMIT %s""",
            params,
        )
        rows = cur.fetchall()
    results = [(float(r["score"] or 0.0), r) for r in rows]
    if post:
        results = [(s, r) for s, r in results if _passes_filter(r, filters)]
    return results


# ---------------------------------------------------------------------------
# Fusion + main entry
# ---------------------------------------------------------------------------

def expand_query(query: str) -> list[str]:
    """LLM query expansion: semantically-equivalent rephrasings + keyword forms.

    Generates QUERY_EXPANSION_VARIANTS total variants (including the original),
    each embedded for dense search AND fed to keyword + sparse, so chunks that
    match the same intent under different wording still get retrieved.
    """
    extra = max(settings.QUERY_EXPANSION_VARIANTS - 1, 1)
    try:
        text = chat_text(
            [{"role": "system", "content": EXPANSION_PROMPT.format(n=extra)},
             {"role": "user", "content": f"Query: {query}"}],
            temperature=0.5,  # a little diversity so rephrasings differ
        )
        result = [query]
        for ln in text.splitlines():
            v = ln.strip().strip("-•*").strip()
            if not v or v.lower() == query.lower():
                continue
            if any(v.lower() == x.lower() for x in result):
                continue
            result.append(v)
            if len(result) >= settings.QUERY_EXPANSION_VARIANTS:
                break
        return result
    except Exception:  # noqa: BLE001
        return [query]


# Matches file references like "chart.png", "report.pdf" (single filename token
# immediately before the extension, so 'describe chart.png' matches 'chart.png'
# and not a greedy 'describe chart.png').
_FILENAME_RE = re.compile(r"([A-Za-z0-9_\-]+)\.(pdf|png|jpe?g|gif|webp|bmp|docx|xlsx|pptx|txt|md|csv|json)",
                          re.IGNORECASE)


def sparse_search(query: str, top_k: int, collection_id: int | None = None, filters=None):
    """BM25-style sparse retrieval over EXACT terms (names, codes, acronyms).

    Matches documents containing the query's exact tokens via inner product on
    sparsevec. Catches what dense embeddings miss — e.g. 'SKU-4471' matches
    only chunks that literally contain '4471'.
    """
    if not db.SPARSE_READY or not settings.USE_SPARSE_SEARCH:
        return []
    qv = sparse.query_vector(query)
    if qv is None:
        return []
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    post = _is_post_filter(filters)
    filter_where, filter_params = _filter_where(filters) if not post else ("", [])
    select_cols = ", d.user_id, d.created_at, d.ingested_by" if post else ""
    oversample = settings.METADATA_FILTER_OVERSAMPLE if post else 1
    params = [qv]
    if collection_id is not None:
        params.append(collection_id)
    params.extend(filter_params)
    params.append(qv)
    params.append(top_k * oversample)
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type{select_cols},
                       - (c.sparse_embedding <#> %s::sparsevec) AS score
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.sparse_embedding IS NOT NULL {coll_filter}{filter_where}
               ORDER BY c.sparse_embedding <#> %s::sparsevec
               LIMIT %s""",
            params,
        )
        rows = cur.fetchall()
    results = [(float(r["score"] or 0.0), r) for r in rows if (r["score"] or 0.0) > 0.0]
    if post:
        results = [(s, r) for s, r in results if _passes_filter(r, filters)]
    return results


def filename_search(query: str, top_k: int, collection_id: int | None = None, filters=None):
    """If the query names a document by its file name (e.g. 'describe chart.png',
    'what is in report.pdf?'), return that document's top chunks directly.
    This guarantees file-referencing questions find their source even when the
    chunk text is OCR noise that vector search can't match."""
    toks = _FILENAME_RE.findall(query)
    if not toks:
        return []
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    post = _is_post_filter(filters)
    filter_where, filter_params = _filter_where(filters) if not post else ("", [])
    select_cols = ", d.user_id, d.created_at, d.ingested_by" if post else ""
    params = [collection_id] if collection_id is not None else []
    params.extend(filter_params)
    results = []
    seen = set()
    with db.get_conn().cursor() as cur:
        for base, ext in toks:
            name = (base.strip() + "." + ext).lower()
            cur.execute(
                f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type{select_cols},
                           {settings.FILENAME_MATCH_SCORE} AS score
                       FROM chunks c JOIN documents d ON d.id = c.document_id
                       WHERE lower(d.title) LIKE %s {coll_filter}{filter_where}
                       ORDER BY c.chunk_index
                       LIMIT %s""",
                tuple(params) + (f"%{name}%", top_k),
            )
            for row in cur.fetchall():
                if row["id"] in seen:
                    continue
                if post and not _passes_filter(row, filters):
                    continue
                seen.add(row["id"])
                results.append((settings.FILENAME_MATCH_SCORE, row))
    return results


def rrf_fuse(ranked_lists, k: int = 60):
    """Reciprocal-rank fusion of multiple ranked result lists."""
    scores: dict[int, float] = {}
    info: dict[int, dict] = {}
    for ranked in ranked_lists:
        for rank, (_score, row) in enumerate(ranked):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            info[rid] = row
    out = []
    for rid, s in sorted(scores.items(), key=lambda x: -x[1]):
        row = dict(info[rid])
        row["rrf_score"] = s
        out.append(row)
    return out


def retrieve(query: str, top_k: int | None = None, collection: str | None = None,
             use_cache: bool = True, filters=None, user_id: str | None = None):
    """Main hybrid retrieval, optionally scoped to a collection.

    Returns dict with 'results' (or 'cached'), 'query_emb'. When the full
    pipeline runs it also includes 'latency_ms' = {"stage1": ..., "rerank": ...,
    "total": ...} (milliseconds) so callers can see the bi-encoder vs
    cross-encoder cost split and tune RERANKER_CANDIDATES.

    use_cache=False bypasses BOTH caches (response + retrieval-results) so
    every call runs the real pipeline — used by benchmarks and tests.

    filters restricts retrieval by metadata (see _norm_filters): user_id,
    date_from/date_to, tags. Filtered queries bypass the caches (a cached
    result is scoped to its unfiltered context). Filter placement follows
    METADATA_FILTER_MODE: 'pre' (WHERE before the ANN scan) or 'post'
    (fetch more, trim after) — each trades speed vs recall.

    user_id scopes the caches per user (None = anonymous/public bucket) so
    per-user private documents can never leak across accounts via a cache hit.
    """
    top_k = top_k or settings.TOP_K
    collection_id = db.get_collection_id(collection) if collection else None
    q_emb = embed_query(query)
    filters = _norm_filters(filters)
    if filters:
        log.info("🔎 Metadata filter applied: %s | mode=%s",
                 filters, settings.METADATA_FILTER_MODE)
    t0 = time.perf_counter()

    # A user_id filter that stays inside the caller's own visibility (their id
    # and/or the shared/global bucket) is redundant — the caches are already
    # scoped per user — so it must NOT disable them. Any other filter
    # (date/tags, or another user's id) still bypasses the caches: a cached
    # result is scoped to its unfiltered context, so those can't be reused.
    def _cache_compatible(filters, user_id):
        if not filters:
            return True
        if set(filters) != {"user_id"}:
            return False
        allowed = filters["user_id"]
        if not isinstance(allowed, (list, tuple)):
            allowed = [allowed]
        scopes = _as_scopes(user_id)
        return all(v is None or v in scopes for v in allowed)

    cache_safe = _cache_compatible(filters, user_id)
    # Cache reads: a real user checks their own bucket first, then the
    # shared/global bucket (the admin's cache). Admin/anonymous read global.
    read_scopes = [None] if user_id is None else [user_id, None]

    if use_cache and cache_safe:
        # 1. Semantic cache fast-path (scoped to collection + user).
        cached = semantic_cache_lookup(q_emb, collection_id, read_scopes)
        if cached:
            log.info("🗄️  Semantic cache hit — reusing previous answer | %.0f ms",
                     (time.perf_counter() - t0) * 1000)
            return {"cached": cached["response"],
                    "cached_sources": cached.get("sources") or [],
                    "query_emb": q_emb, "collection_id": collection_id,
                    "latency_ms": {"stage1": 0.0, "rerank": 0.0,
                                   "total": round((time.perf_counter() - t0) * 1000, 2)}}

        # 1b. Retrieval-results cache: popular queries reuse their cached
        # reranked chunks, skipping hybrid search + RRF + cross-encoder rerank
        # entirely (the LLM answer is still generated fresh downstream).
        rcache = retrieval_cache_lookup(q_emb, collection_id, read_scopes)
        if rcache is not None:
            log.info("🗄️  Retrieval cache hit — reusing previous chunks | %.0f ms",
                     (time.perf_counter() - t0) * 1000)
            return {"results": rcache["results"], "query_emb": q_emb,
                    "collection_id": collection_id,
                    "best_score": rcache["best_score"], "cached_retrieval": True,
                    "latency_ms": {"stage1": 0.0, "rerank": 0.0,
                                   "total": round((time.perf_counter() - t0) * 1000, 2)}}

    # 2. Query expansion.
    queries = [query]
    if settings.USE_QUERY_EXPANSION:
        queries = expand_query(query)
    if len(queries) > 1:
        log.info("🌱 Query expanded into %d variants:\n%s", len(queries),
                 fmt_table(["#", "variant"],
                           [(i + 1, q) for i, q in enumerate(queries)]))

    # 3. Run multiple searches (scoped to collection).
    # Stage-1 recall target: with the cross-encoder reranker ON, the fast
    # hybrid search fetches a WIDE pool (RERANKER_CANDIDATES) that the
    # reranker later trims to top_k. With it OFF, fetch top_k directly.
    # Note: vector_search internally multiplies by RETRIEVAL_MULTIPLIER, so
    # each search oversamples ~2x and RRF dedups the union — still bounded to
    # the reranker's candidate window, never the full corpus.
    candidate_k = settings.RERANKER_CANDIDATES if settings.USE_RERANKER else top_k
    best_score = 0.0
    lists = [vector_search(q_emb, candidate_k, collection_id, filters)]
    if db.SPARSE_READY and settings.USE_SPARSE_SEARCH:
        lists.append(sparse_search(query, candidate_k, collection_id, filters))
    for scored, _row in lists[0]:
        best_score = max(best_score, scored)
    for q in queries[1:]:
        qe = embed_query(q)
        vs = vector_search(qe, candidate_k, collection_id, filters)
        for scored, _row in vs:
            best_score = max(best_score, scored)
        lists.append(vs)
        if settings.USE_KEYWORD_SEARCH:
            lists.append(keyword_search(q, candidate_k, collection_id, filters))
        if db.SPARSE_READY and settings.USE_SPARSE_SEARCH:
            lists.append(sparse_search(q, candidate_k, collection_id, filters))

    # 4. Fuse with reciprocal-rank fusion.
    fused = rrf_fuse(lists, settings.RRF_K)
    log.debug("🧩 Fused %d search channels → %d unique chunks", len(lists), len(fused))

    # Exact file-name references (e.g. "chart.png") are promoted to the top:
    # they unambiguously identify a document, so "describe chart.png" must
    # surface chart.png even though its OCR-y chunks rank low in vector space.
    promoted = []
    seen = set()
    for _score, row in filename_search(query, top_k, collection_id, filters):
        if row["id"] not in seen:
            seen.add(row["id"])
            row["rrf_score"] = 99.0
            promoted.append(row)
    if promoted:
        log.debug("📎 %d exact file-name match(es) promoted", len(promoted))

    if settings.USE_RERANKER:
        # 5. Two-stage: cross-encoder reranks the wide fused pool down to top_k
        # (bi-encoder recall -> cross-encoder precision), keeping filename
        # matches pinned on top.
        t1 = time.perf_counter()  # end of stage-1 (search + fusion + filename)
        pool = fused[: settings.RERANKER_CANDIDATES]
        reranked = rerank.rerank(query, pool, top_k, return_all=True)
        results = list(promoted)
        for row in reranked:
            if row["id"] in seen:
                continue
            results.append(row)
            if len(results) >= top_k:
                break
        t2 = time.perf_counter()  # end of stage-2 (cross-encoder rerank)
        # DEBUG: full ranked pool so you can see exactly what was cut and why.
        log.debug("⚡ Full rerank ranking (%d of %d pool candidates):\n%s",
                  len(reranked), len(pool),
                  fmt_table(["rank", "title", "score", "conf", "snippet"],
                            [(i + 1,
                              (r.get("title") or "")[:44],
                              round(float(r.get("rerank_score") or 0.0), 3),
                              round(float(r.get("rerank_confidence") or 0.0), 3),
                              (r.get("content") or "").replace("\n", " ")[:48])
                             for i, r in enumerate(reranked)]))
    else:
        # 5. Fill remaining slots from the fused ranking.
        t1 = time.perf_counter()  # end of stage-1 (search + fusion + filename)
        results = list(promoted)
        for row in fused:
            if row["id"] in seen:
                continue
            results.append(row)
            if len(results) >= top_k:
                break
        t2 = time.perf_counter()

    for i, row in enumerate(results[:top_k]):
        row["citation"] = i + 1
    results = results[:top_k]
    if results:
        _rows = []
        for i, r in enumerate(results):
            has_rerank = r.get("rerank_score") is not None
            _rows.append((
                i + 1,
                r.get("citation"),
                (r.get("title") or "")[:44],
                round(float(r.get("rerank_score") or r.get("rrf_score") or 0.0), 3),
                round(float(r.get("rerank_confidence") or 0.0), 3) if has_rerank else "—",
                (r.get("content") or "").replace("\n", " ")[:48],
            ))
        log.info("⚡ Ranked candidates (top %d, score=rerank logit / rrf):\n%s",
                 len(results), fmt_table(
                     ["rank", "cit", "title", "score", "conf", "snippet"], _rows))
    log.debug("🔎 Retrieval done: %d chunks | best match=%.2f | search %.1fs, rerank %.1fs",
              len(results), best_score,
              (t1 - t0), (t2 - t1))
    # Cache strongly-grounded queries only (never weak/general ones), so a cache
    # hit can't flip a general-knowledge question onto weak doc chunks. Popular
    # repeats then skip the expensive search + rerank stage entirely.
    # The shared/global cache bucket (user_id=None) is readable by every user,
    # so it may only hold results that contain no private (owned) documents.
    shared_safe = _results_shared_safe(results, user_id)
    if use_cache and cache_safe and results and shared_safe \
            and best_score >= settings.GENERAL_STRONG_THRESHOLD:
        retrieval_cache_store(query, q_emb, results, best_score, collection_id, user_id)
    latency_ms = {
        "stage1": round((t1 - t0) * 1000, 2),
        "rerank": round((t2 - t1) * 1000, 2),
        "total": round((t2 - t0) * 1000, 2),
    }
    return {"results": results, "query_emb": q_emb, "collection_id": collection_id,
            "best_score": best_score, "latency_ms": latency_ms, "filters": filters,
            "user_id": user_id, "shared_safe": shared_safe}
