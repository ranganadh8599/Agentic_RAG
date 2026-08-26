# Agentic RAG - hybrid retrieval orchestration.
#
# Combines the individual retrieval strategies (dense, sparse, keyword,
# filename) with RRF fusion, an optional cross-encoder rerank stage, and the
# query-expansion + cache layers:
#
#   HybridRetriever
#         │
#         ├── DenseRetriever    (vector_search)
#         ├── SparseRetriever   (sparse_search)
#         ├── KeywordRetriever  (keyword_search, filename_search)
#         ├── RRF               (rrf_fuse)
#         └── CrossEncoder      (rerank, optional)

import logging
import re
import time

import db
import app.retrieval.reranker as rerank
from app.core.config import settings
from app.core.logging import fmt_table
from app.retrieval.cache import (_results_shared_safe, retrieval_cache_lookup,
                                 retrieval_cache_store, semantic_cache_lookup)
from app.retrieval.dense import embed_query, vector_search
from app.retrieval.filters import filter_where as build_filter_where
from app.retrieval.filters import is_post_filter, norm_filters, passes_filter
from app.retrieval.fusion import rrf_fuse
from app.retrieval.query_rewriter import expand_query
from app.retrieval.sparse import sparse_search

log = logging.getLogger("retrieval")


def keyword_search(query: str, top_k: int, collection_id: int | None = None, filters=None):
    """Full-text search over chunk content AND the document title.

    Matching the title makes file-name queries work ("describe chart.png",
    "what is in report.pdf?") even when the chunk text itself is OCR noise.
    Title matches are boosted so they surface above weak content matches."""
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    post = is_post_filter(filters)
    filter_where, filter_params = build_filter_where(filters) if not post else ("", [])
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
        results = [(s, r) for s, r in results if passes_filter(r, filters)]
    return results


# Matches file references like "chart.png", "report.pdf" (single filename token
# immediately before the extension, so 'describe chart.png' matches 'chart.png'
# and not a greedy 'describe chart.png').
_FILENAME_RE = re.compile(r"([A-Za-z0-9_\-]+)\.(pdf|png|jpe?g|gif|webp|bmp|docx|xlsx|pptx|txt|md|csv|json)",
                          re.IGNORECASE)


def filename_search(query: str, top_k: int, collection_id: int | None = None, filters=None):
    """If the query names a document by its file name (e.g. 'describe chart.png',
    'what is in report.pdf?'), return that document's top chunks directly.
    This guarantees file-referencing questions find their source even when the
    chunk text is OCR noise that vector search can't match."""
    toks = _FILENAME_RE.findall(query)
    if not toks:
        return []
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    post = is_post_filter(filters)
    filter_where, filter_params = build_filter_where(filters) if not post else ("", [])
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
                if post and not passes_filter(row, filters):
                    continue
                seen.add(row["id"])
                results.append((settings.FILENAME_MATCH_SCORE, row))
    return results


def retrieve(query: str, top_k: int | None = None, collection: str | None = None,
             use_cache: bool = True, filters=None, user_id: str | None = None):
    """Main hybrid retrieval, optionally scoped to a collection.

    Returns dict with 'results' (or 'cached'), 'query_emb'. When the full
    pipeline runs it also includes 'latency_ms' = {"stage1": ..., "rerank": ...,
    "total": ...} (milliseconds) so callers can see the bi-encoder vs
    cross-encoder cost split and tune RERANKER_CANDIDATES.

    use_cache=False bypasses BOTH caches (response + retrieval-results) so
    every call runs the real pipeline — used by benchmarks and tests.

    filters restricts retrieval by metadata (see norm_filters): user_id,
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
    filters = norm_filters(filters)
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
        from app.retrieval.cache import _as_scopes
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
