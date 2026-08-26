# Agentic RAG - dense (bi-encoder) retrieval: query embedding + pgvector search.
#
# Documents & queries are embedded INDEPENDENTLY, so similarity is a cheap dot
# product — fast enough to scan the whole corpus. This is stage-1 recall; the
# cross-encoder reranker (reranker.py) trims the wide pool it returns.

import logging
from functools import lru_cache

import app.database.postgres as db
from app.core.config import settings
from app.llm.embeddings import embed_texts
from app.retrieval.filters import filter_where as build_filter_where
from app.retrieval.filters import is_post_filter, passes_filter

log = logging.getLogger("retrieval")


@lru_cache(maxsize=settings.QUERY_EMBED_CACHE_SIZE)
def embed_query(query: str):
    """Embed a query, cached so repeated or similar queries never re-embed
    the same text."""
    text = query
    if settings.USE_ASYMMETRIC_PREFIX:
        text = settings.QUERY_PREFIX + text
    return embed_texts([text])[0]


def vector_search(query_emb, top_k: int, collection_id: int | None = None, filters=None):
    if db.USE_PGVECTOR:
        coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
        post = is_post_filter(filters)
        filter_where, filter_params = build_filter_where(filters) if not post else ("", [])
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
            results = [(s, r) for s, r in results if passes_filter(r, filters)]
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
        if not passes_filter(r, filters):
            continue
        v = db.from_db_vec(r["embedding"])
        if v is None:
            continue
        s = db.similarity(query_emb, v)
        if s >= settings.RELEVANCE_FLOOR:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return scored[: top_k * 2]
