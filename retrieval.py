# Agentic RAG - hybrid retrieval.
# Combines:
#   * semantic cache with a cosine threshold
#   * LLM query expansion + weighted reciprocal-rank fusion
#   * asymmetric query/document prefixes (when enabled)
#   * vector (pgvector) + keyword (Postgres full-text) hybrid search

import re

import db
from functools import lru_cache

from config import settings
from llm import embed_texts, chat_text
from prompts import EXPANSION_PROMPT


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

def semantic_cache_lookup(query_emb, collection_id: int | None = None):
    if not db.USE_PGVECTOR:
        return None
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT id, response, sources, 1 - (query_embedding <=> %s::{db.VEC_CAST}) AS sim
               FROM semantic_cache
               WHERE collection_id IS NOT DISTINCT FROM %s
               ORDER BY query_embedding <=> %s::{db.VEC_CAST}
               LIMIT 1""",
            (db.to_db_vec(query_emb), collection_id, db.to_db_vec(query_emb)),
        )
        row = cur.fetchone()
    if row and row["sim"] is not None and row["sim"] >= settings.SEMANTIC_CACHE_THRESHOLD:
        return row
    return None


def semantic_cache_store(query: str, query_emb, response: str, model: str, sources=None,
                         collection_id: int | None = None):
    try:
        with db.get_conn().cursor() as cur:
            cur.execute(
                "INSERT INTO semantic_cache (query_embedding, query, response, model, sources, collection_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (db.to_db_vec(query_emb), query, response, model,
                 db.to_json(sources) if sources is not None else "{}", collection_id),
            )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Individual retrievers
# ---------------------------------------------------------------------------

def vector_search(query_emb, top_k: int, collection_id: int | None = None):
    if db.USE_PGVECTOR:
        coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
        # Placeholders: SELECT vec, [WHERE collection], ORDER BY vec, LIMIT
        params = [db.to_db_vec(query_emb)]
        if collection_id is not None:
            params.append(collection_id)
        params.append(db.to_db_vec(query_emb))
        params.append(top_k * settings.RETRIEVAL_MULTIPLIER)
        with db.get_conn().cursor() as cur:
            cur.execute(
                f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type,
                          1 - (c.embedding <=> %s::{db.VEC_CAST}) AS score
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE 1=1 {coll_filter}
                   ORDER BY c.embedding <=> %s::{db.VEC_CAST}
                   LIMIT %s""",
                params,
            )
            rows = cur.fetchall()
        results = [(float(r["score"] or 0.0), r) for r in rows]
        # Relevance threshold: drop weak matches below the floor.
        return [(s, r) for s, r in results if s >= settings.RELEVANCE_FLOOR]

    # JSONB fallback: fetch all embeddings, score in Python.
    with db.get_conn().cursor() as cur:
        cur.execute(
            """SELECT c.id, c.content, c.metadata, c.document_id, d.collection_id, c.embedding
               FROM chunks c JOIN documents d ON d.id = c.document_id"""
        )
        rows = cur.fetchall()
    if collection_id is not None:
        rows = [r for r in rows if r.get("collection_id") == collection_id]
    scored = []
    for r in rows:
        v = db.from_db_vec(r["embedding"])
        if v is None:
            continue
        s = db.cosine(query_emb, v)
        if s >= settings.RELEVANCE_FLOOR:
            scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return scored[: top_k * 2]


def keyword_search(query: str, top_k: int, collection_id: int | None = None):
    """Full-text search over chunk content AND the document title.

    Matching the title makes file-name queries work ("describe chart.png",
    "what is in report.pdf?") even when the chunk text itself is OCR noise.
    Title matches are boosted so they surface above weak content matches."""
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    params = [query, query, query, query]
    if collection_id is not None:
        params.append(collection_id)
    params.append(top_k)
    with db.get_conn().cursor() as cur:
        cur.execute(
            f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type,
                      ts_rank(to_tsvector('english', c.content),
                              plainto_tsquery('english', %s))
                      + {settings.KEYWORD_TITLE_BOOST} * ts_rank(to_tsvector('english', d.title),
                                                               plainto_tsquery('english', %s)) AS score
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
                  OR to_tsvector('english', d.title) @@ plainto_tsquery('english', %s)
                 {coll_filter}
               ORDER BY score DESC
               LIMIT %s""",
            params,
        )
        rows = cur.fetchall()
    return [(float(r["score"] or 0.0), r) for r in rows]


# ---------------------------------------------------------------------------
# Fusion + main entry
# ---------------------------------------------------------------------------

def expand_query(query: str) -> list[str]:
    """LLM query expansion: rephrase + extra keyword queries."""
    try:
        text = chat_text(
            [{"role": "system", "content": EXPANSION_PROMPT},
             {"role": "user", "content": f"Query: {query}"}],
            temperature=0.3,
        )
        lines = [ln.strip("-• *").strip() for ln in text.splitlines() if ln.strip()]
        result = [query] + [ln for ln in lines if ln and ln.lower() != query.lower()]
        return result[:5]
    except Exception:  # noqa: BLE001
        return [query]


# Matches file references like "chart.png", "report.pdf" (single filename token
# immediately before the extension, so 'describe chart.png' matches 'chart.png'
# and not a greedy 'describe chart.png').
_FILENAME_RE = re.compile(r"([A-Za-z0-9_\-]+)\.(pdf|png|jpe?g|gif|webp|bmp|docx|xlsx|pptx|txt|md|csv|json)",
                          re.IGNORECASE)


def filename_search(query: str, top_k: int, collection_id: int | None = None):
    """If the query names a document by its file name (e.g. 'describe chart.png',
    'what is in report.pdf?'), return that document's top chunks directly.
    This guarantees file-referencing questions find their source even when the
    chunk text is OCR noise that vector search can't match."""
    toks = _FILENAME_RE.findall(query)
    if not toks:
        return []
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    params = [collection_id] if collection_id is not None else []
    results = []
    seen = set()
    with db.get_conn().cursor() as cur:
        for base, ext in toks:
            name = (base.strip() + "." + ext).lower()
            cur.execute(
                f"""SELECT c.id, c.content, c.metadata, d.title, d.id AS doc_id, d.source_type,
                           {settings.FILENAME_MATCH_SCORE} AS score
                       FROM chunks c JOIN documents d ON d.id = c.document_id
                       WHERE lower(d.title) LIKE %s {coll_filter}
                       ORDER BY c.chunk_index
                       LIMIT %s""",
                tuple(params) + (f"%{name}%", top_k),
            )
            for row in cur.fetchall():
                if row["id"] not in seen:
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


def retrieve(query: str, top_k: int | None = None, collection: str | None = None):
    """Main hybrid retrieval, optionally scoped to a collection.
    Returns dict with 'results' (or 'cached'), 'query_emb'."""
    top_k = top_k or settings.TOP_K
    collection_id = db.get_collection_id(collection) if collection else None
    q_emb = embed_query(query)

    # 1. Semantic cache fast-path (scoped to collection).
    cached = semantic_cache_lookup(q_emb, collection_id)
    if cached:
        return {"cached": cached["response"],
                "cached_sources": cached.get("sources") or [],
                "query_emb": q_emb, "collection_id": collection_id}

    # 2. Query expansion.
    queries = [query]
    if settings.USE_QUERY_EXPANSION:
        queries = expand_query(query)

    # 3. Run multiple searches (scoped to collection).
    best_score = 0.0
    lists = [vector_search(q_emb, top_k, collection_id)]
    for scored, _row in lists[0]:
        best_score = max(best_score, scored)
    for q in queries[1:]:
        qe = embed_query(q)
        vs = vector_search(qe, top_k, collection_id)
        for scored, _row in vs:
            best_score = max(best_score, scored)
        lists.append(vs)
        lists.append(keyword_search(q, top_k, collection_id))

    # 4. Fuse with reciprocal-rank fusion.
    fused = rrf_fuse(lists, settings.RRF_K)

    # Exact file-name references (e.g. "chart.png") are promoted to the top:
    # they unambiguously identify a document, so "describe chart.png" must
    # surface chart.png even though its OCR-y chunks rank low in vector space.
    promoted = []
    seen = set()
    for _score, row in filename_search(query, top_k, collection_id):
        if row["id"] not in seen:
            seen.add(row["id"])
            row["rrf_score"] = 99.0
            promoted.append(row)

    # 5. Number citations.
    results = list(promoted)
    for row in fused:
        if row["id"] in seen:
            continue
        results.append(row)
        if len(results) >= top_k:
            break

    for i, row in enumerate(results[:top_k]):
        row["citation"] = i + 1
    return {"results": results[:top_k], "query_emb": q_emb, "collection_id": collection_id,
            "best_score": best_score}
