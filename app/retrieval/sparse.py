# Agentic RAG - BM25-style sparse retrieval (pgvector `sparsevec`).
#
# Purpose: catch EXACT terms that dense embeddings miss — names, codes,
# acronyms, IDs (e.g. "SKU-4471", "chart.png", "GCP"). Two channels:
#   * doc vectors   = log-normalized term frequency, stored as a sparsevec
#   * query vector  = IDF-weighted term frequency over the same vocabulary
#   * matching      = inner product (sparsevec_ip_ops) — only documents that
#                     contain the exact term score > 0
# The sparse result list is fused with dense + full-text via RRF in
# retrieval.retrieve(). Requires pgvector >= 0.7 (sparsevec); gated by
# db.SPARSE_READY so it degrades gracefully on older versions.

import logging
import math
import re
from collections import Counter

from app.core.config import settings
from app.retrieval.filters import filter_where as build_filter_where
from app.retrieval.filters import is_post_filter, passes_filter
import db

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Minimal English stopwords. Codes / acronyms / numbers are intentionally kept —
# those are exactly what sparse retrieval must catch.
_STOPWORDS = frozenset(
    "a an the and or but if then else for nor so yet of in on at to from by with "
    "without via per about into over after before during under between what which "
    "who whom whose when where why how is are was were be been being do does did "
    "will would can could should shall may might must has have had this that these "
    "those it its not no yes also only just very more most such than too as up down "
    "out off all any both each few other some your you their our my his her i we "
    "they he she".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased alnum/underscore tokens; stopwords and single chars dropped."""
    return [t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Vocabulary: term -> sparsevec index (1-based, exact, collision-free)
# ---------------------------------------------------------------------------

def ensure_vocab(conn, terms) -> dict[str, int]:
    """Return {term: idx} for the given terms, creating vocab rows as needed."""
    terms = list(dict.fromkeys(t for t in terms if t))
    if not terms:
        return {}
    out = {}
    with conn.cursor() as cur:
        cur.execute("SELECT term, idx FROM sparse_vocab WHERE term = ANY(%s)", (terms,))
        for r in cur.fetchall():
            out[r["term"]] = r["idx"]
        missing = [t for t in terms if t not in out]
        for t in missing:
            cur.execute(
                "INSERT INTO sparse_vocab (term) VALUES (%s) "
                "ON CONFLICT (term) DO NOTHING", (t,)
            )
        if missing:
            cur.execute("SELECT term, idx FROM sparse_vocab WHERE term = ANY(%s)", (missing,))
            for r in cur.fetchall():
                out[r["term"]] = r["idx"]
    return out


# ---------------------------------------------------------------------------
# Term / corpus statistics (IDF + chunk counts)
# ---------------------------------------------------------------------------

def _tf(text: str) -> Counter:
    return Counter(tokenize(text))


def add_terms(conn, tf_list):
    """Increment document frequencies + corpus counters for newly stored chunks."""
    with conn.cursor() as cur:
        for tf in tf_list:
            if not tf:
                continue
            cur.executemany(
                "INSERT INTO sparse_term_stats (term, df) VALUES (%s, 1) "
                "ON CONFLICT (term) DO UPDATE SET df = sparse_term_stats.df + 1",
                [(t,) for t in tf],
            )
            cur.execute(
                "UPDATE sparse_corpus_stats SET chunk_count = chunk_count + 1, "
                "total_tokens = total_tokens + %s WHERE id = 1",
                (sum(tf.values()),),
            )


def remove_terms(conn, tf_list):
    """Decrement document frequencies + corpus counters for removed chunks."""
    with conn.cursor() as cur:
        for tf in tf_list:
            if not tf:
                continue
            cur.executemany(
                "UPDATE sparse_term_stats SET df = GREATEST(df - 1, 0) WHERE term = %s",
                [(t,) for t in tf],
            )
            cur.execute(
                "UPDATE sparse_corpus_stats SET chunk_count = GREATEST(chunk_count - 1, 0), "
                "total_tokens = GREATEST(total_tokens - %s, 0) WHERE id = 1",
                (sum(tf.values()),),
            )


def _chunk_count() -> int:
    try:
        with db.get_conn().cursor() as cur:
            cur.execute("SELECT chunk_count FROM sparse_corpus_stats WHERE id = 1")
            r = cur.fetchone()
        return (r["chunk_count"] or 0) if r else 0
    except Exception:  # noqa: BLE001
        return 0


def load_idf() -> dict[str, float]:
    """idf(term) = ln(1 + (N - df + 0.5) / (df + 0.5))."""
    n = _chunk_count()
    try:
        with db.get_conn().cursor() as cur:
            cur.execute("SELECT term, df FROM sparse_term_stats")
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001
        return {}
    return {r["term"]: math.log(1 + (n - r["df"] + 0.5) / (r["df"] + 0.5)) for r in rows}


# ---------------------------------------------------------------------------
# Vector builders
# ---------------------------------------------------------------------------

def build_doc_batch(conn, texts):
    """Build (sparsevec_literal, token_count) per text; update vocab + term stats."""
    if not db.SPARSE_READY or not settings.USE_SPARSE_SEARCH:
        return [None] * len(texts), [0] * len(texts)
    tf_list = [_tf(t) for t in texts]
    all_terms = [t for tf in tf_list for t in tf]
    idx = ensure_vocab(conn, all_terms)
    vecs, counts = [], []
    for tf in tf_list:
        counts.append(sum(tf.values()))
        pairs = []
        for t, c in tf.most_common(settings.SPARSE_TOP_TERMS):
            i = idx.get(t)
            if i is not None and i <= settings.SPARSE_DIM:
                pairs.append((i, 1.0 + math.log(c)))
        if pairs:
            vecs.append("{" + ",".join(f"{i}:{w:.6f}" for i, w in pairs)
                        + "}/" + str(settings.SPARSE_DIM))
        else:
            vecs.append(None)
    add_terms(conn, tf_list)
    return vecs, counts


def query_vector(query: str):
    """IDF-weighted query sparsevec literal (or None if nothing to match)."""
    if not db.SPARSE_READY or not settings.USE_SPARSE_SEARCH:
        return None
    tf = _tf(query)
    if not tf:
        return None
    try:
        with db.get_conn().cursor() as cur:
            cur.execute("SELECT term, idx FROM sparse_vocab WHERE term = ANY(%s)", (list(tf),))
            idxmap = {r["term"]: r["idx"] for r in cur.fetchall()}
    except Exception:  # noqa: BLE001
        return None
    idf = load_idf()
    pairs = []
    for t, c in tf.items():
        i = idxmap.get(t)
        if i is None or i > settings.SPARSE_DIM:
            continue
        w = idf.get(t, 0.0) * (1.0 + math.log(c))
        if w > 0.0:
            pairs.append((i, w))
    if not pairs:
        return None
    return "{" + ",".join(f"{i}:{w:.6f}" for i, w in pairs) + "}/" + str(settings.SPARSE_DIM)


def sparse_search(query: str, top_k: int, collection_id: int | None = None, filters=None):
    """BM25-style sparse retrieval over EXACT terms (names, codes, acronyms).

    Matches documents containing the query's exact tokens via inner product on
    sparsevec. Catches what dense embeddings miss — e.g. 'SKU-4471' matches
    only chunks that literally contain '4471'.
    """
    if not db.SPARSE_READY or not settings.USE_SPARSE_SEARCH:
        return []
    qv = query_vector(query)
    if qv is None:
        return []
    coll_filter = "AND d.collection_id = %s" if collection_id is not None else ""
    post = is_post_filter(filters)
    filter_where, filter_params = build_filter_where(filters) if not post else ("", [])
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
        results = [(s, r) for s, r in results if passes_filter(r, filters)]
    return results


def remove_texts(conn, contents):
    """Decrement stats for removed chunk contents (used by delta updates)."""
    if not db.SPARSE_READY or not settings.USE_SPARSE_SEARCH:
        return
    remove_terms(conn, [_tf(c) for c in contents])


# ---------------------------------------------------------------------------
# Migration / rebuild
# ---------------------------------------------------------------------------

def rebuild(conn=None, progress=print) -> int:
    """Rebuild vocab, term stats and all chunk sparse vectors from scratch.
    Call once after enabling sparse search so existing chunks get covered."""
    if not db.SPARSE_READY:
        log.warning("Sparse search unavailable (pgvector lacks sparsevec?)")
        return 0
    own_conn = conn is None
    conn = conn or db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE sparse_term_stats, sparse_vocab RESTART IDENTITY CASCADE")
            cur.execute("UPDATE sparse_corpus_stats SET chunk_count = 0, total_tokens = 0 WHERE id = 1")
            cur.execute("SELECT id, content FROM chunks ORDER BY id")
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001
        if own_conn:
            conn.close()
        return 0

    done = 0
    batch = settings.EMBED_BATCH_SIZE
    total = max(len(rows), 1)
    for start in range(0, len(rows), batch):
        chunk_rows = rows[start:start + batch]
        vecs, counts = build_doc_batch(conn, [r["content"] for r in chunk_rows])
        with conn.cursor() as cur:
            for r, v, cnt in zip(chunk_rows, vecs, counts):
                cur.execute(
                    "UPDATE chunks SET sparse_embedding = %s, token_count = %s WHERE id = %s",
                    (v, cnt, r["id"]),
                )
        done += len(chunk_rows)
        if progress is not None:
            try:
                progress(f"  ...sparse {done}/{len(rows)}")
            except Exception:  # noqa: BLE001
                pass
    if own_conn:
        conn.close()
    return done
