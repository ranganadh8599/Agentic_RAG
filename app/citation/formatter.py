# Agentic RAG - citation / source-card formatting.
#
# Turns raw retrieved blocks into the "source cards" the UI shows next to an
# answer: a per-citation snippet chosen from the part of the chunk that
# actually backs the claim, plus page / image / rerank-confidence metadata.
# Also attaches a stored page image when the cited chunk points at a page
# with one.

import db
from app.citation.sanitizer import significant_words
from app.core.config import settings


def best_snippet(content, answer, max_len=None):
    """Return the region of `content` that overlaps most with the answer text,
    so a source card shows the part that actually backs the cited claim instead
    of just the start of the chunk. Among equally-dense windows it picks the one
    centered nearest to the middle of the matched words."""
    max_len = max_len or settings.SNIPPET_MAX_CHARS
    if not content:
        return ""
    flat = " ".join(content.split())
    a_set = set(significant_words(answer))
    if not a_set:
        return flat[:max_len]
    words = flat.split()
    norm = [w.lower() for w in words]
    matches = [j for j, w in enumerate(norm) if w in a_set]
    if not matches:
        return flat[:max_len]
    win = settings.SNIPPET_WINDOW
    scores = []
    for i in range(max(0, len(norm) - win) + 1):
        scores.append(sum(1 for w in norm[i:i + win] if w in a_set))
    best_score = max(scores)
    # Median is robust to stray matches (e.g. a shared word far away) and
    # centers the snippet on the densest cluster of matched terms.
    center = sorted(matches)[len(matches) // 2]
    best_i = min((i for i, s in enumerate(scores) if s == best_score),
                 key=lambda i: abs((i + win / 2) - center))
    offset = sum(len(words[j]) + 1 for j in range(best_i))
    snippet = flat[offset:offset + max_len]
    return snippet or flat[:max_len]


def format_sources(blocks, answer=""):
    """Build the UI source cards for a set of retrieved blocks."""
    out = []
    for r in blocks:
        meta = r.get("metadata") or {}
        content = r.get("content") or ""
        out.append({
            "citation": r.get("citation"),
            "title": r.get("title"),
            "doc_id": r.get("doc_id"),
            "score": round(float(r.get("rrf_score") or 0.0), 4),
            # Cross-encoder relevance in [0,1] (sigmoid of the raw logit) —
            # a confidence measure for this source matching the query.
            "rerank_confidence": round(float(r.get("rerank_confidence") or 0.0), 4),
            "page": meta.get("page"),
            "image_id": meta.get("image_id"),
            "snippet": best_snippet(content, answer),
        })
    return out


def attach_page_images(sources):
    """If a cited source points to a page that has a stored image, attach it."""
    keys = {(s.get("doc_id"), s.get("page")) for s in sources if s.get("page")}
    if not keys:
        return sources
    mapping = {}
    with db.get_conn().cursor() as cur:
        for doc_id, page in keys:
            cur.execute(
                "SELECT id FROM images WHERE document_id=%s AND page=%s ORDER BY id LIMIT 1",
                (doc_id, page),
            )
            row = cur.fetchone()
            if row:
                mapping[(doc_id, page)] = row["id"]
    for s in sources:
        key = (s.get("doc_id"), s.get("page"))
        if key in mapping and not s.get("image_id"):
            s["image_id"] = mapping[key]
    return sources
