# Agentic RAG - ingestion pipeline: load -> chunk -> embed -> store in Postgres.
# Uses asymmetric document prefixes when enabled.

import hashlib
import os
import re

import psycopg

from app.core.config import settings
from app.ingestion.chunking import chunk_text
from app.llm.embeddings import embed_texts
from app.retrieval import sparse
import app.database.postgres as db
import app.ingestion.loaders as loaders
import app.retrieval as retrieval

# Control characters that PostgreSQL text fields cannot store (NUL) or that are
# invalid/unwanted. Keep tab (\t), newline (\n), carriage return (\r).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Filenames that signal a short structured document (resume / CV / form),
# from the comma-separated STRUCTURED_KEYWORDS env setting.
_STRUCTURED_KEYWORDS = tuple(
    k.strip() for k in settings.STRUCTURED_KEYWORDS.split(",") if k.strip())


def sanitize_text(s: str) -> str:
    """Remove control characters that Postgres text can't store (e.g. NUL)."""
    return _CONTROL_RE.sub("", s or "")


def _looks_structured(filename: str, total_chars: int) -> bool:
    """Heuristic: short docs or resume/form-like files get coherent large chunks.
    Resumes/forms are short, section-based documents; fragmenting them at 500
    chars breaks sections mid-sentence and confuses the LLM (e.g. mixing up
    education dates with employment)."""
    name = (filename or "").lower()
    is_short = total_chars <= settings.STRUCTURED_MAX_CHARS
    has_keyword = any(k in name for k in _STRUCTURED_KEYWORDS)
    return is_short or has_keyword


def _chunk_params(title: str, sections) -> tuple[int, int]:
    """Pick (chunk_size, chunk_overlap) for a document."""
    total = sum(len(s.get("text", "")) for s in sections)
    if _looks_structured(title, total):
        return settings.STRUCTURED_CHUNK_SIZE, settings.STRUCTURED_CHUNK_OVERLAP
    return settings.CHUNK_SIZE, settings.CHUNK_OVERLAP

LOADERS = {
    ".pdf": loaders.load_pdf,
    ".png": loaders.load_image,
    ".jpg": loaders.load_image,
    ".jpeg": loaders.load_image,
    ".gif": loaders.load_image,
    ".webp": loaders.load_image,
    ".bmp": loaders.load_image,
    ".txt": loaders.load_text,
    ".md": loaders.load_text,
    ".csv": loaders.load_text,
    ".json": loaders.load_text,
    ".docx": loaders.load_docx,
    ".xlsx": loaders.load_xlsx,
    ".pptx": loaders.load_pptx,
}


def _store_section_image(conn, doc_id: int, sec: dict) -> int | None:
    """Persist a section's image to the `images` table. Returns its id or None."""
    img = sec.get("image")
    if not img:
        return None
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO images (document_id, page, mime_type, data) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (doc_id, img.get("page"), img.get("mime", "image/jpeg"),
             psycopg.Binary(img["data"])),
        )
        return cur.fetchone()["id"]


def _build_chunks(conn, doc_id: int, sections, chunk_size: int, chunk_overlap: int):
    """Flatten sections into [(text, meta)] chunks (storing section images)."""
    payload = []
    for sec in sections:
        # Store any associated image so the UI can show it alongside sources.
        image_id = _store_section_image(conn, doc_id, sec)
        for i, c in enumerate(chunk_text(sec["text"], chunk_size, chunk_overlap)):
            c = sanitize_text(c)
            if not c:
                continue
            meta = dict(sec["metadata"])
            meta["chunk"] = i
            if image_id is not None and i == 0:
                meta["image_id"] = image_id
            payload.append((c, meta))
    return payload


def _chunk_insert(doc_id, content, meta, embedding, content_hash, sparse_vec, token_count):
    """SQL + params to insert a chunk row. Sparse columns are referenced ONLY
    when the schema has them (pgvector >= 0.7); otherwise omitted so ingest
    still works on older pgvector (SPARSE_READY=False). Returns the new row id
    (RETURNING id) so delta updates can keep an exact "keep" list."""
    if db.SPARSE_READY:
        return (
            "INSERT INTO chunks (document_id, content, chunk_index, metadata, embedding, "
            "content_hash, sparse_embedding, token_count) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (doc_id, content, meta.get("chunk"), db.to_json(meta), db.to_db_vec(embedding),
             content_hash, sparse_vec, token_count),
        )
    return (
        "INSERT INTO chunks (document_id, content, chunk_index, metadata, embedding, content_hash) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (doc_id, content, meta.get("chunk"), db.to_json(meta), db.to_db_vec(embedding),
         content_hash),
    )


def _delta_update(conn, doc_id: int, sections, chunk_size: int, chunk_overlap: int,
                  progress=print, on_progress=None) -> tuple[int, int, int]:
    """Re-ingest a document, reusing unchanged chunks and embedding only the
    changed ones. Returns (stored, reused, removed) counts.

    Matching is count-aware and metadata-aware so delta updates stay correct in
    two subtle cases:
      * duplicate identical chunks are matched ONE-TO-ONE (old "A A" -> new
        "A A" reuses both rows; old "A A" -> new "A" deletes the stale
        duplicate) — a plain hash->id map would silently collapse them;
      * a reused chunk's metadata (page / section / chunk ordinal) is refreshed
        from the new payload, so citations never point at stale locations when
        the same text moves between pages/sections.
    """
    # Flatten chunks WITHOUT storing images yet — images are stored below only
    # for sections whose content actually changed (avoids orphaned blobs on
    # repeated delta updates).
    payload = []  # (section_idx, text, meta)
    for sec_idx, sec in enumerate(sections):
        for i, c in enumerate(chunk_text(sec["text"], chunk_size, chunk_overlap)):
            c = sanitize_text(c)
            if not c:
                continue
            meta = dict(sec["metadata"])
            meta["chunk"] = i
            payload.append((sec_idx, c, meta))
    if not payload:
        return 0, 0, 0
    sec_idxs = [s for s, _t, _m in payload]
    texts = [t for _s, t, _m in payload]
    metas = [m for _s, _t, m in payload]
    new_hashes = [hashlib.sha256(c.encode("utf-8")).hexdigest() for c in texts]

    # Load existing chunks, backfilling any that predate the content_hash column
    # so a first delta on an old corpus is still efficient.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content, content_hash, chunk_index, metadata "
            "FROM chunks WHERE document_id = %s", (doc_id,))
        existing_rows = cur.fetchall()
    # Group existing ids by content hash (multi-set, order preserved) so
    # duplicate identical chunks are matched one-for-one instead of collapsing.
    existing_by_hash: dict[str, list[int]] = {}
    id_to_meta: dict[int, dict] = {}
    backfill = []
    for r in existing_rows:
        h = r["content_hash"]
        if not h:
            h = hashlib.sha256((r["content"] or "").encode("utf-8")).hexdigest()
            backfill.append((h, r["id"]))
        existing_by_hash.setdefault(h, []).append(r["id"])
        id_to_meta[r["id"]] = r["metadata"] or {}
    if backfill:
        with conn.cursor() as cur:
            cur.executemany("UPDATE chunks SET content_hash = %s WHERE id = %s", backfill)

    # One-to-one match: for each new chunk, claim an unused existing row with
    # the same content hash (preferring the same chunk ordinal so row identity
    # stays stable across re-orders). Anything unclaimed is a new chunk.
    reuse: dict[int, int] = {}   # new index -> existing chunk id
    used: set[int] = set()
    to_embed: list[int] = []
    for i, h in enumerate(new_hashes):
        pick = None
        pool = existing_by_hash.get(h) or []
        for cid in pool:  # prefer same chunk ordinal
            if cid not in used and id_to_meta.get(cid, {}).get("chunk") == metas[i].get("chunk"):
                pick = cid
                break
        if pick is None:
            for cid in pool:  # any unused row with this hash
                if cid not in used:
                    pick = cid
                    break
        if pick is not None:
            used.add(pick)
            reuse[i] = pick
        else:
            to_embed.append(i)
    reused = len(reuse)

    # Store images ONLY for sections whose FIRST chunk is new (a reused first
    # chunk already references an image). This matches fresh-ingest display
    # behavior (image attached to the section's first chunk) and never creates
    # orphaned image blobs on repeated updates.
    for i in to_embed:
        if metas[i].get("chunk") == 0:
            img_id = _store_section_image(conn, doc_id, sections[sec_idxs[i]])
            if img_id is not None:
                metas[i]["image_id"] = img_id

    # Embed + store ONLY the changed chunks (batched, incremental).
    stored = 0
    inserted_ids: list[int] = []
    batch = settings.EMBED_BATCH_SIZE
    for start in range(0, len(to_embed), batch):
        idxs = to_embed[start:start + batch]
        new_texts = [texts[i] for i in idxs]
        embed_in = ([settings.DOC_PREFIX + t for t in new_texts]
                    if settings.USE_ASYMMETRIC_PREFIX else new_texts)
        embeddings = embed_texts(embed_in)
        sparse_vecs, token_counts = sparse.build_doc_batch(conn, new_texts)
        with conn.cursor() as cur:
            for k, (i, e) in enumerate(zip(idxs, embeddings)):
                sql, params = _chunk_insert(
                    doc_id, texts[i], metas[i], e, new_hashes[i],
                    sparse_vecs[k], token_counts[k])
                cur.execute(sql, params)
                inserted_ids.append(cur.fetchone()["id"])
        stored += len(embeddings)
        pct = min(95, 30 + round(65 * stored / max(len(to_embed), 1)))
        msg = f"  ...updated {stored}/{len(to_embed)} changed chunks"
        _report(progress, on_progress, "embedding", pct, msg)

    # Refresh metadata on reused rows so page/section/ordinal stay current even
    # when the text is unchanged (moved between pages, renumbered sections).
    if reuse:
        with conn.cursor() as cur:
            for i, cid in reuse.items():
                cur.execute(
                    "UPDATE chunks SET metadata = %s, chunk_index = %s WHERE id = %s",
                    (db.to_json(metas[i]), metas[i].get("chunk"), cid))

    # Drop existing rows that were neither reused nor newly inserted. Deletion
    # is by ROW ID (not by hash), so duplicate identical chunks are handled one
    # row at a time and a stale duplicate is never kept just because its text
    # hash still exists elsewhere.
    removed = 0
    removed_contents = []
    keep_ids = list(used) + inserted_ids
    with conn.cursor() as cur:
        if keep_ids:
            cur.execute(
                "DELETE FROM chunks WHERE document_id = %s AND id <> ALL(%s) "
                "RETURNING content",
                (doc_id, keep_ids),
            )
        else:
            cur.execute(
                "DELETE FROM chunks WHERE document_id = %s RETURNING content",
                (doc_id,),
            )
        removed_contents = [r["content"] for r in cur.fetchall()]
        removed = len(removed_contents)
    # Keep sparse term statistics consistent with the removed chunks.
    sparse.remove_texts(conn, removed_contents)

    return stored, reused, removed


def _report(progress, on_progress, phase, percent, message=None):
    """Report structured progress to on_progress (dict) and a human message to
    progress (print). message=None logs only to on_progress (no CLI spam)."""
    if on_progress is not None:
        try:
            on_progress({"phase": phase, "percent": percent, "message": message or ""})
        except Exception:  # noqa: BLE001
            pass
    if message is not None and progress is not None:
        try:
            progress(message)
        except Exception:  # noqa: BLE001
            pass


def ingest_file(path: str, title: str | None = None, progress=print,
                skip_duplicates: bool = True, collection: str = "default",
                on_progress=None, update_existing: bool = False,
                user_id: str | None = None,
                ingested_by: str | None = None) -> tuple[int, int, dict]:
    """Ingest a file. Returns (document_id, chunk_count, info).

    info describes what happened:
      {"mode": "ingested" | "updated" | "skipped" | "empty"}, plus
      {"reused": int, "removed": int} when mode == "updated".

    The file is stored in the named collection (auto-created if missing).
    If skip_duplicates is True and a document with the same title already
    exists in that collection, the file is skipped.

    update_existing=True turns re-uploads of the same file into a DELTA update:
    chunks whose content is unchanged are reused (same content_hash, no
    re-embedding), only changed chunks are embedded + stored, and chunks that
    no longer appear are deleted. No duplicate chunks/embeddings.

    on_progress is an optional callable receiving {"phase", "percent",
    "message"} (e.g. extracting/chunking/embedding/done with a 0-100
    percent) so the UI can show live upload progress.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext or '(none)'}")

    title = title or os.path.basename(path)
    conn = db.get_conn()
    collection_id = db.get_or_create_collection(collection or "default")

    # Look up an existing document with the same name in this collection.
    existing_id = None
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM documents WHERE lower(title) = lower(%s) AND collection_id = %s LIMIT 1",
            (title, collection_id),
        )
        row = cur.fetchone()
    if row:
        if skip_duplicates:
            msg = f"  - skipped duplicate '{title}' (already exists as doc {row['id']})"
            _report(progress, on_progress, "done", 100, msg)
            return row["id"], 0, {"mode": "skipped"}
        if update_existing:
            existing_id = row["id"]
        # else: fall through and create a new (duplicate) document row.

    _report(progress, on_progress, "extracting", 10, f"  - reading '{title}'")
    sections = None
    if ext == ".pdf":
        sections = loaders.load_pdf(path, max_pages=settings.MAX_PAGES)
    else:
        sections = LOADERS[ext](path)
    # Skip files with no extractable text (empty files, blank pages): do NOT
    # create a document row so the caller can report "no content" clearly
    # instead of a misleading "skipped (duplicate)".
    if not sections or not any((s.get("text") or "").strip() for s in sections):
        msg = f"  - no extractable text in '{title}'; skipping"
        _report(progress, on_progress, "done", 100, msg)
        return 0, 0, {"mode": "empty"}
    _report(progress, on_progress, "extracting", 20, f"  - extracted text from {title}")

    source_type = ext.lstrip(".") or "file"

    # Delta update: reuse unchanged chunks (same content_hash), embed only the
    # changed ones, and drop chunks that are no longer present.
    if existing_id is not None:
        if user_id is not None or ingested_by is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET user_id = COALESCE(%s, user_id), "
                    "ingested_by = COALESCE(%s, ingested_by) WHERE id = %s",
                    (user_id, ingested_by, existing_id))
        chunk_size, chunk_overlap = _chunk_params(title, sections)
        stored, reused, removed = _delta_update(
            conn, existing_id, sections, chunk_size, chunk_overlap,
            progress, on_progress)
        msg = (f"  * updated '{title}' (doc {existing_id}): +{stored} embedded, "
               f"{reused} unchanged, -{removed} removed")
        _report(progress, on_progress, "done", 100, msg)
        retrieval.clear_retrieval_cache()
        retrieval.clear_semantic_cache(collection_id=collection_id)
        return existing_id, stored, {"mode": "updated", "reused": reused, "removed": removed}

    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (collection_id, title, source_type, source_path, user_id, ingested_by) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (collection_id, title, source_type, path, user_id, ingested_by),
            )
            doc_id = cur.fetchone()["id"]
    except psycopg.errors.UniqueViolation:
        # A concurrent upload of the same file in this table won the race — the
        # unique index on (collection_id, lower(title)) rejected this one, so it
        # is a duplicate: return the existing document instead of inserting again.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE lower(title) = lower(%s) "
                "AND collection_id = %s LIMIT 1",
                (title, collection_id),
            )
            row = cur.fetchone()
        if row:
            msg = f"  - skipped duplicate '{title}' (already exists as doc {row['id']})"
            _report(progress, on_progress, "done", 100, msg)
            return row["id"], 0, {"mode": "skipped"}
        raise

    # 1. Choose chunk size: short structured docs (resumes/forms) automatically
    #    get larger chunks so their sections stay intact.
    chunk_size, chunk_overlap = _chunk_params(title, sections)

    payload = _build_chunks(conn, doc_id, sections, chunk_size, chunk_overlap)
    chunk_texts = [c for c, _ in payload]
    meta_list = [m for _, m in payload]

    _report(progress, on_progress, "chunking", 30,
            f"  - chunked into {len(chunk_texts)} chunks")

    # 2+3. Embed AND store incrementally (batch by batch), so progress is
    # saved to Postgres as it goes. If the process is stopped mid-way, the
    # batches already embedded+stored are never lost.
    batch = settings.EMBED_BATCH_SIZE
    stored = 0
    total = max(len(chunk_texts), 1)
    for i in range(0, len(chunk_texts), batch):
        batch_texts = chunk_texts[i:i + batch]
        embed_in = ([settings.DOC_PREFIX + t for t in batch_texts]
                    if settings.USE_ASYMMETRIC_PREFIX else batch_texts)
        embeddings = embed_texts(embed_in)
        # BM25-sparse side (exact terms): sparsevec + term stats for this batch.
        sparse_vecs, token_counts = sparse.build_doc_batch(conn, batch_texts)
        with conn.cursor() as cur:
            for j, (c, m, e) in enumerate(zip(batch_texts, meta_list[i:i + batch], embeddings)):
                sql, params = _chunk_insert(
                    doc_id, c, m, e,
                    hashlib.sha256(c.encode("utf-8")).hexdigest(),
                    sparse_vecs[j], token_counts[j])
                cur.execute(sql, params)
        stored += len(embeddings)
        # Report every batch to on_progress (smooth 30% -> 95% progress); only
        # print to the CLI on the throttled schedule to avoid log spam.
        pct = min(95, 30 + round(65 * stored / total))
        msg = (f"  ...stored {stored}/{len(chunk_texts)} chunks"
               if (stored % 400 < batch or stored == len(chunk_texts)) else None)
        _report(progress, on_progress, "embedding", pct, msg)

    _report(progress, on_progress, "done", 100, f"  + {title}: {stored} chunks stored")

    # New content can make cached reranked chunk lists AND cached full answers
    # stale, so invalidate both (scoped to this collection) and let popular
    # queries re-run against fresh data.
    retrieval.clear_retrieval_cache()
    retrieval.clear_semantic_cache(collection_id=collection_id)
    return doc_id, stored, {"mode": "ingested"}


def ingest_path(path: str, title: str | None = None, progress=print,
                skip_duplicates: bool = True, collection: str = "default",
                update_existing: bool = False, user_id: str | None = None,
                ingested_by: str | None = None) -> tuple[int, int]:
    """Ingest a file OR every supported file inside a directory."""
    if os.path.isdir(path):
        total_docs = total_chunks = 0
        for root, _dirs, files in os.walk(path):
            for fname in sorted(files):
                fp = os.path.join(root, fname)
                if os.path.splitext(fname)[1].lower() in LOADERS:
                    doc_id, n, _info = ingest_file(fp, title=fname, progress=progress,
                                                   skip_duplicates=skip_duplicates,
                                                   collection=collection,
                                                   update_existing=update_existing,
                                                   user_id=user_id,
                                                   ingested_by=ingested_by)
                    if n > 0:
                        total_docs += 1
                    total_chunks += n
        return total_docs, total_chunks
    doc_id, n, _info = ingest_file(path, title=title, progress=progress,
                                   skip_duplicates=skip_duplicates,
                                   collection=collection,
                                   update_existing=update_existing,
                                   user_id=user_id,
                                   ingested_by=ingested_by)
    return doc_id, n
