# Agentic RAG - ingestion pipeline: load -> chunk -> embed -> store in Postgres.
# Uses asymmetric document prefixes when enabled.

import os
import re

import psycopg

from config import settings
import db
from chunking import chunk_text
from llm import embed_texts
import loaders

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
                on_progress=None) -> tuple[int, int]:
    """Ingest a file. Returns (document_id, chunk_count).

    The file is stored in the named collection (auto-created if missing).
    If skip_duplicates is True and a document with the same title already
    exists in that collection, the file is skipped.

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

    # Skip if a document with the same name already exists in this collection.
    if skip_duplicates:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE lower(title) = lower(%s) AND collection_id = %s LIMIT 1",
                (title, collection_id),
            )
            row = cur.fetchone()
        if row:
            msg = f"  - skipped duplicate '{title}' (already exists as doc {row['id']})"
            _report(progress, on_progress, "done", 100, msg)
            return row["id"], 0

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
        return 0, 0
    _report(progress, on_progress, "extracting", 20, f"  - extracted text from {title}")

    source_type = ext.lstrip(".") or "file"

    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (collection_id, title, source_type, source_path) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (collection_id, title, source_type, path),
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
            return row["id"], 0
        raise

    # 1. Choose chunk size: short structured docs (resumes/forms) automatically
    #    get larger chunks so their sections stay intact.
    chunk_size, chunk_overlap = _chunk_params(title, sections)

    chunk_texts: list[str] = []
    meta_list: list[dict] = []
    for sec in sections:
        # Store any associated image so the UI can show it alongside sources.
        image_id = _store_section_image(conn, doc_id, sec)

        chunks = chunk_text(sec["text"], chunk_size, chunk_overlap)
        for i, c in enumerate(chunks):
            c = sanitize_text(c)
            if not c:
                continue
            meta = dict(sec["metadata"])
            meta["chunk"] = i
            if image_id is not None and i == 0:
                meta["image_id"] = image_id
            chunk_texts.append(c)
            meta_list.append(meta)

    _report(progress, on_progress, "chunking", 30,
            f"  - chunked into {len(chunk_texts)} chunks")

    # 2+3. Embed AND store incrementally (batch by batch), so progress is
    # saved to Postgres as it goes. If the process is stopped mid-way, the
    # batches already embedded+stored are never lost.
    batch = settings.EMBED_BATCH_SIZE
    stored = 0
    total = max(len(chunk_texts), 1)
    for i in range(0, len(chunk_texts), batch):
        texts = chunk_texts[i:i + batch]
        if settings.USE_ASYMMETRIC_PREFIX:
            texts = [settings.DOC_PREFIX + t for t in texts]
        embeddings = embed_texts(texts)
        with conn.cursor() as cur:
            for c, m, e in zip(chunk_texts[i:i + batch], meta_list[i:i + batch], embeddings):
                cur.execute(
                    "INSERT INTO chunks (document_id, content, chunk_index, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (doc_id, c, m.get("chunk"), db.to_json(m), db.to_db_vec(e)),
                )
        stored += len(embeddings)
        # Report every batch to on_progress (smooth 30% -> 95% progress); only
        # print to the CLI on the throttled schedule to avoid log spam.
        pct = min(95, 30 + round(65 * stored / total))
        msg = (f"  ...stored {stored}/{len(chunk_texts)} chunks"
               if (stored % 400 < batch or stored == len(chunk_texts)) else None)
        _report(progress, on_progress, "embedding", pct, msg)

    _report(progress, on_progress, "done", 100, f"  + {title}: {stored} chunks stored")
    return doc_id, stored


def ingest_path(path: str, title: str | None = None, progress=print,
                skip_duplicates: bool = True, collection: str = "default") -> tuple[int, int]:
    """Ingest a file OR every supported file inside a directory."""
    if os.path.isdir(path):
        total_docs = total_chunks = 0
        for root, _dirs, files in os.walk(path):
            for fname in sorted(files):
                fp = os.path.join(root, fname)
                if os.path.splitext(fname)[1].lower() in LOADERS:
                    doc_id, n = ingest_file(fp, title=fname, progress=progress,
                                            skip_duplicates=skip_duplicates,
                                            collection=collection)
                    if n > 0:
                        total_docs += 1
                    total_chunks += n
        return total_docs, total_chunks
    return ingest_file(path, title=title, progress=progress,
                       skip_duplicates=skip_duplicates, collection=collection)
