"""Integration: a document ingested through the real pipeline is retrievable.

The FTS (keyword) channel is real Postgres full-text search, so even with
mock embeddings the ingested document is found by its terms.
"""
import app.database.postgres as db
import app.retrieval as retrieval


def test_ingest_then_retrieve_finds_document(db_ready, unique_collection, ingest_text):
    text = "The refund policy allows returns within thirty days of purchase."
    doc_id, n = ingest_text(text, unique_collection)
    assert doc_id > 0 and n > 0

    res = retrieval.retrieve("refund policy", top_k=5,
                             collection=unique_collection, use_cache=False)
    doc_ids = {r["doc_id"] for r in res["results"]}
    assert doc_id in doc_ids


def test_retrieve_scoped_to_collection(db_ready, unique_collection, ingest_text):
    ingest_text("Unique token zebra quarantine.", unique_collection)
    other = db.get_or_create_collection(f"other_{unique_collection}")
    try:
        res = retrieval.retrieve("zebra quarantine", top_k=5,
                                 collection=f"other_{unique_collection}",
                                 use_cache=False)
        assert res["results"] == []
    finally:
        with db.get_conn().cursor() as cur:
            cur.execute("DELETE FROM collections WHERE id = %s", (other,))


def test_retrieve_reports_latency_breakdown(db_ready, unique_collection, ingest_text):
    ingest_text("A topic about embeddings and vectors.", unique_collection)
    res = retrieval.retrieve("embeddings", top_k=3,
                             collection=unique_collection, use_cache=False)
    lat = res.get("latency_ms") or {}
    assert "stage1" in lat and "total" in lat


def test_delta_update_matches_duplicates_and_refreshes_metadata(db_ready, unique_collection):
    """Delta updates must match duplicate identical chunks ONE-TO-ONE and refresh
    reused chunks' metadata (page/section) — regression for the hash-only matching
    bug flagged in the v1.0.2 review."""
    import app.ingestion.pipeline as ingest

    coll_id = db.get_or_create_collection(unique_collection)
    conn = db.get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (collection_id, title, source_type) "
                "VALUES (%s, %s, 'txt') RETURNING id", (coll_id, "dup.txt"))
            doc_id = cur.fetchone()["id"]

        def section(page):
            return {"text": "The revenue was ten million dollars. ",
                    "metadata": {"page": page, "section": "s"}}

        # Two sections with byte-identical text -> two identical chunks stored.
        assert ingest._delta_update(
            conn, doc_id, [section(3), section(5)], 40, 0) == (2, 0, 0)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chunks WHERE document_id = %s", (doc_id,))
            assert cur.fetchone()["n"] == 2

        # Removing ONE duplicate deletes exactly one stale row (not both) and
        # the survivor inherits the NEW page metadata.
        assert ingest._delta_update(conn, doc_id, [section(5)], 40, 0) == (0, 1, 1)
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM chunks WHERE document_id = %s", (doc_id,))
            rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["metadata"]["page"] == 5

        # Re-adding the duplicate grows the set again (one reuse, one insert).
        assert ingest._delta_update(
            conn, doc_id, [section(3), section(5)], 40, 0) == (1, 1, 0)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM chunks WHERE document_id = %s", (doc_id,))
            assert cur.fetchone()["n"] == 2
    finally:
        conn.close()
