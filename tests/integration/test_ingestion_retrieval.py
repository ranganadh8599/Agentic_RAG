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
