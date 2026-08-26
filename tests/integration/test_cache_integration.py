"""Integration: the retrieval-results cache serves repeat queries.

Seeded directly (mock embeddings are deterministic), then a real retrieve()
call must hit it and skip the search pipeline.
"""
import pytest

import app.database.postgres as db
import app.retrieval as retrieval
from app.retrieval.cache import clear_retrieval_cache, retrieval_cache_store


@pytest.fixture(autouse=True)
def _clean_caches(db_ready):
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM semantic_cache")
        cur.execute("DELETE FROM retrieval_cache")
    yield
    clear_retrieval_cache()


def test_retrieval_cache_serves_repeat(db_ready, unique_collection, ingest_text):
    ingest_text("Refund policy is thirty days.", unique_collection)
    q = "refund policy"
    emb = retrieval.embed_query(q)
    coll_id = db.get_collection_id(unique_collection)

    # Seed a strongly-grounded cached result for this collection + query.
    retrieval_cache_store(q, emb, [{"id": 1, "content": "x"}], 0.99, coll_id, None)

    res = retrieval.retrieve(q, top_k=3, collection=unique_collection, use_cache=True)
    assert res.get("cached_retrieval") is True
    # cached hits report ~zero pipeline latency
    assert res["latency_ms"]["total"] < 100


def test_retrieval_cache_not_used_when_bypassed(db_ready, unique_collection, ingest_text):
    ingest_text("Refund policy is thirty days.", unique_collection)
    q = "refund policy"
    emb = retrieval.embed_query(q)
    coll_id = db.get_collection_id(unique_collection)
    retrieval_cache_store(q, emb, [{"id": 1, "content": "x"}], 0.99, coll_id, None)

    res = retrieval.retrieve(q, top_k=3, collection=unique_collection, use_cache=False)
    assert res.get("cached_retrieval") is None
