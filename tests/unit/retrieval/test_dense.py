"""Unit tests for dense (bi-encoder) retrieval.

The embedding model is mock in the test env (deterministic hash vectors), so
these tests verify the *mechanics* of the channel — vector shape, L2
normalization, caching, top-K bounding, ordering, collection scoping — not
semantic relevance (that is the job of tests/evaluation with real models).
"""
import numpy as np
import pytest

import app.database.postgres as db
from app.core.config import settings
from app.retrieval.dense import embed_query, vector_search


def test_embed_query_shape_and_normalization():
    emb = embed_query("hello world")
    assert len(emb) == settings.EMBEDDING_DIM
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-4  # L2-normalized


def test_embed_query_is_deterministic_and_cached():
    a = embed_query("same query")
    b = embed_query("same query")
    assert np.allclose(a, b)
    c = embed_query("a different query")
    assert not np.allclose(a, c)


def test_vector_search_empty_collection(db_ready):
    assert vector_search(embed_query("anything"), top_k=5, collection_id=999999) == []


def test_vector_search_bounds_top_k_and_sorts(db_ready, unique_collection, ingest_text):
    ingest_text("The refund policy allows returns within thirty days.", unique_collection)
    coll_id = db.get_collection_id(unique_collection)
    emb = embed_query("refund policy")
    results = vector_search(emb, top_k=5, collection_id=coll_id)
    assert len(results) <= 5
    scores = [s for s, _row in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_search_collection_isolation(db_ready, unique_collection, ingest_text):
    ingest_text("Only here content.", unique_collection)
    other_coll = db.get_or_create_collection(f"iso_{unique_collection}")
    try:
        # the doc lives in unique_collection, so searching the OTHER collection
        # (with a query matching it) must not surface it.
        results = vector_search(embed_query("only here content"), top_k=5,
                                collection_id=other_coll)
        assert len(results) == 0
    finally:
        with db.get_conn().cursor() as cur:
            cur.execute("DELETE FROM collections WHERE id = %s", (other_coll,))
