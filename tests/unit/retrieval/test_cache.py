"""Unit tests for the semantic & retrieval caches (real Postgres, mock
embeddings — deterministic because mock embeddings are hash-based)."""
import pytest

import app.database.postgres as db
from app.llm.embeddings import embed_texts
from app.retrieval.cache import (clear_retrieval_cache, retrieval_cache_lookup,
                                 retrieval_cache_store, semantic_cache_lookup,
                                 semantic_cache_store)


@pytest.fixture(autouse=True)
def _clean_caches(db_ready):
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM semantic_cache")
        cur.execute("DELETE FROM retrieval_cache")
    yield


def _emb(text):
    return embed_texts([text])[0]


# --- semantic cache ----------------------------------------------------------

def test_semantic_cache_store_and_lookup(db_ready):
    q = "what is the refund policy"
    semantic_cache_store(q, _emb(q), "the answer", "mock", [], None, None)
    row = semantic_cache_lookup(_emb(q), None, None)
    assert row is not None
    assert row["response"] == "the answer"


def test_semantic_cache_miss_for_different_query(db_ready):
    semantic_cache_store("query A", _emb("query A"), "answer A", "mock", [], None, None)
    # mock embeddings are near-orthogonal for different text → below threshold.
    assert semantic_cache_lookup(_emb("completely unrelated topic"), None, None) is None


def test_semantic_cache_scoped_per_user(db_ready):
    q = "shared query"
    semantic_cache_store(q, _emb(q), "for userA", "mock", [], None, "userA")
    assert semantic_cache_lookup(_emb(q), None, "userA") is not None
    # other user's bucket is isolated
    assert semantic_cache_lookup(_emb(q), None, "userB") is None
    # global/anonymous bucket does not see userA's private entry
    assert semantic_cache_lookup(_emb(q), None, None) is None


def test_semantic_cache_scoped_per_collection(db_ready):
    q = "query"
    semantic_cache_store(q, _emb(q), "coll 1 answer", "mock", [], 1, None)
    assert semantic_cache_lookup(_emb(q), 1, None) is not None
    assert semantic_cache_lookup(_emb(q), 2, None) is None


# --- retrieval cache ---------------------------------------------------------

def test_retrieval_cache_store_and_lookup(db_ready):
    q = "popular query"
    results = [{"id": 1, "content": "x"}]
    retrieval_cache_store(q, _emb(q), results, 0.99, None, None)
    got = retrieval_cache_lookup(_emb(q), None, None)
    assert got is not None
    assert got["results"] == results
    assert got["best_score"] == 0.99


def test_retrieval_cache_threshold_guards(db_ready):
    q = "popular query"
    retrieval_cache_store(q, _emb(q), [{"id": 1}], 0.99, None, None)
    # a far query is below RETRIEVAL_CACHE_THRESHOLD (0.97) → miss
    assert retrieval_cache_lookup(_emb("something very different"), None, None) is None


def test_clear_retrieval_cache(db_ready):
    q = "q"
    retrieval_cache_store(q, _emb(q), [{"id": 1}], 0.99, None, None)
    assert retrieval_cache_lookup(_emb(q), None, None) is not None
    clear_retrieval_cache()
    assert retrieval_cache_lookup(_emb(q), None, None) is None
