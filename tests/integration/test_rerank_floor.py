"""Integration: the rerank-confidence floor drops clear retrieval noise.

The cross-encoder is mocked (it is never loaded in the hermetic suite); only the
floor logic inside hybrid.retrieve is under test.
"""
import pytest

import app.retrieval as retrieval
from app.core.config import settings

_FAKE_RERANKED = [
    {"id": 101, "content": "Acme revenue is 2.4M", "rerank_score": 5.0,
     "rerank_confidence": 0.99},
    {"id": 102, "content": "noise chunk one", "rerank_score": 1.0,
     "rerank_confidence": 0.10},
    {"id": 103, "content": "noise chunk two", "rerank_score": 0.5,
     "rerank_confidence": 0.05},
    {"id": 104, "content": "noise chunk three", "rerank_score": 0.0,
     "rerank_confidence": 0.01},
]


@pytest.fixture
def _rerank_on(monkeypatch):
    monkeypatch.setattr(settings, "USE_RERANKER", True)
    monkeypatch.setattr(settings, "RERANKER_CANDIDATES", 20)
    monkeypatch.setattr(settings, "RERANK_CONFIDENCE_FLOOR", 0.5)
    monkeypatch.setattr(settings, "RERANK_CONFIDENCE_MIN_KEEP", 2)
    monkeypatch.setattr(
        "app.retrieval.hybrid.rerank.rerank",
        lambda q, docs, top_n, return_all=False: list(_FAKE_RERANKED))


def test_floor_drops_low_confidence_noise(db_ready, unique_collection, ingest_text,
                                          _rerank_on):
    ingest_text("Acme Analytics revenue is 2.4 million.", unique_collection)
    res = retrieval.retrieve("acme revenue", top_k=5,
                             collection=unique_collection, use_cache=False)
    ids = [r["id"] for r in res["results"]]
    assert 101 in ids          # confident chunk served
    assert 102 not in ids      # below-floor noise dropped
    assert 103 not in ids
    assert 104 not in ids


def test_floor_disabled_serves_all(db_ready, unique_collection, ingest_text,
                                   monkeypatch):
    monkeypatch.setattr(settings, "USE_RERANKER", True)
    monkeypatch.setattr(settings, "RERANK_CONFIDENCE_FLOOR", 0.0)
    monkeypatch.setattr(
        "app.retrieval.hybrid.rerank.rerank",
        lambda q, docs, top_n, return_all=False: list(_FAKE_RERANKED))
    ingest_text("Acme Analytics revenue is 2.4 million.", unique_collection)
    res = retrieval.retrieve("acme revenue", top_k=5,
                             collection=unique_collection, use_cache=False)
    ids = [r["id"] for r in res["results"]]
    assert 101 in ids and 102 in ids and 103 in ids and 104 in ids


def test_floor_weak_match_falls_back_to_min_keep(db_ready, unique_collection,
                                                 ingest_text, monkeypatch):
    monkeypatch.setattr(settings, "USE_RERANKER", True)
    monkeypatch.setattr(settings, "RERANK_CONFIDENCE_FLOOR", 0.999)  # nobody meets it
    monkeypatch.setattr(settings, "RERANK_CONFIDENCE_MIN_KEEP", 2)
    monkeypatch.setattr(
        "app.retrieval.hybrid.rerank.rerank",
        lambda q, docs, top_n, return_all=False: list(_FAKE_RERANKED))
    ingest_text("Acme Analytics revenue is 2.4 million.", unique_collection)
    res = retrieval.retrieve("acme revenue", top_k=5,
                             collection=unique_collection, use_cache=False)
    ids = [r["id"] for r in res["results"]]
    assert 101 in ids and 102 in ids   # best MIN_KEEP served, never empty
    assert 103 not in ids
