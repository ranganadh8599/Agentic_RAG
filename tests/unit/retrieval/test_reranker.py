"""Unit tests for the cross-encoder reranker (model mocked).

The heavy sentence-transformers model is never loaded in unit tests — we
monkeypatch `_get_model` with a fake that returns canned scores. One real-model
benchmark lives in tests/evaluation/.
"""
import pytest

from app.retrieval import reranker


class _FakeModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs, **kwargs):
        return self._scores


def _docs(n):
    return [{"id": i, "content": f"chunk {i} content"} for i in range(n)]


def _install_model(monkeypatch, scores):
    monkeypatch.setattr("app.retrieval.reranker._get_model",
                        lambda: _FakeModel(scores))


def test_rerank_returns_exactly_top_n(monkeypatch):
    _install_model(monkeypatch, [0.9, 0.1, 0.5, 0.7, 0.3])
    out = reranker.rerank("q", _docs(5), top_n=3)
    assert len(out) == 3


def test_rerank_orders_by_score_desc(monkeypatch):
    _install_model(monkeypatch, [0.9, 0.1, 0.5, 0.7, 0.3])
    out = reranker.rerank("q", _docs(5), top_n=5)
    scores = [d["rerank_score"] for d in out]
    assert scores == sorted(scores, reverse=True)


def test_rerank_attaches_confidence(monkeypatch):
    _install_model(monkeypatch, [0.9, 0.5])
    out = reranker.rerank("q", _docs(2), top_n=2)
    for d in out:
        assert 0.0 <= d["rerank_confidence"] <= 1.0


def test_rerank_empty_docs(monkeypatch):
    _install_model(monkeypatch, [])
    assert reranker.rerank("q", [], top_n=5) == []


def test_rerank_top_n_larger_than_candidates(monkeypatch):
    _install_model(monkeypatch, [0.8, 0.2])
    out = reranker.rerank("q", _docs(2), top_n=10)
    assert len(out) == 2


def test_rerank_no_duplicate_docs(monkeypatch):
    _install_model(monkeypatch, [0.1, 0.9, 0.5, 0.7])
    out = reranker.rerank("q", _docs(4), top_n=4)
    ids = [d["id"] for d in out]
    assert len(ids) == len(set(ids))


def test_rerank_return_all(monkeypatch):
    _install_model(monkeypatch, [0.1, 0.9])
    out = reranker.rerank("q", _docs(2), top_n=1, return_all=True)
    assert len(out) == 2  # whole ranked list, not just top_n


def test_rerank_failure_keeps_fusion_order(monkeypatch):
    def boom():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr("app.retrieval.reranker._get_model", boom)
    docs = _docs(3)
    out = reranker.rerank("q", docs, top_n=2)
    assert len(out) == 2  # degrades gracefully, no crash
