"""Unit tests for BM25-style sparse retrieval.

The sparsevec schema is unavailable on this box (SPARSE_READY=False), so the
DB-backed search is tested for graceful degradation; the pure tokenization is
tested directly.
"""
import pytest

import app.database.postgres as db
from app.retrieval.sparse import sparse_search, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The RAG system and the embeddings") == ["rag", "system", "embeddings"]


def test_tokenize_keeps_codes_and_numbers():
    toks = tokenize("item sku-4471 in stock")
    assert "sku" in toks and "4471" in toks


def test_tokenize_drops_single_chars():
    assert tokenize("a b c") == []


def test_tokenize_handles_none():
    assert tokenize(None) == []


def test_sparse_search_disabled_returns_empty(db_ready):
    if db.SPARSE_READY:
        pytest.skip("sparse is enabled on this box")
    assert sparse_search("anything", top_k=5) == []
