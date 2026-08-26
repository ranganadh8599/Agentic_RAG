"""Unit tests for citation / source-card formatting (pure functions)."""
from app.citation.formatter import best_snippet, format_sources

BLOCKS = [{
    "citation": 1, "title": "policy.pdf", "doc_id": 5,
    "content": "alpha beta gamma delta epsilon zeta eta theta",
    "rrf_score": 0.9, "rerank_confidence": 0.95, "metadata": {"page": 2},
}]


def test_best_snippet_empty_content():
    assert best_snippet("", "any answer") == ""


def test_best_snippet_is_substring_of_content():
    content = "alpha beta gamma delta epsilon zeta eta theta"
    snip = best_snippet(content, "beta gamma delta")
    assert snip in content


def test_best_snippet_no_match_returns_head():
    content = "alpha beta gamma"
    assert best_snippet(content, "zzz") == content


def test_best_snippet_respects_max_len():
    content = "alpha beta gamma delta epsilon"
    snip = best_snippet(content, "alpha beta", max_len=12)
    assert len(snip) <= 12


def test_format_sources_builds_cards():
    srcs = format_sources(BLOCKS, "answer")
    assert srcs[0]["citation"] == 1
    assert srcs[0]["doc_id"] == 5
    assert srcs[0]["page"] == 2
    assert srcs[0]["rerank_confidence"] == 0.95
    assert srcs[0]["score"] == 0.9
    assert "snippet" in srcs[0]


def test_format_sources_empty_blocks():
    assert format_sources([], "answer") == []


def test_format_sources_tolerates_missing_fields():
    srcs = format_sources([{"citation": 1, "content": "text here"}], "")
    assert srcs[0]["citation"] == 1
    assert srcs[0]["rerank_confidence"] == 0.0
