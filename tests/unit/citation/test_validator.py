"""Unit tests for citation validation (pure, deterministic).

validated_citations() sanitizes the answer, keeps only the sources actually
cited as [n], and attaches stored page images. cited_numbers() extracts the
[n] markers from an answer.
"""
from app.citation.validator import cited_numbers, validated_citations

BLOCKS = [
    {"citation": 1, "content": "The refund period is 30 days.",
     "title": "policy.pdf", "doc_id": 10, "metadata": {},
     "rrf_score": 0.9, "rerank_confidence": 0.95},
    {"citation": 2, "content": "Approvals over $500 need a manager.",
     "title": "policy.pdf", "doc_id": 10, "metadata": {},
     "rrf_score": 0.8, "rerank_confidence": 0.9},
]


def test_cited_numbers_extracts():
    assert cited_numbers("a [1] b [2,3] c") == {1, 2, 3}


def test_cited_numbers_none():
    assert cited_numbers("no citations here") == set()


def test_validated_keeps_only_cited_sources():
    sources, answer = validated_citations(
        "The refund period is 30 days [1].", BLOCKS)
    assert [s["citation"] for s in sources] == [1]
    assert "[1]" in answer


def test_validated_source_metadata():
    sources, _answer = validated_citations(
        "The refund period is 30 days [1].", BLOCKS)
    assert sources[0]["title"] == "policy.pdf"
    assert sources[0]["doc_id"] == 10
    assert "rerank_confidence" in sources[0]
    assert "snippet" in sources[0]


def test_validated_drops_uncited_blocks():
    sources, answer = validated_citations(
        "Approvals need a manager [2].", BLOCKS)
    assert [s["citation"] for s in sources] == [2]
    assert answer == "Approvals need a manager [2]."


def test_validated_out_of_range_citation_removed():
    sources, answer = validated_citations(
        "The refund period is 30 days [9].", BLOCKS)
    assert sources == []
    assert "[9]" not in answer


def test_validated_empty_answer():
    sources, answer = validated_citations("", BLOCKS)
    assert sources == []
    assert answer == ""
