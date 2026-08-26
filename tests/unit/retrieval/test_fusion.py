"""Unit tests for reciprocal-rank fusion (deterministic, pure).

RRF is rank-based, so the per-list scores in the input tuples are ignored —
only the position matters. Each entry is a (score, row) tuple where row has
an ``id`` key (matching the search-channel contract in app.retrieval).
"""
import pytest

from app.retrieval.fusion import rrf_fuse


def _row(rid):
    return {"id": rid, "title": f"doc {rid}"}


def _ranked(ids):
    """Build a ranked list [(score, row), ...] preserving order."""
    return [(1.0, _row(i)) for i in ids]


def test_empty_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_single_list_preserves_order():
    fused = rrf_fuse([_ranked(["A", "B", "C"])])
    assert [r["id"] for r in fused] == ["A", "B", "C"]


def test_doc_in_both_lists_ranks_first():
    dense = _ranked(["A", "B"])
    sparse = _ranked(["A", "C"])
    fused = rrf_fuse([dense, sparse])
    # A is 1st in both lists → clearly top.
    assert fused[0]["id"] == "A"


def test_known_rrf_ordering():
    # Dense:  A(1) B(2) C(3)     Sparse: B(1) A(2) D(3)
    dense = _ranked(["A", "B", "C"])
    sparse = _ranked(["B", "A", "D"])
    fused = rrf_fuse([dense, sparse])
    ids = [r["id"] for r in fused]
    # A and B both appear in both lists (tie) → top two; C, D only once → last two.
    assert set(ids[:2]) == {"A", "B"}
    assert set(ids[2:]) == {"C", "D"}


def test_doc_only_in_one_list_still_included():
    fused = rrf_fuse([_ranked(["A"]), _ranked(["B"])])
    assert {r["id"] for r in fused} == {"A", "B"}


def test_different_list_lengths():
    dense = _ranked(["A", "B", "C", "D"])
    sparse = _ranked(["B"])
    fused = rrf_fuse([dense, sparse])
    assert {r["id"] for r in fused} == {"A", "B", "C", "D"}
    # B appears in both → ranks above the others.
    assert fused[0]["id"] == "B"


def test_duplicate_id_within_one_list_kept_once_in_output():
    dense = _ranked(["A", "A", "B"])
    fused = rrf_fuse([dense])
    ids = [r["id"] for r in fused]
    assert ids.count("A") == 1


def test_rrf_score_attached():
    fused = rrf_fuse([_ranked(["A"])])
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] == 1.0 / (60 + 0 + 1)


def test_custom_k():
    fused = rrf_fuse([_ranked(["A", "B"]), _ranked(["B"])], k=10)
    b = [r for r in fused if r["id"] == "B"][0]
    # rank 0 in list2 → 1/11; rank 1 in list1 → 1/12
    assert abs(b["rrf_score"] - (1 / 11 + 1 / 12)) < 1e-9


def test_rows_preserved():
    fused = rrf_fuse([_ranked(["A"])])
    assert fused[0]["title"] == "doc A"
