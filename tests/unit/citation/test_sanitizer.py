"""Unit tests for citation sanitization (pure, deterministic).

sanitize_citations() is the hard backstop that drops out-of-range markers,
collapses duplicates, expands ranges, and prunes "padding" citations that
have no lexical overlap with the claim they are attached to.
"""
from app.citation.sanitizer import sanitize_citations

# Block 1 is about the sky, block 2 about refunds, block 3 about approvals.
BLOCKS = [
    {"citation": 1, "content": "The sky is blue and the ocean is deep."},
    {"citation": 2, "content": "The refund period is 30 days from purchase."},
    {"citation": 3, "content": "Expenses over $500 require manager approval."},
]


def test_empty_answer():
    assert sanitize_citations("", BLOCKS) == ""


def test_valid_citation_kept():
    out = sanitize_citations("The refund period is 30 days [2].", BLOCKS)
    assert "[2]" in out


def test_out_of_range_citation_dropped():
    out = sanitize_citations("The refund period is 30 days [99].", BLOCKS)
    assert "[99]" not in out
    assert "30 days" in out


def test_multiple_valid_citations_kept():
    out = sanitize_citations(
        "Refunds last 30 days [2]; approvals need a manager [3].", BLOCKS)
    assert "[2]" in out and "[3]" in out


def test_no_blocks_strips_all_citations():
    out = sanitize_citations("Some claim [1] and [2].", [])
    assert "[1]" not in out and "[2]" not in out


def test_duplicate_numbers_collapsed():
    out = sanitize_citations("Refund is 30 days [2,2].", BLOCKS)
    assert "[2,2]" not in out
    assert "[2]" in out


def test_range_expanded():
    # "refund period" overlaps block 2, "expenses" overlaps block 3 → both
    # survive the padding check, so the range [2-3] expands to [2,3].
    out = sanitize_citations("Refund period and expenses [2-3].", BLOCKS)
    assert "[2,3]" in out


def test_range_collapses_when_claim_overlaps_neither_block():
    # "Both rules apply" overlaps no block → the group collapses to the single
    # best-scoring citation (the padding backstop keeps at least one).
    out = sanitize_citations("Both rules apply [2-3].", BLOCKS)
    assert "[2,3]" not in out
    assert "[2]" in out or "[3]" in out


def test_padding_citation_pruned():
    # "sky is blue" claim cites [1,2]; block 2 is unrelated → pruned to [1].
    out = sanitize_citations("The sky is blue [1,2].", BLOCKS)
    assert "[2]" not in out
    assert "[1]" in out


def test_fully_unrelated_group_keeps_best_scoring():
    # Neither block overlaps "pizza"; the backstop must still keep the
    # highest-scoring one so the claim retains at least one citation.
    out = sanitize_citations("Pizza is delicious [2,3].", BLOCKS)
    assert "[2]" in out or "[3]" in out


def test_whitespace_not_left_by_dropped_marker():
    out = sanitize_citations("Refund 30 days [99] for all.", BLOCKS)
    assert "  " not in out
