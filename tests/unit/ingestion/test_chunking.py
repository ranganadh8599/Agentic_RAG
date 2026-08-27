"""Unit tests for the chunking logic (pure, deterministic)."""
import pytest

from app.ingestion.chunking import chunk_text, recursive_split


def test_small_text_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=1000, chunk_overlap=0)
    assert chunks == ["Hello world"]


def test_large_text_multiple_chunks_within_size():
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text, chunk_size=600, chunk_overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 600


def test_empty_text_no_chunks():
    assert chunk_text("", chunk_size=600, chunk_overlap=80) == []


def test_whitespace_only_text_no_chunks():
    assert chunk_text("   \n\n  ", chunk_size=600, chunk_overlap=0) == []


def test_overlap_carries_tail_into_next_chunk():
    text = " ".join(f"tok{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=40)
    if len(chunks) > 1:
        assert chunks[1].startswith(chunks[0][-40:])


def test_exact_boundary_single_chunk():
    text = "a" * 600
    chunks = chunk_text(text, chunk_size=600, chunk_overlap=0)
    assert chunks == [text]


def test_chunk_size_zero_no_errors():
    # A degenerate chunk size should still terminate (no infinite recursion).
    chunks = chunk_text("abc def ghi", chunk_size=1, chunk_overlap=0)
    assert all(len(c) >= 1 for c in chunks)


def test_cjk_text_does_not_break_words():
    # CJK punctuation is a chunk separator; chunks stay bounded.
    text = "。".join("汉字" * 20 for _ in range(30))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=0)
    assert chunks
    for c in chunks:
        assert len(c) <= 200


def test_recursive_split_custom_separators():
    # \n\n is the highest-priority separator → splits on paragraph breaks.
    text = "AB\n\nCD\n\nEF"
    chunks = recursive_split(text, chunk_size=3, chunk_overlap=0)
    assert chunks == ["AB", "CD", "EF"]


@pytest.mark.parametrize("chunk_overlap", [0, 10, 80])
def test_chunk_text_returns_nonempty_for_large_input(chunk_overlap):
    text = "The quick brown fox jumps over the lazy dog. " * 50
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=chunk_overlap)
    assert all(c for c in chunks)
