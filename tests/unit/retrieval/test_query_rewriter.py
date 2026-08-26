"""Unit tests for LLM query expansion (LLM mocked).

expand_query() generates QUERY_EXPANSION_VARIANTS total variants (including
the original), parsing lines and deduplicating case-insensitively. On any LLM
failure it degrades to just the original query.
"""
import pytest

from app.retrieval.query_rewriter import expand_query


def _install(monkeypatch, reply):
    monkeypatch.setattr("app.retrieval.query_rewriter.chat_text",
                        lambda messages, **kw: reply)


def test_returns_original_plus_variants(monkeypatch):
    _install(monkeypatch, "refund period length\nexchange window\n")
    out = expand_query("refund policy")
    assert out[0] == "refund policy"
    assert "refund period length" in out
    assert "exchange window" in out


def test_ignores_bullets_and_dashes(monkeypatch):
    _install(monkeypatch, "- variant one\n* variant two\n- variant three\n")
    out = expand_query("q")
    assert "variant one" in out
    assert "variant two" in out
    assert "variant three" in out


def test_dedupes_case_insensitively(monkeypatch):
    _install(monkeypatch, "REFUND POLICY\nRefund Policy\n")
    out = expand_query("refund policy")
    assert len(out) == 1  # both lines collapse into the original


def test_caps_total_variants(monkeypatch):
    _install(monkeypatch, "\n".join(f"variant {i}" for i in range(20)))
    out = expand_query("original")
    assert out[0] == "original"
    assert len(out) <= 5  # QUERY_EXPANSION_VARIANTS default


def test_llm_error_returns_original(monkeypatch):
    def boom(messages, **kw):
        raise RuntimeError("LLM down")
    monkeypatch.setattr("app.retrieval.query_rewriter.chat_text", boom)
    assert expand_query("q") == ["q"]


def test_empty_llm_reply_returns_original(monkeypatch):
    _install(monkeypatch, "")
    assert expand_query("q") == ["q"]


def test_duplicate_variants_not_repeated(monkeypatch):
    _install(monkeypatch, "same idea\nsame idea\n")
    out = expand_query("original")
    assert out.count("same idea") == 1
