"""Unit tests for the Critic agent (fail-CLOSED grounding check).

The critic is a hard gate: any LLM error or malformed response must raise
CriticUnavailableError (never a silent pass). chat_text is mocked.
"""
import json

import pytest

from app.agents.critic import CriticAgent, CriticUnavailableError


def _fake(reply):
    return lambda messages, **kw: reply


def test_review_pass(monkeypatch):
    monkeypatch.setattr(
        "app.agents.critic.chat_text",
        _fake(json.dumps({"verdict": "pass", "issues": []})))
    ok, issues = CriticAgent().review("q", "context", "answer")
    assert ok is True
    assert issues == []


def test_review_fail_with_issues(monkeypatch):
    monkeypatch.setattr(
        "app.agents.critic.chat_text",
        _fake(json.dumps({"verdict": "fail", "issues": ["hallucination"]})))
    ok, issues = CriticAgent().review("q", "context", "answer")
    assert ok is False
    assert issues == ["hallucination"]


def test_review_malformed_json_raises(monkeypatch):
    monkeypatch.setattr("app.agents.critic.chat_text", _fake("not json"))
    with pytest.raises(CriticUnavailableError):
        CriticAgent().review("q", "context", "answer")


def test_review_llm_error_raises(monkeypatch):
    def boom(messages, **kw):
        raise RuntimeError("LLM down")
    monkeypatch.setattr("app.agents.critic.chat_text", boom)
    with pytest.raises(CriticUnavailableError):
        CriticAgent().review("q", "context", "answer")


def test_review_missing_verdict_treated_as_fail(monkeypatch):
    # No "verdict" key → defaults to fail (never assume pass).
    monkeypatch.setattr(
        "app.agents.critic.chat_text",
        _fake(json.dumps({"issues": []})))
    ok, _issues = CriticAgent().review("q", "context", "answer")
    assert ok is False


def test_review_uppercase_verdict_normalized(monkeypatch):
    monkeypatch.setattr(
        "app.agents.critic.chat_text",
        _fake(json.dumps({"verdict": "PASS", "issues": []})))
    ok, _issues = CriticAgent().review("q", "context", "answer")
    assert ok is True


def test_review_passes_query_context_and_answer_to_llm(monkeypatch):
    seen = {}

    def capture(messages, **kw):
        seen["content"] = messages[-1]["content"]
        return json.dumps({"verdict": "pass", "issues": []})

    monkeypatch.setattr("app.agents.critic.chat_text", capture)
    CriticAgent().review("my query", "the context", "the answer")
    assert "my query" in seen["content"]
    assert "the context" in seen["content"]
    assert "the answer" in seen["content"]
