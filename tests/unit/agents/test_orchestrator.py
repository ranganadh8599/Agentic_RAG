"""Unit tests for the Orchestrator agent decision flow.

The LLM, retriever, writer and critic are all mocked; tests use
conversation_id=None so no MongoDB / embedding writes happen. These verify
the *coordination* logic (which path is taken and in what order).
"""
import pytest

from app.agents.orchestrator import OrchestratorAgent

RESULT = {
    "results": [{
        "citation": 1, "content": "context block", "title": "doc",
        "doc_id": 7, "rrf_score": 0.9, "rerank_confidence": 0.9,
        "metadata": {},
    }],
    "best_score": 0.9,
    "latency_ms": {"stage1": 1.0, "rerank": 0.0, "total": 1.0},
    "collection_id": None, "filters": None, "shared_safe": True,
}


def test_run_greeting_short_circuits(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr("app.agents.orchestrator.chat_text",
                        lambda messages, **kw: "hi!")
    res = agent.run("hi", conversation_id=None)
    assert res["type"] == "greeting"
    assert res["answer"] == "hi!"


def test_run_rag_path_generates_and_cites(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr(agent.router, "classify", lambda q: "rag")
    monkeypatch.setattr(agent.retriever, "run", lambda q, **kw: dict(RESULT))
    monkeypatch.setattr(agent.writer, "run", lambda *a, **kw: "Answer with [1].")
    monkeypatch.setattr(agent.critic, "review", lambda *a, **kw: (True, []))

    res = agent.run("what is x", conversation_id=None)
    assert res["type"] == "rag"
    assert res["answer"] == "Answer with [1]."
    assert [s["citation"] for s in res["sources"]] == [1]
    assert res["unverified"] is False


def test_run_general_fallback_when_no_strong_match(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr(agent.router, "classify", lambda q: "general")
    weak = dict(RESULT, results=[], best_score=0.0)
    monkeypatch.setattr(agent.retriever, "run", lambda q, **kw: weak)
    monkeypatch.setattr("app.agents.orchestrator.chat_text",
                        lambda messages, **kw: "general knowledge answer")

    res = agent.run("hello world", conversation_id=None)
    assert res["type"] == "general"
    assert res["answer"] == "general knowledge answer"
    assert res["sources"] == []


def test_run_cached_answer_path(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr(agent.router, "classify", lambda q: "rag")
    cached = dict(RESULT, cached="cached answer", cached_sources=[], best_score=0.9)
    monkeypatch.setattr(agent.retriever, "run", lambda q, **kw: cached)

    res = agent.run("q", conversation_id=None)
    assert res["type"] == "cache"
    assert res["answer"] == "cached answer"


def test_run_critic_retry_loop(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr(agent.router, "classify", lambda q: "rag")
    monkeypatch.setattr(agent.retriever, "run", lambda q, **kw: dict(RESULT))
    calls = {"writer": 0, "critic": 0}

    def fake_writer(*a, **kw):
        calls["writer"] += 1
        return "draft [1]."

    def fake_critic(*a, **kw):
        calls["critic"] += 1
        return (False, ["fix it"]) if calls["critic"] == 1 else (True, [])

    monkeypatch.setattr(agent.writer, "run", fake_writer)
    monkeypatch.setattr(agent.critic, "review", fake_critic)

    res = agent.run("q", conversation_id=None)
    assert calls["writer"] == 2  # initial + one revision
    assert calls["critic"] == 2
    assert res["unverified"] is False


def test_run_unverified_when_critic_unavailable(monkeypatch):
    from app.agents.critic import CriticUnavailableError

    agent = OrchestratorAgent()
    monkeypatch.setattr(agent.router, "classify", lambda q: "rag")
    monkeypatch.setattr(agent.retriever, "run", lambda q, **kw: dict(RESULT))
    monkeypatch.setattr(agent.writer, "run", lambda *a, **kw: "draft [1].")

    def boom(*a, **kw):
        raise CriticUnavailableError("down")

    monkeypatch.setattr(agent.critic, "review", boom)

    res = agent.run("q", conversation_id=None)
    assert res["unverified"] is True


def test_rewrite_query_resolves_followup(monkeypatch):
    agent = OrchestratorAgent()
    # rewrite is off in the test env → re-enable for this unit test.
    monkeypatch.setattr("app.agents.orchestrator.settings.USE_QUERY_REWRITE", True)
    monkeypatch.setattr(
        "app.agents.orchestrator.memory.get_recent",
        lambda conv, k=8: [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG is a framework."},
        ])
    monkeypatch.setattr("app.agents.orchestrator.chat_text",
                        lambda messages, **kw: "How does RAG work?")

    out = agent._rewrite_query("How does it work?", "conv1")
    assert out == "How does RAG work?"


def test_rewrite_query_unchanged_without_history(monkeypatch):
    agent = OrchestratorAgent()
    monkeypatch.setattr("app.agents.orchestrator.settings.USE_QUERY_REWRITE", True)
    monkeypatch.setattr("app.agents.orchestrator.memory.get_recent",
                        lambda conv, k=8: [])
    assert agent._rewrite_query("What is RAG?", "conv1") == "What is RAG?"
