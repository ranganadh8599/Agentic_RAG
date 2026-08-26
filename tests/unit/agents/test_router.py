"""Unit tests for the router agent.

The greeting detector (is_greeting) is pure and deterministic. Query
classification calls the LLM, so those tests monkeypatch chat_text — the
classifier is lru_cached, hence the cache_clear fixture.
"""
import pytest

from app.agents.router import RouterAgent, _router_classify, is_greeting


@pytest.fixture(autouse=True)
def _clear_router_cache():
    _router_classify.cache_clear()
    yield
    _router_classify.cache_clear()


# --- greeting detector (no LLM) ---------------------------------------------

@pytest.mark.parametrize("query", [
    "hi", "hello", "hey there", "good morning", "good evening", "yo!",
    "what's up", "whats up", "how are you", "hiya",
])
def test_is_greeting_true(query):
    assert is_greeting(query)


@pytest.mark.parametrize("query", [
    "", "What is the refund policy?", "Describe this document",
    "Summarize the report", "hello" * 20,  # too long to be a greeting
])
def test_is_greeting_false(query):
    assert not is_greeting(query)


# --- query classification (LLM mocked) --------------------------------------

def test_router_selects_rag(monkeypatch):
    monkeypatch.setattr("app.agents.router.chat_text", lambda messages, **kw: "rag")
    assert RouterAgent().classify("What is the refund policy?") == "rag"


def test_router_selects_summary(monkeypatch):
    monkeypatch.setattr("app.agents.router.chat_text", lambda messages, **kw: "summary")
    assert RouterAgent().classify("Summarize the document") == "summary"


def test_router_selects_vision(monkeypatch):
    monkeypatch.setattr("app.agents.router.chat_text", lambda messages, **kw: "vision")
    assert RouterAgent().classify("Describe chart.png") == "vision"


def test_router_selects_general(monkeypatch):
    monkeypatch.setattr("app.agents.router.chat_text", lambda messages, **kw: "general")
    assert RouterAgent().classify("What is the weather like?") == "general"


def test_router_defaults_to_rag_on_unknown_output(monkeypatch):
    monkeypatch.setattr("app.agents.router.chat_text",
                        lambda messages, **kw: "something-weird")
    assert RouterAgent().classify("query") == "rag"


def test_router_falls_back_to_rag_on_llm_error(monkeypatch):
    def boom(messages, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr("app.agents.router.chat_text", boom)
    assert RouterAgent().classify("any question") == "rag"


def test_router_passes_query_and_prompt_to_llm(monkeypatch):
    seen = {}

    def capture(messages, **kw):
        seen["messages"] = messages
        return "rag"

    monkeypatch.setattr("app.agents.router.chat_text", capture)
    RouterAgent().classify("What is RAG?")
    assert seen["messages"][0]["role"] == "user"
    assert "What is RAG?" in seen["messages"][0]["content"]
