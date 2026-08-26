"""Unit tests for the Writer agent (LLM mocked)."""
from app.agents.writer import WriterAgent

BLOCKS = [
    {"citation": 1, "title": "doc", "content": "content one"},
    {"citation": 2, "title": "doc", "content": "content two"},
]


def test_build_messages_formats_context():
    msgs = WriterAgent()._build_messages("q", BLOCKS, "")
    system = msgs[0]["content"]
    assert "[1]" in system and "content one" in system
    assert "[2]" in system and "content two" in system
    assert msgs[-1]["content"] == "q"


def test_build_messages_no_context():
    msgs = WriterAgent()._build_messages("q", [], "")
    assert "no matching documents" in msgs[0]["content"]


def test_build_messages_includes_memory():
    msgs = WriterAgent()._build_messages("q", BLOCKS, "Recent:\nuser: hi")
    assert any("Relevant conversation history" in m["content"] for m in msgs)


def test_build_messages_includes_feedback():
    msgs = WriterAgent()._build_messages("q", BLOCKS, "", feedback=["fix the citation"])
    assert "fix the citation" in msgs[-1]["content"]


def test_run_returns_llm_text(monkeypatch):
    monkeypatch.setattr("app.agents.writer.chat_text",
                        lambda messages, **kw: "the generated answer")
    assert WriterAgent().run("q", BLOCKS) == "the generated answer"


def test_run_passes_feedback_through(monkeypatch):
    seen = {}

    def capture(messages, **kw):
        seen["content"] = messages[-1]["content"]
        return "revised"

    monkeypatch.setattr("app.agents.writer.chat_text", capture)
    WriterAgent().run("q", BLOCKS, feedback=["be concise"])
    assert "be concise" in seen["content"]


def test_stream_yields_full_and_delta_pairs(monkeypatch):
    def fake_stream(messages, **kw):
        yield "a"
        yield "b"

    monkeypatch.setattr("app.llm.client.chat_stream", fake_stream)
    out = list(WriterAgent().stream("q", BLOCKS))
    assert out == [("a", "a"), ("ab", "b")]


def test_stream_empty_produces_one_empty_pair(monkeypatch):
    monkeypatch.setattr("app.llm.client.chat_stream", lambda messages, **kw: iter(()))
    assert list(WriterAgent().stream("q", BLOCKS)) == [("", "")]
