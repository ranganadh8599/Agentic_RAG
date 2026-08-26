"""Integration: retrieval produces context that the writer can ground on
(LLM mocked)."""
import app.retrieval as retrieval


def test_retrieve_then_writer_grounded(db_ready, unique_collection, ingest_text, monkeypatch):
    ingest_text("The CEO of Acme Corporation is Jane Smith.", unique_collection)
    from app.agents.writer import WriterAgent

    monkeypatch.setattr("app.agents.writer.chat_text",
                        lambda messages, **kw: "Jane Smith is the CEO [1].")

    res = retrieval.retrieve("Who is the CEO of Acme", top_k=3,
                             collection=unique_collection, use_cache=False)
    assert res["results"]

    answer = WriterAgent().run("Who is the CEO of Acme", res["results"])
    assert "Jane Smith" in answer
