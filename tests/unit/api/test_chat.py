"""API tests for the OpenAI-compatible chat endpoint.

The test env runs with mock LLM / embeddings, so the chat completes offline —
these tests verify the endpoint wiring (request → orchestrator → response),
not answer quality.
"""


def test_chat_returns_answer(client, db_ready):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is the refund policy?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"]
    assert "sources" in body
    assert body["conversation_id"]


def test_chat_requires_message(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_chat_stream_returns_sse(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "stream": True})
    assert r.status_code == 200
    assert "data:" in r.text
    assert "[DONE]" in r.text


def test_chat_resolves_same_conversation(client, db_ready):
    import uuid
    u = f"chat_{uuid.uuid4().hex[:8]}"
    client.post("/api/register", json={"username": u, "password": "pass1234"})
    token = client.post("/api/login",
                        json={"username": u, "password": "pass1234"}).json()["token"]
    auth = {"Authorization": f"Bearer {token}"}

    r1 = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "tell me about RAG"}]},
        headers=auth)
    conv = r1.json()["conversation_id"]
    assert conv
    r2 = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is RAGAS?"}],
        "conversation_id": conv}, headers=auth)
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv
