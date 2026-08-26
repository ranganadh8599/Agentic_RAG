"""E2E: a conversation continues across turns via a shared conversation_id."""
import uuid


def _auth_user(client):
    u = f"mt_{uuid.uuid4().hex[:8]}"
    client.post("/api/register", json={"username": u, "password": "pass1234"})
    token = client.post("/api/login",
                        json={"username": u, "password": "pass1234"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_multiturn_conversation(client, db_ready):
    auth = _auth_user(client)
    r1 = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is RAGAS?"}]},
        headers=auth)
    assert r1.status_code == 200
    conv = r1.json()["conversation_id"]
    assert conv

    r2 = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "What is RAGAS?"},
            {"role": "assistant", "content": "RAGAS is a framework."},
            {"role": "user", "content": "How does it work?"},
        ],
        "conversation_id": conv}, headers=auth)
    assert r2.status_code == 200
    assert r2.json()["conversation_id"] == conv


def test_conversation_not_found(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "conversation_id": "000000000000000000000000"})
    assert r.status_code == 404
