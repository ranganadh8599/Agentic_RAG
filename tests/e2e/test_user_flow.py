"""E2E: upload a document through the API, then ask about it."""
import uuid

import app.database.postgres as db


def test_upload_and_ask(client, db_ready):
    coll = f"e2e_{uuid.uuid4().hex[:8]}"
    files = {"file": ("uptime.txt",
                      b"Server uptime target is ninety nine point nine percent.",
                      "text/plain")}
    r = client.post("/ingest", files=files, data={"collection": coll})
    assert r.status_code == 200
    assert r.json()["document_id"] > 0

    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is the uptime target?"}],
        "collection": coll})
    assert r.status_code == 200
    answer = r.json()["choices"][0]["message"]["content"]
    assert answer

    cid = db.get_collection_id(coll)
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM documents WHERE collection_id = %s", (cid,))
        cur.execute("DELETE FROM collections WHERE id = %s", (cid,))
