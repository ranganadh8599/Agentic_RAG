"""E2E: a logged-in user's private upload is owned by and scoped to them."""
import uuid

import app.database.postgres as db


def test_authenticated_upload_is_owned_by_user(client, db_ready):
    u = f"iso_{uuid.uuid4().hex[:8]}"
    assert client.post("/api/register",
                       json={"username": u, "password": "pass1234"}).status_code == 200
    token = client.post("/api/login",
                        json={"username": u, "password": "pass1234"}).json()["token"]

    coll = f"iso_{uuid.uuid4().hex[:8]}"
    files = {"file": ("private.txt", b"Secret launch date is July 2026.", "text/plain")}
    r = client.post("/ingest", files=files, data={"collection": coll},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    doc_id = r.json()["document_id"]

    me = client.get("/api/me", headers={"Authorization": f"Bearer {token}"}).json()
    with db.get_conn().cursor() as cur:
        cur.execute("SELECT user_id FROM documents WHERE id = %s", (doc_id,))
        assert cur.fetchone()["user_id"] == me["id"]

    cid = db.get_collection_id(coll)
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM documents WHERE collection_id = %s", (cid,))
        cur.execute("DELETE FROM collections WHERE id = %s", (cid,))


def test_anonymous_upload_stays_public_ownerless(client, db_ready):
    coll = f"iso_{uuid.uuid4().hex[:8]}"
    files = {"file": ("shared.txt", b"Shared admin-style note.", "text/plain")}
    r = client.post("/ingest", files=files, data={"collection": coll})
    assert r.status_code == 200
    doc_id = r.json()["document_id"]
    with db.get_conn().cursor() as cur:
        cur.execute("SELECT user_id FROM documents WHERE id = %s", (doc_id,))
        assert cur.fetchone()["user_id"] is None
    cid = db.get_collection_id(coll)
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM documents WHERE collection_id = %s", (cid,))
        cur.execute("DELETE FROM collections WHERE id = %s", (cid,))
