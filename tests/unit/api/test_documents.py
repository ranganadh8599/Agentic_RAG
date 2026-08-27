"""API tests for the document / collection endpoints."""
import uuid

import app.database.postgres as db


def _coll():
    return f"api_{uuid.uuid4().hex[:8]}"


def _cleanup(coll):
    cid = db.get_collection_id(coll)
    if cid is None:
        return
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM documents WHERE collection_id = %s", (cid,))
        cur.execute("DELETE FROM collections WHERE id = %s", (cid,))


def test_documents_list(client, db_ready):
    r = client.get("/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_documents_list_paginated(client, db_ready):
    r = client.get("/documents", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    assert len(r.json()) <= 5


def test_ingest_upload_creates_document(client, db_ready):
    coll = _coll()
    files = {"file": ("hello_upload.txt", b"hello from the upload test", "text/plain")}
    r = client.post("/ingest", files=files, data={"collection": coll})
    assert r.status_code == 200
    body = r.json()
    assert body["document_id"] > 0
    assert body["chunks"] > 0
    assert body["collection"] == coll
    _cleanup(coll)


def test_ingest_unsupported_type(client, db_ready):
    files = {"file": ("x.xyz", b"data", "application/octet-stream")}
    r = client.post("/ingest", files=files)
    assert r.status_code == 415


def test_ingest_duplicate_skipped(client, db_ready):
    coll = _coll()
    files = {"file": ("dup.txt", b"same content", "text/plain")}
    r1 = client.post("/ingest", files=files, data={"collection": coll})
    r2 = client.post("/ingest", files=files, data={"collection": coll})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json()["skipped"] is True
    _cleanup(coll)


def test_collections_create_list_and_idempotent(client, db_ready):
    name = f"coll_{uuid.uuid4().hex[:8]}"
    r = client.post("/collections", json={"name": name})
    assert r.status_code == 200
    assert r.json()["created"] is True
    r = client.post("/collections", json={"name": name})
    assert r.json()["created"] is False
    names = [c["name"] for c in client.get("/collections").json()]
    assert name in names
    _cleanup(name)
