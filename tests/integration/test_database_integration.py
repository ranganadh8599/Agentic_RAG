"""Integration: Postgres data-layer basics against the real schema."""
import uuid

import app.database.postgres as db


def test_collection_create_and_list(db_ready):
    name = f"dbint_{uuid.uuid4().hex[:8]}"
    cid = db.get_or_create_collection(name)
    ids = [c["id"] for c in db.list_collections()]
    assert cid in ids
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM collections WHERE id = %s", (cid,))


def test_collection_idempotent(db_ready):
    name = f"dbint_{uuid.uuid4().hex[:8]}"
    a = db.get_or_create_collection(name)
    b = db.get_or_create_collection(name)
    assert a == b
    with db.get_conn().cursor() as cur:
        cur.execute("DELETE FROM collections WHERE id = %s", (a,))


def test_vector_roundtrip(db_ready):
    v = [0.1, 0.2, 0.3, 0.4]
    stored = db.to_db_vec(v)
    back = db.from_db_vec(stored)
    assert len(back) == 4


def test_connection_pool_checkout(db_ready):
    got = [db.get_conn() for _ in range(3)]
    for g in got:
        g.close()
    with db.get_conn().cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone() is not None
