"""Integration: per-user metadata filtering isolates private documents."""
import uuid

import app.retrieval as retrieval


def test_user_private_doc_isolated(db_ready, unique_collection, ingest_text):
    owner = f"user_{uuid.uuid4().hex[:8]}"
    other = f"user_{uuid.uuid4().hex[:8]}"
    doc_id, _ = ingest_text("Top secret launch date is July 2026.",
                            unique_collection, user_id=owner)

    # The owner (plus the shared corpus) can retrieve it.
    res = retrieval.retrieve("launch date", top_k=5, collection=unique_collection,
                             filters={"user_id": [owner, None]}, use_cache=False)
    assert doc_id in {r["doc_id"] for r in res["results"]}

    # Another user cannot.
    res2 = retrieval.retrieve("launch date", top_k=5, collection=unique_collection,
                              filters={"user_id": [other, None]}, use_cache=False)
    assert doc_id not in {r["doc_id"] for r in res2["results"]}


def test_shared_corpus_visible_to_all(db_ready, unique_collection, ingest_text):
    doc_id, _ = ingest_text("Public onboarding notes are here.", unique_collection)
    for filters in ({"user_id": [None]}, {"user_id": ["someuser", None]}):
        res = retrieval.retrieve("onboarding", top_k=5, collection=unique_collection,
                                 filters=filters, use_cache=False)
        assert doc_id in {r["doc_id"] for r in res["results"]}
