"""Shared fixtures for the Agentic RAG test suite.

Unit tests mock the LLM and reranker so they run offline and fast. Tests that
need the database use the real local Postgres via the ``db_ready`` fixture
(``db.init_db()`` is idempotent).
"""
import os
import sys

import pytest

# Make the repo root importable even when pytest is invoked from a subdir.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session")
def db_ready():
    """Initialize the Postgres schema once per session (idempotent).

    Requires the local Postgres + pgvector service to be running (see
    README / environment notes). Returns the `app.database.postgres` module.
    """
    import app.database.postgres as db
    db.init_db()
    return db
