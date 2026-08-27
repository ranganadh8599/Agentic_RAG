"""Architecture contract tests.

Guard the dependency direction of the app package so a future edit cannot
silently introduce upward/bad coupling (e.g. ``database -> agents``). The
allowed flow is downward:

    api -> agents -> retrieval / ingestion -> database
                    (llm, core, schemas are leaf infrastructure)

A violation here fails CI instead of relying on developer discipline.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"

# Forbidden (from_package, to_package) edges. Package = the top-level dir
# under app/ (e.g. app/database/postgres.py -> "database").
FORBIDDEN_EDGES = {
    # database is the bottom storage layer: never reach up into services.
    ("database", "agents"),
    ("database", "api"),
    ("database", "retrieval"),
    ("database", "ingestion"),
    ("database", "citation"),
    ("database", "memory"),
    ("database", "llm"),
    # llm is a leaf consumed by services: nothing above it may import it back.
    ("llm", "agents"),
    ("llm", "api"),
    ("llm", "retrieval"),
    ("llm", "ingestion"),
    ("llm", "memory"),
    ("llm", "citation"),
    # retrieval is a service: never import the API or agent layers.
    ("retrieval", "api"),
    ("retrieval", "agents"),
    # core is the bottom infrastructure: nothing may import it.
    ("core", "database"),
    ("core", "llm"),
    ("core", "agents"),
    ("core", "api"),
    ("core", "retrieval"),
    ("core", "ingestion"),
    ("core", "citation"),
    ("core", "memory"),
    # schemas are DTOs: no application logic.
    ("schemas", "agents"),
    ("schemas", "api"),
    ("schemas", "retrieval"),
    ("schemas", "ingestion"),
    ("schemas", "database"),
    ("schemas", "llm"),
    ("schemas", "memory"),
}

# Legacy flat modules that the migration removed from the repo root; if one
# comes back, the modularization has regressed.
LEGACY_ROOT_MODULES = [
    "db.py", "mongo.py", "llm.py", "ingest.py", "loaders.py", "memory.py",
    "cli.py", "agents.py", "retrieval.py", "api.py", "config.py",
    "logging_config.py", "prompts.py", "chunking.py", "rerank.py", "sparse.py",
]


def _import_targets(path: Path):
    """Yield every `app.<pkg>...` module this file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("app."):
                yield node.module


def test_no_bad_coupling():
    violations = []
    for py in sorted(APP_ROOT.rglob("*.py")):
        rel = py.relative_to(REPO_ROOT)  # e.g. app/database/postgres.py
        parts = rel.parts
        if len(parts) < 3:  # skip app/__init__.py and app/main.py (composition root)
            continue
        from_pkg = parts[1]
        for mod in _import_targets(py):
            to_pkg = mod.split(".")[1]
            if (from_pkg, to_pkg) in FORBIDDEN_EDGES:
                violations.append(f"{rel} -> {mod}")
    assert not violations, (
        "Dependency direction violated (bad coupling):\n" + "\n".join(violations))


def test_no_legacy_root_modules_reintroduced():
    present = [f for f in LEGACY_ROOT_MODULES if (REPO_ROOT / f).exists()]
    assert not present, (
        f"Legacy root module(s) reintroduced — the modularization regressed: {present}")
