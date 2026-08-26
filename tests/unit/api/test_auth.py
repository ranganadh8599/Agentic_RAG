"""API tests for the auth endpoints (register / login / me / logout)."""
import uuid


def _user():
    return f"user_{uuid.uuid4().hex[:10]}"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "pgvector" in body


def test_api_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    assert "max_upload_mb" in r.json()


def test_root_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200


def test_register_login_me_logout(client):
    u = _user()
    assert client.post("/api/register",
                       json={"username": u, "password": "pass1234"}).status_code == 200

    r = client.post("/api/login", json={"username": u, "password": "pass1234"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert token

    me = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == u

    assert client.post("/api/logout",
                       headers={"Authorization": f"Bearer {token}"}).status_code == 200
    # token is revoked after logout
    assert client.get("/api/me",
                      headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_register_duplicate_username_conflict(client):
    u = _user()
    client.post("/api/register", json={"username": u, "password": "pass1234"})
    r = client.post("/api/register", json={"username": u, "password": "pass1234"})
    assert r.status_code == 409


def test_register_short_password_rejected(client):
    r = client.post("/api/register", json={"username": _user(), "password": "ab"})
    assert r.status_code == 409


def test_login_wrong_password(client):
    r = client.post("/api/login", json={"username": _user(), "password": "wrong"})
    assert r.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/api/me").status_code == 401


def test_register_password_min_length_enforced(client):
    # 7 chars < configured minimum (8) -> rejected
    r = client.post("/api/register", json={"username": _user(), "password": "abcdefg"})
    assert r.status_code == 409
    # exactly the configured minimum -> accepted
    r = client.post("/api/register", json={"username": _user(), "password": "abcdefgh"})
    assert r.status_code == 200


def test_client_ip_ignores_spoofed_xff_unless_proxy_trusted(monkeypatch):
    from types import SimpleNamespace

    from app.api.dependencies import client_ip
    from app.core.config import settings

    req = SimpleNamespace(
        headers={"x-forwarded-for": "6.6.6.6"},
        client=SimpleNamespace(host="127.0.0.1"),
    )
    # Default (no trusted proxy): a spoofed X-Forwarded-For header is ignored.
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    assert client_ip(req) == "127.0.0.1"
    # Behind a trusted proxy the header is honored.
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    assert client_ip(req) == "6.6.6.6"
