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
