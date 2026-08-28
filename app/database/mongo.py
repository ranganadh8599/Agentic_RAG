# Agentic RAG - MongoDB persistence for users, sessions & chat history.
# Conversations & messages live in MongoDB; the RAG vector store —
# documents/chunks/collections/semantic cache — stays in Postgres.
#
# Login is real but basic: username + password, PBKDF2-hashed, with an opaque
# bearer session token. All chat/user/history operations are defensive: if
# MongoDB is unreachable, chat still works (persistence degrades gracefully),
# only login/history calls fail.

import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

from bson import ObjectId
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.core.config import settings

log = logging.getLogger("mongo")

_client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
_db = _client[settings.MONGO_DB]
_users = _db["users"]
_sessions = _db["sessions"]
_conversations = _db["conversations"]
_messages = _db["messages"]


def init_db():
    """Ensure indexes (safe to call at startup; no-op if Mongo is down)."""
    try:
        _users.create_index("username", unique=True)
        _sessions.create_index("token", unique=True)
        _sessions.create_index([("expires_at", 1)], expireAfterSeconds=0)
        _conversations.create_index([("user_id", 1), ("created_at", -1)])
        _messages.create_index([("conversation_id", 1), ("_id", 1)])
    except Exception as exc:  # noqa: BLE001 — auth indexes (unique user/token, session TTL)
        log.warning("Mongo init_db failed (auth indexes may be missing): %s", exc)


def _now():
    return datetime.now(timezone.utc)


def _oid(s):
    try:
        return ObjectId(s)
    except Exception:  # noqa: BLE001
        return None


def _hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt, settings.PBKDF2_ITERATIONS)


# ---------------------------------------------------------------------------
# Users & sessions
# ---------------------------------------------------------------------------

def register_user(username: str, display_name: str | None, password: str) -> dict:
    username = (username or "").strip().lower()
    if not username or not password:
        raise ValueError("username and password are required")
    if len(password) < settings.AUTH_MIN_PASSWORD_LEN:
        raise ValueError(
            f"password must be at least {settings.AUTH_MIN_PASSWORD_LEN} characters")
    salt = os.urandom(16)
    doc = {
        "username": username,
        "display_name": (display_name or username).strip() or username,
        "password_hash": _hash(password, salt).hex(),
        "password_salt": salt.hex(),
        "is_admin": False,
        "created_at": _now(),
    }
    try:
        res = _users.insert_one(doc)
    except DuplicateKeyError:
        raise ValueError("username already taken")
    return {"id": str(res.inserted_id), "username": username, "display_name": doc["display_name"]}


def login_user(username: str, password: str):
    """Verify credentials, create a session. Returns {user, token} or None."""
    username = (username or "").strip().lower()
    user = _users.find_one({"username": username})
    if not user:
        return None
    dk = _hash(password, bytes.fromhex(user["password_salt"]))
    if not hmac.compare_digest(dk.hex(), user["password_hash"]):
        return None
    token = secrets.token_urlsafe(32)
    _sessions.insert_one({
        "token": token,
        "user_id": str(user["_id"]),
        "created_at": _now(),
        "expires_at": _now() + timedelta(seconds=settings.SESSION_TTL_SECONDS),
    })
    return {
        "user": {"id": str(user["_id"]), "username": user["username"],
                 "display_name": user["display_name"],
                 "is_admin": bool(user.get("is_admin"))},
        "token": token,
    }


def change_password(user_id: str, current_password: str, new_password: str,
                    keep_token: str | None = None) -> dict:
    """Verify the current password and set a new one. Also revokes every other
    session so other devices must re-login. Returns {"ok": True} or
    {"ok": False, "error": "..."}."""
    try:
        user = _users.find_one({"_id": _oid(user_id)})
        if not user:
            return {"ok": False, "error": "user not found"}
        dk = _hash(current_password, bytes.fromhex(user["password_salt"]))
        if not hmac.compare_digest(dk.hex(), user["password_hash"]):
            return {"ok": False, "error": "current password is incorrect"}
        if not new_password or len(new_password) < settings.AUTH_MIN_PASSWORD_LEN:
            return {"ok": False,
                    "error": f"new password must be at least {settings.AUTH_MIN_PASSWORD_LEN} characters"}
        salt = os.urandom(16)
        _users.update_one(
            {"_id": user["_id"]},
            {"$set": {"password_hash": _hash(new_password, salt).hex(),
                       "password_salt": salt.hex()}})
        # Revoke all other sessions (keep this one so the user stays logged in).
        try:
            if keep_token:
                _sessions.delete_many({"user_id": user_id, "token": {"$ne": keep_token}})
            else:
                _sessions.delete_many({"user_id": user_id})
        except Exception as exc:  # noqa: BLE001 — failing to revoke sessions weakens password changes
            log.warning("password-change session revocation failed: %s", exc)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("change_password failed: %s", exc)
        return {"ok": False, "error": "could not change password"}


def logout(token: str | None):
    if token:
        try:
            _sessions.delete_many({"token": token})
        except Exception as exc:  # noqa: BLE001
            log.warning("logout failed to revoke session token: %s", exc)


def user_from_token(token: str | None):
    if not token:
        return None
    try:
        sess = _sessions.find_one({"token": token, "expires_at": {"$gt": _now()}})
        if not sess:
            return None
        user = _users.find_one({"_id": _oid(sess["user_id"])})
        if not user:
            return None
        return {"id": str(user["_id"]), "username": user["username"],
                "display_name": user["display_name"],
                "is_admin": bool(user.get("is_admin"))}
    except Exception as exc:  # noqa: BLE001 — never fail open on a token lookup error
        log.warning("session/token lookup failed: %s", exc)
        return None


def get_user(user_id: str):
    try:
        u = _users.find_one({"_id": _oid(user_id)})
        return {"id": str(u["_id"]), "username": u["username"],
                "display_name": u["display_name"],
                "is_admin": bool(u.get("is_admin"))} if u else None
    except Exception:  # noqa: BLE001
        return None


def set_admin(username: str, is_admin: bool = True) -> dict | None:
    """Grant/revoke the admin role for a user (by username). Returns the updated
    user or None when the user doesn't exist. Admins share the global/shared
    retrieval+semantic cache; regular users are scoped to their own."""
    try:
        u = _users.find_one_and_update(
            {"username": (username or "").strip().lower()},
            {"$set": {"is_admin": bool(is_admin)}},
            return_document=ReturnDocument.AFTER,
        )
        if not u:
            return None
        return {"id": str(u["_id"]), "username": u["username"],
                "display_name": u["display_name"],
                "is_admin": bool(u.get("is_admin"))}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Conversations & messages
# ---------------------------------------------------------------------------

def count_conversations() -> int:
    try:
        return _conversations.count_documents({})
    except Exception:  # noqa: BLE001
        return 0


def count_messages() -> int:
    try:
        return _messages.count_documents({})
    except Exception:  # noqa: BLE001
        return 0


def create_conversation(title: str = "New chat", user_id: str | None = None) -> str:
    try:
        res = _conversations.insert_one({
            "user_id": user_id, "title": title or "New chat",
            "created_at": _now(), "updated_at": _now(),
        })
        return str(res.inserted_id)
    except Exception:  # noqa: BLE001
        return ""


def conversation_owner(conversation_id: str):
    try:
        conv = _conversations.find_one({"_id": _oid(conversation_id)}, {"user_id": 1})
        return conv["user_id"] if conv else None
    except Exception:  # noqa: BLE001
        return None


def add_message(conversation_id: str, role: str, content: str, embedding=None,
                sources=None):
    try:
        # Cast to native Python floats — BSON cannot encode numpy.float32 scalars,
        # so embedding arrays from the retrieval layer MUST be converted here
        # (otherwise every insert fails silently and history is lost).
        emb = [float(x) for x in embedding] if embedding is not None else None
        _messages.insert_one({
            "conversation_id": str(conversation_id), "role": role, "content": content,
            "embedding": emb, "created_at": _now(),
            # Persist the grounded sources with the assistant message so citations
            # (source cards) survive switching chats / reloading the page.
            "sources": [dict(s) for s in (sources or [])] or None,
        })
        _conversations.update_one(
            {"_id": _oid(conversation_id)}, {"$set": {"updated_at": _now()}})
    except Exception:  # noqa: BLE001
        pass


def get_conversations(user_id: str, limit: int | None = None,
                      offset: int | None = None) -> list[dict]:
    try:
        q = _conversations.find({"user_id": user_id}) \
            .sort([("created_at", -1), ("_id", -1)])
        if offset and offset > 0:
            q = q.skip(offset)
        if limit and limit > 0:
            q = q.limit(limit)
        convs = list(q)
        out = []
        for c in convs:
            first = _messages.find_one(
                {"conversation_id": str(c["_id"]), "role": "user"}, sort=[("_id", 1)])
            n = _messages.count_documents({"conversation_id": str(c["_id"])})
            out.append({
                "id": str(c["_id"]),
                "title": c.get("title") or "New chat",
                "created_at": c.get("created_at"),
                "messages": n,
                "preview": (first or {}).get("content"),
            })
        return out
    except Exception:  # noqa: BLE001
        return []


def get_conversation_messages(conversation_id: str, user_id: str | None = None):
    """Returns a list of messages, None if not found, False if not owned."""
    try:
        conv = _conversations.find_one({"_id": _oid(conversation_id)})
        if not conv:
            return None
        if user_id is not None and conv.get("user_id") and conv["user_id"] != user_id:
            return False
        msgs = list(_messages.find({"conversation_id": str(conversation_id)})
                    .sort("_id", 1))
        return [{"id": str(m["_id"]), "role": m["role"], "content": m["content"],
                 "created_at": m.get("created_at"),
                 "sources": m.get("sources") or []} for m in msgs]
    except Exception:  # noqa: BLE001
        return None


def delete_conversation(conversation_id: str, user_id: str | None = None):
    try:
        conv = _conversations.find_one({"_id": _oid(conversation_id)})
        if not conv:
            return "not_found"
        if user_id is not None and conv.get("user_id") and conv["user_id"] != user_id:
            return "forbidden"
        _messages.delete_many({"conversation_id": str(conversation_id)})
        _conversations.delete_one({"_id": _oid(conversation_id)})
        return "ok"
    except Exception:  # noqa: BLE001
        return "error"


# ---------------------------------------------------------------------------
# Conversation memory (recent + relevant), computed in Python over the
# conversation's own messages (no local vector index needed).
# ---------------------------------------------------------------------------

def get_recent(conversation_id: str, k: int = 10) -> list[dict]:
    try:
        rows = list(_messages.find({"conversation_id": str(conversation_id)})
                    .sort("_id", -1).limit(k))
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    except Exception:  # noqa: BLE001
        return []


def get_relevant(conversation_id: str, query_emb, k: int = 5) -> list[dict]:
    if query_emb is None:
        return []
    try:
        import numpy as np
        q = np.asarray(query_emb, dtype="float32")
        rows = list(_messages.find({"conversation_id": str(conversation_id),
                                    "embedding": {"$ne": None}}))
        scored = []
        for r in rows:
            e = r.get("embedding")
            if not e:
                continue
            v = np.asarray(e, dtype="float32")
            denom = (np.linalg.norm(q) * np.linalg.norm(v)) + 1e-9
            sim = float(np.dot(q, v) / denom)
            scored.append((sim, r))
        scored.sort(key=lambda x: -x[0])
        return [{"role": r["role"], "content": r["content"]} for _s, r in scored[:k]]
    except Exception:  # noqa: BLE001
        return []


def get_smart_context(conversation_id: str, query_emb,
                      recent_k: int | None = None, relevant_k: int | None = None) -> dict:
    recent_k = recent_k or settings.MEMORY_RECENT_K
    relevant_k = relevant_k or settings.MEMORY_RELEVANT_K
    return {
        "recent": get_recent(conversation_id, recent_k),
        "relevant": get_relevant(conversation_id, query_emb, relevant_k),
    }
