# Agentic RAG - shared FastAPI dependencies & helpers.
#
# Reusable pieces used by the route modules: bearer-token / user resolution,
# pagination normalization, DB row counting, and the auth rate limiter.

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request

import mongo
from app.core.config import settings


def bearer_token(request: Request) -> str | None:
    """Extract the bearer token from the Authorization header (or None)."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_user(request: Request) -> dict:
    """Resolve the authenticated user from the token (401 if missing/invalid)."""
    user = mongo.user_from_token(bearer_token(request))
    if not user:
        raise HTTPException(status_code=401, detail="not authenticated")
    return user


def page_params(limit: int | None, offset: int | None) -> tuple[int | None, int]:
    """Normalize optional limit/offset for list endpoints.

    Returns (limit, offset). limit=None keeps the existing unbounded behaviour
    (the UI sidebar lists everything); when a limit is given it is clamped to
    PAGE_LIMIT_CAP so a single request can't pull the whole table into memory."""
    offset = max(offset or 0, 0)
    if limit is None or limit <= 0:
        return None, offset
    return min(int(limit), settings.PAGE_LIMIT_CAP), offset


def count_rows(table: str) -> int:
    """Row count for a table (used by /health)."""
    import db
    with db.get_conn().cursor() as cur:
        cur.execute(f"SELECT count(*) AS n FROM {table}")
        return cur.fetchone()["n"]


# --- Auth rate limiting ------------------------------------------------------
# Fixed-window throttle per client IP for the (unauthenticated) /api/login and
# /api/register endpoints, so a password can't be brute-forced at full speed.
# Attempts are counted at entry; a successful login/register clears the window
# for that IP. Like upload progress this is per-process — fine for the default
# single worker; for strict multi-worker enforcement front with a reverse proxy
# rate limit (nginx limit_req, etc.).
_auth_attempts: dict[str, deque[float]] = defaultdict(deque)
_auth_lock = Lock()


def client_ip(request: Request) -> str:
    """Best-effort client IP, honoring X-Forwarded-For when behind a proxy."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune_auth_attempts():
    # Bound memory: drop empty buckets once the map grows large.
    if len(_auth_attempts) < 5000:
        return
    for k in [k for k, dq in _auth_attempts.items() if not dq]:
        _auth_attempts.pop(k, None)


def auth_rate_limit(request: Request) -> None:
    """Raise 429 if this client has made too many login/register attempts."""
    ip = client_ip(request)
    now = monotonic()
    with _auth_lock:
        dq = _auth_attempts[ip]
        while dq and now - dq[0] > settings.AUTH_RATE_WINDOW:
            dq.popleft()
        _prune_auth_attempts()
        if len(dq) >= settings.AUTH_RATE_LIMIT:
            raise HTTPException(status_code=429,
                                detail="too many attempts — try again later")
        dq.append(now)


def auth_rate_clear(request: Request) -> None:
    """Reset the attempt window for this client on a successful login/register."""
    ip = client_ip(request)
    with _auth_lock:
        _auth_attempts.pop(ip, None)
