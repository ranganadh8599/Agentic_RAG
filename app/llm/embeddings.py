# Agentic RAG - unified embedding provider via LiteLLM.
# Embeds text with any litellm-supported embedding model; "mock" returns a
# deterministic pseudo-random vector for offline testing.

import hashlib
import logging
import time

import litellm
import numpy as np

from app.core.config import settings
from app.llm.client import _is_mock

log = logging.getLogger("llm")


# ---------------------------------------------------------------------------
# Mock provider (offline testing, no API keys)
# ---------------------------------------------------------------------------

def _mock_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-random vector derived from the text (for offline tests)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """Transient errors worth retrying: rate limits, 429/5xx, connection blips."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(k in name or k in msg for k in
               ("ratelimit", "rate limit", "429", "500", "502", "503", "504",
                "connection", "timeout", "temporarily", "overloaded", "internal server"))


def embed_texts(texts, model=None, _attempts: int = 3) -> list[np.ndarray]:
    """Embed a list of strings. Returns L2-normalized float32 vectors.

    Retries transient API errors (rate limits, 5xx, connection hiccups) with a
    short backoff so uploads don't fail when the embedding provider is
    momentarily throttled or flaky.
    """
    model = model or settings.EMBEDDING_MODEL
    texts = list(texts)
    if not texts:
        return []

    if _is_mock(model):
        return [_mock_embedding(t, settings.EMBEDDING_DIM) for t in texts]

    import time as _time
    delay = 0.5
    for attempt in range(1, _attempts + 1):
        try:
            t0 = _time.perf_counter()
            resp = litellm.embedding(model=model, input=texts)
            vecs = []
            for item in resp.data:
                emb = getattr(item, "embedding", None)
                if emb is None:
                    emb = item["embedding"]
                v = np.asarray(emb, dtype="float32")
                vecs.append(v / (np.linalg.norm(v) + 1e-9))
            log.debug("🧠 Embedding %d text(s) (%s) in %.1fs", len(texts), model,
                      _time.perf_counter() - t0)
            return vecs
        except Exception as exc:  # noqa: BLE001
            if attempt >= _attempts or not _is_retryable(exc):
                raise
            time.sleep(delay)
            delay *= 2


def embed_one(text: str, model=None) -> np.ndarray:
    return embed_texts([text], model=model)[0]
