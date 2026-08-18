# Agentic RAG - unified multi-provider LLM + embeddings via LiteLLM.
# One interface for OpenAI, Gemini, Claude, Ollama, Cohere, etc.
# (The routing idea is borrowed from the Portkey gateway repo: a single
#  OpenAI-style interface in front of many providers.)
#
# "mock" models are supported for offline testing with no API keys.

import hashlib

import litellm
import numpy as np

from config import settings

# Let litellm drop unsupported params per provider instead of erroring.
litellm.drop_params = True


# ---------------------------------------------------------------------------
# Mock provider (offline testing, no API keys)
# ---------------------------------------------------------------------------

def _is_mock(model: str) -> bool:
    return model is None or str(model).strip().lower() == "mock"


def _mock_embedding(text: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-random vector derived from the text (for offline tests)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype("float32")
    return v / (np.linalg.norm(v) + 1e-9)


def _mock_chat(messages, **kwargs) -> str:
    query = messages[-1]["content"] if messages else ""
    query = query if isinstance(query, str) else str(query)
    return (
        "[mock-llm] Configure a real provider in .env to get real answers. "
        f"You asked: {query[:120]}"
    )


# ---------------------------------------------------------------------------
# Chat / completion
# ---------------------------------------------------------------------------

def chat(messages, model=None, temperature=None, max_tokens=None, stream=False, **kwargs):
    """Call any provider via litellm. Returns the litellm response object."""
    model = model or settings.LLM_MODEL
    if _is_mock(model):
        raise RuntimeError("mock model cannot stream; use chat_text()")
    return litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        max_tokens=max_tokens or settings.MAX_TOKENS,
        stream=stream,
        **kwargs,
    )


def chat_text(messages, model=None, temperature=None, max_tokens=None, **kwargs) -> str:
    """Convenience: return just the text reply."""
    model = model or settings.LLM_MODEL
    if _is_mock(model):
        return _mock_chat(messages, **kwargs)
    resp = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
    return resp.choices[0].message.content


def chat_stream(messages, model=None, temperature=None, max_tokens=None, **kwargs):
    """Generator yielding text deltas from any provider."""
    model = model or settings.LLM_MODEL
    if _is_mock(model):
        yield _mock_chat(messages, **kwargs)
        return
    resp = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                stream=True, **kwargs)
    for chunk in resp:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content


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

    import time
    delay = 0.5
    for attempt in range(1, _attempts + 1):
        try:
            resp = litellm.embedding(model=model, input=texts)
            vecs = []
            for item in resp.data:
                emb = getattr(item, "embedding", None)
                if emb is None:
                    emb = item["embedding"]
                v = np.asarray(emb, dtype="float32")
                vecs.append(v / (np.linalg.norm(v) + 1e-9))
            return vecs
        except Exception as exc:  # noqa: BLE001
            if attempt >= _attempts or not _is_retryable(exc):
                raise
            time.sleep(delay)
            delay *= 2


def embed_one(text: str, model=None) -> np.ndarray:
    return embed_texts([text], model=model)[0]
