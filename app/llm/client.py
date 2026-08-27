# Agentic RAG - unified multi-provider LLM chat client via LiteLLM.
# A single OpenAI-style interface in front of many providers
# (OpenAI, Gemini, Claude, Ollama, Cohere, etc.).
#
# "mock" models are supported for offline testing with no API keys.

import logging
import time

import litellm

from app.core.config import settings

log = logging.getLogger("llm")

# Let litellm drop unsupported params per provider instead of erroring.
litellm.drop_params = True


# ---------------------------------------------------------------------------
# Mock provider (offline testing, no API keys)
# ---------------------------------------------------------------------------

def _is_mock(model: str) -> bool:
    return model is None or str(model).strip().lower() == "mock"


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
    t0 = time.perf_counter()
    resp = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
    out = resp.choices[0].message.content
    log.debug("🧠 LLM call %s → %d chars in %.1fs", model, len(out or ""),
              time.perf_counter() - t0)
    return out


def chat_stream(messages, model=None, temperature=None, max_tokens=None, **kwargs):
    """Generator yielding text deltas from any provider."""
    model = model or settings.LLM_MODEL
    if _is_mock(model):
        yield _mock_chat(messages, **kwargs)
        return
    t0 = time.perf_counter()
    n = 0
    resp = chat(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                stream=True, **kwargs)
    for chunk in resp:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                n += len(delta.content)
                yield delta.content
    log.debug("🧠 LLM stream %s → %d chars in %.1fs", model, n, time.perf_counter() - t0)
