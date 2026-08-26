# Agentic RAG - RouterAgent: classifies the query type
# (rag | summary | vision | greeting | general) before retrieval, plus the fast
# greeting detector that short-circuits RAG entirely.

import re
from functools import lru_cache

from app.core.config import settings
from app.llm.prompts import ROUTER_PROMPT
from llm import chat_text

_GREETING_RE = re.compile(
    r"^(?:hi+|hello+|hey+|yo+|hola|howdy|hiya|sup|wassup|whatsup|whats up|what up|"
    r"hi there|hello there|hey there|how are you(?: doing| today)?|how r u|"
    r"hows it going|good morning|good afternoon|good evening|good day|"
    r"greetings|namaste)[\s!?.]*$",
    re.IGNORECASE,
)


def is_greeting(query: str) -> bool:
    """Fast, cheap detector: is this message a pure greeting (no RAG needed)?"""
    if not query or len(query) > 40:
        return False
    # Normalize: lowercase, drop punctuation/apostrophes, collapse whitespace.
    q = re.sub(r"[^a-z\s]", "", query.lower())
    q = re.sub(r"\s+", " ", q).strip()
    return bool(q and _GREETING_RE.match(q))


@lru_cache(maxsize=settings.ROUTER_CACHE_SIZE)
def _router_classify(query: str) -> str:
    try:
        ans = chat_text(
            [{"role": "user",
              "content": f"{ROUTER_PROMPT}\n\nQuery: {query}"}],
            temperature=0.0,
            max_tokens=settings.ROUTER_MAX_TOKENS,
        ).strip().lower()
        if "vision" in ans:
            return "vision"
        if "summary" in ans:
            return "summary"
        if "greeting" in ans:
            return "greeting"
        if "general" in ans:
            return "general"
        return "rag"
    except Exception:  # noqa: BLE001
        return "rag"


class RouterAgent:
    def classify(self, query: str) -> str:
        """Classify the query type. Wrapped in an exact-query LRU cache so a
        repeated question skips the router's full LLM round-trip — routing runs
        before the retrieval/semantic caches can short-circuit a repeat."""
        return _router_classify(query)
