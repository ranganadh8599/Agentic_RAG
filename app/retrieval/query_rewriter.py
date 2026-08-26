# Agentic RAG - LLM query rewriting / expansion.
#
# Generates QUERY_EXPANSION_VARIANTS total query variants (including the
# original), each embedded for dense search AND fed to keyword + sparse, so
# chunks that match the same intent under different wording still get
# retrieved.

import logging

from app.core.config import settings
from app.llm.prompts import EXPANSION_PROMPT
from llm import chat_text

log = logging.getLogger("retrieval")


def expand_query(query: str) -> list[str]:
    """LLM query expansion: semantically-equivalent rephrasings + keyword forms.

    Generates QUERY_EXPANSION_VARIANTS total variants (including the original),
    each embedded for dense search AND fed to keyword + sparse, so chunks that
    match the same intent under different wording still get retrieved.
    """
    extra = max(settings.QUERY_EXPANSION_VARIANTS - 1, 1)
    try:
        text = chat_text(
            [{"role": "system", "content": EXPANSION_PROMPT.format(n=extra)},
             {"role": "user", "content": f"Query: {query}"}],
            temperature=0.5,  # a little diversity so rephrasings differ
        )
        result = [query]
        for ln in text.splitlines():
            v = ln.strip().strip("-•*").strip()
            if not v or v.lower() == query.lower():
                continue
            if any(v.lower() == x.lower() for x in result):
                continue
            result.append(v)
            if len(result) >= settings.QUERY_EXPANSION_VARIANTS:
                break
        return result
    except Exception:  # noqa: BLE001
        return [query]
