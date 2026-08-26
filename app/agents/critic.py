# Agentic RAG - CriticAgent: hallucination/grounding check with a feedback loop.

import json
import logging

from app.core.config import settings
from app.llm.client import chat_text
from app.llm.prompts import CRITIC_PROMPT

log = logging.getLogger("agents")


class CriticUnavailableError(Exception):
    """Raised when the Critic agent cannot run (LLM error / malformed response).

    The orchestrator treats this as "grounding could NOT be verified": the
    answer is still delivered (a transient LLM outage must not brick chat) but
    it is explicitly flagged as UNVERIFIED in logs and in the API/UI response,
    instead of the old behaviour of silently treating a failed critic as a pass.
    The deterministic citation backstop (sanitize_citations) still runs in
    every case."""


class CriticAgent:
    def review(self, query: str, context_text: str, answer: str) -> tuple[bool, list[str]]:
        try:
            resp = chat_text(
                [{"role": "system", "content": CRITIC_PROMPT},
                 {"role": "user", "content": (
                     f"QUERY:\n{query}\n\nCONTEXT:\n{context_text}\n\nANSWER:\n{answer}")}],
                temperature=0.0,
                max_tokens=settings.CRITIC_MAX_TOKENS,
            )
            data = json.loads(resp)
            verdict = str(data.get("verdict", "fail")).lower()
            issues = data.get("issues") or []
            return verdict == "pass", issues
        except Exception as exc:  # noqa: BLE001
            # Fail CLOSED: a critic that cannot run must never report "pass".
            # Surface the failure loudly so the orchestrator can flag the
            # answer as unverified rather than claiming it was grounded.
            log.error("Critic review failed (grounding NOT verified): %s", exc)
            raise CriticUnavailableError("critic could not run") from exc
