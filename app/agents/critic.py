# Agentic RAG - CriticAgent: hallucination/grounding check with a feedback loop.

import json
import logging
import re

from app.core.config import settings
from app.llm.client import chat_text
from app.llm.prompts import CRITIC_PROMPT

log = logging.getLogger("agents")


def _parse_critic_json(resp: str) -> dict:
    """Parse the critic's JSON reply, tolerating markdown fences and prose.

    Providers (e.g. Gemini) sometimes wrap the JSON in ```json ... ``` fences
    or add prose around the object; a plain json.loads would fail-CLOSE on
    every such call. Strip fences, and if direct parsing fails, retry on the
    object between the first { and the last }. Truly malformed output (e.g.
    truncated mid-string) still raises, so the critic stays fail-CLOSED."""
    text = (resp or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


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
            data = _parse_critic_json(resp)
            verdict = str(data.get("verdict", "fail")).lower()
            issues = data.get("issues") or []
            return verdict == "pass", issues
        except Exception as exc:  # noqa: BLE001
            # Fail CLOSED: a critic that cannot run must never report "pass".
            # Surface the failure loudly so the orchestrator can flag the
            # answer as unverified rather than claiming it was grounded.
            log.error("Critic review failed (grounding NOT verified): %s", exc)
            raise CriticUnavailableError("critic could not run") from exc
