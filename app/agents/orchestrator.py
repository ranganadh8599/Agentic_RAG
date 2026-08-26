# Agentic RAG - multi-agent orchestration.
#
#   RouterAgent    -> classify the query (rag | summary | vision | general)
#   RetrieverAgent -> hybrid retrieval (+ query expansion)
#   WriterAgent    -> grounded answer with [n] citations
#   CriticAgent    -> hallucination/grounding check, feedback loop
#   OrchestratorAgent -> coordinates the above + memory + semantic cache

import logging
import time

from app.agents.critic import CriticAgent, CriticUnavailableError
from app.agents.retriever import RetrieverAgent
from app.agents.router import RouterAgent, is_greeting
from app.agents.writer import WriterAgent
from app.citation.formatter import attach_page_images, format_sources
from app.citation.validator import validated_citations
from app.core.config import settings
from app.core.logging import fmt_table
from app.llm.prompts import GENERAL_PROMPT, GREETING_PROMPT, REWRITE_PROMPT
from llm import chat_text
import app.retrieval as retrieval
import memory

log = logging.getLogger("agents")


class OrchestratorAgent:
    def __init__(self):
        self.router = RouterAgent()
        self.retriever = RetrieverAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    # -- shared pre/post steps -------------------------------------------------

    def _memory_for(self, conversation_id, query_emb) -> str:
        if not conversation_id:
            return ""
        mem = memory.get_smart_context(conversation_id, query_emb)
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in mem["recent"][-6:])
        relevant = "\n".join(f"{m['role']}: {m['content']}" for m in mem["relevant"])
        parts = []
        if recent:
            parts.append(f"Recent:\n{recent}")
        if relevant:
            parts.append(f"Also relevant from earlier:\n{relevant}")
        return "\n\n".join(parts)

    def _rewrite_query(self, query: str, conversation_id: str | None) -> str:
        """Multi-turn support: rewrite a follow-up question into a standalone,
        self-contained query using the conversation history so retrieval and
        routing can resolve pronouns/ellipsis.

        "what does it do?"  ->  "What does RAGAS do?"

        Returns the original query unchanged when there is no history, the
        feature is disabled, or the LLM says the query is already standalone.
        The original text is still persisted to memory and shown to the user;
        only retrieval/routing/generation use the resolved query."""
        if not settings.USE_QUERY_REWRITE or not conversation_id:
            return query
        try:
            # Called BEFORE the current turn is persisted, so get_recent holds
            # only PRIOR turns. At least one prior exchange is required for a
            # rewrite to make sense (turn 1 has no history -> no rewrite).
            recent = memory.get_recent(conversation_id, k=8)
            if not recent:
                return query
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            if not transcript.strip():
                return query
            out = (chat_text([
                {"role": "system", "content": REWRITE_PROMPT.format(
                    transcript=transcript, query=query)}]) or "").strip()
            # Defensive: never let a rewrite empty the query or turn it into
            # a refusal-style filler. Take the first non-empty line in case the
            # LLM rambles after the query, and only reject obvious refusals.
            # (Deliberately does NOT reject outputs starting with "I" etc. —
            # "I want to know what scoring means in RAGAS" is a valid rewrite.)
            first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
            if len(first) < 4:
                return query
            low = first.lower()
            if low.startswith(("no ", "cannot", "i can't", "i don't", "i'm not",
                               "not sure", "no rewrite", "please clarify")):
                return query
            return first
        except Exception:  # noqa: BLE001
            pass
        return query

    def _persist_user(self, conversation_id, query, q_emb):
        """Save the user's message immediately so it is never lost if the turn is
        interrupted."""
        if conversation_id:
            memory.add_message(conversation_id, "user", query, embedding=q_emb)

    def _persist(self, conversation_id, query, answer, q_emb, cached, sources=None,
                 collection_id=None, user_id=None, filters=None, shared_safe=True):
        if conversation_id:
            memory.add_message(conversation_id, "assistant", answer,
                               embedding=retrieval.embed_query(answer),
                               sources=sources)
        # Never write into the SHARED cache bucket (user_id=None) unless the
        # retrieval stayed inside the shared corpus — a private document's
        # answer must not leak to every user. Non-admins only ever write to
        # their own bucket, which is safe.
        shared_ok = shared_safe and (
            not filters or (
                set(filters) == {"user_id"}
                and isinstance(filters.get("user_id"), (list, tuple))
                and set(filters["user_id"]) == {None}
            ))
        can_cache = user_id is not None or shared_ok
        if not cached and q_emb is not None and can_cache:
            retrieval.semantic_cache_store(query, q_emb, answer, settings.LLM_MODEL,
                                           sources, collection_id, user_id)

    def _sources(self, blocks, answer=""):
        return format_sources(blocks, answer)

    def _cited_sources(self, answer, blocks):
        """Sanitize the answer's citations, then keep only sources actually
        cited as [n]. Returns (sources, sanitized_answer) so the caller can
        also persist the cleaned answer (e.g. into the semantic cache).
        Also attaches the page's stored image (if any) so the UI can display
        it alongside the text, even when the cited chunk is a text chunk."""
        return validated_citations(answer, blocks)

    def _attach_page_images(self, sources):
        """If a cited source points to a page that has a stored image, attach it."""
        return attach_page_images(sources)

    # -- non-streaming ----------------------------------------------------------

    def _greet(self, query, conversation_id):
        """Pure greeting: reply from the LLM immediately, skipping RAG entirely."""
        answer = chat_text([{"role": "system", "content": GREETING_PROMPT},
                            {"role": "user", "content": query}])
        if conversation_id:
            memory.add_message(conversation_id, "user", query)
            memory.add_message(conversation_id, "assistant", answer,
                               embedding=retrieval.embed_query(answer), sources=[])
        return answer

    def run(self, query: str, conversation_id: str | None = None, top_k: int | None = None,
            collection: str | None = None, filters=None, user_id: str | None = None) -> dict:
        # Greetings short-circuit: no embedding, no router, no RAG retrieval.
        if is_greeting(query):
            answer = self._greet(query, conversation_id)
            return {"answer": answer, "sources": [], "type": "greeting"}

        # Multi-turn: resolve follow-ups ("what does it do?") against the
        # conversation history so retrieval + routing see a self-contained
        # query. The original text is still persisted/shown to the user.
        _t0 = time.perf_counter()
        retrieval_query = self._rewrite_query(query, conversation_id)
        if retrieval_query != query:
            log.info("✏️  Follow-up resolved: %r → %r", query[:80], retrieval_query[:80])
        q_emb = retrieval.embed_query(retrieval_query) if conversation_id else None
        memory_text = self._memory_for(conversation_id, q_emb)
        kind = self.router.classify(retrieval_query)
        log.info("🧭 Query type: %s | user=%s collection=%s filters=%s",
                 kind, user_id, collection, bool(filters))
        self._persist_user(conversation_id, query, q_emb)
        # True when the Critic agent could not run, so this answer was delivered
        # WITHOUT grounding verification (never silently treated as verified).
        unverified = False

        result = self.retriever.run(retrieval_query, top_k=top_k,
                                    collection=collection, filters=filters,
                                    user_id=user_id)
        _lat = result.get("latency_ms") or {}
        log.info("📥 Fetched %d chunks | best match=%.2f | search=%.1fs rerank=%.1fs | cache=%s",
                 len(result.get("results") or []),
                 result.get("best_score") or 0.0,
                 (_lat.get("stage1") or 0) / 1000, (_lat.get("rerank") or 0) / 1000,
                 "hit" if (result.get("cached") or result.get("cached_retrieval")) else "none")

        # General-knowledge fallback: no STRONG document match (top vector cosine
        # below the threshold) AND the router says the question is general -> answer
        # from general knowledge (no citations). Doc questions that fail retrieval
        # are still handled by the grounded writer (it refuses rather than fabricates).
        best = result.get("best_score") or 0.0
        if (not result.get("cached") and kind == "general"
                and best < settings.GENERAL_STRONG_THRESHOLD):
            log.info("🌐 No strong document match (best=%.2f) → answering from general knowledge", best)
            answer = chat_text(
                [{"role": "system", "content": GENERAL_PROMPT},
                 {"role": "user", "content": retrieval_query}])
            if conversation_id:
                memory.add_message(conversation_id, "assistant", answer,
                                   embedding=retrieval.embed_query(answer),
                                   sources=[])
            return {"answer": answer, "sources": [], "type": "general"}

        if result.get("cached"):
            answer = result["cached"]
            blocks = []
            kind = "cache"
            sources = result.get("cached_sources") or []
            log.info("🗄️  Reused cached answer (%d chars)", len(answer))
        else:
            blocks = result.get("results", [])
            log.info("✍️  Generating answer from %d source chunks …", len(blocks))
            context_text = "\n".join(
                f"[{r['citation']}] {r['content']}" for r in blocks)
            answer = self.writer.run(retrieval_query, blocks, memory_text)

            # Critic feedback loop. If the critic cannot run, the answer is
            # delivered but flagged unverified (fail CLOSED, never silent pass).
            for _ in range(settings.MAX_CRITIC_ROUNDS):
                try:
                    ok, issues = self.critic.review(retrieval_query, context_text, answer)
                except CriticUnavailableError:
                    log.error("🔴 Critic unavailable — answer delivered WITHOUT "
                              "grounding verification")
                    unverified = True
                    break
                if ok or not issues:
                    break
                answer = self.writer.run(retrieval_query, blocks, memory_text, feedback=issues)
            sources, answer = self._cited_sources(answer, blocks)

        self._persist(conversation_id, query, answer, q_emb, result.get("cached"),
                      sources, result.get("collection_id"), user_id,
                      filters=result.get("filters"),
                      shared_safe=result.get("shared_safe", True))
        log.info("📝 Generated answer (%d chars):\n%s", len(answer or ""), answer or "")
        if sources:
            log.info("📚 Cited sources:\n%s", fmt_table(
                ["cit", "title", "page", "conf", "snippet"],
                [(s.get("citation"), (s.get("title") or "")[:40],
                  s.get("page"), s.get("rerank_confidence"),
                  (s.get("snippet") or "")[:48]) for s in sources]))
        log.info("✅ Answer ready: %d chars, %d sources | total %.1fs",
                 len(answer or ""), len(sources or []), time.perf_counter() - _t0)
        return {"answer": answer, "sources": sources, "type": kind,
                "unverified": unverified}

    # -- streaming --------------------------------------------------------------

    def run_stream(self, query: str, conversation_id: str | None = None,
                   top_k: int | None = None, collection: str | None = None,
                   filters=None, user_id: str | None = None):
        """Generator yielding events:
        {"type": "content", "delta": str} and finally {"type": "sources", ...}"""
        _t0 = time.perf_counter()
        # True when the Critic agent could not run, so this answer is delivered
        # WITHOUT grounding verification (never silently treated as verified).
        unverified = False
        # Greetings short-circuit: no embedding, no router, no RAG retrieval.
        if is_greeting(query):
            log.info("👋 Greeting reply (no RAG) | conv=%s", conversation_id)
            answer = self._greet(query, conversation_id)
            yield {"type": "content", "delta": answer}
            yield {"type": "sources", "sources": []}
            return

        # Multi-turn: resolve follow-ups ("what does it do?") against the
        # conversation history so retrieval + routing see a self-contained
        # query. The original text is still persisted/shown to the user.
        yield {"type": "status", "status": "understanding your question…"}
        retrieval_query = self._rewrite_query(query, conversation_id)
        if retrieval_query != query:
            log.info("✏️  Follow-up resolved: %r → %r", query[:80], retrieval_query[:80])
        q_emb = retrieval.embed_query(retrieval_query) if conversation_id else None
        memory_text = self._memory_for(conversation_id, q_emb)
        kind = self.router.classify(retrieval_query)
        log.info("🧭 Query type: %s | user=%s collection=%s filters=%s",
                 kind, user_id, collection, bool(filters))
        self._persist_user(conversation_id, query, q_emb)

        yield {"type": "status", "status": "searching your documents…"}
        result = self.retriever.run(retrieval_query, top_k=top_k,
                                    collection=collection, filters=filters,
                                    user_id=user_id)
        _lat = result.get("latency_ms") or {}
        log.info("📥 Fetched %d chunks | best match=%.2f | search=%.1fs rerank=%.1fs | cache=%s",
                 len(result.get("results") or []),
                 result.get("best_score") or 0.0,
                 (_lat.get("stage1") or 0) / 1000, (_lat.get("rerank") or 0) / 1000,
                 "hit" if (result.get("cached") or result.get("cached_retrieval")) else "none")
        if settings.USE_RERANKER and not result.get("cached"):
            yield {"type": "status", "status": "reranking the best matches…"}

        best = result.get("best_score") or 0.0
        if (not result.get("cached") and kind == "general"
                and best < settings.GENERAL_STRONG_THRESHOLD):
            log.info("🌐 No strong document match (best=%.2f) → answering from general knowledge", best)
            answer = chat_text(
                [{"role": "system", "content": GENERAL_PROMPT},
                 {"role": "user", "content": retrieval_query}])
            if conversation_id:
                memory.add_message(conversation_id, "assistant", answer,
                                   embedding=retrieval.embed_query(answer),
                                   sources=[])
            yield {"type": "content", "delta": answer}
            yield {"type": "sources", "sources": []}
            return

        if result.get("cached"):
            answer = result["cached"]
            sources = result.get("cached_sources") or []
            log.info("🗄️  Reused cached answer (%d chars)", len(answer))
            yield {"type": "content", "delta": answer}
        else:
            blocks = result.get("results", [])
            log.info("✍️  Generating answer from %d source chunks …", len(blocks))
            yield {"type": "status", "status": "writing your answer…"}
            full = ""
            for full_so_far, delta in self.writer.stream(retrieval_query, blocks, memory_text):
                full = full_so_far
                if delta:
                    yield {"type": "content", "delta": delta}
            # critic loop (no streaming for the rewrite pass). If the critic
            # cannot run, flag the answer as unverified (fail CLOSED).
            context_text = "\n".join(f"[{r['citation']}] {r['content']}" for r in blocks)
            answer = full
            for _ in range(settings.MAX_CRITIC_ROUNDS):
                try:
                    ok, issues = self.critic.review(retrieval_query, context_text, answer)
                except CriticUnavailableError:
                    log.error("🔴 Critic unavailable — answer delivered WITHOUT "
                              "grounding verification")
                    unverified = True
                    break
                if ok or not issues:
                    break
                answer = self.writer.run(retrieval_query, blocks, memory_text, feedback=issues)
                # The revision REPLACES the flawed first draft: tell the client
                # to drop everything streamed so far, then send the new text —
                # the user must never see draft + revision concatenated.
                yield {"type": "replace"}
                yield {"type": "content", "delta": answer}
            sources, answer = self._cited_sources(answer, blocks)

        self._persist(conversation_id, query, answer, q_emb, result.get("cached"),
                      sources, result.get("collection_id"), user_id,
                      filters=result.get("filters"),
                      shared_safe=result.get("shared_safe", True))
        log.info("📝 Generated answer (%d chars):\n%s", len(answer or ""), answer or "")
        if sources:
            log.info("📚 Cited sources:\n%s", fmt_table(
                ["cit", "title", "page", "conf", "snippet"],
                [(s.get("citation"), (s.get("title") or "")[:40],
                  s.get("page"), s.get("rerank_confidence"),
                  (s.get("snippet") or "")[:48]) for s in sources]))
        log.info("✅ Answer ready: %d chars, %d sources | total %.1fs",
                 len(answer or ""), len(sources or []), time.perf_counter() - _t0)
        if unverified:
            yield {"type": "unverified"}
        yield {"type": "sources", "sources": sources}
