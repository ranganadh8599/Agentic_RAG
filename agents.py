# Agentic RAG - multi-agent orchestration.
# A small hand-rolled agent system:
#   RouterAgent    -> classify the query (rag | summary | vision | general)
#   RetrieverAgent -> hybrid retrieval (+ query expansion)
#   WriterAgent    -> grounded answer with [n] citations
#   CriticAgent    -> hallucination/grounding check, feedback loop
#   OrchestratorAgent -> coordinates the above + memory + semantic cache

import json
import re

from config import settings
import db
from llm import chat_text
from prompts import (ROUTER_PROMPT, WRITER_PROMPT, CRITIC_PROMPT, GENERAL_PROMPT,
                     GREETING_PROMPT)
import retrieval
import memory

# ---- citation robustness helpers -------------------------------------------
# Deterministic post-processing that keeps citations honest:
#   * out-of-range [n] (no matching context block) are dropped entirely;
#   * "padding" citations (a block that has no lexical overlap with the claim
#     it is attached to) are pruned from multi-citation groups, always keeping
#     the best-scoring one so a claim retains at least one citation;
#   * duplicate numbers are collapsed and ranges are expanded.
# This is a hard backstop on top of the (soft) Writer/Critic prompt rules.

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")
# Common English stopwords that would otherwise make any two chunks look
# "overlapping" and defeat the padding-citation check.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "there", "their", "they",
    "them", "we", "our", "you", "your", "he", "she", "his", "her", "i", "me",
    "my", "do", "does", "did", "have", "has", "had", "not", "no", "so", "if",
    "then", "than", "about", "into", "over", "under", "also", "can", "could",
    "should", "will", "would", "may", "might", "must", "etc", "use", "used",
    "using", "one", "two", "three", "via", "such", "which", "what", "when",
    "where", "how", "who", "whom", "why", "any", "all", "each", "every", "some",
    "more", "most", "only", "other", "very", "just", "because", "after",
    "before", "while", "during", "per",
}


def _significant_words(text):
    return [w.lower() for w in _WORD_RE.findall(text or "")
            if w.lower() not in _STOPWORDS]


def _containing_sentence(text, pos):
    """Return the sentence (roughly) that contains position `pos` in `text`."""
    before = text[:pos]
    start = max(before.rfind(". "), before.rfind("! "), before.rfind("? "),
                before.rfind("\n")) + 1
    after = text[pos:]
    m = re.search(r"[.!?\n]", after)
    end = pos + (m.start() if m else len(after))
    return text[start:end]


def _overlap_fraction(claim_words, chunk_text):
    """Fraction of the claim's significant words that appear in the chunk."""
    if not claim_words:
        return 1.0
    chunk_words = set(_significant_words(chunk_text))
    hits = [w for w in claim_words if w in chunk_words]
    return len(hits) / len(claim_words)


def sanitize_citations(answer, blocks):
    """Clean the citation markers in a generated answer (see module docstring)."""
    if not answer:
        return answer
    if not blocks:
        return _CITE_RE.sub("", answer).strip()
    by_num = {r.get("citation"): (r.get("content") or "") for r in blocks}
    parts, last, dropped = [], 0, False
    for m in _CITE_RE.finditer(answer):
        parts.append(answer[last:m.start()])
        nums = []
        for p in re.split(r"[,]", m.group(1)):
            p = p.strip()
            rm = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", p)
            if rm:
                nums.extend(range(int(rm.group(1)), int(rm.group(2)) + 1))
            elif p.isdigit():
                nums.append(int(p))
        nums = list(dict.fromkeys(nums))  # dedupe, keep order
        valid = [n for n in nums if n in by_num]
        if not valid:
            last = m.end()  # drop out-of-range citation entirely
            dropped = True
            continue
        if len(valid) == 1:
            parts.append(f"[{valid[0]}]")
        else:
            claim = _significant_words(_containing_sentence(answer, m.start()))
            if not claim:
                parts.append("[" + ",".join(map(str, valid)) + "]")
            else:
                scored = [(_overlap_fraction(claim, by_num[n]), n) for n in valid]
                best_n = max(scored, key=lambda x: (x[0], -x[1]))[1]  # top score; ties -> smallest n
                keep = sorted({n for score, n in scored
                               if score >= settings.CITATION_OVERLAP_THRESHOLD} or {best_n})
                parts.append("[" + ",".join(map(str, keep)) + "]")
        last = m.end()
    parts.append(answer[last:])
    out = "".join(parts).strip()
    if dropped:
        out = re.sub(r"[ \t]{2,}", " ", out)  # tidy space left by a dropped marker
    return out


def _best_snippet(content, answer, max_len=None):
    """Return the region of `content` that overlaps most with the answer text,
    so a source card shows the part that actually backs the cited claim instead
    of just the start of the chunk. Among equally-dense windows it picks the one
    centered nearest to the middle of the matched words."""
    max_len = max_len or settings.SNIPPET_MAX_CHARS
    if not content:
        return ""
    flat = " ".join(content.split())
    a_set = set(_significant_words(answer))
    if not a_set:
        return flat[:max_len]
    words = flat.split()
    norm = [w.lower() for w in words]
    matches = [j for j, w in enumerate(norm) if w in a_set]
    if not matches:
        return flat[:max_len]
    win = settings.SNIPPET_WINDOW
    scores = []
    for i in range(max(0, len(norm) - win) + 1):
        scores.append(sum(1 for w in norm[i:i + win] if w in a_set))
    best_score = max(scores)
    # Median is robust to stray matches (e.g. a shared word far away) and
    # centers the snippet on the densest cluster of matched terms.
    center = sorted(matches)[len(matches) // 2]
    best_i = min((i for i, s in enumerate(scores) if s == best_score),
                 key=lambda i: abs((i + win / 2) - center))
    offset = sum(len(words[j]) + 1 for j in range(best_i))
    snippet = flat[offset:offset + max_len]
    return snippet or flat[:max_len]


_GREETING_RE = re.compile(
    r"^(?:hi+|hello+|hey+|yo+|hola|howdy|hiya|sup|wassup|whatsup|whats up|what up|"
    r"hi there|hello there|hey there|how are you(?: doing| today)?|how r u|"
    r"hows it going|good morning|good afternoon|good evening|good day|"
    r"greetings|namaste)[\s!?.]*$",
    re.IGNORECASE,
)


def _is_greeting(query: str) -> bool:
    """Fast, cheap detector: is this message a pure greeting (no RAG needed)?"""
    if not query or len(query) > 40:
        return False
    # Normalize: lowercase, drop punctuation/apostrophes, collapse whitespace.
    q = re.sub(r"[^a-z\s]", "", query.lower())
    q = re.sub(r"\s+", " ", q).strip()
    return bool(q and _GREETING_RE.match(q))


class RouterAgent:
    def classify(self, query: str) -> str:
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


class RetrieverAgent:
    def run(self, query: str, top_k: int | None = None, collection: str | None = None):
        return retrieval.retrieve(query, top_k=top_k, collection=collection)


class WriterAgent:
    def _build_messages(self, query, context_blocks, memory_text, feedback=None):
        if context_blocks:
            context = "\n\n".join(
                f"[{r['citation']}] ({r['title']})\n{r['content']}"
                for r in context_blocks
            )
        else:
            context = "(no matching documents found in the collection)"
        messages = [{"role": "system", "content": WRITER_PROMPT.format(context=context)}]
        if memory_text:
            messages.append({"role": "system",
                             "content": f"Relevant conversation history:\n{memory_text}"})
        user_content = query
        if feedback:
            user_content += (
                "\n\nYour previous answer had these issues; please fix them: "
                + "; ".join(feedback)
            )
        messages.append({"role": "user", "content": user_content})
        return messages

    def run(self, query, context_blocks, memory_text="", feedback=None):
        messages = self._build_messages(query, context_blocks, memory_text, feedback)
        return chat_text(messages)

    def stream(self, query, context_blocks, memory_text="", feedback=None):
        """Generator yielding (full_answer, delta) pairs."""
        from llm import chat_stream
        messages = self._build_messages(query, context_blocks, memory_text, feedback)
        parts = []
        for delta in chat_stream(messages):
            parts.append(delta)
            yield "".join(parts), delta
        if not parts:
            yield "", ""


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
        except Exception:  # noqa: BLE001
            return True, []


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

    def _persist_user(self, conversation_id, query, q_emb):
        """Save the user's message immediately so it is never lost if the turn is
        interrupted."""
        if conversation_id:
            memory.add_message(conversation_id, "user", query, embedding=q_emb)

    def _persist(self, conversation_id, query, answer, q_emb, cached, sources=None,
                 collection_id=None):
        if conversation_id:
            memory.add_message(conversation_id, "assistant", answer,
                               embedding=retrieval.embed_query(answer),
                               sources=sources)
        if not cached and q_emb is not None:
            retrieval.semantic_cache_store(query, q_emb, answer, settings.LLM_MODEL,
                                           sources, collection_id)

    def _sources(self, blocks, answer=""):
        out = []
        for r in blocks:
            meta = r.get("metadata") or {}
            content = r.get("content") or ""
            out.append({
                "citation": r.get("citation"),
                "title": r.get("title"),
                "doc_id": r.get("doc_id"),
                "score": round(float(r.get("rrf_score") or 0.0), 4),
                "page": meta.get("page"),
                "image_id": meta.get("image_id"),
                "snippet": _best_snippet(content, answer),
            })
        return out

    def _cited_sources(self, answer, blocks):
        """Sanitize the answer's citations, then keep only sources actually
        cited as [n]. Returns (sources, sanitized_answer) so the caller can
        also persist the cleaned answer (e.g. into the semantic cache).
        Also attaches the page's stored image (if any) so the UI can display
        it alongside the text, even when the cited chunk is a text chunk."""
        answer = sanitize_citations(answer or "", blocks)
        cited = set()
        for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
            for part in m.group(1).split(","):
                part = part.strip()
                if part.isdigit():
                    cited.add(int(part))
        sources = [s for s in self._sources(blocks, answer) if s["citation"] in cited]
        return self._attach_page_images(sources), answer

    def _attach_page_images(self, sources):
        """If a cited source points to a page that has a stored image, attach it."""
        keys = {(s.get("doc_id"), s.get("page")) for s in sources if s.get("page")}
        if not keys:
            return sources
        mapping = {}
        with db.get_conn().cursor() as cur:
            for doc_id, page in keys:
                cur.execute(
                    "SELECT id FROM images WHERE document_id=%s AND page=%s ORDER BY id LIMIT 1",
                    (doc_id, page),
                )
                row = cur.fetchone()
                if row:
                    mapping[(doc_id, page)] = row["id"]
        for s in sources:
            key = (s.get("doc_id"), s.get("page"))
            if key in mapping and not s.get("image_id"):
                s["image_id"] = mapping[key]
        return sources

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
            collection: str | None = None) -> dict:
        # Greetings short-circuit: no embedding, no router, no RAG retrieval.
        if _is_greeting(query):
            answer = self._greet(query, conversation_id)
            return {"answer": answer, "sources": [], "type": "greeting"}

        q_emb = retrieval.embed_query(query) if conversation_id else None
        memory_text = self._memory_for(conversation_id, q_emb)
        kind = self.router.classify(query)
        self._persist_user(conversation_id, query, q_emb)

        result = self.retriever.run(query, top_k=top_k, collection=collection)

        # General-knowledge fallback: no STRONG document match (top vector cosine
        # below the threshold) AND the router says the question is general -> answer
        # from general knowledge (no citations). Doc questions that fail retrieval
        # are still handled by the grounded writer (it refuses rather than fabricates).
        best = result.get("best_score") or 0.0
        if (not result.get("cached") and kind == "general"
                and best < settings.GENERAL_STRONG_THRESHOLD):
            answer = chat_text(
                [{"role": "system", "content": GENERAL_PROMPT},
                 {"role": "user", "content": query}])
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
        else:
            blocks = result.get("results", [])
            context_text = "\n".join(
                f"[{r['citation']}] {r['content']}" for r in blocks)
            answer = self.writer.run(query, blocks, memory_text)

            # Critic feedback loop.
            for _ in range(settings.MAX_CRITIC_ROUNDS):
                ok, issues = self.critic.review(query, context_text, answer)
                if ok or not issues:
                    break
                answer = self.writer.run(query, blocks, memory_text, feedback=issues)
            sources, answer = self._cited_sources(answer, blocks)

        self._persist(conversation_id, query, answer, q_emb, result.get("cached"),
                      sources, result.get("collection_id"))
        return {"answer": answer, "sources": sources, "type": kind}

    # -- streaming --------------------------------------------------------------

    def run_stream(self, query: str, conversation_id: str | None = None,
                   top_k: int | None = None, collection: str | None = None):
        """Generator yielding events:
        {"type": "content", "delta": str} and finally {"type": "sources", ...}"""
        # Greetings short-circuit: no embedding, no router, no RAG retrieval.
        if _is_greeting(query):
            answer = self._greet(query, conversation_id)
            yield {"type": "content", "delta": answer}
            yield {"type": "sources", "sources": []}
            return

        q_emb = retrieval.embed_query(query) if conversation_id else None
        memory_text = self._memory_for(conversation_id, q_emb)
        kind = self.router.classify(query)
        self._persist_user(conversation_id, query, q_emb)

        result = self.retriever.run(query, top_k=top_k, collection=collection)

        best = result.get("best_score") or 0.0
        if (not result.get("cached") and kind == "general"
                and best < settings.GENERAL_STRONG_THRESHOLD):
            answer = chat_text(
                [{"role": "system", "content": GENERAL_PROMPT},
                 {"role": "user", "content": query}])
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
            yield {"type": "content", "delta": answer}
        else:
            blocks = result.get("results", [])
            full = ""
            for full_so_far, delta in self.writer.stream(query, blocks, memory_text):
                full = full_so_far
                if delta:
                    yield {"type": "content", "delta": delta}
            # critic loop (no streaming for the rewrite pass)
            context_text = "\n".join(f"[{r['citation']}] {r['content']}" for r in blocks)
            answer = full
            for _ in range(settings.MAX_CRITIC_ROUNDS):
                ok, issues = self.critic.review(query, context_text, answer)
                if ok or not issues:
                    break
                answer = self.writer.run(query, blocks, memory_text, feedback=issues)
                yield {"type": "content", "delta": f"\n[revised] {answer}"}
            sources, answer = self._cited_sources(answer, blocks)

        self._persist(conversation_id, query, answer, q_emb, result.get("cached"),
                      sources, result.get("collection_id"))
        yield {"type": "sources", "sources": sources}
