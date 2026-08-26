# Agentic RAG - cross-encoder reranking (stage 2 of two-stage retrieval).
#
# Stage 1 (bi-encoder): query & documents are embedded INDEPENDENTLY, so
# similarity is a cheap dot product -> fast enough to scan the whole corpus.
# It fetches a wide candidate pool (~50-100 chunks) via hybrid search + RRF.
#
# Stage 2 (cross-encoder): the query and each candidate chunk are fed through
# the transformer TOGETHER ([CLS] query [SEP] chunk), so attention can model
# fine-grained query<->document interactions. Far more accurate ordering, but
# O(N) transformer passes -> only affordable on the small candidate pool.

import logging
import math

from config import settings

log = logging.getLogger(__name__)

_model = None  # lazy singleton


def _get_model():
    """Load the cross-encoder once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        log.info("⚡ Loading reranker model: %s", settings.RERANKER_MODEL)
        kwargs = {"max_length": settings.RERANKER_MAX_LENGTH}
        if settings.RERANKER_INSTRUCTION:
            # Instruction-aware rerankers (e.g. Qwen3): inject a task instruction
            # through the model's prompt template for better task fit.
            kwargs["prompts"] = {"rerank": settings.RERANKER_INSTRUCTION}
            kwargs["default_prompt_name"] = "rerank"
        # Prefer CUDA when available (offloads the cross-encoder passes to the
        # GPU); falls back to CPU when torch has no CUDA build or no GPU exists.
        torch = None
        device = None
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            torch = None
        _model = CrossEncoder(settings.RERANKER_MODEL, device=device, **kwargs)
        if device == "cuda" and torch is not None:
            try:
                log.info("⚡ Reranker running on GPU: %s", torch.cuda.get_device_name(0))
            except Exception:  # noqa: BLE001
                pass
    return _model


def rerank(query: str, docs: list[dict], top_n: int, return_all: bool = False) -> list[dict]:
    """Score each doc for relevance to the query, return the top_n by score.

    Contract:
      input  -> a query string + a list of candidate docs (dicts with 'content')
      scores -> every candidate is scored for relevance to the query
      output -> up to top_n docs, sorted by relevance score (descending)

    Each returned doc carries two fields so downstream code can judge
    confidence:
      * rerank_score      - raw cross-encoder logit (higher = more relevant;
                            can be negative, not calibrated)
      * rerank_confidence - sigmoid-normalized logit in [0, 1] (probability-
                            like; a natural confidence measure)

    return_all=True returns the ENTIRE ranked candidate list (not just top_n)
    so callers can log/ inspect what was cut. The scoring cost is identical.

    Falls back to a lightweight lexical scorer when sentence-transformers /
    torch is unavailable (e.g. mock/offline test environments).
    """
    if not docs:
        return []

    pairs = [(query, d.get("content") or "") for d in docs]

    try:
        import time as _time
        t0 = _time.perf_counter()
        model = _get_model()
        scores = model.predict(
            pairs,
            batch_size=settings.RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
        scores = [float(s) for s in scores]
        ms = (_time.perf_counter() - t0) * 1000
        device = "cpu"
        try:
            import torch
            device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            pass
        log.info("⚡ Reranked %d candidates → top %d in %.1fs on %s",
                 len(docs), top_n, ms / 1000, device)
    except ImportError:
        log.warning("sentence-transformers not installed; using lexical fallback reranker")
        scores = [_lexical_score(q, c) for q, c in pairs]
    except Exception as exc:  # noqa: BLE001 - never let reranking break retrieval
        log.warning("Cross-encoder failed (%s); keeping fusion order", exc)
        return docs[:top_n]

    # Attach raw score + confidence, then sort by relevance (descending).
    ranked = []
    for row, raw in zip(docs, scores):
        row = dict(row)
        row["rerank_score"] = raw
        # Lexical fallback scores are already in [0,1]; cross-encoder logits are
        # mapped through a sigmoid so both read as a [0,1] confidence.
        row["rerank_confidence"] = raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
        ranked.append(row)
    ranked.sort(key=lambda r: r["rerank_score"], reverse=True)
    if return_all:
        return ranked
    return ranked[:top_n]


def _sigmoid(x: float) -> float:
    """Numerically-stable sigmoid: map a raw logit to a [0, 1] confidence."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _lexical_score(query: str, text: str) -> float:
    """Fallback: fraction of significant query words present in the chunk."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
            "to", "for", "and", "or", "what", "which", "who", "how", "why", "do",
            "does", "did", "with", "about", "tell", "me", "give", "explain"}
    q_words = {w for w in query.lower().split() if w not in stop and len(w) > 1}
    if not q_words or not text:
        return 0.0
    t_words = set(text.lower().split())
    hits = sum(1 for w in q_words if w in t_words)
    return hits / len(q_words)
