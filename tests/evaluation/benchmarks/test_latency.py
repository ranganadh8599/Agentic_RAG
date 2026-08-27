"""Pipeline latency benchmark: P50/P95 per query + component breakdown.

Marked ``evaluation`` — run with real models:

    python -m pytest -m evaluation tests/evaluation/benchmarks/

Measures stage-1 (search+fusion) / rerank / total latency across a fixed
query set and reports percentiles. The reranker must be enabled for the
rerank component to be meaningful (USE_RERANKER=1).
"""
import statistics

import pytest

from app.core.config import settings

QUERIES = [
    "What is the refund policy?",
    "How do I reset my password?",
    "What are the key features of the product?",
    "Explain the architecture of the system",
    "What is the pricing structure?",
]


def _require_real_models():
    if settings.EMBEDDING_MODEL == "mock":
        pytest.skip("latency benchmark needs real embedding models")


@pytest.mark.evaluation
def test_pipeline_latency_percentiles(db_ready):
    _require_real_models()
    import app.retrieval as retrieval

    totals, stage1, rerank = [], [], []
    for q in QUERIES:
        res = retrieval.retrieve(q, top_k=5, use_cache=False)
        lat = res.get("latency_ms") or {}
        totals.append(float(lat.get("total", 0)))
        stage1.append(float(lat.get("stage1", 0)))
        rerank.append(float(lat.get("rerank", 0)))

    def p95(vals):
        vals = sorted(vals)
        return vals[int(len(vals) * 0.95) - 1] if len(vals) > 1 else (vals[0] if vals else 0)

    print(f"\n[latency] n={len(totals)} "
          f"P50={statistics.median(totals):.0f}ms P95={p95(totals):.0f}ms "
          f"stage1 P95={p95(stage1):.0f}ms rerank P95={p95(rerank):.0f}ms")
    # Loose guard: catches catastrophic regressions, tolerates a slow dev box.
    assert statistics.median(totals) < 60000
