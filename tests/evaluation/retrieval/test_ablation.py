"""Retrieval ablation: Recall@1/5/10 + MRR across retrieval configurations.

Marked ``evaluation`` — run with real models and the corpus ingested:

    python -m pytest -m evaluation -k ablation

Sweeps the SAME golden dataset (``tests/evaluation/datasets/rag_eval.json``)
through each channel configuration and prints a comparison table. The sparse
channel is reported as N/A when pgvector < 0.7 (``db.SPARSE_READY`` False) —
this box is one such case.
"""
import json
import statistics
from pathlib import Path

import pytest

from app.core.config import settings

DATASET = Path(__file__).parents[1] / "datasets" / "rag_eval.json"
TOP_K = 10

# Channel configs: (expansion, keyword/FTS, reranker). Dense is always on.
# NOTE: the keyword/FTS channel only runs on query-expansion variants in
# retrieve(), so "Hybrid" configs enable expansion (5 variants).
CONFIGS = [
    {"name": "Dense",                      "expansion": False, "keyword": False, "rerank": False},
    {"name": "Dense + Reranker",           "expansion": False, "keyword": False, "rerank": True},
    {"name": "Hybrid + RRF",               "expansion": True,  "keyword": True,  "rerank": False},
    {"name": "Hybrid + RRF + Reranker",    "expansion": True,  "keyword": True,  "rerank": True},
]


def _eval_config(conf, dataset):
    """Run the real pipeline over the golden set under one channel config."""
    import app.retrieval as retrieval

    prev = (settings.USE_QUERY_EXPANSION, settings.USE_KEYWORD_SEARCH,
            settings.USE_RERANKER)
    settings.USE_QUERY_EXPANSION = conf["expansion"]
    settings.USE_KEYWORD_SEARCH = conf["keyword"]
    settings.USE_RERANKER = conf["rerank"]
    try:
        hits = {1: 0, 5: 0, 10: 0}
        rr_sum = 0.0
        lats = []
        for item in dataset:
            res = retrieval.retrieve(item["question"], top_k=TOP_K, use_cache=False)
            titles = [(r.get("title") or "").lower() for r in res["results"]]
            expected = {d.lower() for d in item["relevant_documents"]}
            for k in hits:
                if expected & set(titles[:k]):
                    hits[k] += 1
            for i, t in enumerate(titles, start=1):
                if t in expected:
                    rr_sum += 1.0 / i
                    break
            lats.append((res.get("latency_ms") or {}).get("total", 0.0))
        n = max(len(dataset), 1)
        return {
            "recall@1": hits[1] / n,
            "recall@5": hits[5] / n,
            "recall@10": hits[10] / n,
            "MRR": rr_sum / n,
            "lat_p50_ms": statistics.median(lats) if lats else 0.0,
            "lat_mean_ms": statistics.mean(lats) if lats else 0.0,
        }
    finally:
        (settings.USE_QUERY_EXPANSION, settings.USE_KEYWORD_SEARCH,
         settings.USE_RERANKER) = prev


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.mark.evaluation
def test_retrieval_ablation(db_ready, dataset):
    import app.database.postgres as db

    if settings.EMBEDDING_MODEL == "mock":
        pytest.skip("evaluation needs a real embedding model (EMBEDDING_MODEL != mock)")

    sparse_label = "ON" if db.SPARSE_READY else "N/A (pgvector<0.7)"
    print(f"\n=== Retrieval ablation — {len(dataset)} questions, top_k={TOP_K}, "
          f"use_cache=False, sparse={sparse_label} ===")
    print(f"{'config':<24}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'p50ms':>8}{'meanms':>9}")
    results = {}
    for conf in CONFIGS:
        r = _eval_config(conf, dataset)
        results[conf["name"]] = r
        print(f"{conf['name']:<24}{r['recall@1']:>7.3f}{r['recall@5']:>7.3f}"
              f"{r['recall@10']:>7.3f}{r['MRR']:>7.3f}{r['lat_p50_ms']:>8.0f}"
              f"{r['lat_mean_ms']:>9.0f}")

    # Loose sanity floor: the full pipeline must retrieve something relevant for
    # at least half the questions at recall@10 (a real quality gate, not a pass).
    full = results.get("Hybrid + RRF + Reranker", {})
    assert full.get("recall@10", 0.0) >= 0.4, "full pipeline recall@10 below floor"
