"""RAG retrieval evaluation: recall@k and MRR against a fixed dataset.

Marked `evaluation` — excluded from the default hermetic suite. Run it with
REAL embedding/LLM models and the corpus ingested:

    python -m pytest -m evaluation

Requires EMBEDDING_MODEL != mock and the documents referenced by
tests/evaluation/datasets/rag_eval.json to be ingested into the default
collection.
"""
import json
from pathlib import Path

import pytest

from app.core.config import settings

DATASET = Path(__file__).parents[1] / "datasets" / "rag_eval.json"
TOP_K = 5
# Modest quality floors — raise them as the pipeline improves.
MIN_RECALL = 0.5
MIN_MRR = 0.3


@pytest.fixture(scope="module")
def dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def _require_real_embeddings():
    if settings.EMBEDDING_MODEL == "mock":
        pytest.skip("evaluation needs a real embedding model (EMBEDDING_MODEL != mock)")


@pytest.mark.evaluation
def test_recall_at_k_and_mrr(db_ready, dataset):
    _require_real_embeddings()
    import app.retrieval as retrieval

    recalls, rr = [], []
    for item in dataset:
        res = retrieval.retrieve(item["question"], top_k=TOP_K, use_cache=False)
        titles = [(r.get("title") or "").lower() for r in res["results"]]
        expected = {d.lower() for d in item["relevant_documents"]}
        recalls.append(bool(expected & set(titles)))
        for i, t in enumerate(titles, start=1):
            if t in expected:
                rr.append(1.0 / i)
                break
        else:
            rr.append(0.0)

    recall = sum(recalls) / len(recalls)
    mrr = sum(rr) / len(rr)
    print(f"\n[recall] recall@{TOP_K}={recall:.3f}  MRR={mrr:.3f}  "
          f"({len(dataset)} queries)")
    assert recall >= MIN_RECALL, f"recall@{TOP_K} below floor {MIN_RECALL}: {recall}"
    assert mrr >= MIN_MRR, f"MRR below floor {MIN_MRR}: {mrr}"
