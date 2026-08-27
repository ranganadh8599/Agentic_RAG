"""RAGAS generation evaluation.

Scores the full pipeline (retrieval + writer) with RAGAS classic metrics:
faithfulness, answer_relevancy, context_precision, context_recall.

Marked ``evaluation`` — excluded from the default suite. Run with REAL
models and the corpus ingested:

    python -m pytest -m evaluation tests/evaluation/generation/

The judge LLM is an instructor-patched litellm client (works for classic and
structured metrics); embeddings use the same model as retrieval so the
retrieved contexts match the corpus.
"""
import json
from pathlib import Path

import pytest

from app.core.config import settings

DATASET = Path(__file__).parents[1] / "datasets" / "rag_eval.json"
MIN_FAITHFULNESS = 0.5


def _require_real_models():
    if settings.EMBEDDING_MODEL == "mock" or settings.LLM_MODEL == "mock":
        pytest.skip("RAGAS evaluation needs real embedding + judge LLM models")


@pytest.fixture(scope="module")
def dataset():
    _require_real_models()  # skip before touching the file under mock models
    return json.loads(DATASET.read_text(encoding="utf-8"))


def _ragas_judge_and_embeddings():
    """Build the judge LLM + embedding wrappers RAGAS needs."""
    import litellm  # noqa: E402
    from langchain_litellm import LiteLLMEmbeddings  # noqa: E402
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
    from ragas.llms import llm_factory  # noqa: E402

    judge = llm_factory(settings.LLM_MODEL, provider="litellm", client=litellm.completion)
    embeddings = LangchainEmbeddingsWrapper(
        LiteLLMEmbeddings(model=settings.EMBEDDING_MODEL))
    return judge, embeddings


@pytest.mark.evaluation
def test_ragas_classic_metrics(db_ready, dataset):
    _require_real_models()
    import app.retrieval as retrieval  # noqa: E402
    from app.agents.writer import WriterAgent  # noqa: E402
    from ragas import evaluate  # noqa: E402
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
    from ragas.metrics import (answer_relevancy, context_precision,  # noqa: E402
                               context_recall, faithfulness)

    judge, embeddings = _ragas_judge_and_embeddings()
    for metric in (faithfulness, answer_relevancy, context_precision, context_recall):
        metric.llm = judge
        metric.embeddings = embeddings

    samples = []
    for item in dataset:
        res = retrieval.retrieve(item["question"], top_k=5, use_cache=False)
        contexts = [r.get("content", "") for r in res["results"]]
        answer = WriterAgent().run(item["question"], res["results"])
        samples.append(SingleTurnSample(
            user_input=item["question"],
            retrieved_contexts=contexts,
            response=answer,
            reference=item.get("ground_truth", ""),
        ))

    result = evaluate(EvaluationDataset(samples=samples),
                      metrics=[faithfulness, answer_relevancy,
                               context_precision, context_recall])
    mean = result.to_pandas().mean(numeric_only=True)
    print(f"\n[RAGAS] {mean.to_dict()}")
    assert float(mean.get("faithfulness", 0)) >= MIN_FAITHFULNESS
