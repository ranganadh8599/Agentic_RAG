"""Citation accuracy evaluation.

Measures what fraction of the sanitizer's output matches the ground-truth
citation set on a fixed sample. A citation is correct only if it survives the
backstop AND belongs to the expected set. This is deterministic and runs
without models, but is kept under the `evaluation` marker to keep the default
suite purely about correctness.
"""
import pytest

from app.citation.sanitizer import sanitize_citations
from app.citation.validator import cited_numbers

# (answer, blocks, expected surviving citation numbers)
SAMPLES = [
    (
        "Refunds last 30 days [1].",
        [{"citation": 1, "content": "Refunds are valid for 30 days."}],
        {1},
    ),
    (
        "Refunds last 30 days [1] and $500 needs approval [2].",
        [
            {"citation": 1, "content": "Refunds are valid for 30 days."},
            {"citation": 2, "content": "Purchases over $500 need manager approval."},
        ],
        {1, 2},
    ),
    # padding citation [2] does not support the claim → pruned
    (
        "Refunds last 30 days [1,2].",
        [
            {"citation": 1, "content": "Refunds are valid for 30 days."},
            {"citation": 2, "content": "Purchases over $500 need manager approval."},
        ],
        {1},
    ),
    # out-of-range citation [9] → dropped entirely
    (
        "Refunds last 30 days [9].",
        [{"citation": 1, "content": "Refunds are valid for 30 days."}],
        set(),
    ),
]


@pytest.mark.evaluation
def test_citation_accuracy():
    correct = 0
    for answer, blocks, expected in SAMPLES:
        cleaned = sanitize_citations(answer, blocks)
        cited = cited_numbers(cleaned)
        if cited == expected:
            correct += 1
    accuracy = correct / len(SAMPLES)
    print(f"\n[citation] accuracy={accuracy:.2f} ({correct}/{len(SAMPLES)} samples)")
    assert accuracy == 1.0
