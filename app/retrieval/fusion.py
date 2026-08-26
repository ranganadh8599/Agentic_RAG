# Agentic RAG - reciprocal-rank fusion (RRF) of multiple ranked result lists.


def rrf_fuse(ranked_lists, k: int = 60):
    """Reciprocal-rank fusion of multiple ranked result lists."""
    scores: dict[int, float] = {}
    info: dict[int, dict] = {}
    for ranked in ranked_lists:
        for rank, (_score, row) in enumerate(ranked):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            info[rid] = row
    out = []
    for rid, s in sorted(scores.items(), key=lambda x: -x[1]):
        row = dict(info[rid])
        row["rrf_score"] = s
        out.append(row)
    return out
