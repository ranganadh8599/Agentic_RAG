# Agentic RAG - RetrieverAgent: wraps the hybrid retrieval pipeline.

import app.retrieval as retrieval


class RetrieverAgent:
    def run(self, query: str, top_k: int | None = None, collection: str | None = None,
            filters=None, user_id: str | None = None):
        return retrieval.retrieve(query, top_k=top_k, collection=collection,
                                  filters=filters, user_id=user_id)
