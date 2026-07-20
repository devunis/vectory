"""RAG reranking helpers."""

from __future__ import annotations

from vectory.rag.embeddings import tokenize
from vectory.rag.models import RetrievalResult


class LexicalReranker:
    """Simple query-token overlap reranker."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return results[:top_k]

        reranked = []
        for result in results:
            result_terms = set(tokenize(result.text))
            overlap = len(query_terms & result_terms) / len(query_terms)
            result.scores["reranker"] = overlap
            result.score = result.score + overlap
            reranked.append(result)

        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
