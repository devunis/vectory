"""Adaptive RAG routing."""

from __future__ import annotations

from dataclasses import dataclass

from vectory.rag.embeddings import tokenize


@dataclass
class RetrievalRoute:
    strategy: str
    candidate_k: int
    mmr_lambda: float | None = None
    reranker: str | None = None


class HeuristicQueryRouter:
    """Route queries to retrieval strategies by simple complexity signals."""

    def route(self, query: str) -> RetrievalRoute:
        terms = tokenize(query)
        lowered = query.lower()
        exact_signals = any(char.isdigit() for char in query) or "-" in query or "_" in query
        multi_hop_signals = {
            "compare",
            "relationship",
            "connect",
            "why",
            "how",
            "between",
            "versus",
            "vs",
            "비교",
            "관계",
            "왜",
            "어떻게",
        }

        if exact_signals and len(terms) <= 6:
            return RetrievalRoute(strategy="bm25", candidate_k=20, reranker="lexical")
        if len(terms) >= 10 or any(signal in lowered for signal in multi_hop_signals):
            return RetrievalRoute(
                strategy="hybrid",
                candidate_k=60,
                mmr_lambda=0.65,
                reranker="lexical",
            )
        return RetrievalRoute(strategy="hybrid", candidate_k=30)
