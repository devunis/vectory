"""Retrieval quality evaluators for corrective RAG."""

from __future__ import annotations

from dataclasses import dataclass

from vectory.rag.embeddings import tokenize
from vectory.rag.models import RetrievalResult


@dataclass
class RetrievalEvaluation:
    confidence: float
    coverage: float
    needs_correction: bool


class HeuristicRetrievalEvaluator:
    """Estimate whether retrieved chunks sufficiently cover the query."""

    def __init__(self, *, min_confidence: float = 0.35) -> None:
        self.min_confidence = min_confidence

    def evaluate(self, query: str, results: list[RetrievalResult]) -> RetrievalEvaluation:
        query_terms = set(tokenize(query))
        if not query_terms or not results:
            return RetrievalEvaluation(confidence=0.0, coverage=0.0, needs_correction=True)

        seen_terms: set[str] = set()
        for result in results[:5]:
            seen_terms.update(tokenize(result.text))

        coverage = len(query_terms & seen_terms) / len(query_terms)
        top_score = max((result.score for result in results[:5]), default=0.0)
        confidence = min(1.0, (coverage * 0.7) + (top_score * 0.3))
        return RetrievalEvaluation(
            confidence=confidence,
            coverage=coverage,
            needs_correction=confidence < self.min_confidence,
        )
