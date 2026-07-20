"""Retrieval fusion and diversity helpers."""

from __future__ import annotations

from vectory.rag.embeddings import HashingEmbedder, cosine_similarity
from vectory.rag.models import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked document IDs with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def apply_mmr(
    query: str,
    results: list[RetrievalResult],
    *,
    top_k: int,
    lambda_mult: float = 0.5,
    embedder: HashingEmbedder | None = None,
) -> list[RetrievalResult]:
    """Apply Maximal Marginal Relevance over candidate result text."""
    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1")
    if len(results) <= top_k:
        return results[:top_k]

    embedder = embedder or HashingEmbedder()
    query_vec = embedder.embed(query)
    result_vectors = {result.id: embedder.embed(result.text) for result in results}
    selected: list[RetrievalResult] = []
    remaining = results[:]

    while remaining and len(selected) < top_k:
        best_result = None
        best_score = float("-inf")
        for candidate in remaining:
            relevance = cosine_similarity(query_vec, result_vectors[candidate.id])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    cosine_similarity(result_vectors[candidate.id], result_vectors[item.id])
                    for item in selected
                )
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity_penalty
            if score > best_score:
                best_score = score
                best_result = candidate
        if best_result is None:
            break
        selected.append(best_result)
        remaining = [result for result in remaining if result.id != best_result.id]

    return selected
