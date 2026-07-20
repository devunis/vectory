"""Lightweight RAPTOR-style hierarchical summaries."""

from __future__ import annotations

from collections import Counter

from vectory.rag.embeddings import tokenize
from vectory.rag.models import Chunk


def build_raptor_summaries(
    chunks: list[Chunk],
    *,
    group_size: int = 4,
    max_terms: int = 48,
) -> list[Chunk]:
    """Build extractive summary chunks over groups of leaf chunks."""
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1")

    summaries: list[Chunk] = []
    for index in range(0, len(chunks), group_size):
        group = chunks[index : index + group_size]
        if len(group) < 2:
            continue
        text = _extractive_summary(group, max_terms=max_terms)
        document_ids = sorted({chunk.document_id for chunk in group})
        summaries.append(
            Chunk(
                id=f"raptor:{document_ids[0]}:{index // group_size}",
                document_id=document_ids[0],
                text=text,
                metadata={
                    "rag": True,
                    "rag_level": 1,
                    "summary_type": "raptor",
                    "child_chunk_ids": [chunk.id for chunk in group],
                    "document_ids": document_ids,
                },
            )
        )
    return summaries


def _extractive_summary(chunks: list[Chunk], *, max_terms: int) -> str:
    text = " ".join(chunk.text for chunk in chunks)
    terms = tokenize(text)
    if not terms:
        return text[:1000]
    counts = Counter(terms)
    important = [term for term, _ in counts.most_common(max_terms)]
    first_sentence = text.split(".")[0].strip()
    summary_terms = " ".join(important)
    if first_sentence:
        return f"{first_sentence}. Key terms: {summary_terms}"
    return f"Key terms: {summary_terms}"
