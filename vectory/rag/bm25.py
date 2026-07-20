"""BM25 lexical retrieval."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from vectory.rag.embeddings import tokenize
from vectory.rag.models import Chunk


class BM25Index:
    """In-memory BM25 index for chunk text."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._doc_terms: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._avgdl = 0.0
        self._build()

    def search(self, query: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []

        scored: list[tuple[str, float]] = []
        total_docs = len(self.chunks)
        for chunk in self.chunks:
            term_counts = self._doc_terms[chunk.id]
            doc_length = self._doc_lengths[chunk.id]
            score = 0.0
            for term in terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (total_docs - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self._avgdl)
                score += idf * (tf * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((chunk.id, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _build(self) -> None:
        total_length = 0
        for chunk in self.chunks:
            terms = tokenize(chunk.text)
            counts = Counter(terms)
            self._doc_terms[chunk.id] = counts
            self._doc_lengths[chunk.id] = len(terms)
            total_length += len(terms)
            for term in counts:
                self._df[term] += 1
        self._avgdl = total_length / len(self.chunks) if self.chunks else 1.0
