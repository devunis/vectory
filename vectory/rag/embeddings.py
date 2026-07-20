"""Small local embedding helpers for dependency-free RAG demos."""

from __future__ import annotations

import hashlib
import math
import re


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    This is not a replacement for production embedding models. It gives Vectory a
    local, dependency-free baseline so RAG indexing and retrieval can work in
    tests and simple demos.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w가-힣]+", text.lower())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return sum(a * b for a, b in zip(left, right))
