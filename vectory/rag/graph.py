"""Lightweight graph-style retrieval expansion."""

from __future__ import annotations

import re
from collections import defaultdict

from vectory.rag.models import Chunk, RetrievalResult


class EntityGraph:
    """Chunk graph connected by shared entity-like tokens."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = {chunk.id: chunk for chunk in chunks}
        self.entities_by_chunk = {chunk.id: extract_entities(chunk.text) for chunk in chunks}
        self.chunk_ids_by_entity: dict[str, set[str]] = defaultdict(set)
        for chunk_id, entities in self.entities_by_chunk.items():
            for entity in entities:
                self.chunk_ids_by_entity[entity].add(chunk_id)

    def expand(
        self,
        query: str,
        seeds: list[RetrievalResult],
        *,
        limit: int,
    ) -> list[Chunk]:
        query_entities = extract_entities(query)
        seed_entities = set(query_entities)
        for seed in seeds:
            seed_entities.update(self.entities_by_chunk.get(seed.id, set()))

        expanded_ids: list[str] = []
        seen = {seed.id for seed in seeds}
        for entity in sorted(seed_entities):
            for chunk_id in sorted(self.chunk_ids_by_entity.get(entity, set())):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    expanded_ids.append(chunk_id)
                if len(expanded_ids) >= limit:
                    return [self.chunks[item_id] for item_id in expanded_ids]
        return [self.chunks[item_id] for item_id in expanded_ids]


def extract_entities(text: str) -> set[str]:
    latin = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)
    codes = re.findall(r"\b[A-Za-z]+-\d+\b", text)
    korean = re.findall(r"[가-힣]{2,}", text)
    return {item.lower() for item in [*latin, *codes, *korean]}
