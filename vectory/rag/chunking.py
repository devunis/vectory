"""Document chunking helpers."""

from __future__ import annotations

import re

from vectory.rag.models import Chunk, Document


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    contextual_prefix: str | None = None,
) -> list[Chunk]:
    """Split a document into overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = re.findall(r"\S+", document.text)
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = chunk_size - chunk_overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        text = " ".join(words[start:end])
        if contextual_prefix:
            text = f"{contextual_prefix.strip()}\n\n{text}"
        chunks.append(
            Chunk(
                id=f"{document.id}:{index}",
                document_id=document.id,
                text=text,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_start": start,
                    "chunk_end": end,
                },
            )
        )
        if end == len(words):
            break
        start += step
        index += 1
    return chunks
