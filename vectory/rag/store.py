"""Persistent RAG corpus storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vectory.rag.models import Chunk, Document


class RagCorpus:
    """JSON-backed corpus metadata for one RAG collection."""

    def __init__(self, data_dir: str | Path, collection_name: str) -> None:
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.path = self.data_dir / "_rag" / f"{collection_name}.json"
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.load()

    def add_document(self, document: Document, chunks: list[Chunk]) -> None:
        self.documents[document.id] = document
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        self.save()

    def list_chunks(self) -> list[Chunk]:
        return sorted(self.chunks.values(), key=lambda chunk: chunk.id)

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self.chunks:
            raise KeyError(f"Chunk '{chunk_id}' not found")
        return self.chunks[chunk_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "documents": [document.to_dict() for document in self.documents.values()],
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()],
        }

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.documents = {
            item["id"]: Document.from_dict(item) for item in data.get("documents", [])
        }
        self.chunks = {item["id"]: Chunk.from_dict(item) for item in data.get("chunks", [])}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
