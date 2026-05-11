"""ChromaDB vector store backend."""

from __future__ import annotations

import uuid
from typing import Any

from vectory.backends.base import SearchResult, VectorStore

METRIC_MAP = {"cosine": "cosine", "euclidean": "l2", "dot_product": "ip"}


class ChromaStore(VectorStore):
    """ChromaDB backend (embedded mode, no server required)."""

    store_type = "chroma"

    def __init__(self, persist_dir: str = ".vectory_chroma"):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)

    def _metric(self, metric: str) -> str:
        return METRIC_MAP.get(metric, "cosine")

    def create_collection(self, name: str, dimension: int, metric: str) -> dict[str, Any]:
        existing = [c.name for c in self._client.list_collections()]
        if name in existing:
            raise ValueError(f"Collection '{name}' already exists")
        self._client.create_collection(
            name=name,
            metadata={
                "dimension": dimension,
                "metric": metric,
                "hnsw:space": self._metric(metric),
            },
        )
        return {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "count": 0,
            "store_type": self.store_type,
        }

    def drop_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def list_collections(self) -> list[dict[str, Any]]:
        result = []
        for col in self._client.list_collections():
            c = self._client.get_collection(col.name)
            meta = c.metadata or {}
            result.append(
                {
                    "name": col.name,
                    "dimension": meta.get("dimension", 0),
                    "metric": meta.get("metric", "cosine"),
                    "count": c.count(),
                    "store_type": self.store_type,
                }
            )
        return result

    def collection_info(self, name: str) -> dict[str, Any]:
        c = self._client.get_collection(name)
        meta = c.metadata or {}
        return {
            "name": name,
            "dimension": meta.get("dimension", 0),
            "metric": meta.get("metric", "cosine"),
            "count": c.count(),
            "store_type": self.store_type,
        }

    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        c = self._client.get_collection(name)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if metadata is None:
            metadata = [{} for _ in vectors]
        # Chroma doesn't accept empty metadata dicts well, use a placeholder
        clean_meta = [m if m else {"_empty": True} for m in metadata]
        c.add(ids=ids, embeddings=vectors, metadatas=clean_meta)
        return ids

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        c = self._client.get_collection(name)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query],
            "n_results": min(top_k, c.count() or 1),
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata
        results = c.query(**kwargs)
        out = []
        if results["ids"] and results["ids"][0]:
            for i, vid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                meta = {k: v for k, v in meta.items() if k != "_empty"}
                score = results["distances"][0][i] if results["distances"] else 0.0
                out.append(SearchResult(id=vid, score=score, metadata=meta))
        return out

    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        c = self._client.get_collection(name)
        results = c.get(ids=ids, include=["embeddings", "metadatas"])
        out = []
        for i, vid in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            meta = {k: v for k, v in meta.items() if k != "_empty"}
            vec = results["embeddings"][i] if results["embeddings"] else []
            out.append({"id": vid, "vector": vec, "metadata": meta})
        return out

    def delete(self, name: str, ids: list[str]) -> int:
        c = self._client.get_collection(name)
        before = c.count()
        c.delete(ids=ids)
        return before - c.count()

    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        c = self._client.get_collection(name)
        existing = c.get(ids=[id], include=["metadatas"])
        if not existing["ids"]:
            raise KeyError(f"ID not found: {id}")
        merged = {**(existing["metadatas"][0] or {}), **metadata}
        c.update(ids=[id], metadatas=[merged])
