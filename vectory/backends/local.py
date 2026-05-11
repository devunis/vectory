"""Local numpy-based vector store (built-in, no external dependencies)."""

from __future__ import annotations

from typing import Any

from vectory.backends.base import SearchResult, VectorStore
from vectory.engine.collection import Collection
from vectory.engine.distance import DistanceMetric
from vectory.storage.backend import FileStorageBackend


class LocalStore(VectorStore):
    """Built-in vector store using numpy brute-force search with JSON file persistence."""

    store_type = "local"

    def __init__(self, data_dir: str = ".vectory_data"):
        self._storage = FileStorageBackend(data_dir)
        self._collections: dict[str, Collection] = {}

    def _get(self, name: str) -> Collection:
        if name not in self._collections:
            data = self._storage.load(name)
            self._collections[name] = Collection.from_dict(data)
        return self._collections[name]

    def _save(self, name: str) -> None:
        self._storage.save(name, self._collections[name].to_dict())

    def create_collection(self, name: str, dimension: int, metric: str) -> dict[str, Any]:
        if self._storage.exists(name) or name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        col = Collection(name=name, dimension=dimension, metric=DistanceMetric(metric))
        self._collections[name] = col
        self._save(name)
        return {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "count": 0,
            "store_type": self.store_type,
        }

    def drop_collection(self, name: str) -> None:
        self._collections.pop(name, None)
        self._storage.delete(name)

    def list_collections(self) -> list[dict[str, Any]]:
        names = set(self._collections.keys()) | set(self._storage.list_collections())
        result = []
        for name in sorted(names):
            col = self._get(name)
            result.append(
                {
                    "name": col.name,
                    "dimension": col.dimension,
                    "metric": col.metric.value,
                    "count": col.count,
                    "store_type": self.store_type,
                }
            )
        return result

    def collection_info(self, name: str) -> dict[str, Any]:
        col = self._get(name)
        return {
            "name": col.name,
            "dimension": col.dimension,
            "metric": col.metric.value,
            "count": col.count,
            "store_type": self.store_type,
        }

    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        col = self._get(name)
        result = col.insert(vectors, ids, metadata)
        self._save(name)
        return result

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        col = self._get(name)
        results = col.search(query, top_k, filter_metadata)
        return [SearchResult(id=r.id, score=r.score, metadata=r.metadata) for r in results]

    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        return self._get(name).get(ids)

    def delete(self, name: str, ids: list[str]) -> int:
        col = self._get(name)
        deleted = col.delete(ids)
        self._save(name)
        return deleted

    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        col = self._get(name)
        col.update_metadata(id, metadata)
        self._save(name)
