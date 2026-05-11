"""Collection - a single vector database instance."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vectory.engine.distance import DISTANCE_FUNCTIONS, DistanceMetric


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any]


class Collection:
    """A named collection of vectors with metadata."""

    def __init__(
        self,
        name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self._ids: list[str] = []
        self._vectors: list[NDArray] = []
        self._metadata: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self._ids)

    def insert(
        self,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Insert vectors into the collection. Returns assigned IDs."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if metadata is None:
            metadata = [{} for _ in vectors]

        if len(vectors) != len(ids) or len(vectors) != len(metadata):
            raise ValueError("vectors, ids, and metadata must have the same length")

        for vec, vid, meta in zip(vectors, ids, metadata):
            arr = np.array(vec, dtype=np.float32)
            if arr.shape != (self.dimension,):
                raise ValueError(f"Expected dimension {self.dimension}, got {arr.shape[0]}")
            if vid in self._id_set:
                raise ValueError(f"Duplicate id: {vid}")
            self._ids.append(vid)
            self._vectors.append(arr)
            self._metadata.append(meta)

        return ids

    @property
    def _id_set(self) -> set[str]:
        return set(self._ids)

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve vectors and metadata by IDs."""
        index_map = {vid: i for i, vid in enumerate(self._ids)}
        results = []
        for vid in ids:
            if vid not in index_map:
                raise KeyError(f"ID not found: {vid}")
            i = index_map[vid]
            results.append(
                {
                    "id": vid,
                    "vector": self._vectors[i].tolist(),
                    "metadata": self._metadata[i],
                }
            )
        return results

    def delete(self, ids: list[str]) -> int:
        """Delete vectors by IDs. Returns number of deleted items."""
        to_delete = set(ids)
        indices_to_keep = [i for i, vid in enumerate(self._ids) if vid not in to_delete]
        deleted = len(self._ids) - len(indices_to_keep)
        self._ids = [self._ids[i] for i in indices_to_keep]
        self._vectors = [self._vectors[i] for i in indices_to_keep]
        self._metadata = [self._metadata[i] for i in indices_to_keep]
        return deleted

    def update_metadata(self, id: str, metadata: dict[str, Any]) -> None:
        """Update metadata for a specific vector."""
        index_map = {vid: i for i, vid in enumerate(self._ids)}
        if id not in index_map:
            raise KeyError(f"ID not found: {id}")
        self._metadata[index_map[id]].update(metadata)

    def search(
        self,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar vectors."""
        if self.count == 0:
            return []

        query_vec = np.array(query, dtype=np.float32)
        if query_vec.shape != (self.dimension,):
            raise ValueError(f"Expected dimension {self.dimension}, got {query_vec.shape[0]}")

        # Apply metadata filter
        if filter_metadata:
            indices = [
                i
                for i, meta in enumerate(self._metadata)
                if all(meta.get(k) == v for k, v in filter_metadata.items())
            ]
        else:
            indices = list(range(self.count))

        if not indices:
            return []

        vectors = np.array([self._vectors[i] for i in indices])
        dist_fn = DISTANCE_FUNCTIONS[self.metric]
        distances = dist_fn(query_vec, vectors)

        # Sort by distance (ascending = most similar first)
        k = min(top_k, len(indices))
        if k >= len(indices):
            top_indices = np.argsort(distances)[:k]
        else:
            top_indices = np.argpartition(distances, k)[:k]
            top_indices = top_indices[np.argsort(distances[top_indices])]

        return [
            SearchResult(
                id=self._ids[indices[i]],
                score=float(distances[i]),
                metadata=self._metadata[indices[i]],
            )
            for i in top_indices
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize collection to a dict."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "ids": self._ids,
            "vectors": [v.tolist() for v in self._vectors],
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Collection:
        """Deserialize collection from a dict."""
        col = cls(
            name=data["name"],
            dimension=data["dimension"],
            metric=DistanceMetric(data["metric"]),
        )
        col._ids = data["ids"]
        col._vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
        col._metadata = data["metadata"]
        return col
