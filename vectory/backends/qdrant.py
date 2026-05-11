"""Qdrant vector store backend."""

from __future__ import annotations

import uuid
from typing import Any

from vectory.backends.base import SearchResult, VectorStore

METRIC_MAP = {
    "cosine": "Cosine",
    "euclidean": "Euclid",
    "dot_product": "Dot",
}


class QdrantStore(VectorStore):
    """Qdrant backend (in-memory mode or connect to server)."""

    store_type = "qdrant"

    def __init__(self, url: str | None = None, path: str | None = ".vectory_qdrant"):
        from qdrant_client import QdrantClient

        if url:
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=path)
        self._dimensions: dict[str, int] = {}
        self._metrics: dict[str, str] = {}

    def create_collection(self, name: str, dimension: int, metric: str) -> dict[str, Any]:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(name):
            raise ValueError(f"Collection '{name}' already exists")
        dist = getattr(Distance, METRIC_MAP.get(metric, "Cosine"))
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=dist),
        )
        self._dimensions[name] = dimension
        self._metrics[name] = metric
        return {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "count": 0,
            "store_type": self.store_type,
        }

    def drop_collection(self, name: str) -> None:
        self._client.delete_collection(name)
        self._dimensions.pop(name, None)
        self._metrics.pop(name, None)

    def _get_info(self, name: str) -> tuple[int, str]:
        if name in self._dimensions:
            return self._dimensions[name], self._metrics[name]
        info = self._client.get_collection(name)
        dim = info.config.params.vectors.size
        dist_name = info.config.params.vectors.distance.name
        reverse_map = {v: k for k, v in METRIC_MAP.items()}
        metric = reverse_map.get(dist_name, "cosine")
        self._dimensions[name] = dim
        self._metrics[name] = metric
        return dim, metric

    def list_collections(self) -> list[dict[str, Any]]:
        result = []
        for col in self._client.get_collections().collections:
            dim, metric = self._get_info(col.name)
            info = self._client.get_collection(col.name)
            result.append(
                {
                    "name": col.name,
                    "dimension": dim,
                    "metric": metric,
                    "count": info.points_count,
                    "store_type": self.store_type,
                }
            )
        return result

    def collection_info(self, name: str) -> dict[str, Any]:
        dim, metric = self._get_info(name)
        info = self._client.get_collection(name)
        return {
            "name": name,
            "dimension": dim,
            "metric": metric,
            "count": info.points_count,
            "store_type": self.store_type,
        }

    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        from qdrant_client.models import PointStruct

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if metadata is None:
            metadata = [{} for _ in vectors]

        points = []
        for vid, vec, meta in zip(ids, vectors, metadata):
            payload = {**meta, "_vectory_id": vid}
            # Qdrant needs int or uuid point ids
            points.append(PointStruct(id=vid, vector=vec, payload=payload))
        self._client.upsert(collection_name=name, points=points)
        return ids

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=name,
            query=query,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        out = []
        for point in results.points:
            payload = dict(point.payload or {})
            vid = payload.pop("_vectory_id", str(point.id))
            out.append(SearchResult(id=vid, score=point.score, metadata=payload))
        return out

    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        results = self._client.retrieve(collection_name=name, ids=ids, with_vectors=True)
        out = []
        for point in results:
            payload = dict(point.payload or {})
            vid = payload.pop("_vectory_id", str(point.id))
            out.append({"id": vid, "vector": list(point.vector), "metadata": payload})
        return out

    def delete(self, name: str, ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList

        before = self._client.get_collection(name).points_count
        self._client.delete(collection_name=name, points_selector=PointIdsList(points=ids))
        after = self._client.get_collection(name).points_count
        return before - after

    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        results = self._client.retrieve(collection_name=name, ids=[id])
        if not results:
            raise KeyError(f"ID not found: {id}")
        payload = dict(results[0].payload or {})
        payload.update(metadata)
        self._client.set_payload(collection_name=name, payload=payload, points=[id])
