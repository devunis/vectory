"""Milvus vector store backend (using milvus-lite for local execution)."""

from __future__ import annotations

import uuid
from typing import Any

from vectory.backends.base import SearchResult, VectorStore

METRIC_MAP = {"cosine": "COSINE", "euclidean": "L2", "dot_product": "IP"}


class MilvusStore(VectorStore):
    """Milvus backend (milvus-lite embedded or remote server)."""

    store_type = "milvus"

    def __init__(self, uri: str = ".vectory_milvus/milvus.db"):
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=uri)

    def create_collection(self, name: str, dimension: int, metric: str) -> dict[str, Any]:
        from pymilvus import CollectionSchema, DataType, FieldSchema

        if self._client.has_collection(name):
            raise ValueError(f"Collection '{name}' already exists")

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

        milvus_metric = METRIC_MAP.get(metric, "COSINE")
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type=milvus_metric, index_type="FLAT")

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        return {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "count": 0,
            "store_type": self.store_type,
        }

    def drop_collection(self, name: str) -> None:
        self._client.drop_collection(name)

    def list_collections(self) -> list[dict[str, Any]]:
        result = []
        for name in self._client.list_collections():
            info = self._get_collection_meta(name)
            result.append(info)
        return result

    def _get_collection_meta(self, name: str) -> dict[str, Any]:
        desc = self._client.describe_collection(name)
        dimension = 0
        for field in desc.get("fields", []):
            if field.get("name") == "vector":
                params = field.get("params", {})
                dimension = params.get("dim", 0)
                break
        stats = self._client.get_collection_stats(name)
        count = stats.get("row_count", 0)
        return {
            "name": name,
            "dimension": dimension,
            "metric": "cosine",
            "count": count,
            "store_type": self.store_type,
        }

    def collection_info(self, name: str) -> dict[str, Any]:
        if not self._client.has_collection(name):
            raise KeyError(f"Collection '{name}' not found")
        return self._get_collection_meta(name)

    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        import json

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if metadata is None:
            metadata = [{} for _ in vectors]

        data = [
            {
                "id": vid,
                "vector": vec,
                "metadata_json": json.dumps(meta, ensure_ascii=False),
            }
            for vid, vec, meta in zip(ids, vectors, metadata)
        ]
        self._client.insert(collection_name=name, data=data)
        return ids

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        import json

        filter_expr = None
        if filter_metadata:
            conditions = []
            for k, v in filter_metadata.items():
                if isinstance(v, str):
                    conditions.append(f'metadata_json like \'%"{k}": "{v}"%\'')
                else:
                    conditions.append(f"metadata_json like '%\"{k}\": {v}%'")
            filter_expr = " and ".join(conditions)

        results = self._client.search(
            collection_name=name,
            data=[query],
            limit=top_k,
            output_fields=["id", "metadata_json"],
            filter=filter_expr,
        )
        out = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                meta_str = entity.get("metadata_json", "{}")
                meta = json.loads(meta_str) if meta_str else {}
                out.append(
                    SearchResult(
                        id=entity.get("id", str(hit["id"])),
                        score=hit["distance"],
                        metadata=meta,
                    )
                )
        return out

    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        import json

        filter_expr = f"id in {ids}"
        results = self._client.query(
            collection_name=name,
            filter=filter_expr,
            output_fields=["id", "vector", "metadata_json"],
        )
        out = []
        for row in results:
            meta = json.loads(row.get("metadata_json", "{}"))
            out.append({"id": row["id"], "vector": row.get("vector", []), "metadata": meta})
        return out

    def delete(self, name: str, ids: list[str]) -> int:
        filter_expr = f"id in {ids}"
        result = self._client.delete(collection_name=name, filter=filter_expr)
        return len(ids) if result else 0

    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        import json

        existing = self.get(name, [id])
        if not existing:
            raise KeyError(f"ID not found: {id}")
        merged = {**existing[0]["metadata"], **metadata}
        self._client.upsert(
            collection_name=name,
            data=[
                {
                    "id": id,
                    "vector": existing[0]["vector"],
                    "metadata_json": json.dumps(merged, ensure_ascii=False),
                },
            ],
        )
