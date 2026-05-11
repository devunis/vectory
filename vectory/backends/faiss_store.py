"""FAISS vector store backend."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from vectory.backends.base import SearchResult, VectorStore


class FAISSStore(VectorStore):
    """FAISS backend (local library, no server needed)."""

    store_type = "faiss"

    def __init__(self, persist_dir: str = ".vectory_faiss"):
        import faiss  # noqa: F401 — validate import

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, Any] = {}  # name -> faiss.Index
        self._ids: dict[str, list[str]] = {}
        self._metadata: dict[str, list[dict[str, Any]]] = {}
        self._dimensions: dict[str, int] = {}
        self._metrics: dict[str, str] = {}
        self._load_all()

    def _meta_path(self, name: str) -> Path:
        return self._persist_dir / f"{name}.json"

    def _index_path(self, name: str) -> Path:
        return self._persist_dir / f"{name}.faiss"

    def _load_all(self) -> None:
        import faiss

        for meta_file in self._persist_dir.glob("*.json"):
            name = meta_file.stem
            index_file = self._index_path(name)
            if not index_file.exists():
                continue
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ids[name] = data["ids"]
            self._metadata[name] = data["metadata"]
            self._dimensions[name] = data["dimension"]
            self._metrics[name] = data["metric"]
            self._indexes[name] = faiss.read_index(str(index_file))

    def _save(self, name: str) -> None:
        import faiss

        faiss.write_index(self._indexes[name], str(self._index_path(name)))
        with open(self._meta_path(name), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": self._ids[name],
                    "metadata": self._metadata[name],
                    "dimension": self._dimensions[name],
                    "metric": self._metrics[name],
                },
                f,
                ensure_ascii=False,
            )

    def _make_index(self, dimension: int, metric: str) -> Any:
        import faiss

        if metric == "cosine":
            index = faiss.IndexFlatIP(dimension)
        elif metric == "dot_product":
            index = faiss.IndexFlatIP(dimension)
        else:  # euclidean
            index = faiss.IndexFlatL2(dimension)
        return index

    def create_collection(self, name: str, dimension: int, metric: str) -> dict[str, Any]:
        if name in self._indexes:
            raise ValueError(f"Collection '{name}' already exists")
        self._indexes[name] = self._make_index(dimension, metric)
        self._ids[name] = []
        self._metadata[name] = []
        self._dimensions[name] = dimension
        self._metrics[name] = metric
        self._save(name)
        return {
            "name": name,
            "dimension": dimension,
            "metric": metric,
            "count": 0,
            "store_type": self.store_type,
        }

    def drop_collection(self, name: str) -> None:
        self._indexes.pop(name, None)
        self._ids.pop(name, None)
        self._metadata.pop(name, None)
        self._dimensions.pop(name, None)
        self._metrics.pop(name, None)
        self._meta_path(name).unlink(missing_ok=True)
        self._index_path(name).unlink(missing_ok=True)

    def list_collections(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "dimension": self._dimensions[name],
                "metric": self._metrics[name],
                "count": len(self._ids[name]),
                "store_type": self.store_type,
            }
            for name in sorted(self._indexes.keys())
        ]

    def collection_info(self, name: str) -> dict[str, Any]:
        if name not in self._indexes:
            raise KeyError(f"Collection '{name}' not found")
        return {
            "name": name,
            "dimension": self._dimensions[name],
            "metric": self._metrics[name],
            "count": len(self._ids[name]),
            "store_type": self.store_type,
        }

    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        if name not in self._indexes:
            raise KeyError(f"Collection '{name}' not found")
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if metadata is None:
            metadata = [{} for _ in vectors]

        arr = np.array(vectors, dtype=np.float32)
        if self._metrics[name] == "cosine":
            # Normalize for cosine similarity via inner product
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-10, norms)
            arr = arr / norms

        self._indexes[name].add(arr)
        self._ids[name].extend(ids)
        self._metadata[name].extend(metadata)
        self._save(name)
        return ids

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if name not in self._indexes:
            raise KeyError(f"Collection '{name}' not found")
        if len(self._ids[name]) == 0:
            return []

        q = np.array([query], dtype=np.float32)
        if self._metrics[name] == "cosine":
            norm = np.linalg.norm(q, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1e-10, norm)
            q = q / norm

        k = min(top_k, len(self._ids[name]))
        distances, indices = self._indexes[name].search(q, k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0:
                continue
            meta = self._metadata[name][idx]
            if filter_metadata and not all(
                meta.get(fk) == fv for fk, fv in filter_metadata.items()
            ):
                continue
            score = float(distances[0][i])
            if self._metrics[name] in ("cosine", "dot_product"):
                score = -score  # Convert IP to distance (lower = better)
            results.append(SearchResult(id=self._ids[name][idx], score=score, metadata=meta))
        return results

    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        if name not in self._indexes:
            raise KeyError(f"Collection '{name}' not found")
        id_map = {vid: i for i, vid in enumerate(self._ids[name])}
        out = []
        for vid in ids:
            if vid not in id_map:
                raise KeyError(f"ID not found: {vid}")
            idx = id_map[vid]
            vec = self._indexes[name].reconstruct(idx)
            out.append(
                {
                    "id": vid,
                    "vector": vec.tolist(),
                    "metadata": self._metadata[name][idx],
                }
            )
        return out

    def delete(self, name: str, ids: list[str]) -> int:
        if name not in self._indexes:
            raise KeyError(f"Collection '{name}' not found")
        to_delete = set(ids)
        keep = [i for i, vid in enumerate(self._ids[name]) if vid not in to_delete]
        deleted = len(self._ids[name]) - len(keep)
        if deleted > 0:
            # Rebuild index with remaining vectors
            new_ids = [self._ids[name][i] for i in keep]
            new_meta = [self._metadata[name][i] for i in keep]
            new_index = self._make_index(self._dimensions[name], self._metrics[name])
            if keep:
                vecs = np.array(
                    [self._indexes[name].reconstruct(i) for i in keep], dtype=np.float32
                )
                new_index.add(vecs)
            self._indexes[name] = new_index
            self._ids[name] = new_ids
            self._metadata[name] = new_meta
            self._save(name)
        return deleted

    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        if name not in self._indexes:
            raise KeyError(f"Collection '{name}' not found")
        id_map = {vid: i for i, vid in enumerate(self._ids[name])}
        if id not in id_map:
            raise KeyError(f"ID not found: {id}")
        self._metadata[name][id_map[id]].update(metadata)
        self._save(name)
