"""CollectionManager - manages multiple vector collections across different backends."""

from __future__ import annotations

from typing import Any

from vectory.backends.base import SearchResult, VectorStore
from vectory.backends.local import LocalStore

# Registry of available store types
STORE_REGISTRY: dict[str, type[VectorStore]] = {}


def _register_stores() -> None:
    """Register all available vector store backends."""
    import importlib

    # Local is always available
    from vectory.backends.local import LocalStore

    STORE_REGISTRY["local"] = LocalStore

    # Each entry: (registry_key, module_name, class_name, required_package)
    _optional = [
        ("chroma", "vectory.backends.chroma", "ChromaStore", "chromadb"),
        ("faiss", "vectory.backends.faiss_store", "FAISSStore", "faiss"),
        ("qdrant", "vectory.backends.qdrant", "QdrantStore", "qdrant_client"),
        ("milvus", "vectory.backends.milvus", "MilvusStore", "pymilvus"),
    ]
    for key, mod_name, cls_name, pkg in _optional:
        try:
            importlib.import_module(pkg)
            mod = importlib.import_module(mod_name)
            STORE_REGISTRY[key] = getattr(mod, cls_name)
        except Exception:
            pass


_register_stores()


class CollectionManager:
    """Platform for creating and managing multiple vector databases."""

    def __init__(self, data_dir: str = ".vectory_data", *, _memory_only: bool = False):
        self._data_dir = data_dir
        self._memory_only = _memory_only
        self._stores: dict[str, VectorStore] = {}
        # Track which store owns each collection
        self._collection_store: dict[str, str] = {}
        if not _memory_only:
            # Init local store with file persistence
            self._stores["local"] = LocalStore(data_dir)
            # Discover existing collections from local store
            for info in self._stores["local"].list_collections():
                self._collection_store[info["name"]] = "local"
        else:
            # In-memory local store (no disk I/O)
            from vectory.storage.backend import MemoryStorageBackend

            store = LocalStore.__new__(LocalStore)
            store._storage = MemoryStorageBackend()
            store._collections = {}
            store.store_type = "local"
            self._stores["local"] = store

    def _get_store(self, store_type: str) -> VectorStore:
        """Get or lazily initialize a store by type."""
        if store_type in self._stores:
            return self._stores[store_type]
        if store_type not in STORE_REGISTRY:
            raise ValueError(
                f"Store type '{store_type}' is not available. Install the required package."
            )
        cls = STORE_REGISTRY[store_type]
        self._stores[store_type] = cls()
        # Discover existing collections in this store
        try:
            for info in self._stores[store_type].list_collections():
                self._collection_store.setdefault(info["name"], store_type)
        except Exception:
            pass
        return self._stores[store_type]

    def available_stores(self) -> list[str]:
        """Return list of available store type names."""
        return sorted(STORE_REGISTRY.keys())

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        store_type: str = "local",
    ) -> dict[str, Any]:
        """Create a new collection on the specified backend."""
        if name in self._collection_store:
            raise ValueError(f"Collection '{name}' already exists")
        store = self._get_store(store_type)
        info = store.create_collection(name, dimension, metric)
        self._collection_store[name] = store_type
        return info

    def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get info about a collection."""
        store_type = self._collection_store.get(name)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        return self._get_store(store_type).collection_info(name)

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections across all initialized stores."""
        result = []
        seen = set()
        for store_type, store in self._stores.items():
            try:
                for info in store.list_collections():
                    if info["name"] not in seen:
                        seen.add(info["name"])
                        self._collection_store.setdefault(info["name"], store_type)
                        result.append(info)
            except Exception:
                pass
        return sorted(result, key=lambda x: x["name"])

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        store_type = self._collection_store.pop(name, None)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        self._get_store(store_type).drop_collection(name)

    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        store_type = self._collection_store.get(name)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        return self._get_store(store_type).insert(name, vectors, ids, metadata)

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        store_type = self._collection_store.get(name)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        return self._get_store(store_type).search(name, query, top_k, filter_metadata)

    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        store_type = self._collection_store.get(name)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        return self._get_store(store_type).get(name, ids)

    def delete_vectors(self, name: str, ids: list[str]) -> int:
        store_type = self._collection_store.get(name)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        return self._get_store(store_type).delete(name, ids)

    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        store_type = self._collection_store.get(name)
        if not store_type:
            raise KeyError(f"Collection '{name}' not found")
        self._get_store(store_type).update_metadata(name, id, metadata)

    # Legacy compatibility
    @classmethod
    def with_file_storage(cls, base_dir: str) -> CollectionManager:
        return cls(data_dir=base_dir)

    @classmethod
    def in_memory(cls) -> CollectionManager:
        return cls(_memory_only=True)
