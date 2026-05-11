"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Unified interface for all vector database backends."""

    store_type: str = "unknown"

    @abstractmethod
    def create_collection(self, name: str, dimension: int, metric: str) -> dict[str, Any]:
        """Create a collection. Returns collection info dict."""
        ...

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete a collection."""
        ...

    @abstractmethod
    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections with info."""
        ...

    @abstractmethod
    def collection_info(self, name: str) -> dict[str, Any]:
        """Get info about a specific collection."""
        ...

    @abstractmethod
    def insert(
        self,
        name: str,
        vectors: list[list[float]],
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Insert vectors. Returns assigned IDs."""
        ...

    @abstractmethod
    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        ...

    @abstractmethod
    def get(self, name: str, ids: list[str]) -> list[dict[str, Any]]:
        """Get vectors by IDs."""
        ...

    @abstractmethod
    def delete(self, name: str, ids: list[str]) -> int:
        """Delete vectors by IDs. Returns count deleted."""
        ...

    @abstractmethod
    def update_metadata(self, name: str, id: str, metadata: dict[str, Any]) -> None:
        """Update metadata for a vector."""
        ...
