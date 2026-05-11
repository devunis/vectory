"""Storage backends for persisting vector collections."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, name: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    def load(self, name: str) -> dict[str, Any]: ...

    @abstractmethod
    def delete(self, name: str) -> None: ...

    @abstractmethod
    def list_collections(self) -> list[str]: ...

    @abstractmethod
    def exists(self, name: str) -> bool: ...


class FileStorageBackend(StorageBackend):
    """JSON file-based storage backend."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.base_dir / f"{name}.json"

    def save(self, name: str, data: dict[str, Any]) -> None:
        with open(self._path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Collection '{name}' not found on disk")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            path.unlink()

    def list_collections(self) -> list[str]:
        collections = []
        required_keys = {"name", "dimension", "metric", "ids", "vectors", "metadata"}
        for path in self.base_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(data, dict)
                and required_keys.issubset(data)
                and data["name"] == path.stem
            ):
                collections.append(path.stem)
        return collections

    def exists(self, name: str) -> bool:
        return self._path(name).exists()


class MemoryStorageBackend(StorageBackend):
    """In-memory storage backend (no persistence)."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, name: str, data: dict[str, Any]) -> None:
        self._store[name] = data

    def load(self, name: str) -> dict[str, Any]:
        if name not in self._store:
            raise KeyError(f"Collection '{name}' not found")
        return self._store[name]

    def delete(self, name: str) -> None:
        self._store.pop(name, None)

    def list_collections(self) -> list[str]:
        return list(self._store.keys())

    def exists(self, name: str) -> bool:
        return name in self._store
