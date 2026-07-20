"""Shared parsing result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseResult:
    """Normalized document parsing response."""

    provider: str
    source: str
    text: str = ""
    markdown: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "source": self.source,
            "text": self.text,
            "markdown": self.markdown,
            "metadata": self.metadata,
            "raw": self.raw,
        }
