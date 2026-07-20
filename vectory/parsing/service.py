"""Parsing service dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vectory.parsing.mineru import MinerUClient
from vectory.parsing.models import ParseResult
from vectory.parsing.paddleocr import parse_with_paddleocr


def parse_document(
    source: str | Path,
    *,
    provider: str = "paddleocr",
    source_type: str = "path",
    output_dir: str | Path | None = None,
    wait: bool = True,
    fetch_markdown: bool = False,
    **options: Any,
) -> ParseResult:
    """Parse a document with a configured provider."""
    if provider == "paddleocr":
        if source_type != "path":
            raise ValueError("PaddleOCR only supports local path sources")
        return parse_with_paddleocr(
            source,
            mode=options.get("mode", "ocr"),
            engine=options.get("engine", "paddle"),
            output_dir=output_dir,
        )

    if provider == "mineru":
        client = MinerUClient(
            poll_interval=options.get("poll_interval", 2.0),
            timeout=options.get("timeout", 120.0),
        )
        mineru_options = {
            "language": options.get("language", "ch"),
            "page_range": options.get("page_range"),
            "enable_table": options.get("enable_table", True),
            "enable_formula": options.get("enable_formula", True),
            "is_ocr": options.get("is_ocr", False),
            "wait": wait,
            "fetch_markdown": fetch_markdown,
        }
        if source_type == "url":
            return client.parse_url(
                str(source), file_name=options.get("file_name"), **mineru_options
            )
        if source_type == "path":
            return client.parse_file(source, **mineru_options)
        raise ValueError("source_type must be one of: path, url")

    raise ValueError("provider must be one of: paddleocr, mineru")
