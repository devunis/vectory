"""PaddleOCR parsing adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vectory.parsing.models import ParseResult


def parse_with_paddleocr(
    source: str | Path,
    *,
    mode: str = "ocr",
    engine: str = "paddle",
    output_dir: str | Path | None = None,
) -> ParseResult:
    """Parse a local image/PDF-like input with PaddleOCR.

    PaddleOCR is optional because it brings large inference dependencies.
    Install it with `pip install -e ".[parse]"`.
    """
    try:
        from paddleocr import PaddleOCR, PPStructureV3
    except ImportError as e:
        raise RuntimeError(
            'PaddleOCR is not installed. Install with: pip install -e ".[parse]"'
        ) from e

    source_path = Path(source)
    if mode == "ocr":
        parser = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine=engine,
        )
        raw_results = parser.predict(str(source_path))
    elif mode == "structure":
        parser = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            engine=engine,
        )
        raw_results = parser.predict(input=str(source_path))
    else:
        raise ValueError("mode must be one of: ocr, structure")

    if output_dir is not None:
        save_path = Path(output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for item in raw_results:
            _call_if_available(item, "save_to_json", save_path=str(save_path))
            _call_if_available(item, "save_to_markdown", save_path=str(save_path))

    raw = _jsonable(raw_results)
    lines = _extract_text_lines(raw)

    return ParseResult(
        provider="paddleocr",
        source=str(source_path),
        text="\n".join(lines),
        metadata={"mode": mode, "engine": engine},
        raw=raw,
    )


def _call_if_available(item: Any, method_name: str, **kwargs: Any) -> None:
    method = getattr(item, method_name, None)
    if method is not None:
        method(**kwargs)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "json"):
        try:
            return _jsonable(value.json)
        except TypeError:
            pass
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _extract_text_lines(value: Any) -> list[str]:
    lines: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            rec_texts = node.get("rec_texts")
            if isinstance(rec_texts, list):
                lines.extend(str(text) for text in rec_texts if str(text).strip())
            rec_text = node.get("rec_text")
            if isinstance(rec_text, str) and rec_text.strip():
                lines.append(rec_text)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return lines
