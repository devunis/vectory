"""MinerU Agent lightweight parsing client."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from vectory.parsing.models import ParseResult


class MinerUClient:
    """Client for MinerU's no-token Agent lightweight document parsing API."""

    def __init__(
        self,
        *,
        base_url: str = "https://mineru.net/api/v1/agent",
        session: Any | None = None,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or _requests()
        self.poll_interval = poll_interval
        self.timeout = timeout

    def parse_url(
        self,
        url: str,
        *,
        file_name: str | None = None,
        language: str = "ch",
        page_range: str | None = None,
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = False,
        wait: bool = True,
        fetch_markdown: bool = False,
    ) -> ParseResult:
        payload = self._payload(
            language=language,
            page_range=page_range,
            enable_table=enable_table,
            enable_formula=enable_formula,
            is_ocr=is_ocr,
            file_name=file_name,
        )
        payload["url"] = url
        submitted = self._post_json("/parse/url", payload)
        return self._result(
            source=url,
            submitted=submitted,
            wait=wait,
            fetch_markdown=fetch_markdown,
        )

    def parse_file(
        self,
        path: str | Path,
        *,
        language: str = "ch",
        page_range: str | None = None,
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = False,
        wait: bool = True,
        fetch_markdown: bool = False,
    ) -> ParseResult:
        source_path = Path(path)
        payload = self._payload(
            language=language,
            page_range=page_range,
            enable_table=enable_table,
            enable_formula=enable_formula,
            is_ocr=is_ocr,
            file_name=source_path.name,
        )
        submitted = self._post_json("/parse/file", payload)
        file_url = submitted["data"]["file_url"]
        with source_path.open("rb") as f:
            upload = self.session.put(file_url, data=f)
        upload.raise_for_status()

        return self._result(
            source=str(source_path),
            submitted=submitted,
            wait=wait,
            fetch_markdown=fetch_markdown,
        )

    def poll(self, task_id: str, *, fetch_markdown: bool = False) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        latest: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            latest = self._get_json(f"/parse/{task_id}")
            data = latest.get("data", {})
            state = data.get("state")
            if state == "done":
                if fetch_markdown and data.get("markdown_url"):
                    markdown_resp = self.session.get(data["markdown_url"])
                    markdown_resp.raise_for_status()
                    data["markdown"] = markdown_resp.text
                return latest
            if state == "failed":
                raise RuntimeError(data.get("err_msg") or "MinerU parsing failed")
            time.sleep(self.poll_interval)

        raise TimeoutError(f"Timed out waiting for MinerU task {task_id}: {latest}")

    def _result(
        self,
        *,
        source: str,
        submitted: dict[str, Any],
        wait: bool,
        fetch_markdown: bool,
    ) -> ParseResult:
        task_id = submitted["data"]["task_id"]
        raw = self.poll(task_id, fetch_markdown=fetch_markdown) if wait else submitted
        data = raw.get("data", {})
        markdown = data.get("markdown")
        return ParseResult(
            provider="mineru",
            source=source,
            text=markdown or "",
            markdown=markdown,
            metadata={
                "task_id": task_id,
                "state": data.get("state"),
                "markdown_url": data.get("markdown_url"),
            },
            raw=raw,
        )

    def _payload(self, **values: Any) -> dict[str, Any]:
        return {k: v for k, v in values.items() if v is not None}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.base_url}{path}", json=payload)
        response.raise_for_status()
        data = response.json()
        self._ensure_ok(data)
        return data

    def _get_json(self, path: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}")
        response.raise_for_status()
        data = response.json()
        self._ensure_ok(data)
        return data

    def _ensure_ok(self, data: dict[str, Any]) -> None:
        if data.get("code") not in (0, None):
            raise RuntimeError(data.get("msg") or f"MinerU API error: {data}")


def _requests() -> Any:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError(
            'requests is not installed. Install with: pip install -e ".[parse]"'
        ) from e
    return requests
