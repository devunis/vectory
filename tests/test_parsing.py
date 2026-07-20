"""Tests for document parsing integrations."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from click.testing import CliRunner
from fastapi.testclient import TestClient
from vectory.api import server
from vectory.api.server import app
from vectory.cli.main import cli
from vectory.parsing.mineru import MinerUClient
from vectory.parsing.models import ParseResult
from vectory.parsing.paddleocr import parse_with_paddleocr
from vectory.parsing.service import parse_document


class _FakePaddleOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, source):
        return [{"res": {"input_path": source, "rec_texts": ["hello", "world"]}}]


class _FakeStructure:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, input):
        return [{"res": {"input_path": input, "rec_texts": ["structured"]}}]


def test_paddleocr_adapter_normalizes_text(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=_FakePaddleOCR, PPStructureV3=_FakeStructure),
    )
    source = tmp_path / "sample.png"
    source.write_text("not an actual image", encoding="utf-8")

    result = parse_with_paddleocr(source)

    assert result.provider == "paddleocr"
    assert result.text == "hello\nworld"
    assert result.raw[0]["res"]["input_path"] == str(source)


def test_paddleocr_adapter_supports_structure_mode(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=_FakePaddleOCR, PPStructureV3=_FakeStructure),
    )
    source = tmp_path / "sample.png"
    source.write_text("not an actual image", encoding="utf-8")

    result = parse_with_paddleocr(source, mode="structure", engine="transformers")

    assert result.text == "structured"
    assert result.metadata == {"mode": "structure", "engine": "transformers"}


def test_paddleocr_adapter_rejects_unknown_mode(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=_FakePaddleOCR, PPStructureV3=_FakeStructure),
    )
    source = tmp_path / "sample.png"
    source.write_text("not an actual image", encoding="utf-8")

    try:
        parse_with_paddleocr(source, mode="unknown")
    except ValueError as e:
        assert "mode must be one of" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_cli_parse_command_outputs_json(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=_FakePaddleOCR, PPStructureV3=_FakeStructure),
    )
    source = tmp_path / "sample.png"
    source.write_text("not an actual image", encoding="utf-8")

    result = CliRunner().invoke(cli, ["parse", str(source)])

    assert result.exit_code == 0
    assert '"provider": "paddleocr"' in result.output
    assert "hello\\nworld" in result.output


def test_cli_parse_command_reports_errors():
    result = CliRunner().invoke(
        cli,
        ["parse", "https://example.com/sample.pdf", "--source-type", "url"],
    )

    assert result.exit_code == 1
    assert "PaddleOCR only supports local path sources" in result.output


class _FakeResponse:
    def __init__(self, payload=None, text=""):
        self.payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeMinerUSession:
    def __init__(self):
        self.uploaded = False

    def post(self, url, json):
        assert url.endswith("/parse/file")
        assert json["file_name"] == "doc.pdf"
        return _FakeResponse(
            {
                "code": 0,
                "data": {
                    "task_id": "task-1",
                    "file_url": "https://upload.example/doc.pdf",
                },
            }
        )

    def put(self, url, data):
        assert url == "https://upload.example/doc.pdf"
        assert data.read()
        self.uploaded = True
        return _FakeResponse()

    def get(self, url):
        if url.endswith("/parse/task-1"):
            return _FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "task_id": "task-1",
                        "state": "done",
                        "markdown_url": "https://cdn.example/doc.md",
                    },
                }
            )
        return _FakeResponse(text="# Parsed")


def test_mineru_client_uploads_and_polls(tmp_path):
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF")
    session = _FakeMinerUSession()
    client = MinerUClient(session=session, poll_interval=0, timeout=1)

    result = client.parse_file(source, fetch_markdown=True)

    assert session.uploaded is True
    assert result.provider == "mineru"
    assert result.markdown == "# Parsed"
    assert result.metadata["task_id"] == "task-1"


class _FakeMinerUUrlSession:
    def __init__(self):
        self.payload = None
        self.get_called = False

    def post(self, url, json):
        assert url.endswith("/parse/url")
        self.payload = json
        return _FakeResponse(
            {
                "code": 0,
                "data": {
                    "task_id": "task-url",
                },
            }
        )

    def get(self, url):
        self.get_called = True
        return _FakeResponse()


def test_mineru_client_url_no_wait_returns_submission():
    session = _FakeMinerUUrlSession()
    client = MinerUClient(session=session, poll_interval=0, timeout=1)

    result = client.parse_url(
        "https://example.com/doc.pdf",
        file_name="doc.pdf",
        language="korean",
        page_range="1-3",
        wait=False,
    )

    assert session.payload == {
        "language": "korean",
        "page_range": "1-3",
        "enable_table": True,
        "enable_formula": True,
        "is_ocr": False,
        "file_name": "doc.pdf",
        "url": "https://example.com/doc.pdf",
    }
    assert session.get_called is False
    assert result.metadata["task_id"] == "task-url"
    assert result.raw["data"]["task_id"] == "task-url"


class _FakeFailedMinerUSession:
    def get(self, url):
        return _FakeResponse(
            {
                "code": 0,
                "data": {
                    "task_id": "task-failed",
                    "state": "failed",
                    "err_msg": "bad document",
                },
            }
        )


def test_mineru_client_failed_task_raises_runtime_error():
    client = MinerUClient(session=_FakeFailedMinerUSession(), poll_interval=0, timeout=1)

    try:
        client.poll("task-failed")
    except RuntimeError as e:
        assert "bad document" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_parse_document_rejects_unsupported_dispatch():
    try:
        parse_document("https://example.com/doc.pdf", source_type="url")
    except ValueError as e:
        assert "PaddleOCR only supports local path sources" in str(e)
    else:
        raise AssertionError("expected ValueError")

    try:
        parse_document("doc.pdf", provider="unknown")
    except ValueError as e:
        assert "provider must be one of" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_api_parse_endpoint_returns_normalized_result(monkeypatch):
    def fake_parse_document(*args, **kwargs):
        assert args == ("sample.pdf",)
        assert kwargs["provider"] == "mineru"
        assert kwargs["source_type"] == "url"
        return ParseResult(
            provider="mineru",
            source="sample.pdf",
            text="# Parsed",
            markdown="# Parsed",
            metadata={"task_id": "task-api"},
            raw={"data": {"state": "done"}},
        )

    monkeypatch.setattr(server, "parse_document", fake_parse_document)

    response = TestClient(app).post(
        "/parse",
        json={
            "provider": "mineru",
            "source": "sample.pdf",
            "source_type": "url",
            "fetch_markdown": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["markdown"] == "# Parsed"
    assert response.json()["metadata"]["task_id"] == "task-api"


def test_api_parse_endpoint_maps_client_errors(monkeypatch):
    def bad_request(*args, **kwargs):
        raise ValueError("bad provider")

    monkeypatch.setattr(server, "parse_document", bad_request)
    response = TestClient(app).post("/parse", json={"source": "sample.pdf"})
    assert response.status_code == 400
    assert response.json()["detail"] == "bad provider"

    def upstream_failure(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(server, "parse_document", upstream_failure)
    response = TestClient(app).post("/parse", json={"source": "sample.pdf"})
    assert response.status_code == 502
    assert response.json()["detail"] == "provider unavailable"
