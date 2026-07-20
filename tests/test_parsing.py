"""Tests for document parsing integrations."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from click.testing import CliRunner
from vectory.cli.main import cli
from vectory.parsing.mineru import MinerUClient
from vectory.parsing.paddleocr import parse_with_paddleocr


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
