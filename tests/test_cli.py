"""Tests for the Click command line interface."""

import json

from click.testing import CliRunner
from vectory.cli.main import cli


def invoke(runner: CliRunner, data_dir, args: list[str], input_text: str | None = None):
    return runner.invoke(cli, ["--data-dir", str(data_dir), *args], input=input_text)


def test_cli_create_insert_search_info_delete_workflow(tmp_path):
    runner = CliRunner()

    result = invoke(runner, tmp_path, ["create", "docs", "--dimension", "3"])
    assert result.exit_code == 0
    assert "Created collection 'docs'" in result.output

    payload_path = tmp_path / "vectors.json"
    payload_path.write_text(
        json.dumps(
            {
                "vectors": [[1, 0, 0], [0, 1, 0]],
                "ids": ["a", "b"],
                "metadata": [{"label": "x"}, {"label": "y"}],
            }
        ),
        encoding="utf-8",
    )

    result = invoke(runner, tmp_path, ["insert", "docs", str(payload_path)])
    assert result.exit_code == 0
    assert "Inserted 2 vectors." in result.output

    result = invoke(runner, tmp_path, ["search", "docs", "[1, 0, 0]", "--top-k", "1"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "label" in result.output

    result = invoke(runner, tmp_path, ["info", "docs"])
    assert result.exit_code == 0
    info = json.loads(result.output)
    assert info["count"] == 2
    assert info["store_type"] == "local"

    result = invoke(runner, tmp_path, ["list"])
    assert result.exit_code == 0
    assert "docs" in result.output

    result = invoke(runner, tmp_path, ["delete", "docs"], input_text="y\n")
    assert result.exit_code == 0
    assert "Deleted collection 'docs'." in result.output

    result = invoke(runner, tmp_path, ["list"])
    assert result.exit_code == 0
    assert "No collections found." in result.output


def test_cli_stores_lists_local_backend(tmp_path):
    runner = CliRunner()

    result = invoke(runner, tmp_path, ["stores"])

    assert result.exit_code == 0
    assert "local" in result.output
