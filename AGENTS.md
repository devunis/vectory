# AGENTS.md

## Project

Vectory is a Python 3.10+ vector database management package. It includes a
NumPy-backed local vector store, optional external vector store adapters, a
FastAPI server, a Click CLI, and a small HTML UI served by the API.

## Setup

Install the package in editable mode with development tooling:

```bash
pip install -e ".[dev]"
```

Optional vector store extras are defined in `pyproject.toml`:

```bash
pip install -e ".[chroma]"
pip install -e ".[faiss]"
pip install -e ".[qdrant]"
pip install -e ".[milvus]"
pip install -e ".[all]"
```

## Quality Checks

Run the full local quality gate before handing off changes:

```bash
python scripts/quality.py
```

Equivalent individual commands:

```bash
python -m ruff check . --fix
python -m ruff format .
python -m pytest tests/ -v
```

## Git Hooks

This repository uses versioned hooks in `.githooks/`. Enable them locally with:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook auto-fixes staged Python files with Ruff, formats them, then
restages the fixed files and runs the test suite.

## Testing Scope

The current tests cover:

- core distance functions and collection behavior
- collection manager persistence and in-memory behavior
- FastAPI collection/vector endpoints and UI entrypoint
- Click CLI create/insert/search/info/list/delete workflow
- built-in local vector store contract

Optional Chroma, FAISS, Qdrant, and Milvus adapters are present but should be
tested separately when those dependencies are installed.

## Repository Hygiene

Do not commit generated data or cache directories such as `.vectory_data/`,
`.vectory_chroma/`, `__pycache__/`, `.pytest_cache/`, or `*.egg-info/`.
