"""Tests for RAG ingestion and retrieval."""

from __future__ import annotations

import json

from click.testing import CliRunner
from fastapi.testclient import TestClient
from vectory.api.server import app, set_manager
from vectory.cli.main import cli
from vectory.engine.manager import CollectionManager
from vectory.rag import RagPipeline
from vectory.rag.bm25 import BM25Index
from vectory.rag.chunking import chunk_document
from vectory.rag.models import Document


def test_chunk_document_creates_overlapping_chunks():
    document = Document(id="doc", text="one two three four five six", metadata={"source": "unit"})

    chunks = chunk_document(document, chunk_size=3, chunk_overlap=1)

    assert [chunk.text for chunk in chunks] == ["one two three", "three four five", "five six"]
    assert chunks[0].metadata["source"] == "unit"
    assert chunks[1].metadata["chunk_start"] == 2


def test_chunk_document_rejects_invalid_overlap():
    document = Document(id="doc", text="one two")

    try:
        chunk_document(document, chunk_size=2, chunk_overlap=2)
    except ValueError as e:
        assert "chunk_overlap must be smaller" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_bm25_prefers_exact_keyword_match():
    chunks = chunk_document(
        Document(
            id="doc",
            text="database vector search TS-999 billing failure unrelated banana",
        ),
        chunk_size=3,
        chunk_overlap=0,
    )
    index = BM25Index(chunks)

    results = index.search("TS-999", top_k=1)

    assert results[0][0] == "doc:1"
    assert results[0][1] > 0


def test_rag_pipeline_ingests_and_hybrid_searches(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)

    ingest = pipeline.ingest_text(
        "docs",
        "Vectory supports hybrid RAG search. Billing code TS-999 needs exact matching.",
        document_id="doc-1",
        metadata={"kind": "guide"},
        chunk_size=6,
        chunk_overlap=1,
        embedding_dimension=64,
    )
    results = pipeline.search("docs", "TS-999 exact billing", strategy="hybrid", top_k=2)

    assert ingest["chunk_count"] == 2
    assert results
    assert results[0].document_id == "doc-1"
    assert "TS-999" in results[0].text
    assert "bm25" in results[0].scores


def test_rag_pipeline_supports_query_expansion_and_hyde(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text(
        "docs",
        "PaddleOCR extracts text from scanned receipts. MinerU parses PDF tables.",
        document_id="doc-1",
        chunk_size=8,
        chunk_overlap=0,
        embedding_dimension=64,
    )

    expanded = pipeline.search(
        "docs",
        "document parser",
        strategy="bm25",
        query_expansions=["PDF tables MinerU"],
        top_k=1,
    )
    hyde = pipeline.search(
        "docs",
        "receipt extraction",
        strategy="vector",
        hypothetical_document="PaddleOCR scanned receipt text extraction",
        top_k=1,
    )

    assert expanded[0].document_id == "doc-1"
    assert hyde[0].document_id == "doc-1"


def test_rag_pipeline_mmr_returns_requested_count(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text(
        "docs",
        "alpha beta gamma. alpha beta delta. unique finance compliance report.",
        document_id="doc-1",
        chunk_size=3,
        chunk_overlap=0,
        embedding_dimension=64,
    )

    results = pipeline.search("docs", "alpha beta", strategy="hybrid", top_k=2, mmr_lambda=0.7)

    assert len(results) == 2
    assert len({result.id for result in results}) == 2


def test_rag_pipeline_rejects_dimension_mismatch(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text("docs", "first document", embedding_dimension=32)

    try:
        pipeline.ingest_text("docs", "second document", embedding_dimension=64)
    except ValueError as e:
        assert "has dimension 32" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_rag_pipeline_adaptive_routes_exact_queries_to_keyword_search(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text(
        "docs",
        "Error code TS-999 means billing verification failed.",
        document_id="doc-1",
        chunk_size=6,
        chunk_overlap=0,
        embedding_dimension=64,
    )

    results = pipeline.search("docs", "TS-999", strategy="adaptive", top_k=1)

    assert results[0].document_id == "doc-1"
    assert "bm25" in results[0].scores
    assert "reranker" in results[0].scores


def test_rag_pipeline_corrective_marks_confidence_scores(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text(
        "docs",
        "CRAG evaluates retrieval quality and performs corrective retrieval.",
        document_id="doc-1",
        chunk_size=6,
        chunk_overlap=0,
        embedding_dimension=64,
    )

    results = pipeline.search(
        "docs",
        "retrieval quality correction",
        strategy="corrective",
        top_k=1,
        corrective_threshold=0.99,
    )

    assert results[0].document_id == "doc-1"
    assert "corrective_confidence" in results[0].scores
    assert "corrective_coverage" in results[0].scores


def test_rag_pipeline_raptor_returns_summary_chunks(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    ingest = pipeline.ingest_text(
        "docs",
        "RAPTOR clusters chunks. It builds summaries. Multi hop questions use hierarchy.",
        document_id="doc-1",
        chunk_size=3,
        chunk_overlap=0,
        embedding_dimension=64,
        enable_raptor=True,
        raptor_group_size=2,
    )

    results = pipeline.search("docs", "hierarchy summaries", strategy="raptor", top_k=3)

    assert ingest["summary_count"] > 0
    assert any(result.metadata.get("summary_type") == "raptor" for result in results)


def test_rag_pipeline_graph_expands_entity_neighbors(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text(
        "docs",
        "Alice owns ProjectX roadmap. Bob reviews ProjectX risks. Carol tracks finance.",
        document_id="doc-1",
        chunk_size=4,
        chunk_overlap=0,
        embedding_dimension=64,
    )

    results = pipeline.search("docs", "Alice", strategy="graph", top_k=3, candidate_k=1)

    assert any("Bob" in result.text for result in results)
    assert any(result.scores.get("graph_expansion") == 1.0 for result in results)


def test_rag_pipeline_lexical_reranker_boosts_overlap(tmp_path):
    manager = CollectionManager(data_dir=str(tmp_path))
    pipeline = RagPipeline(manager)
    pipeline.ingest_text(
        "docs",
        "alpha beta exact answer. unrelated terms only.",
        document_id="doc-1",
        chunk_size=3,
        chunk_overlap=0,
        embedding_dimension=64,
    )

    results = pipeline.search("docs", "alpha beta", strategy="hybrid", top_k=1, reranker="lexical")

    assert "alpha beta" in results[0].text
    assert results[0].scores["reranker"] > 0


def test_cli_rag_ingest_and_search(tmp_path):
    runner = CliRunner()
    source = tmp_path / "doc.txt"
    source.write_text("GraphRAG connects entities. Hybrid search combines BM25 and vectors.")

    ingest = runner.invoke(
        cli,
        [
            "--data-dir",
            str(tmp_path / "data"),
            "rag",
            "ingest",
            "docs",
            str(source),
            "--document-id",
            "doc-cli",
            "--chunk-size",
            "6",
            "--chunk-overlap",
            "1",
            "--embedding-dimension",
            "64",
            "--enable-raptor",
            "--raptor-group-size",
            "2",
        ],
    )
    search = runner.invoke(
        cli,
        [
            "--data-dir",
            str(tmp_path / "data"),
            "rag",
            "search",
            "docs",
            "BM25 vectors",
            "--strategy",
            "hybrid",
            "--top-k",
            "1",
            "--reranker",
            "lexical",
        ],
    )

    assert ingest.exit_code == 0
    assert json.loads(ingest.output)["document_id"] == "doc-cli"
    assert json.loads(ingest.output)["summary_count"] > 0
    assert search.exit_code == 0
    assert json.loads(search.output)[0]["document_id"] == "doc-cli"


def test_api_rag_ingest_and_search(tmp_path):
    set_manager(CollectionManager(data_dir=str(tmp_path)))
    client = TestClient(app)

    ingest = client.post(
        "/rag/ingest",
        json={
            "collection": "docs",
            "text": "Contextual retrieval prepends chunk context before BM25 and embeddings.",
            "document_id": "doc-api",
            "chunk_size": 6,
            "chunk_overlap": 1,
            "embedding_dimension": 64,
            "enable_raptor": True,
            "raptor_group_size": 2,
        },
    )
    search = client.post(
        "/rag/search",
        json={
            "collection": "docs",
            "query": "contextual BM25 embeddings",
            "strategy": "hybrid",
            "top_k": 1,
            "reranker": "lexical",
        },
    )

    assert ingest.status_code == 200
    assert ingest.json()["document_id"] == "doc-api"
    assert ingest.json()["summary_count"] >= 0
    assert search.status_code == 200
    assert search.json()[0]["document_id"] == "doc-api"


def test_api_rag_search_missing_collection_returns_404(tmp_path):
    set_manager(CollectionManager(data_dir=str(tmp_path)))
    response = TestClient(app).post(
        "/rag/search",
        json={"collection": "missing", "query": "anything"},
    )

    assert response.status_code == 404
