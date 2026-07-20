"""Composable RAG ingestion and retrieval pipeline."""

from __future__ import annotations

import uuid
from typing import Any

from vectory.engine.manager import CollectionManager
from vectory.rag.bm25 import BM25Index
from vectory.rag.chunking import chunk_document
from vectory.rag.embeddings import HashingEmbedder
from vectory.rag.fusion import apply_mmr, reciprocal_rank_fusion
from vectory.rag.models import Chunk, Document, RetrievalResult
from vectory.rag.store import RagCorpus


class RagPipeline:
    """RAG indexing and search orchestration on top of Vectory collections."""

    def __init__(self, manager: CollectionManager, *, data_dir: str | None = None) -> None:
        self.manager = manager
        self.data_dir = data_dir or getattr(manager, "_data_dir", ".vectory_data")

    def ingest_text(
        self,
        collection: str,
        text: str,
        *,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        embedding_dimension: int = 384,
        contextual_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Chunk, embed, and store text for RAG retrieval."""
        document = Document(
            id=document_id or str(uuid.uuid4()),
            text=text,
            metadata=metadata or {},
        )
        chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            contextual_prefix=contextual_prefix,
        )
        if not chunks:
            raise ValueError("document text produced no chunks")

        self._ensure_collection(collection, embedding_dimension)
        embedder = HashingEmbedder(embedding_dimension)
        vectors = embedder.embed_many([chunk.text for chunk in chunks])
        vector_metadata = [
            {
                **chunk.metadata,
                "rag": True,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        ids = self.manager.insert(
            collection,
            vectors,
            ids=[chunk.id for chunk in chunks],
            metadata=vector_metadata,
        )
        RagCorpus(self.data_dir, collection).add_document(document, chunks)

        return {
            "collection": collection,
            "document_id": document.id,
            "chunk_count": len(chunks),
            "chunk_ids": ids,
            "embedding_dimension": embedding_dimension,
        }

    def search(
        self,
        collection: str,
        query: str,
        *,
        strategy: str = "hybrid",
        top_k: int = 5,
        candidate_k: int = 30,
        rrf_k: int = 60,
        mmr_lambda: float | None = None,
        query_expansions: list[str] | None = None,
        hypothetical_document: str | None = None,
    ) -> list[RetrievalResult]:
        """Search a RAG collection with vector, BM25, or hybrid retrieval."""
        if strategy not in {"vector", "bm25", "hybrid"}:
            raise ValueError("strategy must be one of: vector, bm25, hybrid")
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be greater than 0")

        info = self.manager.get_collection_info(collection)
        embedder = HashingEmbedder(info["dimension"])
        corpus = RagCorpus(self.data_dir, collection)
        chunks = corpus.list_chunks()
        if not chunks:
            return []

        queries = [query, *(query_expansions or [])]
        vector_query = hypothetical_document or query
        candidate_scores: dict[str, dict[str, float]] = {}
        ranked_lists: list[list[str]] = []

        if strategy in {"vector", "hybrid"}:
            vector_ranked = self._vector_ranked(
                collection,
                embedder.embed(vector_query),
                candidate_k,
            )
            ranked_lists.append([item_id for item_id, _ in vector_ranked])
            for item_id, score in vector_ranked:
                candidate_scores.setdefault(item_id, {})["vector"] = score

        if strategy in {"bm25", "hybrid"}:
            bm25 = BM25Index(chunks)
            for expanded_query in queries:
                bm25_ranked = bm25.search(expanded_query, top_k=candidate_k)
                ranked_lists.append([item_id for item_id, _ in bm25_ranked])
                for item_id, score in bm25_ranked:
                    scores = candidate_scores.setdefault(item_id, {})
                    scores["bm25"] = max(scores.get("bm25", 0.0), score)

        fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
        results = [
            self._to_result(corpus.get_chunk(item_id), score, candidate_scores.get(item_id, {}))
            for item_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)
        ]

        if mmr_lambda is not None:
            return apply_mmr(query, results, top_k=top_k, lambda_mult=mmr_lambda, embedder=embedder)
        return results[:top_k]

    def _ensure_collection(self, collection: str, dimension: int) -> None:
        try:
            info = self.manager.get_collection_info(collection)
        except KeyError:
            self.manager.create_collection(collection, dimension, store_type="local")
            return
        if info["dimension"] != dimension:
            raise ValueError(
                f"Collection '{collection}' has dimension {info['dimension']}, "
                f"but ingest requested {dimension}"
            )

    def _vector_ranked(
        self,
        collection: str,
        query_vector: list[float],
        candidate_k: int,
    ) -> list[tuple[str, float]]:
        results = self.manager.search(
            collection,
            query_vector,
            top_k=candidate_k,
            filter_metadata={"rag": True},
        )
        return [(result.id, 1.0 / (1.0 + result.score)) for result in results]

    def _to_result(
        self,
        chunk: Chunk,
        score: float,
        scores: dict[str, float],
    ) -> RetrievalResult:
        return RetrievalResult(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=score,
            metadata=chunk.metadata,
            scores=scores,
        )
