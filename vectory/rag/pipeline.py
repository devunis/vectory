"""Composable RAG ingestion and retrieval pipeline."""

from __future__ import annotations

import uuid
from typing import Any

from vectory.engine.manager import CollectionManager
from vectory.rag.adaptive import HeuristicQueryRouter
from vectory.rag.bm25 import BM25Index
from vectory.rag.chunking import chunk_document
from vectory.rag.embeddings import HashingEmbedder
from vectory.rag.evaluators import HeuristicRetrievalEvaluator
from vectory.rag.fusion import apply_mmr, reciprocal_rank_fusion
from vectory.rag.graph import EntityGraph
from vectory.rag.models import Chunk, Document, RetrievalResult
from vectory.rag.raptor import build_raptor_summaries
from vectory.rag.rerankers import LexicalReranker
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
        enable_raptor: bool = False,
        raptor_group_size: int = 4,
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
        summary_chunks = (
            build_raptor_summaries(chunks, group_size=raptor_group_size) if enable_raptor else []
        )
        all_chunks = [*chunks, *summary_chunks]

        self._ensure_collection(collection, embedding_dimension)
        embedder = HashingEmbedder(embedding_dimension)
        vectors = embedder.embed_many([chunk.text for chunk in all_chunks])
        vector_metadata = [
            {
                **chunk.metadata,
                "rag": True,
                "document_id": chunk.document_id,
                "chunk_id": chunk.id,
                "text": chunk.text,
            }
            for chunk in all_chunks
        ]
        ids = self.manager.insert(
            collection,
            vectors,
            ids=[chunk.id for chunk in all_chunks],
            metadata=vector_metadata,
        )
        RagCorpus(self.data_dir, collection).add_document(document, all_chunks)

        return {
            "collection": collection,
            "document_id": document.id,
            "chunk_count": len(chunks),
            "summary_count": len(summary_chunks),
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
        reranker: str | None = None,
        corrective_threshold: float = 0.35,
        graph_expand_k: int = 5,
    ) -> list[RetrievalResult]:
        """Search a RAG collection with baseline or advanced retrieval."""
        valid_strategies = {
            "vector",
            "bm25",
            "hybrid",
            "adaptive",
            "corrective",
            "raptor",
            "graph",
        }
        if strategy not in valid_strategies:
            raise ValueError(
                "strategy must be one of: adaptive, bm25, corrective, graph, hybrid, raptor, vector"
            )
        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be greater than 0")

        if strategy == "adaptive":
            route = HeuristicQueryRouter().route(query)
            return self.search(
                collection,
                query,
                strategy=route.strategy,
                top_k=top_k,
                candidate_k=max(candidate_k, route.candidate_k),
                rrf_k=rrf_k,
                mmr_lambda=mmr_lambda if mmr_lambda is not None else route.mmr_lambda,
                query_expansions=query_expansions,
                hypothetical_document=hypothetical_document,
                reranker=reranker or route.reranker,
                corrective_threshold=corrective_threshold,
                graph_expand_k=graph_expand_k,
            )

        if strategy == "corrective":
            results = self.search(
                collection,
                query,
                strategy="hybrid",
                top_k=max(top_k, candidate_k),
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                mmr_lambda=None,
                query_expansions=query_expansions,
                hypothetical_document=hypothetical_document,
            )
            evaluator = HeuristicRetrievalEvaluator(min_confidence=corrective_threshold)
            evaluation = evaluator.evaluate(query, results)
            if evaluation.needs_correction:
                correction_queries = query_expansions or []
                correction_queries = [*correction_queries, *query.split()]
                corrections = self.search(
                    collection,
                    query,
                    strategy="bm25",
                    top_k=candidate_k,
                    candidate_k=candidate_k,
                    rrf_k=rrf_k,
                    query_expansions=correction_queries,
                )
                results = self._merge_results(results, corrections)
            for result in results:
                result.scores["corrective_confidence"] = evaluation.confidence
                result.scores["corrective_coverage"] = evaluation.coverage
            return self._finalize(query, results, top_k, mmr_lambda, reranker)

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
        retrieval_strategy = "hybrid" if strategy in {"graph", "raptor"} else strategy

        if retrieval_strategy in {"vector", "hybrid"}:
            vector_ranked = self._vector_ranked(
                collection,
                embedder.embed(vector_query),
                candidate_k,
            )
            ranked_lists.append([item_id for item_id, _ in vector_ranked])
            for item_id, score in vector_ranked:
                candidate_scores.setdefault(item_id, {})["vector"] = score

        if retrieval_strategy in {"bm25", "hybrid"}:
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

        if strategy == "raptor":
            for result in results:
                if result.metadata.get("summary_type") == "raptor":
                    result.score += 0.1
                    result.scores["raptor_boost"] = 0.1
            results = sorted(results, key=lambda item: item.score, reverse=True)

        if strategy == "graph":
            graph = EntityGraph(chunks)
            expanded = graph.expand(query, results[:top_k], limit=graph_expand_k)
            expanded_results = [
                self._to_result(chunk, 0.01, {"graph_expansion": 1.0}) for chunk in expanded
            ]
            results = self._merge_results(results, expanded_results)

        return self._finalize(query, results, top_k, mmr_lambda, reranker, embedder=embedder)

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

    def _merge_results(
        self,
        primary: list[RetrievalResult],
        secondary: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        merged: dict[str, RetrievalResult] = {}
        for result in [*primary, *secondary]:
            if result.id not in merged:
                merged[result.id] = result
                continue
            existing = merged[result.id]
            existing.score = max(existing.score, result.score)
            existing.scores.update(result.scores)
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def _finalize(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
        mmr_lambda: float | None,
        reranker: str | None,
        *,
        embedder: HashingEmbedder | None = None,
    ) -> list[RetrievalResult]:
        if mmr_lambda is not None:
            results = apply_mmr(
                query, results, top_k=top_k, lambda_mult=mmr_lambda, embedder=embedder
            )
        if reranker == "lexical":
            return LexicalReranker().rerank(query, results, top_k=top_k)
        if reranker not in {None, "none"}:
            raise ValueError("reranker must be one of: lexical, none")
        return results[:top_k]
