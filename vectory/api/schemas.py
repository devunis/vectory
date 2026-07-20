"""Pydantic schemas for the REST API."""

from typing import Any

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int = Field(gt=0)
    metric: str = "cosine"
    store_type: str = "local"


class InsertRequest(BaseModel):
    vectors: list[list[float]]
    ids: list[str] | None = None
    metadata: list[dict[str, Any]] | None = None


class SearchRequest(BaseModel):
    query: list[float]
    top_k: int = Field(default=10, gt=0, le=1000)
    filter_metadata: dict[str, Any] | None = None


class DeleteRequest(BaseModel):
    ids: list[str]


class UpdateMetadataRequest(BaseModel):
    id: str
    metadata: dict[str, Any]


class ParseRequest(BaseModel):
    provider: str = "paddleocr"
    source: str
    source_type: str = "path"
    mode: str = "ocr"
    engine: str = "paddle"
    output_dir: str | None = None
    wait: bool = True
    fetch_markdown: bool = False
    language: str = "ch"
    page_range: str | None = None
    enable_table: bool = True
    enable_formula: bool = True
    is_ocr: bool = False


class RagIngestRequest(BaseModel):
    collection: str
    text: str
    document_id: str | None = None
    metadata: dict[str, Any] | None = None
    chunk_size: int = Field(default=200, gt=0)
    chunk_overlap: int = Field(default=40, ge=0)
    embedding_dimension: int = Field(default=384, gt=0)
    contextual_prefix: str | None = None


class RagSearchRequest(BaseModel):
    collection: str
    query: str
    strategy: str = "hybrid"
    top_k: int = Field(default=5, gt=0, le=100)
    candidate_k: int = Field(default=30, gt=0, le=1000)
    rrf_k: int = Field(default=60, gt=0)
    mmr_lambda: float | None = Field(default=None, ge=0, le=1)
    query_expansions: list[str] | None = None
    hypothetical_document: str | None = None


class SearchResultResponse(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any]


class ParseResponse(BaseModel):
    provider: str
    source: str
    text: str = ""
    markdown: str | None = None
    metadata: dict[str, Any]
    raw: Any


class RagIngestResponse(BaseModel):
    collection: str
    document_id: str
    chunk_count: int
    chunk_ids: list[str]
    embedding_dimension: int


class RagSearchResultResponse(BaseModel):
    id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    scores: dict[str, float]


class CollectionInfoResponse(BaseModel):
    name: str
    dimension: int
    metric: str
    count: int
    store_type: str = "local"
