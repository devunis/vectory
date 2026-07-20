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


class CollectionInfoResponse(BaseModel):
    name: str
    dimension: int
    metric: str
    count: int
    store_type: str = "local"
