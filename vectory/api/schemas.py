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


class SearchResultResponse(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any]


class CollectionInfoResponse(BaseModel):
    name: str
    dimension: int
    metric: str
    count: int
    store_type: str = "local"
