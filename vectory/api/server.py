"""FastAPI REST API server for Vectory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from vectory.api.schemas import (
    CollectionInfoResponse,
    CreateCollectionRequest,
    DeleteRequest,
    InsertRequest,
    ParseRequest,
    ParseResponse,
    RagIngestRequest,
    RagIngestResponse,
    RagSearchRequest,
    RagSearchResultResponse,
    SearchRequest,
    SearchResultResponse,
    UpdateMetadataRequest,
)
from vectory.engine.manager import CollectionManager
from vectory.parsing import parse_document
from vectory.rag import RagPipeline

app = FastAPI(title="Vectory", version="0.1.0", description="Vector DB Platform")

_UI_HTML = (Path(__file__).resolve().parent.parent / "ui" / "index.html").read_text(
    encoding="utf-8"
)

# Global manager instance — configured at startup
manager: CollectionManager | None = None


def get_manager() -> CollectionManager:
    global manager
    if manager is None:
        manager = CollectionManager(data_dir=".vectory_data")
    return manager


def set_manager(m: CollectionManager) -> None:
    global manager
    manager = m


# --- UI ---


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI_HTML


# --- Store info ---


@app.get("/stores")
def list_stores() -> list[str]:
    return get_manager().available_stores()


# --- Document parsing ---


@app.post("/parse")
def parse_source(req: ParseRequest) -> ParseResponse:
    try:
        parsed = parse_document(
            req.source,
            provider=req.provider,
            source_type=req.source_type,
            output_dir=req.output_dir,
            wait=req.wait,
            fetch_markdown=req.fetch_markdown,
            mode=req.mode,
            engine=req.engine,
            language=req.language,
            page_range=req.page_range,
            enable_table=req.enable_table,
            enable_formula=req.enable_formula,
            is_ocr=req.is_ocr,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (RuntimeError, TimeoutError) as e:
        raise HTTPException(502, str(e))
    return ParseResponse(**parsed.to_dict())


# --- RAG ---


@app.post("/rag/ingest")
def rag_ingest(req: RagIngestRequest) -> RagIngestResponse:
    try:
        result = RagPipeline(get_manager()).ingest_text(
            req.collection,
            req.text,
            document_id=req.document_id,
            metadata=req.metadata,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            embedding_dimension=req.embedding_dimension,
            contextual_prefix=req.contextual_prefix,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return RagIngestResponse(**result)


@app.post("/rag/search")
def rag_search(req: RagSearchRequest) -> list[RagSearchResultResponse]:
    try:
        results = RagPipeline(get_manager()).search(
            req.collection,
            req.query,
            strategy=req.strategy,
            top_k=req.top_k,
            candidate_k=req.candidate_k,
            rrf_k=req.rrf_k,
            mmr_lambda=req.mmr_lambda,
            query_expansions=req.query_expansions,
            hypothetical_document=req.hypothetical_document,
        )
    except KeyError:
        raise HTTPException(404, f"Collection '{req.collection}' not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return [RagSearchResultResponse(**result.to_dict()) for result in results]


# --- Collection endpoints ---


@app.post("/collections", status_code=201)
def create_collection(req: CreateCollectionRequest) -> CollectionInfoResponse:
    try:
        info = get_manager().create_collection(
            req.name,
            req.dimension,
            req.metric,
            req.store_type,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return CollectionInfoResponse(**info)


@app.get("/collections")
def list_collections() -> list[CollectionInfoResponse]:
    return [CollectionInfoResponse(**c) for c in get_manager().list_collections()]


@app.get("/collections/{name}")
def get_collection(name: str) -> CollectionInfoResponse:
    try:
        info = get_manager().get_collection_info(name)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, f"Collection '{name}' not found")
    return CollectionInfoResponse(**info)


@app.delete("/collections/{name}", status_code=204, response_model=None)
def delete_collection(name: str):
    try:
        get_manager().delete_collection(name)
    except KeyError:
        raise HTTPException(404, f"Collection '{name}' not found")


# --- Vector endpoints ---


@app.post("/collections/{name}/vectors")
def insert_vectors(name: str, req: InsertRequest) -> dict:
    try:
        ids = get_manager().insert(name, req.vectors, req.ids, req.metadata)
    except KeyError:
        raise HTTPException(404, f"Collection '{name}' not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"inserted_ids": ids}


@app.post("/collections/{name}/search")
def search_vectors(name: str, req: SearchRequest) -> list[SearchResultResponse]:
    try:
        results = get_manager().search(name, req.query, req.top_k, req.filter_metadata)
    except KeyError:
        raise HTTPException(404, f"Collection '{name}' not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return [SearchResultResponse(id=r.id, score=r.score, metadata=r.metadata) for r in results]


@app.post("/collections/{name}/get")
def get_vectors(name: str, req: DeleteRequest) -> list[dict]:
    try:
        return get_manager().get(name, req.ids)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/collections/{name}/delete")
def delete_vectors(name: str, req: DeleteRequest) -> dict:
    try:
        deleted = get_manager().delete_vectors(name, req.ids)
    except KeyError:
        raise HTTPException(404, f"Collection '{name}' not found")
    return {"deleted": deleted}


@app.put("/collections/{name}/metadata")
def update_metadata(name: str, req: UpdateMetadataRequest) -> dict:
    try:
        get_manager().update_metadata(name, req.id, req.metadata)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"status": "updated"}
