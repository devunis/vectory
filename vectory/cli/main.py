"""CLI interface for Vectory."""

from __future__ import annotations

import json

import click

from vectory.engine.manager import CollectionManager

DEFAULT_DATA_DIR = ".vectory_data"


@click.group()
@click.option("--data-dir", default=DEFAULT_DATA_DIR, help="Storage directory")
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Vectory - Vector DB Platform CLI."""
    ctx.ensure_object(dict)
    ctx.obj["manager"] = CollectionManager(data_dir=data_dir)


@cli.command()
@click.argument("name")
@click.option("--dimension", "-d", required=True, type=int, help="Vector dimension")
@click.option(
    "--metric",
    "-m",
    default="cosine",
    type=click.Choice(["cosine", "euclidean", "dot_product"]),
    help="Distance metric",
)
@click.option(
    "--store",
    "-s",
    default="local",
    help="Vector store backend (local, chroma, faiss, qdrant, milvus)",
)
@click.pass_context
def create(ctx: click.Context, name: str, dimension: int, metric: str, store: str) -> None:
    """Create a new vector collection."""
    mgr: CollectionManager = ctx.obj["manager"]
    try:
        info = mgr.create_collection(name, dimension, metric, store)
        click.echo(
            f"Created collection '{info['name']}' (dim={info['dimension']}, metric={info['metric']}, store={info['store_type']})"
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command("list")
@click.pass_context
def list_collections(ctx: click.Context) -> None:
    """List all collections."""
    mgr: CollectionManager = ctx.obj["manager"]
    collections = mgr.list_collections()
    if not collections:
        click.echo("No collections found.")
        return
    for c in collections:
        click.echo(
            f"  {c['name']}  dim={c['dimension']}  metric={c['metric']}  count={c['count']}  store={c['store_type']}"
        )


@cli.command()
@click.argument("name")
@click.pass_context
def info(ctx: click.Context, name: str) -> None:
    """Show collection info."""
    mgr: CollectionManager = ctx.obj["manager"]
    try:
        info_data = mgr.get_collection_info(name)
        click.echo(json.dumps(info_data, indent=2))
    except (KeyError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this collection?")
@click.pass_context
def delete(ctx: click.Context, name: str) -> None:
    """Delete a collection."""
    mgr: CollectionManager = ctx.obj["manager"]
    mgr.delete_collection(name)
    click.echo(f"Deleted collection '{name}'.")


@cli.command()
@click.argument("name")
@click.argument("vectors_json", type=click.Path(exists=True))
@click.pass_context
def insert(ctx: click.Context, name: str, vectors_json: str) -> None:
    """Insert vectors from a JSON file into a collection.

    The JSON file should contain: {"vectors": [[...], ...], "ids": [...], "metadata": [...]}
    """
    mgr: CollectionManager = ctx.obj["manager"]

    with open(vectors_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        ids = mgr.insert(
            name,
            data["vectors"],
            data.get("ids"),
            data.get("metadata"),
        )
        click.echo(f"Inserted {len(ids)} vectors.")
    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("name")
@click.argument("query", type=str)
@click.option("--top-k", "-k", default=5, type=int, help="Number of results")
@click.pass_context
def search(ctx: click.Context, name: str, query: str, top_k: int) -> None:
    """Search for similar vectors. QUERY is a JSON array of floats."""
    mgr: CollectionManager = ctx.obj["manager"]
    query_vec = json.loads(query)
    try:
        results = mgr.search(name, query_vec, top_k)
    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    for r in results:
        click.echo(f"  {r.id}  score={r.score:.6f}  metadata={json.dumps(r.metadata)}")


@cli.command()
def stores() -> None:
    """List available vector store backends."""
    from vectory.engine.manager import STORE_REGISTRY

    for name in sorted(STORE_REGISTRY.keys()):
        click.echo(f"  {name}")


@cli.command("parse")
@click.argument("source")
@click.option(
    "--provider",
    default="paddleocr",
    type=click.Choice(["paddleocr", "mineru"]),
    help="Parsing provider",
)
@click.option(
    "--source-type",
    default="path",
    type=click.Choice(["path", "url"]),
    help="Source location type",
)
@click.option(
    "--mode",
    default="ocr",
    type=click.Choice(["ocr", "structure"]),
    help="PaddleOCR parsing mode",
)
@click.option(
    "--engine",
    default="paddle",
    type=click.Choice(["paddle", "transformers"]),
    help="PaddleOCR inference engine",
)
@click.option("--output-dir", type=click.Path(), help="Directory for provider output files")
@click.option("--language", default="ch", help="MinerU language code")
@click.option("--page-range", help="MinerU page range, for example 1-3")
@click.option("--disable-table", is_flag=True, help="Disable MinerU table recognition")
@click.option("--disable-formula", is_flag=True, help="Disable MinerU formula recognition")
@click.option("--ocr/--no-ocr", "is_ocr", default=False, help="Enable MinerU OCR mode")
@click.option("--no-wait", is_flag=True, help="Return after MinerU task submission")
@click.option("--fetch-markdown", is_flag=True, help="Fetch MinerU markdown when available")
def parse(
    source: str,
    provider: str,
    source_type: str,
    mode: str,
    engine: str,
    output_dir: str | None,
    language: str,
    page_range: str | None,
    disable_table: bool,
    disable_formula: bool,
    is_ocr: bool,
    no_wait: bool,
    fetch_markdown: bool,
) -> None:
    """Parse a document or image with PaddleOCR or MinerU."""
    from vectory.parsing import parse_document

    try:
        result = parse_document(
            source,
            provider=provider,
            source_type=source_type,
            output_dir=output_dir,
            wait=not no_wait,
            fetch_markdown=fetch_markdown,
            mode=mode,
            engine=engine,
            language=language,
            page_range=page_range,
            enable_table=not disable_table,
            enable_formula=not disable_formula,
            is_ocr=is_ocr,
        )
    except (RuntimeError, TimeoutError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@cli.group()
def rag() -> None:
    """RAG ingestion and retrieval commands."""


@rag.command("ingest")
@click.argument("collection")
@click.argument("text_file", type=click.Path(exists=True))
@click.option("--document-id", help="Stable document ID")
@click.option("--metadata-json", help="JSON metadata object for the document")
@click.option("--chunk-size", default=200, type=int, help="Chunk size in words")
@click.option("--chunk-overlap", default=40, type=int, help="Chunk overlap in words")
@click.option("--embedding-dimension", default=384, type=int, help="Hash embedding dimension")
@click.option("--contextual-prefix", help="Context prefix prepended to every chunk")
@click.pass_context
def rag_ingest(
    ctx: click.Context,
    collection: str,
    text_file: str,
    document_id: str | None,
    metadata_json: str | None,
    chunk_size: int,
    chunk_overlap: int,
    embedding_dimension: int,
    contextual_prefix: str | None,
) -> None:
    """Ingest a text file into a RAG collection."""
    from vectory.rag import RagPipeline

    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    metadata = json.loads(metadata_json) if metadata_json else None
    try:
        result = RagPipeline(ctx.obj["manager"]).ingest_text(
            collection,
            text,
            document_id=document_id,
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dimension=embedding_dimension,
            contextual_prefix=contextual_prefix,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


@rag.command("search")
@click.argument("collection")
@click.argument("query")
@click.option(
    "--strategy",
    default="hybrid",
    type=click.Choice(["vector", "bm25", "hybrid"]),
    help="Retrieval strategy",
)
@click.option("--top-k", "-k", default=5, type=int, help="Number of final results")
@click.option("--candidate-k", default=30, type=int, help="Number of candidates per retriever")
@click.option("--rrf-k", default=60, type=int, help="RRF rank constant")
@click.option("--mmr-lambda", type=float, help="Apply MMR with this relevance/diversity weight")
@click.option("--expand", "query_expansions", multiple=True, help="Additional rewritten query")
@click.option(
    "--hyde", "hypothetical_document", help="Hypothetical document text for HyDE-style search"
)
@click.pass_context
def rag_search(
    ctx: click.Context,
    collection: str,
    query: str,
    strategy: str,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    mmr_lambda: float | None,
    query_expansions: tuple[str, ...],
    hypothetical_document: str | None,
) -> None:
    """Search a RAG collection."""
    from vectory.rag import RagPipeline

    try:
        results = RagPipeline(ctx.obj["manager"]).search(
            collection,
            query,
            strategy=strategy,
            top_k=top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            mmr_lambda=mmr_lambda,
            query_expansions=list(query_expansions),
            hypothetical_document=hypothetical_document,
        )
    except (KeyError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the REST API server."""
    import uvicorn

    from vectory.api.server import set_manager

    set_manager(ctx.obj["manager"])
    click.echo(f"Starting Vectory API server at {host}:{port}")
    uvicorn.run("vectory.api.server:app", host=host, port=port)


if __name__ == "__main__":
    cli()
