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
