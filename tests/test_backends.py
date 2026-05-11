"""Tests for built-in vector store backends."""

from vectory.backends.local import LocalStore
from vectory.engine.manager import STORE_REGISTRY


def test_store_registry_always_includes_local():
    assert STORE_REGISTRY["local"] is LocalStore


def test_local_store_contract_round_trip(tmp_path):
    store = LocalStore(str(tmp_path))

    info = store.create_collection("docs", dimension=2, metric="cosine")
    assert info == {
        "name": "docs",
        "dimension": 2,
        "metric": "cosine",
        "count": 0,
        "store_type": "local",
    }

    ids = store.insert(
        "docs",
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        ids=["a", "b"],
        metadata=[{"tag": "x"}, {"tag": "y"}],
    )
    assert ids == ["a", "b"]
    assert store.collection_info("docs")["count"] == 2

    results = store.search("docs", [1.0, 0.0], top_k=2, filter_metadata={"tag": "x"})
    assert [r.id for r in results] == ["a"]

    vectors = store.get("docs", ["a"])
    assert vectors == [{"id": "a", "vector": [1.0, 0.0], "metadata": {"tag": "x"}}]

    store.update_metadata("docs", "a", {"tag": "z", "source": "test"})
    assert store.get("docs", ["a"])[0]["metadata"] == {"tag": "z", "source": "test"}

    assert store.delete("docs", ["a"]) == 1
    assert store.collection_info("docs")["count"] == 1

    store.drop_collection("docs")
    assert store.list_collections() == []


def test_local_store_ignores_non_collection_json_files(tmp_path):
    (tmp_path / "payload.json").write_text('{"vectors": [[1, 0]]}', encoding="utf-8")

    store = LocalStore(str(tmp_path))

    assert store.list_collections() == []
