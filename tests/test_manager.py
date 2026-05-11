"""Tests for CollectionManager."""

import pytest
from vectory.engine.manager import CollectionManager


class TestCollectionManager:
    def test_create_and_list(self):
        mgr = CollectionManager.in_memory()
        mgr.create_collection("col1", dimension=4)
        mgr.create_collection("col2", dimension=8, metric="euclidean")
        collections = mgr.list_collections()
        assert len(collections) == 2
        names = {c["name"] for c in collections}
        assert names == {"col1", "col2"}

    def test_duplicate_name(self):
        mgr = CollectionManager.in_memory()
        mgr.create_collection("test", dimension=4)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_collection("test", dimension=4)

    def test_delete_collection(self):
        mgr = CollectionManager.in_memory()
        mgr.create_collection("test", dimension=4)
        mgr.delete_collection("test")
        assert mgr.list_collections() == []

    def test_save_and_load(self, tmp_path):
        mgr = CollectionManager(data_dir=str(tmp_path))
        mgr.create_collection("test", dimension=3, store_type="local")
        mgr.insert("test", [[1.0, 2.0, 3.0]], ids=["v1"])

        # New manager should load from disk
        mgr2 = CollectionManager(data_dir=str(tmp_path))
        info = mgr2.get_collection_info("test")
        assert info["count"] == 1
        vecs = mgr2.get("test", ["v1"])
        assert vecs[0]["vector"] == [1.0, 2.0, 3.0]

    def test_list_includes_persisted(self, tmp_path):
        mgr = CollectionManager(data_dir=str(tmp_path))
        mgr.create_collection("persisted", dimension=2, store_type="local")

        mgr2 = CollectionManager(data_dir=str(tmp_path))
        collections = mgr2.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "persisted"

    def test_store_type_in_info(self):
        mgr = CollectionManager.in_memory()
        info = mgr.create_collection("test", dimension=4, store_type="local")
        assert info["store_type"] == "local"

    def test_insert_search_via_manager(self):
        mgr = CollectionManager.in_memory()
        mgr.create_collection("test", dimension=3)
        mgr.insert("test", [[1, 0, 0], [0, 1, 0]], ids=["a", "b"])
        results = mgr.search("test", [1, 0, 0], top_k=1)
        assert len(results) == 1
        assert results[0].id == "a"
