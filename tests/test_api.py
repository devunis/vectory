"""Tests for the REST API."""

import pytest
from fastapi.testclient import TestClient
from vectory.api.server import app, set_manager
from vectory.engine.manager import CollectionManager


@pytest.fixture(autouse=True)
def _setup_manager():
    mgr = CollectionManager.in_memory()
    set_manager(mgr)
    yield


client = TestClient(app)


class TestCollectionAPI:
    def test_ui_index(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Vectory" in resp.text
        assert "RAG Ingest" in resp.text
        assert "RAG Search" in resp.text
        assert "Document Parsing" in resp.text
        assert "Ingest parsed text" in resp.text
        assert "/rag/ingest" in resp.text
        assert "/rag/search" in resp.text
        assert "/parse" in resp.text

    def test_list_stores(self):
        resp = client.get("/stores")
        assert resp.status_code == 200
        assert "local" in resp.json()

    def test_create_collection(self):
        resp = client.post("/collections", json={"name": "test", "dimension": 3})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test"
        assert data["dimension"] == 3

    def test_create_duplicate(self):
        client.post("/collections", json={"name": "test", "dimension": 3})
        resp = client.post("/collections", json={"name": "test", "dimension": 3})
        assert resp.status_code == 409

    def test_list_collections(self):
        client.post("/collections", json={"name": "a", "dimension": 2})
        client.post("/collections", json={"name": "b", "dimension": 4})
        resp = client.get("/collections")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_collection(self):
        client.post("/collections", json={"name": "test", "dimension": 3})
        resp = client.get("/collections/test")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test"

    def test_get_missing_collection(self):
        resp = client.get("/collections/missing")
        assert resp.status_code == 404

    def test_delete_collection(self):
        client.post("/collections", json={"name": "test", "dimension": 3})
        resp = client.delete("/collections/test")
        assert resp.status_code == 204


class TestVectorAPI:
    def _create_collection(self):
        client.post("/collections", json={"name": "vecs", "dimension": 3})

    def test_insert_and_search(self):
        self._create_collection()
        resp = client.post(
            "/collections/vecs/vectors",
            json={
                "vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "ids": ["a", "b", "c"],
                "metadata": [{"t": "x"}, {"t": "y"}, {"t": "z"}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["inserted_ids"] == ["a", "b", "c"]

        resp = client.post(
            "/collections/vecs/search",
            json={
                "query": [1, 0, 0],
                "top_k": 2,
            },
        )
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2
        assert results[0]["id"] == "a"

    def test_get_vectors(self):
        self._create_collection()
        client.post(
            "/collections/vecs/vectors",
            json={
                "vectors": [[1, 2, 3]],
                "ids": ["v1"],
            },
        )
        resp = client.post("/collections/vecs/get", json={"ids": ["v1"]})
        assert resp.status_code == 200
        assert resp.json()[0]["id"] == "v1"

    def test_delete_vectors(self):
        self._create_collection()
        client.post(
            "/collections/vecs/vectors",
            json={
                "vectors": [[1, 0, 0], [0, 1, 0]],
                "ids": ["a", "b"],
            },
        )
        resp = client.post("/collections/vecs/delete", json={"ids": ["a"]})
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1

    def test_update_metadata(self):
        self._create_collection()
        client.post(
            "/collections/vecs/vectors",
            json={
                "vectors": [[1, 0, 0]],
                "ids": ["a"],
                "metadata": [{"key": "old"}],
            },
        )
        resp = client.put(
            "/collections/vecs/metadata",
            json={
                "id": "a",
                "metadata": {"key": "new"},
            },
        )
        assert resp.status_code == 200
