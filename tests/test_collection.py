"""Tests for the core Collection and distance functions."""

import numpy as np
import pytest
from vectory.engine.collection import Collection
from vectory.engine.distance import (
    DistanceMetric,
    cosine_distance,
    dot_product_distance,
    euclidean_distance,
)


class TestDistanceFunctions:
    def test_cosine_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[1.0, 0.0, 0.0]])
        result = cosine_distance(a, b)
        assert abs(result[0]) < 1e-6

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([[0.0, 1.0]])
        result = cosine_distance(a, b)
        assert abs(result[0] - 1.0) < 1e-6

    def test_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([[3.0, 4.0]])
        result = euclidean_distance(a, b)
        assert abs(result[0] - 5.0) < 1e-6

    def test_dot_product(self):
        a = np.array([1.0, 2.0])
        b = np.array([[3.0, 4.0]])
        result = dot_product_distance(a, b)
        assert abs(result[0] - (-11.0)) < 1e-6


class TestCollection:
    def test_create_and_insert(self):
        col = Collection("test", dimension=3)
        ids = col.insert([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert col.count == 2
        assert len(ids) == 2

    def test_insert_with_ids_and_metadata(self):
        col = Collection("test", dimension=2)
        ids = col.insert(
            [[1.0, 0.0], [0.0, 1.0]],
            ids=["a", "b"],
            metadata=[{"label": "x"}, {"label": "y"}],
        )
        assert ids == ["a", "b"]
        results = col.get(["a"])
        assert results[0]["metadata"]["label"] == "x"

    def test_insert_wrong_dimension(self):
        col = Collection("test", dimension=3)
        with pytest.raises(ValueError, match="Expected dimension 3"):
            col.insert([[1.0, 2.0]])

    def test_duplicate_id(self):
        col = Collection("test", dimension=2)
        col.insert([[1.0, 0.0]], ids=["a"])
        with pytest.raises(ValueError, match="Duplicate id"):
            col.insert([[0.0, 1.0]], ids=["a"])

    def test_get_missing_id(self):
        col = Collection("test", dimension=2)
        with pytest.raises(KeyError):
            col.get(["nonexistent"])

    def test_delete(self):
        col = Collection("test", dimension=2)
        col.insert([[1.0, 0.0], [0.0, 1.0]], ids=["a", "b"])
        deleted = col.delete(["a"])
        assert deleted == 1
        assert col.count == 1

    def test_update_metadata(self):
        col = Collection("test", dimension=2)
        col.insert([[1.0, 0.0]], ids=["a"], metadata=[{"key": "val"}])
        col.update_metadata("a", {"key": "new", "extra": 1})
        result = col.get(["a"])
        assert result[0]["metadata"] == {"key": "new", "extra": 1}

    def test_search_cosine(self):
        col = Collection("test", dimension=3, metric=DistanceMetric.COSINE)
        col.insert(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
            ids=["a", "b", "c"],
        )
        results = col.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "a"  # exact match should be first

    def test_search_with_filter(self):
        col = Collection("test", dimension=2)
        col.insert(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            ids=["a", "b", "c"],
            metadata=[{"cat": "x"}, {"cat": "y"}, {"cat": "x"}],
        )
        results = col.search([1.0, 0.0], top_k=10, filter_metadata={"cat": "x"})
        assert all(r.metadata["cat"] == "x" for r in results)
        assert len(results) == 2

    def test_search_empty_collection(self):
        col = Collection("test", dimension=2)
        results = col.search([1.0, 0.0])
        assert results == []

    def test_serialization_roundtrip(self):
        col = Collection("test", dimension=2, metric=DistanceMetric.EUCLIDEAN)
        col.insert([[1.0, 2.0], [3.0, 4.0]], ids=["a", "b"], metadata=[{"k": 1}, {"k": 2}])
        data = col.to_dict()
        col2 = Collection.from_dict(data)
        assert col2.name == "test"
        assert col2.dimension == 2
        assert col2.metric == DistanceMetric.EUCLIDEAN
        assert col2.count == 2
        results = col2.get(["a", "b"])
        assert results[0]["vector"] == [1.0, 2.0]
