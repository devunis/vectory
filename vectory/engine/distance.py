"""Distance metrics for vector similarity search."""

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


def cosine_distance(a: NDArray, b: NDArray) -> NDArray:
    """Compute cosine distance between vector a and matrix b."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    # Avoid division by zero
    denom = norm_a * norm_b
    denom = np.where(denom == 0, 1e-10, denom)
    similarity = np.dot(b, a) / denom
    return 1.0 - similarity


def euclidean_distance(a: NDArray, b: NDArray) -> NDArray:
    """Compute euclidean distance between vector a and matrix b."""
    return np.linalg.norm(b - a, axis=1)


def dot_product_distance(a: NDArray, b: NDArray) -> NDArray:
    """Compute negative dot product (lower = more similar)."""
    return -np.dot(b, a)


DISTANCE_FUNCTIONS = {
    DistanceMetric.COSINE: cosine_distance,
    DistanceMetric.EUCLIDEAN: euclidean_distance,
    DistanceMetric.DOT_PRODUCT: dot_product_distance,
}
