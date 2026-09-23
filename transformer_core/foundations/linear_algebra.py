"""Small linear algebra primitives implemented without NumPy shortcuts."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def dot_product(left: FloatArray, right: FloatArray) -> float:
    """Return the dot product of two one-dimensional vectors."""
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("dot_product expects one-dimensional vectors")
    if left.shape != right.shape:
        raise ValueError("dot_product expects vectors with the same shape")

    total = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        total += float(left_value) * float(right_value)
    return total


def matrix_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return the matrix product of two two-dimensional arrays."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("matrix_multiply expects two-dimensional matrices")
    if left.shape[1] != right.shape[0]:
        raise ValueError("matrix shapes are not aligned for multiplication")

    result = np.zeros((left.shape[0], right.shape[1]), dtype=float)
    for row_index in range(left.shape[0]):
        for column_index in range(right.shape[1]):
            result[row_index, column_index] = dot_product(
                left[row_index, :],
                right[:, column_index],
            )
    return result

