import numpy as np
import pytest

from transformer_core.foundations import dot_product, matrix_multiply


def test_dot_product_matches_hand_computed_result() -> None:
    left = np.array([1.0, -2.0, 3.0])
    right = np.array([4.0, 5.0, -6.0])

    assert dot_product(left, right) == pytest.approx(-24.0)


def test_dot_product_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        dot_product(np.array([1.0]), np.array([1.0, 2.0]))


def test_matrix_multiply_matches_numpy_reference() -> None:
    left = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

    np.testing.assert_allclose(matrix_multiply(left, right), left @ right)


def test_matrix_multiply_rejects_unaligned_shapes() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        matrix_multiply(np.ones((2, 3)), np.ones((2, 1)))

