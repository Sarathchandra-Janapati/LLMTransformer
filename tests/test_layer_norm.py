import numpy as np
import pytest

from transformer_core.blocks import LayerNorm


def test_layer_norm_normalizes_each_token_row() -> None:
    layer_norm = LayerNorm(gamma=np.ones(3), beta=np.zeros(3), epsilon=1e-8)
    inputs = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

    outputs = layer_norm.forward(inputs)

    np.testing.assert_allclose(np.mean(outputs, axis=1), np.zeros(2), atol=1e-7)
    np.testing.assert_allclose(np.var(outputs, axis=1), np.ones(2), atol=1e-7)


def test_layer_norm_applies_gamma_and_beta_after_normalization() -> None:
    layer_norm = LayerNorm(
        gamma=np.array([2.0, 0.5]),
        beta=np.array([1.0, -1.0]),
        epsilon=1e-8,
    )

    outputs = layer_norm.forward(np.array([[1.0, 3.0]]))

    np.testing.assert_allclose(outputs, np.array([[-1.0, -0.5]]), atol=1e-7)


def test_layer_norm_handles_constant_rows_with_epsilon() -> None:
    layer_norm = LayerNorm(gamma=np.ones(3), beta=np.zeros(3))

    outputs = layer_norm.forward(np.array([[4.0, 4.0, 4.0]]))

    np.testing.assert_allclose(outputs, np.zeros((1, 3)))


def test_layer_norm_forward_with_cache_matches_forward() -> None:
    layer_norm = LayerNorm(gamma=np.ones(3), beta=np.zeros(3), epsilon=1e-8)
    inputs = np.array([[1.0, 2.0, 3.0]])

    outputs, cache = layer_norm.forward_with_cache(inputs)

    np.testing.assert_allclose(outputs, layer_norm.forward(inputs))
    assert cache.inputs.shape == inputs.shape
    assert cache.normalized.shape == inputs.shape


def test_layer_norm_backward_matches_finite_difference_for_inputs() -> None:
    layer_norm = LayerNorm(
        gamma=np.array([1.2, 0.7, -0.5]),
        beta=np.array([0.1, -0.2, 0.3]),
        epsilon=1e-6,
    )
    inputs = np.array([[1.0, 2.0, 4.0], [0.5, -1.0, 3.0]])
    upstream = np.array([[0.2, -0.3, 0.7], [1.0, -0.4, 0.1]])

    _outputs, cache = layer_norm.forward_with_cache(inputs)
    gradients = layer_norm.backward(upstream, cache)

    def loss(candidate_inputs: np.ndarray) -> float:
        return float(np.sum(layer_norm.forward(candidate_inputs) * upstream))

    numerical = np.zeros_like(inputs)
    epsilon = 1e-5
    for row in range(inputs.shape[0]):
        for column in range(inputs.shape[1]):
            plus = inputs.copy()
            minus = inputs.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(gradients.inputs, numerical, atol=1e-5)
    np.testing.assert_allclose(gradients.gamma, np.sum(upstream * cache.normalized, axis=0))
    np.testing.assert_allclose(gradients.beta, np.sum(upstream, axis=0))


def test_layer_norm_rejects_parameter_width_mismatch() -> None:
    layer_norm = LayerNorm(gamma=np.ones(2), beta=np.zeros(2))

    with pytest.raises(ValueError, match="activation width"):
        layer_norm.forward(np.ones((2, 3)))


def test_layer_norm_rejects_non_positive_epsilon() -> None:
    layer_norm = LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon"):
        layer_norm.forward(np.ones((1, 2)))
