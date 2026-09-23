import numpy as np
import pytest

from transformer_core.blocks import FeedForwardNetwork


def test_feed_forward_expands_activates_and_projects_each_token() -> None:
    network = FeedForwardNetwork(
        input_weights=np.array([[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]]),
        input_biases=np.array([0.0, 0.5, -0.25]),
        output_weights=np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.25]]),
        output_biases=np.array([0.1, -0.2]),
    )
    inputs = np.array([[2.0, 1.0], [-1.0, 3.0]])

    outputs = network.forward(inputs)

    hidden = inputs @ network.input_weights + network.input_biases
    expected = np.maximum(hidden, 0.0) @ network.output_weights + network.output_biases
    np.testing.assert_allclose(outputs, expected)


def test_feed_forward_output_width_comes_from_output_biases() -> None:
    network = FeedForwardNetwork(
        input_weights=np.ones((3, 4)),
        input_biases=np.zeros(4),
        output_weights=np.ones((4, 3)),
        output_biases=np.zeros(3),
    )

    assert network.forward(np.ones((2, 3))).shape == (2, 3)


def test_feed_forward_forward_with_cache_matches_forward() -> None:
    network = FeedForwardNetwork(
        input_weights=np.array([[1.0, -1.0], [0.5, 2.0]]),
        input_biases=np.array([0.0, 0.5]),
        output_weights=np.array([[1.0, 0.25], [-0.5, 0.75]]),
        output_biases=np.array([0.1, -0.2]),
    )
    inputs = np.array([[2.0, 1.0]])

    outputs, cache = network.forward_with_cache(inputs)

    np.testing.assert_allclose(outputs, network.forward(inputs))
    assert cache.hidden.shape == (1, 2)
    assert cache.activated.shape == (1, 2)


def test_feed_forward_backward_matches_finite_difference_for_input_weights() -> None:
    network = FeedForwardNetwork(
        input_weights=np.array([[0.5, -0.25], [0.4, 0.8]]),
        input_biases=np.array([0.1, -0.2]),
        output_weights=np.array([[0.3, -0.7], [0.6, 0.2]]),
        output_biases=np.array([0.05, -0.1]),
    )
    inputs = np.array([[1.5, -0.5], [0.25, 2.0]])
    upstream = np.array([[0.4, -0.2], [0.1, 0.7]])

    _outputs, cache = network.forward_with_cache(inputs)
    gradients = network.backward(upstream, cache)

    def loss(candidate_weights: np.ndarray) -> float:
        original = network.input_weights
        network.input_weights = candidate_weights
        try:
            return float(np.sum(network.forward(inputs) * upstream))
        finally:
            network.input_weights = original

    numerical = np.zeros_like(network.input_weights)
    epsilon = 1e-5
    for row in range(network.input_weights.shape[0]):
        for column in range(network.input_weights.shape[1]):
            plus = network.input_weights.copy()
            minus = network.input_weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(gradients.input_weights, numerical, atol=1e-5)
    assert gradients.inputs.shape == inputs.shape
    assert gradients.input_biases.shape == network.input_biases.shape
    assert gradients.output_weights.shape == network.output_weights.shape
    assert gradients.output_biases.shape == network.output_biases.shape


def test_feed_forward_rejects_input_width_mismatch() -> None:
    network = FeedForwardNetwork(
        input_weights=np.ones((3, 4)),
        input_biases=np.zeros(4),
        output_weights=np.ones((4, 3)),
        output_biases=np.zeros(3),
    )

    with pytest.raises(ValueError, match="model width"):
        network.forward(np.ones((2, 2)))


def test_feed_forward_rejects_hidden_width_mismatch() -> None:
    network = FeedForwardNetwork(
        input_weights=np.ones((3, 4)),
        input_biases=np.zeros(4),
        output_weights=np.ones((5, 3)),
        output_biases=np.zeros(3),
    )

    with pytest.raises(ValueError, match="hidden width"):
        network.forward(np.ones((2, 3)))
