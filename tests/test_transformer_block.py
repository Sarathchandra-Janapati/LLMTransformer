import numpy as np
import pytest

from transformer_core.attention import MultiHeadSelfAttention, causal_attention_mask
from transformer_core.blocks import FeedForwardNetwork, LayerNorm, TransformerBlock


def build_transformer_block(output_weights: np.ndarray | None = None) -> TransformerBlock:
    return TransformerBlock(
        attention=MultiHeadSelfAttention(
            query_weights=[np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            key_weights=[np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            value_weights=[np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            output_weights=(
                output_weights
                if output_weights is not None
                else np.array([[0.5, 0.0], [0.0, 0.5]])
            ),
        ),
        attention_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
        feed_forward=FeedForwardNetwork(
            input_weights=np.array([[1.0, -1.0], [0.5, 1.0]]),
            input_biases=np.array([0.0, 0.0]),
            output_weights=np.array([[0.25, 0.0], [0.0, 0.25]]),
            output_biases=np.array([0.0, 0.0]),
        ),
        feed_forward_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
    )


def test_transformer_block_returns_intermediate_tensors_and_output_shape() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0], [1.0, 0.0]])

    result = block.forward(inputs)

    assert result.normalized_attention_input.shape == inputs.shape
    assert len(result.attention.heads) == 2
    assert result.attention.output.shape == inputs.shape
    assert result.attention_residual.shape == inputs.shape
    assert result.normalized_feed_forward_input.shape == inputs.shape
    assert result.feed_forward_output.shape == inputs.shape
    assert result.output.shape == inputs.shape


def test_transformer_block_output_matches_residual_formula() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])

    result = block.forward(inputs)

    np.testing.assert_allclose(result.attention_residual, inputs + result.attention.output)
    np.testing.assert_allclose(result.output, result.attention_residual + result.feed_forward_output)


def test_transformer_block_forward_with_cache_matches_forward() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])

    result, cache = block.forward_with_cache(inputs)

    np.testing.assert_allclose(result.output, block.forward(inputs).output)
    assert cache.inputs.shape == inputs.shape
    assert cache.feed_forward.inputs.shape == inputs.shape


def test_transformer_block_backward_matches_finite_difference_for_feed_forward_weights() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])
    upstream = np.array([[0.2, -0.1], [0.4, 0.7]])

    _result, cache = block.forward_with_cache(inputs)
    gradients = block.backward(upstream, cache)

    def loss(candidate_weights: np.ndarray) -> float:
        original = block.feed_forward.output_weights
        block.feed_forward.output_weights = candidate_weights
        try:
            return float(np.sum(block.forward(inputs).output * upstream))
        finally:
            block.feed_forward.output_weights = original

    numerical = np.zeros_like(block.feed_forward.output_weights)
    epsilon = 1e-5
    for row in range(block.feed_forward.output_weights.shape[0]):
        for column in range(block.feed_forward.output_weights.shape[1]):
            plus = block.feed_forward.output_weights.copy()
            minus = block.feed_forward.output_weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(
        gradients.feed_forward.output_weights,
        numerical,
        atol=1e-5,
    )


def test_transformer_block_backward_matches_input_gradient() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])
    upstream = np.array([[0.2, -0.1], [0.4, 0.7]])

    _result, cache = block.forward_with_cache(inputs)
    gradients = block.backward(upstream, cache)

    def loss(candidate_inputs: np.ndarray) -> float:
        return float(np.sum(block.forward(candidate_inputs).output * upstream))

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


def test_transformer_block_apply_gradients_updates_trainable_parameters() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])
    upstream = np.array([[0.2, -0.1], [0.4, 0.7]])
    original_attention_weights = block.attention.output_weights.copy()
    original_feed_forward_weights = block.feed_forward.output_weights.copy()

    _result, cache = block.forward_with_cache(inputs)
    gradients = block.backward(upstream, cache)
    block.apply_gradients(gradients, learning_rate=0.1)

    assert not np.allclose(block.feed_forward.output_weights, original_feed_forward_weights)
    assert not np.allclose(block.attention.output_weights, original_attention_weights)


def test_transformer_block_passes_attention_mask_into_heads() -> None:
    block = build_transformer_block()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])

    result = block.forward(inputs, attention_mask=causal_attention_mask(2))

    for head in result.attention.heads:
        assert head.weights[0, 1] == pytest.approx(0.0)


def test_transformer_block_rejects_non_matrix_inputs() -> None:
    block = build_transformer_block()

    with pytest.raises(ValueError, match="token rows"):
        block.forward(np.ones(2))


def test_transformer_block_rejects_attention_output_width_mismatch() -> None:
    block = build_transformer_block(output_weights=np.ones((2, 3)))

    with pytest.raises(ValueError, match="attention output"):
        block.forward(np.ones((2, 2)))
