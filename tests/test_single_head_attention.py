from math import sqrt

import numpy as np
import pytest

from transformer_core.attention import (
    SingleHeadSelfAttention,
    causal_attention_mask,
    softmax,
)


def test_softmax_normalizes_each_token_row() -> None:
    weights = softmax(np.array([[1.0, 2.0], [4.0, 4.0]]))

    np.testing.assert_allclose(np.sum(weights, axis=1), np.array([1.0, 1.0]))
    np.testing.assert_allclose(weights[1], np.array([0.5, 0.5]))


def test_single_head_attention_exposes_each_intermediate_tensor() -> None:
    attention = SingleHeadSelfAttention(
        query_weights=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        key_weights=np.array([[0.5, 0.0], [0.0, 0.5], [1.0, -1.0]]),
        value_weights=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
    )
    embeddings = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    result = attention.forward(embeddings)

    expected_queries = embeddings @ attention.query_weights
    expected_keys = embeddings @ attention.key_weights
    expected_values = embeddings @ attention.value_weights
    expected_scores = (expected_queries @ expected_keys.T) / sqrt(2)

    np.testing.assert_allclose(result.queries, expected_queries)
    np.testing.assert_allclose(result.keys, expected_keys)
    np.testing.assert_allclose(result.values, expected_values)
    np.testing.assert_allclose(result.scores, expected_scores)
    np.testing.assert_allclose(result.context, result.weights @ result.values)
    assert result.weights.shape == (3, 3)


def test_scaled_attention_weights_sum_to_one_for_each_query_token() -> None:
    identity = np.eye(2)
    attention = SingleHeadSelfAttention(identity, identity, identity)

    result = attention.forward(np.array([[1.0, 0.0], [0.0, 1.0]]))

    np.testing.assert_allclose(np.sum(result.weights, axis=1), np.ones(2))
    assert result.weights[0, 0] > result.weights[0, 1]
    assert result.weights[1, 1] > result.weights[1, 0]


def test_single_head_attention_forward_with_cache_matches_forward() -> None:
    identity = np.eye(2)
    attention = SingleHeadSelfAttention(identity, identity, identity)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    result, cache = attention.forward_with_cache(embeddings)

    np.testing.assert_allclose(result.context, attention.forward(embeddings).context)
    np.testing.assert_allclose(cache.embeddings, embeddings)


def test_single_head_attention_backward_matches_query_weight_finite_difference() -> None:
    attention = SingleHeadSelfAttention(
        query_weights=np.array([[0.8, -0.2], [0.1, 0.5]]),
        key_weights=np.array([[0.4, 0.7], [-0.3, 0.6]]),
        value_weights=np.array([[0.9, -0.4], [0.2, 0.3]]),
    )
    embeddings = np.array([[1.0, 0.5], [-0.25, 1.2]])
    upstream = np.array([[0.3, -0.1], [0.2, 0.4]])

    _result, cache = attention.forward_with_cache(embeddings)
    gradients = attention.backward(upstream, cache)

    def loss(candidate_weights: np.ndarray) -> float:
        original = attention.query_weights
        attention.query_weights = candidate_weights
        try:
            return float(np.sum(attention.forward(embeddings).context * upstream))
        finally:
            attention.query_weights = original

    numerical = np.zeros_like(attention.query_weights)
    epsilon = 1e-5
    for row in range(attention.query_weights.shape[0]):
        for column in range(attention.query_weights.shape[1]):
            plus = attention.query_weights.copy()
            minus = attention.query_weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(gradients.query_weights, numerical, atol=1e-5)


def test_single_head_attention_backward_matches_embedding_finite_difference() -> None:
    attention = SingleHeadSelfAttention(
        query_weights=np.array([[0.8, -0.2], [0.1, 0.5]]),
        key_weights=np.array([[0.4, 0.7], [-0.3, 0.6]]),
        value_weights=np.array([[0.9, -0.4], [0.2, 0.3]]),
    )
    embeddings = np.array([[1.0, 0.5], [-0.25, 1.2]])
    upstream = np.array([[0.3, -0.1], [0.2, 0.4]])

    _result, cache = attention.forward_with_cache(embeddings)
    gradients = attention.backward(upstream, cache)

    def loss(candidate_embeddings: np.ndarray) -> float:
        return float(np.sum(attention.forward(candidate_embeddings).context * upstream))

    numerical = np.zeros_like(embeddings)
    epsilon = 1e-5
    for row in range(embeddings.shape[0]):
        for column in range(embeddings.shape[1]):
            plus = embeddings.copy()
            minus = embeddings.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(gradients.embeddings, numerical, atol=1e-5)


def test_single_head_attention_backward_respects_causal_mask() -> None:
    identity = np.eye(2)
    attention = SingleHeadSelfAttention(identity, identity, identity)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    upstream = np.array([[1.0, 0.0], [0.0, 0.0]])

    _result, cache = attention.forward_with_cache(embeddings, causal_mask=True)
    gradients = attention.backward(upstream, cache)

    np.testing.assert_allclose(gradients.query_weights, np.zeros((2, 2)))
    np.testing.assert_allclose(gradients.key_weights, np.zeros((2, 2)))
    np.testing.assert_allclose(gradients.value_weights[:, 1], np.zeros(2))


def test_single_head_attention_apply_gradients_updates_projection_weights() -> None:
    identity = np.eye(2)
    attention = SingleHeadSelfAttention(identity.copy(), identity.copy(), identity.copy())
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    upstream = np.array([[0.3, -0.1], [0.2, 0.4]])

    _result, cache = attention.forward_with_cache(embeddings)
    gradients = attention.backward(upstream, cache)
    original_values = attention.value_weights.copy()
    attention.apply_gradients(gradients, learning_rate=0.1)

    assert not np.allclose(attention.value_weights, original_values)


def test_causal_attention_blocks_future_token_weights() -> None:
    identity = np.eye(2)
    attention = SingleHeadSelfAttention(identity, identity, identity)

    result = attention.forward(np.array([[1.0, 0.0], [0.0, 1.0]]), causal_mask=True)

    assert result.mask is not None
    assert np.isneginf(result.scores[0, 1])
    assert result.weights[0, 1] == pytest.approx(0.0)
    assert result.weights[0, 0] == pytest.approx(1.0)


def test_causal_attention_mask_marks_future_positions() -> None:
    np.testing.assert_array_equal(
        causal_attention_mask(3),
        np.array(
            [
                [False, True, True],
                [False, False, True],
                [False, False, False],
            ]
        ),
    )


def test_attention_rejects_query_and_key_width_mismatch() -> None:
    attention = SingleHeadSelfAttention(
        query_weights=np.ones((3, 2)),
        key_weights=np.ones((3, 4)),
        value_weights=np.ones((3, 2)),
    )

    with pytest.raises(ValueError, match="attention width"):
        attention.forward(np.ones((2, 3)))
