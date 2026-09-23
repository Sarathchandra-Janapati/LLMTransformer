import numpy as np
import pytest

from transformer_core.attention import MultiHeadSelfAttention, causal_attention_mask


def build_two_head_attention() -> MultiHeadSelfAttention:
    return MultiHeadSelfAttention(
        query_weights=[
            np.array([[1.0], [0.0], [0.5]]),
            np.array([[0.0], [1.0], [0.25]]),
        ],
        key_weights=[
            np.array([[0.5], [0.0], [1.0]]),
            np.array([[0.0], [0.5], [1.0]]),
        ],
        value_weights=[
            np.array([[1.0], [0.0], [0.0]]),
            np.array([[0.0], [1.0], [0.0]]),
        ],
        output_weights=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_multi_head_attention_returns_each_head_and_projected_output() -> None:
    attention = build_two_head_attention()
    embeddings = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    result = attention.forward(embeddings)

    assert len(result.heads) == 2
    assert result.heads[0].queries.shape == (3, 1)
    assert result.heads[1].weights.shape == (3, 3)
    assert result.concatenated_context.shape == (3, 2)
    assert result.output.shape == (3, 2)
    np.testing.assert_allclose(result.output, result.concatenated_context)


def test_multi_head_attention_concatenates_head_contexts_in_order() -> None:
    attention = build_two_head_attention()
    embeddings = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    result = attention.forward(embeddings)

    np.testing.assert_allclose(
        result.concatenated_context,
        np.concatenate(
            [result.heads[0].context, result.heads[1].context],
            axis=1,
        ),
    )


def test_multi_head_attention_forward_with_cache_matches_forward() -> None:
    attention = build_two_head_attention()
    embeddings = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    result, cache = attention.forward_with_cache(embeddings)

    np.testing.assert_allclose(result.output, attention.forward(embeddings).output)
    np.testing.assert_allclose(cache.embeddings, embeddings)
    assert len(cache.head_caches) == 2


def test_multi_head_attention_backward_matches_output_weight_finite_difference() -> None:
    attention = build_two_head_attention()
    embeddings = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    upstream = np.array([[0.3, -0.1], [0.2, 0.4], [-0.2, 0.1]])

    _result, cache = attention.forward_with_cache(embeddings)
    gradients = attention.backward(upstream, cache)

    def loss(candidate_weights: np.ndarray) -> float:
        original = attention.output_weights
        attention.output_weights = candidate_weights
        try:
            return float(np.sum(attention.forward(embeddings).output * upstream))
        finally:
            attention.output_weights = original

    numerical = np.zeros_like(attention.output_weights)
    epsilon = 1e-5
    for row in range(attention.output_weights.shape[0]):
        for column in range(attention.output_weights.shape[1]):
            plus = attention.output_weights.copy()
            minus = attention.output_weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(gradients.output_weights, numerical, atol=1e-5)


def test_multi_head_attention_backward_matches_embedding_finite_difference() -> None:
    attention = build_two_head_attention()
    embeddings = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    upstream = np.array([[0.3, -0.1], [0.2, 0.4], [-0.2, 0.1]])

    _result, cache = attention.forward_with_cache(embeddings)
    gradients = attention.backward(upstream, cache)

    def loss(candidate_embeddings: np.ndarray) -> float:
        return float(np.sum(attention.forward(candidate_embeddings).output * upstream))

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


def test_multi_head_attention_apply_gradients_updates_attention_parameters() -> None:
    attention = build_two_head_attention()
    embeddings = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    upstream = np.array([[0.3, -0.1], [0.2, 0.4]])
    original_output_weights = attention.output_weights.copy()

    _result, cache = attention.forward_with_cache(embeddings)
    gradients = attention.backward(upstream, cache)
    attention.apply_gradients(gradients, learning_rate=0.1)

    assert not np.allclose(attention.output_weights, original_output_weights)


def test_multi_head_attention_applies_mask_to_each_head() -> None:
    attention = build_two_head_attention()
    embeddings = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    result = attention.forward(embeddings, attention_mask=causal_attention_mask(2))

    for head in result.heads:
        assert head.weights[0, 1] == pytest.approx(0.0)
        assert head.weights[0, 0] == pytest.approx(1.0)


def test_multi_head_attention_rejects_mismatched_head_counts() -> None:
    attention = MultiHeadSelfAttention(
        query_weights=[np.ones((3, 1))],
        key_weights=[],
        value_weights=[np.ones((3, 1))],
        output_weights=np.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="head counts"):
        attention.forward(np.ones((2, 3)))


def test_multi_head_attention_rejects_bad_output_projection_shape() -> None:
    attention = MultiHeadSelfAttention(
        query_weights=[np.ones((3, 1)), np.ones((3, 1))],
        key_weights=[np.ones((3, 1)), np.ones((3, 1))],
        value_weights=[np.ones((3, 1)), np.ones((3, 1))],
        output_weights=np.ones((3, 2)),
    )

    with pytest.raises(ValueError, match="concatenated context"):
        attention.forward(np.ones((2, 3)))
