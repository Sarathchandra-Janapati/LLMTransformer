import numpy as np
import pytest

from transformer_core import DecoderStack, LanguageModelHead
from transformer_core.attention import MultiHeadSelfAttention
from transformer_core.blocks import FeedForwardNetwork, LayerNorm, TransformerBlock


def build_block(scale: float = 0.25) -> TransformerBlock:
    return TransformerBlock(
        attention=MultiHeadSelfAttention(
            query_weights=[np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            key_weights=[np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            value_weights=[np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            output_weights=np.array([[scale, 0.0], [0.0, scale]]),
        ),
        attention_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
        feed_forward=FeedForwardNetwork(
            input_weights=np.array([[1.0, -1.0], [0.5, 1.0]]),
            input_biases=np.zeros(2),
            output_weights=np.array([[0.1, 0.0], [0.0, 0.1]]),
            output_biases=np.zeros(2),
        ),
        feed_forward_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
    )


def build_language_model() -> LanguageModelHead:
    return LanguageModelHead(
        decoder=DecoderStack(blocks=[build_block(0.25), build_block(0.1)]),
        final_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
        vocabulary_weights=np.array(
            [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
            ]
        ),
        vocabulary_biases=np.array([0.0, 0.0, 0.1]),
    )


def test_decoder_stack_runs_blocks_in_sequence() -> None:
    stack = DecoderStack(blocks=[build_block(0.25), build_block(0.1)])
    inputs = np.array([[1.0, 3.0], [2.0, 5.0], [1.0, 0.0]])

    result = stack.forward(inputs)

    assert len(result.blocks) == 2
    np.testing.assert_allclose(
        result.blocks[1].normalized_attention_input,
        stack.blocks[1].attention_norm.forward(result.blocks[0].output),
    )
    np.testing.assert_allclose(result.hidden_states, result.blocks[-1].output)


def test_decoder_stack_passes_causal_mask_to_each_block() -> None:
    stack = DecoderStack(blocks=[build_block()])
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])

    result = stack.forward(inputs, causal_mask=True)

    for head in result.blocks[0].attention.heads:
        assert head.weights[0, 1] == pytest.approx(0.0)


def test_decoder_stack_forward_with_cache_matches_forward() -> None:
    stack = DecoderStack(blocks=[build_block(0.25), build_block(0.1)])
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])

    result, cache = stack.forward_with_cache(inputs)

    np.testing.assert_allclose(result.hidden_states, stack.forward(inputs).hidden_states)
    assert len(cache.block_caches) == 2


def test_decoder_stack_backward_matches_input_finite_difference() -> None:
    stack = DecoderStack(blocks=[build_block(0.25), build_block(0.1)])
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])
    upstream = np.array([[0.2, -0.1], [0.4, 0.7]])

    _result, cache = stack.forward_with_cache(inputs)
    gradients = stack.backward(upstream, cache)

    def loss(candidate_inputs: np.ndarray) -> float:
        return float(np.sum(stack.forward(candidate_inputs).hidden_states * upstream))

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


def test_language_model_head_returns_logits_probabilities_and_predictions() -> None:
    model = build_language_model()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0], [1.0, 0.0]])

    result = model.forward(inputs)

    assert len(result.decoder.blocks) == 2
    assert result.normalized_hidden_states.shape == (3, 2)
    assert result.logits.shape == (3, 3)
    assert result.probabilities.shape == (3, 3)
    assert result.predicted_token_ids.shape == (3,)
    np.testing.assert_allclose(np.sum(result.probabilities, axis=1), np.ones(3))
    np.testing.assert_array_equal(
        result.predicted_token_ids,
        np.argmax(result.probabilities, axis=1),
    )


def test_language_model_head_forward_with_cache_matches_forward() -> None:
    model = build_language_model()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])

    result, cache = model.forward_with_cache(inputs)

    np.testing.assert_allclose(result.logits, model.forward(inputs).logits)
    assert len(cache.decoder.block_caches) == 2


def test_language_model_head_backward_matches_vocabulary_weight_finite_difference() -> None:
    model = build_language_model()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])
    logit_gradients = np.array([[0.2, -0.1, -0.1], [0.4, -0.2, -0.2]])

    result, cache = model.forward_with_cache(inputs)
    gradients = model.backward(logit_gradients, result, cache)

    def loss(candidate_weights: np.ndarray) -> float:
        original = model.vocabulary_weights
        model.vocabulary_weights = candidate_weights
        try:
            return float(np.sum(model.forward(inputs).logits * logit_gradients))
        finally:
            model.vocabulary_weights = original

    numerical = np.zeros_like(model.vocabulary_weights)
    epsilon = 1e-5
    for row in range(model.vocabulary_weights.shape[0]):
        for column in range(model.vocabulary_weights.shape[1]):
            plus = model.vocabulary_weights.copy()
            minus = model.vocabulary_weights.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (loss(plus) - loss(minus)) / (2.0 * epsilon)

    np.testing.assert_allclose(gradients.vocabulary_weights, numerical, atol=1e-5)


def test_language_model_head_apply_gradients_updates_decoder_and_head() -> None:
    model = build_language_model()
    inputs = np.array([[1.0, 3.0], [2.0, 5.0]])
    logit_gradients = np.array([[0.2, -0.1, -0.1], [0.4, -0.2, -0.2]])
    original_vocab = model.vocabulary_weights.copy()
    original_attention = model.decoder.blocks[0].attention.output_weights.copy()

    result, cache = model.forward_with_cache(inputs)
    gradients = model.backward(logit_gradients, result, cache)
    model.apply_gradients(gradients, learning_rate=0.01)

    assert not np.allclose(model.vocabulary_weights, original_vocab)
    assert np.linalg.norm(gradients.decoder.blocks[0].attention.output_weights) > 0.0
    np.testing.assert_allclose(
        model.decoder.blocks[0].attention.output_weights,
        original_attention - 0.01 * gradients.decoder.blocks[0].attention.output_weights,
    )


def test_decoder_stack_rejects_empty_blocks() -> None:
    stack = DecoderStack(blocks=[])

    with pytest.raises(ValueError, match="at least one"):
        stack.forward(np.ones((2, 2)))


def test_language_model_head_rejects_vocabulary_width_mismatch() -> None:
    model = LanguageModelHead(
        decoder=DecoderStack(blocks=[build_block()]),
        final_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2)),
        vocabulary_weights=np.ones((3, 4)),
        vocabulary_biases=np.zeros(4),
    )

    with pytest.raises(ValueError, match="model width"):
        model.forward(np.ones((2, 2)))
