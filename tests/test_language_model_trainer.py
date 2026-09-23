import numpy as np
import pytest

from training import (
    NextTokenTrainingExample,
    build_next_token_examples,
    cross_entropy_loss,
    next_token_accuracy,
    train_projection_and_embeddings,
    train_full_model,
    train_full_model_step,
    train_token_embeddings_step,
    train_output_projection,
    train_output_projection_step,
)
from transformer_core import DecoderStack, LanguageModelHead, TinyGPTModel
from transformer_core import TokenEmbeddingTable
from transformer_core.attention import MultiHeadSelfAttention
from transformer_core.blocks import FeedForwardNetwork, LayerNorm, TransformerBlock


def build_block(scale: float = 0.1) -> TransformerBlock:
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
            output_weights=np.array([[0.05, 0.0], [0.0, 0.05]]),
            output_biases=np.zeros(2),
        ),
        feed_forward_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
    )


def build_trainable_tiny_gpt() -> TinyGPTModel:
    return TinyGPTModel(
        token_embedding_table=TokenEmbeddingTable(
            embeddings=np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.2],
                    [0.4, 1.0],
                ]
            )
        ),
        language_model=LanguageModelHead(
            decoder=DecoderStack(blocks=[build_block()]),
            final_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
            vocabulary_weights=np.zeros((2, 3)),
            vocabulary_biases=np.zeros(3),
        ),
    )


def test_build_next_token_examples_creates_sliding_windows() -> None:
    examples = build_next_token_examples([1, 2, 0, 2], context_size=2)

    assert len(examples) == 2
    np.testing.assert_array_equal(examples[0].input_token_ids, np.array([1, 2]))
    np.testing.assert_array_equal(examples[0].target_token_ids, np.array([2, 0]))
    np.testing.assert_array_equal(examples[1].input_token_ids, np.array([2, 0]))
    np.testing.assert_array_equal(examples[1].target_token_ids, np.array([0, 2]))


def test_cross_entropy_loss_uses_correct_token_probabilities() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

    loss = cross_entropy_loss(probabilities, [0, 1])

    assert loss == pytest.approx(-np.mean(np.log([0.7, 0.8])))


def test_next_token_accuracy_counts_exact_token_matches() -> None:
    assert next_token_accuracy([2, 1, 0], [2, 0, 0]) == pytest.approx(2 / 3)


def test_train_output_projection_step_updates_weights_and_returns_metrics() -> None:
    model = build_trainable_tiny_gpt()
    example = NextTokenTrainingExample(
        input_token_ids=np.array([1, 2]),
        target_token_ids=np.array([2, 0]),
    )
    weights_before = model.language_model.vocabulary_weights.copy()

    result = train_output_projection_step(
        model=model,
        example=example,
        learning_rate=0.1,
        step=3,
    )

    assert result.metrics.step == 3
    assert result.metrics.loss > 0
    assert result.metrics.token_count == 2
    assert result.gradients.vocabulary_weights.shape == (2, 3)
    assert result.gradients.vocabulary_biases.shape == (3,)
    assert not np.allclose(model.language_model.vocabulary_weights, weights_before)


def test_train_output_projection_reduces_loss_on_repeated_tiny_sequence() -> None:
    model = build_trainable_tiny_gpt()
    example = NextTokenTrainingExample(
        input_token_ids=np.array([1, 2]),
        target_token_ids=np.array([2, 0]),
    )
    initial_result = model.forward(example.input_token_ids, causal_mask=True)
    initial_loss = cross_entropy_loss(
        initial_result.language_model.probabilities,
        example.target_token_ids,
    )

    train_output_projection(model, [example], learning_rate=0.05, epochs=50)
    final_result = model.forward(example.input_token_ids, causal_mask=True)
    final_loss = cross_entropy_loss(
        final_result.language_model.probabilities,
        example.target_token_ids,
    )

    assert final_loss < initial_loss


def test_train_token_embeddings_step_updates_touched_embedding_rows() -> None:
    model = build_trainable_tiny_gpt()
    example = NextTokenTrainingExample(
        input_token_ids=np.array([1]),
        target_token_ids=np.array([2]),
    )
    train_output_projection(model, [example], learning_rate=0.05, epochs=10)
    embeddings_before = model.token_embedding_table.embeddings.copy()

    result = train_token_embeddings_step(
        model=model,
        example=example,
        learning_rate=0.01,
        finite_difference_epsilon=1e-4,
        step=7,
    )

    assert result.metrics.step == 7
    assert result.gradients.token_embeddings.shape == embeddings_before.shape
    assert np.linalg.norm(result.gradients.token_embeddings[1]) > 0.0
    np.testing.assert_allclose(
        model.token_embedding_table.embeddings[1],
        embeddings_before[1] - 0.01 * result.gradients.token_embeddings[1],
    )
    np.testing.assert_allclose(
        model.token_embedding_table.embeddings[0],
        embeddings_before[0],
    )


def test_train_projection_and_embeddings_returns_both_step_types() -> None:
    model = build_trainable_tiny_gpt()
    example = NextTokenTrainingExample(
        input_token_ids=np.array([1]),
        target_token_ids=np.array([2]),
    )

    history = train_projection_and_embeddings(
        model=model,
        examples=[example],
        projection_learning_rate=0.05,
        embedding_learning_rate=0.01,
        epochs=2,
    )

    assert len(history) == 4
    assert history[0].metrics.step == 0
    assert history[-1].metrics.step == 3


def test_train_full_model_step_updates_embeddings_and_decoder() -> None:
    model = build_trainable_tiny_gpt()
    example = NextTokenTrainingExample(
        input_token_ids=np.array([1, 2]),
        target_token_ids=np.array([2, 0]),
    )
    embeddings_before = model.token_embedding_table.embeddings.copy()
    model.language_model.vocabulary_weights = np.array(
        [
            [0.2, -0.1, 0.3],
            [0.1, 0.25, -0.2],
        ]
    )
    attention_before = model.language_model.decoder.blocks[0].attention.output_weights.copy()

    result = train_full_model_step(
        model=model,
        example=example,
        learning_rate=0.01,
        step=5,
    )

    assert result.metrics.step == 5
    assert result.gradients.token_embeddings.shape == embeddings_before.shape
    assert np.linalg.norm(result.gradients.language_model.decoder.blocks[0].attention.output_weights) > 0.0
    assert np.linalg.norm(result.gradients.token_embeddings) > 0.0
    np.testing.assert_allclose(
        model.token_embedding_table.embeddings,
        embeddings_before - 0.01 * result.gradients.token_embeddings,
    )
    np.testing.assert_allclose(
        model.language_model.decoder.blocks[0].attention.output_weights,
        attention_before
        - 0.01 * result.gradients.language_model.decoder.blocks[0].attention.output_weights,
    )


def test_train_full_model_reduces_loss_on_tiny_sequence() -> None:
    model = build_trainable_tiny_gpt()
    example = NextTokenTrainingExample(
        input_token_ids=np.array([1]),
        target_token_ids=np.array([2]),
    )
    initial_result = model.forward(example.input_token_ids, causal_mask=True)
    initial_loss = cross_entropy_loss(
        initial_result.language_model.probabilities,
        example.target_token_ids,
    )

    history = train_full_model(model, [example], learning_rate=0.02, epochs=30)
    final_result = model.forward(example.input_token_ids, causal_mask=True)
    final_loss = cross_entropy_loss(
        final_result.language_model.probabilities,
        example.target_token_ids,
    )

    assert len(history) == 30
    assert final_loss < initial_loss


def test_build_next_token_examples_rejects_short_sequences() -> None:
    with pytest.raises(ValueError, match="longer than the context size"):
        build_next_token_examples([1, 2], context_size=2)
