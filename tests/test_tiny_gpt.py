import numpy as np
import pytest

from transformer_core import DecoderStack, LanguageModelHead, TinyGPTModel
from transformer_core import TokenEmbeddingTable
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


def build_tiny_gpt() -> TinyGPTModel:
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
            decoder=DecoderStack(blocks=[build_block(0.25), build_block(0.1)]),
            final_norm=LayerNorm(gamma=np.ones(2), beta=np.zeros(2), epsilon=1e-8),
            vocabulary_weights=np.array(
                [
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -1.0],
                ]
            ),
            vocabulary_biases=np.array([0.0, 0.0, 0.1]),
        ),
    )


def test_tiny_gpt_runs_from_token_ids_to_token_predictions() -> None:
    model = build_tiny_gpt()

    result = model.forward([1, 2, 0])

    np.testing.assert_array_equal(result.token_ids, np.array([1, 2, 0]))
    assert result.token_embeddings.shape == (3, 2)
    assert result.position_encodings.shape == (3, 2)
    assert result.model_inputs.shape == (3, 2)
    assert result.language_model.logits.shape == (3, 3)
    assert result.language_model.probabilities.shape == (3, 3)
    assert result.language_model.predicted_token_ids.shape == (3,)
    np.testing.assert_allclose(
        np.sum(result.language_model.probabilities, axis=1),
        np.ones(3),
    )


def test_tiny_gpt_adds_position_encodings_to_token_embeddings() -> None:
    model = build_tiny_gpt()

    result = model.forward(np.array([1, 2]))

    np.testing.assert_allclose(
        result.model_inputs,
        result.token_embeddings + result.position_encodings,
    )


def test_token_embedding_table_rejects_unknown_token_id() -> None:
    model = build_tiny_gpt()

    with pytest.raises(ValueError, match="outside the embedding vocabulary"):
        model.forward([1, 4])


def test_token_embedding_table_rejects_non_vector_token_ids() -> None:
    table = TokenEmbeddingTable(embeddings=np.ones((3, 2)))

    with pytest.raises(ValueError, match="one-dimensional"):
        table.forward(np.ones((2, 2), dtype=int))


def test_tiny_gpt_rejects_embedding_and_output_vocabulary_mismatch() -> None:
    model = build_tiny_gpt()
    model.token_embedding_table = TokenEmbeddingTable(embeddings=np.ones((4, 2)))

    with pytest.raises(ValueError, match="input vocabulary size"):
        model.forward([1, 2])
