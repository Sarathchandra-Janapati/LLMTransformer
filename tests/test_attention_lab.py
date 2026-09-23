import numpy as np

from tokenizer import WordTokenizer
import pytest

from transformer_core.attention import (
    build_attention_lab_example,
    build_i_love_transformers_example,
    sinusoidal_position_encodings,
    task_attention_mask,
)


def test_word_tokenizer_encodes_known_and_unknown_tokens() -> None:
    tokenizer = WordTokenizer({"<unk>": 0, "i": 1, "love": 2})

    assert tokenizer.tokenize("I love attention") == ["i", "love", "attention"]
    assert tokenizer.encode("I love attention") == [1, 2, 0]


def test_attention_lab_example_uses_one_embedding_per_sentence_token() -> None:
    example = build_i_love_transformers_example()

    assert example.tokens == ["i", "love", "transformers"]
    assert example.token_ids == [1, 2, 3]
    assert example.embeddings.shape == (3, 3)
    assert example.position_encodings.shape == (3, 3)
    np.testing.assert_allclose(
        example.attention_inputs,
        example.embeddings + example.position_encodings,
    )
    assert example.attention.weights.shape == (3, 3)
    np.testing.assert_allclose(np.sum(example.attention.weights, axis=1), np.ones(3))


def test_attention_lab_payload_is_ready_for_json_serialization() -> None:
    payload = build_i_love_transformers_example().to_payload()

    assert payload["text"] == "I love transformers"
    assert payload["tokens"] == ["i", "love", "transformers"]
    assert payload["token_ids"] == [1, 2, 3]
    assert isinstance(payload["weights"], list)
    assert payload["attention_width"] == 2
    assert payload["attention_scale"] == pytest.approx(np.sqrt(2))
    assert len(payload["query_weights"]) == 3
    assert len(payload["position_encodings"]) == len(payload["tokens"])
    assert len(payload["attention_inputs"]) == len(payload["tokens"])
    assert len(payload["scores"]) == len(payload["tokens"])


def test_attention_lab_supports_controlled_vocabulary_and_unknown_words() -> None:
    example = build_attention_lab_example("tokens learn mystery")

    assert example.tokens == ["tokens", "learn", "mystery"]
    assert example.token_ids == [5, 6, 0]
    assert example.attention.weights.shape == (3, 3)


def test_attention_lab_rejects_empty_text_before_attention_math() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        build_attention_lab_example("   ")


def test_attention_lab_payload_exposes_causal_mask_cells() -> None:
    payload = build_attention_lab_example("i love attention", causal_mask=True).to_payload()

    assert payload["causal_mask"] is True
    assert payload["mask"][0] == [False, True, True]
    assert payload["weights"][0][1] == pytest.approx(0.0)


def test_attention_lab_supports_local_window_task_mode() -> None:
    payload = build_attention_lab_example(
        "i love attention tokens",
        task_mode="local_window",
    ).to_payload()

    assert payload["task_mode"] == "local_window"
    assert payload["mask"][0] == [False, False, True, True]
    assert payload["weights"][0][2] == pytest.approx(0.0)


def test_task_attention_mask_returns_none_for_bidirectional_tasks() -> None:
    assert task_attention_mask(3, "classification") is None
    assert task_attention_mask(3, "fill_blank") is None


def test_sinusoidal_position_encodings_start_with_zero_and_one_pair() -> None:
    encodings = sinusoidal_position_encodings(token_count=3, width=3)

    np.testing.assert_allclose(encodings[0], np.array([0.0, 1.0, 0.0]))
    assert encodings[1, 0] != encodings[2, 0]
