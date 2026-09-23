"""Deterministic attention examples for the first visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from tokenizer import WordTokenizer
from transformer_core.foundations.linear_algebra import FloatArray

from .single_head import AttentionResult, SingleHeadSelfAttention

DEFAULT_SENTENCE = "I love transformers"
LAB_VOCABULARY = {
    "<unk>": 0,
    "i": 1,
    "love": 2,
    "transformers": 3,
    "attention": 4,
    "tokens": 5,
    "learn": 6,
}

TASK_MODES = {
    "next_token": {
        "label": "Next-token prediction",
        "mask_rule": "Block every future token where column position is greater than row position.",
        "reason": "A GPT-style decoder must predict the next token without seeing the answer or anything after it.",
    },
    "fill_blank": {
        "label": "Fill-in-the-blank",
        "mask_rule": "Allow both left and right context.",
        "reason": "A masked-word task can use surrounding words to infer the missing token.",
    },
    "classification": {
        "label": "Sentence classification",
        "mask_rule": "Allow every token to see every other token.",
        "reason": "Classification uses the whole sentence representation, so future-word hiding is unnecessary.",
    },
    "local_window": {
        "label": "Local window",
        "mask_rule": "Only allow each token to see itself and immediate neighbors.",
        "reason": "A custom task may restrict communication to nearby tokens to study local context.",
    },
}


@dataclass(frozen=True)
class AttentionLabExample:
    """Token ids, embeddings, and attention tensors for one sentence."""

    text: str
    tokens: list[str]
    token_ids: list[int]
    causal_mask: bool
    task_mode: str
    task_label: str
    mask_rule: str
    task_reason: str
    embeddings: FloatArray
    position_encodings: FloatArray
    attention_inputs: FloatArray
    query_weights: FloatArray
    key_weights: FloatArray
    value_weights: FloatArray
    attention: AttentionResult

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-ready values for a frontend attention heatmap."""
        return {
            "text": self.text,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "causal_mask": self.causal_mask,
            "task_mode": self.task_mode,
            "task_label": self.task_label,
            "mask_rule": self.mask_rule,
            "task_reason": self.task_reason,
            "embeddings": self.embeddings.tolist(),
            "position_encodings": self.position_encodings.tolist(),
            "attention_inputs": self.attention_inputs.tolist(),
            "query_weights": self.query_weights.tolist(),
            "key_weights": self.key_weights.tolist(),
            "value_weights": self.value_weights.tolist(),
            "attention_width": self.attention.keys.shape[1],
            "attention_scale": sqrt(self.attention.keys.shape[1]),
            "queries": self.attention.queries.tolist(),
            "keys": self.attention.keys.tolist(),
            "values": self.attention.values.tolist(),
            "scores": _masked_score_payload(self.attention),
            "mask": (
                self.attention.mask.tolist()
                if self.attention.mask is not None
                else None
            ),
            "weights": self.attention.weights.tolist(),
            "context": self.attention.context.tolist(),
        }


def build_attention_lab_example(
    text: str = DEFAULT_SENTENCE,
    causal_mask: bool = False,
    task_mode: str | None = None,
) -> AttentionLabExample:
    """Build an attention-lab sentence with deterministic tiny embeddings."""
    tokenizer = WordTokenizer(LAB_VOCABULARY)
    tokens = tokenizer.tokenize(text)
    if not tokens:
        raise ValueError("attention lab text must include at least one token")

    token_ids = tokenizer.encode(text)
    embeddings = _embedding_table()[token_ids]
    position_encodings = sinusoidal_position_encodings(
        token_count=len(tokens),
        width=embeddings.shape[1],
    )
    attention_inputs = embeddings + position_encodings

    resolved_task_mode = _resolve_task_mode(task_mode, causal_mask)
    task = TASK_MODES[resolved_task_mode]
    attention_mask = task_attention_mask(len(tokens), resolved_task_mode)

    query_weights, key_weights, value_weights = _projection_weights()
    attention = SingleHeadSelfAttention(
        query_weights=query_weights,
        key_weights=key_weights,
        value_weights=value_weights,
    ).forward(attention_inputs, attention_mask=attention_mask)

    return AttentionLabExample(
        text=text,
        tokens=tokens,
        token_ids=token_ids,
        causal_mask=resolved_task_mode == "next_token",
        task_mode=resolved_task_mode,
        task_label=task["label"],
        mask_rule=task["mask_rule"],
        task_reason=task["reason"],
        embeddings=embeddings,
        position_encodings=position_encodings,
        attention_inputs=attention_inputs,
        query_weights=query_weights,
        key_weights=key_weights,
        value_weights=value_weights,
        attention=attention,
    )


def build_i_love_transformers_example(
    text: str = DEFAULT_SENTENCE,
    causal_mask: bool = False,
    task_mode: str | None = None,
) -> AttentionLabExample:
    """Build the original fixed attention-lab sentence example."""
    return build_attention_lab_example(text, causal_mask=causal_mask, task_mode=task_mode)


def task_attention_mask(
    token_count: int,
    task_mode: str,
) -> np.ndarray | None:
    """Return a blocking mask for the selected learning task."""
    if token_count <= 0:
        raise ValueError("task masks need at least one token")
    if task_mode == "next_token":
        return np.triu(np.ones((token_count, token_count), dtype=bool), k=1)
    if task_mode in {"fill_blank", "classification"}:
        return None
    if task_mode == "local_window":
        positions = np.arange(token_count)
        distance = np.abs(positions[:, None] - positions[None, :])
        return distance > 1
    raise ValueError(f"unknown task mode: {task_mode}")


def _resolve_task_mode(task_mode: str | None, causal_mask: bool) -> str:
    """Keep older causal_mask calls working while exposing task modes."""
    if task_mode is None:
        return "next_token" if causal_mask else "classification"
    if task_mode not in TASK_MODES:
        raise ValueError(f"unknown task mode: {task_mode}")
    return task_mode


def _embedding_table() -> FloatArray:
    """Return handcrafted embeddings for a tiny attention-lab vocabulary."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [0.4, 1.0, 0.3],
            [0.2, 0.5, 1.0],
            [0.9, 0.4, 0.8],
            [0.3, 0.9, 0.7],
            [0.7, 0.6, 0.2],
        ]
    )


def _projection_weights() -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return deterministic Q, K, and V projection matrices for the lab."""
    return (
        np.array(
            [
                [0.8, 0.1],
                [0.2, 0.7],
                [0.5, -0.3],
            ]
        ),
        np.array(
            [
                [0.6, 0.0],
                [0.1, 0.8],
                [0.4, -0.2],
            ]
        ),
        np.array(
            [
                [0.9, 0.1],
                [0.0, 0.8],
                [0.3, 0.5],
            ]
        ),
    )


def sinusoidal_position_encodings(token_count: int, width: int) -> FloatArray:
    """Return sinusoidal position vectors for one short token sequence."""
    if token_count <= 0 or width <= 0:
        raise ValueError("position encodings need positive token and width sizes")

    encodings = np.zeros((token_count, width), dtype=float)
    positions = np.arange(token_count, dtype=float)
    for dimension in range(0, width, 2):
        scale = 10000 ** (dimension / width)
        encodings[:, dimension] = np.sin(positions / scale)
        if dimension + 1 < width:
            encodings[:, dimension + 1] = np.cos(positions / scale)
    return encodings


def _masked_score_payload(attention: AttentionResult) -> list[list[float | None]]:
    """Return JSON-safe scores, leaving masked cells empty for the UI."""
    if attention.mask is None:
        return attention.scores.tolist()

    return [
        [
            None if attention.mask[row_index, column_index] else float(score)
            for column_index, score in enumerate(row)
        ]
        for row_index, row in enumerate(attention.scores)
    ]
