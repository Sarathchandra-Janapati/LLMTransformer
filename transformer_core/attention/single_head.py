"""Single-head self-attention for the first attention visualizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from transformer_core.foundations.linear_algebra import FloatArray, matrix_multiply


def softmax(values: FloatArray) -> FloatArray:
    """Return row-wise softmax probabilities for a score matrix."""
    if values.ndim != 2:
        raise ValueError("softmax expects a two-dimensional score matrix")

    shifted_values = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted_values)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


@dataclass(frozen=True)
class AttentionResult:
    """All tensors needed to explain one self-attention forward pass."""

    queries: FloatArray
    keys: FloatArray
    values: FloatArray
    scores: FloatArray
    mask: NDArray[np.bool_] | None
    weights: FloatArray
    context: FloatArray


@dataclass(frozen=True)
class AttentionCache:
    """Saved inputs and outputs needed for attention backpropagation."""

    embeddings: FloatArray
    result: AttentionResult


@dataclass(frozen=True)
class AttentionGradients:
    """Gradients for single-head attention parameters and inputs."""

    embeddings: FloatArray
    query_weights: FloatArray
    key_weights: FloatArray
    value_weights: FloatArray


@dataclass
class SingleHeadSelfAttention:
    """Project embeddings into Q, K, and V before mixing token values."""

    query_weights: FloatArray
    key_weights: FloatArray
    value_weights: FloatArray

    def forward(
        self,
        embeddings: FloatArray,
        causal_mask: bool = False,
        attention_mask: NDArray[np.bool_] | None = None,
    ) -> AttentionResult:
        """Compute scaled dot-product self-attention for token embeddings."""
        result, _cache = self.forward_with_cache(
            embeddings,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
        )
        return result

    def forward_with_cache(
        self,
        embeddings: FloatArray,
        causal_mask: bool = False,
        attention_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[AttentionResult, AttentionCache]:
        """Compute attention and keep values needed for backpropagation."""
        self._validate_inputs(embeddings)
        if causal_mask and attention_mask is not None:
            raise ValueError("provide either causal_mask or attention_mask, not both")

        queries = matrix_multiply(embeddings, self.query_weights)
        keys = matrix_multiply(embeddings, self.key_weights)
        values = matrix_multiply(embeddings, self.value_weights)

        scores = matrix_multiply(queries, keys.T) / sqrt(keys.shape[1])
        mask = attention_mask if attention_mask is not None else None
        if causal_mask:
            mask = causal_attention_mask(scores.shape[0])
        if mask is not None and mask.shape != scores.shape:
            raise ValueError("attention mask must match the score matrix shape")
        if mask is not None:
            scores = np.where(mask, -np.inf, scores)

        weights = softmax(scores)
        context = matrix_multiply(weights, values)

        result = AttentionResult(
            queries=queries,
            keys=keys,
            values=values,
            scores=scores,
            mask=mask,
            weights=weights,
            context=context,
        )
        return result, AttentionCache(embeddings=embeddings, result=result)

    def backward(
        self,
        upstream_gradients: FloatArray,
        cache: AttentionCache,
    ) -> AttentionGradients:
        """Backpropagate through scaled dot-product self-attention."""
        self._validate_inputs(cache.embeddings)
        result = cache.result
        if upstream_gradients.shape != result.context.shape:
            raise ValueError("upstream gradients must match the attention context shape")

        weight_gradients = matrix_multiply(upstream_gradients, result.values.T)
        value_gradients = matrix_multiply(result.weights.T, upstream_gradients)

        score_gradients = result.weights * (
            weight_gradients
            - np.sum(weight_gradients * result.weights, axis=1, keepdims=True)
        )
        if result.mask is not None:
            score_gradients = np.where(result.mask, 0.0, score_gradients)

        scale = sqrt(result.keys.shape[1])
        query_gradients = matrix_multiply(score_gradients, result.keys) / scale
        key_gradients = matrix_multiply(score_gradients.T, result.queries) / scale

        query_weight_gradients = matrix_multiply(cache.embeddings.T, query_gradients)
        key_weight_gradients = matrix_multiply(cache.embeddings.T, key_gradients)
        value_weight_gradients = matrix_multiply(cache.embeddings.T, value_gradients)
        embedding_gradients = (
            matrix_multiply(query_gradients, self.query_weights.T)
            + matrix_multiply(key_gradients, self.key_weights.T)
            + matrix_multiply(value_gradients, self.value_weights.T)
        )

        return AttentionGradients(
            embeddings=embedding_gradients,
            query_weights=query_weight_gradients,
            key_weights=key_weight_gradients,
            value_weights=value_weight_gradients,
        )

    def apply_gradients(
        self,
        gradients: AttentionGradients,
        learning_rate: float,
    ) -> None:
        """Update Q, K, and V projection parameters."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.query_weights -= learning_rate * gradients.query_weights
        self.key_weights -= learning_rate * gradients.key_weights
        self.value_weights -= learning_rate * gradients.value_weights

    def _validate_inputs(self, embeddings: FloatArray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("attention expects one embedding row per token")
        if any(
            weights.ndim != 2
            for weights in (self.query_weights, self.key_weights, self.value_weights)
        ):
            raise ValueError("attention projection weights must be matrices")
        if embeddings.shape[1] != self.query_weights.shape[0]:
            raise ValueError("query projection must match the embedding width")
        if embeddings.shape[1] != self.key_weights.shape[0]:
            raise ValueError("key projection must match the embedding width")
        if embeddings.shape[1] != self.value_weights.shape[0]:
            raise ValueError("value projection must match the embedding width")
        if self.query_weights.shape[1] != self.key_weights.shape[1]:
            raise ValueError("query and key projections must share an attention width")


def causal_attention_mask(token_count: int) -> NDArray[np.bool_]:
    """Mark future token positions hidden from decoder queries."""
    if token_count <= 0:
        raise ValueError("causal masks need at least one token")

    return np.triu(np.ones((token_count, token_count), dtype=bool), k=1)
