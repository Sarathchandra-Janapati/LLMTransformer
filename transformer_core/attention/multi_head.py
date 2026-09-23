"""Multi-head self-attention built from inspectable single heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transformer_core.foundations.linear_algebra import FloatArray, matrix_multiply

from .single_head import (
    AttentionCache,
    AttentionGradients,
    AttentionResult,
    SingleHeadSelfAttention,
)


@dataclass(frozen=True)
class MultiHeadAttentionResult:
    """All tensors needed to inspect one multi-head attention pass."""

    heads: list[AttentionResult]
    concatenated_context: FloatArray
    output: FloatArray


@dataclass(frozen=True)
class MultiHeadAttentionCache:
    """Saved values needed for multi-head attention backpropagation."""

    embeddings: FloatArray
    head_caches: list[AttentionCache]
    concatenated_context: FloatArray


@dataclass(frozen=True)
class MultiHeadAttentionGradients:
    """Gradients for multi-head attention parameters and inputs."""

    embeddings: FloatArray
    heads: list[AttentionGradients]
    output_weights: FloatArray


@dataclass
class MultiHeadSelfAttention:
    """Run multiple attention heads and project their concatenated outputs."""

    query_weights: list[FloatArray]
    key_weights: list[FloatArray]
    value_weights: list[FloatArray]
    output_weights: FloatArray

    def forward(
        self,
        embeddings: FloatArray,
        attention_mask: NDArray[np.bool_] | None = None,
    ) -> MultiHeadAttentionResult:
        """Compute multi-head attention for one token sequence."""
        result, _cache = self.forward_with_cache(embeddings, attention_mask=attention_mask)
        return result

    def forward_with_cache(
        self,
        embeddings: FloatArray,
        attention_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[MultiHeadAttentionResult, MultiHeadAttentionCache]:
        """Compute multi-head attention and retain values for backprop."""
        self._validate_inputs(embeddings)

        head_results: list[AttentionResult] = []
        head_caches: list[AttentionCache] = []
        for query_weights, key_weights, value_weights in zip(
            self.query_weights,
            self.key_weights,
            self.value_weights,
            strict=True,
        ):
            head_result, head_cache = SingleHeadSelfAttention(
                query_weights=query_weights,
                key_weights=key_weights,
                value_weights=value_weights,
            ).forward_with_cache(embeddings, attention_mask=attention_mask)
            head_results.append(head_result)
            head_caches.append(head_cache)
        concatenated_context = np.concatenate(
            [head.context for head in head_results],
            axis=1,
        )
        output = matrix_multiply(concatenated_context, self.output_weights)

        result = MultiHeadAttentionResult(
            heads=head_results,
            concatenated_context=concatenated_context,
            output=output,
        )
        cache = MultiHeadAttentionCache(
            embeddings=embeddings,
            head_caches=head_caches,
            concatenated_context=concatenated_context,
        )
        return result, cache

    def backward(
        self,
        upstream_gradients: FloatArray,
        cache: MultiHeadAttentionCache,
    ) -> MultiHeadAttentionGradients:
        """Backpropagate through output projection and each attention head."""
        self._validate_inputs(cache.embeddings)
        if upstream_gradients.shape[0] != cache.embeddings.shape[0]:
            raise ValueError("upstream gradients must have one row per token")
        if upstream_gradients.shape[1] != self.output_weights.shape[1]:
            raise ValueError("upstream gradients must match attention output width")

        output_weight_gradients = matrix_multiply(
            cache.concatenated_context.T,
            upstream_gradients,
        )
        concatenated_gradients = matrix_multiply(upstream_gradients, self.output_weights.T)

        head_gradients: list[AttentionGradients] = []
        embedding_gradients = np.zeros_like(cache.embeddings)
        start = 0
        for head_index, head_cache in enumerate(cache.head_caches):
            head_width = head_cache.result.context.shape[1]
            head_upstream = concatenated_gradients[:, start : start + head_width]
            gradients = SingleHeadSelfAttention(
                query_weights=self.query_weights[head_index],
                key_weights=self.key_weights[head_index],
                value_weights=self.value_weights[head_index],
            ).backward(head_upstream, head_cache)
            head_gradients.append(gradients)
            embedding_gradients += gradients.embeddings
            start += head_width

        return MultiHeadAttentionGradients(
            embeddings=embedding_gradients,
            heads=head_gradients,
            output_weights=output_weight_gradients,
        )

    def apply_gradients(
        self,
        gradients: MultiHeadAttentionGradients,
        learning_rate: float,
    ) -> None:
        """Update output projection and per-head Q/K/V projections."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if len(gradients.heads) != len(self.query_weights):
            raise ValueError("one set of gradients is required for each head")

        self.output_weights -= learning_rate * gradients.output_weights
        for head_index, head_gradients in enumerate(gradients.heads):
            self.query_weights[head_index] -= learning_rate * head_gradients.query_weights
            self.key_weights[head_index] -= learning_rate * head_gradients.key_weights
            self.value_weights[head_index] -= learning_rate * head_gradients.value_weights

    def _validate_inputs(self, embeddings: FloatArray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("multi-head attention expects two-dimensional inputs")
        head_count = len(self.query_weights)
        if head_count == 0:
            raise ValueError("multi-head attention needs at least one head")
        if len(self.key_weights) != head_count or len(self.value_weights) != head_count:
            raise ValueError("query, key, and value head counts must match")
        if self.output_weights.ndim != 2:
            raise ValueError("output projection weights must be a matrix")

        context_width = 0
        for head_index, (query_weights, key_weights, value_weights) in enumerate(
            zip(self.query_weights, self.key_weights, self.value_weights, strict=True)
        ):
            if query_weights.ndim != 2 or key_weights.ndim != 2 or value_weights.ndim != 2:
                raise ValueError("all projection weights must be matrices")
            if embeddings.shape[1] != query_weights.shape[0]:
                raise ValueError(f"head {head_index} query weights do not match input width")
            if embeddings.shape[1] != key_weights.shape[0]:
                raise ValueError(f"head {head_index} key weights do not match input width")
            if embeddings.shape[1] != value_weights.shape[0]:
                raise ValueError(f"head {head_index} value weights do not match input width")
            if query_weights.shape[1] != key_weights.shape[1]:
                raise ValueError(f"head {head_index} query/key widths must match")
            context_width += value_weights.shape[1]

        if self.output_weights.shape[0] != context_width:
            raise ValueError("output projection must match concatenated context width")
