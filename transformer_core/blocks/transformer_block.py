"""A small inspectable pre-norm transformer block."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from transformer_core.attention import (
    MultiHeadAttentionCache,
    MultiHeadAttentionGradients,
    MultiHeadAttentionResult,
    MultiHeadSelfAttention,
)
from transformer_core.foundations.linear_algebra import FloatArray

from .feed_forward import FeedForwardCache, FeedForwardGradients, FeedForwardNetwork
from .normalization import LayerNorm, LayerNormCache, LayerNormGradients


@dataclass(frozen=True)
class TransformerBlockResult:
    """Intermediate tensors from one transformer block forward pass."""

    normalized_attention_input: FloatArray
    attention: MultiHeadAttentionResult
    attention_residual: FloatArray
    normalized_feed_forward_input: FloatArray
    feed_forward_output: FloatArray
    output: FloatArray


@dataclass(frozen=True)
class TransformerBlockCache:
    """Saved values needed for transformer-block backprop."""

    inputs: FloatArray
    attention_norm: LayerNormCache
    attention: MultiHeadAttentionCache
    feed_forward_norm: LayerNormCache
    feed_forward: FeedForwardCache


@dataclass(frozen=True)
class TransformerBlockGradients:
    """Gradients for the trainable pieces of one transformer block."""

    inputs: FloatArray
    attention_norm: LayerNormGradients
    attention: MultiHeadAttentionGradients
    feed_forward_norm: LayerNormGradients
    feed_forward: FeedForwardGradients


@dataclass
class TransformerBlock:
    """Pre-norm transformer block with attention, MLP, and residuals."""

    attention: MultiHeadSelfAttention
    attention_norm: LayerNorm
    feed_forward: FeedForwardNetwork
    feed_forward_norm: LayerNorm

    def forward(
        self,
        inputs: FloatArray,
        attention_mask: NDArray | None = None,
    ) -> TransformerBlockResult:
        """Run one transformer block over token rows."""
        result, _cache = self.forward_with_cache(inputs, attention_mask=attention_mask)
        return result

    def forward_with_cache(
        self,
        inputs: FloatArray,
        attention_mask: NDArray | None = None,
    ) -> tuple[TransformerBlockResult, TransformerBlockCache]:
        """Run one block and retain cache values for partial backprop."""
        self._validate_inputs(inputs)

        normalized_attention_input, attention_norm_cache = (
            self.attention_norm.forward_with_cache(inputs)
        )
        attention_result, attention_cache = self.attention.forward_with_cache(
            normalized_attention_input,
            attention_mask=attention_mask,
        )
        if attention_result.output.shape != inputs.shape:
            raise ValueError("attention output must match residual input shape")

        attention_residual = inputs + attention_result.output
        normalized_feed_forward_input, feed_forward_norm_cache = (
            self.feed_forward_norm.forward_with_cache(attention_residual)
        )
        feed_forward_output, feed_forward_cache = self.feed_forward.forward_with_cache(
            normalized_feed_forward_input
        )
        if feed_forward_output.shape != attention_residual.shape:
            raise ValueError("feed-forward output must match residual input shape")

        output = attention_residual + feed_forward_output

        result = TransformerBlockResult(
            normalized_attention_input=normalized_attention_input,
            attention=attention_result,
            attention_residual=attention_residual,
            normalized_feed_forward_input=normalized_feed_forward_input,
            feed_forward_output=feed_forward_output,
            output=output,
        )
        cache = TransformerBlockCache(
            inputs=inputs,
            attention_norm=attention_norm_cache,
            attention=attention_cache,
            feed_forward_norm=feed_forward_norm_cache,
            feed_forward=feed_forward_cache,
        )
        return result, cache

    def backward(
        self,
        upstream_gradients: FloatArray,
        cache: TransformerBlockCache,
    ) -> TransformerBlockGradients:
        """Backpropagate through attention, MLP, norms, and residual paths."""
        self._validate_inputs(cache.inputs)
        if upstream_gradients.shape != cache.inputs.shape:
            raise ValueError("upstream gradients must match transformer block output")

        attention_residual_gradients = upstream_gradients.copy()
        feed_forward_gradients = self.feed_forward.backward(
            upstream_gradients,
            cache.feed_forward,
        )
        feed_forward_norm_gradients = self.feed_forward_norm.backward(
            feed_forward_gradients.inputs,
            cache.feed_forward_norm,
        )
        attention_residual_gradients += feed_forward_norm_gradients.inputs

        attention_gradients = self.attention.backward(
            attention_residual_gradients,
            cache.attention,
        )
        attention_norm_gradients = self.attention_norm.backward(
            attention_gradients.embeddings,
            cache.attention_norm,
        )
        input_gradients = attention_residual_gradients + attention_norm_gradients.inputs

        return TransformerBlockGradients(
            inputs=input_gradients,
            attention_norm=attention_norm_gradients,
            attention=attention_gradients,
            feed_forward_norm=feed_forward_norm_gradients,
            feed_forward=feed_forward_gradients,
        )

    def apply_gradients(
        self,
        gradients: TransformerBlockGradients,
        learning_rate: float,
    ) -> None:
        """Update block parameters."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.attention.apply_gradients(gradients.attention, learning_rate)
        self.attention_norm.gamma -= learning_rate * gradients.attention_norm.gamma
        self.attention_norm.beta -= learning_rate * gradients.attention_norm.beta
        self.feed_forward.input_weights -= learning_rate * gradients.feed_forward.input_weights
        self.feed_forward.input_biases -= learning_rate * gradients.feed_forward.input_biases
        self.feed_forward.output_weights -= learning_rate * gradients.feed_forward.output_weights
        self.feed_forward.output_biases -= learning_rate * gradients.feed_forward.output_biases
        self.feed_forward_norm.gamma -= learning_rate * gradients.feed_forward_norm.gamma
        self.feed_forward_norm.beta -= learning_rate * gradients.feed_forward_norm.beta

    def _validate_inputs(self, inputs: FloatArray) -> None:
        if inputs.ndim != 2:
            raise ValueError("TransformerBlock expects token rows in a matrix")
