"""Tiny decoder stack and language-model head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transformer_core.attention import causal_attention_mask, softmax
from transformer_core.blocks import (
    LayerNorm,
    LayerNormCache,
    LayerNormGradients,
    TransformerBlock,
    TransformerBlockCache,
    TransformerBlockGradients,
    TransformerBlockResult,
)
from transformer_core.foundations.linear_algebra import FloatArray, matrix_multiply


@dataclass(frozen=True)
class DecoderStackResult:
    """Intermediate tensors from a decoder stack forward pass."""

    blocks: list[TransformerBlockResult]
    hidden_states: FloatArray


@dataclass(frozen=True)
class DecoderStackCache:
    """Saved block caches for decoder backpropagation."""

    block_caches: list[TransformerBlockCache]


@dataclass(frozen=True)
class DecoderStackGradients:
    """Gradients flowing through a decoder stack."""

    inputs: FloatArray
    blocks: list[TransformerBlockGradients]


@dataclass
class DecoderStack:
    """Stack transformer blocks with a shared decoder attention mask."""

    blocks: list[TransformerBlock]

    def forward(
        self,
        inputs: FloatArray,
        attention_mask: NDArray[np.bool_] | None = None,
        causal_mask: bool = True,
    ) -> DecoderStackResult:
        """Run each transformer block over token rows."""
        result, _cache = self.forward_with_cache(
            inputs,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        return result

    def forward_with_cache(
        self,
        inputs: FloatArray,
        attention_mask: NDArray[np.bool_] | None = None,
        causal_mask: bool = True,
    ) -> tuple[DecoderStackResult, DecoderStackCache]:
        """Run each transformer block and retain caches for backprop."""
        self._validate_inputs(inputs)
        if attention_mask is not None and causal_mask:
            raise ValueError("provide either attention_mask or causal_mask, not both")

        resolved_mask = attention_mask
        if causal_mask:
            resolved_mask = causal_attention_mask(inputs.shape[0])

        hidden_states = inputs
        block_results: list[TransformerBlockResult] = []
        block_caches: list[TransformerBlockCache] = []
        for block in self.blocks:
            result, cache = block.forward_with_cache(
                hidden_states,
                attention_mask=resolved_mask,
            )
            block_results.append(result)
            block_caches.append(cache)
            hidden_states = result.output

        return (
            DecoderStackResult(blocks=block_results, hidden_states=hidden_states),
            DecoderStackCache(block_caches=block_caches),
        )

    def backward(
        self,
        upstream_gradients: FloatArray,
        cache: DecoderStackCache,
    ) -> DecoderStackGradients:
        """Backpropagate from the final block to the first block."""
        if len(cache.block_caches) != len(self.blocks):
            raise ValueError("one cache is required for each decoder block")

        gradients_by_block: list[TransformerBlockGradients] = []
        current_gradients = upstream_gradients
        for block, block_cache in zip(
            reversed(self.blocks),
            reversed(cache.block_caches),
            strict=True,
        ):
            block_gradients = block.backward(current_gradients, block_cache)
            gradients_by_block.append(block_gradients)
            current_gradients = block_gradients.inputs

        gradients_by_block.reverse()
        return DecoderStackGradients(
            inputs=current_gradients,
            blocks=gradients_by_block,
        )

    def apply_gradients(
        self,
        gradients: DecoderStackGradients,
        learning_rate: float,
    ) -> None:
        """Update each decoder block with its matching gradients."""
        if len(gradients.blocks) != len(self.blocks):
            raise ValueError("one gradient object is required for each decoder block")

        for block, block_gradients in zip(self.blocks, gradients.blocks, strict=True):
            block.apply_gradients(block_gradients, learning_rate)

    def _validate_inputs(self, inputs: FloatArray) -> None:
        if inputs.ndim != 2:
            raise ValueError("DecoderStack expects token rows in a matrix")
        if len(self.blocks) == 0:
            raise ValueError("DecoderStack needs at least one transformer block")


@dataclass(frozen=True)
class LanguageModelResult:
    """Outputs from a tiny language-model forward pass."""

    decoder: DecoderStackResult
    normalized_hidden_states: FloatArray
    logits: FloatArray
    probabilities: FloatArray
    predicted_token_ids: NDArray[np.integer]


@dataclass(frozen=True)
class LanguageModelCache:
    """Saved values needed for language-model head backprop."""

    inputs: FloatArray
    decoder: DecoderStackCache
    final_norm: LayerNormCache


@dataclass(frozen=True)
class LanguageModelGradients:
    """Gradients for the language-model head and decoder."""

    inputs: FloatArray
    decoder: DecoderStackGradients
    final_norm: LayerNormGradients
    vocabulary_weights: FloatArray
    vocabulary_biases: FloatArray


@dataclass
class LanguageModelHead:
    """Project decoder states into vocabulary logits and probabilities."""

    decoder: DecoderStack
    final_norm: LayerNorm
    vocabulary_weights: FloatArray
    vocabulary_biases: FloatArray

    def forward(
        self,
        inputs: FloatArray,
        attention_mask: NDArray[np.bool_] | None = None,
        causal_mask: bool = True,
    ) -> LanguageModelResult:
        """Run a decoder stack and predict a token distribution per position."""
        result, _cache = self.forward_with_cache(
            inputs,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        return result

    def forward_with_cache(
        self,
        inputs: FloatArray,
        attention_mask: NDArray[np.bool_] | None = None,
        causal_mask: bool = True,
    ) -> tuple[LanguageModelResult, LanguageModelCache]:
        """Run the language model and retain values for backprop."""
        self._validate_inputs(inputs)

        decoder_result, decoder_cache = self.decoder.forward_with_cache(
            inputs,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
        )
        normalized_hidden_states, final_norm_cache = self.final_norm.forward_with_cache(
            decoder_result.hidden_states
        )
        logits = (
            matrix_multiply(normalized_hidden_states, self.vocabulary_weights)
            + self.vocabulary_biases
        )
        probabilities = softmax(logits)
        predicted_token_ids = np.argmax(probabilities, axis=1)

        result = LanguageModelResult(
            decoder=decoder_result,
            normalized_hidden_states=normalized_hidden_states,
            logits=logits,
            probabilities=probabilities,
            predicted_token_ids=predicted_token_ids,
        )
        cache = LanguageModelCache(
            inputs=inputs,
            decoder=decoder_cache,
            final_norm=final_norm_cache,
        )
        return result, cache

    def backward(
        self,
        logit_gradients: FloatArray,
        result: LanguageModelResult,
        cache: LanguageModelCache,
    ) -> LanguageModelGradients:
        """Backpropagate from vocabulary logits into the decoder inputs."""
        if logit_gradients.shape != result.logits.shape:
            raise ValueError("logit gradients must match logits")

        vocabulary_weight_gradients = matrix_multiply(
            result.normalized_hidden_states.T,
            logit_gradients,
        )
        vocabulary_bias_gradients = np.sum(logit_gradients, axis=0)
        normalized_hidden_gradients = matrix_multiply(
            logit_gradients,
            self.vocabulary_weights.T,
        )
        final_norm_gradients = self.final_norm.backward(
            normalized_hidden_gradients,
            cache.final_norm,
        )
        decoder_gradients = self.decoder.backward(
            final_norm_gradients.inputs,
            cache.decoder,
        )

        return LanguageModelGradients(
            inputs=decoder_gradients.inputs,
            decoder=decoder_gradients,
            final_norm=final_norm_gradients,
            vocabulary_weights=vocabulary_weight_gradients,
            vocabulary_biases=vocabulary_bias_gradients,
        )

    def apply_gradients(
        self,
        gradients: LanguageModelGradients,
        learning_rate: float,
    ) -> None:
        """Update vocabulary projection, final norm, and decoder blocks."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.vocabulary_weights -= learning_rate * gradients.vocabulary_weights
        self.vocabulary_biases -= learning_rate * gradients.vocabulary_biases
        self.final_norm.gamma -= learning_rate * gradients.final_norm.gamma
        self.final_norm.beta -= learning_rate * gradients.final_norm.beta
        self.decoder.apply_gradients(gradients.decoder, learning_rate)

    def _validate_inputs(self, inputs: FloatArray) -> None:
        if inputs.ndim != 2:
            raise ValueError("LanguageModelHead expects token rows in a matrix")
        if self.vocabulary_weights.ndim != 2 or self.vocabulary_biases.ndim != 1:
            raise ValueError("vocabulary projection parameters have invalid dimensions")
        if inputs.shape[1] != self.vocabulary_weights.shape[0]:
            raise ValueError("vocabulary weights must match the model width")
        if self.vocabulary_weights.shape[1] != self.vocabulary_biases.shape[0]:
            raise ValueError("one vocabulary bias is required for each token id")
