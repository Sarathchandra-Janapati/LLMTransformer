"""Layer normalization for token-wise transformer activations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transformer_core.foundations.linear_algebra import FloatArray


@dataclass
class LayerNormCache:
    """Saved values needed to backpropagate through layer normalization."""

    inputs: FloatArray
    normalized: FloatArray
    inverse_standard_deviation: FloatArray


@dataclass(frozen=True)
class LayerNormGradients:
    """Gradients for LayerNorm parameters and inputs."""

    inputs: FloatArray
    gamma: FloatArray
    beta: FloatArray


@dataclass
class LayerNorm:
    """Normalize each token row, then apply scale and bias parameters."""

    gamma: FloatArray
    beta: FloatArray
    epsilon: float = 1e-5

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Return row-wise normalized activations with affine parameters."""
        outputs, _cache = self.forward_with_cache(inputs)
        return outputs

    def forward_with_cache(self, inputs: FloatArray) -> tuple[FloatArray, LayerNormCache]:
        """Return normalized outputs plus values needed for backprop."""
        self._validate_inputs(inputs)

        means = np.mean(inputs, axis=1, keepdims=True)
        variances = np.var(inputs, axis=1, keepdims=True)
        inverse_standard_deviation = 1.0 / np.sqrt(variances + self.epsilon)
        normalized = (inputs - means) * inverse_standard_deviation
        outputs = normalized * self.gamma + self.beta
        return outputs, LayerNormCache(
            inputs=inputs,
            normalized=normalized,
            inverse_standard_deviation=inverse_standard_deviation,
        )

    def backward(
        self,
        upstream_gradients: FloatArray,
        cache: LayerNormCache,
    ) -> LayerNormGradients:
        """Backpropagate gradients through row-wise layer normalization."""
        self._validate_inputs(cache.inputs)
        if upstream_gradients.shape != cache.inputs.shape:
            raise ValueError("upstream gradients must match LayerNorm inputs")

        beta_gradients = np.sum(upstream_gradients, axis=0)
        gamma_gradients = np.sum(upstream_gradients * cache.normalized, axis=0)

        normalized_gradients = upstream_gradients * self.gamma
        width = cache.inputs.shape[1]
        input_gradients = (
            cache.inverse_standard_deviation
            / width
            * (
                width * normalized_gradients
                - np.sum(normalized_gradients, axis=1, keepdims=True)
                - cache.normalized
                * np.sum(
                    normalized_gradients * cache.normalized,
                    axis=1,
                    keepdims=True,
                )
            )
        )

        return LayerNormGradients(
            inputs=input_gradients,
            gamma=gamma_gradients,
            beta=beta_gradients,
        )

    def _validate_inputs(self, inputs: FloatArray) -> None:
        if inputs.ndim != 2:
            raise ValueError("LayerNorm expects token rows in a two-dimensional matrix")
        if self.gamma.ndim != 1 or self.beta.ndim != 1:
            raise ValueError("LayerNorm gamma and beta must be one-dimensional")
        if self.gamma.shape != self.beta.shape:
            raise ValueError("LayerNorm gamma and beta must have the same shape")
        if inputs.shape[1] != self.gamma.shape[0]:
            raise ValueError("LayerNorm parameters must match the activation width")
        if self.epsilon <= 0.0:
            raise ValueError("LayerNorm epsilon must be positive")
