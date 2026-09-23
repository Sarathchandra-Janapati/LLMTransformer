"""Transformer feed-forward network for token-wise processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from transformer_core.foundations.linear_algebra import FloatArray
from transformer_core.foundations.neural_network import DenseLayer, relu


@dataclass
class FeedForwardCache:
    """Saved activations needed for feed-forward backpropagation."""

    inputs: FloatArray
    hidden: FloatArray
    activated: FloatArray


@dataclass(frozen=True)
class FeedForwardGradients:
    """Gradients for feed-forward parameters and inputs."""

    inputs: FloatArray
    input_weights: FloatArray
    input_biases: FloatArray
    output_weights: FloatArray
    output_biases: FloatArray


@dataclass
class FeedForwardNetwork:
    """Two-layer token-wise MLP used inside transformer blocks."""

    input_weights: FloatArray
    input_biases: FloatArray
    output_weights: FloatArray
    output_biases: FloatArray

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Expand, activate, and project token rows back to model width."""
        outputs, _cache = self.forward_with_cache(inputs)
        return outputs

    def forward_with_cache(
        self,
        inputs: FloatArray,
    ) -> tuple[FloatArray, FeedForwardCache]:
        """Return feed-forward outputs plus values needed for backprop."""
        self._validate_inputs(inputs)

        hidden = DenseLayer(self.input_weights, self.input_biases).forward(inputs)
        activated = relu(hidden)
        outputs = DenseLayer(self.output_weights, self.output_biases).forward(activated)
        return outputs, FeedForwardCache(
            inputs=inputs,
            hidden=hidden,
            activated=activated,
        )

    def backward(
        self,
        upstream_gradients: FloatArray,
        cache: FeedForwardCache,
    ) -> FeedForwardGradients:
        """Backpropagate through output projection, ReLU, and input projection."""
        self._validate_inputs(cache.inputs)
        if upstream_gradients.shape[0] != cache.inputs.shape[0]:
            raise ValueError("upstream gradients must have one row per input token")
        if upstream_gradients.shape[1] != self.output_biases.shape[0]:
            raise ValueError("upstream gradients must match the output width")

        output_weight_gradients = cache.activated.T @ upstream_gradients
        output_bias_gradients = np.sum(upstream_gradients, axis=0)

        activated_gradients = upstream_gradients @ self.output_weights.T
        hidden_gradients = activated_gradients * (cache.hidden > 0.0)

        input_weight_gradients = cache.inputs.T @ hidden_gradients
        input_bias_gradients = np.sum(hidden_gradients, axis=0)
        input_gradients = hidden_gradients @ self.input_weights.T

        return FeedForwardGradients(
            inputs=input_gradients,
            input_weights=input_weight_gradients,
            input_biases=input_bias_gradients,
            output_weights=output_weight_gradients,
            output_biases=output_bias_gradients,
        )

    def _validate_inputs(self, inputs: FloatArray) -> None:
        if inputs.ndim != 2:
            raise ValueError("FeedForwardNetwork expects token rows in a matrix")
        if self.input_weights.ndim != 2 or self.output_weights.ndim != 2:
            raise ValueError("FeedForwardNetwork weights must be matrices")
        if self.input_biases.ndim != 1 or self.output_biases.ndim != 1:
            raise ValueError("FeedForwardNetwork biases must be vectors")
        if inputs.shape[1] != self.input_weights.shape[0]:
            raise ValueError("input weights must match the model width")
        if self.input_weights.shape[1] != self.input_biases.shape[0]:
            raise ValueError("input bias width must match hidden width")
        if self.input_weights.shape[1] != self.output_weights.shape[0]:
            raise ValueError("output weights must accept the hidden width")
        if self.output_weights.shape[1] != self.output_biases.shape[0]:
            raise ValueError("output bias width must match model output width")
