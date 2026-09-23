"""Inspectable neural-network pieces built from the foundation math helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear_algebra import FloatArray, dot_product, matrix_multiply


def relu(values: FloatArray) -> FloatArray:
    """Return ReLU activations without changing the input array."""
    return np.maximum(values, 0.0)


def sigmoid(values: FloatArray) -> FloatArray:
    """Return sigmoid activations for a NumPy array."""
    return 1.0 / (1.0 + np.exp(-values))


def mean_squared_error(predictions: FloatArray, targets: FloatArray) -> float:
    """Return average squared error for equally shaped tensors."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")

    squared_error = (predictions - targets) ** 2
    return float(np.mean(squared_error))


@dataclass
class Neuron:
    """A single weighted sum with a bias term."""

    weights: FloatArray
    bias: float

    def forward(self, inputs: FloatArray) -> float:
        """Compute one neuron output for one input vector."""
        return dot_product(inputs, self.weights) + self.bias


@dataclass
class DenseLayer:
    """A batch-friendly dense layer with one weight column per output."""

    weights: FloatArray
    biases: FloatArray

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Compute dense outputs for a batch of input rows."""
        if inputs.ndim != 2:
            raise ValueError("DenseLayer expects batched two-dimensional inputs")
        if self.weights.ndim != 2 or self.biases.ndim != 1:
            raise ValueError("DenseLayer parameters have invalid dimensions")
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError("input width must match layer weight height")
        if self.weights.shape[1] != self.biases.shape[0]:
            raise ValueError("one bias is required for each layer output")

        return matrix_multiply(inputs, self.weights) + self.biases


@dataclass
class TwoLayerRegressionNetwork:
    """A tiny ReLU network used to show forward and backward passes."""

    input_weights: FloatArray
    input_biases: FloatArray
    output_weights: FloatArray
    output_biases: FloatArray

    def predict(self, inputs: FloatArray) -> FloatArray:
        """Run a forward pass and return predictions."""
        hidden_linear = DenseLayer(self.input_weights, self.input_biases).forward(inputs)
        hidden_activations = relu(hidden_linear)
        return DenseLayer(self.output_weights, self.output_biases).forward(
            hidden_activations
        )

    def train_step(
        self,
        inputs: FloatArray,
        targets: FloatArray,
        learning_rate: float,
    ) -> float:
        """Run explicit backpropagation for one mean-squared-error step."""
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        hidden_linear = DenseLayer(self.input_weights, self.input_biases).forward(inputs)
        hidden_activations = relu(hidden_linear)
        predictions = DenseLayer(self.output_weights, self.output_biases).forward(
            hidden_activations
        )
        loss = mean_squared_error(predictions, targets)

        prediction_count = predictions.size
        prediction_gradient = 2.0 * (predictions - targets) / prediction_count

        output_weight_gradient = hidden_activations.T @ prediction_gradient
        output_bias_gradient = np.sum(prediction_gradient, axis=0)

        hidden_gradient = prediction_gradient @ self.output_weights.T
        relu_gradient = hidden_linear > 0.0
        hidden_linear_gradient = hidden_gradient * relu_gradient

        input_weight_gradient = inputs.T @ hidden_linear_gradient
        input_bias_gradient = np.sum(hidden_linear_gradient, axis=0)

        self.output_weights -= learning_rate * output_weight_gradient
        self.output_biases -= learning_rate * output_bias_gradient
        self.input_weights -= learning_rate * input_weight_gradient
        self.input_biases -= learning_rate * input_bias_gradient

        return loss
