import numpy as np
import pytest

from transformer_core.foundations import (
    DenseLayer,
    Neuron,
    TwoLayerRegressionNetwork,
    mean_squared_error,
    relu,
    sigmoid,
)


def test_neuron_combines_weighted_inputs_and_bias() -> None:
    neuron = Neuron(weights=np.array([0.5, -1.0, 2.0]), bias=0.25)

    assert neuron.forward(np.array([2.0, 3.0, -1.0])) == pytest.approx(-3.75)


def test_dense_layer_returns_one_output_row_per_input_row() -> None:
    layer = DenseLayer(
        weights=np.array([[1.0, -1.0], [0.5, 2.0]]),
        biases=np.array([0.25, -0.5]),
    )

    outputs = layer.forward(np.array([[2.0, 4.0], [-1.0, 3.0]]))

    np.testing.assert_allclose(outputs, np.array([[4.25, 5.5], [0.75, 6.5]]))


def test_activations_keep_expected_ranges() -> None:
    values = np.array([-2.0, 0.0, 2.0])

    np.testing.assert_allclose(relu(values), np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(sigmoid(values), np.array([0.11920292, 0.5, 0.88079708]))


def test_mean_squared_error_averages_elementwise_error() -> None:
    predictions = np.array([[1.0], [3.0]])
    targets = np.array([[2.0], [1.0]])

    assert mean_squared_error(predictions, targets) == pytest.approx(2.5)


def test_backpropagation_step_reduces_loss_for_tiny_regression_problem() -> None:
    network = TwoLayerRegressionNetwork(
        input_weights=np.array([[0.2, 0.4], [0.1, 0.3]]),
        input_biases=np.array([0.0, 0.1]),
        output_weights=np.array([[0.5], [0.25]]),
        output_biases=np.array([0.0]),
    )
    inputs = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    targets = np.array([[1.0], [1.5], [1.5]])

    starting_loss = mean_squared_error(network.predict(inputs), targets)
    network.train_step(inputs, targets, learning_rate=0.1)
    ending_loss = mean_squared_error(network.predict(inputs), targets)

    assert ending_loss < starting_loss

