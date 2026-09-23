"""First tiny next-token training loop for MiniGPT Studio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from transformer_core.foundations.linear_algebra import FloatArray, matrix_multiply
from transformer_core.attention import sinusoidal_position_encodings
from transformer_core.language_modeling import LanguageModelGradients
from transformer_core.tiny_gpt import TinyGPTModel


@dataclass(frozen=True)
class NextTokenTrainingExample:
    """One context window and its next-token labels."""

    input_token_ids: NDArray[np.integer]
    target_token_ids: NDArray[np.integer]


@dataclass(frozen=True)
class ProjectionGradients:
    """Gradients for the final vocabulary projection."""

    vocabulary_weights: FloatArray
    vocabulary_biases: FloatArray


@dataclass(frozen=True)
class EmbeddingGradients:
    """Gradients for token embedding rows touched by one example."""

    token_embeddings: FloatArray


@dataclass(frozen=True)
class TrainingStepMetrics:
    """Small metrics that can be plotted in the future dashboard."""

    step: int
    loss: float
    accuracy: float
    token_count: int


@dataclass(frozen=True)
class TrainingStepResult:
    """Inspectable output from one gradient update."""

    metrics: TrainingStepMetrics
    input_token_ids: NDArray[np.integer]
    target_token_ids: NDArray[np.integer]
    predicted_token_ids: NDArray[np.integer]
    gradients: ProjectionGradients


@dataclass(frozen=True)
class EmbeddingTrainingStepResult:
    """Inspectable output from one token-embedding update."""

    metrics: TrainingStepMetrics
    input_token_ids: NDArray[np.integer]
    target_token_ids: NDArray[np.integer]
    predicted_token_ids: NDArray[np.integer]
    gradients: EmbeddingGradients


@dataclass(frozen=True)
class FullModelGradients:
    """Gradients for one full tiny-GPT training step."""

    token_embeddings: FloatArray
    language_model: LanguageModelGradients


@dataclass(frozen=True)
class FullModelTrainingStepResult:
    """Inspectable output from one full-model update."""

    metrics: TrainingStepMetrics
    input_token_ids: NDArray[np.integer]
    target_token_ids: NDArray[np.integer]
    predicted_token_ids: NDArray[np.integer]
    gradients: FullModelGradients


def build_next_token_examples(
    token_ids: Sequence[int] | NDArray[np.integer],
    context_size: int,
) -> list[NextTokenTrainingExample]:
    """Create sliding next-token examples from one token-id sequence."""
    token_id_array = np.asarray(token_ids, dtype=int)
    _validate_token_sequence(token_id_array)
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if token_id_array.size <= context_size:
        raise ValueError("token sequence must be longer than the context size")

    examples: list[NextTokenTrainingExample] = []
    for start_index in range(token_id_array.size - context_size):
        window = token_id_array[start_index : start_index + context_size + 1]
        examples.append(
            NextTokenTrainingExample(
                input_token_ids=window[:-1],
                target_token_ids=window[1:],
            )
        )
    return examples


def cross_entropy_loss(
    probabilities: FloatArray,
    target_token_ids: Sequence[int] | NDArray[np.integer],
) -> float:
    """Return average negative log probability of the correct next tokens."""
    target_id_array = np.asarray(target_token_ids, dtype=int)
    _validate_probabilities_and_targets(probabilities, target_id_array)

    correct_probabilities = probabilities[
        np.arange(target_id_array.size),
        target_id_array,
    ]
    return float(-np.mean(np.log(np.clip(correct_probabilities, 1e-12, 1.0))))


def next_token_accuracy(
    predicted_token_ids: Sequence[int] | NDArray[np.integer],
    target_token_ids: Sequence[int] | NDArray[np.integer],
) -> float:
    """Return the fraction of token positions predicted exactly."""
    predictions = np.asarray(predicted_token_ids, dtype=int)
    targets = np.asarray(target_token_ids, dtype=int)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if targets.size == 0:
        raise ValueError("accuracy needs at least one target token")
    return float(np.mean(predictions == targets))


def train_output_projection_step(
    model: TinyGPTModel,
    example: NextTokenTrainingExample,
    learning_rate: float,
    step: int = 0,
) -> TrainingStepResult:
    """Train the final vocabulary projection for one next-token example.

    This is the first training milestone: the transformer stack is frozen, and
    only the final matrix that maps hidden states to vocabulary logits is updated.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    _validate_example(example)

    result = model.forward(example.input_token_ids, causal_mask=True)
    probabilities = result.language_model.probabilities
    loss = cross_entropy_loss(probabilities, example.target_token_ids)
    predictions = result.language_model.predicted_token_ids
    accuracy = next_token_accuracy(predictions, example.target_token_ids)

    gradient_logits = probabilities.copy()
    gradient_logits[
        np.arange(example.target_token_ids.size),
        example.target_token_ids,
    ] -= 1.0
    gradient_logits = gradient_logits / example.target_token_ids.size

    hidden_states = result.language_model.normalized_hidden_states
    weight_gradients = matrix_multiply(hidden_states.T, gradient_logits)
    bias_gradients = np.sum(gradient_logits, axis=0)

    model.language_model.vocabulary_weights -= learning_rate * weight_gradients
    model.language_model.vocabulary_biases -= learning_rate * bias_gradients

    return TrainingStepResult(
        metrics=TrainingStepMetrics(
            step=step,
            loss=loss,
            accuracy=accuracy,
            token_count=example.target_token_ids.size,
        ),
        input_token_ids=example.input_token_ids,
        target_token_ids=example.target_token_ids,
        predicted_token_ids=predictions,
        gradients=ProjectionGradients(
            vocabulary_weights=weight_gradients,
            vocabulary_biases=bias_gradients,
        ),
    )


def train_output_projection(
    model: TinyGPTModel,
    examples: Sequence[NextTokenTrainingExample],
    learning_rate: float,
    epochs: int,
) -> list[TrainingStepResult]:
    """Run several passes of projection-only next-token training."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if len(examples) == 0:
        raise ValueError("training needs at least one example")

    history: list[TrainingStepResult] = []
    step = 0
    for _epoch in range(epochs):
        for example in examples:
            history.append(
                train_output_projection_step(
                    model=model,
                    example=example,
                    learning_rate=learning_rate,
                    step=step,
                )
            )
            step += 1
    return history


def train_token_embeddings_step(
    model: TinyGPTModel,
    example: NextTokenTrainingExample,
    learning_rate: float,
    finite_difference_epsilon: float = 1e-4,
    step: int = 0,
) -> EmbeddingTrainingStepResult:
    """Train touched token embeddings using finite-difference gradients.

    This is intentionally slow and tiny. It is a learning bridge before full
    analytic backprop through layer norm, attention, residuals, and MLP layers.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if finite_difference_epsilon <= 0:
        raise ValueError("finite_difference_epsilon must be positive")
    _validate_example(example)

    result = model.forward(example.input_token_ids, causal_mask=True)
    loss = cross_entropy_loss(
        result.language_model.probabilities,
        example.target_token_ids,
    )
    accuracy = next_token_accuracy(
        result.language_model.predicted_token_ids,
        example.target_token_ids,
    )

    embedding_gradients = np.zeros_like(model.token_embedding_table.embeddings)
    touched_token_ids = np.unique(example.input_token_ids)
    for token_id in touched_token_ids:
        for dimension in range(model.token_embedding_table.embeddings.shape[1]):
            original_value = model.token_embedding_table.embeddings[token_id, dimension]

            model.token_embedding_table.embeddings[token_id, dimension] = (
                original_value + finite_difference_epsilon
            )
            plus_loss = _loss_for_example(model, example)

            model.token_embedding_table.embeddings[token_id, dimension] = (
                original_value - finite_difference_epsilon
            )
            minus_loss = _loss_for_example(model, example)

            model.token_embedding_table.embeddings[token_id, dimension] = original_value
            embedding_gradients[token_id, dimension] = (
                plus_loss - minus_loss
            ) / (2.0 * finite_difference_epsilon)

    model.token_embedding_table.embeddings -= learning_rate * embedding_gradients

    return EmbeddingTrainingStepResult(
        metrics=TrainingStepMetrics(
            step=step,
            loss=loss,
            accuracy=accuracy,
            token_count=example.target_token_ids.size,
        ),
        input_token_ids=example.input_token_ids,
        target_token_ids=example.target_token_ids,
        predicted_token_ids=result.language_model.predicted_token_ids,
        gradients=EmbeddingGradients(token_embeddings=embedding_gradients),
    )


def train_projection_and_embeddings(
    model: TinyGPTModel,
    examples: Sequence[NextTokenTrainingExample],
    projection_learning_rate: float,
    embedding_learning_rate: float,
    epochs: int,
    finite_difference_epsilon: float = 1e-4,
) -> list[TrainingStepResult | EmbeddingTrainingStepResult]:
    """Train output projection analytically, then embeddings numerically."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if len(examples) == 0:
        raise ValueError("training needs at least one example")

    history: list[TrainingStepResult | EmbeddingTrainingStepResult] = []
    step = 0
    for _epoch in range(epochs):
        for example in examples:
            history.append(
                train_output_projection_step(
                    model=model,
                    example=example,
                    learning_rate=projection_learning_rate,
                    step=step,
                )
            )
            step += 1
            history.append(
                train_token_embeddings_step(
                    model=model,
                    example=example,
                    learning_rate=embedding_learning_rate,
                    finite_difference_epsilon=finite_difference_epsilon,
                    step=step,
                )
            )
            step += 1
    return history


def train_full_model_step(
    model: TinyGPTModel,
    example: NextTokenTrainingExample,
    learning_rate: float,
    step: int = 0,
) -> FullModelTrainingStepResult:
    """Train embeddings, decoder blocks, final norm, and vocabulary projection."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    _validate_example(example)

    token_embeddings = model.token_embedding_table.forward(example.input_token_ids)
    position_encodings = sinusoidal_position_encodings(
        token_count=token_embeddings.shape[0],
        width=token_embeddings.shape[1],
    )
    model_inputs = token_embeddings + position_encodings
    result, cache = model.language_model.forward_with_cache(model_inputs, causal_mask=True)

    loss = cross_entropy_loss(result.probabilities, example.target_token_ids)
    accuracy = next_token_accuracy(
        result.predicted_token_ids,
        example.target_token_ids,
    )
    logit_gradients = _cross_entropy_logit_gradients(
        result.probabilities,
        example.target_token_ids,
    )
    language_model_gradients = model.language_model.backward(
        logit_gradients,
        result,
        cache,
    )

    token_embedding_gradients = np.zeros_like(model.token_embedding_table.embeddings)
    for row_index, token_id in enumerate(example.input_token_ids):
        token_embedding_gradients[token_id] += language_model_gradients.inputs[row_index]

    model.language_model.apply_gradients(language_model_gradients, learning_rate)
    model.token_embedding_table.embeddings -= learning_rate * token_embedding_gradients

    return FullModelTrainingStepResult(
        metrics=TrainingStepMetrics(
            step=step,
            loss=loss,
            accuracy=accuracy,
            token_count=example.target_token_ids.size,
        ),
        input_token_ids=example.input_token_ids,
        target_token_ids=example.target_token_ids,
        predicted_token_ids=result.predicted_token_ids,
        gradients=FullModelGradients(
            token_embeddings=token_embedding_gradients,
            language_model=language_model_gradients,
        ),
    )


def train_full_model(
    model: TinyGPTModel,
    examples: Sequence[NextTokenTrainingExample],
    learning_rate: float,
    epochs: int,
) -> list[FullModelTrainingStepResult]:
    """Run full analytic next-token training over a tiny dataset."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if len(examples) == 0:
        raise ValueError("training needs at least one example")

    history: list[FullModelTrainingStepResult] = []
    step = 0
    for _epoch in range(epochs):
        for example in examples:
            history.append(
                train_full_model_step(
                    model=model,
                    example=example,
                    learning_rate=learning_rate,
                    step=step,
                )
            )
            step += 1
    return history


def _loss_for_example(
    model: TinyGPTModel,
    example: NextTokenTrainingExample,
) -> float:
    result = model.forward(example.input_token_ids, causal_mask=True)
    return cross_entropy_loss(
        result.language_model.probabilities,
        example.target_token_ids,
    )


def _cross_entropy_logit_gradients(
    probabilities: FloatArray,
    target_token_ids: NDArray[np.integer],
) -> FloatArray:
    _validate_probabilities_and_targets(probabilities, target_token_ids)
    gradients = probabilities.copy()
    gradients[np.arange(target_token_ids.size), target_token_ids] -= 1.0
    return gradients / target_token_ids.size


def _validate_token_sequence(token_ids: NDArray[np.integer]) -> None:
    if token_ids.ndim != 1:
        raise ValueError("token ids must be a one-dimensional sequence")
    if token_ids.size == 0:
        raise ValueError("token ids must include at least one token")


def _validate_example(example: NextTokenTrainingExample) -> None:
    if example.input_token_ids.ndim != 1 or example.target_token_ids.ndim != 1:
        raise ValueError("training examples must contain one-dimensional token ids")
    if example.input_token_ids.shape != example.target_token_ids.shape:
        raise ValueError("input and target token ids must have the same shape")
    if example.target_token_ids.size == 0:
        raise ValueError("training examples need at least one target token")


def _validate_probabilities_and_targets(
    probabilities: FloatArray,
    target_token_ids: NDArray[np.integer],
) -> None:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a token-by-vocabulary matrix")
    if target_token_ids.ndim != 1:
        raise ValueError("target token ids must be one-dimensional")
    if probabilities.shape[0] != target_token_ids.size:
        raise ValueError("one target token id is required for each probability row")
    if target_token_ids.size == 0:
        raise ValueError("loss needs at least one target token")
    if np.any(target_token_ids < 0) or np.any(target_token_ids >= probabilities.shape[1]):
        raise ValueError("target token id is outside the probability vocabulary")
